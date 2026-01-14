"""
Stage 2: EEG-to-Text alignment with VL-JEPA objective.

Architecture:
    Multi-View Transformer (10 brain regions) -> Predictor -> Text Embedding Space
    
Training objective: Predict text embeddings from EEG (L2 loss in embedding space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.brain_regions import BRAIN_REGIONS, NUM_BANDS


class RegionEncoder(nn.Module):
    """
    Encoder for a single brain region.
    
    Input: (batch, num_region_channels, num_bands)
    Output: (batch, embed_dim)
    """
    
    def __init__(self, 
                 num_channels: int, 
                 num_bands: int = 8, 
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        input_dim = num_channels * num_bands
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Region EEG (B, num_region_channels, num_bands)
        Returns:
            Region features (B, embed_dim)
        """
        B = x.size(0)
        x = x.reshape(B, -1)  # Flatten
        return self.net(x)


class MultiViewTransformer(nn.Module):
    """
    Multi-View Transformer with 10 brain region encoders.
    
    Each region has its own encoder, then a global transformer fuses all regions.
    
    Input: (batch, 105, 8) - full EEG
    Output: (batch, embed_dim) - fused representation
    """
    
    def __init__(self,
                 brain_regions: Dict[str, List[int]] = None,
                 num_bands: int = 8,
                 embed_dim: int = 256,
                 global_transformer_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.regions = brain_regions or BRAIN_REGIONS
        self.num_regions = len(self.regions)
        
        # Create one encoder per brain region
        self.region_encoders = nn.ModuleDict({
            name: RegionEncoder(len(channels), num_bands, embed_dim, dropout)
            for name, channels in self.regions.items()
        })
        
        # Learnable region embeddings
        self.region_embed = nn.Parameter(
            torch.zeros(1, self.num_regions, embed_dim)
        )
        
        # Global transformer for cross-region fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=global_transformer_layers
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.region_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Full EEG (B, 105, 8)
        Returns:
            Fused features (B, embed_dim)
        """
        B = x.size(0)
        
        # Process each region
        region_features = []
        for name, channels in self.regions.items():
            region_x = x[:, channels, :]  # (B, num_region_channels, 8)
            feat = self.region_encoders[name](region_x)  # (B, embed_dim)
            region_features.append(feat)
        
        # Stack: (B, num_regions, embed_dim)
        features = torch.stack(region_features, dim=1)
        
        # Add region embeddings
        features = features + self.region_embed
        
        # Global fusion
        features = self.global_transformer(features)
        features = self.norm(features)
        
        # Global average pooling
        return features.mean(dim=1)  # (B, embed_dim)


class EEG2TextJEPA(nn.Module):
    """
    Stage 2: EEG-to-Text alignment with VL-JEPA training objective.
    
    Training: Predict text embeddings from EEG using L2 loss
    Inference: Generate text embeddings for retrieval or decoder
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 text_encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 pretrained_path: Optional[str] = None,
                 freeze_text_encoder: bool = True):
        super().__init__()
        
        # Multi-View EEG Encoder
        self.eeg_encoder = MultiViewTransformer(embed_dim=embed_dim)
        
        # Load pretrained Stage 1 weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # Text Encoder (frozen by default)
        print(f"Loading text encoder: {text_encoder_name}")
        self.text_encoder = SentenceTransformer(text_encoder_name)
        self.text_embed_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Predictor: maps EEG embedding to text embedding space
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, self.text_embed_dim),
            nn.LayerNorm(self.text_embed_dim)
        )
    
    def _load_pretrained(self, path: str):
        """Load pretrained Stage 1 encoder weights."""
        print(f"Loading pretrained weights from {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # Try to load encoder weights
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
        if encoder_keys:
            # Filter and rename keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    new_key = k.replace('encoder.', 'eeg_encoder.')
                    new_state_dict[new_key] = v
            
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f"  Loaded {len(new_state_dict)} weights")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using frozen text encoder.
        
        Args:
            texts: List of text strings
        Returns:
            Text embeddings (B, text_embed_dim)
        """
        with torch.no_grad():
            embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        return embeddings
    
    def forward(self, 
                eeg: torch.Tensor, 
                texts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            eeg: EEG tensor (B, 105, 8)
            texts: Optional target texts for training
            
        Returns:
            predicted_embed: Predicted text embeddings (B, text_embed_dim)
            target_embed: Target text embeddings (if texts provided)
            loss: L2 loss in embedding space (if texts provided)
        """
        # Encode EEG through multi-view transformer
        eeg_features = self.eeg_encoder(eeg)  # (B, embed_dim)
        
        # Predict text embedding
        predicted_embed = self.predictor(eeg_features)
        predicted_embed = F.normalize(predicted_embed, p=2, dim=-1)
        
        if texts is not None:
            # Get target embeddings from frozen text encoder
            target_embed = self.encode_text(texts).to(predicted_embed.device)
            
            # VL-JEPA loss: L2 in embedding space
            mse_loss = F.mse_loss(predicted_embed, target_embed)
            
            # Optional: cosine similarity loss for better alignment
            cos_loss = 1 - F.cosine_similarity(predicted_embed, target_embed).mean()
            
            loss = mse_loss + 0.5 * cos_loss
            
            return predicted_embed, target_embed, loss
        
        return predicted_embed, None, None
    
    @torch.no_grad()
    def get_similarity(self, 
                       eeg: torch.Tensor, 
                       candidate_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute similarity between EEG and candidate texts (for retrieval).
        
        Args:
            eeg: EEG tensor (B, 105, 8)
            candidate_texts: List of N candidate texts
            
        Returns:
            Similarity matrix (B, N)
        """
        self.eval()
        
        # Get EEG embeddings
        predicted_embed, _, _ = self.forward(eeg)  # (B, text_embed_dim)
        
        # Get text embeddings
        text_embeds = self.encode_text(candidate_texts).to(predicted_embed.device)  # (N, text_embed_dim)
        
        # Cosine similarity
        similarity = F.cosine_similarity(
            predicted_embed.unsqueeze(1),  # (B, 1, D)
            text_embeds.unsqueeze(0),       # (1, N, D)
            dim=-1
        )  # (B, N)
        
        return similarity
