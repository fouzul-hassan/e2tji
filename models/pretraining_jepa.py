"""
Stage 1: True JEPA Self-supervised pretraining with latent prediction.

Architecture:
    Context EEG Regions -> Context Encoder -> Predictor -> Predicted Latent
    Target EEG Regions  -> Target Encoder (EMA) -> Target Latent
    
    Loss: L2(Predicted Latent, Target Latent) in LATENT SPACE
    
Key differences from MAE:
    - No decoder - operates entirely in latent/representation space
    - Target encoder is EMA copy of context encoder
    - Predicts masked region latents, not raw signal
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.brain_regions import BRAIN_REGIONS


class RegionPatchEncoder(nn.Module):
    """
    Encode a single brain region patch to latent representation.
    
    Handles both formats:
    - Regular: (batch, num_region_channels, 8) - frequency bands
    - Spectro: (batch, num_region_channels, T) - time series
    """
    
    def __init__(self, 
                 num_channels: int, 
                 num_features: int = 500, 
                 embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_features = num_features
        
        # For spectro data (large feature dim), use CNN to compress
        if num_features > 100:
            # CNN-based encoder for time series
            self.conv = nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=15, stride=5, padding=7),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=3),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(8),  # Always output 8 time steps
            )
            # Output: (B, 128, 8) -> flatten: 1024
            self.proj = nn.Sequential(
                nn.Linear(128 * 8, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            self.use_cnn = True
        else:
            # Linear encoder for frequency band data
            self.use_cnn = False
            input_dim = num_channels * num_features
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
            x: Region EEG (B, num_region_channels, num_features)
        Returns:
            Region latent (B, embed_dim)
        """
        B = x.size(0)
        if self.use_cnn:
            x = self.conv(x)  # (B, 128, 8)
            x = x.reshape(B, -1)  # (B, 1024)
            return self.proj(x)
        else:
            x = x.reshape(B, -1)  # Flatten
            return self.net(x)


class RegionTransformer(nn.Module):
    """
    Transformer for processing region tokens.
    
    Takes N region embeddings and produces N contextualized embeddings.
    Supports variable-length input with indexed positional embeddings.
    """
    
    def __init__(self,
                 num_regions: int = 10,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_regions = num_regions
        self.embed_dim = embed_dim
        
        # Learnable region positional embeddings (indexed by region ID)
        self.region_embed = nn.Parameter(
            torch.zeros(1, num_regions, embed_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.region_embed, std=0.02)
    
    def forward(self, 
                x: torch.Tensor, 
                region_indices: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Region tokens (B, N, embed_dim) where N can be any number <= num_regions
            region_indices: Optional tensor of shape (N,) indicating which region each token is
                           If None, assumes x contains all num_regions in order
            mask: Optional attention mask
        Returns:
            Contextualized tokens (B, N, embed_dim)
        """
        B, N, D = x.shape
        
        # Get positional embeddings
        if region_indices is not None:
            # Index into region embeddings for specific regions
            pos_embed = self.region_embed[:, region_indices, :]  # (1, N, D)
        elif N == self.num_regions:
            # Full regions, use all embeddings
            pos_embed = self.region_embed
        else:
            # Partial regions without indices - just use first N embeddings
            pos_embed = self.region_embed[:, :N, :]
        
        # Add positional embeddings
        x = x + pos_embed
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


class JEPAPredictor(nn.Module):
    """
    JEPA Predictor: Predicts target latents from context latents.
    
    Takes context region embeddings and predicts the missing target embeddings.
    Uses a smaller transformer to predict masked positions.
    """
    
    def __init__(self,
                 num_regions: int = 10,
                 embed_dim: int = 256,
                 predictor_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_regions = num_regions
        self.predictor_dim = predictor_dim
        
        # Project to predictor dimension
        self.proj_in = nn.Linear(embed_dim, predictor_dim)
        
        # Learnable mask tokens (for predicting masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        
        # Region positional embeddings
        self.region_embed = nn.Parameter(
            torch.zeros(1, num_regions, predictor_dim)
        )
        
        # Predictor transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Project back to embed dimension
        self.proj_out = nn.Sequential(
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.region_embed, std=0.02)
    
    def forward(self, 
                context_embed: torch.Tensor, 
                context_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict target latents from context.
        
        Args:
            context_embed: Context region embeddings (B, num_context, embed_dim)
            context_mask: Binary mask indicating context (1) vs target (0) positions
                          Shape: (B, num_regions)
        Returns:
            Predicted target embeddings (B, num_targets, embed_dim)
        """
        B, N = context_mask.shape
        device = context_embed.device
        
        # Project context embeddings
        context_proj = self.proj_in(context_embed)  # (B, num_context, predictor_dim)
        
        # Expand mask tokens for target positions
        num_targets = (~context_mask.bool()).sum(dim=1).max().item()
        
        # Create full sequence with mask tokens at target positions
        # Use same dtype as context_proj for FP16 compatibility
        full_seq = self.mask_token.expand(B, N, -1).clone().to(context_proj.dtype)  # (B, N, predictor_dim)
        
        # Place context embeddings at context positions
        for i in range(B):
            ctx_idx = context_mask[i].bool()
            full_seq[i, ctx_idx] = context_proj[i]
        
        # Add positional embeddings
        full_seq = full_seq + self.region_embed
        
        # Transformer prediction
        predicted = self.transformer(full_seq)  # (B, N, predictor_dim)
        
        # Project back
        predicted = self.proj_out(predicted)  # (B, N, embed_dim)
        
        # Extract only target predictions
        target_preds = []
        for i in range(B):
            tgt_idx = ~context_mask[i].bool()
            target_preds.append(predicted[i, tgt_idx])  # (num_targets_i, embed_dim)
        
        # Stack (assuming same num_targets per sample)
        target_preds = torch.stack(target_preds, dim=0)  # (B, num_targets, embed_dim)
        
        return target_preds


class JEPAPretrainingModel(nn.Module):
    """
    Stage 1: True JEPA Self-supervised Pretraining.
    
    Architecture:
        - Context Encoder: Encodes visible/context EEG regions
        - Target Encoder: EMA copy, encodes target EEG regions (frozen)
        - Predictor: Predicts target latents from context latents
        
    Training objective:
        L2 loss between predicted and target latents (NO DECODER)
    """
    
    def __init__(self,
                 num_channels: int = 105,
                 num_features: int = 500,
                 embed_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_predictor_layers: int = 4,
                 predictor_dim: int = 128,
                 context_ratio: float = 0.6,
                 ema_momentum: float = 0.996,
                 brain_regions: Dict[str, List[int]] = None):
        super().__init__()
        
        self.brain_regions = brain_regions or BRAIN_REGIONS
        self.num_regions = len(self.brain_regions)
        self.context_ratio = context_ratio
        self.ema_momentum = ema_momentum
        self.embed_dim = embed_dim
        
        # Region patch encoders (one per brain region)
        self.region_encoders = nn.ModuleDict({
            name: RegionPatchEncoder(len(channels), num_features, embed_dim)
            for name, channels in self.brain_regions.items()
        })
        
        # Context encoder (trainable)
        self.context_transformer = RegionTransformer(
            num_regions=self.num_regions,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers
        )
        
        # Target encoder (EMA copy, frozen)
        self.target_encoders = copy.deepcopy(self.region_encoders)
        self.target_transformer = copy.deepcopy(self.context_transformer)
        
        # Freeze target encoder
        for param in self.target_encoders.parameters():
            param.requires_grad = False
        for param in self.target_transformer.parameters():
            param.requires_grad = False
        
        # Predictor
        self.predictor = JEPAPredictor(
            num_regions=self.num_regions,
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            num_layers=num_predictor_layers
        )
    
    def _encode_regions(self, 
                        x: torch.Tensor, 
                        encoders: nn.ModuleDict,
                        region_indices: List[int] = None) -> torch.Tensor:
        """
        Encode EEG into region embeddings.
        
        Args:
            x: EEG tensor (B, 105, num_features)
            encoders: Region encoder modules
            region_indices: Optional subset of regions to encode
        Returns:
            Region embeddings (B, num_regions or len(region_indices), embed_dim)
        """
        B = x.size(0)
        region_names = list(self.brain_regions.keys())
        
        if region_indices is None:
            region_indices = list(range(self.num_regions))
        
        embeddings = []
        for idx in region_indices:
            name = region_names[idx]
            channels = self.brain_regions[name]
            region_x = x[:, channels, :]  # (B, num_region_channels, num_features)
            embed = encoders[name](region_x)  # (B, embed_dim)
            embeddings.append(embed)
        
        return torch.stack(embeddings, dim=1)  # (B, num_regions, embed_dim)
    
    def create_context_target_mask(self, 
                                    batch_size: int, 
                                    device: torch.device) -> torch.Tensor:
        """
        Create random context/target split.
        
        Returns:
            context_mask: Binary mask where 1 = context, 0 = target
                          Shape: (B, num_regions)
        """
        num_context = int(self.num_regions * self.context_ratio)
        
        # Create masks for each sample
        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(self.num_regions, device=device)
            mask = torch.zeros(self.num_regions, device=device)
            mask[perm[:num_context]] = 1
            masks.append(mask)
        
        return torch.stack(masks, dim=0)  # (B, num_regions)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        JEPA forward pass with latent prediction.
        
        Args:
            x: EEG tensor (B, 105, num_features)
            
        Returns:
            predicted_latent: Predicted target latents (B, num_targets, embed_dim)
            target_latent: Target latents from EMA encoder (B, num_targets, embed_dim)
            loss: L2 loss in latent space
        """
        B = x.size(0)
        device = x.device
        
        # Create context/target split
        context_mask = self.create_context_target_mask(B, device)  # (B, num_regions)
        
        # Get context and target indices
        # Note: context_mask is 1 for context, 0 for target
        
        # Encode ALL regions first (more efficient)
        # Context encoder path
        all_context_embed = self._encode_regions(x, self.region_encoders)  # (B, N, D)
        
        # Target encoder path (no gradient)
        with torch.no_grad():
            all_target_embed = self._encode_regions(x, self.target_encoders)  # (B, N, D)
        
        # Extract context embeddings for transformer
        # Use first sample's context indices for all (same mask structure per batch)
        first_ctx_mask = context_mask[0].bool()
        context_indices = torch.where(first_ctx_mask)[0]  # (num_ctx,)
        
        context_embeds = []
        for i in range(B):
            ctx_idx = context_mask[i].bool()
            context_embeds.append(all_context_embed[i, ctx_idx])  # (num_ctx, D)
        context_embeds = torch.stack(context_embeds, dim=0)  # (B, num_ctx, D)
        
        # Context transformer with proper region indices
        context_transformed = self.context_transformer(
            context_embeds, region_indices=context_indices
        )  # (B, num_ctx, D)
        
        # Target transformer (no gradient)
        with torch.no_grad():
            all_target_transformed = self.target_transformer(all_target_embed)  # (B, N, D)
            
            # Extract target latents
            target_latents = []
            for i in range(B):
                tgt_idx = ~context_mask[i].bool()
                target_latents.append(all_target_transformed[i, tgt_idx])  # (num_tgt, D)
            target_latents = torch.stack(target_latents, dim=0)  # (B, num_tgt, D)
        
        # Predict target latents from context
        predicted_latents = self.predictor(context_transformed, context_mask)  # (B, num_tgt, D)
        
        # ============================================================
        # VICReg-style loss (Variance-Invariance-Covariance Regularization)
        # STRONGER VERSION - no L2 normalization, original VICReg weights
        # ============================================================
        
        # DO NOT normalize - let VICReg handle the scale
        # Flatten to (N, D) for loss computation
        pred_flat = predicted_latents.reshape(-1, self.embed_dim)  # (B*num_tgt, D)
        target_flat = target_latents.detach().reshape(-1, self.embed_dim)  # (B*num_tgt, D)
        
        # 1. Invariance loss (MSE between predicted and target)
        # Normalize only for MSE to keep scale reasonable
        pred_norm = F.normalize(pred_flat, p=2, dim=-1)
        target_norm = F.normalize(target_flat, p=2, dim=-1)
        mse_loss = F.mse_loss(pred_norm, target_norm)
        
        # 2. Variance loss (encourage variance >= 1 per dimension)
        # Use standard deviation, target std = 1
        pred_std = torch.sqrt(pred_flat.var(dim=0) + 1e-4)  # (D,)
        var_loss = torch.mean(F.relu(1 - pred_std))  # Hinge loss
        
        # 3. Covariance loss (decorrelate dimensions)
        pred_centered = pred_flat - pred_flat.mean(dim=0)
        cov = (pred_centered.T @ pred_centered) / (pred_flat.size(0) - 1)  # (D, D)
        # Off-diagonal elements should be zero
        off_diag_mask = ~torch.eye(self.embed_dim, device=cov.device).bool()
        cov_loss = cov[off_diag_mask].pow(2).mean()
        
        # Combined loss with ORIGINAL VICReg weights (strong regularization)
        # VICReg original: lambda=25 (invariance), mu=25 (variance), nu=1 (covariance)
        inv_weight = 25.0   # Invariance (MSE) weight
        var_weight = 25.0   # Variance weight - STRONG
        cov_weight = 1.0    # Covariance weight
        
        loss = inv_weight * mse_loss + var_weight * var_loss + cov_weight * cov_loss
        
        return predicted_latents, target_latents, loss
    
    @torch.no_grad()
    def update_target_encoder(self, momentum: float = None):
        """
        EMA update of target encoder.
        
        target = momentum * target + (1 - momentum) * context
        """
        if momentum is None:
            momentum = self.ema_momentum
        
        # Update region encoders
        for name in self.brain_regions.keys():
            for p_ctx, p_tgt in zip(self.region_encoders[name].parameters(), 
                                     self.target_encoders[name].parameters()):
                p_tgt.data = momentum * p_tgt.data + (1 - momentum) * p_ctx.data
        
        # Update transformer
        for p_ctx, p_tgt in zip(self.context_transformer.parameters(), 
                                 self.target_transformer.parameters()):
            p_tgt.data = momentum * p_tgt.data + (1 - momentum) * p_ctx.data
    
    def get_encoder(self) -> nn.Module:
        """Return encoder for transfer to Stage 2."""
        class CombinedEncoder(nn.Module):
            def __init__(self, region_encoders, transformer, brain_regions):
                super().__init__()
                self.region_encoders = region_encoders
                self.transformer = transformer
                self.brain_regions = brain_regions
            
            def forward(self, x):
                B = x.size(0)
                embeddings = []
                for name, channels in self.brain_regions.items():
                    region_x = x[:, channels, :]
                    embed = self.region_encoders[name](region_x)
                    embeddings.append(embed)
                features = torch.stack(embeddings, dim=1)
                features = self.transformer(features)
                return features.mean(dim=1)  # Global average
        
        return CombinedEncoder(
            self.region_encoders, 
            self.context_transformer,
            self.brain_regions
        )
