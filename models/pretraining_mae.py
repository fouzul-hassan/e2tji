"""
Stage 1: Self-supervised pretraining model with masked EEG reconstruction.

Architecture:
    Masked EEG -> CNN Encoder -> Conv Transformer -> CNN Decoder -> Reconstructed EEG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CNNEncoder(nn.Module):
    """
    CNN Encoder for compressing EEG signals.
    
    Input: (batch, 105, 8) - channels x frequency_bands
    Output: (batch, embed_dim)
    """
    
    def __init__(self, 
                 num_channels: int = 105, 
                 num_bands: int = 8, 
                 embed_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Conv over channels and bands (treat as 2D image)
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        # Calculate output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_bands)
            out = self.encoder(dummy)
            self.feat_shape = out.shape[1:]  # (C, H, W)
            self.feat_size = out.view(1, -1).size(1)
        
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG tensor (B, 105, 8)
        Returns:
            Encoded features (B, embed_dim)
        """
        x = x.unsqueeze(1)  # (B, 1, 105, 8)
        x = self.encoder(x)
        x = self.proj(x)
        return x


class CNNDecoder(nn.Module):
    """
    CNN Decoder for reconstructing EEG signals.
    
    Input: (batch, embed_dim)
    Output: (batch, 105, 8)
    """
    
    def __init__(self, 
                 num_channels: int = 105, 
                 num_bands: int = 8, 
                 embed_dim: int = 256,
                 encoder_feat_shape: Tuple[int, int, int] = (256, 14, 8)):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.feat_shape = encoder_feat_shape
        
        feat_size = encoder_feat_shape[0] * encoder_feat_shape[1] * encoder_feat_shape[2]
        
        self.expand = nn.Sequential(
            nn.Linear(embed_dim, feat_size),
            nn.LayerNorm(feat_size),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(5, 3), stride=(2, 1), 
                              padding=(2, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 3), stride=(2, 1), 
                              padding=(2, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.ConvTranspose2d(64, 1, kernel_size=(5, 3), stride=(2, 1), 
                              padding=(2, 1), output_padding=(1, 0)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features (B, embed_dim)
        Returns:
            Reconstructed EEG (B, 105, 8)
        """
        B = x.size(0)
        x = self.expand(x)
        x = x.view(B, *self.feat_shape)  # (B, 256, H, W)
        x = self.decoder(x)  # (B, 1, ~105, ~8)
        
        # Interpolate to exact target size
        x = F.interpolate(x, size=(self.num_channels, self.num_bands), 
                         mode='bilinear', align_corners=False)
        
        return x.squeeze(1)  # (B, 105, 8)


class ConvTransformer(nn.Module):
    """
    Transformer encoder for processing EEG features.
    """
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_layers: int = 4, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, embed_dim)
        Returns:
            Transformed features (B, embed_dim)
        """
        x = x.unsqueeze(1)  # (B, 1, embed_dim)
        x = self.transformer(x)
        x = self.norm(x)
        return x.squeeze(1)  # (B, embed_dim)


class PretrainingModel(nn.Module):
    """
    Stage 1: Self-supervised pretraining with masked EEG reconstruction.
    
    Training objective: Reconstruct masked EEG signals
    """
    
    def __init__(self,
                 num_channels: int = 105,
                 num_bands: int = 8,
                 embed_dim: int = 256,
                 num_transformer_layers: int = 4,
                 mask_ratio: float = 0.15,
                 mask_type: str = 'channel'):  # 'channel' or 'random'
        super().__init__()
        
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        
        # Encoder
        self.encoder = CNNEncoder(num_channels, num_bands, embed_dim)
        
        # Transformer
        self.transformer = ConvTransformer(embed_dim, num_layers=num_transformer_layers)
        
        # Decoder
        self.decoder = CNNDecoder(
            num_channels, num_bands, embed_dim,
            encoder_feat_shape=self.encoder.feat_shape
        )
    
    def create_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create mask for EEG signals.
        
        Args:
            x: EEG tensor (B, 105, 8)
            
        Returns:
            masked_x: Masked EEG (B, 105, 8)
            mask: Binary mask (B, 105) indicating masked channels
        """
        B, C, F = x.shape
        device = x.device
        
        if self.mask_type == 'channel':
            # Mask entire channels
            num_mask = int(C * self.mask_ratio)
            mask = torch.zeros(B, C, device=device)
            
            for i in range(B):
                idx = torch.randperm(C, device=device)[:num_mask]
                mask[i, idx] = 1
            
            # Expand mask to cover all frequency bands
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, F)
            
        else:  # random
            # Random element-wise masking
            mask_expanded = (torch.rand(B, C, F, device=device) < self.mask_ratio).float()
            mask = mask_expanded.mean(dim=-1)  # Average for summary
        
        # Apply mask (zero out)
        masked_x = x * (1 - mask_expanded)
        
        return masked_x, mask
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with reconstruction.
        
        Args:
            x: EEG tensor (B, 105, 8)
            
        Returns:
            reconstructed: Reconstructed EEG (B, 105, 8)
            loss: Reconstruction MSE loss
            mask: Applied mask (B, 105)
        """
        # Create mask
        masked_x, mask = self.create_mask(x)
        
        # Encode
        encoded = self.encoder(masked_x)
        
        # Transform
        transformed = self.transformer(encoded)
        
        # Decode
        reconstructed = self.decoder(transformed)
        
        # Reconstruction loss (full signal)
        loss = F.mse_loss(reconstructed, x)
        
        return reconstructed, loss, mask
    
    def get_encoder(self) -> nn.Module:
        """Return encoder for transfer to Stage 2."""
        return nn.Sequential(self.encoder, self.transformer)
