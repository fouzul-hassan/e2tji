#!/usr/bin/env python3
"""
Stage 1: JEPA Self-Supervised Pretraining CLI.

This is TRUE JEPA - predicts in latent space with EMA target encoder.
For MAE (pixel-space reconstruction), use train_stage1_mae.py.

Usage:
    python scripts/train_stage1.py --gpu_profile t4 --epochs 100
    python scripts/train_stage1.py --gpu_profile rtx4000 --epochs 100 --ema_momentum 0.996
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from models.pretraining_jepa import JEPAPretrainingModel
from training.stage1_jepa import train_stage1_jepa, evaluate_jepa
from data.zuco_dataset import get_dataloaders
from utils.gpu_profiles import (
    get_gpu_profile, 
    auto_detect_gpu_profile, 
    apply_gpu_optimizations,
    print_profile_info
)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: JEPA Self-Supervised Pretraining (Latent Prediction)'
    )
    
    # GPU Profile
    parser.add_argument('--gpu_profile', type=str, default='auto',
                        choices=['auto', 't4', 'a4000', 'rtx4000', 'a100', 'cpu'],
                        help='GPU optimization profile (default: auto-detect)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory with processed .pt files')
    
    # Training (can override profile defaults)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Override options (optional, uses profile defaults if not set)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override profile batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=None,
                        help='Override profile gradient accumulation')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Override profile embedding dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=None,
                        help='Override profile encoder layers')
    parser.add_argument('--fp16', action='store_true', default=None,
                        help='Force enable FP16')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Force disable FP16')
    
    # JEPA-specific
    parser.add_argument('--ema_momentum', type=float, default=0.996,
                        help='EMA momentum for target encoder')
    parser.add_argument('--context_ratio', type=float, default=0.6,
                        help='Ratio of context regions (vs target)')
    parser.add_argument('--predictor_dim', type=int, default=128,
                        help='Predictor hidden dimension')
    parser.add_argument('--num_predictor_layers', type=int, default=4,
                        help='Number of predictor transformer layers')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (epochs without improvement, 0=disabled)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get GPU profile
    if args.gpu_profile == 'auto':
        profile = auto_detect_gpu_profile()
    else:
        profile = get_gpu_profile(args.gpu_profile)
    
    print_profile_info(profile)
    
    # Apply GPU optimizations
    apply_gpu_optimizations(args.device, profile)
    
    # Get settings (use overrides if provided, else profile defaults)
    batch_size = args.batch_size or profile.batch_size
    grad_accum_steps = args.grad_accum_steps or profile.grad_accum_steps
    embed_dim = args.embed_dim or profile.embed_dim
    num_encoder_layers = args.num_encoder_layers or profile.num_transformer_layers
    num_workers = args.num_workers or profile.num_workers
    
    # Handle FP16
    if args.no_fp16:
        use_fp16 = False
    elif args.fp16:
        use_fp16 = True
    else:
        use_fp16 = profile.use_fp16
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Get feature dimensions from data
    sample = next(iter(train_loader))
    num_channels = sample['eeg'].shape[1]
    num_features = sample['eeg'].shape[2]
    print(f"EEG shape: {sample['eeg'].shape}")
    
    print("\n" + "="*60)
    print("STAGE 1: JEPA SELF-SUPERVISED PRETRAINING")
    print("="*60)
    
    model = JEPAPretrainingModel(
        num_channels=num_channels,
        num_features=num_features,
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_predictor_layers=args.num_predictor_layers,
        predictor_dim=args.predictor_dim,
        context_ratio=args.context_ratio,
        ema_momentum=args.ema_momentum
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")
    
    # Determine resume path
    resume_path = None
    if args.resume:
        checkpoint_path = Path(args.save_dir) / 'stage1_jepa_best.pt'
        if checkpoint_path.exists():
            resume_path = str(checkpoint_path)
        else:
            print("⚠ No checkpoint found to resume from, starting fresh")
    
    # Train
    model = train_stage1_jepa(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        grad_accum_steps=grad_accum_steps,
        use_fp16=use_fp16,
        ema_momentum=args.ema_momentum,
        max_grad_norm=profile.max_grad_norm,
        resume_path=resume_path,
        early_stopping_patience=args.early_stopping
    )
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    test_metrics = evaluate_jepa(model, test_loader, args.device)
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test Cosine Similarity: {test_metrics['cosine_sim']:.4f}")
    
    print("\n✓ Stage 1 JEPA pretraining complete!")
    print(f"  Best model saved to: {args.save_dir}/stage1_jepa_best.pt")


if __name__ == '__main__':
    main()
