"""
Stage 1: Self-supervised pretraining with masked EEG reconstruction.

Usage:
    # Auto-detect GPU and use optimal settings
    python scripts/train_stage1.py --gpu_profile auto --epochs 100
    
    # Manually specify GPU profile
    python scripts/train_stage1.py --gpu_profile t4 --epochs 100
    python scripts/train_stage1.py --gpu_profile a4000 --epochs 100
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.zuco_dataset import get_dataloaders
from models.pretraining_model import PretrainingModel
from training.stage1_pretrain import train_stage1, evaluate_pretraining
from utils.gpu_profiles import (
    get_gpu_profile, 
    auto_detect_gpu_profile, 
    apply_gpu_optimizations,
    print_profile_info
)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Self-supervised Pretraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Profiles:
  t4      - Tesla T4 (15GB) - Google Colab, most cloud
  a4000   - NVIDIA A4000 Ada (16GB) - Workstations
  a100    - NVIDIA A100 (40GB) - High-end compute
  auto    - Auto-detect GPU and select profile
  
Examples:
  python scripts/train_stage1.py --gpu_profile t4 --epochs 100
  python scripts/train_stage1.py --gpu_profile auto --epochs 100
        """
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
    parser.add_argument('--num_transformer_layers', type=int, default=None,
                        help='Override profile transformer layers')
    parser.add_argument('--fp16', action='store_true', default=None,
                        help='Force enable FP16')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Force disable FP16')
    
    # Model
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    
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
    
    # Apply GPU optimizations
    apply_gpu_optimizations(args.device, profile)
    
    # Use profile defaults, allow overrides
    batch_size = args.batch_size if args.batch_size is not None else profile.batch_size
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else profile.grad_accum_steps
    embed_dim = args.embed_dim if args.embed_dim is not None else profile.embed_dim
    num_transformer_layers = args.num_transformer_layers if args.num_transformer_layers is not None else profile.num_transformer_layers
    num_workers = args.num_workers if args.num_workers is not None else profile.num_workers
    
    # Handle FP16 flags
    if args.no_fp16:
        use_fp16 = False
    elif args.fp16:
        use_fp16 = True
    else:
        use_fp16 = profile.use_fp16
    
    # Print profile info
    print_profile_info(profile)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Get EEG shape from first batch
    sample_batch = next(iter(train_loader))
    eeg_shape = sample_batch['eeg'].shape
    print(f"EEG shape: {eeg_shape}")  # (B, channels, features)
    
    num_channels = eeg_shape[1]
    num_features = eeg_shape[2]
    
    # Create model
    print("\n" + "="*60)
    print("STAGE 1: SELF-SUPERVISED PRETRAINING")
    print("="*60)
    
    model = PretrainingModel(
        num_channels=num_channels,
        num_bands=num_features,
        embed_dim=embed_dim,
        num_transformer_layers=num_transformer_layers,
        mask_ratio=args.mask_ratio
    )
    
    # Apply torch.compile if enabled and available (PyTorch 2.0+)
    if profile.use_compile and hasattr(torch, 'compile'):
        print("Applying torch.compile() optimization...")
        model = torch.compile(model, mode='reduce-overhead')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")
    
    # Determine resume path
    resume_path = None
    if args.resume:
        checkpoint_path = Path(args.save_dir) / 'stage1_best.pt'
        if checkpoint_path.exists():
            resume_path = str(checkpoint_path)
        else:
            print("⚠ No checkpoint found to resume from, starting fresh")
    
    # Train
    model = train_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        grad_accum_steps=grad_accum_steps,
        use_fp16=use_fp16,
        use_gradient_checkpointing=profile.use_gradient_checkpointing,
        max_grad_norm=profile.max_grad_norm,
        resume_path=resume_path
    )
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_pretraining(model, test_loader, args.device)
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test Correlation: {test_metrics['correlation']:.4f}")
    print(f"Test SNR: {test_metrics['snr_db']:.2f} dB")
    
    print(f"\n✓ Stage 1 complete! Checkpoint saved to: {args.save_dir}/stage1_best.pt")
    print("\nNext step: Run train_stage2.py")


if __name__ == '__main__':
    main()
