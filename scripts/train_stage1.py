"""
Stage 1: Self-supervised pretraining with masked EEG reconstruction.

Usage:
    python scripts/train_stage1.py --data_dir ./data/processed --epochs 100
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.zuco_dataset import get_dataloaders
from models.pretraining_model import PretrainingModel
from training.stage1_pretrain import train_stage1, evaluate_pretraining


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Self-supervised Pretraining')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory with processed .pt files')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (reduce for spectro data to avoid OOM)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension (128 for memory efficiency)')
    parser.add_argument('--num_transformer_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Device: {args.device}")
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
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
        embed_dim=args.embed_dim,
        num_transformer_layers=args.num_transformer_layers,
        mask_ratio=args.mask_ratio
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    model = train_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        grad_accum_steps=args.grad_accum_steps,
        use_fp16=args.fp16
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
