"""
Stage 2: EEG-Text alignment with VL-JEPA objective.

Usage:
    # Auto-detect GPU and use optimal settings
    python scripts/train_stage2.py --gpu_profile auto --epochs 50
    
    # Manually specify GPU profile
    python scripts/train_stage2.py --gpu_profile t4 --epochs 50
    python scripts/train_stage2.py --gpu_profile a4000 --epochs 50
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.zuco_dataset import get_dataloaders
from models.eeg2text_jepa import EEG2TextJEPA
from training.stage2_alignment import train_stage2, evaluate_alignment
from utils.gpu_profiles import (
    get_gpu_profile, 
    auto_detect_gpu_profile, 
    apply_gpu_optimizations,
    print_profile_info
)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: EEG-Text Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Profiles:
  t4      - Tesla T4 (15GB) - Google Colab, most cloud
  a4000   - NVIDIA A4000 Ada (16GB) - Workstations
  a100    - NVIDIA A100 (40GB) - High-end compute
  auto    - Auto-detect GPU and select profile
        """
    )
    
    # GPU Profile
    parser.add_argument('--gpu_profile', type=str, default='auto',
                        choices=['auto', 't4', 'a4000', 'rtx4000', 'a100', 'cpu'],
                        help='GPU optimization profile (default: auto-detect)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory with processed .pt files')
    
    # Model
    parser.add_argument('--text_encoder', type=str, 
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to Stage 1 checkpoint (optional)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    
    # Override options
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
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
    embed_dim = args.embed_dim if args.embed_dim is not None else profile.embed_dim
    num_workers = args.num_workers if args.num_workers is not None else profile.num_workers
    
    # Print profile info
    print_profile_info(profile)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for Stage 1 checkpoint
    stage1_path = Path(args.save_dir) / 'stage1_best.pt'
    if args.pretrained_path is None and stage1_path.exists():
        args.pretrained_path = str(stage1_path)
        print(f"Found Stage 1 checkpoint: {args.pretrained_path}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Detect EEG shape from first batch
    sample_batch = next(iter(train_loader))
    eeg_shape = sample_batch['eeg'].shape
    num_features = eeg_shape[2]  # (B, 105, num_features)
    print(f"EEG shape: {eeg_shape}")
    print(f"Detected num_features: {num_features}")
    
    # Create model
    print("\n" + "="*60)
    print("STAGE 2: EEG-TEXT ALIGNMENT (VL-JEPA)")
    print("="*60)
    
    model = EEG2TextJEPA(
        embed_dim=embed_dim,
        num_features=num_features,
        text_encoder_name=args.text_encoder,
        pretrained_path=args.pretrained_path
    )
    
    # Apply torch.compile if enabled
    if profile.use_compile and hasattr(torch, 'compile'):
        print("Applying torch.compile() optimization...")
        model = torch.compile(model, mode='reduce-overhead')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    model = train_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        use_fp16=profile.use_fp16,
        grad_accum_steps=profile.grad_accum_steps,
        max_grad_norm=profile.max_grad_norm
    )
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_alignment(model, test_loader, args.device)
    print(f"Test Cosine Similarity: {test_metrics['cosine_similarity']:.4f}")
    print(f"Test Acc@1: {test_metrics['acc@1']:.4f}")
    print(f"Test Acc@5: {test_metrics['acc@5']:.4f}")
    print(f"Test Acc@10: {test_metrics['acc@10']:.4f}")
    print(f"Test MRR: {test_metrics['mrr']:.4f}")
    
    print(f"\n✓ Stage 2 complete! Checkpoint saved to: {args.save_dir}/stage2_best.pt")
    print("\nNext step: Run train_stage3.py (optional, for text generation)")


if __name__ == '__main__':
    main()
