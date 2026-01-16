"""
Stage 3: Text decoder fine-tuning with BART.

Usage:
    # Auto-detect GPU and use optimal settings
    python scripts/train_stage3.py --gpu_profile auto --epochs 20
    
    # Manually specify GPU profile
    python scripts/train_stage3.py --gpu_profile t4 --epochs 20
    python scripts/train_stage3.py --gpu_profile a4000 --epochs 20
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.zuco_dataset import get_dataloaders
from models.eeg2text_jepa import EEG2TextJEPA
from models.decoder import EEG2TextDecoder
from training.stage3_decoder import train_stage3, evaluate_decoder
from utils.gpu_profiles import (
    get_gpu_profile, 
    auto_detect_gpu_profile, 
    apply_gpu_optimizations,
    print_profile_info
)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: Text Decoder Fine-tuning',
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
                        choices=['auto', 't4', 'a4000', 'a100', 'cpu'],
                        help='GPU optimization profile (default: auto-detect)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory with processed .pt files')
    
    # Model
    parser.add_argument('--text_encoder', type=str, 
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--bart_model', type=str, default='facebook/bart-base')
    parser.add_argument('--stage2_path', type=str, default=None,
                        help='Path to Stage 2 checkpoint (required)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    
    # Override options
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
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
    
    # Use profile defaults with reduced batch for decoder (more memory intensive)
    # Decoder uses about 2x memory compared to encoder stages
    decoder_batch_size = max(2, profile.batch_size // 2)
    batch_size = args.batch_size if args.batch_size is not None else decoder_batch_size
    embed_dim = args.embed_dim if args.embed_dim is not None else profile.embed_dim
    num_workers = args.num_workers if args.num_workers is not None else profile.num_workers
    
    # Print profile info
    print_profile_info(profile)
    print(f"Decoder batch size (reduced): {batch_size}")
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for Stage 2 checkpoint
    stage2_path = Path(args.save_dir) / 'stage2_best.pt'
    if args.stage2_path is None:
        if stage2_path.exists():
            args.stage2_path = str(stage2_path)
        else:
            raise ValueError("Stage 2 checkpoint not found! Run train_stage2.py first, or specify --stage2_path")
    
    print(f"Loading Stage 2 checkpoint: {args.stage2_path}")
    
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
    
    # Load Stage 2 model
    print("\n" + "="*60)
    print("STAGE 3: TEXT DECODER FINE-TUNING")
    print("="*60)
    
    jepa_model = EEG2TextJEPA(
        embed_dim=embed_dim,
        num_features=num_features,
        text_encoder_name=args.text_encoder
    )
    
    checkpoint = torch.load(args.stage2_path, map_location='cpu', weights_only=False)
    jepa_model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded Stage 2 model")
    
    # Create decoder model
    model = EEG2TextDecoder(
        jepa_model=jepa_model.to(args.device),
        bart_model_name=args.bart_model,
        freeze_jepa=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    model = train_stage3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        use_fp16=profile.use_fp16,
        grad_accum_steps=profile.grad_accum_steps * 2,  # More accumulation for decoder
        max_grad_norm=profile.max_grad_norm
    )
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    test_metrics, predictions, references = evaluate_decoder(model, test_loader, args.device)
    print(f"Test BLEU-1: {test_metrics.get('bleu-1', 0):.4f}")
    print(f"Test BLEU-4: {test_metrics.get('bleu-4', 0):.4f}")
    print(f"Test ROUGE-L: {test_metrics.get('rougeL', 0):.4f}")
    if 'bertscore_f1' in test_metrics:
        print(f"Test BERTScore: {test_metrics['bertscore_f1']:.4f}")
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}] Prediction: {predictions[i]}")
        print(f"    Reference:  {references[i]}")
    
    print(f"\n✓ Stage 3 complete! Checkpoint saved to: {args.save_dir}/stage3_best.pt")
    print("\n🎉 Training pipeline complete!")


if __name__ == '__main__':
    main()
