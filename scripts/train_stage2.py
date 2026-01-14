"""
Stage 2: EEG-Text alignment with VL-JEPA objective.

Usage:
    python scripts/train_stage2.py --data_dir ./data/processed --epochs 50
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.zuco_dataset import get_dataloaders
from models.eeg2text_jepa import EEG2TextJEPA
from training.stage2_alignment import train_stage2, evaluate_alignment


def main():
    parser = argparse.ArgumentParser(description='Stage 2: EEG-Text Alignment')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory with processed .pt files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--text_encoder', type=str, 
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to Stage 1 checkpoint (optional)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    
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
    
    # Check for Stage 1 checkpoint
    stage1_path = Path(args.save_dir) / 'stage1_best.pt'
    if args.pretrained_path is None and stage1_path.exists():
        args.pretrained_path = str(stage1_path)
        print(f"Found Stage 1 checkpoint: {args.pretrained_path}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
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
        embed_dim=args.embed_dim,
        num_features=num_features,
        text_encoder_name=args.text_encoder,
        pretrained_path=args.pretrained_path
    )
    
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
        save_dir=args.save_dir
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
