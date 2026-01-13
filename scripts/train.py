"""
Main training script for EEG2TEXT JEPA.

Usage:
    # Prepare data first
    python scripts/prepare_pytorch_data.py --input_dir <ZUCO_DIR> --output_dir ./data/processed
    
    # Run all stages
    python scripts/train.py --stage all
    
    # Run specific stage
    python scripts/train.py --stage 1  # Pretraining
    python scripts/train.py --stage 2  # Alignment
    python scripts/train.py --stage 3  # Decoder
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.zuco_dataset import get_dataloaders
from models.pretraining_model import PretrainingModel
from models.eeg2text_jepa import EEG2TextJEPA
from models.decoder import EEG2TextDecoder
from training.stage1_pretrain import train_stage1, evaluate_pretraining
from training.stage2_alignment import train_stage2, evaluate_alignment
from training.stage3_decoder import train_stage3, evaluate_decoder


def run_stage1(args, train_loader, val_loader, test_loader):
    """Run Stage 1: Self-supervised pretraining."""
    print("\n" + "="*70)
    print("STAGE 1: SELF-SUPERVISED PRETRAINING")
    print("="*70)
    
    model = PretrainingModel(
        num_channels=105,
        num_bands=8,
        embed_dim=args.embed_dim,
        num_transformer_layers=4,
        mask_ratio=0.15
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs_stage1,
        lr=args.lr_stage1,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = evaluate_pretraining(model, test_loader, args.device)
    print(f"  Test MSE: {test_metrics['mse']:.4f}")
    print(f"  Test Correlation: {test_metrics['correlation']:.4f}")
    print(f"  Test SNR: {test_metrics['snr_db']:.2f} dB")
    
    return model


def run_stage2(args, train_loader, val_loader, test_loader, pretrained_path=None):
    """Run Stage 2: EEG-Text alignment."""
    print("\n" + "="*70)
    print("STAGE 2: EEG-TEXT ALIGNMENT (VL-JEPA)")
    print("="*70)
    
    model = EEG2TextJEPA(
        embed_dim=args.embed_dim,
        text_encoder_name=args.text_encoder,
        pretrained_path=pretrained_path
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    model = train_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs_stage2,
        lr=args.lr_stage2,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = evaluate_alignment(model, test_loader, args.device)
    print(f"  Test Cosine Sim: {test_metrics['cosine_similarity']:.4f}")
    print(f"  Test Acc@1: {test_metrics['acc@1']:.4f}")
    print(f"  Test Acc@5: {test_metrics['acc@5']:.4f}")
    print(f"  Test MRR: {test_metrics['mrr']:.4f}")
    
    return model


def run_stage3(args, train_loader, val_loader, test_loader, jepa_model=None):
    """Run Stage 3: Text decoder fine-tuning."""
    print("\n" + "="*70)
    print("STAGE 3: TEXT DECODER FINE-TUNING")
    print("="*70)
    
    # Load JEPA model if not provided
    if jepa_model is None:
        jepa_model = EEG2TextJEPA(
            embed_dim=args.embed_dim,
            text_encoder_name=args.text_encoder
        )
        checkpoint = torch.load(Path(args.save_dir) / 'stage2_best.pt')
        jepa_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded Stage 2 checkpoint")
    
    model = EEG2TextDecoder(
        jepa_model=jepa_model.to(args.device),
        bart_model_name=args.bart_model,
        freeze_jepa=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    model = train_stage3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs_stage3,
        lr=args.lr_stage3,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics, preds, refs = evaluate_decoder(model, test_loader, args.device)
    print(f"  Test BLEU-1: {test_metrics.get('bleu-1', 0):.4f}")
    print(f"  Test BLEU-4: {test_metrics.get('bleu-4', 0):.4f}")
    print(f"  Test ROUGE-L: {test_metrics.get('rougeL', 0):.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train EEG2TEXT JEPA')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory with processed .pt files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--text_encoder', type=str, 
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--bart_model', type=str, default='facebook/bart-base')
    
    # Training
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Which stage(s) to run')
    parser.add_argument('--epochs_stage1', type=int, default=100)
    parser.add_argument('--epochs_stage2', type=int, default=50)
    parser.add_argument('--epochs_stage3', type=int, default=20)
    parser.add_argument('--lr_stage1', type=float, default=1e-4)
    parser.add_argument('--lr_stage2', type=float, default=5e-5)
    parser.add_argument('--lr_stage3', type=float, default=1e-5)
    
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
    
    # Run stages
    if args.stage == 'all' or args.stage == '1':
        stage1_model = run_stage1(args, train_loader, val_loader, test_loader)
    
    if args.stage == 'all' or args.stage == '2':
        pretrained_path = Path(args.save_dir) / 'stage1_best.pt' if args.stage == 'all' else None
        stage2_model = run_stage2(args, train_loader, val_loader, test_loader, 
                                   pretrained_path=str(pretrained_path) if pretrained_path and pretrained_path.exists() else None)
    
    if args.stage == 'all' or args.stage == '3':
        jepa_model = stage2_model if args.stage == 'all' else None
        stage3_model = run_stage3(args, train_loader, val_loader, test_loader, jepa_model)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
