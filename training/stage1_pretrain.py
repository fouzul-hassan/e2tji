"""
Stage 1: Self-supervised pretraining with masked EEG reconstruction.

Training and evaluation functions.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pretraining_model import PretrainingModel


def evaluate_pretraining(
    model: PretrainingModel,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate Stage 1 pretraining model.
    
    Metrics:
        - MSE: Mean Squared Error
        - Correlation: Pearson correlation between original and reconstructed
        - SNR: Signal-to-Noise Ratio in dB
    """
    model.eval()
    
    total_mse = 0
    total_corr = 0
    total_snr = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Stage 1"):
            eeg = batch['eeg'].to(device)  # (B, 105, 8)
            
            reconstructed, loss, mask = model(eeg)
            
            # MSE
            total_mse += loss.item() * eeg.size(0)
            
            # Correlation and SNR (per sample)
            eeg_np = eeg.cpu().numpy()
            recon_np = reconstructed.cpu().numpy()
            
            for i in range(eeg.size(0)):
                # Flatten for correlation
                orig_flat = eeg_np[i].flatten()
                recon_flat = recon_np[i].flatten()
                
                # Pearson correlation
                try:
                    corr, _ = pearsonr(orig_flat, recon_flat)
                    if not np.isnan(corr):
                        total_corr += corr
                except:
                    pass
                
                # SNR = 10 * log10(signal_power / noise_power)
                signal_power = np.mean(orig_flat ** 2) + 1e-8
                noise_power = np.mean((orig_flat - recon_flat) ** 2) + 1e-8
                snr = 10 * np.log10(signal_power / noise_power)
                total_snr += snr
                
                n_samples += 1
    
    metrics = {
        'mse': total_mse / n_samples,
        'correlation': total_corr / n_samples,
        'snr_db': total_snr / n_samples
    }
    
    return metrics


def train_stage1(
    model: PretrainingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    log_interval: int = 10,
    grad_accum_steps: int = 1,
    use_fp16: bool = False
) -> PretrainingModel:
    """
    Train Stage 1 pretraining model.
    
    Args:
        model: PretrainingModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Log every N batches
        grad_accum_steps: Gradient accumulation steps
        use_fp16: Use mixed precision training
        
    Returns:
        Trained model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_fp16 and device == 'cuda' else None
    
    best_val_mse = float('inf')
    history = {'train_loss': [], 'val_mse': [], 'val_corr': [], 'val_snr': []}
    
    effective_batch = train_loader.batch_size * grad_accum_steps
    
    print(f"\n{'='*60}")
    print("Stage 1: Self-Supervised Pretraining")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Gradient accumulation: {grad_accum_steps}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Mixed precision: {use_fp16}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg'].to(device)
            
            # Forward pass with optional mixed precision
            if use_fp16 and scaler is not None:
                with torch.cuda.amp.autocast():
                    reconstructed, loss, mask = model(eeg)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                reconstructed, loss, mask = model(eeg)
                loss = loss / grad_accum_steps
                loss.backward()
            
            train_loss += loss.item() * grad_accum_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
        
        # Handle remaining gradients
        if (batch_idx + 1) % grad_accum_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Clear cache before validation
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Validation
        val_metrics = evaluate_pretraining(model, val_loader, device)
        history['val_mse'].append(val_metrics['mse'])
        history['val_corr'].append(val_metrics['correlation'])
        history['val_snr'].append(val_metrics['snr_db'])
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print(f"  Val Correlation: {val_metrics['correlation']:.4f}")
        print(f"  Val SNR: {val_metrics['snr_db']:.2f} dB")
        
        # Save best model
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / 'stage1_best.pt')
            print(f"  ✓ Saved best model (MSE: {best_val_mse:.4f})")
    
    # Load best model
    checkpoint = torch.load(save_dir / 'stage1_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Save training history
    torch.save(history, save_dir / 'stage1_history.pt')
    
    return model
