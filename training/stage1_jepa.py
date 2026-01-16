"""
Stage 1: JEPA Training Function.

Trains JEPAPretrainingModel with:
- EMA target encoder updates
- L2 loss in latent space
- Mixed precision support
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pretraining_jepa import JEPAPretrainingModel


def evaluate_jepa(
    model: JEPAPretrainingModel,
    val_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate JEPA model on validation set.
    
    Returns:
        Dictionary with metrics: mse, cosine_sim
    """
    model.eval()
    
    total_mse = 0
    total_cosine = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating JEPA"):
            eeg = batch['eeg'].to(device)
            predicted, target, loss = model(eeg)
            
            total_mse += loss.item()
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                predicted.reshape(-1, predicted.size(-1)),
                target.reshape(-1, target.size(-1)),
                dim=-1
            ).mean().item()
            total_cosine += cos_sim
            
            num_batches += 1
    
    return {
        'mse': total_mse / num_batches,
        'cosine_sim': total_cosine / num_batches,
    }


def train_stage1_jepa(
    model: JEPAPretrainingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    log_interval: int = 10,
    grad_accum_steps: int = 1,
    use_fp16: bool = False,
    ema_momentum: float = 0.996,
    max_grad_norm: float = 1.0,
    resume_path: str = None,
    early_stopping_patience: int = 10
) -> JEPAPretrainingModel:
    """
    Train Stage 1 JEPA pretraining model.
    
    Args:
        model: JEPAPretrainingModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Log every N batches
        grad_accum_steps: Gradient accumulation steps
        use_fp16: Use mixed precision training
        ema_momentum: Momentum for EMA target encoder update
        max_grad_norm: Maximum gradient norm for clipping
        resume_path: Path to checkpoint to resume from
        early_stopping_patience: Stop if no improvement for N epochs (0 = disabled)
        
    Returns:
        Trained model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    # Optimizer (only context encoder + predictor, not target encoder)
    trainable_params = []
    for name, param in model.named_parameters():
        if 'target' not in name:  # Skip target encoder
            trainable_params.append(param)
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=lr, 
        weight_decay=0.05,  # JEPA typically uses higher weight decay
        betas=(0.9, 0.95),
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_mse': [], 'val_cosine': []}
    
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_metrics'].get('mse', float('inf'))
        
        # Load history if available
        history_path = save_dir / 'stage1_jepa_history.pt'
        if history_path.exists():
            history = torch.load(history_path, weights_only=False)
        
        print(f"  Resumed from epoch {start_epoch}, best loss: {best_val_loss:.4f}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, 
        last_epoch=start_epoch-1 if start_epoch > 0 else -1
    )
    
    # Mixed precision scaler
    if use_fp16 and device == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    effective_batch = train_loader.batch_size * grad_accum_steps
    
    # Early stopping counter
    epochs_without_improvement = 0
    
    print(f"\n{'='*60}")
    print("Stage 1: JEPA Self-Supervised Pretraining")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {start_epoch+1}-{epochs} (total {epochs})")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Gradient accumulation: {grad_accum_steps}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Mixed precision: {use_fp16}")
    print(f"EMA momentum: {ema_momentum}")
    print(f"Context ratio: {model.context_ratio}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg'].to(device)
            
            # Forward
            if use_fp16 and scaler is not None:
                with torch.amp.autocast('cuda'):
                    _, _, loss = model(eeg)
                    loss = loss / grad_accum_steps
                
                scaler.scale(loss).backward()
            else:
                _, _, loss = model(eeg)
                loss = loss / grad_accum_steps
                loss.backward()
            
            train_loss += loss.item() * grad_accum_steps
            
            # Update weights
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                # EMA update of target encoder
                model.update_target_encoder(momentum=ema_momentum)
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item()*grad_accum_steps:.4f}")
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Clear cache before validation
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Validation
        val_metrics = evaluate_jepa(model, val_loader, device)
        history['val_mse'].append(val_metrics['mse'])
        history['val_cosine'].append(val_metrics['cosine_sim'])
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print(f"  Val Cosine Sim: {val_metrics['cosine_sim']:.4f}")
        
        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / 'stage1_jepa_best.pt')
            print(f"  ✓ Saved best model (MSE: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Early stopping check
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"\n⚠ Early stopping triggered: no improvement for {early_stopping_patience} epochs")
            break
    
    # Load best model
    best_path = save_dir / 'stage1_jepa_best.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Save training history
    torch.save(history, save_dir / 'stage1_jepa_history.pt')
    
    return model
