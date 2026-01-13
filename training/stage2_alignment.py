"""
Stage 2: EEG-Text alignment training with VL-JEPA objective.

Training and evaluation functions.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.eeg2text_jepa import EEG2TextJEPA


def compute_retrieval_metrics(
    pred_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    ks: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    For each predicted embedding, find the closest target embedding
    and check if it's the correct one.
    
    Args:
        pred_embeds: Predicted embeddings (N, D)
        target_embeds: Target embeddings (N, D)
        ks: Top-k values for accuracy
        
    Returns:
        Dictionary with acc@k and MRR
    """
    # Cosine similarity matrix: (N, N)
    sim_matrix = F.cosine_similarity(
        pred_embeds.unsqueeze(1),   # (N, 1, D)
        target_embeds.unsqueeze(0),  # (1, N, D)
        dim=-1
    )  # (N, N)
    
    # Get rankings (descending similarity)
    rankings = sim_matrix.argsort(dim=1, descending=True)
    
    # Ground truth: diagonal (each pred should match its own target)
    N = pred_embeds.size(0)
    correct_indices = torch.arange(N, device=pred_embeds.device)
    
    # Compute accuracies
    metrics = {}
    for k in ks:
        top_k = rankings[:, :k]
        correct = (top_k == correct_indices.unsqueeze(1)).any(dim=1)
        metrics[f'acc@{k}'] = correct.float().mean().item()
    
    # Mean Reciprocal Rank
    # Find rank of correct answer for each sample
    ranks = (rankings == correct_indices.unsqueeze(1)).float().argmax(dim=1) + 1
    mrr = (1.0 / ranks.float()).mean().item()
    metrics['mrr'] = mrr
    
    return metrics


def evaluate_alignment(
    model: EEG2TextJEPA,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate Stage 2 alignment model.
    
    Metrics:
        - Cosine Similarity (mean)
        - Retrieval Accuracy @1, @5, @10
        - Mean Reciprocal Rank (MRR)
    """
    model.eval()
    
    all_pred_embeds = []
    all_target_embeds = []
    total_loss = 0
    total_cos_sim = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Stage 2"):
            eeg = batch['eeg'].to(device)
            texts = batch['text']
            
            pred_embed, target_embed, loss = model(eeg, texts)
            
            all_pred_embeds.append(pred_embed.cpu())
            all_target_embeds.append(target_embed.cpu())
            
            total_loss += loss.item()
            cos_sim = F.cosine_similarity(pred_embed, target_embed).mean().item()
            total_cos_sim += cos_sim
            n_batches += 1
    
    # Stack all embeddings
    all_pred_embeds = torch.cat(all_pred_embeds, dim=0)
    all_target_embeds = torch.cat(all_target_embeds, dim=0)
    
    # Retrieval metrics (on subset if too large)
    max_samples = min(len(all_pred_embeds), 1000)
    retrieval_metrics = compute_retrieval_metrics(
        all_pred_embeds[:max_samples],
        all_target_embeds[:max_samples]
    )
    
    metrics = {
        'loss': total_loss / n_batches,
        'cosine_similarity': total_cos_sim / n_batches,
        **retrieval_metrics
    }
    
    return metrics


def train_stage2(
    model: EEG2TextJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 5e-5,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    log_interval: int = 10
) -> EEG2TextJEPA:
    """
    Train Stage 2 alignment model.
    
    Args:
        model: EEG2TextJEPA instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Log every N batches
        
    Returns:
        Trained model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    # Only optimize EEG encoder and predictor (text encoder is frozen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_cos_sim = 0
    history = {'train_loss': [], 'val_loss': [], 'val_cos_sim': [], 'val_acc1': []}
    
    print(f"\n{'='*60}")
    print("Stage 2: EEG-Text Alignment (VL-JEPA)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg'].to(device)
            texts = batch['text']
            
            optimizer.zero_grad()
            pred_embed, target_embed, loss = model(eeg, texts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_metrics = evaluate_alignment(model, val_loader, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_cos_sim'].append(val_metrics['cosine_similarity'])
        history['val_acc1'].append(val_metrics['acc@1'])
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
        print(f"  Val Acc@1: {val_metrics['acc@1']:.4f}")
        print(f"  Val Acc@5: {val_metrics['acc@5']:.4f}")
        print(f"  Val Acc@10: {val_metrics['acc@10']:.4f}")
        print(f"  Val MRR: {val_metrics['mrr']:.4f}")
        
        # Save best model
        if val_metrics['cosine_similarity'] > best_cos_sim:
            best_cos_sim = val_metrics['cosine_similarity']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / 'stage2_best.pt')
            print(f"  ✓ Saved best model (CosSim: {best_cos_sim:.4f})")
    
    # Load best model
    checkpoint = torch.load(save_dir / 'stage2_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Save training history
    torch.save(history, save_dir / 'stage2_history.pt')
    
    return model
