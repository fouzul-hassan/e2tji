"""
Stage 3: Text decoder training with BLEU/ROUGE evaluation.

Training and evaluation functions.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress some warnings
warnings.filterwarnings('ignore', category=UserWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.decoder import EEG2TextDecoder


def compute_generation_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute text generation metrics.
    
    Metrics:
        - BLEU-1, BLEU-2, BLEU-3, BLEU-4
        - ROUGE-1, ROUGE-2, ROUGE-L
        - BERTScore (optional)
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    try:
        from rouge_score import rouge_scorer
        has_rouge = True
    except ImportError:
        has_rouge = False
        print("Warning: rouge_score not installed. Skipping ROUGE metrics.")
    
    smooth = SmoothingFunction().method1
    
    # Initialize scores
    bleu_scores = {f'bleu-{i}': [] for i in range(1, 5)}
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    if has_rouge:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
        
        # BLEU scores
        for i in range(1, 5):
            weights = tuple([1.0/i] * i + [0.0] * (4-i))
            try:
                score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
            except:
                score = 0.0
            bleu_scores[f'bleu-{i}'].append(score)
        
        # ROUGE scores
        if has_rouge:
            try:
                rouge = scorer.score(ref.lower(), pred.lower())
                rouge_scores['rouge1'].append(rouge['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(rouge['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(rouge['rougeL'].fmeasure)
            except:
                rouge_scores['rouge1'].append(0.0)
                rouge_scores['rouge2'].append(0.0)
                rouge_scores['rougeL'].append(0.0)
    
    # Average scores
    metrics = {}
    for k, v in bleu_scores.items():
        metrics[k] = sum(v) / len(v) if v else 0.0
    
    if has_rouge:
        for k, v in rouge_scores.items():
            metrics[k] = sum(v) / len(v) if v else 0.0
    
    # Optional: BERTScore
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(predictions, references, lang='en', verbose=False)
        metrics['bertscore_f1'] = F1.mean().item()
    except ImportError:
        pass
    except Exception as e:
        print(f"BERTScore error: {e}")
    
    return metrics


def evaluate_decoder(
    model: EEG2TextDecoder,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: int = None
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Evaluate Stage 3 decoder model.
    
    Args:
        model: EEG2TextDecoder instance
        dataloader: Data loader
        device: Device
        max_samples: Maximum samples to evaluate (for speed)
        
    Returns:
        (metrics, predictions, references)
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    n_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Stage 3"):
            eeg = batch['eeg'].to(device)
            texts = batch['text']
            
            # Generate
            predictions = model.generate(eeg, max_length=50, num_beams=5)
            
            all_predictions.extend(predictions)
            all_references.extend(texts)
            
            n_samples += len(texts)
            if max_samples and n_samples >= max_samples:
                break
    
    # Compute metrics
    metrics = compute_generation_metrics(all_predictions, all_references)
    
    return metrics, all_predictions, all_references


def train_stage3(
    model: EEG2TextDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-5,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    log_interval: int = 10,
    eval_samples: int = 500,
    use_fp16: bool = False,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0
) -> EEG2TextDecoder:
    """
    Train Stage 3 decoder model.
    
    Args:
        model: EEG2TextDecoder instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Log every N batches
        eval_samples: Max samples for evaluation (for speed)
        use_fp16: Use mixed precision training
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Trained model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    # Only train embed_proj and bart (JEPA is frozen)
    trainable_params = (
        list(model.embed_proj.parameters()) + 
        list(model.bart.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_fp16 and device == 'cuda' else None
    
    best_bleu4 = 0
    history = {'train_loss': [], 'val_bleu4': [], 'val_rougeL': []}
    
    effective_batch = train_loader.batch_size * grad_accum_steps
    
    print(f"\n{'='*60}")
    print("Stage 3: Text Decoder Fine-tuning")
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
            texts = batch['text']
            
            # Forward pass with optional mixed precision
            if use_fp16 and scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(eeg, texts=texts)
                    loss = outputs.loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(eeg, texts=texts)
                loss = outputs.loss / grad_accum_steps
                loss.backward()
            
            train_loss += loss.item() * grad_accum_steps
            
            # Gradient accumulation step
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
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_metrics, preds, refs = evaluate_decoder(
            model, val_loader, device, max_samples=eval_samples
        )
        history['val_bleu4'].append(val_metrics.get('bleu-4', 0))
        history['val_rougeL'].append(val_metrics.get('rougeL', 0))
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val BLEU-1: {val_metrics.get('bleu-1', 0):.4f}")
        print(f"  Val BLEU-4: {val_metrics.get('bleu-4', 0):.4f}")
        print(f"  Val ROUGE-L: {val_metrics.get('rougeL', 0):.4f}")
        if 'bertscore_f1' in val_metrics:
            print(f"  Val BERTScore: {val_metrics['bertscore_f1']:.4f}")
        
        # Sample predictions
        print(f"\n  Sample predictions:")
        for i in range(min(3, len(preds))):
            print(f"    Pred: {preds[i][:60]}...")
            print(f"    Ref:  {refs[i][:60]}...")
            print()
        
        # Save best model
        if val_metrics.get('bleu-4', 0) > best_bleu4:
            best_bleu4 = val_metrics.get('bleu-4', 0)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / 'stage3_best.pt')
            print(f"  ✓ Saved best model (BLEU-4: {best_bleu4:.4f})")
    
    # Load best model
    checkpoint = torch.load(save_dir / 'stage3_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Save training history
    torch.save(history, save_dir / 'stage3_history.pt')
    
    return model
