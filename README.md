# EEG2TEXT JEPA

**EEG-to-Text Decoding with Joint Embedding Predictive Architecture**

This project implements a state-of-the-art EEG-to-Text decoding system combining:
- **EEG2TEXT** architecture (Multi-View Transformer for different brain regions)
- **VL-JEPA** training objective (predict text embeddings instead of tokens)
- **3-stage training pipeline** (pretrain → align → decode)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Download NLTK data (for BLEU)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Prepare Data

Convert your EEG2TEXT pickle files to PyTorch format:

```bash
python scripts/prepare_pytorch_data.py \
    --pickle_files "path/to/task1-SR-dataset.pickle" "path/to/task2-NR-dataset.pickle" \
    --output_dir "./data/processed"
```

This creates:
```
data/processed/
├── train_data.pt   # 8 subjects
├── val_data.pt     # 2 subjects
├── test_data.pt    # 2 subjects
└── metadata.pt     # Configuration
```

### 3. Train Model (Step by Step)

**Stage 1: Self-supervised pretraining (masked reconstruction)**
```bash
python scripts/train_stage1.py \
    --data_dir ./data/processed \
    --epochs 100 \
    --device cuda
```

**Stage 2: EEG-Text alignment (VL-JEPA)**
```bash
python scripts/train_stage2.py \
    --data_dir ./data/processed \
    --epochs 50 \
    --device cuda
```

**Stage 3: Text decoder fine-tuning (BART)**
```bash
python scripts/train_stage3.py \
    --data_dir ./data/processed \
    --epochs 20 \
    --device cuda
```

### Alternative: Run All Stages at Once

```bash
python scripts/train.py --stage all --device cuda
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    3-STAGE TRAINING PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STAGE 1: Self-Supervised Pretraining                           │
│  ─────────────────────────────────────                          │
│  Masked EEG → CNN Encoder → ConvTransformer → CNN Decoder → EEG │
│                                   └── Reconstruction Loss        │
│                                                                  │
│  STAGE 2: EEG-Text Alignment (VL-JEPA)                          │
│  ─────────────────────────────────────                          │
│  EEG → Multi-View Transformer → Predictor → Ŝ_Y                 │
│                                               ↓                  │
│  Text → Frozen SentenceBERT → S_Y ──────→ L2 Loss               │
│                                                                  │
│  STAGE 3: Text Decoder                                          │
│  ────────────────────────                                       │
│  EEG → Frozen JEPA → Embedding → BART → Generated Text          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-View Transformer

The model processes 10 brain regions separately:

| Region | Channels | Function |
|--------|----------|----------|
| Prefrontal L/R | 0-10, 11-21 | Executive function |
| Frontal L/R | 22-32, 33-43 | Motor planning |
| Central L/R | 44-54, 55-65 | Sensorimotor |
| Temporal L/R | 66-76, 77-87 | Language processing |
| Parietal-Occipital L/R | 88-96, 97-104 | Visual processing |

## Evaluation Metrics

| Stage | Metrics |
|-------|---------|
| **Stage 1** | MSE, Correlation, SNR (dB) |
| **Stage 2** | Cosine Similarity, Acc@1/5/10, MRR |
| **Stage 3** | BLEU-1/4, ROUGE-1/2/L, BERTScore |

## Project Structure

```
llm-jepa-for-eeg-to-text/
├── data/
│   ├── brain_regions.py      # Channel mappings
│   └── zuco_dataset.py       # PyTorch dataset
├── models/
│   ├── pretraining_model.py  # Stage 1: CNN + Transformer
│   ├── eeg2text_jepa.py      # Stage 2: Multi-View + JEPA
│   └── decoder.py            # Stage 3: BART decoder
├── training/
│   ├── stage1_pretrain.py    # Pretraining loop + eval
│   ├── stage2_alignment.py   # Alignment loop + eval
│   └── stage3_decoder.py     # Decoder loop + eval
├── scripts/
│   ├── prepare_pytorch_data.py
│   └── train.py
├── checkpoints/              # Saved models
└── requirements.txt
```

## Training Arguments

```
--data_dir          Data directory (default: ./data/processed)
--batch_size        Batch size (default: 32)
--embed_dim         Embedding dimension (default: 256)
--text_encoder      Text encoder model (default: all-MiniLM-L6-v2)
--bart_model        BART model for decoding (default: bart-base)
--epochs_stage1     Stage 1 epochs (default: 100)
--epochs_stage2     Stage 2 epochs (default: 50)
--epochs_stage3     Stage 3 epochs (default: 20)
--device            cuda or cpu
--save_dir          Checkpoint directory
```

## Citation

If you use this code, please cite:
- EEG2TEXT paper (IEEE BigData 2024)
- VL-JEPA paper (arXiv:2512.10942)
- ZuCo dataset
