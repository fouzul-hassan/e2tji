"""
PyTorch Dataset for pre-processed ZuCo data.
Loads from .pt files created by scripts/prepare_pytorch_data.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List


class ZuCoDataset(Dataset):
    """
    PyTorch Dataset for pre-processed ZuCo EEG data.
    
    Expects .pt files with structure:
        {
            'eeg': Tensor (N, 105, 8),
            'texts': List[str],
            'subjects': List[str]
        }
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to {split}_data.pt file
        """
        data = torch.load(data_path, weights_only=False)
        
        self.eeg = data['eeg']           # (N, 105, 8)
        self.texts = data['texts']        # List[str]
        self.subjects = data['subjects']  # List[str]
        
        print(f"Loaded {len(self)} samples from {data_path}")
        print(f"  EEG shape: {self.eeg.shape}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'eeg': self.eeg[idx],
            'text': self.texts[idx],
            'subject': self.subjects[idx]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable-length text."""
    return {
        'eeg': torch.stack([b['eeg'] for b in batch]),
        'text': [b['text'] for b in batch],
        'subject': [b['subject'] for b in batch]
    }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from processed data.
    
    Args:
        data_dir: Directory containing train_data.pt, val_data.pt, test_data.pt
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    train_dataset = ZuCoDataset(data_dir / "train_data.pt")
    val_dataset = ZuCoDataset(data_dir / "val_data.pt")
    test_dataset = ZuCoDataset(data_dir / "test_data.pt")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
