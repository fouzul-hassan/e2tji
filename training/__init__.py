"""Training utilities for EEG2TEXT JEPA."""

from .stage1_pretrain import train_stage1, evaluate_pretraining
from .stage2_alignment import train_stage2, evaluate_alignment
from .stage3_decoder import train_stage3, evaluate_decoder

__all__ = [
    'train_stage1', 'evaluate_pretraining',
    'train_stage2', 'evaluate_alignment',
    'train_stage3', 'evaluate_decoder',
]
