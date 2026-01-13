"""Model components for EEG2TEXT JEPA."""

from .pretraining_model import PretrainingModel, CNNEncoder, CNNDecoder, ConvTransformer
from .eeg2text_jepa import EEG2TextJEPA, MultiViewTransformer, RegionEncoder
from .decoder import EEG2TextDecoder

__all__ = [
    # Stage 1
    'PretrainingModel',
    'CNNEncoder', 
    'CNNDecoder',
    'ConvTransformer',
    # Stage 2
    'EEG2TextJEPA',
    'MultiViewTransformer',
    'RegionEncoder',
    # Stage 3
    'EEG2TextDecoder',
]
