"""Model components for EEG2TEXT JEPA."""

# Stage 1: MAE (legacy backup)
from .pretraining_mae import PretrainingModel as MAEPretrainingModel
from .pretraining_mae import CNNEncoder, CNNDecoder, ConvTransformer

# Stage 1: JEPA (true latent prediction)
from .pretraining_jepa import JEPAPretrainingModel, JEPAPredictor, RegionPatchEncoder

# Stage 2: VL-JEPA alignment
from .eeg2text_jepa import EEG2TextJEPA, MultiViewTransformer, RegionEncoder

# Stage 3: Decoder
from .decoder import EEG2TextDecoder

__all__ = [
    # Stage 1 - MAE (legacy)
    'MAEPretrainingModel',
    'CNNEncoder', 
    'CNNDecoder',
    'ConvTransformer',
    # Stage 1 - JEPA
    'JEPAPretrainingModel',
    'JEPAPredictor',
    'RegionPatchEncoder',
    # Stage 2
    'EEG2TextJEPA',
    'MultiViewTransformer',
    'RegionEncoder',
    # Stage 3
    'EEG2TextDecoder',
]

