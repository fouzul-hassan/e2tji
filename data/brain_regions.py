"""
Brain region to EEG channel mapping for ZuCo 105-channel setup.
Used by Multi-View Transformer to process different brain areas separately.
"""

# EEG Frequency bands from ZuCo dataset
FREQUENCY_BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
# t1, t2: Theta (4-8 Hz)
# a1, a2: Alpha (8-13 Hz)
# b1, b2: Beta (13-30 Hz)
# g1, g2: Gamma (30-100 Hz)

NUM_CHANNELS = 105
NUM_BANDS = 8

# Brain region to channel indices mapping
# Based on 10-20 system electrode placement for 105 channels
BRAIN_REGIONS = {
    'prefrontal_left': list(range(0, 11)),       # Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, FC1
    'prefrontal_right': list(range(11, 22)),     # Fp2, AF8, AF4, F2, F4, F6, F8, FT8, FC6, FC4, FC2
    'frontal_left': list(range(22, 33)),         # Left frontal electrodes
    'frontal_right': list(range(33, 44)),        # Right frontal electrodes
    'central_left': list(range(44, 55)),         # C3, C1, CP3, CP1, etc.
    'central_right': list(range(55, 66)),        # C4, C2, CP4, CP2, etc.
    'temporal_left': list(range(66, 77)),        # T7, TP7, T9, etc.
    'temporal_right': list(range(77, 88)),       # T8, TP8, T10, etc.
    'parietal_occipital_left': list(range(88, 97)),   # P7, P5, P3, P1, PO7, PO3, O1
    'parietal_occipital_right': list(range(97, 105)), # P8, P6, P4, P2, PO8, PO4, O2
}

# Subject splits for ZuCo v1.0 (12 subjects)
ZUCO_V1_SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 
                     'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 'ZRP']

# Default subject-based splits (8 train / 2 val / 2 test)
DEFAULT_TRAIN_SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZKB', 'ZKH']
DEFAULT_VAL_SUBJECTS = ['ZKW', 'ZMG']
DEFAULT_TEST_SUBJECTS = ['ZPH', 'ZRP']


def get_region_channels(region_name: str) -> list:
    """Get channel indices for a specific brain region."""
    if region_name not in BRAIN_REGIONS:
        raise ValueError(f"Unknown region: {region_name}. Available: {list(BRAIN_REGIONS.keys())}")
    return BRAIN_REGIONS[region_name]


def get_num_region_channels() -> dict:
    """Get number of channels per region."""
    return {name: len(channels) for name, channels in BRAIN_REGIONS.items()}
