# Utils module for EEG2Text JEPA
from utils.gpu_profiles import (
    GPUProfile,
    GPU_PROFILES,
    get_gpu_profile,
    auto_detect_gpu_profile,
    apply_gpu_optimizations,
    print_profile_info,
    T4_PROFILE,
    A4000_PROFILE,
    A100_40GB_PROFILE,
    CPU_PROFILE,
)

__all__ = [
    'GPUProfile',
    'GPU_PROFILES',
    'get_gpu_profile',
    'auto_detect_gpu_profile',
    'apply_gpu_optimizations',
    'print_profile_info',
    'T4_PROFILE',
    'A4000_PROFILE',
    'A100_40GB_PROFILE',
    'CPU_PROFILE',
]
