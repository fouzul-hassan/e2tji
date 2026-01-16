"""
GPU Profile Configurations for EEG2Text JEPA Training.

Provides optimized settings for different GPU types:
- Tesla T4 (15GB VRAM)
- NVIDIA A4000 Ada (16GB VRAM)

Usage:
    python scripts/train_stage1.py --gpu_profile t4 ...
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUProfile:
    """GPU optimization profile."""
    name: str
    vram_gb: int
    
    # Batch settings
    batch_size: int
    grad_accum_steps: int
    
    # Model settings
    embed_dim: int
    num_transformer_layers: int
    global_transformer_layers: int
    
    # Memory optimizations
    use_fp16: bool
    use_gradient_checkpointing: bool
    use_compile: bool  # torch.compile for PyTorch 2.0+
    
    # DataLoader settings
    num_workers: int
    pin_memory: bool
    
    # Additional settings
    max_grad_norm: float
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


# ============================================================================
# GPU PROFILES
# ============================================================================

# Tesla T4 - 15GB VRAM (Google Colab, most cloud providers)
T4_PROFILE = GPUProfile(
    name="Tesla T4",
    vram_gb=15,
    
    # Batch settings - optimized for 15GB
    batch_size=8,
    grad_accum_steps=4,  # Effective batch = 32
    
    # Model settings - balanced for memory
    embed_dim=256,
    num_transformer_layers=4,
    global_transformer_layers=4,
    
    # Memory optimizations
    use_fp16=True,
    use_gradient_checkpointing=True,
    use_compile=False,  # May cause issues on T4
    
    # DataLoader
    num_workers=4,
    pin_memory=True,
    
    max_grad_norm=1.0,
)

# NVIDIA A4000 Ada - 16GB VRAM (Higher compute capability)
A4000_PROFILE = GPUProfile(
    name="NVIDIA A4000 Ada",
    vram_gb=16,
    
    # Batch settings - slightly larger batches
    batch_size=12,
    grad_accum_steps=4,  # Effective batch = 48
    
    # Model settings - can handle larger model
    embed_dim=384,
    num_transformer_layers=6,
    global_transformer_layers=6,
    
    # Memory optimizations
    use_fp16=True,
    use_gradient_checkpointing=True,
    use_compile=True,  # Ampere+ benefits from torch.compile
    
    # DataLoader
    num_workers=6,
    pin_memory=True,
    
    max_grad_norm=1.0,
)

# A100 - 40GB/80GB (for reference)
A100_40GB_PROFILE = GPUProfile(
    name="NVIDIA A100 40GB",
    vram_gb=40,
    
    batch_size=32,
    grad_accum_steps=2,  # Effective batch = 64
    
    embed_dim=512,
    num_transformer_layers=8,
    global_transformer_layers=8,
    
    use_fp16=True,  # Can use BF16 on A100
    use_gradient_checkpointing=False,  # Not needed with 40GB
    use_compile=True,
    
    num_workers=8,
    pin_memory=True,
    
    max_grad_norm=1.0,
)

# CPU fallback
CPU_PROFILE = GPUProfile(
    name="CPU",
    vram_gb=0,
    
    batch_size=4,
    grad_accum_steps=8,  # Effective batch = 32
    
    embed_dim=128,
    num_transformer_layers=2,
    global_transformer_layers=2,
    
    use_fp16=False,
    use_gradient_checkpointing=False,
    use_compile=False,
    
    num_workers=2,
    pin_memory=False,
    
    max_grad_norm=1.0,
)


# Profile registry
GPU_PROFILES = {
    "t4": T4_PROFILE,
    "a4000": A4000_PROFILE,
    "a100": A100_40GB_PROFILE,
    "cpu": CPU_PROFILE,
}


def get_gpu_profile(profile_name: str) -> GPUProfile:
    """Get GPU profile by name."""
    profile_name = profile_name.lower()
    if profile_name not in GPU_PROFILES:
        available = ", ".join(GPU_PROFILES.keys())
        raise ValueError(f"Unknown GPU profile: {profile_name}. Available: {available}")
    return GPU_PROFILES[profile_name]


def auto_detect_gpu_profile() -> GPUProfile:
    """Auto-detect GPU and return appropriate profile."""
    if not torch.cuda.is_available():
        print("No CUDA GPU detected, using CPU profile")
        return CPU_PROFILE
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_memory:.1f} GB")
    
    # Match by name
    if "t4" in gpu_name:
        return T4_PROFILE
    elif "a4000" in gpu_name or "ada" in gpu_name:
        return A4000_PROFILE
    elif "a100" in gpu_name:
        return A100_40GB_PROFILE
    elif "v100" in gpu_name or "p100" in gpu_name:
        return T4_PROFILE  # Similar specs to T4
    
    # Match by VRAM
    if total_memory >= 35:
        return A100_40GB_PROFILE
    elif total_memory >= 14:
        return T4_PROFILE  # Conservative
    else:
        print(f"Warning: Small GPU ({total_memory:.1f}GB), using CPU profile")
        return CPU_PROFILE


def apply_gpu_optimizations(device: str, profile: GPUProfile):
    """Apply GPU-specific optimizations."""
    if device != 'cuda':
        return
    
    # Set memory allocation strategy
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
    
    # Enable TF32 for Ampere+ GPUs (A4000, A100)
    if torch.cuda.get_device_capability()[0] >= 8:  # Compute capability 8.0+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 for Ampere+ GPU")
    
    # cuDNN optimization
    torch.backends.cudnn.benchmark = True
    
    # Enable memory efficient attention if available (PyTorch 2.0+)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("Using memory-efficient scaled dot product attention (PyTorch 2.0+)")
    
    print(f"Applied optimizations for {profile.name}")


def print_profile_info(profile: GPUProfile):
    """Print GPU profile configuration."""
    print(f"\n{'='*60}")
    print(f"GPU Profile: {profile.name}")
    print(f"{'='*60}")
    print(f"VRAM: {profile.vram_gb} GB")
    print(f"Batch size: {profile.batch_size}")
    print(f"Gradient accumulation: {profile.grad_accum_steps}")
    print(f"Effective batch size: {profile.effective_batch_size}")
    print(f"Embed dim: {profile.embed_dim}")
    print(f"Transformer layers: {profile.num_transformer_layers}")
    print(f"FP16: {profile.use_fp16}")
    print(f"Gradient checkpointing: {profile.use_gradient_checkpointing}")
    print(f"torch.compile: {profile.use_compile}")
    print(f"{'='*60}\n")
