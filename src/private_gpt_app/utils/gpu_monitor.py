"""GPU monitoring and detection utilities."""

import os
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class GPUInfo:
    """Information about the GPU."""
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    cuda_available: bool
    compute_capability: Optional[tuple] = None


@dataclass
class OptimalSettings:
    """Optimal model settings for detected GPU."""
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    cpu_offload_gb: float
    description: str


def detect_gpu() -> Optional[GPUInfo]:
    """Detect GPU and return information."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available")
            return None
        
        # Get GPU info
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        
        total_vram = props.total_memory / (1024 ** 3)  # Convert to GB
        
        # Get memory usage
        torch.cuda.empty_cache()
        used_vram = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        free_vram = total_vram - used_vram
        
        info = GPUInfo(
            name=props.name,
            vram_total_gb=total_vram,
            vram_free_gb=free_vram,
            vram_used_gb=used_vram,
            cuda_available=True,
            compute_capability=(props.major, props.minor)
        )
        
        return info
        
    except ImportError:
        print("⚠️  PyTorch not installed, cannot detect GPU")
        return None
    except Exception as e:
        print(f"⚠️  Error detecting GPU: {e}")
        return None


def get_current_vram_usage() -> Optional[Dict[str, float]]:
    """
    Get current VRAM usage without full GPU detection.
    Returns dict with 'used_gb', 'free_gb', 'total_gb', 'utilization_pct'.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        device_id = 0
        torch.cuda.empty_cache()
        
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
        used = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        free = total - used
        utilization = (used / total) * 100
        
        return {
            "used_gb": used,
            "free_gb": free,
            "total_gb": total,
            "utilization_pct": utilization
        }
    except Exception:
        return None


def recommend_settings(gpu_info: GPUInfo) -> OptimalSettings:
    """
    Recommend optimal settings based on GPU VRAM.
    
    Args:
        gpu_info: Detected GPU information
        
    Returns:
        OptimalSettings with recommended configuration
    """
    vram = gpu_info.vram_total_gb
    
    if vram >= 10:
        # High-end GPUs (RTX 3080, 4070, 4080, etc.)
        return OptimalSettings(
            gpu_memory_utilization=0.75,
            max_model_len=6144,
            max_num_seqs=8,
            cpu_offload_gb=0.0,
            description="Optimal - 6K context, maximum performance"
        )
    elif vram >= 8:
        # Mid-high GPUs (RTX 3070, 4060 Ti, 5060, etc.)
        return OptimalSettings(
            gpu_memory_utilization=0.70,
            max_model_len=4096,
            max_num_seqs=6,
            cpu_offload_gb=0.0,
            description="High - 4K context, excellent performance"
        )
    elif vram >= 6:
        # Mid-range GPUs (RTX 3060, 2060, 4060, etc.)
        return OptimalSettings(
            gpu_memory_utilization=0.65,
            max_model_len=3072,
            max_num_seqs=4,
            cpu_offload_gb=1.0,
            description="Balanced - 3K context, good performance"
        )
    elif vram >= 4:
        # Low-end GPUs (GTX 1650, RTX 3050 4GB, etc.)
        return OptimalSettings(
            gpu_memory_utilization=0.55,
            max_model_len=2048,
            max_num_seqs=2,
            cpu_offload_gb=2.0,
            description="Conservative - 2K context, stable"
        )
    else:
        # Very low VRAM
        return OptimalSettings(
            gpu_memory_utilization=0.50,
            max_model_len=1536,
            max_num_seqs=1,
            cpu_offload_gb=3.0,
            description="Minimal - 1.5K context, maximum stability"
        )


def validate_hardware_requirements(gpu_info: Optional[GPUInfo], min_vram_gb: float = 6.0) -> tuple[bool, str]:
    """
    Validate that hardware meets minimum requirements.
    
    Returns:
        (is_valid, message)
    """
    if gpu_info is None:
        return False, (
            "❌ No CUDA-capable GPU detected.\n\n"
            "Private-GPT requires an NVIDIA GPU with CUDA support.\n"
            "You can try running on CPU (much slower) by setting use_cpu=True."
        )
    
    if gpu_info.vram_total_gb < min_vram_gb:
        return False, (
            f"❌ Insufficient VRAM detected.\n\n"
            f"Your GPU has {gpu_info.vram_total_gb:.1f}GB VRAM.\n"
            f"Minimum required: {min_vram_gb}GB\n\n"
            f"Consider using a cloud GPU service or a model with lower requirements."
        )
    
    if gpu_info.vram_free_gb < (min_vram_gb * 0.8):  # Need 80% of min free
        return False, (
            f"⚠️  Low free VRAM detected.\n\n"
            f"Free VRAM: {gpu_info.vram_free_gb:.1f}GB / {gpu_info.vram_total_gb:.1f}GB\n"
            f"Recommended free: {min_vram_gb * 0.8:.1f}GB\n\n"
            f"Close other GPU-intensive applications and try again."
        )
    
    # All checks passed
    return True, (
        f"✅ Hardware check passed!\n\n"
        f"GPU: {gpu_info.name}\n"
        f"VRAM: {gpu_info.vram_free_gb:.1f}GB free / {gpu_info.vram_total_gb:.1f}GB total\n"
        f"CUDA Compute: {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}"
    )


def print_gpu_info(gpu_info: Optional[GPUInfo]) -> None:
    """Print GPU information to console."""
    if gpu_info is None:
        print("❌ No GPU detected")
        return
    
    print("=" * 60)
    print("🎮 GPU Information")
    print("=" * 60)
    print(f"Device: {gpu_info.name}")
    print(f"VRAM: {gpu_info.vram_free_gb:.1f}GB free / {gpu_info.vram_total_gb:.1f}GB total")
    print(f"CUDA Available: {gpu_info.cuda_available}")
    if gpu_info.compute_capability:
        print(f"Compute Capability: {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}")
    print()


def set_vram_limit(limit_gb: Optional[float] = None) -> None:
    """Set VRAM limit for PyTorch to prevent OOM."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return
        
        if limit_gb:
            # Set memory fraction
            fraction = limit_gb / (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            print(f"✓ Set VRAM limit to {limit_gb}GB")
    except Exception as e:
        print(f"⚠️  Could not set VRAM limit: {e}")
