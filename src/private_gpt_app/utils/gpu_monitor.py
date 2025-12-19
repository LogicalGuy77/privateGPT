"""GPU monitoring and detection utilities."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about the GPU."""
    name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_available: bool
    compute_capability: Optional[tuple] = None


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
        
        # Get free VRAM
        torch.cuda.empty_cache()
        free_vram = (torch.cuda.get_device_properties(device_id).total_memory - 
                     torch.cuda.memory_allocated(device_id)) / (1024 ** 3)
        
        info = GPUInfo(
            name=props.name,
            vram_total_gb=total_vram,
            vram_free_gb=free_vram,
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
