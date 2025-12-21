"""
PyInstaller hook for NVIDIA CUDA libraries.
Ensures all CUDA .so files are collected.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect all NVIDIA CUDA shared libraries
binaries = []

nvidia_packages = [
    'nvidia.cublas',
    'nvidia.cuda_cupti',
    'nvidia.cuda_nvrtc',
    'nvidia.cuda_runtime',
    'nvidia.cudnn',
    'nvidia.cufft',
    'nvidia.curand',
    'nvidia.cusolver',
    'nvidia.cusparse',
    'nvidia.nccl',
    'nvidia.nvjitlink',
    'nvidia.nvtx',
]

for package in nvidia_packages:
    try:
        binaries.extend(collect_dynamic_libs(package))
    except:
        pass
