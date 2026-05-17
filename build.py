#!/usr/bin/env python
"""
Build script using PyInstaller for Private-GPT.
Fast build (5-10 min) with low memory usage.
"""

import subprocess
import sys
import shutil
from pathlib import Path

def clean_build():
    """Remove old build artifacts."""
    dirs_to_clean = ['dist', 'build']
    files_to_clean = ['*.spec']
    
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                print(f"🗑️  Removing {path}")
                shutil.rmtree(path, ignore_errors=True)
    
    for pattern in files_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_file():
                print(f"🗑️  Removing {path}")
                path.unlink()

def verify_model_exists():
    """Check if model is available to bundle."""
    model_path = Path("models/Qwen2.5-1.5B-Instruct-AWQ")
    
    if not model_path.exists():
        print("❌ Model not found at models/Qwen2.5-1.5B-Instruct-AWQ")
        print("Please ensure the model is downloaded first.")
        return False
    
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    missing = [f for f in required_files if not (model_path / f).exists()]
    
    if missing:
        print(f"❌ Missing required model files: {', '.join(missing)}")
        return False
    
    print(f"✅ Model found at {model_path}")
    return True

def build():
    """Build with PyInstaller."""
    print("🔨 Building Private-GPT with PyInstaller...")
    print("⏱️  This will take 5-10 minutes...")
    print("💾 Expected size: ~3-4GB (includes bundled 1.5B model)\n")
    
    if not verify_model_exists():
        print("\n❌ Build aborted - model not found")
        return False
    
    # Check for PyInstaller
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} found\n")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=True)
    
    clean_build()
    
    # Determine separator based on OS
    separator = ';' if sys.platform == 'win32' else ':'
    
    # PyInstaller command
    args = [
        sys.executable, '-m', 'PyInstaller',
        '--name=PrivateGPT',
        '--onedir',  # Create directory (faster than onefile)
        '--noconfirm',
        # Don't use --windowed for apps with multiprocessing (vLLM)
        
        # Use custom hooks for NVIDIA libraries
        '--additional-hooks-dir=hooks',
        
        # Bundle system NVIDIA driver library (critical for GPU access)
        '--add-binary=/lib/x86_64-linux-gnu/libcuda.so.1:nvidia/cuda/lib',
        '--add-binary=/lib/x86_64-linux-gnu/libcuda.so:nvidia/cuda/lib',
        
        # Add data files
        f'--add-data=src/private_gpt_app/ui/styles.qss{separator}ui',
        f'--add-data=src/private_gpt_app/ui/styles_modern.qss{separator}ui',
        f'--add-data=models/Qwen2.5-1.5B-Instruct-AWQ{separator}models/Qwen2.5-1.5B-Instruct-AWQ',
        
        # Hidden imports (modules not auto-detected)
        '--hidden-import=vllm',
        '--hidden-import=torch',
        '--hidden-import=PyQt6',
        '--hidden-import=qdrant_client',
        '--hidden-import=sentence_transformers',
        '--hidden-import=transformers',
        '--hidden-import=rank_bm25',
        '--hidden-import=unittest.mock',
        '--hidden-import=private_gpt_app.backend.vllm_service',
        '--hidden-import=private_gpt_app.backend.session_manager',
        '--hidden-import=private_gpt_app.backend.router',
        '--hidden-import=private_gpt_app.rag.vector_store',
        '--hidden-import=private_gpt_app.rag.embeddings',
        
        # Collect all data for these packages
        '--collect-all=vllm',
        '--collect-all=torch',
        '--collect-all=transformers',
        '--collect-all=sentence_transformers',
        
        # Copy CUDA metadata and binaries (critical for GPU detection)
        '--copy-metadata=torch',
        '--copy-metadata=nvidia-cublas-cu12',
        '--copy-metadata=nvidia-cuda-cupti-cu12',
        '--copy-metadata=nvidia-cuda-nvrtc-cu12',
        '--copy-metadata=nvidia-cuda-runtime-cu12',
        '--copy-metadata=nvidia-cudnn-cu12',
        '--copy-metadata=nvidia-cufft-cu12',
        '--copy-metadata=nvidia-curand-cu12',
        '--copy-metadata=nvidia-cusolver-cu12',
        '--copy-metadata=nvidia-cusparse-cu12',
        '--copy-metadata=nvidia-nccl-cu12',
        '--copy-metadata=nvidia-nvjitlink-cu12',
        '--copy-metadata=nvidia-nvtx-cu12',
        '--copy-metadata=triton',
        
        # Entry point
        'src/private_gpt_app/main.py'
    ]
    
    print(f"Command: {' '.join(args)}\n")
    
    try:
        subprocess.run(args, check=True)
        
        print("\n✅ Build successful!")
        print(f"📦 Executable: dist/PrivateGPT/PrivateGPT")
        print("\n📝 Next steps:")
        print("1. Test: cd dist/PrivateGPT && ./PrivateGPT --mock")
        print("2. Full test: cd dist/PrivateGPT && ./PrivateGPT")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Build cancelled")
        return False

if __name__ == '__main__':
    success = build()
    sys.exit(0 if success else 1)
