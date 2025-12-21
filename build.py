#!/usr/bin/env python
"""
Optimized Nuitka build script for Private-GPT with bundled model.
Includes the Qwen2.5-3B-Instruct-AWQ model in the distribution.
"""

import subprocess
import sys
import shutil
from pathlib import Path

def clean_build():
    """Remove old build artifacts."""
    dirs_to_clean = ['dist', 'build', '*.dist', '*.build', '*.onefile-build']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                print(f"🗑️  Removing {path}")
                shutil.rmtree(path, ignore_errors=True)

def verify_model_exists():
    """Check if model is available to bundle."""
    model_path = Path("models/Qwen2.5-3B-Instruct-AWQ")
    
    if not model_path.exists():
        print("❌ Model not found at models/Qwen2.5-3B-Instruct-AWQ")
        print("Please ensure the model is downloaded first.")
        return False
    
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    missing = [f for f in required_files if not (model_path / f).exists()]
    
    if missing:
        print(f"❌ Missing required model files: {', '.join(missing)}")
        return False
    
    print(f"✅ Model found at {model_path}")
    return True

def get_nuitka_args():
    """Build Nuitka command arguments with bundled model."""
    
    is_windows = sys.platform == "win32"
    
    # Core options
    args = [
        sys.executable, '-m', 'nuitka',
        '--standalone',
        '--enable-plugin=pyqt6',
        
        # Output
        '--output-dir=dist',
        f'--output-filename=PrivateGPT{".exe" if is_windows else ""}',
        
        # Performance
        '--assume-yes-for-downloads',
        '--show-progress',
        '--jobs=4',
        
        # Include only what we need
        '--include-package=vllm',
        '--include-package=torch',
        '--include-package=PyQt6',
        '--include-package=qdrant_client',
        '--include-package=sentence_transformers',
        '--include-package=transformers',  # Needed for tokenizer
        '--include-package-data=tokenizers',
        '--include-package=unittest.mock',  # Required by some dependencies
        '--include-package=rank_bm25',  # For hybrid search
        
        # Include metadata (prevents runtime errors)
        '--include-distribution-metadata=triton',  # Required by transformers
        '--include-distribution-metadata=torch',  # Required by vllm
        '--include-distribution-metadata=vllm',  # Required for version checks
        '--include-distribution-metadata=transformers',  # Required for model loading
        '--include-distribution-metadata=sentencepiece',  # Required by some tokenizers
        
        # Exclude heavy stuff we don't use
        '--nofollow-import-to=transformers.commands',
        '--nofollow-import-to=transformers.models.albert',
        '--nofollow-import-to=transformers.models.bart',
        '--nofollow-import-to=transformers.models.bert',
        '--nofollow-import-to=transformers.models.gpt2',
        '--nofollow-import-to=transformers.models.llama',  # We use qwen
        '--nofollow-import-to=sklearn',
        '--nofollow-import-to=scipy',
        '--nofollow-import-to=matplotlib',
        '--nofollow-import-to=pandas',
        '--nofollow-import-to=pytest',
        '--nofollow-import-to=setuptools',
        '--nofollow-import-to=pip',
        '--nofollow-import-to=tensorboard',
        '--nofollow-import-to=wandb',
        '--nofollow-import-to=notebook',
        '--nofollow-import-to=IPython',
        
        # Include UI data files
        '--include-data-file=src/private_gpt_app/ui/styles.qss=private_gpt_app/ui/styles.qss',
        '--include-data-file=src/private_gpt_app/ui/styles_modern.qss=private_gpt_app/ui/styles_modern.qss',
        
        # Bundle the entire model directory
        '--include-data-dir=models/Qwen2.5-3B-Instruct-AWQ=models/Qwen2.5-3B-Instruct-AWQ',
        
        # Entry point
        'src/private_gpt_app/main.py'
    ]
    
    return args

def build():
    """Execute Nuitka build with bundled model."""
    print("🔨 Building Private-GPT with Nuitka (with bundled model)...")
    print("⏱️  This will take 10-20 minutes...")
    print("💾 Expected size: ~3-4GB (includes 2GB model)\n")
    
    # Verify model exists
    if not verify_model_exists():
        print("\n❌ Build aborted - model not found")
        return False
    
    clean_build()
    
    args = get_nuitka_args()
    
    print(f"Command: {' '.join(args)}\n")
    
    try:
        # Run nuitka
        subprocess.run(args, check=True)
        
        # Rename main.dist to PrivateGPT.dist
        main_dist = Path("dist/main.dist")
        private_gpt_dist = Path("dist/PrivateGPT.dist")
        if main_dist.exists():
            if private_gpt_dist.exists():
                shutil.rmtree(private_gpt_dist)
            main_dist.rename(private_gpt_dist)
            print(f"📁 Renamed {main_dist} -> {private_gpt_dist}")
        
        # Rename main.build to PrivateGPT.build
        main_build = Path("dist/main.build")
        private_gpt_build = Path("dist/PrivateGPT.build")
        if main_build.exists():
            if private_gpt_build.exists():
                shutil.rmtree(private_gpt_build)
            main_build.rename(private_gpt_build)
        
        print("\n✅ Build successful!")
        print(f"📦 Executable: dist/PrivateGPT.dist/PrivateGPT")
        print("\n📝 Next steps:")
        print("1. Test: cd dist/PrivateGPT.dist && ./PrivateGPT --mock")
        print("2. For single-file build, add --onefile flag (slower)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Build cancelled by user")
        return False

if __name__ == '__main__':
    success = build()
    sys.exit(0 if success else 1)
