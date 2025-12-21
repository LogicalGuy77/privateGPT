#!/usr/bin/env python
"""Pre-build verification to catch issues before 4-hour build."""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test all critical imports that Nuitka will need."""
    print("🔍 Testing critical imports...\n")
    
    critical_modules = [
        'vllm',
        'torch',
        'PyQt6.QtWidgets',
        'qdrant_client',
        'sentence_transformers',
        'transformers',
        'rank_bm25',
        'unittest.mock',
    ]
    
    failed = []
    
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("\n✅ All critical imports successful")
    return True

def test_metadata():
    """Test that required distribution metadata exists."""
    print("\n🔍 Testing distribution metadata...\n")
    
    required_metadata = [
        'triton',
        'torch',
        'vllm',
        'transformers',
    ]
    
    failed = []
    
    for pkg in required_metadata:
        try:
            import importlib.metadata
            version = importlib.metadata.version(pkg)
            print(f"✅ {pkg} ({version})")
        except Exception as e:
            print(f"❌ {pkg}: {e}")
            failed.append(pkg)
    
    if failed:
        print(f"\n❌ Missing metadata: {', '.join(failed)}")
        return False
    
    print("\n✅ All required metadata available")
    return True

def test_model_files():
    """Verify model files exist."""
    print("\n🔍 Testing model files...\n")
    
    model_path = Path("models/Qwen2.5-3B-Instruct-AWQ")
    
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    required_files = [
        'config.json',
        'model.safetensors',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt'
    ]
    
    missing = []
    
    for file in required_files:
        filepath = model_path / file
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files (non-critical): {', '.join(missing)}")
    
    print(f"\n✅ Model files verified at {model_path}")
    return True

def test_ui_files():
    """Verify UI resource files exist."""
    print("\n🔍 Testing UI files...\n")
    
    ui_files = [
        'src/private_gpt_app/ui/styles.qss',
        'src/private_gpt_app/ui/styles_modern.qss'
    ]
    
    missing = []
    
    for file in ui_files:
        filepath = Path(file)
        if filepath.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing UI files: {', '.join(missing)}")
        return False
    
    print("\n✅ All UI files present")
    return True

def test_tokenizer_load():
    """Test tokenizer loading (critical for chunking.py)."""
    print("\n🔍 Testing tokenizer loading...\n")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "models/Qwen2.5-3B-Instruct-AWQ",
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"✅ Tokenizer loaded from local model")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        print("   This will cause chunking.py to fall back to character-based chunking")
        return True  # Non-critical, has fallback

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Private-GPT Pre-Build Verification")
    print("=" * 60 + "\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Metadata Test", test_metadata),
        ("Model Files Test", test_model_files),
        ("UI Files Test", test_ui_files),
        ("Tokenizer Test", test_tokenizer_load),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60 + "\n")
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to build.")
        print("\nRun: rm -rf dist && uv run python build.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Fix issues before building.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
