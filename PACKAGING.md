# Private-GPT Packaging Guide

This guide covers building and packaging Private-GPT into a standalone executable with **bundled model** included.

## Prerequisites

### For Building Executable (Nuitka)
```bash
pip install nuitka
pip install ordered-set
sudo apt install patchelf  # Linux only
```

### For Creating Installer (Inno Setup - Windows only)
Download and install Inno Setup: https://jrsoftware.org/isdl.php

### Model Requirement
The Qwen2.5-3B-Instruct-AWQ model must be present at:
```
private-gpt-app/models/Qwen2.5-3B-Instruct-AWQ/
```

## Build Process

### Step 1: Verify Model is Present

```bash
cd private-gpt-app
ls models/Qwen2.5-3B-Instruct-AWQ/
# Should show: config.json, model.safetensors, tokenizer.json, etc.
```

### Step 2: Build Executable with Nuitka (includes model)

```bash
uv run python build_exe_optimized.py
```

This will:
- Verify model exists
- Compile Python code to C
- Bundle all dependencies + the 2GB model
- Create standalone package at `dist/PrivateGPT.dist/`
- Take 10-20 minutes depending on your system

**Expected Output Size:** ~3-4GB (includes 2GB model + 1-2GB dependencies)

### Step 3: Test the Build

```bash
cd dist/PrivateGPT.dist
./PrivateGPT --mock  # Test UI without loading model
./PrivateGPT         # Full test (will use bundled model)
```

### Step 4: Create Windows Installer (Optional)

```bash
# On Windows with Inno Setup installed:
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
```

Output: `installer/installer_output/PrivateGPT-Setup-0.2.0.exe`

**Installer size:** ~3-4GB (everything included)

## Distribution

### Complete Package (Recommended)
- **What:** Single installer with everything (app + model)
- **Size:** ~3-4GB
- **User Experience:** Download → Install → Launch (no setup needed)
- **Distribution:** Upload to GitHub Releases or file hosting

### Folder Distribution (Alternative)
- **What:** Just zip the `dist/PrivateGPT.dist` folder
- **Size:** ~3-4GB compressed to ~2.5GB
- **User Experience:** Download → Extract → Run executable
- **Distribution:** Easier for testing, no installer needed

## First-Run Experience

With bundled model:

1. User installs/extracts Private-GPT
2. Launches executable
3. App detects bundled model automatically
4. Shows "Model Ready" message
5. Immediately ready to use - **no downloads, no setup!**

## Optimization Tips

### Keep Model Size Reasonable
- ✅ Using AWQ quantized model (2GB vs 6GB FP16)
- Consider GPTQ or GGUF for even smaller sizes

### Reduce Dependencies
- Already excluding sklearn, scipy, matplotlib, pandas
- Only including tokenizer from transformers
- Using `--nofollow-import-to` aggressively

### Distribution Options

**Option A: Full Package (Current)**
- Pros: Zero setup, works offline immediately
- Cons: Large download (3-4GB)
- Best for: Users with good internet, one-time install

**Option B: App + Separate Model Download**  
- Pros: Smaller initial download (~800MB app)
- Cons: Requires internet on first run
- Implementation: Remove `--include-data-dir=models` from build script

**Option C: Portable Package**
- Just zip `dist/PrivateGPT.dist/` folder
- No installer, direct extraction
- Good for USB stick distribution

## Distribution Checklist

- [ ] Test on clean Windows VM
- [ ] Verify CUDA/GPU detection works
- [ ] Test model download on slow connection
- [ ] Check installer creates shortcuts
- [ ] Verify uninstaller removes all files
- [ ] Test offline mode (mock/skip setup)
- [ ] Check startup time (<10s)
- [ ] Verify crash recovery works

## Troubleshooting

### "CUDA Not Found" Error
- Bundle CUDA DLLs with installer (uncomment in setup.iss)
- Or require NVIDIA drivers in system requirements

### "DLL Load Failed" Error
- Install Visual C++ Redistributable (handled by installer)
- Check if all dependencies are bundled

### Large Executable Size
- Normal for ML apps with PyTorch
- Consider splitting into "Core" and "GPU" installers
- Use LZMA2 compression in Inno Setup

### Slow Startup
- Enable vLLM prefix caching
- Pre-compile Python bytecode
- Use `--onefile` carefully (slower unpacking)

## Alternative: PyInstaller (Simpler but Larger)

```bash
pip install pyinstaller

pyinstaller --onefile \
  --windowed \
  --name PrivateGPT \
  --icon assets/icon.ico \
  --hidden-import vllm \
  --hidden-import torch \
  --collect-all PyQt6 \
  src/private_gpt_app/main.py
```

**Pros:** Faster builds, easier to configure
**Cons:** Larger binaries, slower startup

## Model Packaging Options

### Option A: Separate Model Installer (Recommended)
- Small app installer (~500MB)
- Model downloaded on first run
- User can update model independently

### Option B: Bundle Model with Installer
- One-click install
- 2.5GB+ installer
- Slower distribution

### Option C: Portable Zip
- No installer needed
- Just extract and run
- Requires manual model setup

## Publishing

### GitHub Releases
```bash
gh release create v0.2.0 \
  installer/installer_output/PrivateGPT-Setup-0.2.0.exe \
  --title "Private-GPT v0.2.0" \
  --notes "First release with Qwen2.5-3B and RAG"
```

### Code Signing (Optional)
- Prevents Windows SmartScreen warnings
- Requires code signing certificate ($100+/year)
- Use SignTool.exe from Windows SDK

## System Requirements

**Minimum:**
- Windows 10/11 or Linux (64-bit)
- 8GB RAM (10GB+ recommended)
- 4GB VRAM (NVIDIA GPU with CUDA 11.8+)
- 8GB disk space (4GB package + 4GB runtime)
- **No internet required** (model bundled)

**Recommended:**
- 16GB RAM
- 8GB+ VRAM (RTX 3060 or better)
- SSD for faster model loading
- Good CPU for non-GPU fallback

## Build Summary

**What gets bundled:**
- ✅ Python runtime
- ✅ PyQt6 UI framework
- ✅ vLLM inference engine
- ✅ PyTorch + CUDA libraries
- ✅ Qdrant vector database
- ✅ Sentence transformers (CPU)
- ✅ **Qwen2.5-3B-Instruct-AWQ model (2GB)**
- ✅ All application code

**What user needs:**
- ❌ No Python installation
- ❌ No pip or package managers
- ❌ No internet for model download
- ✅ Just NVIDIA drivers (for GPU acceleration)

## Quick Start for End Users

1. Download `PrivateGPT-Setup-0.2.0.exe` (3-4GB)
2. Run installer
3. Launch Private-GPT from Start Menu
4. Start chatting immediately - everything included!
