# Private-GPT Desktop App

A local, privacy-focused desktop chat application powered by Qwen2.5-3B-Instruct-AWQ with vLLM acceleration.

## Features

- 🔒 **100% Local & Private** - All data stays on your machine
- 💬 **Chat Interface** - Modern PyQt6-based UI with real-time token streaming
- 🚀 **Powered by vLLM** - AWQ Marlin quantization for maximum efficiency
- 🎯 **Low VRAM Optimized** - Runs on 4GB+ VRAM GPUs
- 📜 **2K Context Window** - Balanced performance for low-end hardware
- ⚡ **Fast Generation** - 66-378 tokens/sec on RTX 5060 Laptop
- 🤖 **Apache 2.0 Licensed Model** - Commercial use ready

## Prerequisites

- **Hardware:**
  - NVIDIA GPU with 4GB+ VRAM (GTX 1650, RTX 3050, RTX 4060, RTX 5060, etc.)
  - CUDA-capable drivers (vLLM bundles CUDA runtime)
  - 8GB+ system RAM
  - 5GB disk space for model

- **Software:**
  - Python 3.10+
  - Linux (tested on Ubuntu 22.04+)
  - NVIDIA drivers 525+ (for CUDA 12 support)

## Quick Start

### 1. Install Dependencies

```bash
cd /home/harshit/coding/private-gpt/private-gpt-app

# Install vLLM and dependencies
uv pip install vllm

# Or sync all dependencies
uv sync
```

### 2. Run the Application

```bash
# Run the application
uv run python run.py --dev

# Or without dev mode
uv run python run.py
```

### 3. First Run

On first launch, the app will:
1. Check your GPU (4GB VRAM minimum)
2. Auto-download Qwen2.5-3B-Instruct-AWQ from HuggingFace (~2.7GB)
3. Load model with vLLM using AWQ Marlin kernels
4. Ready to chat!

**Model Info:**
- Model: Qwen/Qwen2.5-3B-Instruct-AWQ (Apache 2.0 license)
- Size: 2.7GB download, 1.93GB loaded
- Context: 2048 tokens (~1500 words)
- VRAM Usage: ~3.1GB total (model + KV cache)
- Automatically cached in `models/Qwen2.5-3B-Instruct-AWQ/`

## Usage

### Chat Interface

The app provides a clean chat interface with:
- Real-time token streaming
- Message bubbles with user/assistant distinction
- Auto-scrolling to latest messages
- Responsive PyQt6 design

### Performance Tuning

Edit `src/private_gpt_app/ui/main_window.py` to adjust VRAM/context trade-offs:

```python
VLLMService(
    gpu_memory_utilization=0.55,  # 4GB VRAM: 0.55 | 6GB: 0.65 | 8GB+: 0.70
    max_model_len=2048,            # 4GB VRAM: 2048 | 6GB: 3072 | 8GB+: 4096
    cpu_offload_gb=2.0,            # Offload 2GB to system RAM for headroom
)
```

**GPU Compatibility Matrix:**
- **4GB VRAM** (GTX 1650, RTX 3050 4GB): 0.55 utilization, 2K context
- **6GB VRAM** (RTX 3060 Mobile, RTX 2060): 0.65 utilization, 3K context
- **8GB+ VRAM** (RTX 3070, RTX 4060, RTX 5060): 0.70 utilization, 4K context

## Project Structure

```
private-gpt-app/
├── src/private_gpt_app/
│   ├── ui/              # PyQt6 interface (main_window, chat_widget, message_bubble)
│   ├── backend/         # vLLM service with AWQ quantization
│   └── utils/           # GPU monitoring
├── data/
│   ├── faiss_index/     # (Future) Local vector database
│   └── crash_recovery/  # (Future) Auto-save temp files
├── models/
│   └── Qwen2.5-3B-Instruct-AWQ/  # Downloaded model files
└── docs/                # Documentation
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=private_gpt_app
```

### Hot Reload (QSS Styles)

When running in development mode, QSS stylesheets auto-reload on file changes.

### Memory Profiling

```bash
# Profile VRAM usage
uv run python -m memory_profiler src/private_gpt_app/main.py
```

## Phase 1 Status (Current)

✅ **Project Setup** - Complete
- [x] uv project initialization
- [x] pyproject.toml configuration
- [x] Directory structure
- [x] Python package structure
- [x] .gitignore for Python/PyQt
- [x] README with quick-start

✅ **Basic UI with Async** - Complete
- [x] PyQt6 main window with chat interface
- [x] qasync event loop integration
- [x] Token streaming UI with progressive rendering
- [x] Premium Black theme (QSS stylesheet)
- [x] Message bubbles with Markdown support

✅ **vLLM Integration** - Complete
- [x] VLLMService with lazy loading
- [x] GPU detection and VRAM validation
- [x] Qwen2.5-3B-Instruct-AWQ model (Apache 2.0)
- [x] AWQ Marlin quantization for efficiency
- [x] Token streaming from vLLM
- [x] ChatML conversation format
- [x] Error handling and graceful degradation
- [x] 4GB VRAM optimization (0.55 utilization, 2K context)

✅ **Codebase Cleanup** - Complete
- [x] Removed unused backend services (llama.cpp, transformers)
- [x] Removed unused model files (7B model, GGUF files)
- [x] Fixed duplicate code and import issues
- [x] Optimized for low-end GPU compatibility

⏳ **Next Steps (Phase 1 Remaining):**
- [ ] Add settings panel (memory/context tuning UI)
- [ ] Implement auto GPU detection with optimal settings
- [ ] Add VRAM usage monitoring in UI
- [ ] Implement crash recovery auto-save

## Roadmap

- **Phase 1** (Current): Foundation & Basic Chat
- **Phase 2**: RAG Pipeline with FAISS
- **Phase 3**: Routing & Session Management
- **Phase 4**: Optimization & Packaging

## Configuration

### VRAM Optimization

The app is currently configured for maximum compatibility (4GB+ VRAM).

To adjust for your specific GPU, edit `src/private_gpt_app/ui/main_window.py`:

```python
# 4GB VRAM (current default)
gpu_memory_utilization=0.55
max_model_len=2048

# 6GB VRAM (recommended for better context)
gpu_memory_utilization=0.65
max_model_len=3072

# 8GB+ VRAM (optimal performance)
gpu_memory_utilization=0.70
max_model_len=4096
```

**Trade-offs:**
- Lower utilization = more stable, less risk of OOM
- Higher context = better conversation memory, more VRAM usage

## Troubleshooting

### Low VRAM Error

If you see OOM (Out of Memory) errors:

1. Reduce `gpu_memory_utilization` from 0.55 to 0.50
2. Reduce `max_model_len` from 2048 to 1536
3. Increase `cpu_offload_gb` from 2.0 to 3.0
4. Close other GPU applications (Chrome, games, etc.)

Check current GPU usage:
```bash
nvidia-smi
```

### Model Download Failed

Model auto-downloads from HuggingFace on first run. If interrupted:
1. Delete `models/Qwen2.5-3B-Instruct-AWQ/`
2. Restart the app - download will resume

### Lingering Processes

If the app crashes and GPU is still occupied:
```bash
pkill -9 -f "python.*run.py"
```

### UI Freezing

Ensure qasync is properly initialized. Check terminal output for vLLM errors.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Qwen team for the efficient 3B Instruct model
- vLLM team for the high-performance inference engine
- AWQ team for the quantization method
- PyQt6 for the cross-platform UI framework
