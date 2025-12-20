# Private-GPT Desktop App

A local, privacy-focused desktop chat application with RAG (Retrieval-Augmented Generation) capabilities, powered by NVIDIA Nemotron-Nano-9B-v2.

## Features

- 🔒 **100% Local & Private** - All data stays on your machine
- 💬 **Chat Interface** - Modern PyQt6-based UI with token streaming
- 🚀 **Powered by vLLM** - Optimized inference with 50% less VRAM
- 🧠 **Reasoning Modes** - `/think` for step-by-step reasoning, `/no_think` for fast responses
- 📚 **RAG Pipeline** (Coming soon) - Upload documents and ask questions
- 💾 **Session Management** (Coming soon) - Persistent chat history
- ⚡ **Fast & Lightweight** - Native widgets, optimized CUDA kernels

## Prerequisites

- **Hardware:**
  - NVIDIA GPU with 6GB+ VRAM (RTX 3060, 4060, 5060 or better)
  - CUDA-capable drivers (vLLM bundles CUDA runtime)
  - 8GB+ system RAM
  - 10GB disk space

- **Software:**
  - Python 3.10+
  - Linux (tested on Ubuntu 22.04+) or Windows 10/11
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
# Mock mode (no model - for UI testing)
uv run python run.py --mock --dev

# Real mode (auto-downloads model from HuggingFace)
uv run python run.py
```

### 3. First Run

On first launch, the app will:
1. Check your GPU (6GB VRAM minimum)
2. Auto-download Nemotron Nano 9B v2 from HuggingFace (~4.5GB)
3. Load model with vLLM (takes ~30 seconds)
4. Ready to chat!

**Model Download:**
- Automatically cached in `~/.cache/huggingface/hub/`
- Only downloads once
- Takes 5-10 minutes depending on internet speed

## Usage

### Reasoning Modes

Nemotron Nano 9B v2 supports controllable reasoning:

- **`/no_think`** (default): Fast, direct answers
- **`/think`**: Shows step-by-step reasoning traces

Currently set in code (`main_window.py`), UI toggle coming in Phase 2.

### Performance Tuning

Edit `src/private_gpt_app/ui/main_window.py` to adjust:

```python
VLLMService(
    gpu_memory_utilization=0.90,  # Use 90% of VRAM (adjust to 0.85 if needed)
    max_model_len=8192,            # Context window (reduce to 4096 for lower VRAM)
)
```

## Project Structure

```
private-gpt-app/
├── src/private_gpt_app/
│   ├── ui/              # PyQt6 interface components
│   ├── backend/         # LLM service and session manager
│   ├── rag/             # Vector store and embeddings
│   └── utils/           # GPU monitoring, crash recovery
├── data/
│   ├── faiss_index/     # Local vector database
│   ├── crash_recovery/  # Auto-save temp files
│   └── sessions.db      # Chat history (SQLite)
├── models/              # Downloaded GGUF models
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
- [x] Hot-reload for QSS in dev mode
- [x] Message bubbles with Markdown support

✅ **llama.cpp Integration** - Complete
- [x] LlamaCppService with lazy loading
- [x] GPU detection and VRAM validation
- [x] Model downloader utility
- [x] Token streaming from llama.cpp
- [x] Conversation history management
- [x] Error handling and graceful degradation

⏳ **Next Steps (Phase 1 Remaining):**
- [ ] Add settings panel (context window, temperature, etc.)
- [ ] Implement reasoning mode toggle (/think vs /no_think)
- [ ] Add VRAM usage monitoring in UI
- [ ] Implement crash recovery auto-save

## Roadmap

- **Phase 1** (Current): Foundation & Basic Chat
- **Phase 2**: RAG Pipeline with FAISS
- **Phase 3**: Routing & Session Management
- **Phase 4**: Optimization & Packaging

## Configuration

### VRAM Optimization

Edit `~/.private-gpt/config.json`:

```json
{
  "context_window": 8192,  // Default: 8k, max: 32k
  "n_gpu_layers": -1,      // -1 = all layers on GPU
  "reasoning_mode": true   // Enable /think mode
}
```

## Troubleshooting

### Low VRAM Error

If you see "Low VRAM detected", the app will:
1. Reduce context window to 4k
2. Offload some layers to CPU
3. Suggest closing other GPU applications

### Model Download Failed

The app uses resumable downloads. If interrupted:
1. Restart the app
2. Download will resume from where it stopped

### UI Freezing

Ensure you're not running in mock mode and qasync is properly initialized.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- NVIDIA Nemotron team for the amazing model
- llama.cpp community for the inference engine
- PyQt6 for the cross-platform UI framework
