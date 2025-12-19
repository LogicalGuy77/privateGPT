# Private-GPT Desktop App

A local, privacy-focused desktop chat application with RAG (Retrieval-Augmented Generation) capabilities, powered by NVIDIA Nemotron-Nano-9B-v2.

## Features

- 🔒 **100% Local & Private** - All data stays on your machine
- 💬 **Chat Interface** - Modern PyQt6-based UI with token streaming
- 📚 **RAG Pipeline** - Upload documents and ask questions about them
- 🚀 **Optimized for 8GB VRAM** - Uses llama.cpp with INT4 quantization
- 💾 **Session Management** - Persistent chat history with full-text search
- ⚡ **Fast & Lightweight** - Native widgets, minimal RAM overhead

## Prerequisites

- **Hardware:**
  - NVIDIA GPU with 6GB+ VRAM (8GB recommended)
  - CUDA 12.x drivers
  - 16GB RAM

- **Software:**
  - Python 3.10+
  - Linux (tested on Ubuntu 22.04+) or Windows 10/11

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd private-gpt-app

# Install dependencies with uv
uv sync

# For development dependencies
uv sync --dev
```

### 2. Run the Application

```bash
# Normal mode
uv run private-gpt

# Development mode with mock responses (no model loading)
uv run python -m private_gpt_app.main --mock
```

### 3. First Run

On first launch, the app will:
1. Check your GPU and VRAM
2. Download the Nemotron-Nano-9B-v2 model (~4.5GB)
3. Download the embedding model (~2GB)

This may take 10-20 minutes depending on your internet connection.

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

⏳ **Next Steps:**
- [ ] Basic PyQt6 UI with chat area
- [ ] llama.cpp integration with lazy loading
- [ ] Token streaming implementation
- [ ] GPU detection and validation

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
