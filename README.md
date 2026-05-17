# Private-GPT Desktop App

A local, privacy-focused desktop chat application powered by
`Qwen/Qwen2.5-1.5B-Instruct-AWQ`, vLLM, PyQt6, and a local Qdrant-based RAG
pipeline.

Blog: https://medium.com/@harshitweb3/building-a-fully-private-gpt-549c0935d307

## Features

- 100% local chat: conversations and documents stay on your machine.
- PyQt6 desktop UI with sessions, searchable history, settings, and a knowledge base.
- Local vLLM inference with AWQ Marlin quantization.
- RAG over PDF, DOCX, TXT, and Markdown files.
- Qdrant local vector store plus BM25 hybrid reranking.
- CPU embeddings to keep GPU VRAM available for the LLM.
- Crash recovery, VRAM monitoring, and performance stats.
- Mock mode for UI testing without loading the main LLM.

## Current Models

| Purpose | Model | Where It Runs |
| --- | --- | --- |
| Chat / generation | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | NVIDIA GPU through vLLM |
| Embeddings / RAG | `sentence-transformers/all-MiniLM-L6-v2` | CPU |

The embedding model may download on first use even in `--mock` mode, because the
RAG stack is still initialized. Mock mode skips the main Qwen/vLLM model.

## Requirements

- Linux desktop environment, tested on Ubuntu-style systems.
- Python 3.10+.
- NVIDIA GPU with CUDA support for real model mode.
- 4GB+ VRAM is the current runtime validation threshold for the 1.5B AWQ model.
- 6GB+ VRAM is recommended for a smoother 3072-token default context.
- Docker is not required. The app uses your NVIDIA GPU directly through PyTorch/vLLM.

The code is now tuned for the lighter 1.5B model by default.

## Quick Start

From this repo:

```bash
cd ~/coding/private-gpt/privateGPT
uv sync
```

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Start with mock mode:

```bash
uv run python run.py --mock --dev
```

Run the real local model:

```bash
uv run python run.py --dev
```

You can monitor GPU usage in another terminal:

```bash
watch -n 1 nvidia-smi
```

## Runtime Behavior

On startup the app:

1. Cleans stale vLLM GPU processes from previous crashes.
2. Creates a PyQt6 + qasync event loop.
3. Loads `styles_modern.qss` when available.
4. Creates the main window and session sidebar.
5. Checks GPU requirements in real model mode.
6. Loads Qwen through vLLM after the UI appears.
7. Initializes local RAG components lazily as needed.

The UI displays streamed chunks, but vLLM generation is currently performed as a
single synchronous call in an executor and then split into small chunks for the
chat display. So it behaves like streaming in the UI, but it is not true token
streaming from vLLM yet.

## Main Commands

```bash
# UI smoke test, no main LLM load
uv run python run.py --mock --dev

# Real model mode
uv run python run.py --dev

# Run tests
uv run pytest

# Build packaged app with PyInstaller
uv run python build.py
```

## Project Structure

```text
privateGPT/
├── run.py                         # Dev-friendly entrypoint
├── build.py                       # PyInstaller build script
├── src/private_gpt_app/
│   ├── main.py                    # Real application entrypoint
│   ├── ui/                        # PyQt6 windows, dialogs, widgets, styles
│   ├── backend/                   # vLLM, sessions, retrieval router, document DB
│   ├── rag/                       # Qdrant, embeddings, ingestion, hybrid search
│   └── utils/                     # GPU, paths, setup, performance, crash recovery
├── data/
│   ├── qdrant_db/                 # Local Qdrant vector database
│   ├── documents.db               # Document registry
│   ├── sessions.db                # Chat sessions, created at runtime
│   └── crash_recovery/            # Recovery files
├── models/
│   └── Qwen2.5-1.5B-Instruct-AWQ/   # Optional bundled/local model directory
└── docs/
```

## RAG Flow

1. Add files through `Tools > Knowledge Base`.
2. `IngestionWorker` extracts text from PDF, DOCX, TXT, or MD.
3. Text is split with the current character-based `TextSplitter`.
4. Chunks are embedded on CPU with `all-MiniLM-L6-v2`.
5. Chunks are stored in local Qdrant with source metadata.
6. The document registry is stored in SQLite.
7. On a query, `RetrievalService` searches Qdrant, optionally reranks with BM25,
   truncates context by token budget, and injects the context into the latest
   user message before vLLM generation.

Note: token-based utilities exist in `rag/chunking.py` and are used for context
counting/truncation. Ingestion still uses the character-based splitter.

## Configuration

Default runtime settings are in `MainWindow.current_settings`:

```python
{
    "gpu_memory_utilization": 0.55,
    "max_model_len": 3072,
    "cpu_offload_gb": 1.0,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 768,
    "rag_strategy": "always",
    "relevance_threshold": 0.5,
}
```

The settings dialog can adjust generation and RAG settings. Model memory settings
are applied when the model service is created, so restart the app after changing
them if you need a clean model reload.

## Troubleshooting

### `uv` Not Found

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### CUDA / GPU Check

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Low VRAM or OOM

Try lowering:

- `gpu_memory_utilization`
- `max_model_len`
- `max_tokens`

Also close other GPU-heavy applications.

### Model Download

If no bundled model exists under `models/Qwen2.5-1.5B-Instruct-AWQ/`, vLLM will use
the HuggingFace model ID and download/cache through the normal HuggingFace cache.

### Qdrant or RAG Issues

The local vector database lives in `data/qdrant_db`. The document list is tracked
separately in `data/documents.db`.

## Packaging

This repo currently packages with PyInstaller:

```bash
uv run python build.py
```

The build expects a local model at:

```text
models/Qwen2.5-1.5B-Instruct-AWQ/
```

with required files such as `config.json`, `model.safetensors`, and
`tokenizer.json`.

## Notes

- `src/private_gpt_app/main.py` is the real app entrypoint.
- The root `main.py` is not the desktop app entrypoint.
- `main_broken.py`, if present in older checkouts, is a stale backup and is not
  referenced by the current run/build path.
