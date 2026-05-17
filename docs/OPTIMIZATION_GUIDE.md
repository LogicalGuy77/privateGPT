# Optimization Guide

This guide summarizes the current optimization state and the highest-value next
changes for Private-GPT.

## Current Optimizations

### Inference

- Main model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- Runtime: vLLM
- Quantization: `awq_marlin`
- `dtype="float16"`
- `gpu_memory_utilization=0.55`
- `max_model_len=4096`
- `max_tokens=1024`
- `enable_prefix_caching=True`
- `enforce_eager=True` for stability
- optional `cpu_offload_gb` from settings

The UI currently receives chunked output, but vLLM generation itself happens as
a full synchronous call in an executor.

### RAG

- Embeddings run on CPU with `sentence-transformers/all-MiniLM-L6-v2`.
- Qdrant local mode stores vectors under `data/qdrant_db`.
- Hybrid reranking blends semantic results with BM25.
- Retrieved context is truncated with token counting before generation.
- RAG strategy is configurable: `always`, `smart`, or `explicit`.

### Persistence

- Chat sessions use SQLite.
- WAL mode is enabled.
- FTS5 powers session search.
- Recent sessions are cached.
- Conversations are trimmed to the last 10 messages before generation.

### UI and Reliability

- PyQt6 desktop UI.
- Mock mode for UI testing.
- QSS hot reload in `--dev`.
- VRAM monitoring.
- Crash recovery.
- Performance stats dialog.

## Important Caveats

### 1. 4GB vs 6GB VRAM

Some settings are conservative enough to experiment with 4GB GPUs, but current
runtime validation warns below 6GB VRAM. The active default context is 4096
tokens, not 2048.

### 2. Simulated Streaming

`VLLMService.generate_stream()` calls `LLM.generate()` once, then yields the full
text in small chunks. This is UI streaming, not true token streaming from vLLM.

### 3. Token-Based Chunking Is Not Used for Ingestion

`rag/chunking.py` provides token utilities, and retrieval uses them for context
counting/truncation. Ingestion still uses the character-based `TextSplitter` in
`rag/ingestion.py`.

### 4. Duplicate Detection Runs Too Late

`DocumentStore` can detect duplicate files by hash, but ingestion currently
upserts chunks into Qdrant before inserting the SQLite document record. Move the
duplicate check before text extraction/embedding to avoid duplicate vectors.

## Recommended Next Fixes

### 1. Check Duplicates Before Ingestion

Current rough flow:

```text
extract -> chunk -> embed/upsert -> add document row
```

Recommended flow:

```text
hash/check duplicate -> extract -> chunk -> embed/upsert -> add document row
```

Impact:

- avoids duplicate Qdrant chunks
- saves embedding time
- keeps document registry and vector store aligned

### 2. Use Token-Based Chunking During Ingestion

Replace `TextSplitter` usage in `IngestionWorker` with `token_chunker`:

```python
chunks = token_chunker.chunk_by_tokens(text, chunk_size=256, overlap=50)
```

Impact:

- more predictable context usage
- better alignment with LLM token budget
- easier chunk-size tuning by model context length

### 3. Persist Session Rename

`SessionSidebar._rename_session()` currently makes the item editable but does
not write the new title back to SQLite. Add an `itemChanged` handler and a
session-manager method or reuse `update_session()`.

### 4. Add Focused Tests

Priority tests:

- `trim_conversation()`
- `SessionManager` CRUD and FTS search
- duplicate detection before ingestion
- `RetrievalService` strategies
- context truncation budget
- vector-store filename filtering

### 5. True vLLM Streaming

If you want real token streaming rather than UI chunking, investigate the
currently supported vLLM async/streaming API for the installed vLLM version and
adapt the service around that API.

## Tuning Suggestions

### Lower VRAM

Reduce:

- `max_model_len`
- `max_tokens`
- `gpu_memory_utilization`

Increase only if needed:

- `cpu_offload_gb`

### Faster RAG

- Lower `top_k`.
- Disable hybrid reranking for speed-sensitive cases.
- Use `explicit` or `smart` RAG strategy instead of `always`.

### More Accurate RAG

- Keep hybrid search enabled.
- Improve ingestion chunking.
- Add page-aware PDF metadata.
- Add a reranker as a later second-stage retrieval step.

## Useful Commands

```bash
# UI only
uv run python run.py --mock --dev

# Real model
uv run python run.py --dev

# Tests
uv run pytest

# GPU watch
watch -n 1 nvidia-smi

# Build
uv run python build.py
```

## Current Strengths

- Fully local app architecture.
- Practical model choice for consumer GPUs.
- CPU embeddings preserve GPU memory for generation.
- Qdrant local mode avoids a separate vector DB service.
- SQLite keeps sessions simple and portable.
- Mock mode makes UI development possible without loading the LLM.
