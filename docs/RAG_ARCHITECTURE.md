# RAG Architecture (Phase 2)

This document outlines the architecture for the Retrieval-Augmented Generation (RAG) pipeline in Private-GPT.

## Components

### 1. Embeddings (`src/private_gpt_app/rag/embeddings.py`)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Device:** CPU (to reserve VRAM for the LLM)
- **Dimension:** 384
- **Caching:** `lru_cache` used for repeated query embeddings.

### 2. Vector Store (`src/private_gpt_app/rag/vector_store.py`)
- **Database:** Qdrant (Local Mode)
- **Path:** `data/qdrant_db`
- **Collection:** `private_gpt_docs`
- **Features:**
    - Persistent on-disk storage.
    - Metadata filtering (filename, page number).
    - Cosine similarity search.

### 3. Ingestion Pipeline (Planned)
- **PDFs:** `PyMuPDF` (fitz) for fast extraction.
- **Word:** `python-docx`.
- **Chunking:** Recursive character splitter (512 tokens, 50 overlap).
- **Concurrency:** `QThreadPool` for background processing.

## Data Flow

1. **Ingestion:**
   User uploads file -> Text extracted -> Chunked -> Embedded (CPU) -> Stored in Qdrant.

2. **Retrieval:**
   User query -> Embedded (CPU) -> Qdrant Search (Top-k) -> Context constructed.

3. **Generation:**
   Context + Query -> vLLM (GPU) -> Response.

## Usage

```python
from private_gpt_app.rag.vector_store import vector_store

# Add documents
vector_store.add_documents(
    texts=["Private-GPT is a local chat app.", "It uses Qwen2.5."],
    metadatas=[{"source": "intro.txt"}, {"source": "intro.txt"}]
)

# Search
results = vector_store.search("What model does it use?")
for res in results:
    print(f"{res['text']} (Score: {res['score']})")
```
