# RAG Architecture

Private-GPT uses a local Retrieval-Augmented Generation pipeline for
document-grounded answers.

## Components

### Embeddings

File: `src/private_gpt_app/rag/embeddings.py`

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Device: CPU
- Vector dimension: 384
- Query embeddings are cached with `lru_cache`.
- CPU execution keeps GPU VRAM available for Qwen/vLLM.

### Vector Store

File: `src/private_gpt_app/rag/vector_store.py`

- Database: Qdrant local mode
- Path: `data/qdrant_db`
- Collection: `private_gpt_docs`
- Distance: cosine
- Supports metadata filtering, currently used mostly by source filename.

### Document Registry

File: `src/private_gpt_app/backend/document_store.py`

- Database: SQLite
- Path: `data/documents.db`
- Tracks filename, original path, file hash, upload date, and status.
- Includes duplicate detection by SHA256 hash.

Important caveat: ingestion currently writes chunks to Qdrant before inserting
the document registry row. If a duplicate is rejected by SQLite, duplicate chunks
may already have been inserted into Qdrant. Fixing that means checking
`document_store.is_duplicate()` before embedding/upserting chunks.

### Ingestion

File: `src/private_gpt_app/rag/ingestion.py`

Supported files:

- `.pdf` through PyMuPDF / `fitz`
- `.docx` through `python-docx`
- `.txt`
- `.md`

Current chunking for ingestion uses `TextSplitter`, a character-based splitter
with a default chunk size of 512 characters. Token-based chunking utilities exist
in `rag/chunking.py`, but they are not currently used by ingestion.

### Retrieval Router

File: `src/private_gpt_app/backend/router.py`

`RetrievalService` decides whether to use RAG and builds context.

Strategies:

- `always`: use RAG when any document search result exists.
- `smart`: use semantic relevance checks before using RAG.
- `explicit`: use RAG only when file references are explicit.

The current default is `always`.

### Hybrid Search

File: `src/private_gpt_app/rag/hybrid_search.py`

Retrieval first gets semantic Qdrant results, then optionally reranks those
candidate chunks with BM25:

- Semantic weight: 0.6
- BM25 weight: 0.4

BM25 is built over the retrieved candidate set, not over the entire knowledge
base.

### Context Truncation

File: `src/private_gpt_app/rag/chunking.py`

`TokenBasedChunker` is used by retrieval to count/truncate context before it is
sent to the LLM. `RetrievalService.max_context_tokens` is currently 1000, with
some tokens reserved for formatting.

## Data Flow

### Ingestion

```text
User adds file
-> text extraction
-> character-based chunks
-> CPU embeddings
-> Qdrant upsert
-> SQLite document registry insert
```

### Query

```text
User sends message
-> RetrievalService strategy check
-> Qdrant semantic search
-> optional BM25 rerank
-> token-budgeted context construction
-> context injected into latest user message
-> vLLM generation
-> UI response with source names
```

## Example

```python
from private_gpt_app.rag.vector_store import vector_store

vector_store.add_documents(
    texts=["Private-GPT is a local chat app.", "It uses Qwen2.5."],
    metadatas=[{"source": "intro.txt"}, {"source": "intro.txt"}],
)

results = vector_store.search("What model does it use?")
for result in results:
    print(result["metadata"], result["score"])
```
