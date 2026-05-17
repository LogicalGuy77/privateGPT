# Phase 4 Implementation Guide

Phase 4 added hybrid retrieval, context-budgeting helpers, performance metrics,
duplicate-detection support, and a more complete desktop UI. This document
describes what is implemented in the current code and calls out the remaining
caveats.

## Implemented Features

### 1. Hybrid BM25 + Semantic Search

File: `src/private_gpt_app/rag/hybrid_search.py`

`HybridSearchEngine` reranks retrieved chunks using a weighted blend of semantic
similarity and BM25 keyword score.

Current retrieval flow:

```python
semantic_results = vector_store.search(query, limit=top_k * 2)

documents = [
    {"text": result["text"], "metadata": result["metadata"]}
    for result in semantic_results
]
hybrid_search.index_documents(documents)

bm25_results = hybrid_search.search_bm25(query, top_k=top_k * 2)

final_results = hybrid_search.merge_results(
    semantic_results,
    bm25_results,
    semantic_weight=0.6,
    bm25_weight=0.4,
)[:top_k]
```

Important detail: BM25 is built over the semantic candidate set returned by
Qdrant, not over every document in the knowledge base.

### 2. Token-Based Context Utilities

File: `src/private_gpt_app/rag/chunking.py`

`TokenBasedChunker` can:

- count tokens
- truncate text to a token budget
- split text by tokens

Current production use is in retrieval context construction:

```python
chunk_tokens = token_chunker.count_tokens(text)
text = token_chunker.truncate_to_tokens(text, remaining_tokens)
```

Caveat: document ingestion still uses `TextSplitter` from `rag/ingestion.py`,
which is character-based. Token-based ingestion is available as a future
improvement but is not wired into the ingestion worker yet.

### 3. Smart Context Truncation

File: `src/private_gpt_app/backend/router.py`

`RetrievalService._construct_smart_context()` keeps retrieved context within a
small token budget before passing it to the LLM.

Current behavior:

- `max_context_tokens = 1000`
- 50 tokens are reserved for formatting.
- Chunks are added in ranked order.
- If a chunk is too large, it is truncated when at least 50 tokens remain.
- Source filenames are collected for citation text in the UI.

### 4. Performance Monitoring

Files:

- `src/private_gpt_app/utils/performance.py`
- `src/private_gpt_app/ui/performance_dialog.py`

Tracked operations include:

- `rag_retrieval`
- `semantic_search`
- `hybrid_rerank`
- `context_construction`

The Performance Stats dialog is accessible from:

```text
Tools -> Performance Stats
```

Metrics populate after the app performs retrieval work.

### 5. Duplicate Document Detection Support

File: `src/private_gpt_app/backend/document_store.py`

The document registry computes a SHA256 hash and can reject duplicate document
records:

```python
duplicate = document_store.is_duplicate(file_path)
```

Caveat: the current ingestion worker calls `vector_store.add_documents()` before
`document_store.add_document()`. That means a duplicate file can still insert
duplicate chunks into Qdrant before SQLite rejects the duplicate registry row.

Recommended fix:

```text
check duplicate -> skip if duplicate -> extract/chunk/embed/upsert -> add registry row
```

### 6. Modern UI and Navigation

Files:

- `src/private_gpt_app/ui/styles_modern.qss`
- `src/private_gpt_app/ui/main_window.py`
- `src/private_gpt_app/ui/session_sidebar.py`
- `src/private_gpt_app/ui/knowledge_base_dialog.py`
- `src/private_gpt_app/ui/settings_dialog.py`
- `src/private_gpt_app/ui/file_picker_widget.py`

The app includes:

- menu bar
- session sidebar
- session search
- knowledge base dialog
- settings dialog
- performance dialog
- RAG toggle
- file attachment picker for document-focused queries

## Current RAG Strategies

Configured by `RetrievalService.rag_strategy`:

- `always`: default, use RAG when document results exist.
- `smart`: use semantic relevance threshold.
- `explicit`: use RAG only for explicit file references.

The Settings dialog can change the strategy at runtime.

## Known Gaps

- Token-based chunking is not used during ingestion.
- Duplicate detection should happen before embedding/upserting chunks.
- Session rename UI is partial; right-click Rename makes the list item editable,
  but the new title is not persisted.
- The app simulates streaming by chunking a full vLLM response; vLLM itself is
  not streaming tokens yet.
- Benchmark numbers in older notes should be treated as targets/observations,
  not as verified automated test results.

## Testing

Manual checks:

```bash
uv run python run.py --mock --dev
uv run python run.py --dev
uv run pytest
```

Exercise:

1. Add a document through Knowledge Base.
2. Ask a document question with RAG on.
3. Check source citation names.
4. Open Performance Stats after a RAG query.
5. Toggle between RAG strategies in Settings.

## Packaging

Current packaging uses PyInstaller through:

```bash
uv run python build.py
```

The build script expects a local bundled model at:

```text
models/Qwen2.5-1.5B-Instruct-AWQ/
```

Nuitka is not the current build path in this repo.

## Next Improvements

- Move duplicate detection before vector upsert.
- Use `TokenBasedChunker` during ingestion.
- Persist session rename.
- Add true vLLM streaming if/when the chosen vLLM API path supports it cleanly.
- Add automated tests for session CRUD, document ingestion, duplicate handling,
  retrieval strategy decisions, and context truncation.
