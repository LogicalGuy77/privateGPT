# Private-GPT Testing Guide

This guide reflects the current PyQt6 desktop app.

## Smoke Test: Mock Mode

Mock mode skips the main Qwen/vLLM model and uses fake assistant responses. It
may still initialize Qdrant and download/load the CPU embedding model because the
RAG stack is imported by the UI.

```bash
cd ~/coding/private-gpt/privateGPT
uv sync
uv run python run.py --mock --dev
```

Expected terminal output includes:

- `Running in MOCK mode`
- `Development mode enabled`
- `Loaded stylesheet: styles_modern.qss`
- `Hot-reload enabled for QSS`

Expected UI behavior:

- Window opens with title `Private-GPT`.
- Left sidebar contains New Chat, session search/list, Knowledge Base, RAG toggle,
  VRAM/status labels, and Settings.
- Chat area shows the welcome message.
- Message input, Attach, Clear, and Send controls are visible.
- Sending a message returns a mock response with incremental UI updates.

## Real Model Test

Use this after mock mode works:

```bash
uv run python run.py --dev
```

Before running, verify GPU access:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected behavior:

- GPU check runs.
- Qwen2.5-1.5B-Instruct-AWQ loads through vLLM.
- Status changes to model ready.
- `nvidia-smi` shows Python/vLLM using VRAM.

## Manual Checklist

### Chat

- Send with the Send button.
- Send with `Ctrl+Enter`.
- Clear input with Clear.
- Start a new chat with `Ctrl+N`.
- Confirm session appears in sidebar.
- Switch sessions and verify messages reload.

### RAG

1. Open `Tools > Knowledge Base` or press `Ctrl+K`.
2. Add a small `.txt`, `.md`, `.pdf`, or `.docx` file.
3. Confirm the document appears in the list.
4. Ask a question about the document.
5. Confirm the response includes source citation text when RAG is used.
6. Toggle RAG off and confirm future queries skip retrieval.

### Attachments

1. Click Attach.
2. Select one or more knowledge-base files.
3. Ask a question.
4. Confirm the RAG label shows selected source files.
5. Confirm file selection clears after the response.

### Settings

- Open Settings.
- Change generation settings.
- Change RAG strategy:
  - Always
  - Smart
  - Explicit Only
- Confirm RAG settings apply immediately.
- Restart app after changing model memory settings if you need a clean model reload.

### Performance Stats

1. Send a RAG query.
2. Open `Tools > Performance Stats`.
3. Confirm retrieval/search metrics appear after usage.

### Crash Recovery

Crash recovery files live under `data/crash_recovery/`. A clean window close
ends the recovery session. Unexpected termination should leave recoverable data
for the next launch.

## Development Notes

### Stylesheet Hot Reload

With `--dev`, the loaded stylesheet reloads when modified. The app prefers:

```text
src/private_gpt_app/ui/styles_modern.qss
```

and falls back to:

```text
src/private_gpt_app/ui/styles.qss
```

### Automated Tests

The current repo has minimal test files. Run:

```bash
uv run pytest
```

This is currently more of an environment sanity check than a comprehensive test
suite.

### Useful One-Off Checks

```bash
# Import check
uv run python -c "from private_gpt_app.backend.vllm_service import VLLMService; print('ok')"

# GPU check
uv run python -c "import torch; print(torch.cuda.is_available())"

# Qdrant/vector store check
uv run python -c "from private_gpt_app.rag.vector_store import vector_store; print(vector_store.collection_name)"
```
