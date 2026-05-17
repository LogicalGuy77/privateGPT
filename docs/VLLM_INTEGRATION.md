# vLLM Integration

This document describes the current vLLM integration in Private-GPT.

## Current Status

The app uses vLLM to run:

```text
Qwen/Qwen2.5-3B-Instruct-AWQ
```

The old Nemotron/Nano experiment is no longer the active runtime path. There is
no `/think` or `/no_think` reasoning-mode toggle in the current code.

## Main Files

- `src/private_gpt_app/backend/vllm_service.py`
- `src/private_gpt_app/ui/main_window.py`
- `src/private_gpt_app/utils/setup_manager.py`

`MainWindow.initialize_llm()` creates `VLLMService`. `VLLMService.load_model()`
then constructs a vLLM `LLM` object in a thread executor so the PyQt event loop
does not freeze during model loading.

## Runtime Configuration

Current defaults:

```python
VLLMService(
    model_name="Qwen/Qwen2.5-3B-Instruct-AWQ",
    dtype="float16",
    quantization="awq_marlin",
    gpu_memory_utilization=0.55,
    max_model_len=4096,
    max_num_seqs=4,
    cpu_offload_gb=2.0,  # passed from MainWindow settings
)
```

vLLM is called with:

```python
LLM(
    model=model_name,
    dtype="float16",
    quantization="awq_marlin",
    gpu_memory_utilization=...,
    max_model_len=...,
    max_num_seqs=...,
    enforce_eager=True,
    enable_prefix_caching=True,
    cpu_offload_gb=...,  # only when > 0
)
```

`enforce_eager=True` favors stability over CUDA graph performance.
`enable_prefix_caching=True` lets vLLM reuse repeated prompt prefixes where
possible.

## Model Resolution

`setup_manager.get_model_path()` resolves the model in this order:

1. Bundled model under `models/Qwen2.5-3B-Instruct-AWQ/`.
2. Saved config under `~/.private-gpt/model_config.txt`.
3. Common local model locations.
4. HuggingFace model ID `Qwen/Qwen2.5-3B-Instruct-AWQ`.

In packaged mode, `get_bundled_model_path()` also checks PyInstaller/Nuitka-style
bundle locations, but the current build script uses PyInstaller.

## Generation

`GenerationConfig` currently exposes:

```python
GenerationConfig(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    system_prompt=...
)
```

The system prompt is inserted if the conversation does not already begin with a
system message. When RAG context is available, the context is appended to the
latest user message before formatting the prompt with the Qwen chat template.

## Streaming Behavior

The UI displays text incrementally, but the current vLLM call is not true token
streaming:

1. `LLM.generate()` produces the full response synchronously.
2. The synchronous call runs in an executor.
3. The full response is yielded back to the UI in 24-character chunks.

This preserves a streaming-like UI while avoiding direct event-loop blocking.

## Running

Mock mode, no main LLM:

```bash
uv run python run.py --mock --dev
```

Real model mode:

```bash
uv run python run.py --dev
```

Check GPU availability:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Docker is not required for normal local usage.

## Hardware Notes

The code currently warns below 6GB VRAM:

```python
validate_hardware_requirements(gpu_info, min_vram_gb=6.0)
```

The memory settings are conservative, but the active default context is 4096
tokens. For lower VRAM GPUs, reduce `max_model_len`, `max_tokens`, and/or
`gpu_memory_utilization`.

## Troubleshooting

### vLLM Import Failure

Run:

```bash
uv sync
uv run python -c "import vllm; print('vllm ok')"
```

### CUDA Not Available

Check:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Lower these settings in `MainWindow.current_settings` or through the settings UI
followed by a restart:

- `gpu_memory_utilization`
- `max_model_len`
- `cpu_offload_gb`
- `max_tokens`

### Stale GPU Processes

The app attempts to clean stale vLLM processes on startup. You can also inspect
with:

```bash
nvidia-smi
```
