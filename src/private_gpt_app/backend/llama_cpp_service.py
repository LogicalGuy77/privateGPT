"""LLM service using llama.cpp for Nemotron Nano 9B v2 GGUF - optimized for 8GB VRAM."""

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator, Callable
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    reasoning_mode: str = "/no_think"  # "/think" or "/no_think"
    
    def __post_init__(self):
        if self.reasoning_mode not in ["/think", "/no_think"]:
            self.reasoning_mode = "/no_think"


class LlamaCppService:
    """Service for LLM inference using llama.cpp - optimized for low VRAM."""
    
    # Default model path relative to project root
    DEFAULT_MODEL_PATH = "models/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_S.gguf"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,  # Context length
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_batch: int = 512,  # Batch size for prompt processing
        verbose: bool = False
    ):
        """
        Initialize llama.cpp service for Nemotron Nano 9B v2.
        
        Args:
            model_path: Path to GGUF model file (relative or absolute)
            n_ctx: Context length (max tokens in conversation)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_batch: Batch size for prompt processing
            verbose: Enable verbose logging
        """
        # Resolve model path
        if model_path is None:
            # Use default path relative to private-gpt-app folder
            # Path: backend/ -> private_gpt_app/ -> src/ -> private-gpt-app/
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = str(project_root / self.DEFAULT_MODEL_PATH)
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.verbose = verbose
        
        self._llm = None
        self._is_loaded = False
        self._load_lock = asyncio.Lock()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    async def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Load the llama.cpp model.
        
        Args:
            progress_callback: Called with status updates
        """
        async with self._load_lock:
            if self._is_loaded:
                return
            
            if progress_callback:
                progress_callback(f"🔄 Loading model from {self.model_path}...")
            
            print(f"📦 Loading Nemotron Nano 9B v2 GGUF with llama.cpp")
            print(f"📁 Model Path: {self.model_path}")
            print(f"📏 Context Length: {self.n_ctx}")
            print(f"🎮 GPU Layers: {self.n_gpu_layers}")
            print(f"📦 Batch Size: {self.n_batch}")
            
            # Verify model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Download it with:\n"
                    f"  wget -c 'https://huggingface.co/bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF/"
                    f"resolve/main/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_S.gguf' -O {self.model_path}"
                )
            
            try:
                from llama_cpp import Llama
                
                # Load in executor to not block event loop
                loop = asyncio.get_event_loop()
                
                def _load():
                    return Llama(
                        model_path=self.model_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        n_batch=self.n_batch,
                        verbose=self.verbose,
                        # Use chat format for proper message handling
                        chat_format="chatml",
                    )
                
                self._llm = await loop.run_in_executor(None, _load)
                
                self._is_loaded = True
                print("✅ llama.cpp model loaded successfully!")
                print(f"💾 Using GGUF Q4_K_S quantization (~6.2GB)")
                
                if progress_callback:
                    progress_callback("✅ Model ready!")
            
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python not installed. "
                    "Install with: uv pip install llama-cpp-python --extra-index-url "
                    "https://abetlen.github.io/llama-cpp-python/whl/cu124"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load model with llama.cpp: {e}") from e
    
    async def generate_stream(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        Generate text with token streaming.
        
        Args:
            messages: Chat messages in format [{"role": "user", "content": "..."}]
            config: Generation configuration
        
        Yields:
            Generated tokens one at a time
        """
        if not self._is_loaded:
            await self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        print(f"\n🤖 Generating response (max {config.max_tokens} tokens, {config.reasoning_mode})...")
        
        try:
            # Add reasoning mode to messages
            enhanced_messages = self._add_reasoning_mode(messages, config.reasoning_mode)
            
            # Generate with streaming
            loop = asyncio.get_event_loop()
            
            # Create chat completion with streaming
            def _create_stream():
                return self._llm.create_chat_completion(
                    messages=enhanced_messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    stream=True,
                )
            
            # Get the stream generator
            stream = await loop.run_in_executor(None, _create_stream)
            
            # Yield tokens from stream
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
                    await asyncio.sleep(0)  # Allow other tasks to run
        
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise
    
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text (non-streaming).
        
        Args:
            messages: Chat messages
            config: Generation configuration
        
        Returns:
            Complete generated text
        """
        if not self._is_loaded:
            await self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        try:
            # Add reasoning mode
            enhanced_messages = self._add_reasoning_mode(messages, config.reasoning_mode)
            
            # Generate in executor
            loop = asyncio.get_event_loop()
            
            def _generate():
                response = self._llm.create_chat_completion(
                    messages=enhanced_messages,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_tokens,
                    stream=False,
                )
                return response["choices"][0]["message"]["content"]
            
            return await loop.run_in_executor(None, _generate)
        
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise
    
    def _add_reasoning_mode(self, messages: list[dict], reasoning_mode: str) -> list[dict]:
        """
        Add reasoning mode to messages.
        
        Args:
            messages: Original chat messages
            reasoning_mode: "/think" or "/no_think"
        
        Returns:
            Enhanced messages with reasoning mode
        """
        if not messages:
            return messages
        
        # Clone messages
        enhanced = [msg.copy() for msg in messages]
        
        # Check if first message is system message
        if enhanced and enhanced[0].get('role') == 'system':
            # Append to existing system message
            enhanced[0] = {
                'role': 'system',
                'content': f"{enhanced[0]['content']} {reasoning_mode}".strip()
            }
        else:
            # Insert new system message
            enhanced.insert(0, {'role': 'system', 'content': reasoning_mode})
        
        return enhanced
    
    def format_chat_messages(self, conversation_history: list[dict]) -> list[dict]:
        """
        Format conversation history (already in correct format for llama.cpp).
        
        Args:
            conversation_history: List of {role: str, content: str} dicts
        
        Returns:
            Formatted messages
        """
        return conversation_history
    
    def unload_model(self) -> None:
        """Unload model from memory to free VRAM."""
        if self._is_loaded:
            print("🔄 Unloading llama.cpp model...")
            
            # llama.cpp cleanup
            if self._llm is not None:
                del self._llm
            
            self._llm = None
            self._is_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            print("✓ llama.cpp model unloaded")
