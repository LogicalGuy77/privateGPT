"""LLM service using llama.cpp for inference."""

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator, Callable
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["</s>", "[/INST]"]


class LlamaCppService:
    """Service for LLM inference using llama.cpp."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        """
        Initialize llama.cpp service.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default 8k, max 32k)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
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
        Lazily load the model (only when first needed).
        
        Args:
            progress_callback: Called with status updates
        """
        async with self._load_lock:
            if self._is_loaded:
                return
            
            if self.model_path is None:
                raise ValueError("No model path specified. Please download a model first.")
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
            if progress_callback:
                progress_callback("🔄 Loading model into memory...")
            
            print(f"📦 Loading model: {self.model_path}")
            print(f"⚙️  Context window: {self.n_ctx} tokens")
            print(f"🎮 GPU layers: {self.n_gpu_layers}")
            
            try:
                # Import here to avoid loading at startup
                from llama_cpp import Llama
                
                # Load model (this happens in a thread pool to not block)
                loop = asyncio.get_event_loop()
                self._llm = await loop.run_in_executor(
                    None,
                    lambda: Llama(
                        model_path=str(self.model_path),
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        verbose=self.verbose,
                        n_threads=4,  # CPU threads for CPU operations
                    )
                )
                
                self._is_loaded = True
                print("✅ Model loaded successfully!")
                
                if progress_callback:
                    progress_callback("✅ Model ready!")
            
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed. "
                    "Install with: uv pip install llama-cpp-python"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")
    
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        Generate text with token streaming.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
        
        Yields:
            Generated tokens one at a time
        """
        if not self._is_loaded:
            await self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        print(f"\n🤖 Generating response (max {config.max_tokens} tokens)...")
        
        try:
            # Run generation in executor to not block event loop
            loop = asyncio.get_event_loop()
            
            # Create generator
            stream = await loop.run_in_executor(
                None,
                lambda: self._llm(
                    prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    stop=config.stop_sequences,
                    stream=True,
                )
            )
            
            # Stream tokens
            for output in stream:
                token = output['choices'][0]['text']
                yield token
                await asyncio.sleep(0)  # Allow other tasks to run
        
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text (non-streaming).
        
        Args:
            prompt: Input prompt
            config: Generation configuration
        
        Returns:
            Complete generated text
        """
        if not self._is_loaded:
            await self.load_model()
        
        if config is None:
            config = GenerationConfig()
        
        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._llm(
                    prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    stop=config.stop_sequences,
                    stream=False,
                )
            )
            
            return output['choices'][0]['text']
        
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise
    
    def format_chat_prompt(self, messages: list[dict]) -> str:
        """
        Format chat messages into a prompt string.
        
        Args:
            messages: List of {role: str, content: str} dicts
        
        Returns:
            Formatted prompt
        """
        # Llama-style chat format
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == 'user':
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"<|assistant|>\n{content}\n")
        
        # Add final assistant tag to prompt completion
        prompt_parts.append("<|assistant|>\n")
        
        return "".join(prompt_parts)
    
    def unload_model(self) -> None:
        """Unload model from memory to free VRAM."""
        if self._is_loaded:
            print("🔄 Unloading model...")
            self._llm = None
            self._is_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            print("✓ Model unloaded")
