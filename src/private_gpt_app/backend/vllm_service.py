"""LLM service using vLLM for Qwen2.5 - optimized for 8GB VRAM."""

import asyncio
import os
from typing import Optional, AsyncIterator, Callable
from dataclasses import dataclass

# Memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful, harmless, and honest AI assistant."


class VLLMService:
    """Service for LLM inference using vLLM - optimized for 8GB VRAM GPUs."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct-AWQ",
        dtype: str = "float16",
        quantization: str = "awq_marlin",  # Efficient AWQ with Marlin kernels
        gpu_memory_utilization: float = 0.55,  # Works on 4GB+ VRAM GPUs
        max_model_len: int = 2048,  # Reduced context for 4GB compatibility
        max_num_seqs: int = 4,  # Limit concurrent sequences
        cpu_offload_gb: float = 0.0,  # CPU offload for extra headroom
        verbose: bool = False
    ):
        """
        Initialize vLLM service for Qwen2.5-Instruct-AWQ.
        
        Args:
            model_name: HuggingFace model ID or local path
            dtype: Data type ("float16" for AWQ)
            quantization: Quantization method ("awq_marlin" for efficiency)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum context length
            max_num_seqs: Maximum concurrent sequences
            cpu_offload_gb: Amount of memory to offload to CPU (GB)
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.dtype = dtype
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.cpu_offload_gb = cpu_offload_gb
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
        Load the vLLM model.
        
        Args:
            progress_callback: Called with status updates
        """
        async with self._load_lock:
            if self._is_loaded:
                return
            
            if progress_callback:
                progress_callback(f"🔄 Loading {self.model_name}...")
            
            print(f"📦 Loading Qwen2.5 with vLLM")
            print(f"⚙️  GPU Memory Utilization: {self.gpu_memory_utilization * 100}%")
            print(f"📏 Max Context Length: {self.max_model_len}")
            print(f"🎮 Quantization: {self.quantization}")
            if self.cpu_offload_gb > 0:
                print(f"💾 CPU Offload: {self.cpu_offload_gb}GB")
            
            try:
                from vllm import LLM
                
                # Load in executor to not block event loop
                loop = asyncio.get_event_loop()
                
                def _load():
                    kwargs = {
                        "model": self.model_name,
                        "dtype": self.dtype,
                        "quantization": self.quantization,
                        "gpu_memory_utilization": self.gpu_memory_utilization,
                        "max_model_len": self.max_model_len,
                        "max_num_seqs": self.max_num_seqs,
                        "enforce_eager": True,  # Disable CUDA graphs for stability
                    }
                    # Add CPU offload if specified
                    if self.cpu_offload_gb > 0:
                        kwargs["cpu_offload_gb"] = self.cpu_offload_gb
                    return LLM(**kwargs)
                
                self._llm = await loop.run_in_executor(None, _load)
                
                self._is_loaded = True
                print("✅ Qwen2.5 model loaded successfully!")
                
                if progress_callback:
                    progress_callback("✅ Model ready!")
            
            except ImportError as e:
                raise ImportError(
                    "vLLM not installed. "
                    "Install with: uv pip install vllm"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load model with vLLM: {e}") from e
    
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
        
        print(f"\n🤖 Generating response (max {config.max_tokens} tokens)...")
        
        try:
            from vllm import SamplingParams
            
            # Add system prompt if not present
            enhanced_messages = self._ensure_system_prompt(messages, config.system_prompt)
            
            # Format messages into prompt using chat template
            prompt = self._format_chat_messages(enhanced_messages)
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            
            # Note: vLLM doesn't have native async streaming in the SDK
            # We simulate streaming by generating and yielding chunks
            loop = asyncio.get_event_loop()
            
            def _generate():
                outputs = self._llm.generate([prompt], sampling_params=sampling_params)
                return outputs[0].outputs[0].text
            
            # Generate full response
            full_text = await loop.run_in_executor(None, _generate)
            
            # Simulate streaming by yielding in chunks
            chunk_size = 5  # Characters per chunk
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate streaming
        
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
            from vllm import SamplingParams
            
            # Add system prompt if not present
            enhanced_messages = self._ensure_system_prompt(messages, config.system_prompt)
            prompt = self._format_chat_messages(enhanced_messages)
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            
            # Generate in executor
            loop = asyncio.get_event_loop()
            
            def _generate():
                outputs = self._llm.generate([prompt], sampling_params=sampling_params)
                return outputs[0].outputs[0].text
            
            return await loop.run_in_executor(None, _generate)
        
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise
    
    def _ensure_system_prompt(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """
        Ensure messages have a system prompt.
        
        Args:
            messages: Original chat messages
            system_prompt: Default system prompt to add if none exists
        
        Returns:
            Messages with system prompt
        """
        if not messages:
            return [{'role': 'system', 'content': system_prompt}]
        
        # Clone messages
        enhanced = messages.copy()
        
        # Check if first message is system message
        if enhanced[0].get('role') != 'system':
            enhanced.insert(0, {'role': 'system', 'content': system_prompt})
        
        return enhanced
    
    def _format_chat_messages(self, messages: list[dict]) -> str:
        """
        Format messages into Qwen2.5 ChatML format.
        
        Args:
            messages: Chat messages
        
        Returns:
            Formatted prompt string
        """
        # Qwen2.5 uses ChatML format
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        
        # Add assistant prompt for generation
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "".join(prompt_parts)
    
    def format_chat_messages(self, conversation_history: list[dict]) -> list[dict]:
        """
        Format conversation history (already in correct format for vLLM).
        
        Args:
            conversation_history: List of {role: str, content: str} dicts
        
        Returns:
            Formatted messages
        """
        return conversation_history
    
    def unload_model(self) -> None:
        """Unload model from memory to free VRAM."""
        if self._is_loaded:
            print("🔄 Unloading vLLM model...")
            
            # vLLM cleanup
            if self._llm is not None:
                # Force cleanup of CUDA resources
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
            except:
                pass
            
            print("✓ vLLM model unloaded")
