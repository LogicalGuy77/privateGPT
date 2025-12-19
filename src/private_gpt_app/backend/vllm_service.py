"""LLM service using vLLM for Nemotron Nano 9B v2 - optimized for 8GB VRAM + 32GB RAM."""

import asyncio
import os
from typing import Optional, AsyncIterator, Callable
from dataclasses import dataclass

# CRITICAL: Set memory optimizations BEFORE importing vLLM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
# Force V0 engine which has different memory handling
os.environ["VLLM_USE_V1"] = "0"


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


class VLLMService:
    """Service for LLM inference using vLLM - optimized for low VRAM with CPU offloading."""
    
    def __init__(
        self,
        model_name: str = "RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16",
        dtype: str = "float16",
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,  # Use 85% of VRAM for model
        max_model_len: int = 2048,  # Context length
        kv_cache_dtype: str = "fp8",  # 8-bit KV cache saves ~50% cache memory
        cpu_offload_gb: float = 8.0,  # Offload 8GB to CPU RAM
        verbose: bool = False
    ):
        """
        Initialize vLLM service for Nemotron Nano 9B v2.
        
        Args:
            model_name: HuggingFace model ID
            dtype: Data type ("bfloat16", "float16", "auto")
            quantization: Quantization method ("awq", "gptq", "squeezellm", None)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum context length
            kv_cache_dtype: Key/Value cache data type ("fp8" saves memory)
            cpu_offload_gb: Amount of model to offload to CPU RAM (GB)
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.dtype = dtype
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.kv_cache_dtype = kv_cache_dtype
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
                progress_callback(f"🔄 Loading {self.model_name} with vLLM...")
            
            print(f"📦 Loading Nemotron Nano 9B v2 with vLLM (V0 Engine)")
            print(f"⚙️  GPU Memory Utilization: {self.gpu_memory_utilization * 100}%")
            print(f"📏 Max Context Length: {self.max_model_len}")
            print(f"🎮 Data Type: {self.dtype}")
            print(f"🧠 KV Cache Type: {self.kv_cache_dtype}")
            print(f"💾 CPU Offload: {self.cpu_offload_gb}GB to RAM")
            
            try:
                from vllm import LLM
                
                # Load in executor to not block event loop
                loop = asyncio.get_event_loop()
                
                def _load():
                    return LLM(
                        model=self.model_name,
                        trust_remote_code=True,
                        dtype=self.dtype,
                        quantization=self.quantization,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        kv_cache_dtype=self.kv_cache_dtype,
                        enforce_eager=True,  # Disable CUDA graphs to save VRAM
                        cpu_offload_gb=self.cpu_offload_gb,  # Offload to CPU RAM
                        swap_space=16,  # 16GB swap space for KV cache overflow
                        max_num_seqs=1,  # CRITICAL: Only 1 sequence at a time (reduces Mamba cache)
                        disable_custom_all_reduce=True,  # Save memory
                    )
                
                self._llm = await loop.run_in_executor(None, _load)
                
                self._is_loaded = True
                print("✅ vLLM model loaded successfully!")
                print(f"💾 Using CPU offloading: {self.cpu_offload_gb}GB offloaded to RAM")
                
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
        
        print(f"\n🤖 Generating response (max {config.max_tokens} tokens, {config.reasoning_mode})...")
        
        try:
            from vllm import SamplingParams
            
            # Add reasoning mode to messages
            enhanced_messages = self._add_reasoning_mode(messages, config.reasoning_mode)
            
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
            
            # Add reasoning mode
            enhanced_messages = self._add_reasoning_mode(messages, config.reasoning_mode)
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
        enhanced = messages.copy()
        
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
    
    def _format_chat_messages(self, messages: list[dict]) -> str:
        """
        Format messages into Nemotron chat template format.
        
        Args:
            messages: Chat messages
        
        Returns:
            Formatted prompt string
        """
        # Nemotron uses a simple chat format
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"<extra_id_0>System\n{content}\n")
            elif role == 'user':
                prompt_parts.append(f"<extra_id_1>User\n{content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"<extra_id_1>Assistant\n{content}\n")
        
        # Add assistant prompt
        prompt_parts.append("<extra_id_1>Assistant\n")
        
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
