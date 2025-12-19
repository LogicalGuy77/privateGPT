"""LLM service using HuggingFace Transformers for Nemotron Nano 9B v2."""

import asyncio
import threading
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


class TransformersService:
    """Service for LLM inference using HuggingFace Transformers with 4-bit quantization."""
    
    def __init__(
        self,
        model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        device_map: str = "auto",
        dtype: str = "float16",
        use_4bit: bool = True,
        verbose: bool = False
    ):
        """
        Initialize Transformers service for Nemotron Nano 9B v2.
        
        Args:
            model_name: HuggingFace model ID
            device_map: Device mapping strategy ("auto", "cuda", "cpu")
            dtype: Data type ("bfloat16", "float16", "float32")
            use_4bit: Use 4-bit quantization (recommended for 8GB VRAM)
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.use_4bit = use_4bit
        self.verbose = verbose
        
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._load_lock = asyncio.Lock()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    async def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Lazily load the model and tokenizer.
        
        Args:
            progress_callback: Called with status updates
        """
        async with self._load_lock:
            if self._is_loaded:
                return
            
            if progress_callback:
                progress_callback(f"🔄 Loading {self.model_name}...")
            
            print(f"📦 Loading Nemotron Nano 9B v2 from HuggingFace Hub")
            print(f"🎮 Device: {self.device_map}")
            print(f"⚙️  Dtype: {self.dtype}")
            print(f"🔢 4-bit quantization: {self.use_4bit}")
            
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                
                # Determine torch dtype
                if self.dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif self.dtype == "float16":
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                
                # Load in executor to not block event loop
                loop = asyncio.get_event_loop()
                
                def _load():
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Configure 4-bit quantization for 8GB VRAM
                    if self.use_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_use_double_quant=True,  # Further memory savings
                            bnb_4bit_quant_type="nf4",  # Normalized float4
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            quantization_config=quantization_config,
                            trust_remote_code=True,
                            device_map=self.device_map,
                            low_cpu_mem_usage=True,
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch_dtype,
                            trust_remote_code=True,
                            device_map=self.device_map,
                            low_cpu_mem_usage=True,
                        )
                    return tokenizer, model
                
                self._tokenizer, self._model = await loop.run_in_executor(None, _load)
                
                self._is_loaded = True
                print("✅ Model loaded successfully!")
                
                if progress_callback:
                    progress_callback("✅ Model ready!")
            
            except ImportError as e:
                raise ImportError(
                    "Required dependencies not installed. "
                    "Install with: uv pip install transformers accelerate torch bitsandbytes"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}") from e
    
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
            from transformers import TextIteratorStreamer
            import queue
            
            # Add reasoning mode to system message
            enhanced_messages = self._add_reasoning_mode(messages, config.reasoning_mode)
            
            # Prepare inputs
            inputs = self._tokenizer.apply_chat_template(
                enhanced_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self._model.device)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=30.0
            )
            
            # Generation kwargs
            gen_kwargs = dict(
                inputs=inputs,
                streamer=streamer,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                eos_token_id=self._tokenizer.eos_token_id,
                do_sample=config.temperature > 0
            )
            
            # Run generation in background thread
            thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
            thread.start()
            
            # Stream tokens as they arrive
            loop = asyncio.get_event_loop()
            
            async def stream_wrapper():
                try:
                    for new_text in streamer:
                        yield new_text
                        await asyncio.sleep(0)  # Allow other tasks to run
                except queue.Empty:
                    print("⚠️ Stream timeout")
                except Exception as e:
                    print(f"❌ Stream error: {e}")
                finally:
                    thread.join(timeout=5.0)
            
            async for token in stream_wrapper():
                yield token
        
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
            import torch
            
            # Add reasoning mode
            enhanced_messages = self._add_reasoning_mode(messages, config.reasoning_mode)
            
            # Prepare inputs
            inputs = self._tokenizer.apply_chat_template(
                enhanced_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self._model.device)
            
            # Generate in executor
            loop = asyncio.get_event_loop()
            
            def _generate():
                with torch.no_grad():
                    outputs = self._model.generate(
                        inputs,
                        max_new_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        eos_token_id=self._tokenizer.eos_token_id,
                        do_sample=config.temperature > 0
                    )
                
                # Decode only the new tokens
                response = self._tokenizer.decode(
                    outputs[0][inputs.shape[1]:],
                    skip_special_tokens=True
                )
                return response
            
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
        
        # Clone messages to avoid modifying original
        enhanced = messages.copy()
        
        # Check if first message is system message
        if enhanced[0].get('role') == 'system':
            # Append reasoning mode to existing system message
            enhanced[0] = {
                'role': 'system',
                'content': f"{enhanced[0]['content']} {reasoning_mode}".strip()
            }
        else:
            # Insert new system message with reasoning mode
            enhanced.insert(0, {'role': 'system', 'content': reasoning_mode})
        
        return enhanced
    
    def format_chat_messages(self, conversation_history: list[dict]) -> list[dict]:
        """
        Format conversation history into messages format.
        
        Args:
            conversation_history: List of {role: str, content: str} dicts
        
        Returns:
            Formatted messages
        """
        # Already in correct format for transformers
        return conversation_history
    
    def unload_model(self) -> None:
        """Unload model from memory to free VRAM."""
        if self._is_loaded:
            print("🔄 Unloading model...")
            self._model = None
            self._tokenizer = None
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
