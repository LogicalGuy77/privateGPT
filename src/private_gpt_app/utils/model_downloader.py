"""Model downloader utility - simplified for vLLM (models auto-download from HuggingFace)."""

from pathlib import Path
from typing import Optional


class ModelDownloader:
    """Handles model cache directory setup. vLLM auto-downloads models from HuggingFace Hub."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model downloader.
        
        Args:
            cache_dir: Directory to cache models. Defaults to ~/.cache/huggingface/hub/
        """
        if cache_dir is None:
            # Use HuggingFace's default cache (vLLM uses this)
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_dir(self) -> Path:
        """Get the HuggingFace cache directory."""
        return self.cache_dir
    
    def check_model_cached(self, model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2") -> bool:
        """
        Check if model is already cached.
        
        Args:
            model_name: HuggingFace model ID
        
        Returns:
            True if model exists in cache
        """
        # HuggingFace cache format: models--nvidia--NVIDIA-Nemotron-Nano-9B-v2
        cache_name = "models--" + model_name.replace("/", "--")
        model_cache_path = self.cache_dir / cache_name
        
        return model_cache_path.exists() and any(model_cache_path.iterdir())
