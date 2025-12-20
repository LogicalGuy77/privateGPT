from functools import lru_cache
from typing import List
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Force CPU to save VRAM for the LLM
        self.device = "cpu"
        # Use a lightweight model optimized for speed/quality balance
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading embedding model {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Enable multi-threading for CPU inference
        torch.set_num_threads(4)
        print("Embedding model loaded.")

    @lru_cache(maxsize=2000)  # Increased cache size
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query string (cached)."""
        # encode returns a numpy array, convert to list
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for documents with batching for efficiency."""
        if len(texts) <= batch_size:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        
        # Process in batches for very large document sets
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self.embed_query.cache_clear()
        print("Embedding cache cleared.")

# Global instance
embedding_service = EmbeddingService()
