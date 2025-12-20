from functools import lru_cache
from typing import List
import torch
from sentence_transformers import SentenceTransformer

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
        print("Embedding model loaded.")

    @lru_cache(maxsize=1000)
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query string."""
        # encode returns a numpy array, convert to list
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

# Global instance
embedding_service = EmbeddingService()
