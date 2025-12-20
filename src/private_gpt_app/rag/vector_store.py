import os
import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .embeddings import embedding_service

class VectorStore:
    def __init__(self, collection_name: str = "private_gpt_docs"):
        self.collection_name = collection_name
        # Ensure data directory exists
        self.db_path = os.path.join(os.getcwd(), "data", "qdrant_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        print(f"Initializing Qdrant at {self.db_path}...")
        self.client = QdrantClient(path=self.db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Dimension for all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to the vector store."""
        if not texts:
            return
            
        embeddings = embedding_service.embed_documents(texts)
        
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": text, **metadata}
            )
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(texts)} documents to Qdrant.")

    def search(self, query: str, limit: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant documents."""
        query_vector = embedding_service.embed_query(query)
        
        query_filter = None
        if filter_metadata:
            # Simple equality match for now
            conditions = [
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
                for key, value in filter_metadata.items()
            ]
            if conditions:
                query_filter = models.Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score
            }
            for hit in results
        ]
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        print(f"Cleared collection '{self.collection_name}'.")

# Global instance
vector_store = VectorStore()
