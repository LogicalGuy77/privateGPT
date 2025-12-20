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
        
        # Lazy initialization - don't create client until first use
        self.client = None
    
    def _ensure_client(self):
        """Lazy initialization of Qdrant client."""
        if self.client is None:
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

    def add_documents(self, texts: List[str], metadatas: List[Dict], batch_size: int = 100):
        """Add documents to vector store with batching for large document sets."""
        if not texts:
            return
        
        self._ensure_client()  # Lazy init
        
        # Process embeddings in batches
        from .embeddings import embedding_service
        embeddings = embedding_service.embed_documents(texts, batch_size=batch_size)
        
        # Batch upsert for efficiency
        total_points = len(texts)
        for i in range(0, total_points, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": text, **metadata}
                )
                for text, embedding, metadata in zip(batch_texts, batch_embeddings, batch_metadatas)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            if total_points > batch_size:
                print(f"Progress: {min(i + batch_size, total_points)}/{total_points} documents")
        
        print(f"Added {len(texts)} documents to Qdrant.")

    def search(self, query: str, limit: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant documents."""
        self._ensure_client()  # Lazy init
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

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score
            }
            for hit in results.points
        ]
    
    def clear(self):
        """Clear all documents from the collection."""
        self._ensure_client()  # Lazy init
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        print(f"Cleared collection '{self.collection_name}'.")

    def delete_document(self, filename: str):
        """Delete all chunks associated with a specific filename."""
        self._ensure_client()  # Lazy init
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
        )
        print(f"Deleted document '{filename}' from Qdrant.")

# Global instance
vector_store = VectorStore()
