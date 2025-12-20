"""Hybrid search combining BM25 keyword matching with semantic search."""

from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import re


class HybridSearchEngine:
    """Combines BM25 keyword search with semantic vector search."""
    
    def __init__(self):
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def index_documents(self, documents: List[Dict]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc['text']) for doc in documents]
        
        if self.tokenized_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase + split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of documents with BM25 scores
        """
        if not self.bm25_index or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return docs with positive scores
                results.append({
                    'text': self.documents[idx]['text'],
                    'metadata': self.documents[idx]['metadata'],
                    'bm25_score': float(scores[idx])
                })
        
        return results
    
    def merge_results(
        self, 
        semantic_results: List[Dict], 
        bm25_results: List[Dict],
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Dict]:
        """
        Merge and rerank results from semantic and BM25 search.
        
        Args:
            semantic_results: Results from vector search (with 'score' key)
            bm25_results: Results from BM25 search (with 'bm25_score' key)
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            
        Returns:
            Merged and reranked results with combined scores
        """
        # Normalize scores to 0-1 range
        def normalize_scores(results, score_key):
            if not results:
                return results
            
            scores = [r[score_key] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                # All scores are the same
                for r in results:
                    r[f'{score_key}_normalized'] = 1.0
            else:
                for r in results:
                    r[f'{score_key}_normalized'] = (r[score_key] - min_score) / (max_score - min_score)
            
            return results
        
        # Normalize both result sets
        semantic_results = normalize_scores(semantic_results, 'score')
        bm25_results = normalize_scores(bm25_results, 'bm25_score')
        
        # Create a map of documents by text (for deduplication)
        merged_map = {}
        
        # Add semantic results
        for result in semantic_results:
            text = result['text']
            merged_map[text] = {
                'text': text,
                'metadata': result['metadata'],
                'semantic_score': result.get('score_normalized', 0.0),
                'bm25_score': 0.0,
                'combined_score': 0.0
            }
        
        # Add/merge BM25 results
        for result in bm25_results:
            text = result['text']
            if text in merged_map:
                # Document found in both - update BM25 score
                merged_map[text]['bm25_score'] = result.get('bm25_score_normalized', 0.0)
            else:
                # Document only in BM25 results
                merged_map[text] = {
                    'text': text,
                    'metadata': result['metadata'],
                    'semantic_score': 0.0,
                    'bm25_score': result.get('bm25_score_normalized', 0.0),
                    'combined_score': 0.0
                }
        
        # Calculate combined scores
        for text, doc in merged_map.items():
            doc['combined_score'] = (
                semantic_weight * doc['semantic_score'] +
                bm25_weight * doc['bm25_score']
            )
        
        # Sort by combined score
        merged_results = sorted(
            merged_map.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return merged_results
    
    def clear(self):
        """Clear the BM25 index."""
        self.bm25_index = None
        self.documents = []
        self.tokenized_corpus = []


# Global instance
hybrid_search = HybridSearchEngine()
