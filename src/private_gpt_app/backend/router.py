"""Advanced query router with hybrid search and smart context truncation."""

import re
from typing import Optional, Dict, List
from private_gpt_app.rag.vector_store import vector_store
from private_gpt_app.rag.hybrid_search import hybrid_search
from private_gpt_app.rag.chunking import token_chunker
from private_gpt_app.utils.performance import perf_monitor

class QueryRouter:
    """Lightweight router to decide between RAG and direct LLM inference."""
    
    # Keywords that suggest the user wants to query documents
    RAG_KEYWORDS = [
        r'\bdocument\b', r'\bfile\b', r'\bpdf\b', r'\bpaper\b', r'\breport\b',
        r'\bin the\b.*\b(document|file|pdf|text)\b',
        r'\baccording to\b', r'\bbased on\b', r'\bfrom the\b',
        r'\bwhat does.*say\b', r'\bsummarize\b', r'\bsummary\b',
        r'\bfind\b.*\bin\b', r'\bsearch\b.*\bfor\b',
        r'\bknowledge base\b', r'\buploaded\b', r'\bfiles?\b'
    ]
    
    def __init__(self):
        self.rag_pattern = re.compile('|'.join(self.RAG_KEYWORDS), re.IGNORECASE)
    
    def should_use_rag(self, query: str) -> bool:
        """
        Determine if RAG should be used for this query.
        
        Args:
            query: User's query string
            
        Returns:
            True if RAG should be used, False for direct LLM
        """
        # Check if query matches RAG keywords
        if self.rag_pattern.search(query):
            return True
        
        # Check if there are any documents in the knowledge base
        # If no documents, always use direct LLM
        try:
            # Do a quick test search - if it returns nothing, no documents exist
            test_results = vector_store.search("test", limit=1)
            if not test_results:
                return False
        except Exception:
            return False
        
        return False
    
    def extract_filename_filter(self, query: str) -> Optional[str]:
        """
        Extract filename if user asks about a specific file.
        
        Examples:
            "in report.pdf" -> "report.pdf"
            "from the analysis.docx" -> "analysis.docx"
        """
        patterns = [
            r'in\s+([a-zA-Z0-9_\-]+\.(pdf|docx|txt|md))',
            r'from\s+(?:the\s+)?([a-zA-Z0-9_\-]+\.(pdf|docx|txt|md))',
            r'according to\s+([a-zA-Z0-9_\-]+\.(pdf|docx|txt|md))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

class RetrievalService:
    """Service for retrieving relevant context with hybrid search and smart truncation."""
    
    def __init__(self):
        self.router = QueryRouter()
        self.use_hybrid_search = True  # Enable hybrid search by default
        self.max_context_tokens = 1500  # Maximum tokens for context
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        use_hybrid: bool = None
    ) -> Dict:
        """
        Retrieve relevant context for a query with hybrid search.
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            use_hybrid: Override hybrid search setting
            
        Returns:
            Dict with 'context' (str), 'sources' (list), and 'used_rag' (bool)
        """
        # Start performance tracking
        perf_monitor.start_timer('rag_retrieval')
        
        # Check if RAG should be used
        if not self.router.should_use_rag(query):
            perf_monitor.end_timer('rag_retrieval')
            return {
                'context': '',
                'sources': [],
                'used_rag': False
            }
        
        # Check for filename filter
        filename = self.router.extract_filename_filter(query)
        filter_metadata = {'source': filename} if filename else None
        
        # Search vector store
        try:
            # Get semantic search results
            perf_monitor.start_timer('semantic_search')
            semantic_results = vector_store.search(
                query, 
                limit=top_k * 2,  # Get more for hybrid reranking
                filter_metadata=filter_metadata
            )
            perf_monitor.end_timer('semantic_search')
            
            if not semantic_results:
                perf_monitor.end_timer('rag_retrieval')
                return {
                    'context': '',
                    'sources': [],
                    'used_rag': False
                }
            
            # Optionally use hybrid search
            should_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid_search
            
            if should_hybrid and len(semantic_results) > 0:
                perf_monitor.start_timer('hybrid_rerank')
                
                # Build BM25 index from results
                documents = [
                    {'text': r['text'], 'metadata': r['metadata']}
                    for r in semantic_results
                ]
                hybrid_search.index_documents(documents)
                
                # Get BM25 results
                bm25_results = hybrid_search.search_bm25(query, top_k=top_k * 2)
                
                # Merge and rerank
                merged_results = hybrid_search.merge_results(
                    semantic_results,
                    bm25_results,
                    semantic_weight=0.6,
                    bm25_weight=0.4
                )
                
                # Take top-k from merged results
                final_results = merged_results[:top_k]
                perf_monitor.end_timer('hybrid_rerank')
            else:
                # Use semantic results only
                final_results = semantic_results[:top_k]
            
            # Smart context construction with token truncation
            perf_monitor.start_timer('context_construction')
            context, sources = self._construct_smart_context(final_results, query)
            perf_monitor.end_timer('context_construction')
            
            perf_monitor.end_timer('rag_retrieval')
            
            return {
                'context': context,
                'sources': sources,
                'used_rag': True
            }
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            perf_monitor.end_timer('rag_retrieval')
            return {
                'context': '',
                'sources': [],
                'used_rag': False
            }
    
    def _construct_smart_context(self, results: List[Dict], query: str) -> tuple:
        """
        Construct context with smart token-based truncation.
        
        Args:
            results: Search results with scores
            query: Original query
            
        Returns:
            Tuple of (context_str, sources_list)
        """
        context_parts = []
        sources = []
        total_tokens = 0
        
        # Reserve tokens for formatting
        formatting_tokens = 50
        available_tokens = self.max_context_tokens - formatting_tokens
        
        for i, result in enumerate(results, 1):
            text = result['text']
            source = result['metadata'].get('source', 'Unknown')
            score = result.get('combined_score') or result.get('score', 0.0)
            
            # Count tokens for this chunk
            chunk_tokens = token_chunker.count_tokens(text)
            
            # Check if adding this chunk would exceed limit
            if total_tokens + chunk_tokens > available_tokens:
                # Try to truncate the chunk to fit
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 50:  # Only include if at least 50 tokens
                    text = token_chunker.truncate_to_tokens(text, remaining_tokens)
                    chunk_tokens = remaining_tokens
                else:
                    break  # Stop adding chunks
            
            # Add to context
            context_parts.append(f"[{i}] {text}")
            total_tokens += chunk_tokens
            
            if source not in sources:
                sources.append(source)
            
            # Stop if we've reached token limit
            if total_tokens >= available_tokens:
                break
        
        # Format final context
        if context_parts:
            context = "Relevant information:\n\n" + "\n\n".join(context_parts)
        else:
            context = ""
        
        return context, sources

# Global instance
retrieval_service = RetrievalService()
