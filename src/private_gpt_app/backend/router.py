"""Advanced query router with hybrid search and smart context truncation.

RAG Context Management:
----------------------
This module provides intelligent RAG (Retrieval Augmented Generation) with three strategies:

1. SMART (Default, Recommended):
   - Uses semantic similarity to decide if documents are relevant
   - Adjustable relevance threshold (default 0.5)
   - Automatically uses RAG when documents match the query well
   - Example: "what is in the cover letter?" → checks similarity with all docs
   
2. ALWAYS:
   - Always uses RAG if any documents exist in knowledge base
   - Good for dedicated document Q&A systems
   - May include irrelevant context for general queries
   
3. EXPLICIT:
   - Only uses RAG when user explicitly mentions files
   - Example: "in report.pdf" or "from the analysis.docx"
   - Most conservative, but requires explicit user intent

Configuration:
-------------
retrieval_service.set_rag_strategy("smart")  # Change strategy
retrieval_service.set_relevance_threshold(0.7)  # Make similarity check stricter
"""

import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
from private_gpt_app.rag.vector_store import vector_store
from private_gpt_app.rag.hybrid_search import hybrid_search
from private_gpt_app.rag.chunking import token_chunker
from private_gpt_app.utils.performance import perf_monitor

# Thread pool for async operations
_executor = ThreadPoolExecutor(max_workers=4)

class QueryRouter:
    """Intelligent router using semantic similarity to decide RAG usage."""
    
    def __init__(self):
        self.relevance_threshold = 0.5  # Minimum similarity score to use RAG
        self.always_rag_patterns = [
            # Only explicit file references require RAG
            r'\b(?:in|from|according to)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+\.(?:pdf|docx|txt|md))\b',
        ]
        self.always_rag_regex = re.compile('|'.join(self.always_rag_patterns), re.IGNORECASE)
    
    def should_use_rag(self, query: str, top_k: int = 3) -> bool:
        """
        Intelligently determine if RAG should be used based on semantic relevance.
        
        Strategy:
        1. If user explicitly mentions a file -> always use RAG
        2. Otherwise, check semantic similarity with knowledge base
        3. If top results have high similarity -> use RAG
        4. If no relevant documents -> direct LLM
        
        Args:
            query: User's query string
            top_k: Number of top results to check for relevance
            
        Returns:
            True if RAG should be used, False for direct LLM
        """
        # 1. Check for explicit file references
        if self.always_rag_regex.search(query):
            return True
        
        # 2. Check if documents exist in knowledge base
        try:
            # Semantic search to find most relevant documents
            results = vector_store.search(query, limit=top_k)
            
            if not results:
                # No documents in knowledge base
                return False
            
            # 3. Check relevance scores
            # Most vector stores return scores between 0-1 or use distance metrics
            # Higher score = more relevant
            top_score = results[0].get('score', 0) if results else 0
            
            # If top result is highly relevant, use RAG
            if top_score >= self.relevance_threshold:
                return True
            
            # For lower scores, be more conservative
            # Only use RAG if multiple results are somewhat relevant
            relevant_count = sum(1 for r in results if r.get('score', 0) >= self.relevance_threshold * 0.8)
            if relevant_count >= 2:
                return True
            
            return False
            
        except Exception as e:
            # If vector store fails, don't use RAG
            print(f"RAG check failed: {e}")
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
    
    # RAG Strategy modes
    RAG_ALWAYS = "always"  # Always use RAG if documents exist
    RAG_SMART = "smart"    # Use semantic similarity scoring
    RAG_EXPLICIT = "explicit"  # Only when user explicitly mentions files
    
    def __init__(self, rag_strategy: str = "always"):
        self.router = QueryRouter()
        self.rag_strategy = rag_strategy
        self.use_hybrid_search = True  # Enable hybrid search by default
        self.max_context_tokens = 1000  # Reduced to leave more room for conversation
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        use_hybrid: bool = None,
        filter_filename: str = None
    ) -> Dict:
        """
        Retrieve relevant context for a query with hybrid search (sync version).
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            use_hybrid: Override hybrid search setting
            filter_filename: Optional filename to filter results to specific document
            
        Returns:
            Dict with 'context' (str), 'sources' (list), and 'used_rag' (bool)
        """
        return self._retrieve_impl(query, top_k, use_hybrid, filter_filename)
    
    async def retrieve_context_async(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = None,
        filter_filename: str = None
    ) -> Dict:
        """
        Async version of retrieve_context for non-blocking RAG.
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            use_hybrid: Override hybrid search setting
            filter_filename: Optional filename to filter results to specific document
            
        Returns:
            Dict with 'context' (str), 'sources' (list), and 'used_rag' (bool)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._retrieve_impl,
            query, top_k, use_hybrid, filter_filename
        )
    
    def _retrieve_impl(
        self,
        query: str,
        top_k: int,
        use_hybrid: bool,
        filter_filename: str
    ) -> Dict:
        """Internal implementation of context retrieval."""
        # Start performance tracking
        perf_monitor.start_timer('rag_retrieval')
        
        # Check if RAG should be used based on strategy
        should_use = False
        
        if self.rag_strategy == self.RAG_ALWAYS:
            # Always use RAG if documents exist
            try:
                test = vector_store.search(query, limit=1)
                should_use = len(test) > 0
            except:
                should_use = False
        elif self.rag_strategy == self.RAG_EXPLICIT:
            # Only explicit file references
            should_use = self.router.always_rag_regex.search(query) is not None
        else:  # RAG_SMART (default)
            # Intelligent semantic similarity based routing
            should_use = self.router.should_use_rag(query)
        
        if not should_use:
            perf_monitor.end_timer('rag_retrieval')
            return {
                'context': '',
                'sources': [],
                'used_rag': False
            }
        
        # Check for filename filter (from UI attachment or query parsing)
        if filter_filename:
            filename = filter_filename
        else:
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
    
    def set_rag_strategy(self, strategy: str):
        """
        Change RAG strategy on the fly.
        
        Args:
            strategy: One of 'always', 'smart', 'explicit'
        """
        valid_strategies = [self.RAG_ALWAYS, self.RAG_SMART, self.RAG_EXPLICIT]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        self.rag_strategy = strategy
        print(f"RAG strategy changed to: {strategy}")
    
    def set_relevance_threshold(self, threshold: float):
        """
        Adjust semantic similarity threshold for smart mode.
        
        Args:
            threshold: Float between 0-1 (higher = more strict)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.router.relevance_threshold = threshold
        print(f"Relevance threshold set to: {threshold}")

# Global instance
retrieval_service = RetrievalService()
