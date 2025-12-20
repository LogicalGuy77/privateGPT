"""Simple query router for RAG vs direct LLM calls."""

import re
from typing import Optional, Dict, List
from private_gpt_app.rag.vector_store import vector_store

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
    """Service for retrieving relevant context from the knowledge base."""
    
    def __init__(self):
        self.router = QueryRouter()
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with 'context' (str), 'sources' (list), and 'used_rag' (bool)
        """
        # Check if RAG should be used
        if not self.router.should_use_rag(query):
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
            results = vector_store.search(
                query, 
                limit=top_k,
                filter_metadata=filter_metadata
            )
            
            if not results:
                return {
                    'context': '',
                    'sources': [],
                    'used_rag': False
                }
            
            # Construct context from results
            context_parts = []
            sources = []
            
            for i, result in enumerate(results, 1):
                text = result['text']
                source = result['metadata'].get('source', 'Unknown')
                score = result['score']
                
                context_parts.append(f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}\n")
                
                if source not in sources:
                    sources.append(source)
            
            context = "\n".join(context_parts)
            
            return {
                'context': context,
                'sources': sources,
                'used_rag': True
            }
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return {
                'context': '',
                'sources': [],
                'used_rag': False
            }

# Global instance
retrieval_service = RetrievalService()
