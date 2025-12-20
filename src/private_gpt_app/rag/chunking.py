"""Advanced text chunking with token-based splitting."""

from typing import List, Dict
from transformers import AutoTokenizer


class TokenBasedChunker:
    """Token-aware text chunker for better LLM context alignment."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct-AWQ"):
        """
        Initialize tokenizer.
        
        Args:
            model_name: HuggingFace model ID for tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer, falling back to character-based: {e}")
            self.tokenizer = None
    
    def chunk_by_tokens(
        self, 
        text: str, 
        chunk_size: int = 256, 
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            
        Returns:
            List of text chunks
        """
        if not self.tokenizer:
            # Fallback to character-based chunking
            return self._chunk_by_chars(text, chunk_size * 4, overlap * 4)
        
        try:
            # Encode text to tokens
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                # Get chunk
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                
                # Decode back to text
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                
                # Move to next chunk with overlap
                start += chunk_size - overlap
            
            return chunks
        
        except Exception as e:
            print(f"⚠️ Token chunking failed, falling back: {e}")
            return self._chunk_by_chars(text, chunk_size * 4, overlap * 4)
    
    def _chunk_by_chars(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback character-based chunking."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '.\n', '! ', '?\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap
        
        return chunks
    
    def chunk_with_metadata(
        self,
        text: str,
        chunk_size: int = 256,
        overlap: int = 50,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Chunk text and return with metadata.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            metadata: Base metadata to attach to all chunks
            
        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        chunks = self.chunk_by_tokens(text, chunk_size, overlap)
        
        base_metadata = metadata or {}
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            result.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return result
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer:
            return len(text) // 4  # Rough estimate
        
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except:
            return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        if not self.tokenizer:
            return text[:max_tokens * 4]
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        except:
            return text[:max_tokens * 4]


# Global instance
token_chunker = TokenBasedChunker()
