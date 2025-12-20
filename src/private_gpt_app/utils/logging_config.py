"""Enhanced logging configuration for production."""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    if log_to_file:
        # File handler with rotation
        log_file = log_dir / f"private_gpt_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    # Suppress noisy libraries
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('vllm').setLevel(logging.INFO)
    
    logging.info(f"Logging initialized - Level: {log_level}, File: {log_to_file}")
    
    return root_logger


class PerformanceLogger:
    """Log performance metrics for monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
    
    def log_query(self, query_time: float, tokens: int, used_rag: bool):
        """Log a query performance metric."""
        tokens_per_sec = tokens / query_time if query_time > 0 else 0
        self.logger.info(
            f"Query completed - Time: {query_time:.2f}s, "
            f"Tokens: {tokens}, Speed: {tokens_per_sec:.1f} tok/s, "
            f"RAG: {used_rag}"
        )
    
    def log_rag_retrieval(self, retrieval_time: float, num_results: int):
        """Log RAG retrieval performance."""
        self.logger.info(
            f"RAG retrieval - Time: {retrieval_time:.3f}s, Results: {num_results}"
        )
    
    def log_embedding(self, num_docs: int, time_taken: float):
        """Log embedding performance."""
        docs_per_sec = num_docs / time_taken if time_taken > 0 else 0
        self.logger.info(
            f"Embedding - Docs: {num_docs}, Time: {time_taken:.2f}s, "
            f"Speed: {docs_per_sec:.1f} docs/s"
        )


class ErrorTracker:
    """Track and categorize errors for monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("errors")
        self.error_counts = {}
    
    def log_error(self, error_type: str, error: Exception, context: dict = None):
        """Log an error with context."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(
            f"Error: {error_type} - {str(error)}",
            extra={"context": context or {}},
            exc_info=True
        )
    
    def get_error_stats(self) -> dict:
        """Get error statistics."""
        return self.error_counts.copy()


# Global instances
perf_logger = PerformanceLogger()
error_tracker = ErrorTracker()
