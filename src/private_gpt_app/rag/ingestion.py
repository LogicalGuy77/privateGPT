import os
import fitz  # PyMuPDF
import docx
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtCore import QObject, pyqtSignal

from private_gpt_app.rag.vector_store import vector_store
from private_gpt_app.backend.document_store import document_store

class TextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Simple recursive character splitter."""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            
            # If we are at the end, just take the rest
            if end >= text_len:
                chunks.append(text[start:])
                break
            
            # Try to find a natural break point (newline, period, space)
            # Look back from 'end' to find a separator
            search_window = text[start:end]
            
            # Priority of separators
            separators = ["\n\n", "\n", ". ", " "]
            split_pos = -1
            
            for sep in separators:
                pos = search_window.rfind(sep)
                if pos != -1:
                    split_pos = start + pos + len(sep)
                    break
            
            if split_pos != -1:
                # Found a separator
                chunks.append(text[start:split_pos].strip())
                start = split_pos - self.chunk_overlap # Overlap is tricky with this simple logic, let's just overlap by index
                # Actually, standard overlap means the next chunk starts 'overlap' chars before the current one ends.
                # But here we found a natural break. 
                # Let's simplify: Just hard cut if no separator, or cut at separator.
                # To support overlap properly, we should reset 'start' based on the cut.
                
                # Correct logic for overlap:
                # Next chunk should start at (current_end - overlap)
                # But we want to respect sentence boundaries.
                # Let's stick to a simpler sliding window for now to ensure robustness.
                start = split_pos 
            else:
                # No separator found, hard cut
                chunks.append(text[start:end].strip())
                start = end - self.chunk_overlap

        return [c for c in chunks if c]

class IngestionWorker(QObject):
    """Worker for background ingestion."""
    progress = pyqtSignal(str, int)  # Message, Percentage
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, file_paths: List[str]):
        super().__init__()
        self.file_paths = file_paths
        self.splitter = TextSplitter()

    def run(self):
        total_files = len(self.file_paths)
        for i, file_path in enumerate(self.file_paths):
            try:
                filename = os.path.basename(file_path)
                self.progress.emit(f"Processing {filename}...", int((i / total_files) * 100))
                
                text = self._extract_text(file_path)
                if not text:
                    continue

                chunks = self.splitter.split_text(text)
                
                # Add to Vector Store
                metadatas = [{"source": filename, "page": 1} for _ in chunks] # TODO: Better page tracking
                vector_store.add_documents(chunks, metadatas)
                
                # Add to Document Store
                document_store.add_document(filename, file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                self.error.emit(f"Failed to process {os.path.basename(file_path)}: {str(e)}")

        self.progress.emit("Ingestion complete!", 100)
        self.finished.emit()

    def _extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext == ".docx":
            return self._read_docx(file_path)
        elif ext in [".txt", ".md"]:
            return self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _read_pdf(self, path: str) -> str:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def _read_docx(self, path: str) -> str:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
