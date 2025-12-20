import sqlite3
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

class DocumentStore:
    def __init__(self, db_path: str = "data/documents.db"):
        self.db_path = os.path.join(os.getcwd(), db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'indexed',
                UNIQUE(filename)
            )
        """)
        # Add file_hash column if it doesn't exist (migration)
        try:
            cursor.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
        conn.close()
    
    def _hash_file(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Warning: Failed to hash file {file_path}: {e}")
            return ""
    
    def is_duplicate(self, file_path: str) -> Optional[str]:
        """
        Check if file is duplicate by hash.
        
        Returns:
            Existing filename if duplicate, None otherwise
        """
        file_hash = self._hash_file(file_path)
        if not file_hash:
            return None
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        conn.close()
        
        return row['filename'] if row else None

    def add_document(self, filename: str, file_path: str, check_duplicate: bool = True):
        """
        Add document to store.
        
        Args:
            filename: Document filename
            file_path: Path to document file
            check_duplicate: Check for duplicate by hash
            
        Returns:
            True if added, False if duplicate
        """
        if check_duplicate:
            duplicate = self.is_duplicate(file_path)
            if duplicate:
                print(f"⚠️ Duplicate document detected: {duplicate}")
                return False
        
        file_hash = self._hash_file(file_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO documents (filename, file_path, file_hash, status) VALUES (?, ?, ?, ?)",
                (filename, file_path, file_hash, 'indexed')
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Update existing if needed, or ignore
            return False
        finally:
            conn.close()

    def remove_document(self, filename: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()

    def get_all_documents(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_document(self, filename: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

document_store = DocumentStore()
