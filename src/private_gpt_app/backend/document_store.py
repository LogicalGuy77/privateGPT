import sqlite3
import os
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
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'indexed',
                UNIQUE(filename)
            )
        """)
        conn.commit()
        conn.close()

    def add_document(self, filename: str, file_path: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO documents (filename, file_path, status) VALUES (?, ?, ?)",
                (filename, file_path, 'indexed')
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # Update existing if needed, or ignore
            pass
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
