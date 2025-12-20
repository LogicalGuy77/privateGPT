"""Session manager with SQLite WAL mode and FTS5 search."""

import sqlite3
import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from functools import lru_cache

class SessionManager:
    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = os.path.join(os.getcwd(), db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with optimizations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for 3x faster writes
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                messages TEXT NOT NULL DEFAULT '[]'
            )
        """)
        
        # Create FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                title, content, content_rowid=id
            )
        """)
        
        # Create triggers to keep FTS5 in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
                INSERT INTO sessions_fts(rowid, title, content)
                VALUES (new.id, new.title, new.messages);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
                DELETE FROM sessions_fts WHERE rowid = old.id;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
                UPDATE sessions_fts SET title=new.title, content=new.messages
                WHERE rowid = new.id;
            END
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, title: str = "New Chat") -> int:
        """Create a new session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (title, messages) VALUES (?, ?)",
            (title, "[]")
        )
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id
    
    def get_session(self, session_id: int) -> Optional[Dict]:
        """Get a session by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'messages': json.loads(row['messages'])
            }
        return None
    
    @lru_cache(maxsize=5)
    def get_session_cached(self, session_id: int) -> Optional[Dict]:
        """Get session with LRU caching (keeps last 5 sessions in RAM)."""
        return self.get_session(session_id)
    
    def update_session(self, session_id: int, messages: List[Dict], title: Optional[str] = None):
        """Update session messages and optionally title."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if title:
            cursor.execute(
                "UPDATE sessions SET messages = ?, title = ?, updated_at = ? WHERE id = ?",
                (json.dumps(messages), title, datetime.now().isoformat(), session_id)
            )
        else:
            cursor.execute(
                "UPDATE sessions SET messages = ?, updated_at = ? WHERE id = ?",
                (json.dumps(messages), datetime.now().isoformat(), session_id)
            )
        
        conn.commit()
        conn.close()
        
        # Invalidate cache
        self.get_session_cached.cache_clear()
    
    def delete_session(self, session_id: int):
        """Delete a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        
        # Invalidate cache
        self.get_session_cached.cache_clear()
    
    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """List all sessions ordered by update time."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_sessions(self, query: str, limit: int = 20) -> List[Dict]:
        """Search sessions using FTS5."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # FTS5 query
        cursor.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at
            FROM sessions_fts fts
            JOIN sessions s ON fts.rowid = s.id
            WHERE sessions_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def auto_generate_title(self, messages: List[Dict]) -> str:
        """Auto-generate session title from first user message."""
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Take first 50 chars
                title = content[:50].strip()
                if len(content) > 50:
                    title += "..."
                return title or "New Chat"
        return "New Chat"

def trim_conversation(history: List[Dict], max_messages: int = 10) -> List[Dict]:
    """
    Implement sliding window to prevent context overflow.
    Keeps system prompt + last N messages.
    """
    if len(history) <= max_messages:
        return history
    
    # Find system prompt
    system_prompt = None
    if history and history[0].get('role') == 'system':
        system_prompt = history[0]
        history = history[1:]
    
    # Keep last N messages
    trimmed = history[-max_messages:]
    
    # Re-add system prompt
    if system_prompt:
        trimmed.insert(0, system_prompt)
    
    return trimmed

# Global instance
session_manager = SessionManager()
