"""Crash recovery and auto-save utilities."""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class CrashRecovery:
    """Manages auto-save and crash recovery for chat sessions."""
    
    def __init__(self, recovery_dir: Optional[Path] = None):
        """
        Initialize crash recovery system.
        
        Args:
            recovery_dir: Directory for recovery files (default: data/crash_recovery)
        """
        if recovery_dir is None:
            recovery_dir = Path(__file__).parent.parent.parent.parent / "data" / "crash_recovery"
        
        self.recovery_dir = recovery_dir
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_file: Optional[Path] = None
        self.auto_save_interval = 30  # seconds
        self.last_save_time = 0
    
    def start_session(self) -> Path:
        """
        Start a new recovery session.
        
        Returns:
            Path to the session recovery file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_file = self.recovery_dir / f"session_{timestamp}.json"
        
        # Initialize empty session
        self._save_session([])
        return self.current_session_file
    
    def save_conversation(self, conversation: List[Dict[str, str]], force: bool = False) -> bool:
        """
        Save conversation to recovery file.
        
        Args:
            conversation: List of message dicts with 'role' and 'content'
            force: Force save even if interval hasn't elapsed
            
        Returns:
            True if saved, False if skipped
        """
        if self.current_session_file is None:
            self.start_session()
        
        current_time = time.time()
        if not force and (current_time - self.last_save_time) < self.auto_save_interval:
            return False
        
        self._save_session(conversation)
        self.last_save_time = current_time
        return True
    
    def _save_session(self, conversation: List[Dict[str, str]]) -> None:
        """Internal method to save session data."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "conversation": conversation,
                "message_count": len(conversation)
            }
            
            with open(self.current_session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️  Failed to save recovery session: {e}")
    
    def load_latest_session(self) -> Optional[List[Dict[str, str]]]:
        """
        Load the most recent recovery session.
        
        Returns:
            Conversation history or None if no recovery files found
        """
        recovery_files = sorted(self.recovery_dir.glob("session_*.json"), reverse=True)
        
        if not recovery_files:
            return None
        
        latest_file = recovery_files[0]
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("conversation", [])
        except Exception as e:
            print(f"⚠️  Failed to load recovery session: {e}")
            return None
    
    def has_recovery_data(self) -> bool:
        """Check if there are any recovery files."""
        recovery_files = list(self.recovery_dir.glob("session_*.json"))
        return len(recovery_files) > 0
    
    def get_recovery_info(self) -> Optional[Dict]:
        """
        Get information about the latest recovery session.
        
        Returns:
            Dict with 'timestamp', 'message_count', 'file_path' or None
        """
        recovery_files = sorted(self.recovery_dir.glob("session_*.json"), reverse=True)
        
        if not recovery_files:
            return None
        
        latest_file = recovery_files[0]
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "timestamp": data.get("timestamp", "Unknown"),
                    "message_count": data.get("message_count", 0),
                    "file_path": str(latest_file)
                }
        except Exception:
            return None
    
    def clear_session(self) -> None:
        """Clear the current session file."""
        if self.current_session_file and self.current_session_file.exists():
            try:
                self.current_session_file.unlink()
            except Exception as e:
                print(f"⚠️  Failed to clear recovery session: {e}")
        
        self.current_session_file = None
    
    def clear_all_recovery_data(self) -> int:
        """
        Delete all recovery files.
        
        Returns:
            Number of files deleted
        """
        recovery_files = list(self.recovery_dir.glob("session_*.json"))
        count = 0
        
        for file in recovery_files:
            try:
                file.unlink()
                count += 1
            except Exception as e:
                print(f"⚠️  Failed to delete {file}: {e}")
        
        return count
    
    def end_session(self) -> None:
        """End the current session and mark it as completed."""
        # Simply clear the session file on clean exit
        self.clear_session()
