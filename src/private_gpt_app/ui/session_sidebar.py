"""Session sidebar with FTS5 search and session management."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from typing import Optional


class SessionSidebar(QWidget):
    """Sidebar widget for session management."""
    
    session_selected = pyqtSignal(int)  # Emits session_id
    new_session_requested = pyqtSignal()
    session_deleted = pyqtSignal(int)
    
    def __init__(self, session_manager, parent=None):
        super().__init__(parent)
        self.session_manager = session_manager
        self.current_session_id: Optional[int] = None
        self._init_ui()
        self.load_sessions()
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # New Chat button
        self.new_chat_btn = QPushButton("➕ New Chat")
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.clicked.connect(self.new_session_requested.emit)
        layout.addWidget(self.new_chat_btn)
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search sessions...")
        self.search_box.textChanged.connect(self._on_search)
        layout.addWidget(self.search_box)
        
        # Session list
        self.session_list = QListWidget()
        self.session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._show_context_menu)
        self.session_list.itemClicked.connect(self._on_session_clicked)
        layout.addWidget(self.session_list)
    
    def load_sessions(self):
        """Load all sessions into the list."""
        self.session_list.clear()
        sessions = self.session_manager.list_sessions(limit=50)
        
        for session in sessions:
            item = QListWidgetItem(session['title'])
            item.setData(Qt.ItemDataRole.UserRole, session['id'])
            
            # Highlight current session
            if session['id'] == self.current_session_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            
            self.session_list.addItem(item)
    
    def _on_search(self, query: str):
        """Handle search query."""
        self.session_list.clear()
        
        if not query.strip():
            # Empty search - show all sessions
            self.load_sessions()
            return
        
        # FTS5 search
        results = self.session_manager.search_sessions(query, limit=20)
        
        for session in results:
            item = QListWidgetItem(session['title'])
            item.setData(Qt.ItemDataRole.UserRole, session['id'])
            
            if session['id'] == self.current_session_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            
            self.session_list.addItem(item)
    
    def _on_session_clicked(self, item: QListWidgetItem):
        """Handle session selection."""
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id != self.current_session_id:
            self.session_selected.emit(session_id)
    
    def _show_context_menu(self, position):
        """Show context menu for session operations."""
        item = self.session_list.itemAt(position)
        if not item:
            return
        
        session_id = item.data(Qt.ItemDataRole.UserRole)
        
        menu = QMenu(self)
        rename_action = menu.addAction("✏️ Rename")
        delete_action = menu.addAction("🗑️ Delete")
        
        action = menu.exec(self.session_list.mapToGlobal(position))
        
        if action == delete_action:
            self._delete_session(session_id, item.text())
        elif action == rename_action:
            self._rename_session(session_id, item)
    
    def _delete_session(self, session_id: int, title: str):
        """Delete a session with confirmation."""
        reply = QMessageBox.question(
            self,
            "Delete Session",
            f"Are you sure you want to delete '{title}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.session_manager.delete_session(session_id)
            self.session_deleted.emit(session_id)
            self.load_sessions()
    
    def _rename_session(self, session_id: int, item: QListWidgetItem):
        """Rename a session (simple implementation - double-click to edit)."""
        # For now, make item editable
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.session_list.editItem(item)
        
        # Note: In production, you'd connect to itemChanged signal to persist
    
    def set_current_session(self, session_id: int):
        """Update the current session highlight."""
        self.current_session_id = session_id
        self.load_sessions()
    
    def refresh(self):
        """Refresh the session list."""
        query = self.search_box.text()
        if query.strip():
            self._on_search(query)
        else:
            self.load_sessions()
