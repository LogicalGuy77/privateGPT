"""File picker widget for attaching knowledge base documents to queries."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from private_gpt_app.backend.document_store import document_store


class FilePickerWidget(QWidget):
    """Compact file picker for attaching documents to queries."""
    
    files_selected = pyqtSignal(list)  # Emits list of selected filenames
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setVisible(False)  # Hidden by default
        
    def setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("📎 Attach files from Knowledge Base")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(title)
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #888;
                font-size: 16px;
            }
            QPushButton:hover {
                color: #fff;
            }
        """)
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn)
        
        layout.addLayout(header)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.file_list.setMaximumHeight(150)
        self.file_list.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.file_list)
        
        # Info label
        self.info_label = QLabel("Select files to focus the query on specific documents")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.info_label)
        
        # Style
        self.setStyleSheet("""
            QWidget {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 8px;
            }
            QListWidget {
                background-color: #1A1A1A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                color: #E5E5E5;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: #3A3A3A;
            }
            QListWidget::item:selected {
                background-color: #2563EB;
                color: white;
            }
        """)
    
    def show_files(self):
        """Load and show available documents."""
        self.file_list.clear()
        docs = document_store.get_all_documents()
        
        if not docs:
            item = QListWidgetItem("No documents in knowledge base")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.file_list.addItem(item)
            self.info_label.setText("Add documents via Knowledge Base menu")
        else:
            for doc in docs:
                item = QListWidgetItem(f"📄 {doc['filename']}")
                item.setData(Qt.ItemDataRole.UserRole, doc['filename'])
                self.file_list.addItem(item)
            self.info_label.setText(f"{len(docs)} documents available - select to attach")
        
        self.setVisible(True)
    
    def on_selection_changed(self):
        """Emit selected files."""
        selected_files = []
        for item in self.file_list.selectedItems():
            filename = item.data(Qt.ItemDataRole.UserRole)
            if filename:
                selected_files.append(filename)
        
        self.files_selected.emit(selected_files)
    
    def clear_selection(self):
        """Clear all selections."""
        self.file_list.clearSelection()
    
    def get_selected_files(self) -> list:
        """Get currently selected filenames."""
        selected = []
        for item in self.file_list.selectedItems():
            filename = item.data(Qt.ItemDataRole.UserRole)
            if filename:
                selected.append(filename)
        return selected
