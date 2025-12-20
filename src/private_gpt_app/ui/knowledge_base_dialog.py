from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
    QLabel, QFileDialog, QProgressBar, QMessageBox, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread
from private_gpt_app.backend.document_store import document_store
from private_gpt_app.rag.vector_store import vector_store
from private_gpt_app.rag.ingestion import IngestionWorker

class KnowledgeBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Knowledge Base Management")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        self.load_documents()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Manage Your Documents")
        header.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Document List
        self.doc_list = QListWidget()
        layout.addWidget(self.doc_list)

        # Progress Bar (Hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add Files...")
        self.add_btn.clicked.connect(self.add_files)
        btn_layout.addWidget(self.add_btn)

        self.del_btn = QPushButton("Delete Selected")
        self.del_btn.clicked.connect(self.delete_file)
        btn_layout.addWidget(self.del_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        # Apply Styles
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QListWidget { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3d3d3d; }
            QPushButton { background-color: #007acc; color: white; padding: 5px 15px; border-radius: 4px; }
            QPushButton:hover { background-color: #0098ff; }
            QLabel { color: #ffffff; }
        """)

    def load_documents(self):
        self.doc_list.clear()
        docs = document_store.get_all_documents()
        for doc in docs:
            item = QListWidgetItem(f"{doc['filename']} ({doc['upload_date']})")
            item.setData(Qt.ItemDataRole.UserRole, doc['filename'])
            self.doc_list.addItem(item)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents", "", "Documents (*.pdf *.docx *.txt *.md)"
        )
        if not files:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.add_btn.setEnabled(False)
        self.del_btn.setEnabled(False)

        # Start Worker Thread
        self.thread = QThread()
        self.worker = IngestionWorker(files)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.ingestion_finished)
        self.worker.error.connect(self.ingestion_error)
        
        # Clean up
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def update_progress(self, msg, val):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"{msg} %p%")

    def ingestion_finished(self):
        self.progress_bar.setVisible(False)
        self.add_btn.setEnabled(True)
        self.del_btn.setEnabled(True)
        self.load_documents()
        QMessageBox.information(self, "Success", "Documents added successfully!")

    def ingestion_error(self, msg):
        self.progress_bar.setVisible(False)
        self.add_btn.setEnabled(True)
        self.del_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def delete_file(self):
        item = self.doc_list.currentItem()
        if not item:
            return

        filename = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to delete '{filename}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Delete from Vector Store
            vector_store.delete_document(filename)
            # Delete from Document Store
            document_store.remove_document(filename)
            # Refresh UI
            self.load_documents()
