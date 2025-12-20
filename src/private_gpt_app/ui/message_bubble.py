"""Message bubble component for chat display - ChatGPT style."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextBrowser, QPushButton, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from datetime import datetime


class MessageBubble(QWidget):
    """A single message row in the chat - ChatGPT style (full width, clean layout)."""
    
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        self.text = text
        self.is_user = is_user
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the message UI."""
        # Main horizontal layout to control alignment
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 0, 8)
        main_layout.setSpacing(0)
        
        # Create the message content widget with responsive width
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Background color for the message bubble
        if self.is_user:
            content_widget.setStyleSheet("background-color: #2563EB; border-radius: 12px; padding: 12px;")
        else:
            content_widget.setStyleSheet("background-color: rgba(30, 30, 30, 0.8); border-radius: 12px; border: 1px solid #2A2A2A; padding: 12px;")
        
        # Align user messages to the right (20% space on left), assistant to the left (20% space on right)
        if self.is_user:
            main_layout.addStretch(20)  # 20% left margin
            main_layout.addWidget(content_widget, 80)  # 80% width
        else:
            main_layout.addWidget(content_widget, 80)  # 80% width
            main_layout.addStretch(20)  # 20% right margin
        
        # Header row: icon + name + spacer + copy button + timestamp
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        
        # Role icon and name
        role_icon = "👤" if self.is_user else "🤖"
        role_text = "You" if self.is_user else "Assistant"
        role_label = QLabel(f"{role_icon} {role_text}")
        role_color = "#FFFFFF" if self.is_user else "#e0e0e0"
        role_label.setStyleSheet(f"""
            color: {role_color}; 
            font-weight: 600; 
            font-size: 13px; 
            background: transparent;
        """)
        header.addWidget(role_label)
        
        header.addStretch()
        
        # Copy button (small, subtle)
        copy_btn = QPushButton("Copy")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_message_text)
        copy_btn.setFixedHeight(24)
        btn_color = "rgba(255, 255, 255, 0.2)" if self.is_user else "rgba(255, 255, 255, 0.12)"
        btn_hover_color = "rgba(255, 255, 255, 0.3)" if self.is_user else "rgba(255, 255, 255, 0.08)"
        btn_text_color = "#e0e0e0" if self.is_user else "#777777"
        btn_hover_text_color = "#FFFFFF" if self.is_user else "#cccccc"
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {btn_color};
                color: {btn_text_color};
                padding: 2px 10px;
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {btn_hover_color};
                color: {btn_hover_text_color};
                border-color: rgba(255, 255, 255, 0.3);
            }}
        """)
        header.addWidget(copy_btn)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        time_label = QLabel(timestamp)
        time_color = "rgba(255, 255, 255, 0.7)" if self.is_user else "#555555"
        time_label.setStyleSheet(f"color: {time_color}; font-size: 11px; background: transparent;")
        header.addWidget(time_label)
        
        layout.addLayout(header)
        
        # Message text
        self.text_label = QTextBrowser()
        self.text_label.setOpenExternalLinks(True)
        self.text_label.setReadOnly(True)
        self.text_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_label.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        # Different text color for user vs assistant
        text_color = "#FFFFFF" if self.is_user else "#e0e0e0"
        self.text_label.setStyleSheet(f"""
            QTextBrowser {{
                background: transparent;
                color: {text_color};
                font-size: 14px;
                padding: 0px;
                border: none;
            }}
        """)
        
        # Set content
        if self.text:
            if self.is_user:
                self.text_label.setPlainText(self.text)
            else:
                self.text_label.setMarkdown(self.text)
        
        # Auto-adjust height
        self.text_label.document().documentLayout().documentSizeChanged.connect(
            self._update_height
        )
        
        layout.addWidget(self.text_label)
        
        # Schedule initial height update
        QTimer.singleShot(10, self._update_height)
    
    def _update_height(self):
        """Update text browser height to fit content."""
        doc = self.text_label.document()
        viewport_width = self.text_label.viewport().width()
        if viewport_width > 0:
            doc.setTextWidth(viewport_width)
        height = int(doc.size().height()) + 8
        self.text_label.setMinimumHeight(max(20, height))
        self.text_label.setMaximumHeight(max(20, height))
    
    def _copy_message_text(self):
        """Copy message text to clipboard."""
        text = self.text_label.toPlainText()
        QApplication.clipboard().setText(text)
