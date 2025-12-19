"""Message bubble component for chat display."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from datetime import datetime


class MessageBubble(QWidget):
    """A single message bubble in the chat."""
    
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        self.text = text
        self.is_user = is_user
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the message bubble UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Message container with alignment
        message_container = QWidget()
        message_layout = QHBoxLayout(message_container)
        message_layout.setContentsMargins(0, 0, 0, 0)
        message_layout.setSpacing(10)
        
        # Add stretch on appropriate side
        if self.is_user:
            message_layout.addStretch()
        
        # Bubble widget
        bubble = self.create_bubble()
        message_layout.addWidget(bubble)
        
        # Add stretch on appropriate side
        if not self.is_user:
            message_layout.addStretch()
        
        layout.addWidget(message_container)
    
    def create_bubble(self) -> QWidget:
        """Create the actual bubble widget."""
        bubble = QWidget()
        bubble.setObjectName("userBubble" if self.is_user else "assistantBubble")
        bubble.setMaximumWidth(700)
        
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(15, 10, 15, 10)
        bubble_layout.setSpacing(5)
        
        # Header (role + timestamp)
        header = QHBoxLayout()
        header.setSpacing(10)
        
        # Role label
        role_icon = "👤" if self.is_user else "🤖"
        role_text = "You" if self.is_user else "Assistant"
        role_label = QLabel(f"{role_icon} <b>{role_text}</b>")
        role_label.setObjectName("roleLabel")
        role_label.setTextFormat(Qt.TextFormat.RichText)
        header.addWidget(role_label)
        
        header.addStretch()
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        time_label = QLabel(timestamp)
        time_label.setObjectName("timestampLabel")
        header.addWidget(time_label)
        
        bubble_layout.addLayout(header)
        
        # Message text (use QTextBrowser for markdown support)
        self.text_label = QTextBrowser()
        self.text_label.setObjectName("messageText")
        self.text_label.setOpenExternalLinks(True)
        self.text_label.setReadOnly(True)
        self.text_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_label.setFrameShape(QTextBrowser.Shape.NoFrame)
        
        # Set content
        if self.text:
            if self.is_user:
                # User messages as plain text
                self.text_label.setPlainText(self.text)
            else:
                # Assistant messages as markdown
                self.text_label.setMarkdown(self.text)
        
        # Auto-adjust height to content
        self.text_label.document().documentLayout().documentSizeChanged.connect(
            lambda size: self.text_label.setFixedHeight(int(size.height()) + 5)
        )
        
        bubble_layout.addWidget(self.text_label)
        
        return bubble
