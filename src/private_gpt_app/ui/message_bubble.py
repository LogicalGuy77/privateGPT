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
        layout.setContentsMargins(0, 5, 0, 5)  # Reduced top/bottom margins
        layout.setSpacing(5)
        
        # Message container with alignment
        message_container = QWidget()
        message_layout = QHBoxLayout(message_container)
        message_layout.setContentsMargins(0, 0, 0, 0)
        message_layout.setSpacing(0)  # Remove spacing between stretches
        
        # Add stretch on appropriate side with more weight
        if self.is_user:
            message_layout.addStretch(2)  # More stretch on left for user
        
        # Bubble widget
        bubble = self.create_bubble()
        message_layout.addWidget(bubble)
        
        # Add stretch on appropriate side with more weight
        if not self.is_user:
            message_layout.addStretch(2)  # More stretch on right for assistant
        
        layout.addWidget(message_container)
    
    def create_bubble(self) -> QWidget:
        """Create the actual bubble widget."""
        bubble = QWidget()
        bubble.setObjectName("userBubble" if self.is_user else "assistantBubble")
        bubble.setMaximumWidth(650)  # Reduced from 700
        bubble.setMinimumWidth(200)  # Add minimum width
        
        # Add explicit styling for visibility
        if self.is_user:
            bubble.setStyleSheet("""
                #userBubble {
                    background-color: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 12px;
                }
            """)
        else:
            bubble.setStyleSheet("""
                #assistantBubble {
                    background-color: #1a1a1a;
                    border: 1px solid #2a2a2a;
                    border-radius: 12px;
                }
            """)
        
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
        role_label.setStyleSheet("color: #e0e0e0; font-size: 13px;")
        header.addWidget(role_label)
        
        header.addStretch()
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M")
        time_label = QLabel(timestamp)
        time_label.setObjectName("timestampLabel")
        time_label.setStyleSheet("color: #888888; font-size: 11px;")
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
        # Ensure message text stays visible on dark themes even if the app stylesheet
        # fails to load (Qt will fall back to platform defaults).
        self.text_label.setStyleSheet("background: transparent; color: #e0e0e0;")
        
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
