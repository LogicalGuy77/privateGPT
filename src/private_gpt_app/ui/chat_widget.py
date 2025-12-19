"""Chat widget for displaying messages with token streaming support."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from private_gpt_app.ui.message_bubble import MessageBubble


class ChatWidget(QWidget):
    """Widget for displaying chat messages with streaming support."""
    
    message_added = pyqtSignal(str, bool)  # message, is_user
    
    def __init__(self):
        super().__init__()
        self.current_message_bubble = None
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the chat display area."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scroll area for messages
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setObjectName("chatScrollArea")
        
        # Container for messages
        self.messages_container = QWidget()
        self.messages_container.setObjectName("messagesContainer")
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(20, 20, 20, 20)
        self.messages_layout.setSpacing(15)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Welcome message
        self.add_welcome_message()
        
        self.scroll_area.setWidget(self.messages_container)
        layout.addWidget(self.scroll_area)
    
    def add_welcome_message(self):
        """Add initial welcome message."""
        welcome = QLabel(
            "👋 <b>Welcome to Private-GPT</b><br><br>"
            "Ask me anything! Your conversations stay completely private and local."
        )
        welcome.setObjectName("welcomeMessage")
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setWordWrap(True)
        welcome.setTextFormat(Qt.TextFormat.RichText)
        self.messages_layout.addWidget(welcome)
        self.messages_layout.addStretch()
    
    def add_message(self, text: str, is_user: bool):
        """Add a complete message to the chat."""
        # Remove welcome message and stretch if present
        if self.messages_layout.count() > 0:
            first_item = self.messages_layout.itemAt(0)
            if first_item.widget() and first_item.widget().objectName() == "welcomeMessage":
                first_item.widget().deleteLater()
            # Remove stretch
            if self.messages_layout.count() > 0:
                last_item = self.messages_layout.itemAt(self.messages_layout.count() - 1)
                if last_item.spacerItem():
                    self.messages_layout.removeItem(last_item)
        
        # Create message bubble
        bubble = MessageBubble(text, is_user)
        self.messages_layout.addWidget(bubble)
        
        # Scroll to bottom
        self.scroll_to_bottom()
        
        self.message_added.emit(text, is_user)
    
    def start_assistant_message(self):
        """Start a new assistant message for streaming."""
        # Remove stretch if present
        if self.messages_layout.count() > 0:
            last_item = self.messages_layout.itemAt(self.messages_layout.count() - 1)
            if last_item.spacerItem():
                self.messages_layout.removeItem(last_item)
        
        # Remove welcome message if present
        if self.messages_layout.count() > 0:
            first_item = self.messages_layout.itemAt(0)
            if first_item.widget() and first_item.widget().objectName() == "welcomeMessage":
                first_item.widget().deleteLater()
        
        # Create new message bubble
        self.current_message_bubble = MessageBubble("", is_user=False)
        self.messages_layout.addWidget(self.current_message_bubble)
        self.scroll_to_bottom()
    
    def append_to_current_message(self, text_chunk: str):
        """Append text to the currently streaming message."""
        if self.current_message_bubble:
            current_text = self.current_message_bubble.text_label.toPlainText()
            self.current_message_bubble.text_label.setPlainText(current_text + text_chunk)
            self.scroll_to_bottom()
    
    def finish_current_message(self):
        """Finish the current streaming message and render as markdown."""
        if self.current_message_bubble:
            # Convert plain text to markdown rendering
            final_text = self.current_message_bubble.text_label.toPlainText()
            self.current_message_bubble.text_label.setMarkdown(final_text)
            self.current_message_bubble = None
            self.scroll_to_bottom()
    
    def clear_messages(self):
        """Clear all messages from the chat."""
        # Remove all widgets
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Re-add welcome message
        self.add_welcome_message()
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
