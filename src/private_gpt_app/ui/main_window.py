"""Main window for Private-GPT application."""

import asyncio
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSplitter, QListWidget, QMessageBox
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QFont

from private_gpt_app.ui.chat_widget import ChatWidget
from private_gpt_app.ui.settings_dialog import SettingsDialog
from private_gpt_app.ui.knowledge_base_dialog import KnowledgeBaseDialog
from private_gpt_app.backend.vllm_service import VLLMService, GenerationConfig
from private_gpt_app.backend.router import retrieval_service
from private_gpt_app.utils.gpu_monitor import (
    detect_gpu, validate_hardware_requirements, print_gpu_info,
    recommend_settings, get_current_vram_usage
)
from private_gpt_app.utils.crash_recovery import CrashRecovery


class MainWindow(QMainWindow):
    """Main application window with chat interface."""
    
    def __init__(self, mock_mode: bool = False):
        super().__init__()
        self.mock_mode = mock_mode
        self.llm_service = None
        self.conversation_history = []
        self.gpu_info = None
        self.current_settings = {
            "gpu_memory_utilization": 0.55,
            "max_model_len": 2048,
            "cpu_offload_gb": 2.0,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048
        }
        
        # Crash recovery system
        self.crash_recovery = CrashRecovery()
        
        # Check hardware if not in mock mode
        if not mock_mode:
            self.check_hardware()
            self.check_recovery_data()
        
        self.setup_ui()
        self.setup_signals()
        self.start_vram_monitoring()
    
    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Private-GPT")
        self.setMinimumSize(QSize(1200, 800))
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for sidebar and chat area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Sidebar (left panel)
        sidebar = self.create_sidebar()
        splitter.addWidget(sidebar)
        
        # Chat area (right panel)
        chat_container = self.create_chat_area()
        splitter.addWidget(chat_container)
        
        # Set initial splitter sizes (25% sidebar, 75% chat)
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def create_sidebar(self) -> QWidget:
        """Create the sidebar with session list."""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(400)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header_label = QLabel("Chat History")
        header_label.setObjectName("sidebarHeader")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        header_label.setFont(font)
        layout.addWidget(header_label)
        
        # Knowledge Base Button
        kb_btn = QPushButton("📚 Knowledge Base")
        kb_btn.clicked.connect(self.open_knowledge_base)
        kb_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3d3d3d; }
        """)
        layout.addWidget(kb_btn)

        header_label.setFont(font)
        layout.addWidget(header_label)
        
        # New Chat button
        self.new_chat_btn = QPushButton("+ New Chat")
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.setMinimumHeight(40)
        layout.addWidget(self.new_chat_btn)
        
        # Session list
        self.session_list = QListWidget()
        self.session_list.setObjectName("sessionList")
        layout.addWidget(self.session_list)
        
        # Add placeholder sessions for demo
        self.session_list.addItems([
            "Today - New Chat",
            "Yesterday - Project Discussion",
            "Dec 18 - Code Review",
            "Dec 17 - Documentation",
        ])
        
        # VRAM Monitor
        self.vram_label = QLabel("VRAM: --")
        self.vram_label.setObjectName("vramLabel")
        self.vram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vram_label.setStyleSheet("color: #4CAF50; font-size: 11px; padding: 5px;")
        layout.addWidget(self.vram_label)
        
        # Settings button
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setObjectName("settingsButton")
        settings_btn.clicked.connect(self.show_settings)
        layout.addWidget(settings_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        return sidebar
    
    def create_chat_area(self) -> QWidget:
        """Create the main chat area."""
        container = QWidget()
        container.setObjectName("chatContainer")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Chat display area
        self.chat_widget = ChatWidget()
        layout.addWidget(self.chat_widget, stretch=1)
        
        # Input area
        input_container = self.create_input_area()
        layout.addWidget(input_container)
        
        return container
    
    def create_input_area(self) -> QWidget:
        """Create the message input area."""
        container = QWidget()
        container.setObjectName("inputContainer")
        container.setMinimumHeight(100)
        container.setMaximumHeight(200)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 10, 20, 20)
        layout.setSpacing(10)
        
        # Input field
        self.input_field = QTextEdit()
        self.input_field.setObjectName("messageInput")
        self.input_field.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        self.input_field.setMaximumHeight(120)
        layout.addWidget(self.input_field)
        
        # Button row
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        
        # Mode indicator
        mode_text = "🎭 Mock Mode" if self.mock_mode else "🤖 LLM Ready"
        self.mode_label = QLabel(mode_text)
        self.mode_label.setObjectName("modeLabel")
        button_row.addWidget(self.mode_label)
        
        # RAG status indicator
        self.rag_label = QLabel("")
        self.rag_label.setObjectName("ragLabel")
        self.rag_label.setStyleSheet("color: #00ff88; font-size: 11px;")
        button_row.addWidget(self.rag_label)
        
        button_row.addStretch()
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearButton")
        button_row.addWidget(self.clear_btn)
        
        # Send button
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendButton")
        self.send_btn.setMinimumWidth(100)
        button_row.addWidget(self.send_btn)
        
        layout.addLayout(button_row)
        
        return container
    
    def setup_signals(self):
        """Connect signals and slots."""
        self.send_btn.clicked.connect(self.on_send_message)
        self.clear_btn.clicked.connect(self.on_clear_input)
        self.new_chat_btn.clicked.connect(self.on_new_chat)
        
        # Enable Ctrl+Enter to send
        self.input_field.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        return super().eventFilter(obj, event)
    
    def check_hardware(self):
        """Check hardware requirements and show warnings if needed."""
        gpu_info = detect_gpu()
        print_gpu_info(gpu_info)
        
        is_valid, message = validate_hardware_requirements(gpu_info, min_vram_gb=6.0)
        
        if not is_valid:
            reply = QMessageBox.warning(
                self,
                "Hardware Check",
                message + "\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                import sys
                sys.exit(0)
        else:
            print(message)
    
    async def initialize_llm(self):
        """Initialize the LLM service with Qwen2.5-3B-Instruct-AWQ."""
        if self.llm_service and self.llm_service.is_loaded:
            return
        
        # Show loading status
        self.status_label.setText("🔄 Loading Qwen2.5-3B...")
        self.send_btn.setEnabled(False)
        
        try:
            # Initialize VLLMService with Qwen2.5-3B-Instruct-AWQ
            # Apache 2.0 license - fully commercial, AWQ 4-bit quantized (~2.7GB)
            models_dir = Path(__file__).parent.parent.parent.parent / "models" / "Qwen2.5-3B-Instruct-AWQ"
            if models_dir.exists():
                model_path = str(models_dir)
                print(f"📁 Loading model from local folder: {model_path}")
            else:
                model_path = "Qwen/Qwen2.5-3B-Instruct-AWQ"
                print(f"☁️ Loading model from HuggingFace: {model_path}")
            
            self.llm_service = VLLMService(
                model_name=model_path,
                quantization="awq_marlin",
                gpu_memory_utilization=self.current_settings["gpu_memory_utilization"],
                max_model_len=self.current_settings["max_model_len"],
                cpu_offload_gb=self.current_settings["cpu_offload_gb"],
                verbose=False
            )
            
            await self.llm_service.load_model(
                progress_callback=lambda msg: self.status_label.setText(msg)
            )
            
            self.status_label.setText("✅ Model ready")
            self.send_btn.setEnabled(True)
            self.mode_label.setText("🤖 Qwen2.5-3B")
        
        except Exception as e:
            QMessageBox.critical(
                self,
                "Model Load Failed",
                f"Failed to load Qwen2.5-3B-Instruct-AWQ:\n\n{str(e)}\n\n"
                "Make sure you have:\n"
                "• CUDA-capable GPU with 6GB+ VRAM\n"
                "• Python packages: vllm, torch\n\n"
                "Install with: uv pip install vllm"
            )
            self.status_label.setText("❌ Model load failed")
            self.send_btn.setEnabled(True)
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_send_message(self):
        """Handle send button click."""
        message = self.input_field.toPlainText().strip()
        if not message:
            return
        
        # Add user message to chat
        self.chat_widget.add_message(message, is_user=True)
        self.conversation_history.append({"role": "user", "content": message})
        self.input_field.clear()
        
        # Generate response (async)
        asyncio.create_task(self.generate_response(message))
    
    async def generate_response(self, user_message: str):
        """Generate and display response."""
        self.send_btn.setEnabled(False)
        self.status_label.setText("⏳ Generating...")
        self.rag_label.setText("")
        
        # Retrieve context if RAG should be used
        retrieval_result = retrieval_service.retrieve_context(user_message)
        context = retrieval_result['context']
        sources = retrieval_result['sources']
        used_rag = retrieval_result['used_rag']
        
        # Update RAG indicator
        if used_rag and sources:
            self.rag_label.setText(f"📚 Using: {', '.join(sources[:3])}")
        
        if self.mock_mode:
            # Simulate thinking time
            await asyncio.sleep(0.5)
            
            # Mock response
            response = (
                f"**Mock Response** to: \"{user_message[:50]}...\"\n\n"
                "This is a mock response for testing the UI. "
                "In production, this will be replaced with actual LLM output.\n\n"
                "Features:\n"
                "- Token streaming\n"
                "- Markdown rendering\n"
                "- Code syntax highlighting"
            )
            
            # Simulate token streaming
            self.chat_widget.start_assistant_message()
            for i in range(0, len(response), 10):
                chunk = response[i:i+10]
                self.chat_widget.append_to_current_message(chunk)
                await asyncio.sleep(0.05)
            self.chat_widget.finish_current_message()
        
        else:
            # Real LLM inference with Nemotron Nano 9B v2
            try:
                # Initialize LLM if not loaded
                if self.llm_service is None or not self.llm_service.is_loaded:
                    await self.initialize_llm()
                    
                    if self.llm_service is None:
                        # User canceled or model not found
                        self.send_btn.setEnabled(True)
                        self.status_label.setText("Ready")
                        return
                
                # Generate with streaming (pass messages directly)
                self.chat_widget.start_assistant_message()
                
                full_response = ""
                config = GenerationConfig(
                    temperature=self.current_settings["temperature"],
                    max_tokens=self.current_settings["max_tokens"],
                    top_p=self.current_settings["top_p"]
                )
                
                # Pass context to LLM if RAG is used
                async for token in self.llm_service.generate_stream(self.conversation_history, config, context=context):
                    self.chat_widget.append_to_current_message(token)
                    full_response += token
                    await asyncio.sleep(0)  # Allow UI updates
                
                # Add sources citation if RAG was used
                if used_rag and sources:
                    citation = f"\n\n---\n*Sources: {', '.join(sources)}*"
                    self.chat_widget.append_to_current_message(citation)
                    full_response += citation
                
                self.chat_widget.finish_current_message()
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Auto-save conversation
                self.crash_recovery.save_conversation(self.conversation_history)
                
                print(f"✓ Generated {len(full_response)} characters")
            
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                print(error_msg)
                self.chat_widget.add_message(
                    f"**Error:** {str(e)}\n\nCheck console for details.",
                    is_user=False
                )
        
        self.send_btn.setEnabled(True)
        self.status_label.setText("Ready")
    
    def on_clear_input(self):
        """Clear the input field."""
        self.input_field.clear()
        self.input_field.setFocus()
    
    def on_new_chat(self):
        """Start a new chat session."""
        self.chat_widget.clear_messages()
        self.input_field.clear()
        self.conversation_history.clear()  # Reset conversation history
        self.crash_recovery.start_session()  # Start new recovery session
        self.status_label.setText("New chat started")
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self, self.current_settings)
        dialog.settings_changed.connect(self.apply_settings)
        dialog.exec()
    
    def apply_settings(self, new_settings: dict):
        """Apply new settings (requires restart)."""
        self.current_settings.update(new_settings)
        
        # Show restart message
        QMessageBox.information(
            self,
            "Settings Updated",
            "Settings have been updated.\n\n"
            "Please start a new chat or restart the application\n"
            "for changes to take effect.",
            QMessageBox.StandardButton.Ok
        )
    
    def start_vram_monitoring(self):
        """Start periodic VRAM monitoring."""
        if self.mock_mode:
            return
        
        # Create timer for VRAM updates
        self.vram_timer = QTimer(self)
        self.vram_timer.timeout.connect(self.update_vram_display)
        self.vram_timer.start(2000)  # Update every 2 seconds
    
    def update_vram_display(self):
        """Update VRAM usage display."""
        usage = get_current_vram_usage()
        
        if usage:
            used = usage["used_gb"]
            total = usage["total_gb"]
            pct = usage["utilization_pct"]
            
            # Color code based on usage
            if pct < 60:
                color = "#4CAF50"  # Green
            elif pct < 80:
                color = "#FF9800"  # Orange
            else:
                color = "#F44336"  # Red
            
            self.vram_label.setText(f"VRAM: {used:.1f}/{total:.1f} GB ({pct:.0f}%)")
            self.vram_label.setStyleSheet(f"color: {color}; font-size: 11px; padding: 5px;")
    
    def check_recovery_data(self):
        """Check for crash recovery data."""
        if not self.crash_recovery.has_recovery_data():
            self.crash_recovery.start_session()
            return
        
        info = self.crash_recovery.get_recovery_info()
        if not info:
            self.crash_recovery.start_session()
            return
        
        # Ask user if they want to restore
        reply = QMessageBox.question(
            self,
            "Restore Previous Session?",
            f"Found unsaved session from {info['timestamp']}\n"
            f"Messages: {info['message_count']}\n\n"
            "Would you like to restore it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            conversation = self.crash_recovery.load_latest_session()
            if conversation:
                self.conversation_history = conversation
                
                # Display recovered messages
                for msg in conversation:
                    is_user = (msg["role"] == "user")
                    self.chat_widget.add_message(msg["content"], is_user=is_user)
                
                self.status_label.setText(f"✓ Restored {len(conversation)} messages")
        else:
            self.crash_recovery.clear_all_recovery_data()
        
        # Start new session
        self.crash_recovery.start_session()
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up recovery data on clean exit
        self.crash_recovery.end_session()
        event.accept()
    
    def open_knowledge_base(self):
        """Open the Knowledge Base management dialog."""
        dialog = KnowledgeBaseDialog(self)
        dialog.exec()
