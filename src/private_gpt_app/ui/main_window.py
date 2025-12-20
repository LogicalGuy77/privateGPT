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
from private_gpt_app.ui.session_sidebar import SessionSidebar
from private_gpt_app.ui.performance_dialog import PerformanceDialog
from private_gpt_app.ui.file_picker_widget import FilePickerWidget
from private_gpt_app.backend.vllm_service import VLLMService, GenerationConfig
from private_gpt_app.backend.router import retrieval_service
from private_gpt_app.backend.session_manager import session_manager, trim_conversation
from private_gpt_app.utils.gpu_monitor import (
    detect_gpu, validate_hardware_requirements, print_gpu_info,
    recommend_settings, get_current_vram_usage
)
from private_gpt_app.utils.crash_recovery import CrashRecovery
from private_gpt_app.utils.performance import perf_monitor


class MainWindow(QMainWindow):
    """Main application window with chat interface."""
    
    def __init__(self, mock_mode: bool = False, model_path: str = None):
        super().__init__()
        self.mock_mode = mock_mode
        self.model_path = model_path or "Qwen/Qwen2.5-3B-Instruct-AWQ"
        self.llm_service = None
        self.conversation_history = []
        self.gpu_info = None
        self.current_session_id = None
        self.rag_enabled = True  # RAG toggle
        self.selected_files = []  # Files selected for current query
        self.current_settings = {
            "gpu_memory_utilization": 0.55,
            "max_model_len": 4096,
            "cpu_offload_gb": 2.0,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "rag_strategy": "always",
            "relevance_threshold": 0.5
        }
        
        # Crash recovery system
        self.crash_recovery = CrashRecovery()
        
        # Start with a new session
        self.current_session_id = session_manager.create_session()
        
        # Check hardware if not in mock mode
        if not mock_mode:
            self.check_hardware()
        
        self.setup_ui()
        self.setup_signals()
        self.start_vram_monitoring()

        # Crash recovery needs the UI (chat widget) to be initialized.
        if not mock_mode:
            self.check_recovery_data()
    
    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Private-GPT")
        self.setMinimumSize(QSize(1200, 800))
        
        # Create menu bar
        self.create_menu_bar()
        
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
    
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_chat_action = file_menu.addAction("New Chat")
        new_chat_action.setShortcut("Ctrl+N")
        new_chat_action.triggered.connect(self.on_new_chat)
        
        file_menu.addSeparator()
        
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        kb_action = tools_menu.addAction("Knowledge Base")
        kb_action.setShortcut("Ctrl+K")
        kb_action.triggered.connect(self.open_knowledge_base)
        
        tools_menu.addSeparator()
        
        perf_action = tools_menu.addAction("Performance Stats")
        perf_action.triggered.connect(self.show_performance_stats)
        
        settings_action = tools_menu.addAction("Settings")
        settings_action.triggered.connect(self.show_settings)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
    
    def create_sidebar(self) -> QWidget:
        """Create the sidebar with session management."""
        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebar")
        sidebar_container.setMinimumWidth(250)
        sidebar_container.setMaximumWidth(400)
        
        layout = QVBoxLayout(sidebar_container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Session sidebar
        self.session_sidebar = SessionSidebar(session_manager)
        self.session_sidebar.session_selected.connect(self.on_session_selected)
        self.session_sidebar.new_session_requested.connect(self.on_new_chat)
        self.session_sidebar.session_deleted.connect(self.on_session_deleted)
        layout.addWidget(self.session_sidebar)
        
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
        
        # RAG Toggle
        self.rag_toggle_btn = QPushButton("📚 RAG: ON")
        self.rag_toggle_btn.setCheckable(True)
        self.rag_toggle_btn.setChecked(True)
        self.rag_toggle_btn.clicked.connect(self.toggle_rag)
        self.rag_toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
            QPushButton:checked { background-color: #00aa44; }
            QPushButton:hover { background-color: #3d3d3d; }
        """)
        layout.addWidget(self.rag_toggle_btn)
        
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
        
        return sidebar_container
    
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
        container.setMaximumHeight(300)  # Increased for file picker
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 10, 20, 20)
        layout.setSpacing(10)
        
        # File picker (hidden by default)
        self.file_picker = FilePickerWidget()
        self.file_picker.files_selected.connect(self.on_files_selected)
        layout.addWidget(self.file_picker)
        
        # Input field
        self.input_field = QTextEdit()
        self.input_field.setObjectName("messageInput")
        self.input_field.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        self.input_field.setMaximumHeight(120)
        layout.addWidget(self.input_field)
        
        # Button row
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        
        # Attach files button
        self.attach_btn = QPushButton("📎 Attach")
        self.attach_btn.setObjectName("attachButton")
        self.attach_btn.setToolTip("Attach files from knowledge base")
        self.attach_btn.clicked.connect(self.toggle_file_picker)
        self.attach_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #E5E5E5;
                border: 1px solid #4A4A4A;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
            }
            QPushButton:pressed {
                background-color: #2A2A2A;
            }
        """)
        button_row.addWidget(self.attach_btn)
        
        # Selected files label
        self.selected_files_label = QLabel("")
        self.selected_files_label.setStyleSheet("color: #2563EB; font-size: 11px;")
        button_row.addWidget(self.selected_files_label)
        
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
        
        # Enable keyboard shortcuts
        self.input_field.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent, QKeySequence
        
        if obj == self.input_field and event.type() == QEvent.Type.KeyPress:
            # Ctrl+Enter to send
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.on_send_message()
                return True
            # Ctrl+N for new chat
            elif event.key() == Qt.Key.Key_N and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.on_new_chat()
                return True
            # Ctrl+K for knowledge base
            elif event.key() == Qt.Key.Key_K and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.open_knowledge_base()
                return True
            # Ctrl+L to clear chat
            elif event.key() == Qt.Key.Key_L and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.on_new_chat()
                return True
        
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
                model_path = self.model_path
                print(f"📦 Loading model: {model_path}")
            
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
    
    def toggle_file_picker(self):
        """Toggle file picker visibility."""
        if self.file_picker.isVisible():
            self.file_picker.hide()
        else:
            self.file_picker.show_files()
    
    def on_files_selected(self, filenames: list):
        """Handle file selection from picker."""
        self.selected_files = filenames
        if filenames:
            count = len(filenames)
            display_names = ", ".join([f.split("/")[-1][:15] for f in filenames[:2]])
            if count > 2:
                display_names += f" +{count-2}"
            self.selected_files_label.setText(f"📎 {display_names}")
            self.attach_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2563EB;
                    color: white;
                    border: 1px solid #1D4ED8;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #1D4ED8;
                }
            """)
        else:
            self.selected_files_label.setText("")
            self.attach_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3A3A3A;
                    color: #E5E5E5;
                    border: 1px solid #4A4A4A;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #4A4A4A;
                }
            """)
    
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
        
        # Retrieve context if RAG should be used and is enabled
        context = ""
        sources = []
        used_rag = False
        
        if self.rag_enabled:
            # If files are selected, filter by those files
            if self.selected_files:
                # Retrieve context for each selected file and combine (async)
                all_contexts = []
                all_sources = set()
                
                # Use asyncio.gather for parallel retrieval
                tasks = [
                    retrieval_service.retrieve_context_async(
                        user_message,
                        filter_filename=filename
                    )
                    for filename in self.selected_files
                ]
                results = await asyncio.gather(*tasks)
                
                for result in results:
                    if result['context']:
                        all_contexts.append(result['context'])
                        all_sources.update(result['sources'])
                
                if all_contexts:
                    context = "\n\n".join(all_contexts)
                    sources = list(all_sources)
                    used_rag = True
            else:
                # Normal RAG retrieval (async for non-blocking)
                retrieval_result = await retrieval_service.retrieve_context_async(user_message)
                context = retrieval_result['context']
                sources = retrieval_result['sources']
                used_rag = retrieval_result['used_rag']
        
        # Clear file selection after use
        if self.selected_files:
            self.selected_files = []
            self.selected_files_label.setText("")
            self.file_picker.clear_selection()
            self.file_picker.hide()
            self.attach_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3A3A3A;
                    color: #E5E5E5;
                    border: 1px solid #4A4A4A;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #4A4A4A;
                }
            """)
        
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
                
                # Trim conversation to prevent context overflow (sliding window)
                trimmed_history = trim_conversation(self.conversation_history, max_messages=10)
                
                # Generate with streaming (pass trimmed messages)
                self.chat_widget.start_assistant_message()
                
                full_response = ""
                config = GenerationConfig(
                    temperature=self.current_settings["temperature"],
                    max_tokens=self.current_settings["max_tokens"],
                    top_p=self.current_settings["top_p"]
                )
                
                # Pass context to LLM if RAG is used
                async for token in self.llm_service.generate_stream(trimmed_history, config, context=context):
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
                
                # Auto-save to session database
                if self.current_session_id:
                    # Auto-generate title from first message if still "New Chat"
                    session = session_manager.get_session(self.current_session_id)
                    if session and (session['title'] == 'New Chat' and len(self.conversation_history) >= 2):
                        title = session_manager.auto_generate_title(self.conversation_history)
                        session_manager.update_session(self.current_session_id, self.conversation_history, title=title)
                        self.session_sidebar.refresh()
                    else:
                        session_manager.update_session(self.current_session_id, self.conversation_history)
                
                # Crash recovery backup
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
        # Create new session in database
        self.current_session_id = session_manager.create_session()
        
        # Clear UI
        self.chat_widget.clear_messages()
        self.input_field.clear()
        self.conversation_history.clear()
        
        # Update sidebar
        self.session_sidebar.set_current_session(self.current_session_id)
        self.session_sidebar.refresh()
        
        # Start new recovery session
        self.crash_recovery.start_session()
        self.status_label.setText("New chat started")
    
    def on_session_selected(self, session_id: int):
        """Handle session selection from sidebar."""
        # Save current session before switching
        if self.current_session_id and self.conversation_history:
            session_manager.update_session(self.current_session_id, self.conversation_history)
        
        # Load selected session
        session = session_manager.get_session(session_id)
        if not session:
            self.status_label.setText("❌ Session not found")
            return
        
        # Update current session
        self.current_session_id = session_id
        self.conversation_history = session['messages']
        
        # Clear and reload chat display
        self.chat_widget.clear_messages()
        for msg in self.conversation_history:
            is_user = (msg["role"] == "user")
            self.chat_widget.add_message(msg["content"], is_user=is_user)
        
        # Update UI
        self.session_sidebar.set_current_session(session_id)
        self.status_label.setText(f"Loaded: {session['title']}")
    
    def on_session_deleted(self, session_id: int):
        """Handle session deletion."""
        # If deleted session was current, start new chat
        if self.current_session_id == session_id:
            self.on_new_chat()
        
        self.status_label.setText("Session deleted")
    
    def toggle_rag(self):
        """Toggle RAG on/off."""
        self.rag_enabled = not self.rag_enabled
        
        if self.rag_enabled:
            self.rag_toggle_btn.setText("📚 RAG: ON")
            self.rag_toggle_btn.setChecked(True)
            self.status_label.setText("RAG enabled")
        else:
            self.rag_toggle_btn.setText("📚 RAG: OFF")
            self.rag_toggle_btn.setChecked(False)
            self.status_label.setText("RAG disabled")
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self, self.current_settings)
        dialog.settings_changed.connect(self.apply_settings)
        dialog.exec()
    
    def apply_settings(self, new_settings: dict):
        """Apply new settings (requires restart for model settings, immediate for RAG)."""
        self.current_settings.update(new_settings)
        
        # Apply RAG settings immediately (no restart needed)
        if "rag_strategy" in new_settings:
            retrieval_service.set_rag_strategy(new_settings["rag_strategy"])
        
        if "relevance_threshold" in new_settings:
            retrieval_service.set_relevance_threshold(new_settings["relevance_threshold"])
        
        # Show restart message for model settings
        QMessageBox.information(
            self,
            "Settings Updated",
            "Settings have been updated.\n\n"
            "RAG settings applied immediately.\n"
            "Model settings will take effect on next chat or restart.",
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
        print("🛑 Shutting down Private-GPT...")
        
        # Unload LLM to free GPU memory
        if self.llm_service and self.llm_service.is_loaded:
            print("🔄 Cleaning up vLLM resources...")
            self.llm_service.unload_model()
        
        # Clean up recovery data on clean exit
        self.crash_recovery.end_session()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("✓ Shutdown complete")
        event.accept()
    
    def open_knowledge_base(self):
        """Open the Knowledge Base management dialog."""
        dialog = KnowledgeBaseDialog(self)
        dialog.exec()
    
    def show_performance_stats(self):
        """Show performance statistics dialog."""
        dialog = PerformanceDialog(self)
        dialog.exec()
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Private-GPT",
            "<h2>Private-GPT</h2>"
            "<p>Version 0.2.0</p>"
            "<p>A local, privacy-focused desktop chat application<br>"
            "powered by Qwen2.5-3B-Instruct-AWQ with vLLM acceleration.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>100% Local & Private</li>"
            "<li>RAG with Qdrant & Hybrid Search</li>"
            "<li>Session Management with FTS5 Search</li>"
            "<li>Optimized for 4GB+ VRAM GPUs</li>"
            "</ul>"
            "<p><b>Developer:</b></p>"
            "<p>Email: <a href='mailto:harshitacademia@gmail.com'>harshitacademia@gmail.com</a><br>"
            "Twitter: <a href='https://x.com/HarshitNay80531'>@HarshitNay80531</a><br>"
            "GitHub: <a href='https://github.com/LogicalGuy77'>LogicalGuy77</a></p>"
            "<p>Licensed under MIT License</p>"
        )
