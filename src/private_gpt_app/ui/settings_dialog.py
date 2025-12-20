"""Settings dialog for adjusting model parameters."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QComboBox, QPushButton, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal


class SettingsDialog(QDialog):
    """Dialog for adjusting model and performance settings."""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.setup_ui()
        self.load_current_settings()
        
    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # GPU Memory Settings
        gpu_group = self.create_gpu_settings()
        layout.addWidget(gpu_group)
        
        # Generation Settings
        gen_group = self.create_generation_settings()
        layout.addWidget(gen_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reset_btn = QPushButton("Reset to Default")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.apply_btn.setDefault(True)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
        
    def create_gpu_settings(self) -> QGroupBox:
        """Create GPU memory settings group."""
        group = QGroupBox("GPU Memory Settings")
        layout = QFormLayout()
        
        # GPU Memory Utilization
        mem_layout = QVBoxLayout()
        self.memory_slider = QSlider(Qt.Orientation.Horizontal)
        self.memory_slider.setMinimum(40)
        self.memory_slider.setMaximum(85)
        self.memory_slider.setValue(55)
        self.memory_slider.setTickInterval(5)
        self.memory_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.memory_slider.valueChanged.connect(self.update_memory_label)
        
        self.memory_label = QLabel("55%")
        self.memory_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        mem_header = QHBoxLayout()
        mem_header.addWidget(QLabel("GPU Memory Utilization"))
        mem_header.addWidget(self.memory_label)
        
        mem_layout.addLayout(mem_header)
        mem_layout.addWidget(self.memory_slider)
        
        mem_info = QLabel("Lower = more stable, Higher = better performance")
        mem_info.setStyleSheet("color: #888; font-size: 11px;")
        mem_layout.addWidget(mem_info)
        
        layout.addRow(mem_layout)
        
        # Context Window
        ctx_layout = QVBoxLayout()
        self.context_combo = QComboBox()
        self.context_combo.addItems([
            "1536 tokens (4GB VRAM)",
            "2048 tokens (4GB+ VRAM)",
            "3072 tokens (6GB VRAM)",
            "4096 tokens (8GB VRAM)",
            "6144 tokens (10GB+ VRAM)"
        ])
        self.context_combo.setCurrentIndex(1)  # 2048 default
        self.context_combo.currentIndexChanged.connect(self.update_context_info)
        
        ctx_layout.addWidget(QLabel("Context Window"))
        ctx_layout.addWidget(self.context_combo)
        
        self.context_info = QLabel("~1500 words of conversation memory")
        self.context_info.setStyleSheet("color: #888; font-size: 11px;")
        ctx_layout.addWidget(self.context_info)
        
        layout.addRow(ctx_layout)
        
        # CPU Offload
        offload_layout = QVBoxLayout()
        self.offload_slider = QSlider(Qt.Orientation.Horizontal)
        self.offload_slider.setMinimum(0)
        self.offload_slider.setMaximum(40)
        self.offload_slider.setValue(20)
        self.offload_slider.setTickInterval(10)
        self.offload_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.offload_slider.valueChanged.connect(self.update_offload_label)
        
        self.offload_label = QLabel("2.0 GB")
        self.offload_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        offload_header = QHBoxLayout()
        offload_header.addWidget(QLabel("CPU Memory Offload"))
        offload_header.addWidget(self.offload_label)
        
        offload_layout.addLayout(offload_header)
        offload_layout.addWidget(self.offload_slider)
        
        offload_info = QLabel("Offload model data to RAM for extra headroom")
        offload_info.setStyleSheet("color: #888; font-size: 11px;")
        offload_layout.addWidget(offload_info)
        
        layout.addRow(offload_layout)
        
        group.setLayout(layout)
        return group
        
    def create_generation_settings(self) -> QGroupBox:
        """Create text generation settings group."""
        group = QGroupBox("Generation Settings")
        layout = QFormLayout()
        
        # Temperature
        temp_layout = QVBoxLayout()
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(100)
        self.temp_slider.setValue(70)
        self.temp_slider.setTickInterval(10)
        self.temp_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temp_slider.valueChanged.connect(self.update_temp_label)
        
        self.temp_label = QLabel("0.70")
        self.temp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        temp_header = QHBoxLayout()
        temp_header.addWidget(QLabel("Temperature"))
        temp_header.addWidget(self.temp_label)
        
        temp_layout.addLayout(temp_header)
        temp_layout.addWidget(self.temp_slider)
        
        temp_info = QLabel("Lower = focused, Higher = creative")
        temp_info.setStyleSheet("color: #888; font-size: 11px;")
        temp_layout.addWidget(temp_info)
        
        layout.addRow(temp_layout)
        
        # Top P
        topp_layout = QVBoxLayout()
        self.topp_slider = QSlider(Qt.Orientation.Horizontal)
        self.topp_slider.setMinimum(50)
        self.topp_slider.setMaximum(100)
        self.topp_slider.setValue(95)
        self.topp_slider.setTickInterval(5)
        self.topp_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.topp_slider.valueChanged.connect(self.update_topp_label)
        
        self.topp_label = QLabel("0.95")
        self.topp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        topp_header = QHBoxLayout()
        topp_header.addWidget(QLabel("Top P"))
        topp_header.addWidget(self.topp_label)
        
        topp_layout.addLayout(topp_header)
        topp_layout.addWidget(self.topp_slider)
        
        topp_info = QLabel("Nucleus sampling threshold")
        topp_info.setStyleSheet("color: #888; font-size: 11px;")
        topp_layout.addWidget(topp_info)
        
        layout.addRow(topp_layout)
        
        # Max Tokens
        tokens_layout = QVBoxLayout()
        self.tokens_slider = QSlider(Qt.Orientation.Horizontal)
        self.tokens_slider.setMinimum(256)
        self.tokens_slider.setMaximum(4096)
        self.tokens_slider.setValue(2048)
        self.tokens_slider.setTickInterval(512)
        self.tokens_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tokens_slider.valueChanged.connect(self.update_tokens_label)
        
        self.tokens_label = QLabel("2048")
        self.tokens_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        tokens_header = QHBoxLayout()
        tokens_header.addWidget(QLabel("Max Response Tokens"))
        tokens_header.addWidget(self.tokens_label)
        
        tokens_layout.addLayout(tokens_header)
        tokens_layout.addWidget(self.tokens_slider)
        
        tokens_info = QLabel("Maximum length of generated responses")
        tokens_info.setStyleSheet("color: #888; font-size: 11px;")
        tokens_layout.addWidget(tokens_info)
        
        layout.addRow(tokens_layout)
        
        group.setLayout(layout)
        return group
    
    def update_memory_label(self, value: int):
        """Update memory utilization label."""
        self.memory_label.setText(f"{value}%")
        
    def update_offload_label(self, value: int):
        """Update CPU offload label."""
        gb = value / 10.0
        self.offload_label.setText(f"{gb:.1f} GB")
        
    def update_temp_label(self, value: int):
        """Update temperature label."""
        temp = value / 100.0
        self.temp_label.setText(f"{temp:.2f}")
        
    def update_topp_label(self, value: int):
        """Update top_p label."""
        topp = value / 100.0
        self.topp_label.setText(f"{topp:.2f}")
        
    def update_tokens_label(self, value: int):
        """Update max tokens label."""
        self.tokens_label.setText(str(value))
        
    def update_context_info(self, index: int):
        """Update context window info label."""
        context_sizes = {
            0: "~1100 words",
            1: "~1500 words",
            2: "~2300 words",
            3: "~3000 words",
            4: "~4600 words"
        }
        self.context_info.setText(f"{context_sizes.get(index, '')} of conversation memory")
    
    def load_current_settings(self):
        """Load current settings into UI."""
        if not self.current_settings:
            return
            
        # GPU settings
        if "gpu_memory_utilization" in self.current_settings:
            value = int(self.current_settings["gpu_memory_utilization"] * 100)
            self.memory_slider.setValue(value)
            
        if "max_model_len" in self.current_settings:
            context_map = {1536: 0, 2048: 1, 3072: 2, 4096: 3, 6144: 4}
            idx = context_map.get(self.current_settings["max_model_len"], 1)
            self.context_combo.setCurrentIndex(idx)
            
        if "cpu_offload_gb" in self.current_settings:
            value = int(self.current_settings["cpu_offload_gb"] * 10)
            self.offload_slider.setValue(value)
        
        # Generation settings
        if "temperature" in self.current_settings:
            value = int(self.current_settings["temperature"] * 100)
            self.temp_slider.setValue(value)
            
        if "top_p" in self.current_settings:
            value = int(self.current_settings["top_p"] * 100)
            self.topp_slider.setValue(value)
            
        if "max_tokens" in self.current_settings:
            self.tokens_slider.setValue(self.current_settings["max_tokens"])
    
    def get_settings(self) -> dict:
        """Get current settings from UI."""
        context_map = [1536, 2048, 3072, 4096, 6144]
        
        return {
            # GPU settings
            "gpu_memory_utilization": self.memory_slider.value() / 100.0,
            "max_model_len": context_map[self.context_combo.currentIndex()],
            "cpu_offload_gb": self.offload_slider.value() / 10.0,
            
            # Generation settings
            "temperature": self.temp_slider.value() / 100.0,
            "top_p": self.topp_slider.value() / 100.0,
            "max_tokens": self.tokens_slider.value()
        }
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.memory_slider.setValue(55)
        self.context_combo.setCurrentIndex(1)  # 2048
        self.offload_slider.setValue(20)  # 2.0 GB
        self.temp_slider.setValue(70)  # 0.70
        self.topp_slider.setValue(95)  # 0.95
        self.tokens_slider.setValue(2048)
    
    def apply_settings(self):
        """Emit settings and close dialog."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
        self.accept()
