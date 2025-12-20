"""Performance statistics dialog."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from private_gpt_app.utils.performance import perf_monitor


class PerformanceDialog(QDialog):
    """Dialog showing performance metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Performance Statistics")
        self.setMinimumSize(600, 400)
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("📊 Performance Metrics")
        header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        
        # Metrics tab
        metrics_tab = self.create_metrics_tab()
        tabs.addTab(metrics_tab, "Metrics")
        
        # Summary tab
        summary_tab = self.create_summary_tab()
        tabs.addTab(summary_tab, "Summary")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        button_layout.addWidget(refresh_btn)
        
        clear_btn = QPushButton("Clear Metrics")
        clear_btn.clicked.connect(self.clear_metrics)
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_metrics_tab(self) -> QWidget:
        """Create metrics table tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(7)
        self.metrics_table.setHorizontalHeaderLabels([
            "Metric", "Latest", "Avg", "Min", "Max", "P95", "Count"
        ])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.metrics_table)
        
        return widget
    
    def create_summary_tab(self) -> QWidget:
        """Create summary text tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("padding: 15px; font-family: monospace;")
        
        layout.addWidget(self.summary_label)
        
        return widget
    
    def refresh_data(self):
        """Refresh performance data."""
        # Get all summaries
        summaries = perf_monitor.get_all_summaries()
        
        # Update table
        self.metrics_table.setRowCount(len(summaries))
        
        for i, (metric_name, summary) in enumerate(summaries.items()):
            if summary is None:
                continue
            
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric_name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(self.format_time(summary['latest'])))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(self.format_time(summary['avg'])))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(self.format_time(summary['min'])))
            self.metrics_table.setItem(i, 4, QTableWidgetItem(self.format_time(summary['max'])))
            self.metrics_table.setItem(i, 5, QTableWidgetItem(self.format_time(summary['p95'])))
            self.metrics_table.setItem(i, 6, QTableWidgetItem(str(summary['count'])))
        
        # Update summary text
        summary_text = ""
        for metric_name in summaries.keys():
            summary_text += perf_monitor.format_summary(metric_name) + "\n\n"
        
        self.summary_label.setText(summary_text or "No metrics available")
    
    def format_time(self, seconds: float) -> str:
        """Format time value."""
        if seconds is None:
            return "N/A"
        if seconds < 1:
            return f"{seconds * 1000:.1f} ms"
        return f"{seconds:.2f} s"
    
    def clear_metrics(self):
        """Clear all metrics."""
        perf_monitor.clear()
        self.refresh_data()
    
    def showEvent(self, event):
        """Refresh data when dialog is shown."""
        super().showEvent(event)
        self.refresh_data()
