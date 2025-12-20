"""Performance monitoring for tracking app metrics."""

import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricSnapshot:
    """Single metric measurement."""
    timestamp: float
    value: float
    metadata: Dict = field(default_factory=dict)


class PerformanceMonitor:
    """Track and report performance metrics."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of recent measurements to keep
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self._active_timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self._active_timers[operation] = time.time()
    
    def end_timer(self, operation: str, metadata: Dict = None) -> float:
        """
        End timing and record metric.
        
        Args:
            operation: Operation name
            metadata: Optional metadata
            
        Returns:
            Duration in seconds
        """
        if operation not in self._active_timers:
            return 0.0
        
        start_time = self._active_timers.pop(operation)
        duration = time.time() - start_time
        
        self.record(operation, duration, metadata)
        return duration
    
    def record(self, metric_name: str, value: float, metadata: Dict = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Optional metadata
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
        
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics[metric_name].append(snapshot)
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get average value for a metric.
        
        Args:
            metric_name: Name of the metric
            last_n: Only consider last N measurements
            
        Returns:
            Average value or None if no data
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        measurements = list(self.metrics[metric_name])
        if last_n:
            measurements = measurements[-last_n:]
        
        if not measurements:
            return None
        
        return sum(m.value for m in measurements) / len(measurements)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get most recent value for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        return self.metrics[metric_name][-1].value
    
    def get_min_max(self, metric_name: str) -> Optional[tuple]:
        """Get min and max values for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = [m.value for m in self.metrics[metric_name]]
        return (min(values), max(values))
    
    def get_percentile(self, metric_name: str, percentile: float) -> Optional[float]:
        """Get percentile value for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = sorted([m.value for m in self.metrics[metric_name]])
        if not values:
            return None
        
        index = int(len(values) * percentile / 100)
        index = min(index, len(values) - 1)
        return values[index]
    
    def get_summary(self, metric_name: str) -> Optional[Dict]:
        """
        Get comprehensive summary for a metric.
        
        Returns:
            Dict with avg, min, max, p50, p95, p99
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        return {
            'avg': self.get_average(metric_name),
            'latest': self.get_latest(metric_name),
            'min': self.get_min_max(metric_name)[0],
            'max': self.get_min_max(metric_name)[1],
            'p50': self.get_percentile(metric_name, 50),
            'p95': self.get_percentile(metric_name, 95),
            'p99': self.get_percentile(metric_name, 99),
            'count': len(self.metrics[metric_name])
        }
    
    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all tracked metrics."""
        return {
            metric_name: self.get_summary(metric_name)
            for metric_name in self.metrics.keys()
        }
    
    def clear(self, metric_name: Optional[str] = None):
        """
        Clear metrics.
        
        Args:
            metric_name: Specific metric to clear, or None for all
        """
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
        else:
            self.metrics.clear()
    
    def format_summary(self, metric_name: str) -> str:
        """Format metric summary as human-readable string."""
        summary = self.get_summary(metric_name)
        if not summary:
            return f"{metric_name}: No data"
        
        # Convert to milliseconds for readability
        def fmt(val):
            if val is None:
                return "N/A"
            if val < 1:
                return f"{val * 1000:.1f}ms"
            return f"{val:.2f}s"
        
        return (
            f"{metric_name}:\n"
            f"  Latest: {fmt(summary['latest'])}\n"
            f"  Avg: {fmt(summary['avg'])}\n"
            f"  Min/Max: {fmt(summary['min'])}/{fmt(summary['max'])}\n"
            f"  P50/P95/P99: {fmt(summary['p50'])}/{fmt(summary['p95'])}/{fmt(summary['p99'])}\n"
            f"  Samples: {summary['count']}"
        )


# Global instance
perf_monitor = PerformanceMonitor(window_size=100)
