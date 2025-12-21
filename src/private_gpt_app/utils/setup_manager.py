"""
Setup manager for Private-GPT - handles bundled model detection.
For packaged version with bundled model.
"""

import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox


def get_bundled_model_path() -> Path | None:
    """
    Detect bundled model path in packaged executable.
    
    Returns:
        Path to bundled model or None if not found
    """
    # When running from packaged executable (PyInstaller or Nuitka)
    if getattr(sys, 'frozen', False):
        # PyInstaller sets sys._MEIPASS
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller extracts to temp dir, but data files are in _internal
            base_path = Path(sys.executable).parent / "_internal"
            # Also check the _MEIPASS location
            meipass_path = Path(sys._MEIPASS)
            
            # Try both locations
            for bp in [base_path, meipass_path]:
                model_path = bp / "models" / "Qwen2.5-3B-Instruct-AWQ"
                if model_path.exists() and (model_path / "config.json").exists():
                    print(f"✅ Found bundled model at: {model_path}")
                    return model_path
        else:
            # Nuitka standalone
            base_path = Path(sys.executable).parent
            model_path = base_path / "models" / "Qwen2.5-3B-Instruct-AWQ"
            if model_path.exists() and (model_path / "config.json").exists():
                print(f"✅ Found bundled model at: {model_path}")
                return model_path
    else:
        # Development mode
        base_path = Path(__file__).parent.parent.parent.parent
        model_path = base_path / "models" / "Qwen2.5-3B-Instruct-AWQ"
        if model_path.exists() and (model_path / "config.json").exists():
            return model_path
    
    return None


def check_first_time_setup() -> bool:
    """
    Check if setup is needed.
    With bundled model, setup is always complete.
    
    Returns:
        True (setup not needed with bundled model)
    """
    # Check if model is bundled
    bundled = get_bundled_model_path()
    
    if bundled:
        print(f"✅ Using bundled model at: {bundled}")
        return True
    
    # Fallback: check if setup was done before
    setup_marker = Path.home() / ".private-gpt" / ".setup_complete"
    return setup_marker.exists()


def get_model_path() -> str:
    """
    Get model path - prioritizes bundled model.
    
    Returns:
        Path to model (bundled, local config, or HuggingFace)
    """
    # First, check for bundled model
    bundled = get_bundled_model_path()
    if bundled:
        return str(bundled)
    
    # Second, check saved config
    config_file = Path.home() / ".private-gpt" / "model_config.txt"
    if config_file.exists():
        saved_path = config_file.read_text().strip()
        if Path(saved_path).exists():
            return saved_path
    
    # Third, check common locations
    common_paths = [
        Path("models/Qwen2.5-3B-Instruct-AWQ"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-3B-Instruct-AWQ",
        Path.home() / ".private-gpt" / "models" / "Qwen2.5-3B-Instruct-AWQ",
    ]
    
    for path in common_paths:
        if path.exists() and (path / "config.json").exists():
            print(f"✅ Found local model at: {path}")
            return str(path)
    
    # Fallback to HuggingFace (will download on first use)
    print("⚠️  No local model found, will use HuggingFace (requires internet)")
    return "Qwen/Qwen2.5-3B-Instruct-AWQ"


class FirstTimeSetupDialog:
    """
    Setup dialog - not needed with bundled model.
    Kept for compatibility but shows info message.
    """
    
    @staticmethod
    def show_info(parent=None):
        """Show info about bundled model."""
        bundled = get_bundled_model_path()
        
        if bundled:
            QMessageBox.information(
                parent,
                "Model Ready",
                f"<h3>Private-GPT is ready to use!</h3>"
                f"<p>Using bundled model:</p>"
                f"<p><code>{bundled}</code></p>"
                f"<p>No additional setup required.</p>"
            )
        else:
            QMessageBox.warning(
                parent,
                "Model Not Found",
                "<h3>Model not found</h3>"
                "<p>The application will attempt to download the model from HuggingFace.</p>"
                "<p>This requires an internet connection (~2GB download).</p>"
            )
