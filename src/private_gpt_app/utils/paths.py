"""Path utilities for PyInstaller compatibility."""

import sys
from pathlib import Path


def get_base_path() -> Path:
    """
    Get base path that works in both dev and packaged mode.
    
    Returns:
        Base path for the application
    """
    if getattr(sys, 'frozen', False):
        # Running as packaged executable
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller
            return Path(sys.executable).parent
        else:
            # Nuitka
            return Path(sys.executable).parent
    else:
        # Development mode
        # Assumes this file is in src/private_gpt_app/utils/
        return Path(__file__).parent.parent.parent.parent


def get_resource_path(relative_path: str) -> Path:
    """
    Get path to a resource file (works in packaged and dev mode).
    
    Args:
        relative_path: Path relative to base (e.g., "ui/styles.qss")
    
    Returns:
        Absolute path to the resource
    """
    if getattr(sys, 'frozen', False):
        # Packaged mode - resources are in _internal
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller extracts to _internal
            base = Path(sys.executable).parent / "_internal"
        else:
            # Nuitka
            base = Path(sys.executable).parent
    else:
        # Development mode
        base = Path(__file__).parent.parent.parent.parent / "src" / "private_gpt_app"
    
    return base / relative_path


def get_data_dir() -> Path:
    """
    Get data directory for user data (always writable).
    
    Returns:
        Path to data directory
    """
    if getattr(sys, 'frozen', False):
        # Packaged mode - use user's home directory
        data_dir = Path.home() / ".private-gpt" / "data"
    else:
        # Development mode - use project data directory
        data_dir = get_base_path() / "data"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
