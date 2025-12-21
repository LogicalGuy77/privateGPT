"""Main entry point for Private-GPT application."""

import sys
import argparse
import asyncio
from pathlib import Path
import os
import torch
import multiprocessing

# CRITICAL: Prevent worker processes from running GUI code
# This environment variable is NOT set in worker processes
_IS_MAIN_PROCESS = os.environ.get('PRIVATE_GPT_MAIN_PROCESS') == '1'

# Set for child processes to detect
if __name__ == "__main__":
    os.environ['PRIVATE_GPT_MAIN_PROCESS'] = '1'
    _IS_MAIN_PROCESS = True
    multiprocessing.set_start_method('spawn', force=True)

# Only import GUI modules in main process
if _IS_MAIN_PROCESS:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    import qasync

    from private_gpt_app.ui.main_window import MainWindow
    from private_gpt_app.utils.setup_manager import check_first_time_setup, FirstTimeSetupDialog, get_model_path
    from private_gpt_app.utils.paths import get_resource_path


    def setup_app() -> QApplication:
        """Initialize QApplication with proper settings."""
        # Enable high DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        
        app = QApplication(sys.argv)
        app.setApplicationName("Private-GPT")
        app.setOrganizationName("Private-GPT")
        app.setApplicationVersion("0.1.0")
        
        return app


    def load_stylesheet(app: QApplication, dev_mode: bool = False) -> None:
    """Load and apply QSS stylesheet."""
    # Try modern stylesheet first
    style_path = get_resource_path("ui/styles_modern.qss")
    
    if not style_path.exists():
        # Fallback to original stylesheet
        style_path = get_resource_path("ui/styles.qss")
    
    if style_path.exists():
        with open(style_path, "r") as f:
            stylesheet = f.read()
            app.setStyleSheet(stylesheet)
        print(f"✓ Loaded stylesheet: {style_path.name}")
    else:
        print(f"⚠ Stylesheet not found at {style_path}")
    
    # Setup hot-reload in dev mode
    if dev_mode:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class StylesheetReloader(FileSystemEventHandler):
                def __init__(self, app: QApplication, path: Path):
                    self.app = app
                    self.path = path
                
                def on_modified(self, event):
                    if event.src_path == str(self.path):
                        with open(self.path, "r") as f:
                            self.app.setStyleSheet(f.read())
                        print("🔄 Reloaded stylesheet")
            
            observer = Observer()
            observer.schedule(
                StylesheetReloader(app, style_path),
                str(style_path.parent),
                recursive=False
            )
            observer.start()
            print("🔥 Hot-reload enabled for QSS")
        except ImportError:
            print("⚠ watchdog not installed - hot-reload disabled")


async def main_async(mock_mode: bool = False, dev_mode: bool = False) -> None:
    """Async main function for the application."""
    app = qasync.QApplication.instance()
    
    # With bundled model, setup is automatic - just show info on first run
    if not mock_mode and not check_first_time_setup():
        # Show info about bundled model (non-blocking)
        FirstTimeSetupDialog.show_info()
    
    # Load stylesheet
    load_stylesheet(app, dev_mode=dev_mode)
    
    # Pre-initialize embedding service BEFORE creating MainWindow
    # This prevents it from being lazily initialized in vLLM worker processes
    if not mock_mode:
        from private_gpt_app.rag.embeddings import get_embedding_service
        print("🔧 Pre-loading embedding model...")
        get_embedding_service()  # Force initialization in main process
    
    # Get model path (bundled or default)
    model_path = get_model_path() if not mock_mode else None
    
    # Create and show main window
    window = MainWindow(mock_mode=mock_mode, model_path=model_path)
    window.show()
    
    # Wait for window to close
    close_event = asyncio.Event()
    
    def on_close():
        close_event.set()
    
    window.destroyed.connect(on_close)
    
    await close_event.wait()


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Private-GPT Desktop Application")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no model loading, fake responses)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development features (QSS hot-reload, debug logging)"
    )
    
    # Parse only known args to ignore multiprocessing arguments from vLLM
    args, unknown = parser.parse_known_args()
    
    print("=" * 60)
    print("🔒 Private-GPT Desktop Application")
    print("=" * 60)
    
    if args.mock:
        print("🎭 Running in MOCK mode (no model loading)")
    if args.dev:
        print("🛠️  Development mode enabled")
    
    print()
    
    # Clean up any stale processes from previous crashes
    from private_gpt_app.utils.gpu_cleanup import kill_stale_vllm_processes, cleanup_gpu_memory
    if kill_stale_vllm_processes():
        import time
        time.sleep(1)  # Wait for processes to die
        cleanup_gpu_memory()
    
    try:
        # Create qasync event loop
        app = QApplication(sys.argv)
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        with loop:
            loop.run_until_complete(main_async(mock_mode=args.mock, dev_mode=args.dev))
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        # Ensure cleanup on Ctrl+C
        from private_gpt_app.utils.gpu_cleanup import cleanup_gpu_memory
        cleanup_gpu_memory()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def debug_cuda_env():
    """Debug CUDA environment information."""
    print("\n🔍 Debugging CUDA Environment:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
    print(f"PATH: {os.environ.get('PATH', 'Not Set')}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
    else:
        print(f"✅ CUDA Device: {torch.cuda.get_device_name(0)}")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    # Only run main in the primary process, not in vLLM workers
    if _IS_MAIN_PROCESS:
        debug_cuda_env()
        main()
    else:
        # Worker process - do nothing
        pass
