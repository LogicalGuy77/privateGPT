"""Main entry point for Private-GPT application."""

import sys
import argparse
import asyncio
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import qasync

from private_gpt_app.ui.main_window import MainWindow


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
    style_path = Path(__file__).parent / "ui" / "styles.qss"
    
    if style_path.exists():
        with open(style_path, "r") as f:
            stylesheet = f.read()
            app.setStyleSheet(stylesheet)
        print(f"✓ Loaded stylesheet from {style_path}")
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
    
    # Load stylesheet
    load_stylesheet(app, dev_mode=dev_mode)
    
    # Create and show main window
    window = MainWindow(mock_mode=mock_mode)
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔒 Private-GPT Desktop Application")
    print("=" * 60)
    
    if args.mock:
        print("🎭 Running in MOCK mode (no model loading)")
    if args.dev:
        print("🛠️  Development mode enabled")
    
    print()
    
    try:
        # Create qasync event loop
        app = QApplication(sys.argv)
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        with loop:
            loop.run_until_complete(main_async(mock_mode=args.mock, dev_mode=args.dev))
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
