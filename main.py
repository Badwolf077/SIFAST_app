"""
PyPulse - Spatiotemporal Pulse Characterization
Main entry point for the application.
"""

import os
import sys

# Set OpenGL backend for Qt to avoid graphics pipeline issues
os.environ["QSG_RHI_BACKEND"] = "opengl"

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from app.main_window import PyPulseMainWindow
from app.styles import apply_modern_style


def main():
    """Main entry point for the PyPulse application."""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("PyPulse")
    app.setOrganizationName("PyPulse")

    # Enable high DPI support
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # Apply modern style
    apply_modern_style(app)

    # Create and show main window
    window = PyPulseMainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
