"""
Modern style definitions for PyPulse application.
"""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_modern_style(app: QApplication):
    """Apply modern, elegant style to the application."""

    # Set Fusion style as base
    app.setStyle("Fusion")

    # Define color palette
    palette = QPalette()

    # Window colors
    palette.setColor(QPalette.Window, QColor(248, 249, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))

    # Base colors
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(248, 249, 250))

    # Text colors
    palette.setColor(QPalette.Text, QColor(33, 37, 41))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))

    # Button colors
    palette.setColor(QPalette.Button, QColor(248, 249, 250))
    palette.setColor(QPalette.ButtonText, QColor(33, 37, 41))

    # Highlight colors
    palette.setColor(QPalette.Highlight, QColor(13, 110, 253))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    # Link colors
    palette.setColor(QPalette.Link, QColor(13, 110, 253))
    palette.setColor(QPalette.LinkVisited, QColor(85, 46, 132))

    app.setPalette(palette)

    # Global stylesheet
    app.setStyleSheet("""
        /* Global font */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Tooltips */
        QToolTip {
            background-color: #212529;
            color: white;
            border: 1px solid #495057;
            border-radius: 4px;
            padding: 5px;
            font-size: 12px;
        }

        /* Scroll bars */
        QScrollBar:vertical {
            background: #f8f9fa;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background: #adb5bd;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background: #6c757d;
        }

        QScrollBar:horizontal {
            background: #f8f9fa;
            height: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:horizontal {
            background: #adb5bd;
            border-radius: 6px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background: #6c757d;
        }

        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }

        /* Splitter handles */
        QSplitter::handle {
            background: #dee2e6;
        }

        QSplitter::handle:hover {
            background: #adb5bd;
        }

        QSplitter::handle:horizontal {
            width: 5px;
        }

        QSplitter::handle:vertical {
            height: 5px;
        }

        /* Menu bar */
        QMenuBar {
            background: white;
            border-bottom: 1px solid #dee2e6;
        }

        QMenuBar::item {
            padding: 5px 10px;
        }

        QMenuBar::item:selected {
            background: #e9ecef;
            border-radius: 4px;
        }

        /* Status bar */
        QStatusBar {
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            font-size: 12px;
        }

        /* Tab widget */
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 4px;
        }

        QTabBar::tab {
            background: #f8f9fa;
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #dee2e6;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background: white;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }

        QTabBar::tab:hover:!selected {
            background: #e9ecef;
        }

        /* Progress bars */
        QProgressBar {
            border: 1px solid #dee2e6;
            border-radius: 4px;
            text-align: center;
            background: #f8f9fa;
            height: 20px;
        }

        QProgressBar::chunk {
            background: #0d6efd;
            border-radius: 3px;
        }
    """)


# Button styles
BUTTON_STYLE = """
    QPushButton {
        background-color: #0d6efd;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: 500;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #0b5ed7;
    }
    QPushButton:pressed {
        background-color: #0a58ca;
    }
    QPushButton:disabled {
        background-color: #6c757d;
        color: #dee2e6;
    }
"""

SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #6c757d;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: 500;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #5c636a;
    }
    QPushButton:pressed {
        background-color: #565e64;
    }
"""

OUTLINE_BUTTON_STYLE = """
    QPushButton {
        background-color: transparent;
        color: #0d6efd;
        border: 1px solid #0d6efd;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: 500;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #0d6efd;
        color: white;
    }
    QPushButton:pressed {
        background-color: #0b5ed7;
        color: white;
    }
"""

# Input styles
INPUT_STYLE = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 6px 8px;
        background: white;
        font-size: 13px;
    }

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border-color: #86b7fe;
        outline: none;
    }

    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
        background: #e9ecef;
        color: #6c757d;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid #6c757d;
        width: 0;
        height: 0;
        margin-right: 4px;
    }
"""

# Group box styles
GROUP_BOX_STYLE = """
    QGroupBox {
        font-weight: 600;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 12px;
        background: white;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 8px;
        background: white;
        color: #495057;
    }
"""

# Collapsible group box styles
COLLAPSIBLE_GROUP_STYLE = """
    QGroupBox {
        font-weight: 600;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 8px;
        background: white;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 4px 8px;
        background: white;
        color: #495057;
    }

    QGroupBox::indicator {
        width: 12px;
        height: 12px;
        margin-right: 4px;
    }

    QGroupBox::indicator:unchecked {
        image: none;
        border: 2px solid #6c757d;
        border-left: none;
        border-top: none;
        width: 6px;
        height: 6px;
        transform: rotate(-45deg);
    }

    QGroupBox::indicator:checked {
        image: none;
        border: 2px solid #6c757d;
        border-right: none;
        border-top: none;
        width: 6px;
        height: 6px;
        transform: rotate(45deg);
    }
"""
