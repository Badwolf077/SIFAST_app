"""
Compact log dock widget with filtering capabilities.
"""

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from ..styles import OUTLINE_BUTTON_STYLE


class LogDock(QWidget):
    """Compact log widget with filtering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_logs = []  # Store all logs for filtering
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Header with controls
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        # Title
        title = QLabel("Logs")
        title.setStyleSheet("font-weight: 600; color: #495057;")
        header_layout.addWidget(title)

        # Filter
        header_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "INFO", "WARNING", "ERROR", "SUCCESS", "DEBUG"])
        self.filter_combo.setMaximumWidth(100)
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        header_layout.addWidget(self.filter_combo)

        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumWidth(60)
        self.clear_btn.setStyleSheet(OUTLINE_BUTTON_STYLE)
        self.clear_btn.clicked.connect(self.clear_logs)
        header_layout.addWidget(self.clear_btn)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    def log(self, message: str, level: str = "INFO"):
        """Add a log message with timestamp and color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Store log entry
        log_entry = {"timestamp": timestamp, "level": level, "message": message}
        self.all_logs.append(log_entry)

        # Apply filter
        current_filter = self.filter_combo.currentText()
        if current_filter == "All" or current_filter == level:
            self._append_log_entry(log_entry)

    def _append_log_entry(self, log_entry: dict):
        """Append a log entry to the text widget."""
        # Define colors for different levels
        colors = {"INFO": "#495057", "WARNING": "#ff6b6b", "ERROR": "#dc3545", "SUCCESS": "#28a745", "DEBUG": "#6c757d"}

        color = colors.get(log_entry["level"], "#495057")

        # Format and append
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Timestamp format
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(QColor("#6c757d"))
        cursor.insertText(f"[{log_entry['timestamp']}] ", timestamp_format)

        # Level format
        level_format = QTextCharFormat()
        level_format.setForeground(QColor(color))
        level_format.setFontWeight(600)
        cursor.insertText(f"[{log_entry['level']}] ", level_format)

        # Message format
        message_format = QTextCharFormat()
        message_format.setForeground(QColor("#212529"))
        cursor.insertText(f"{log_entry['message']}\n", message_format)

        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def apply_filter(self, filter_text: str):
        """Apply filter to show only selected log level."""
        self.log_text.clear()

        for log_entry in self.all_logs:
            if filter_text == "All" or log_entry["level"] == filter_text:
                self._append_log_entry(log_entry)

    def clear_logs(self):
        """Clear all logs."""
        self.all_logs.clear()
        self.log_text.clear()
