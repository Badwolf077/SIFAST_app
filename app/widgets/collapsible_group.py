"""
Collapsible group box widget with smooth animations.
"""

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QWidget


class CollapsibleGroupBox(QGroupBox):
    """A QGroupBox that can be collapsed/expanded with animation."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)

        # Custom style for better appearance
        self.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 32px;
                padding: 0 8px;
                background: white;
                color: #495057;
            }
            QGroupBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)

        # Animation setup
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Content widget to hold the layout
        self.content_widget = QWidget()
        self.content_layout = None
        self._collapsed = False

        # Setup main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 16, 8, 8)
        main_layout.addWidget(self.content_widget)
        super().setLayout(main_layout)

        # Connect toggle signal
        self.toggled.connect(self._on_toggled)

    def setLayout(self, layout):
        """Override setLayout to add to content widget."""
        self.content_layout = layout
        self.content_widget.setLayout(layout)

    def _on_toggled(self, checked: bool):
        """Handle toggle with animation."""
        self._collapsed = not checked

        if checked:
            # Expand
            self.content_widget.setVisible(True)
            self.setMaximumHeight(16777215)  # Remove height restriction

            # Calculate the full height
            full_height = self.sizeHint().height()

            self.animation.setStartValue(40)
            self.animation.setEndValue(full_height)
            self.animation.finished.disconnect()  # Disconnect any previous connections

        else:
            # Collapse
            current_height = self.height()

            self.animation.setStartValue(current_height)
            self.animation.setEndValue(40)  # Just enough for title

            # Hide content after animation
            self.animation.finished.disconnect()  # Disconnect any previous connections
            self.animation.finished.connect(lambda: self.content_widget.setVisible(False))

        self.animation.start()

    def paintEvent(self, event):
        """Custom paint event for the chevron indicator."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw chevron indicator
        chevron_rect = QRect(10, 5, 12, 12)

        # Create chevron path
        chevron = QPainterPath()
        if self._collapsed:
            # Right-pointing chevron
            chevron.moveTo(chevron_rect.left() + 3, chevron_rect.top() + 2)
            chevron.lineTo(chevron_rect.left() + 8, chevron_rect.center().y())
            chevron.lineTo(chevron_rect.left() + 3, chevron_rect.bottom() - 2)
        else:
            # Down-pointing chevron
            chevron.moveTo(chevron_rect.left() + 2, chevron_rect.top() + 3)
            chevron.lineTo(chevron_rect.center().x(), chevron_rect.top() + 8)
            chevron.lineTo(chevron_rect.right() - 2, chevron_rect.top() + 3)

        # Draw with proper stroke
        painter.setPen(QColor("#6c757d"))
        painter.setBrush(Qt.NoBrush)
        painter.strokePath(chevron, painter.pen())

    def mousePressEvent(self, event):
        """Handle mouse press to toggle on title area click."""
        # Check if click is in the title area
        title_rect = QRect(0, 0, self.width(), 30)
        if title_rect.contains(event.pos()):
            self.setChecked(not self.isChecked())
        else:
            super().mousePressEvent(event)
