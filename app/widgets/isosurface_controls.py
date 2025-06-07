"""
Compact isosurface control panel.
"""

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from ..styles import BUTTON_STYLE, INPUT_STYLE


class IsosurfaceControls(QWidget):
    """Compact control panel for isosurface parameters."""

    reconstructRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Group box
        group = QGroupBox("Isosurface Controls")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 13px;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding-top: 20px;
                padding-bottom: 10px;
                background: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background: #f8f9fa;
                color: #495057;
            }
        """)

        # Grid layout for compact arrangement
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(10, 5, 10, 10)

        # Apply input styles
        input_style = """
            QDoubleSpinBox, QSpinBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 6px;
                background: white;
                font-size: 12px;
                min-width: 80px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #86b7fe;
            }
            QLabel {
                font-size: 12px;
                color: #495057;
            }
        """

        # Time range controls
        grid_layout.addWidget(QLabel("t_min:"), 0, 0, Qt.AlignRight)
        self.t_min_spinbox = QDoubleSpinBox()
        self.t_min_spinbox.setRange(-1000, 1000)
        self.t_min_spinbox.setValue(-250)
        self.t_min_spinbox.setSuffix(" fs")
        self.t_min_spinbox.setDecimals(0)
        self.t_min_spinbox.setStyleSheet(input_style)
        grid_layout.addWidget(self.t_min_spinbox, 0, 1)

        grid_layout.addWidget(QLabel("t_max:"), 1, 0, Qt.AlignRight)
        self.t_max_spinbox = QDoubleSpinBox()
        self.t_max_spinbox.setRange(-1000, 1000)
        self.t_max_spinbox.setValue(700)
        self.t_max_spinbox.setSuffix(" fs")
        self.t_max_spinbox.setDecimals(0)
        self.t_max_spinbox.setStyleSheet(input_style)
        grid_layout.addWidget(self.t_max_spinbox, 1, 1)

        # Frequency scale
        grid_layout.addWidget(QLabel("Freq Scale:"), 2, 0, Qt.AlignRight)
        self.freq_scale_spinbox = QDoubleSpinBox()
        self.freq_scale_spinbox.setRange(-1.0, 1.0)
        self.freq_scale_spinbox.setValue(0.0)
        self.freq_scale_spinbox.setDecimals(2)
        self.freq_scale_spinbox.setSingleStep(0.01)
        self.freq_scale_spinbox.setStyleSheet(input_style)
        grid_layout.addWidget(self.freq_scale_spinbox, 2, 1)

        # Isovalue
        grid_layout.addWidget(QLabel("Isovalue:"), 3, 0, Qt.AlignRight)
        self.isovalue_spinbox = QDoubleSpinBox()
        self.isovalue_spinbox.setRange(0.01, 1.0)
        self.isovalue_spinbox.setValue(0.05)
        self.isovalue_spinbox.setDecimals(2)
        self.isovalue_spinbox.setSingleStep(0.01)
        self.isovalue_spinbox.setStyleSheet(input_style)
        grid_layout.addWidget(self.isovalue_spinbox, 3, 1)

        # Opacity
        grid_layout.addWidget(QLabel("Opacity:"), 4, 0, Qt.AlignRight)
        self.opacity_spinbox = QDoubleSpinBox()
        self.opacity_spinbox.setRange(0.0, 1.0)
        self.opacity_spinbox.setValue(0.9)
        self.opacity_spinbox.setDecimals(2)
        self.opacity_spinbox.setSingleStep(0.1)
        self.opacity_spinbox.setStyleSheet(input_style)
        grid_layout.addWidget(self.opacity_spinbox, 4, 1)

        # Reconstruct button
        self.reconstruct_btn = QPushButton("Reconstruct")
        self.reconstruct_btn.setStyleSheet(
            BUTTON_STYLE
            + """
            QPushButton {
                margin-top: 8px;
                padding: 8px 16px;
                font-weight: 600;
            }
        """
        )
        self.reconstruct_btn.clicked.connect(self.reconstructRequested.emit)
        grid_layout.addWidget(self.reconstruct_btn, 5, 0, 1, 2)

        group.setLayout(grid_layout)
        layout.addWidget(group)

        # Add stretch to push everything up
        layout.addStretch()

        self.setLayout(layout)

        # Set up tooltips
        self.t_min_spinbox.setToolTip("Minimum time value for isosurface")
        self.t_max_spinbox.setToolTip("Maximum time value for isosurface")
        self.freq_scale_spinbox.setToolTip("Frequency scaling factor")
        self.isovalue_spinbox.setToolTip("Isosurface threshold value")
        self.opacity_spinbox.setToolTip("Surface opacity (0=transparent, 1=opaque)")

    def get_parameters(self) -> dict[str, Any]:
        """Get all control parameters."""
        return {
            "t_min": self.t_min_spinbox.value(),
            "t_max": self.t_max_spinbox.value(),
            "frequency_scale": self.freq_scale_spinbox.value(),
            "isovalue": self.isovalue_spinbox.value(),
            "opacity": self.opacity_spinbox.value(),
        }
