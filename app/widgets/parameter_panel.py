"""
Parameter panel widget for PyPulse application.
"""

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .parameter_widgets import ProcessingParametersWidget, ScanParametersWidget


class ParameterPanel(QWidget):
    """Panel containing all parameter settings."""

    parametersChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #f8f9fa;
            }
        """)

        # Container widget
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(12, 12, 12, 12)
        container_layout.setSpacing(12)

        # Processing parameters (larger section)
        self.processing_params = ProcessingParametersWidget()
        self.processing_params.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        container_layout.addWidget(self.processing_params, 3)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background: #dee2e6; max-height: 1px;")
        container_layout.addWidget(separator)

        # Scan parameters (smaller section)
        self.scan_params = ScanParametersWidget()
        self.scan_params.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        container_layout.addWidget(self.scan_params, 1)

        # Add stretch to push everything up
        container_layout.addStretch()

        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        layout.addWidget(scroll_area)
        self.setLayout(layout)

        # Connect signals
        self.processing_params.parametersChanged.connect(self._on_parameters_changed)
        self.scan_params.parametersChanged.connect(self._on_parameters_changed)

    def _on_parameters_changed(self):
        """Handle parameter changes from child widgets."""
        self.parametersChanged.emit(self.get_all_parameters())

    def get_all_parameters(self) -> dict[str, Any]:
        """Get all parameters from both sections."""
        params = {}
        params.update(self.processing_params.get_parameters())
        params.update(self.scan_params.get_parameters())
        return params

    def get_processing_parameters(self) -> dict[str, Any]:
        """Get processing parameters only."""
        return self.processing_params.get_parameters()

    def get_scan_parameters(self) -> dict[str, Any]:
        """Get scan parameters only."""
        return self.scan_params.get_parameters()

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        self.processing_params.set_parameters(params)
        self.scan_params.set_parameters(params)
