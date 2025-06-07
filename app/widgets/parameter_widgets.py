"""
Parameter input widgets for PyPulse application.
"""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..styles import INPUT_STYLE, OUTLINE_BUTTON_STYLE
from .collapsible_group import CollapsibleGroupBox


class ProcessingParametersWidget(QWidget):
    """Widget for processing parameters."""

    parametersChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setup_tooltips()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Apply input styles
        self.setStyleSheet(INPUT_STYLE)

        # Acquisition Settings
        acq_group = CollapsibleGroupBox("Acquisition Settings")
        acq_layout = QGridLayout()
        acq_layout.setSpacing(8)

        acq_layout.addWidget(QLabel("Mode Acquire:"), 0, 0)
        self.mode_acquire = QComboBox()
        self.mode_acquire.addItems(["single", "double", "triple"])
        self.mode_acquire.setCurrentIndex(2)  # Default to triple
        acq_layout.addWidget(self.mode_acquire, 0, 1)

        acq_layout.addWidget(QLabel("Gate Noise:"), 1, 0)
        self.gate_noise = QDoubleSpinBox()
        self.gate_noise.setRange(0, 1000)
        self.gate_noise.setValue(200.0)
        self.gate_noise.setSuffix(" a.u.")
        self.gate_noise.setDecimals(1)
        acq_layout.addWidget(self.gate_noise, 1, 1)

        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

        # Wavelength Settings
        wave_group = CollapsibleGroupBox("Wavelength Settings")
        wave_layout = QGridLayout()
        wave_layout.setSpacing(8)

        wave_layout.addWidget(QLabel("Center:"), 0, 0)
        self.wavelength_center = QDoubleSpinBox()
        self.wavelength_center.setRange(200, 2000)
        self.wavelength_center.setValue(793.0)
        self.wavelength_center.setSuffix(" nm")
        self.wavelength_center.setDecimals(1)
        wave_layout.addWidget(self.wavelength_center, 0, 1)

        wave_layout.addWidget(QLabel("Width:"), 1, 0)
        self.wavelength_width = QDoubleSpinBox()
        self.wavelength_width.setRange(10, 500)
        self.wavelength_width.setValue(100.0)
        self.wavelength_width.setSuffix(" nm")
        self.wavelength_width.setDecimals(1)
        wave_layout.addWidget(self.wavelength_width, 1, 1)

        wave_group.setLayout(wave_layout)
        layout.addWidget(wave_group)

        # FFT Settings
        fft_group = CollapsibleGroupBox("FFT Settings")
        fft_layout = QGridLayout()
        fft_layout.setSpacing(8)

        fft_layout.addWidget(QLabel("n_omega:"), 0, 0)
        self.n_omega = QSpinBox()
        self.n_omega.setRange(256, 8192)
        self.n_omega.setSingleStep(256)
        self.n_omega.setValue(2048)
        fft_layout.addWidget(self.n_omega, 0, 1)

        fft_layout.addWidget(QLabel("n_fft:"), 1, 0)
        self.n_fft = QSpinBox()
        self.n_fft.setRange(1024, 131072)
        self.n_fft.setSingleStep(1024)
        self.n_fft.setValue(65536)
        fft_layout.addWidget(self.n_fft, 1, 1)

        fft_group.setLayout(fft_layout)
        layout.addWidget(fft_group)

        # Processing Settings
        proc_group = CollapsibleGroupBox("Processing Settings")
        proc_layout = QGridLayout()
        proc_layout.setSpacing(8)

        proc_layout.addWidget(QLabel("Fiber Position:"), 0, 0)
        self.mode_fiber_position = QComboBox()
        self.mode_fiber_position.addItems(["calibration", "calculation"])
        proc_layout.addWidget(self.mode_fiber_position, 0, 1)

        proc_layout.addWidget(QLabel("Method:"), 1, 0)
        self.method = QComboBox()
        self.method.addItems(["linear", "slinear", "quadratic", "cubic"])
        proc_layout.addWidget(self.method, 1, 1)

        proc_layout.addWidget(QLabel("Delay Min:"), 2, 0)
        self.delay_min = QDoubleSpinBox()
        self.delay_min.setRange(-10000, 10000)
        self.delay_min.setValue(3000)
        self.delay_min.setSpecialValueText("Auto")
        self.delay_min.setSuffix(" fs")
        proc_layout.addWidget(self.delay_min, 2, 1)

        self.as_calibration = QCheckBox("As Calibration")
        proc_layout.addWidget(self.as_calibration, 3, 0, 1, 2)

        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)

        # Fiber Array Settings
        fiber_group = CollapsibleGroupBox("Fiber Array Settings")
        fiber_layout = QGridLayout()
        fiber_layout.setSpacing(8)

        fiber_layout.addWidget(QLabel("Array ID:"), 0, 0)
        self.fiber_array_id = QComboBox()
        self.fiber_array_id.addItems(["default_14x14", "Fiber_array_14x14_1.1", "custom"])
        fiber_layout.addWidget(self.fiber_array_id, 0, 1)

        fiber_layout.addWidget(QLabel("Config Path:"), 1, 0)
        config_layout = QHBoxLayout()
        config_layout.setSpacing(4)
        self.config_folder_path = QLineEdit()
        self.config_folder_path.setPlaceholderText("Optional config path...")
        self.browse_config_btn = QPushButton("Browse")
        self.browse_config_btn.setStyleSheet(OUTLINE_BUTTON_STYLE)
        self.browse_config_btn.clicked.connect(self.browse_config_folder)
        config_layout.addWidget(self.config_folder_path)
        config_layout.addWidget(self.browse_config_btn)
        fiber_layout.addLayout(config_layout, 1, 1)

        fiber_group.setLayout(fiber_layout)
        layout.addWidget(fiber_group)

        self.setLayout(layout)

        # Connect signals
        self.connect_signals()

    def setup_tooltips(self):
        """Set up helpful tooltips."""
        self.mode_acquire.setToolTip("Acquisition mode for spectral data")
        self.gate_noise.setToolTip("Noise threshold for fiber detection")
        self.wavelength_center.setToolTip("Center wavelength of the pulse")
        self.wavelength_width.setToolTip("Spectral width to analyze")
        self.n_omega.setToolTip("Number of frequency points")
        self.n_fft.setToolTip("FFT size for processing")
        self.mode_fiber_position.setToolTip("Method for fiber position detection")
        self.method.setToolTip("Interpolation method for resampling")
        self.delay_min.setToolTip("Minimum delay for peak detection (set to 0 for auto)")
        self.as_calibration.setToolTip("Use this measurement for calibration")
        self.fiber_array_id.setToolTip("Fiber array configuration")
        self.config_folder_path.setToolTip("Optional custom configuration folder")

    def connect_signals(self):
        """Connect all parameter change signals."""
        widgets = [
            self.mode_acquire,
            self.gate_noise,
            self.wavelength_center,
            self.wavelength_width,
            self.n_omega,
            self.n_fft,
            self.mode_fiber_position,
            self.method,
            self.delay_min,
            self.as_calibration,
            self.fiber_array_id,
            self.config_folder_path,
        ]

        for widget in widgets:
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.on_parameter_changed)
            elif isinstance(widget, QSpinBox | QDoubleSpinBox):
                widget.valueChanged.connect(self.on_parameter_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.on_parameter_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.on_parameter_changed)

    def on_parameter_changed(self):
        """Emit signal when any parameter changes."""
        self.parametersChanged.emit()

    def browse_config_folder(self):
        """Open file dialog to select config folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Config Folder", str(Path.cwd() / "config" / "device"))
        if folder:
            self.config_folder_path.setText(folder)

    def get_parameters(self) -> dict[str, Any]:
        """Get all parameters as dictionary."""
        params = {
            "mode_acquire": self.mode_acquire.currentText(),
            "gate_noise_intensity": self.gate_noise.value(),
            "wavelength_center": self.wavelength_center.value(),
            "wavelength_width": self.wavelength_width.value(),
            "n_omega": self.n_omega.value(),
            "n_fft": self.n_fft.value(),
            "mode_fiber_position": self.mode_fiber_position.currentText(),
            "method": self.method.currentText(),
            "fiber_array_id": self.fiber_array_id.currentText(),
            "as_calibration": self.as_calibration.isChecked(),
        }

        if self.delay_min.value() != 0:
            params["delay_min"] = self.delay_min.value()

        if self.config_folder_path.text():
            params["config_folder_path"] = self.config_folder_path.text()

        return params

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        if "mode_acquire" in params:
            self.mode_acquire.setCurrentText(params["mode_acquire"])
        if "gate_noise_intensity" in params:
            self.gate_noise.setValue(params["gate_noise_intensity"])
        if "wavelength_center" in params:
            self.wavelength_center.setValue(params["wavelength_center"])
        if "wavelength_width" in params:
            self.wavelength_width.setValue(params["wavelength_width"])
        if "n_omega" in params:
            self.n_omega.setValue(params["n_omega"])
        if "n_fft" in params:
            self.n_fft.setValue(params["n_fft"])
        if "mode_fiber_position" in params:
            self.mode_fiber_position.setCurrentText(params["mode_fiber_position"])
        if "method" in params:
            self.method.setCurrentText(params["method"])
        if "delay_min" in params:
            self.delay_min.setValue(params["delay_min"])
        if "fiber_array_id" in params:
            self.fiber_array_id.setCurrentText(params["fiber_array_id"])
        if "as_calibration" in params:
            self.as_calibration.setChecked(params["as_calibration"])
        if "config_folder_path" in params:
            self.config_folder_path.setText(params["config_folder_path"])


class ScanParametersWidget(QWidget):
    """Widget for scan parameters."""

    parametersChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setup_tooltips()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Apply input styles
        self.setStyleSheet(INPUT_STYLE)

        # Position Settings
        pos_group = QGroupBox("Position Settings")
        pos_group.setStyleSheet("QGroupBox { font-weight: 600; }")
        pos_layout = QGridLayout()
        pos_layout.setSpacing(8)

        pos_layout.addWidget(QLabel("X Offset:"), 0, 0)
        self.x_offset = QDoubleSpinBox()
        self.x_offset.setRange(-100, 100)
        self.x_offset.setValue(0.0)
        self.x_offset.setSuffix(" mm")
        self.x_offset.setDecimals(2)
        pos_layout.addWidget(self.x_offset, 0, 1)

        pos_layout.addWidget(QLabel("Y Offset:"), 1, 0)
        self.y_offset = QDoubleSpinBox()
        self.y_offset.setRange(-100, 100)
        self.y_offset.setValue(0.0)
        self.y_offset.setSuffix(" mm")
        self.y_offset.setDecimals(2)
        pos_layout.addWidget(self.y_offset, 1, 1)

        pos_layout.addWidget(QLabel("X Position:"), 2, 0)
        self.x_position = QDoubleSpinBox()
        self.x_position.setRange(-100, 100)
        self.x_position.setValue(0.0)
        self.x_position.setSuffix(" mm")
        self.x_position.setDecimals(2)
        pos_layout.addWidget(self.x_position, 2, 1)

        pos_layout.addWidget(QLabel("Y Position:"), 3, 0)
        self.y_position = QDoubleSpinBox()
        self.y_position.setRange(-100, 100)
        self.y_position.setValue(0.0)
        self.y_position.setSuffix(" mm")
        self.y_position.setDecimals(2)
        pos_layout.addWidget(self.y_position, 3, 1)

        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        # Merge Settings
        merge_group = QGroupBox("Merge Settings")
        merge_group.setStyleSheet("QGroupBox { font-weight: 600; }")
        merge_layout = QVBoxLayout()
        merge_layout.setSpacing(8)

        self.unwrap_before_merge = QCheckBox("Unwrap phase before merge")
        self.unwrap_before_merge.setChecked(False)
        merge_layout.addWidget(self.unwrap_before_merge)

        neighbors_layout = QHBoxLayout()
        neighbors_layout.addWidget(QLabel("Neighbors:"))
        self.n_neighbors = QSpinBox()
        self.n_neighbors.setRange(1, 10)
        self.n_neighbors.setValue(3)
        neighbors_layout.addWidget(self.n_neighbors)
        neighbors_layout.addStretch()
        merge_layout.addLayout(neighbors_layout)

        merge_group.setLayout(merge_layout)
        layout.addWidget(merge_group)

        self.setLayout(layout)

        # Connect signals
        self.connect_signals()

    def setup_tooltips(self):
        """Set up helpful tooltips."""
        self.x_offset.setToolTip("X-axis offset for fiber array position")
        self.y_offset.setToolTip("Y-axis offset for fiber array position")
        self.x_position.setToolTip("X position for phase calibration")
        self.y_position.setToolTip("Y position for phase calibration")
        self.unwrap_before_merge.setToolTip("Apply 2D phase unwrapping before merging scans")
        self.n_neighbors.setToolTip("Number of nearest neighbors for phase interpolation")

    def connect_signals(self):
        """Connect all parameter change signals."""
        widgets = [
            self.x_offset,
            self.y_offset,
            self.x_position,
            self.y_position,
            self.unwrap_before_merge,
            self.n_neighbors,
        ]

        for widget in widgets:
            if isinstance(widget, QSpinBox | QDoubleSpinBox):
                widget.valueChanged.connect(self.on_parameter_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.on_parameter_changed)

    def on_parameter_changed(self):
        """Emit signal when any parameter changes."""
        self.parametersChanged.emit()

    def get_parameters(self) -> dict[str, Any]:
        """Get scan parameters."""
        return {
            "x_offset": self.x_offset.value(),
            "y_offset": self.y_offset.value(),
            "x_position": self.x_position.value(),
            "y_position": self.y_position.value(),
            "unwrap_before_merge": self.unwrap_before_merge.isChecked(),
            "n_neighbors": self.n_neighbors.value(),
        }

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        if "x_offset" in params:
            self.x_offset.setValue(params["x_offset"])
        if "y_offset" in params:
            self.y_offset.setValue(params["y_offset"])
        if "x_position" in params:
            self.x_position.setValue(params["x_position"])
        if "y_position" in params:
            self.y_position.setValue(params["y_position"])
        if "unwrap_before_merge" in params:
            self.unwrap_before_merge.setChecked(params["unwrap_before_merge"])
        if "n_neighbors" in params:
            self.n_neighbors.setValue(params["n_neighbors"])
