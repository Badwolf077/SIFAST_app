"""
Main window for PyPulse application.
"""

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QSize, Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .processing.processor import ProcessingThread
from .utils.icons import IconManager
from .widgets.log_dock import LogDock
from .widgets.parameter_panel import ParameterPanel
from .widgets.visualization_panel import VisualizationPanel


class PyPulseMainWindow(QMainWindow):
    """Main application window for PyPulse GUI."""

    # Signals
    pulseProcessed = Signal(object)

    def __init__(self):
        super().__init__()
        self.settings = QSettings("PyPulse", "MainWindow")
        self.pulse = None
        self.processing_thread = None
        self.icon_manager = IconManager()

        self.init_ui()
        self.setup_connections()
        self.load_settings()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("PyPulse - Spatiotemporal Pulse Characterization")
        self.setMinimumSize(1280, 720)

        # Create toolbar with compact design
        self.create_toolbar()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main splitter (horizontal)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(5)

        # Left panel - Parameters
        self.parameter_panel = ParameterPanel()
        self.parameter_panel.setMinimumWidth(320)
        self.parameter_panel.setMaximumWidth(400)

        # Right panel - Visualization
        self.visualization_panel = VisualizationPanel()

        # Add to splitter
        self.main_splitter.addWidget(self.parameter_panel)
        self.main_splitter.addWidget(self.visualization_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

        # Create vertical splitter for main content and logs
        self.vertical_splitter = QSplitter(Qt.Vertical)
        self.vertical_splitter.setHandleWidth(5)
        self.vertical_splitter.addWidget(self.main_splitter)

        # Create log dock
        self.log_dock = LogDock()
        self.log_dock.setMaximumHeight(150)
        self.vertical_splitter.addWidget(self.log_dock)

        # Set splitter sizes
        self.vertical_splitter.setSizes([600, 120])

        main_layout.addWidget(self.vertical_splitter)
        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("QStatusBar { background: #f8f9fa; border-top: 1px solid #dee2e6; }")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_toolbar(self):
        """Create the main toolbar with compact design."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.setStyleSheet("""
            QToolBar {
                background: #ffffff;
                border-bottom: 1px solid #dee2e6;
                padding: 4px;
                spacing: 4px;
            }
            QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                margin: 2px;
                color: #495057;
                font-size: 11px;
            }
            QToolButton:hover {
                background: #e9ecef;
                border-color: #dee2e6;
            }
            QToolButton:pressed {
                background: #dee2e6;
            }
            QToolButton:disabled {
                color: #adb5bd;
            }
        """)
        self.addToolBar(toolbar)

        # Calibration action
        self.calibration_action = QAction("Calibration", self)
        self.calibration_action.setIcon(self.icon_manager.get_icon("calibration"))
        toolbar.addAction(self.calibration_action)

        toolbar.addSeparator()

        # Open folder action
        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.setIcon(self.icon_manager.get_icon("folder_open"))
        toolbar.addAction(self.open_folder_action)

        toolbar.addSeparator()

        # Acquire action
        self.acquire_action = QAction("Acquire", self)
        self.acquire_action.setIcon(self.icon_manager.get_icon("acquire"))
        toolbar.addAction(self.acquire_action)

        # Save data action
        self.save_data_action = QAction("Save Data", self)
        self.save_data_action.setIcon(self.icon_manager.get_icon("save"))
        self.save_data_action.setEnabled(False)
        toolbar.addAction(self.save_data_action)

        toolbar.addSeparator()

        # Scan action
        self.scan_action = QAction("Scan", self)
        self.scan_action.setIcon(self.icon_manager.get_icon("scan"))
        toolbar.addAction(self.scan_action)

        # Open scan folder action
        self.open_scan_action = QAction("Open Scan", self)
        self.open_scan_action.setIcon(self.icon_manager.get_icon("folder_scan"))
        toolbar.addAction(self.open_scan_action)

        toolbar.addSeparator()

        # Save results action
        self.save_results_action = QAction("Save Results", self)
        self.save_results_action.setIcon(self.icon_manager.get_icon("save_results"))
        self.save_results_action.setEnabled(False)
        toolbar.addAction(self.save_results_action)

    def setup_connections(self):
        """Connect all signals and slots."""
        # Toolbar actions
        self.calibration_action.triggered.connect(self.open_calibration)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.acquire_action.triggered.connect(self.acquire_data)
        self.save_data_action.triggered.connect(self.save_data)
        self.scan_action.triggered.connect(self.start_scan)
        self.open_scan_action.triggered.connect(self.open_scan_folder)
        self.save_results_action.triggered.connect(self.save_results)

        # Parameter changes
        self.parameter_panel.parametersChanged.connect(self.on_parameters_changed)

        # Visualization actions
        self.visualization_panel.exportRequested.connect(self.export_plots)
        self.visualization_panel.saveRequested.connect(self.save_results)

        # Processing signals
        self.pulseProcessed.connect(self.on_pulse_processed)

    def open_calibration(self):
        """Open calibration application."""
        self.log_dock.log("Opening calibration application...", "INFO")
        # TODO: Launch external calibration app
        QMessageBox.information(self, "Calibration", "Calibration application will be launched.")

    def open_folder(self):
        """Open folder for single measurement processing."""
        folder = QFileDialog.getExistingDirectory(self, "Select Measurement Folder", str(Path.cwd() / "data"))

        if folder:
            self.log_dock.log(f"Opening folder: {folder}", "INFO")
            self.process_single_measurement(folder)

    def process_single_measurement(self, folder_path: str):
        """Process a single measurement folder."""
        try:
            # Get parameters
            params = self.parameter_panel.get_all_parameters()
            params["folder_path"] = folder_path
            params["mode_input"] = "read"  # Force read mode

            # Disable UI during processing
            self.set_ui_enabled(False)
            self.status_bar.showMessage("Processing measurement...")

            # Create and start processing thread
            self.processing_thread = ProcessingThread(params, mode="single")
            self.processing_thread.status.connect(self.log_dock.log)
            self.processing_thread.error.connect(lambda e: self.log_dock.log(e, "ERROR"))
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.start()

        except Exception as e:
            self.log_dock.log(f"Error: {str(e)}", "ERROR")
            self.set_ui_enabled(True)

    def on_processing_finished(self, pulse):
        """Handle processing completion."""
        self.set_ui_enabled(True)

        if pulse is not None:
            self.pulse = pulse
            self.pulseProcessed.emit(pulse)
            self.save_results_action.setEnabled(True)
            self.status_bar.showMessage("Processing completed successfully")
            self.log_dock.log("Processing completed successfully", "SUCCESS")
        else:
            self.status_bar.showMessage("Processing failed")

    @Slot(object)
    def on_pulse_processed(self, pulse):
        """Handle pulse processed signal."""
        self.visualization_panel.set_pulse(pulse)

    def acquire_data(self):
        """Acquire data from spectrometer."""
        self.log_dock.log("Starting data acquisition...", "INFO")
        # TODO: Implement spectrometer SDK integration
        QMessageBox.information(self, "Acquisition", "Spectrometer SDK integration pending.")
        # After acquisition, enable save
        self.save_data_action.setEnabled(True)

    def save_data(self):
        """Save acquired data."""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", str(Path.cwd() / "data"))

        if folder:
            self.log_dock.log(f"Saving data to: {folder}", "INFO")
            # TODO: Implement data saving
            self.save_data_action.setEnabled(False)

    def start_scan(self):
        """Start scanning process."""
        self.log_dock.log("Starting scan...", "INFO")

        # Get scan parameters
        scan_params = self.parameter_panel.get_scan_parameters()

        # Create progress dialog
        progress = QProgressDialog("Scanning in progress...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setStyleSheet("""
            QProgressDialog {
                background: white;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #0d6efd;
                border-radius: 3px;
            }
        """)

        # TODO: Implement actual scanning
        for i in range(101):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                self.log_dock.log("Scan cancelled", "WARNING")
                break

        progress.close()

    def open_scan_folder(self):
        """Open folder containing scan data."""
        folder = QFileDialog.getExistingDirectory(self, "Select Scan Data Folder", str(Path.cwd() / "data"))

        if folder:
            self.log_dock.log(f"Processing scan data from: {folder}", "INFO")
            # TODO: Implement scan data processing with merge_spatial_scans

    def save_results(self):
        """Save processed results (axes and Et)."""
        if self.pulse is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", str(Path.cwd() / "results.h5"), "HDF5 Files (*.h5);;All Files (*)"
        )

        if file_path:
            self.log_dock.log(f"Saving results to: {file_path}", "INFO")
            # TODO: Implement saving of axes and Et data

    def export_plots(self):
        """Export current plots."""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Directory", str(Path.cwd() / "exports"))

        if folder:
            self.log_dock.log(f"Exporting plots to: {folder}", "INFO")
            # TODO: Implement plot export functionality

    def on_parameters_changed(self, params: dict[str, Any]):
        """Handle parameter changes."""
        # Log parameter changes for debugging
        self.log_dock.log("Parameters updated", "DEBUG")

    def set_ui_enabled(self, enabled: bool):
        """Enable/disable UI during processing."""
        self.parameter_panel.setEnabled(enabled)
        self.open_folder_action.setEnabled(enabled)
        self.acquire_action.setEnabled(enabled)
        self.scan_action.setEnabled(enabled)
        self.open_scan_action.setEnabled(enabled)

    def load_settings(self):
        """Load saved settings."""
        # Window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Window state
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

        # Splitter states
        main_splitter_state = self.settings.value("mainSplitter")
        if main_splitter_state:
            self.main_splitter.restoreState(main_splitter_state)

        vertical_splitter_state = self.settings.value("verticalSplitter")
        if vertical_splitter_state:
            self.vertical_splitter.restoreState(vertical_splitter_state)

        # Parameters
        params = self.settings.value("parameters")
        if params:
            self.parameter_panel.set_parameters(params)

        self.log_dock.log("Settings loaded from previous session", "INFO")

    def save_settings(self):
        """Save current settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("mainSplitter", self.main_splitter.saveState())
        self.settings.setValue("verticalSplitter", self.vertical_splitter.saveState())
        self.settings.setValue("parameters", self.parameter_panel.get_all_parameters())

    def closeEvent(self, event):
        """Handle window close event."""
        self.save_settings()
        event.accept()
