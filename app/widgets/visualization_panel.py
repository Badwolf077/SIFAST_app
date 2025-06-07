"""
Visualization panel containing all plots and controls.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget

from ..styles import BUTTON_STYLE, SECONDARY_BUTTON_STYLE
from .isosurface_controls import IsosurfaceControls
from .plot_widgets import IsosurfacePlot, PhasePlot, PulseFrontPlot, SpectralProfilePlot, TemporalProfilePlot


class VisualizationPanel(QWidget):
    """Main visualization panel with tabs for different plot types."""

    exportRequested = Signal()
    saveRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pulse = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background: white;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QTabBar::tab {
                background: #f8f9fa;
                padding: 10px 20px;
                margin-right: 2px;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
            }
            QTabBar::tab:hover:!selected {
                background: #e9ecef;
            }
        """)

        # Create tabs
        self._create_pulse_analysis_tab()
        self._create_3d_visualization_tab()

        layout.addWidget(self.tab_widget)

        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(12, 8, 12, 8)
        button_layout.setSpacing(8)

        self.export_btn = QPushButton("Export Plots")
        self.export_btn.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.export_btn.clicked.connect(self.exportRequested.emit)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.setStyleSheet(BUTTON_STYLE)
        self.save_btn.clicked.connect(self.saveRequested.emit)

        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _create_pulse_analysis_tab(self):
        """Create pulse analysis tab with side-by-side plots."""
        tab = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Create splitter for resizable plots
        splitter = QSplitter(Qt.Horizontal)

        # Pulse front plot
        self.pulse_front_plot = PulseFrontPlot()
        splitter.addWidget(self.pulse_front_plot)

        # Phase plot
        self.phase_plot = PhasePlot()
        splitter.addWidget(self.phase_plot)

        # Equal sizes by default
        splitter.setSizes([640, 640])

        layout.addWidget(splitter)
        tab.setLayout(layout)

        self.tab_widget.addTab(tab, "Pulse Analysis")

    def _create_3d_visualization_tab(self):
        """Create 3D visualization tab with multiple plots."""
        tab = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side - Spectral and Temporal profiles
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Spectral profile
        self.spectral_profile_plot = SpectralProfilePlot()
        left_layout.addWidget(self.spectral_profile_plot)

        # Temporal profile
        self.temporal_profile_plot = TemporalProfilePlot()
        left_layout.addWidget(self.temporal_profile_plot)

        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)

        # Right side - 3D Isosurface with controls
        right_widget = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Isosurface plot
        self.isosurface_plot = IsosurfacePlot()
        right_layout.addWidget(self.isosurface_plot, 3)

        # Isosurface controls (compact)
        self.isosurface_controls = IsosurfaceControls()
        self.isosurface_controls.reconstructRequested.connect(self._reconstruct_isosurface)
        right_layout.addWidget(self.isosurface_controls, 1)

        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)

        # Set initial sizes (40% left, 60% right)
        main_splitter.setSizes([500, 750])

        layout.addWidget(main_splitter)
        tab.setLayout(layout)

        self.tab_widget.addTab(tab, "3D Visualization")

    def set_pulse(self, pulse):
        """Set the pulse object and update all plots."""
        self.pulse = pulse
        self.update_plots()

    def update_plots(self):
        """Update all plots with current pulse data."""
        if self.pulse is None:
            return

        # Update pulse analysis plots
        self.pulse_front_plot.update_plot(self.pulse)
        self.phase_plot.update_plot(self.pulse)

        # Update profile plots
        self.spectral_profile_plot.update_plot(self.pulse)
        self.temporal_profile_plot.update_plot(self.pulse)

        # Update isosurface
        self._reconstruct_isosurface()

    def _reconstruct_isosurface(self):
        """Reconstruct isosurface with current parameters."""
        if self.pulse is None:
            return

        params = self.isosurface_controls.get_parameters()
        self.isosurface_plot.update_plot(self.pulse, **params)
