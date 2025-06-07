"""
Individual plot widgets for PyPulse visualization.
"""

import json
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlotlyWidget(QWebEngineView):
    """Base widget for displaying Plotly plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channel = QWebChannel()
        self.page().setWebChannel(self.channel)

    def plot_data(self, fig_dict: dict[str, Any]):
        """Display a plotly figure dictionary."""
        html = self._generate_plotly_html(fig_dict)
        self.setHtml(html)

    def _generate_plotly_html(self, fig_dict: dict[str, Any]) -> str:
        """Generate HTML with Plotly plot."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: white;
                }}
                #plot {{
                    width: 100%;
                    height: 100vh;
                }}
            </style>
        </head>
        <body>
            <div id="plot"></div>
            <script>
                var figure = {json.dumps(fig_dict)};
                var config = {{
                    responsive: true,
                    toImageButtonOptions: {{
                        format: 'png',
                        filename: 'pypulse_plot',
                        height: 800,
                        width: 1200,
                        scale: 2
                    }},
                    modeBarButtonsToAdd: ['hovercompare', 'hoverclosest']
                }};
                Plotly.newPlot('plot', figure.data, figure.layout, config);
            </script>
        </body>
        </html>
        """


class PulseFrontPlot(QWidget):
    """Widget for pulse front visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        title = QLabel("Pulse Front")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #495057;")
        layout.addWidget(title)

        # Plot
        self.plot_widget = PlotlyWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def update_plot(self, pulse):
        """Update plot with pulse data."""
        if pulse is None:
            return

        # Create heatmap of pulse front
        fig = {
            "data": [
                {
                    "type": "heatmap",
                    "z": pulse.pulse_front.tolist() if hasattr(pulse, "pulse_front") else [[0]],
                    "x": pulse.x_axis.tolist() if hasattr(pulse, "x_axis") else [0],
                    "y": pulse.y_axis.tolist() if hasattr(pulse, "y_axis") else [0],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Time (fs)", "titleside": "right"},
                    "hovertemplate": "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Time: %{z:.2f} fs<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {"title": "X (mm)", "showgrid": True, "gridcolor": "#e9ecef"},
                "yaxis": {"title": "Y (mm)", "showgrid": True, "gridcolor": "#e9ecef"},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif"},
                "margin": {"l": 60, "r": 20, "t": 20, "b": 60},
                "hoverlabel": {"bgcolor": "white", "font": {"size": 12}},
            },
        }

        self.plot_widget.plot_data(fig)


class PhasePlot(QWidget):
    """Widget for phase distribution visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        title = QLabel("Phase")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #495057;")
        layout.addWidget(title)

        # Plot
        self.plot_widget = PlotlyWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def update_plot(self, pulse):
        """Update plot with pulse data."""
        if pulse is None:
            return

        # Get phase at center frequency
        if hasattr(pulse, "phase") and pulse.phase.ndim >= 3:
            phase_data = pulse.phase[:, :, pulse.n_omega // 2]
        else:
            phase_data = np.zeros((10, 10))

        fig = {
            "data": [
                {
                    "type": "heatmap",
                    "z": phase_data.tolist(),
                    "x": pulse.x_axis.tolist() if hasattr(pulse, "x_axis") else list(range(10)),
                    "y": pulse.y_axis.tolist() if hasattr(pulse, "y_axis") else list(range(10)),
                    "colorscale": "RdBu",
                    "colorbar": {"title": "Phase (rad)", "titleside": "right"},
                    "hovertemplate": "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Phase: %{z:.3f} rad<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {"title": "X (mm)", "showgrid": True, "gridcolor": "#e9ecef"},
                "yaxis": {"title": "Y (mm)", "showgrid": True, "gridcolor": "#e9ecef"},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif"},
                "margin": {"l": 60, "r": 20, "t": 20, "b": 60},
            },
        }

        self.plot_widget.plot_data(fig)


class SpectralProfilePlot(QWidget):
    """Widget for spectral profile visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        title = QLabel("Spectral Profile")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #495057;")
        layout.addWidget(title)

        # Plot
        self.plot_widget = PlotlyWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def update_plot(self, pulse):
        """Update plot with pulse data."""
        if pulse is None:
            return

        # Get spectral data (averaged over spatial dimensions)
        if hasattr(pulse, "Sw_unknown"):
            spectral_data = np.nanmean(pulse.Sw_unknown, axis=(0, 1))
            wavelength = (
                pulse.wavelength_axis
                if hasattr(pulse, "wavelength_axis")
                else np.linspace(700, 900, len(spectral_data))
            )
        else:
            wavelength = np.linspace(700, 900, 100)
            spectral_data = np.exp(-(((wavelength - 800) / 50) ** 2))

        fig = {
            "data": [
                {
                    "type": "scatter",
                    "x": wavelength.tolist(),
                    "y": spectral_data.tolist(),
                    "mode": "lines",
                    "line": {"color": "#0d6efd", "width": 2},
                    "fill": "tozeroy",
                    "fillcolor": "rgba(13, 110, 253, 0.1)",
                    "hovertemplate": "λ: %{x:.1f} nm<br>Intensity: %{y:.3f}<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {"title": "Wavelength (nm)", "showgrid": True, "gridcolor": "#e9ecef"},
                "yaxis": {"title": "Intensity (a.u.)", "showgrid": True, "gridcolor": "#e9ecef"},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif"},
                "margin": {"l": 60, "r": 20, "t": 20, "b": 60},
            },
        }

        self.plot_widget.plot_data(fig)


class TemporalProfilePlot(QWidget):
    """Widget for temporal profile visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        title = QLabel("Temporal Profile")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #495057;")
        layout.addWidget(title)

        # Plot
        self.plot_widget = PlotlyWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def update_plot(self, pulse):
        """Update plot with pulse data."""
        if pulse is None:
            return

        # Get temporal data
        if hasattr(pulse, "Et") and hasattr(pulse, "t_axis"):
            Et = pulse.Et
            # Average over spatial dimensions
            temporal_data = np.nanmean(np.abs(Et) ** 2, axis=(0, 1))
            t_axis = pulse.t_axis
        else:
            t_axis = np.linspace(-500, 500, 1000)
            temporal_data = np.exp(-(((t_axis) / 100) ** 2))

        fig = {
            "data": [
                {
                    "type": "scatter",
                    "x": t_axis.tolist(),
                    "y": temporal_data.tolist(),
                    "mode": "lines",
                    "line": {"color": "#dc3545", "width": 2},
                    "fill": "tozeroy",
                    "fillcolor": "rgba(220, 53, 69, 0.1)",
                    "hovertemplate": "Time: %{x:.1f} fs<br>Intensity: %{y:.3f}<extra></extra>",
                }
            ],
            "layout": {
                "xaxis": {"title": "Time (fs)", "showgrid": True, "gridcolor": "#e9ecef"},
                "yaxis": {"title": "Intensity (a.u.)", "showgrid": True, "gridcolor": "#e9ecef"},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif"},
                "margin": {"l": 60, "r": 20, "t": 20, "b": 60},
            },
        }

        self.plot_widget.plot_data(fig)


class IsosurfacePlot(QWidget):
    """Widget for 3D isosurface visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title
        title = QLabel("3D Isosurface")
        title.setStyleSheet("font-weight: 600; font-size: 14px; color: #495057;")
        layout.addWidget(title)

        # Plot
        self.plot_widget = PlotlyWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def update_plot(self, pulse, t_min=-250, t_max=700, frequency_scale=0, isovalue=0.05, opacity=0.9):
        """Update plot with pulse data and parameters."""
        if pulse is None:
            return

        # For now, create a placeholder 3D visualization
        # In real implementation, this would use pulse.Et data
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        z = np.linspace(t_min, t_max, 50)

        X, Y, Z = np.meshgrid(x, y, z)
        values = np.exp(-(X**2 + Y**2) / 10) * np.exp(-((Z / 200) ** 2))

        fig = {
            "data": [
                {
                    "type": "isosurface",
                    "x": X.flatten().tolist(),
                    "y": Y.flatten().tolist(),
                    "z": Z.flatten().tolist(),
                    "value": values.flatten().tolist(),
                    "isomin": isovalue * 0.8,
                    "isomax": isovalue * 1.2,
                    "opacity": opacity,
                    "surface": {"show": True},
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Intensity", "titleside": "right"},
                    "caps": {"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
                }
            ],
            "layout": {
                "scene": {
                    "xaxis": {"title": "X (mm)", "showgrid": True, "gridcolor": "#e9ecef", "backgroundcolor": "white"},
                    "yaxis": {"title": "Y (mm)", "showgrid": True, "gridcolor": "#e9ecef", "backgroundcolor": "white"},
                    "zaxis": {
                        "title": "Time (fs)",
                        "showgrid": True,
                        "gridcolor": "#e9ecef",
                        "backgroundcolor": "white",
                    },
                    "bgcolor": "white",
                    "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
                },
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif"},
                "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            },
        }

        self.plot_widget.plot_data(fig)
