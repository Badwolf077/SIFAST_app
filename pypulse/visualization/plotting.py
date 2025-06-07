"""Visualization utilities for SIFAST data."""

from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.io as pio
from matplotlib import colors

from ..utils.math import rescale

# Check for Mayavi availability
try:
    import mayavi.core.ui.api
    import mayavi.mlab

    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False

# Check for Plotly availability
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class SIFASTVisualizer:
    """Visualization methods for SIFAST data."""

    def __init__(self, sifast_instance, backend: str = "mayavi"):
        """
        Initialize visualizer.

        Parameters
        ----------
        sifast_instance : SIFAST
            SIFAST instance to visualize
        backend : str, optional
            The plotting backend to use, either "mayavi" or "plotly".
            Defaults to "mayavi".
        """
        self.sifast = sifast_instance
        self.backend = backend.lower()

        if self.backend == "mayavi" and not MAYAVI_AVAILABLE:
            raise ImportError(
                "Mayavi is selected as backend, but it's not installed. Please install Mayavi to use this backend."
            )
        if self.backend == "plotly" and not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is selected as backend, but it's not installed. Please install Plotly to use this backend."
            )
        if self.backend not in ["mayavi", "plotly"]:
            raise ValueError(f"Unsupported backend: {self.backend}. Choose 'mayavi' or 'plotly'.")

    def _plot_scatter_mayavi(
        self,
        values: npt.NDArray[np.float64],
        scene_model: Any | None = None,
    ) -> None:
        """Create 3D scatter plot using Mayavi."""
        if not MAYAVI_AVAILABLE:
            raise ImportError("Mayavi is required for 3D plotting with the 'mayavi' backend.")
        # Set up Mayavi
        if scene_model is None:
            fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            active_mlab = mayavi.mlab
            target_scene = fig
        else:
            active_mlab = scene_model.mlab
            target_scene = scene_model.mayavi_scene

        # Prepare data
        x_range = self.sifast.x_axis.max() - self.sifast.x_axis.min()
        values_flat = values.flatten()
        values_range = np.nanmax(values_flat) - np.nanmin(values_flat)

        scale_factor = x_range / values_range if values_range != 0 else 1.0
        values_scaled = values_flat * scale_factor

        # Create scatter plot
        points = active_mlab.points3d(
            self.sifast.x_matrix.flatten(),
            self.sifast.y_matrix.flatten(),
            values_scaled,
            values_flat,
            mode="sphere",
            colormap="inferno",
            scale_factor=0.5,
            figure=target_scene,
            resolution=20,
            scale_mode="none",
        )

        # Add outline and axes
        active_mlab.outline(points, color=(0.7, 0.7, 0.7))

        axes = active_mlab.axes(points)
        axes_range = list(axes.axes.bounds)
        axes_range[4:6] /= scale_factor
        axes.remove()

        axes = active_mlab.axes(points, xlabel="x (mm)", ylabel="y (mm)", nb_labels=5, ranges=axes_range)
        axes.axes.label_format = "%.1f"
        axes.title_text_property.color = (0.0, 0.0, 0.0)
        axes.label_text_property.color = (0.0, 0.0, 0.0)
        axes.axes.fly_mode = "none"

        active_mlab.colorbar(points, orientation="vertical")

        if target_scene:
            target_scene.scene.reset_zoom()

        if scene_model is None:
            mayavi.mlab.show()

    def _plot_scatter_plotly(
        self,
        values: npt.NDArray[np.float64],
        scene_model: Any | None = None,
    ) -> None:
        """Create 3D scatter plot using Plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D plotting with the 'plotly' backend.")

        x_flat = self.sifast.x_matrix.flatten()
        y_flat = self.sifast.y_matrix.flatten()
        values_flat = values.flatten()

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_flat,
                    y=y_flat,
                    z=values_flat,
                    mode="markers",
                    marker=dict(
                        size=5, color=values_flat, colorscale="Inferno", opacity=0.8, colorbar=dict(title="Values")
                    ),
                )
            ]
        )

        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title="x (mm)",
                yaxis_title="y (mm)",
                zaxis_title="Values",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        if scene_model is None:
            fig.show()
        else:
            scene_model.setHtml(fig.to_html(include_plotlyjs="cdn"))

    def plot_scatter(
        self,
        values: npt.NDArray[np.float64],
        scene_model: Any | None = None,
    ) -> None:
        """
        Create 3D scatter plot.

        Parameters
        ----------
        values : array_like
            Values to plot
        scene_model : optional
            Mayavi scene model (only used if backend is 'mayavi')
        """
        if self.backend == "mayavi":
            self._plot_scatter_mayavi(values, scene_model)
        elif self.backend == "plotly":
            self._plot_scatter_plotly(values, scene_model)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _prepare_isosurface_data(
        self,
        t_min: float,
        t_max: float,
        frequency_scale: float,
        indexing: str,
    ):
        """Helper to prepare data for isosurface plotting."""
        Et = self.sifast.Et
        if indexing == "xy":
            if self.backend == "mayavi":
                Et = np.transpose(Et, (1, 0, 2))
        elif indexing == "ij":
            if self.backend == "plotly":
                Et = np.transpose(Et, (1, 0, 2))

        # Process data
        if frequency_scale != 0:
            values = np.real(
                rescale(np.abs(Et))
                * np.exp(
                    1j * frequency_scale * self.sifast.t_axis[np.newaxis, np.newaxis, :] * self.sifast.omega_center
                )
                * np.exp(1j * frequency_scale * np.angle(Et))
            )
        else:
            values = rescale(np.abs(Et) ** 2)

        # Select time range
        t_mask = (self.sifast.t_axis > t_min) & (self.sifast.t_axis < t_max)
        values_plot = values[:, :, t_mask]
        t_axis_plot = self.sifast.t_axis[t_mask]

        return values_plot, t_axis_plot

    def _plot_isosurface_mayavi(
        self,
        values_plot: npt.NDArray[np.float64],
        t_axis_plot: npt.NDArray[np.float64],
        isovalue: float,
        scene_model: Any | None = None,
        **kwargs,
    ) -> None:
        """Create 3D isosurface plot using Mayavi."""
        if not MAYAVI_AVAILABLE:
            raise ImportError("Mayavi is required for 3D plotting with the 'mayavi' backend.")

        opacity = kwargs.pop("opacity", 1)

        if scene_model is None:
            fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            active_mlab = mayavi.mlab
            target_scene = fig
        else:
            active_mlab = scene_model.mlab
            target_scene = scene_model.mayavi_scene

        # Create meshgrid
        x_plot, y_plot, z_plot = np.meshgrid(self.sifast.x_axis, self.sifast.y_axis, t_axis_plot, indexing="ij")

        # Initial isosurface for scaling
        iso_temp = active_mlab.contour3d(
            x_plot, y_plot, z_plot, values_plot, contours=[isovalue], color=colors.to_rgb("lightblue"), opacity=0.5
        )

        # Get scaling
        axes = active_mlab.axes(iso_temp)
        axes_range = axes.axes.bounds
        iso_temp.remove()
        axes.remove()

        x_range = axes_range[1] - axes_range[0]
        z_range = axes_range[5] - axes_range[4]
        zoom_factor = 1.618 if kwargs.get("zoom", None) is None else kwargs["zoom"]
        scale_factor_z = x_range / z_range * zoom_factor if z_range != 0 else 1.0

        # Rescale z-axis
        z_plot_scaled = t_axis_plot * scale_factor_z
        x_plot, y_plot, z_plot = np.meshgrid(self.sifast.x_axis, self.sifast.y_axis, z_plot_scaled, indexing="ij")

        # Final isosurface
        isosurface = active_mlab.contour3d(
            x_plot, y_plot, z_plot, values_plot, contours=[isovalue], color=colors.to_rgb("lightblue"), opacity=opacity
        )

        # Add outline and axes
        active_mlab.outline(isosurface, color=(0.7, 0.7, 0.7))

        axes = active_mlab.axes(
            isosurface, xlabel="x (mm)", ylabel="y (mm)", zlabel="t (fs)", nb_labels=5, ranges=axes_range
        )
        axes.axes.label_format = "%.1f"
        axes.title_text_property.color = (0.0, 0.0, 0.0)
        axes.label_text_property.color = (0.0, 0.0, 0.0)
        axes.axes.fly_mode = "none"

        active_mlab.colorbar(isosurface, orientation="vertical")

        if target_scene:
            target_scene.scene.reset_zoom()

        if scene_model is None:
            mayavi.mlab.show()

    def _plot_isosurface_plotly(
        self,
        values_plot: npt.NDArray[np.float64],  # Should be (nx, ny, nt)
        t_axis_plot: npt.NDArray[np.float64],
        isovalue: float,
        scene_model: Any | None = None,
        **kwargs,
    ) -> None:
        """Create 3D isosurface plot using Plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D plotting with the 'plotly' backend.")

        opacity = kwargs.pop("opacity", 1)

        x_plot, y_plot, z_plot = np.meshgrid(self.sifast.x_axis, self.sifast.y_axis, t_axis_plot)

        fig = go.Figure(
            data=go.Isosurface(
                x=x_plot.flatten(),
                y=y_plot.flatten(),
                z=z_plot.flatten(),
                value=values_plot.flatten(),
                isomin=isovalue,
                isomax=isovalue,
                surface_count=1,
                opacity=opacity,
                colorscale=kwargs.get("colorscale", "Viridis"),
                showscale=True,  # Show color bar
                colorbar=dict(title="Value"),
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )

        aspectratio_plotly = kwargs.get("aspectratio", dict(x=1, y=1, z=1))

        fig.update_layout(
            title="3D Isosurface Plot",
            scene=dict(
                xaxis_title="x (mm)",
                yaxis_title="y (mm)",
                zaxis_title="t (fs)",
                aspectmode="manual",
                aspectratio=aspectratio_plotly,
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        if scene_model is None:
            fig.show(renderer=kwargs.get("renderer", "browser"))
        else:
            scene_model.setHtml(fig.to_html(include_plotlyjs="cdn"))

    def plot_isosurface(
        self,
        t_min: float,
        t_max: float,
        frequency_scale: float,
        isovalue: float,
        indexing: str = "xy",
        scene_model: Any | None = None,
        **kwargs,
    ) -> None:
        """
        Create 3D isosurface plot.

        Parameters
        ----------
        t_min, t_max : float
            Time range
        frequency_scale : float
            Frequency scaling factor
        isovalue : float
            Isosurface value
        indexing : str
            Coordinate indexing ('xy' or 'ij'). 'xy' means input Et is (ny, nx, nt).
            'ij' means input Et is (nx, ny, nt).
            Internal processing assumes (nx, ny, nt).
        scene_model : optional
            Mayavi scene model (only used if backend is 'mayavi')
        **kwargs
            Additional options (e.g., opacity, color, colorscale, aspectratio for plotly)
        """
        values_plot, t_axis_plot = self._prepare_isosurface_data(t_min, t_max, frequency_scale, indexing)

        if self.backend == "mayavi":
            self._plot_isosurface_mayavi(values_plot, t_axis_plot, isovalue, scene_model, **kwargs)
        elif self.backend == "plotly":
            self._plot_isosurface_plotly(values_plot, t_axis_plot, isovalue, scene_model, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
