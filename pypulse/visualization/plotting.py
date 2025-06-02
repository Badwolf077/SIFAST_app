"""Visualization utilities for SIFAST data."""

from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import colors

from ..utils.math import rescale


class SIFASTVisualizer:
    """Visualization methods for SIFAST data."""

    def __init__(self, sifast_instance):
        """
        Initialize visualizer.

        Parameters
        ----------
        sifast_instance : SIFAST
            SIFAST instance to visualize
        """
        self.sifast = sifast_instance

    def plot_scatter(self, values: npt.NDArray[np.float64], scene_model: Any | None = None) -> None:
        """
        Create 3D scatter plot.

        Parameters
        ----------
        values : array_like
            Values to plot
        scene_model : optional
            Mayavi scene model
        """
        try:
            import mayavi.core.ui.api
            import mayavi.mlab
        except ImportError:
            raise ImportError("Mayavi is required for 3D plotting")

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
            Coordinate indexing ('xy' or 'ij')
        scene_model : optional
            Mayavi scene model
        **kwargs
            Additional options (e.g., opacity)
        """
        try:
            import mayavi.core.ui.api
            import mayavi.mlab
        except ImportError:
            raise ImportError("Mayavi is required for 3D plotting")

        # Set up Mayavi
        if scene_model is None:
            fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            active_mlab = mayavi.mlab
            target_scene = fig
        else:
            active_mlab = scene_model.mlab
            target_scene = scene_model.mayavi_scene

        opacity = kwargs.pop("opacity", 1)

        # Get electric field
        Et = self.sifast.Et
        if indexing == "xy":
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
