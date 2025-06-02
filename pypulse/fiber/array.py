"""Fiber array definitions and management."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class FiberArrayInterface(ABC):
    """Abstract interface for fiber arrays."""

    @property
    @abstractmethod
    def x_axis(self) -> npt.NDArray[np.float64]:
        """X-axis coordinates."""
        pass

    @property
    @abstractmethod
    def y_axis(self) -> npt.NDArray[np.float64]:
        """Y-axis coordinates."""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Array shape (ny, nx)."""
        pass

    @abstractmethod
    def get_properties(self) -> dict[str, Any]:
        """Get all array properties."""
        pass


class FiberArray(FiberArrayInterface):
    """Standard fiber array implementation."""

    def __init__(self, config: dict[str, Any], dx: float = 0, dy: float = 0):
        """
        Initialize fiber array.

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - 'type': array type identifier
            - 'nx', 'ny': array dimensions
            - 'spacing': fiber spacing (or 'spacing_x', 'spacing_y')
            - Optional: 'pattern' for non-rectangular arrays
        dx, dy : float
            X and Y offsets
        """
        self._config = config
        self._dx = dx
        self._dy = dy
        self._initialize_array()

    def _initialize_array(self) -> None:
        """Initialize array based on configuration."""
        nx = self._config["nx"]
        ny = self._config["ny"]

        # Handle different spacing configurations
        if "spacing" in self._config:
            spacing_x = spacing_y = self._config["spacing"]
        else:
            spacing_x = self._config.get("spacing_x", 1.0)
            spacing_y = self._config.get("spacing_y", 1.0)

        # Create coordinate arrays
        self._x_axis = (np.arange(nx) - (nx - 1) / 2) * spacing_x + self._dx
        self._y_axis = (np.arange(ny) - (ny - 1) / 2) * spacing_y + self._dy

        # Create meshgrid
        self._x_matrix, self._y_matrix = np.meshgrid(self._x_axis, self._y_axis)

        # Create fiber numbering
        self._fiber_number = np.arange(nx * ny).reshape(ny, nx)

    @property
    def x_axis(self) -> npt.NDArray[np.float64]:
        return self._x_axis

    @property
    def y_axis(self) -> npt.NDArray[np.float64]:
        return self._y_axis

    @property
    def x_matrix(self) -> npt.NDArray[np.float64]:
        return self._x_matrix

    @property
    def y_matrix(self) -> npt.NDArray[np.float64]:
        return self._y_matrix

    @property
    def fiber_number(self) -> npt.NDArray[np.int64]:
        return self._fiber_number

    @property
    def shape(self) -> tuple[int, int]:
        return (self._config["ny"], self._config["nx"])

    @property
    def number_x(self) -> int:
        return self._config["nx"]

    @property
    def number_y(self) -> int:
        return self._config["ny"]

    def get_properties(self) -> dict[str, Any]:
        """Get all fiber array properties."""
        return {
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "x_matrix": self.x_matrix,
            "y_matrix": self.y_matrix,
            "fiber_number": self.fiber_number,
            "number_x": self.number_x,
            "number_y": self.number_y,
            "shape": self.shape,
            "config": self._config,
        }

    @classmethod
    def from_legacy_14x14(cls, dx: float = 0, dy: float = 0) -> "FiberArray":
        """Create legacy 14x14 array for backward compatibility."""
        config = {"type": "rectangular_14x14", "nx": 14, "ny": 14, "spacing": 1.1}
        return cls(config, dx, dy)
