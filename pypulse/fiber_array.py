import numpy as np

from .base import FiberArrayBasic


class FiberArray(FiberArrayBasic):
    def __init__(self, dx: float, dy: float) -> None:
        """Initializes the fiber array with the offset dx and dy."""
        self._x_axis = (np.arange(14) - 13 / 2) * 1.1 + dx
        self._y_axis = (np.arange(14) - 13 / 2) * 1.1 + dy
        self._x_matrix, self._y_matrix = np.meshgrid(self._x_axis, self._y_axis)
        self._fiber_number = np.reshape(np.arange(14 * 14), (14, 14))
        self._number_x = 14
        self._number_y = 14

    @property
    def x_axis(self) -> np.ndarray:
        """Returns the x-axis of the fiber array."""
        return self._x_axis

    @property
    def y_axis(self) -> np.ndarray:
        """Returns the y-axis of the fiber array."""
        return self._y_axis

    @property
    def x_matrix(self) -> np.ndarray:
        """Returns the x matrix of the fiber array."""
        return self._x_matrix

    @property
    def y_matrix(self) -> np.ndarray:
        """Returns the y matrix of the fiber array."""
        return self._y_matrix

    @property
    def fiber_number(self) -> np.ndarray:
        """Returns the fiber number matrix of the fiber array."""
        return self._fiber_number

    @property
    def number_x(self) -> int:
        """Returns the number of fibers in the x direction."""
        return self._number_x

    @property
    def number_y(self) -> int:
        """Returns the number of fibers in the y direction."""
        return self._number_y
