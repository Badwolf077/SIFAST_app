"""Base classes for pulse analysis."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class PulseInterface(ABC):
    """Abstract interface for pulse objects."""

    SPEED_OF_LIGHT: float = 299792458 * 1e9 * 1e-15  # in fs/nm

    @property
    @abstractmethod
    def t_axis(self) -> npt.NDArray[np.float64]:
        """Time axis of the pulse."""
        pass

    @property
    @abstractmethod
    def omega_axis(self) -> npt.NDArray[np.float64]:
        """Frequency axis of the pulse."""
        pass

    @property
    @abstractmethod
    def wavelength_axis(self) -> npt.NDArray[np.float64]:
        """Wavelength axis of the pulse."""
        pass
