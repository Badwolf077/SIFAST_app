"""Core pulse representation and operations."""

import numpy as np
import numpy.typing as npt
import scipy.interpolate as interp
from scipy.signal import find_peaks

from ..utils.math import rescale
from .base import PulseInterface
from .transforms import FourierTransforms


class PulseBase(PulseInterface, FourierTransforms):
    """Base class for pulse representations with common operations."""

    def __init__(self):
        self._t_axis: npt.NDArray[np.float64] | None = None
        self._omega_axis: npt.NDArray[np.float64] | None = None
        self._wavelength_axis: npt.NDArray[np.float64] | None = None

    @property
    def t_axis(self) -> npt.NDArray[np.float64]:
        if self._t_axis is None:
            raise ValueError("Time axis not initialized")
        return self._t_axis

    @t_axis.setter
    def t_axis(self, value: npt.NDArray[np.float64]) -> None:
        self._t_axis = np.asarray(value, dtype=np.float64)

    @property
    def omega_axis(self) -> npt.NDArray[np.float64]:
        if self._omega_axis is None:
            raise ValueError("Frequency axis not initialized")
        return self._omega_axis

    @omega_axis.setter
    def omega_axis(self, value: npt.NDArray[np.float64]) -> None:
        self._omega_axis = np.asarray(value, dtype=np.float64)

    @property
    def wavelength_axis(self) -> npt.NDArray[np.float64]:
        if self._wavelength_axis is None:
            raise ValueError("Wavelength axis not initialized")
        return self._wavelength_axis

    @wavelength_axis.setter
    def wavelength_axis(self, value: npt.NDArray[np.float64]) -> None:
        self._wavelength_axis = np.asarray(value, dtype=np.float64)

    def resample_spectrum(
        self,
        spectrum: npt.NDArray[np.float64],
        wavelength_center: float,
        n_omega: int,
        wavelength_width: float,
        method: str = "linear",
    ) -> npt.NDArray[np.float64]:
        """
        Resample spectrum to new wavelength axis.

        Parameters
        ----------
        spectrum : array_like
            Input spectrum
        wavelength_center : float
            Center wavelength
        n_omega : int
            Number of frequency points
        wavelength_width : float
            Wavelength range width
        method : str, optional
            Interpolation method

        Returns
        -------
        array_like
            Resampled spectrum
        """
        # Validate attributes
        if not hasattr(self, "wavelength") or not hasattr(self, "omega_center"):
            raise ValueError("Object must have 'wavelength' and 'omega_center' attributes")

        # Remove background
        mask = np.abs(self.wavelength - wavelength_center) > wavelength_width / 2
        background = np.mean(spectrum[mask]) if np.any(mask) else 0
        spectrum = spectrum - background

        # Convert to frequency domain
        omega = 2 * np.pi * self.SPEED_OF_LIGHT / self.wavelength - self.omega_center
        delta_omega = (
            2 * np.pi * self.SPEED_OF_LIGHT * (1 / (wavelength_center - wavelength_width / 2) - 1 / wavelength_center)
        )

        # Create new axes
        self.omega_axis = np.linspace(-delta_omega, delta_omega, n_omega)
        self.wavelength_axis = 2 * np.pi * self.SPEED_OF_LIGHT / (self.omega_axis + self.omega_center)

        # Interpolate
        interpolator = interp.interp1d(omega, spectrum, kind=method, bounds_error=False, fill_value=0)
        spectrum_resampled = interpolator(self.omega_axis)

        # Ensure non-negative
        spectrum_resampled[spectrum_resampled < 0] = 0

        return spectrum_resampled

    def fourier_transform_spectral_interferometry(
        self, n_omega: int, n_fft: int, delay_min: float | None = None, filter_order: int = 8
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Perform Fourier transform spectral interferometry.

        Parameters
        ----------
        n_omega : int
            Number of frequency points
        n_fft : int
            FFT size
        delay_min : float, optional
            Minimum delay for peak detection
        filter_order : int, optional
            Filter order (must be even)

        Returns
        -------
        phase : array_like
            Extracted phase
        delay : array_like
            Delay map
        Su : array_like
            Unknown spectrum
        """
        # Validate inputs
        if filter_order % 2 != 0:
            raise ValueError("Filter order must be even")

        required_attrs = ["Sw_interference", "row", "col"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f"'{attr}' attribute is required")

        # Set up time axis
        f_max = (
            np.max(self.omega_axis) + (n_fft - n_omega) / 2 * np.abs(self.omega_axis[1] - self.omega_axis[0])
        ) / np.pi
        self.t_axis = (np.arange(n_fft) - (n_fft - 1) / 2) / f_max

        # Transform to time domain
        St = self.iFt(self.Sw_interference, n_omega, n_fft)

        # Extract delays
        delay = self._extract_delays(St, n_fft, delay_min)

        # Apply filters
        phase, Su = self._apply_filters(St, delay, filter_order, n_omega, n_fft)

        return phase, delay, Su

    def _extract_delays(
        self, St: npt.NDArray[np.complex128], n_fft: int, delay_min: float | None
    ) -> npt.NDArray[np.float64]:
        """Extract delay values from time-domain signal."""
        delay = np.full((St.shape[0], St.shape[1]), np.nan)

        if delay_min is None:
            t_start = n_fft // 2
            t_axis_temp = self.t_axis[t_start:]
        else:
            mask = self.t_axis > delay_min / 2
            t_axis_temp = self.t_axis[mask]
            t_start = np.where(mask)[0][0]

        for i, (r, c) in enumerate(zip(self.row, self.col)):
            signal = rescale(np.abs(St[r, c, t_start:]))
            peaks, _ = find_peaks(signal, height=0.01)

            if len(peaks) > 1:
                # Get strongest peak
                peak_idx = peaks[np.argmax(signal[peaks])]
                delay[r, c] = t_axis_temp[peak_idx]

        return delay

    def _apply_filters(
        self,
        St: npt.NDArray[np.complex128],
        delay: npt.NDArray[np.float64],
        filter_order: int,
        n_omega: int,
        n_fft: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Apply AC/DC filters and extract phase."""
        # Create filter widths
        filter_width = (-np.log(0.001)) ** (-1 / filter_order) * delay / 2

        # Broadcast for vectorized operations
        t_broadcast = self.t_axis.reshape(1, 1, -1)
        delay_broadcast = delay[:, :, np.newaxis]
        width_broadcast = filter_width[:, :, np.newaxis]

        # Create filters
        filter_AC = np.exp(-(((t_broadcast - delay_broadcast) / width_broadcast) ** filter_order))
        filter_DC = np.exp(-((t_broadcast / width_broadcast) ** filter_order))

        # Apply filters
        St_AC = St * filter_AC
        St_DC = St * filter_DC

        # Transform back to frequency domain
        Sw_AC = self.Ft(St_AC, n_omega, n_fft)
        Sw_DC = self.Ft(St_DC, n_omega, n_fft)

        # Extract phase
        omega_broadcast = self.omega_axis.reshape(1, 1, -1)
        phase = np.angle(Sw_AC * np.exp(1j * omega_broadcast * delay_broadcast))

        # Calculate unknown spectrum
        a = np.abs(Sw_DC) - 2 * np.abs(Sw_AC)
        a[a < 0] = 0
        Su = (0.5 * (np.sqrt(np.abs(Sw_DC) + 2 * np.abs(Sw_AC)) + np.sqrt(a))) ** 2

        return phase, Su
