"""Self-Referenced Spectral Interferometry (SRSI) implementation."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.pulse import PulseBase
from ..io.readers import SpectrumReader


class SRSI(PulseBase):
    """Self-Referenced Spectral Interferometry processor."""

    def __init__(
        self,
        folder_path: str | Path,
        mode_acquire: str,
        wavelength_center: float,
        wavelength_width: float,
        n_omega: int,
        n_fft: int,
        n_iteration: int,
        method: str = "linear",
    ):
        """
        Initialize SRSI processor.

        Parameters
        ----------
        folder_path : str or Path
            Data folder path
        mode_acquire : str
            Acquisition mode ('single', 'double', 'triple')
        wavelength_center : float
            Center wavelength (nm)
        wavelength_width : float
            Wavelength range (nm)
        n_omega : int
            Number of frequency points
        n_fft : int
            FFT size
        n_iteration : int
            Number of phase retrieval iterations
        method : str
            Interpolation method
        """
        super().__init__()

        # Validate inputs
        if mode_acquire not in ["single", "double", "triple"]:
            raise ValueError(f"Invalid mode_acquire: {mode_acquire}")

        if method not in ["linear", "slinear", "quadratic", "cubic"]:
            raise ValueError(f"Invalid interpolation method: {method}")

        # Store parameters
        self.params = {
            "folder_path": str(folder_path),
            "mode_acquire": mode_acquire,
            "wavelength_center": wavelength_center,
            "wavelength_width": wavelength_width,
            "n_omega": n_omega,
            "n_fft": n_fft,
            "n_iteration": n_iteration,
            "method": method,
        }

        # Initialize
        self.omega_center = 2 * np.pi * self.SPEED_OF_LIGHT / wavelength_center
        self.n_omega = n_omega
        self.n_fft = n_fft
        self.row = [0]
        self.col = [0]

        # Read and process data
        self._process_data(folder_path, mode_acquire, wavelength_center, wavelength_width, method, n_iteration)

    def _process_data(
        self,
        folder_path: str | Path,
        mode_acquire: str,
        wavelength_center: float,
        wavelength_width: float,
        method: str,
        n_iteration: int,
    ) -> None:
        """Process spectrum data."""
        # Read spectra
        reader = SpectrumReader()
        spectra = reader.read_srsi_spectra(folder_path, mode_acquire)

        self.wavelength = spectra["wavelength"]

        # Resample spectra
        self.Sw_interference = self.resample_spectrum(
            spectra["interference"], wavelength_center, self.n_omega, wavelength_width, method
        ).reshape(1, 1, -1)

        if mode_acquire in ["double", "triple"]:
            self.Sw_unknown = self.resample_spectrum(
                spectra["unknown"], wavelength_center, self.n_omega, wavelength_width, method
            ).reshape(1, 1, -1)

        if mode_acquire == "triple":
            self.Sw_reference = self.resample_spectrum(
                spectra["reference"], wavelength_center, self.n_omega, wavelength_width, method
            ).reshape(1, 1, -1)

        # Ensure non-negative
        self.Sw_unknown[self.Sw_unknown < 0] = 0

        # Perform FTSI
        self.phase_diff, self.delay, Su = self.fourier_transform_spectral_interferometry(self.n_omega, self.n_fft)

        if mode_acquire == "single":
            self.Sw_unknown = Su

        # Retrieve phase
        self._retrieve_phase(n_iteration)

    def _retrieve_phase(self, n_iteration: int) -> None:
        """Iterative phase retrieval."""
        phase = np.unwrap(self.phase_diff, axis=2)
        phase = phase - phase[:, :, self.n_omega // 2]

        phase_diff_history = []

        for i in range(n_iteration - 1):
            # E-field in time domain
            Ew_unknown = np.sqrt(self.Sw_unknown) * np.exp(-1j * phase)
            Et_unknown = self.iFt(Ew_unknown, self.n_omega, self.n_fft)

            # Third-order nonlinearity
            Et_reference = Et_unknown * Et_unknown * Et_unknown.conjugate()
            Ew_reference = self.Ft(Et_reference, self.n_omega, self.n_fft)

            # Extract phase
            phase_reference = np.unwrap(-np.angle(Ew_reference), axis=2)
            phase_reference = phase_reference - phase_reference[:, :, self.n_omega // 2]

            # Store difference
            phase_diff_history.append(phase_reference - self.phase_diff - phase)

            # Update phase
            phase = phase_reference - self.phase_diff

        self.phase_diff_between_iteration = np.array(phase_diff_history).squeeze()
        self.phase = phase

    @property
    def Et(self) -> npt.NDArray[np.complex128]:
        """Electric field in time domain."""
        return self.iFt(np.sqrt(self.Sw_unknown) * np.exp(-1j * self.phase), self.n_omega, self.n_fft)

    @property
    def Et_FTL(self) -> npt.NDArray[np.complex128]:
        """Fourier transform limited electric field."""
        return self.iFt(np.sqrt(self.Sw_unknown), self.n_omega, self.n_fft)

    def to_dict(self) -> dict[str, Any]:
        """Export parameters as dictionary."""
        return self.params.copy()
