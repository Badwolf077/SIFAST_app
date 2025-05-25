import pathlib

import numpy as np

from .base import PulseBasic


class SRSI(PulseBasic):
    @property
    def Et(self) -> np.ndarray:
        """Returns the electric field in the time domain."""
        return self.iFt(np.sqrt(self.Sw_unknown) * np.exp(-1j * self.phase), self.n_omega, self.n_fft)

    @property
    def Et_FTL(self) -> np.ndarray:
        """Returns the fourier transform limit of the electric field."""
        return self.iFt(np.sqrt(self.Sw_unknown), self.n_omega, self.n_fft)

    @property
    def t_axis(self) -> np.ndarray:
        """Returns the time axis of the pulse."""
        return self._t_axis

    @t_axis.setter
    def t_axis(self, value: np.ndarray) -> None:
        """Sets the time axis of the pulse."""
        self._t_axis = value

    @property
    def omega_axis(self) -> np.ndarray:
        """Returns the frequency axis of the pulse."""
        return self._omega_axis

    @omega_axis.setter
    def omega_axis(self, value: np.ndarray) -> None:
        """Sets the frequency axis of the pulse."""
        self._omega_axis = value

    @property
    def wavelength_axis(self) -> np.ndarray:
        """Returns the wavelength axis of the pulse."""
        return self._wavelength_axis

    @wavelength_axis.setter
    def wavelength_axis(self, value: np.ndarray) -> None:
        """Sets the wavelength axis of the pulse."""
        self._wavelength_axis = value

    def __init__(
        self,
        folder_path: str | pathlib.Path,
        mode_acquire: str,
        wavelength_center: float,
        wavelength_width: float,
        n_omega: int,
        n_fft: int,
        n_iteration: int,
        method: str = "linear",
    ) -> None:
        params = locals()
        del params["self"]
        self.params = params

        if method not in ["linear", "slinear", "quadratic", "cubic"]:
            raise ValueError("method must be 'linear', 'slinear', 'quadratic', or 'cubic'")

        self.omega_center = 2 * np.pi * self.SPEED_OF_LIGHT / wavelength_center
        self.n_omega = n_omega
        self.n_fft = n_fft
        self.row = [0]
        self.col = [0]

        self._read_spectrum_from_txt(folder_path, mode_acquire)
        self.Sw_interference = self.resample_signal(
            self.Sw_interference, wavelength_center, n_omega, wavelength_width, method
        )

        match mode_acquire:
            case "double":
                self.Sw_unknown = self.resample_signal(
                    self.Sw_unknown, wavelength_center, n_omega, wavelength_width, method
                )
            case "triple":
                self.Sw_unknown = self.resample_signal(
                    self.Sw_unknown, wavelength_center, n_omega, wavelength_width, method
                )
                self.Sw_reference = self.resample_signal(
                    self.Sw_reference, wavelength_center, n_omega, wavelength_width, method
                )

        self.Sw_unknown[self.Sw_unknown < 0] = 0
        self.Sw_interference = self.Sw_interference.reshape(1, 1, -1)
        self.Sw_unknown = self.Sw_unknown.reshape(1, 1, -1)
        self.phase_diff, self.delay, Su = self.fourier_transform_spectral_interferometry(n_omega, n_fft)
        if mode_acquire == "single":
            self.Sw_unknown = Su

        self._retrieve_phase(n_iteration)

    def _read_spectrum_from_txt(self, folder_path: str, mode_acquire: str) -> None:
        """
        Reads the spectrum data from text files in the specified folder.

        Parameters:
        - folder_path: The path to the folder containing the text files.
        - mode_acquire: The acquisition mode (e.g., "single", "double", "triple").

        Returns:
        - None
        """
        matching_file = list(pathlib.Path(folder_path).glob("*inter*.txt"))
        if len(matching_file) == 0:
            raise FileNotFoundError(f"No interference spectrum found in {folder_path}")
        self.Sw_interference = np.loadtxt(matching_file[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[:, 1]
        self.wavelength = np.loadtxt(matching_file[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[:, 0]

        match mode_acquire:
            case "double":
                matching_file = list(pathlib.Path(folder_path).glob("*unk*.txt"))
                if len(matching_file) == 0:
                    raise FileNotFoundError(f"No unknown spectrum found in {folder_path}")
                self.Sw_unknown = np.loadtxt(matching_file[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[:, 1]
            case "triple":
                matching_file = list(pathlib.Path(folder_path).glob("*unk*.txt"))
                if len(matching_file) == 0:
                    raise FileNotFoundError(f"No unknown spectrum found in {folder_path}")
                self.Sw_unknown = np.loadtxt(matching_file[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[:, 1]

                matching_file = list(pathlib.Path(folder_path).glob("*ref*.txt"))
                if len(matching_file) == 0:
                    raise FileNotFoundError(f"No reference spectrum found in {folder_path}")
                self.Sw_reference = np.loadtxt(matching_file[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[
                    :, 1
                ]

    def _retrieve_phase(self, n_iteration: int) -> None:
        """
        Retrieves the phase information from the interference spectrum.

        Parameters:
        - n_iteration: The number of iterations for phase retrieval.

        Returns:
        - None
        """

        phase = self.phase_diff
        phase = np.unwrap(phase, axis=2)
        phase = phase - phase[:, :, self.n_omega // 2]
        phase_diff = np.full((n_iteration - 1, self.n_omega), np.nan)

        for i in range(n_iteration - 1):
            Ew_unknown = np.sqrt(self.Sw_unknown) * np.exp(-1j * phase)
            Et_unknown = self.iFt(Ew_unknown, self.n_omega, self.n_fft)
            Et_reference = Et_unknown * Et_unknown * Et_unknown.conjugate()
            Ew_reference = self.Ft(Et_reference, self.n_omega, self.n_fft)
            phase_reference = np.unwrap(-np.angle(Ew_reference), axis=2)
            phase_reference = phase_reference - phase_reference[:, :, self.n_omega // 2]
            phase_diff[i, :] = phase_reference - self.phase_diff - phase
            phase = phase_reference - self.phase_diff

        self.phase_diff_between_iteration = phase_diff
        self.phase = phase

    def to_dict(self) -> dict:
        """
        Returns the input parameters of SRSI initialization as a dictionary.
        """
        return self.params
