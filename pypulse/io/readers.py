"""Data readers for various file formats."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


class SpectrumReader:
    """Reader for spectrum data files."""

    def read_sifast_data(self, folder_path: str | Path, mode_acquire: str) -> dict[str, Any]:
        """
        Read SIFAST data from CSV files.

        Parameters
        ----------
        folder_path : str or Path
            Folder containing data files
        mode_acquire : str
            Acquisition mode

        Returns
        -------
        dict
            Dictionary with 'interference', 'wavelength', and optionally
            'unknown' and 'reference' arrays
        """
        folder = Path(folder_path)
        data = {}

        # Read interference
        inter_files = list(folder.glob("*inter*.csv"))
        if not inter_files:
            raise FileNotFoundError(f"No interference spectrum found in {folder}")

        inter_data = np.loadtxt(inter_files[0], delimiter=",", skiprows=3)
        data["wavelength"] = inter_data[0, 1:]
        data["interference"] = inter_data[1:, 1:]

        # Read unknown if needed
        if mode_acquire in ["double", "triple"]:
            unk_files = list(folder.glob("*unk*.csv"))
            if not unk_files:
                raise FileNotFoundError(f"No unknown spectrum found in {folder}")
            data["unknown"] = np.loadtxt(unk_files[0], delimiter=",", skiprows=3)[1:, 1:]

        # Read reference if needed
        if mode_acquire == "triple":
            ref_files = list(folder.glob("*ref*.csv"))
            if not ref_files:
                raise FileNotFoundError(f"No reference spectrum found in {folder}")
            data["reference"] = np.loadtxt(ref_files[0], delimiter=",", skiprows=3)[1:, 1:]

        return data

    def read_srsi_spectra(self, folder_path: str | Path, mode_acquire: str) -> dict[str, npt.NDArray[np.float64]]:
        """
        Read SRSI spectrum data from text files.

        Parameters
        ----------
        folder_path : str or Path
            Folder containing spectrum files
        mode_acquire : str
            Acquisition mode

        Returns
        -------
        dict
            Dictionary with spectrum arrays
        """
        folder = Path(folder_path)
        spectra = {}

        # Read interference spectrum
        inter_files = list(folder.glob("*inter*.txt"))
        if not inter_files:
            raise FileNotFoundError(f"No interference spectrum found in {folder}")

        inter_data = np.loadtxt(inter_files[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")
        spectra["wavelength"] = inter_data[:, 0]
        spectra["interference"] = inter_data[:, 1]

        # Read unknown spectrum
        if mode_acquire in ["double", "triple"]:
            unk_files = list(folder.glob("*unk*.txt"))
            if not unk_files:
                raise FileNotFoundError(f"No unknown spectrum found in {folder}")
            spectra["unknown"] = np.loadtxt(unk_files[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[:, 1]

        # Read reference spectrum
        if mode_acquire == "triple":
            ref_files = list(folder.glob("*ref*.txt"))
            if not ref_files:
                raise FileNotFoundError(f"No reference spectrum found in {folder}")
            spectra["reference"] = np.loadtxt(ref_files[0], delimiter="\t", skiprows=14, encoding="iso-8859-1")[:, 1]

        return spectra


class ConfigReader:
    """Reader for configuration files."""

    @staticmethod
    def read_reference_parameters(file_path: str | Path) -> dict[str, float]:
        """Read reference parameters from JSON file."""
        import json

        with open(file_path) as f:
            return json.load(f)

    @staticmethod
    def read_fiber_calibration(file_path: str | Path) -> npt.NDArray[np.float64]:
        """Read fiber calibration data."""
        return np.loadtxt(file_path, delimiter=",")
