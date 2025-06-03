"""Data readers for various file formats."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt


class SpectrumReader:
    """Reader for spectrum data files."""

    def read_sifast_data(self, folder_path: str | Path, mode_acquire: str) -> dict[str, Any]:
        """
        Read SIFAST data from HDF5 or CSV files.

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

        # Check for HDF5 files first
        hdf5_files = list(folder.glob("*.h5")) + list(folder.glob("*.hdf5"))
        if hdf5_files:
            return self._read_hdf5_data(folder, mode_acquire)

        # Fall back to CSV if no HDF5 files found
        csv_files = list(folder.glob("*.csv"))
        if csv_files:
            return self._read_csv_data(folder, mode_acquire)

        raise FileNotFoundError(f"No data files (HDF5 or CSV) found in {folder}")

    def _read_hdf5_data(self, folder: Path, mode_acquire: str) -> dict[str, Any]:
        """Read SIFAST data from HDF5 format."""
        data = {}

        # Read interference
        inter_files = list(folder.glob("*inter*.h5")) + list(folder.glob("*inter*.hdf5"))
        if not inter_files:
            raise FileNotFoundError(f"No interference HDF5 file found in {folder}")

        with h5py.File(inter_files[0], "r") as f:
            data["wavelength"] = f["wavelength"][:]
            data["interference"] = f["image"][:]

        # Read unknown if needed
        if mode_acquire in ["double", "triple"]:
            unk_files = list(folder.glob("*unk*.h5")) + list(folder.glob("*unk*.hdf5"))
            if not unk_files:
                raise FileNotFoundError(f"No unknown HDF5 file found in {folder}")

            with h5py.File(unk_files[0], "r") as f:
                data["unknown"] = f["image"][:]

        # Read reference if needed
        if mode_acquire == "triple":
            ref_files = list(folder.glob("*ref*.h5")) + list(folder.glob("*ref*.hdf5"))
            if not ref_files:
                raise FileNotFoundError(f"No reference HDF5 file found in {folder}")

            with h5py.File(ref_files[0], "r") as f:
                data["reference"] = f["image"][:]

        return data

    def _read_csv_data(self, folder: Path, mode_acquire: str) -> dict[str, Any]:
        """Read SIFAST data from legacy CSV format."""
        data = {}

        # Read interference
        inter_files = list(folder.glob("*inter*.csv"))
        if not inter_files:
            raise FileNotFoundError(f"No interference CSV file found in {folder}")

        inter_data = np.loadtxt(inter_files[0], delimiter=",", skiprows=3)
        data["wavelength"] = inter_data[0, 1:]
        data["interference"] = inter_data[1:, 1:].astype(np.int32)

        # Read unknown if needed
        if mode_acquire in ["double", "triple"]:
            unk_files = list(folder.glob("*unk*.csv"))
            if not unk_files:
                raise FileNotFoundError(f"No unknown CSV file found in {folder}")
            data["unknown"] = np.loadtxt(unk_files[0], delimiter=",", skiprows=3)[1:, 1:].astype(np.int32)

        # Read reference if needed
        if mode_acquire == "triple":
            ref_files = list(folder.glob("*ref*.csv"))
            if not ref_files:
                raise FileNotFoundError(f"No reference CSV file found in {folder}")
            data["reference"] = np.loadtxt(ref_files[0], delimiter=",", skiprows=3)[1:, 1:].astype(np.int32)

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
