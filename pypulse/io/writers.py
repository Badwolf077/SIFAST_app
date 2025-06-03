"""Data writers for various file formats."""

import datetime
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt


class DataWriter:
    """Writer for data files."""

    @staticmethod
    def save_sifast_data(
        folder_path: str | Path,
        wavelength: npt.NDArray[np.float64],
        interference: npt.NDArray[np.int32],
        unknown: npt.NDArray[np.int32] | None = None,
        reference: npt.NDArray[np.int32] | None = None,
        save_format: str = "hdf5",
    ) -> None:
        """
        Save SIFAST data to HDF5 or CSV files.

        Parameters
        ----------
        folder_path : str or Path
            Output folder
        wavelength : array_like
            Wavelength array
        interference : array_like
            Interference image
        unknown : array_like, optional
            Unknown image
        reference : array_like, optional
            Reference image
        save_format : str
            Output format ('hdf5' or 'csv')
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        if save_format.lower() in ["hdf5", "h5"]:
            DataWriter._save_hdf5(folder, wavelength, interference, unknown, reference)
        elif save_format.lower() == "csv":
            DataWriter._save_csv(folder, wavelength, interference, unknown, reference)
        else:
            raise ValueError(f"Unsupported format: {save_format}. Use 'hdf5' or 'csv'")

    @staticmethod
    def _save_hdf5(
        folder: Path,
        wavelength: npt.NDArray[np.float64],
        interference: npt.NDArray[np.int32],
        unknown: npt.NDArray[np.int32] | None = None,
        reference: npt.NDArray[np.int32] | None = None,
    ) -> None:
        """Save data in HDF5 format."""
        # Save interference
        with h5py.File(folder / "inter.h5", "w") as f:
            f.create_dataset("wavelength", data=wavelength, compression="gzip")
            f.create_dataset("image", data=interference, compression="gzip")

            # Add metadata
            f.attrs["description"] = "SIFAST interference data"
            f.attrs["wavelength_unit"] = "nm"
            f.attrs["timestamp"] = datetime.datetime.now().isoformat()

        # Save unknown if provided
        if unknown is not None:
            with h5py.File(folder / "unk.h5", "w") as f:
                f.create_dataset("wavelength", data=wavelength, compression="gzip")
                f.create_dataset("image", data=unknown, compression="gzip")
                f.attrs["description"] = "SIFAST unknown data"
                f.attrs["wavelength_unit"] = "nm"
                f.attrs["timestamp"] = datetime.datetime.now().isoformat()

        # Save reference if provided
        if reference is not None:
            with h5py.File(folder / "ref.h5", "w") as f:
                f.create_dataset("wavelength", data=wavelength, compression="gzip")
                f.create_dataset("image", data=reference, compression="gzip")
                f.attrs["description"] = "SIFAST reference data"
                f.attrs["wavelength_unit"] = "nm"
                f.attrs["timestamp"] = datetime.datetime.now().isoformat()

    @staticmethod
    def _save_csv(
        folder: Path,
        wavelength: npt.NDArray[np.float64],
        interference: npt.NDArray[np.int32],
        unknown: npt.NDArray[np.int32] | None = None,
        reference: npt.NDArray[np.int32] | None = None,
    ) -> None:
        """Save data in legacy CSV format."""
        # Create template array
        n_pixels, n_wavelengths = interference.shape
        template_1 = np.zeros((4, n_wavelengths + 1))
        template_2 = np.zeros((n_pixels, n_wavelengths + 1))
        template_1[3, 1:] = wavelength

        # Save interference
        inter_data = template_2.copy()
        inter_data[:, 1:] = interference
        with open(folder / "inter.csv", "wb") as f:
            np.savetxt(f, template_1, delimiter=",")
            np.savetxt(f, inter_data, fmt="%d", delimiter=",")

        # Save unknown if provided
        if unknown is not None:
            unk_data = template_2.copy()
            unk_data[:, 1:] = unknown
            with open(folder / "unk.csv", "wb") as f:
                np.savetxt(f, template_1, delimiter=",")
                np.savetxt(f, unk_data, fmt="%d", delimiter=",")

        # Save reference if provided
        if reference is not None:
            ref_data = template_2.copy()
            ref_data[:, 1:] = reference
            with open(folder / "ref.csv", "wb") as f:
                np.savetxt(f, template_1, delimiter=",")
                np.savetxt(f, ref_data, fmt="%d", delimiter=",")


class ConfigWriter:
    """Writer for configuration files."""

    @staticmethod
    def save_reference_parameters(file_path: str | Path, parameters: dict) -> None:
        """Save reference parameters to JSON file."""
        import json

        with open(file_path, "w") as f:
            json.dump(parameters, f, indent=2)
