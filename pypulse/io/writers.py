"""Data writers for various file formats."""

from pathlib import Path

import numpy as np
import numpy.typing as npt


class DataWriter:
    """Writer for data files."""

    @staticmethod
    def save_sifast_data(
        folder_path: str | Path,
        wavelength: npt.NDArray[np.float64],
        interference: npt.NDArray[np.float64],
        unknown: npt.NDArray[np.float64] | None = None,
        reference: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Save SIFAST data to CSV files.

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
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        # Create template array
        n_pixels, n_wavelengths = interference.shape
        template = np.zeros((n_pixels + 4, n_wavelengths + 1))
        template[3, 1:] = wavelength

        # Save interference
        inter_data = template.copy()
        inter_data[4:, 1:] = interference
        np.savetxt(folder / "inter.csv", inter_data, delimiter=",")

        # Save unknown if provided
        if unknown is not None:
            unk_data = template.copy()
            unk_data[4:, 1:] = unknown
            np.savetxt(folder / "unk.csv", unk_data, delimiter=",")

        # Save reference if provided
        if reference is not None:
            ref_data = template.copy()
            ref_data[4:, 1:] = reference
            np.savetxt(folder / "ref.csv", ref_data, delimiter=",")


class ConfigWriter:
    """Writer for configuration files."""

    @staticmethod
    def save_reference_parameters(file_path: str | Path, parameters: dict) -> None:
        """Save reference parameters to JSON file."""
        import json

        with open(file_path, "w") as f:
            json.dump(parameters, f, indent=2)
