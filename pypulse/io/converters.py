"""Data format converters for pypulse."""

from pathlib import Path
from typing import Literal

from .readers import SpectrumReader
from .writers import DataWriter


def detect_acquisition_mode(folder_path: Path) -> str:
    """
    Automatically detect acquisition mode based on CSV files present.

    Parameters
    ----------
    folder_path : Path
        Folder containing CSV files

    Returns
    -------
    str
        Detected acquisition mode ('single', 'double', or 'triple')
    """
    inter_files = list(folder_path.glob("*inter*.csv"))
    unk_files = list(folder_path.glob("*unk*.csv"))
    ref_files = list(folder_path.glob("*ref*.csv"))

    if inter_files and ref_files and unk_files:
        return "triple"
    elif inter_files and unk_files:
        return "double"
    elif inter_files:
        return "single"
    else:
        raise FileNotFoundError("No interference CSV file found")


def convert_csv_to_hdf5(
    folder_path: str | Path,
    mode_acquire: Literal["single", "double", "triple", "auto"] = "auto",
    remove_csv: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Convert CSV files in a folder to HDF5 format.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing CSV files
    mode_acquire : str
        Acquisition mode ('single', 'double', 'triple', or 'auto' for automatic detection)
    remove_csv : bool
        Whether to remove CSV files after conversion
    verbose : bool
        Whether to print status messages

    Returns
    -------
    bool
        True if conversion successful, False otherwise
    """
    folder_path = Path(folder_path)
    reader = SpectrumReader()

    # Check if HDF5 files already exist
    hdf5_files = list(folder_path.glob("*.h5")) + list(folder_path.glob("*.hdf5"))
    if hdf5_files:
        if verbose:
            print(f"HDF5 files already exist in {folder_path}, skipping...")
        return False

    # Auto-detect mode if requested
    if mode_acquire == "auto":
        try:
            mode_acquire = detect_acquisition_mode(folder_path)
            if verbose:
                print(f"Auto-detected acquisition mode: {mode_acquire}")
        except FileNotFoundError as e:
            if verbose:
                print(f"Error detecting acquisition mode: {e}")
            return False

    # Read CSV data
    try:
        data = reader._read_csv_data(folder_path, mode_acquire)
    except FileNotFoundError as e:
        if verbose:
            print(f"Error reading CSV files: {e}")
        return False

    # Save as HDF5
    writer = DataWriter()
    writer.save_sifast_data(
        folder_path,
        data["wavelength"],
        data["interference"],
        data.get("unknown"),
        data.get("reference"),
        save_format="hdf5",
    )

    if verbose:
        print(f"Successfully converted {mode_acquire} mode CSV files to HDF5 in {folder_path}")

    # Remove CSV files if requested
    if remove_csv:
        csv_files = list(folder_path.glob("*.csv"))
        # Only remove data CSV files, not configuration files
        for csv_file in csv_files:
            if any(x in csv_file.name for x in ["inter", "unk", "ref"]):
                csv_file.unlink()
                if verbose:
                    print(f"Removed {csv_file.name}")

    return True


def batch_convert_csv_to_hdf5(
    root_path: str | Path,
    mode_acquire: Literal["single", "double", "triple", "auto"] = "auto",
    remove_csv: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Convert all CSV files in subdirectories to HDF5 format.

    Parameters
    ----------
    root_path : str or Path
        Root directory to search for CSV files
    mode_acquire : str
        Acquisition mode ('single', 'double', 'triple', or 'auto' for automatic detection)
    remove_csv : bool
        Whether to remove CSV files after conversion
    verbose : bool
        Whether to print status messages

    Returns
    -------
    tuple[int, int]
        Number of (successful conversions, total folders found)
    """
    root_path = Path(root_path)

    # Find all folders containing inter.csv files
    inter_csv_files = list(root_path.rglob("*inter*.csv"))
    folders = set(f.parent for f in inter_csv_files)

    if not folders:
        if verbose:
            print(f"No SIFAST CSV files found in {root_path}")
        return 0, 0

    if verbose:
        print(f"Found {len(folders)} folders with CSV files")

    # Group folders by detected mode if auto
    if mode_acquire == "auto" and verbose:
        mode_stats = {"single": 0, "double": 0, "triple": 0}
        for folder in folders:
            try:
                detected_mode = detect_acquisition_mode(folder)
                mode_stats[detected_mode] += 1
            except:  # noqa: E722
                pass
        print(
            f"Detected modes: single={mode_stats['single']}, "
            f"double={mode_stats['double']}, "
            f"triple={mode_stats['triple']}"
        )

    successful = 0
    for folder in sorted(folders):
        if verbose:
            print(f"\nProcessing {folder}")
        if convert_csv_to_hdf5(folder, mode_acquire, remove_csv, verbose):
            successful += 1

    return successful, len(folders)
