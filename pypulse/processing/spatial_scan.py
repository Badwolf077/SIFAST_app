"""Spatial scanning enhancement for SIFAST to improve spatial resolution."""

import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from .sifast import SIFAST


class SpatialScanner:
    """Handles spatial scanning and merging of SIFAST measurements."""

    def __init__(self, unwrap_before_merge: bool = True, n_neighbors: int = 3):
        """
        Initialize spatial scanner.

        Parameters
        ----------
        unwrap_before_merge : bool
            Whether to apply 2D phase unwrapping before merging
        n_neighbors : int
            Number of nearest neighbors for phase calibration
        """
        self.unwrap_before_merge = unwrap_before_merge
        self.n_neighbors = n_neighbors

    def merge_sifast_measurements(
        self,
        pulses: list["SIFAST"],
        calibration_index: tuple[int, int] | None = None,
        calibration_point: tuple[float, float] | None = None,
    ) -> "SIFAST":
        """
        Merge multiple SIFAST measurements from different positions.

        Parameters
        ----------
        pulses : List[SIFAST]
            List of SIFAST instances to merge
        calibration_index : Tuple[int, int], optional
            (row, col) index in first pulse for phase calibration
        calibration_point : Tuple[float, float], optional
            (x, y) spatial position for phase calibration

        Returns
        -------
        SIFAST
            Merged SIFAST instance
        """
        if len(pulses) < 2:
            raise ValueError("Need at least 2 pulses to merge")

        # Step 1: Collect all spatial points
        all_x, all_y = self._collect_all_points(pulses)

        # Step 2: Create merged spatial arrays
        x_merged, y_merged, x_matrix, y_matrix = self._create_merged_arrays(all_x, all_y)

        # Step 3: Create and fill merged data arrays using vectorized operations
        merged_data = self._create_and_fill_merged_arrays(pulses, x_matrix, y_matrix, pulses[0].n_omega)

        # Step 4: Update row and col indices for valid measurements
        row_merged, col_merged = self._get_valid_indices(merged_data["time_interval"])

        # Step 5: Merge phase with spatial calibration
        phase_merged = self._merge_phase_with_calibration(
            pulses, x_matrix, y_matrix, merged_data["Sw_unknown"], calibration_index, calibration_point
        )

        # Step 6: Create merged SIFAST instance
        merged_pulse = self._create_merged_instance(
            pulses[0], x_merged, y_merged, x_matrix, y_matrix, row_merged, col_merged, merged_data, phase_merged
        )

        return merged_pulse

    def _collect_all_points(self, pulses: list["SIFAST"]) -> tuple[np.ndarray, np.ndarray]:
        """Collect all unique x and y coordinates from all pulses."""
        all_x_coords = []
        all_y_coords = []

        for pulse in pulses:
            all_x_coords.append(pulse.x_axis)
            all_y_coords.append(pulse.y_axis)

        # Concatenate and get unique values
        all_x = np.unique(np.concatenate(all_x_coords))
        all_y = np.unique(np.concatenate(all_y_coords))

        return all_x, all_y

    def _create_merged_arrays(
        self, all_x: np.ndarray, all_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create merged spatial arrays from all unique coordinates."""
        # Sort arrays
        x_merged = np.sort(all_x)
        y_merged = np.sort(all_y)

        # Create meshgrid
        x_matrix, y_matrix = np.meshgrid(x_merged, y_merged)

        return x_merged, y_merged, x_matrix, y_matrix

    def _create_and_fill_merged_arrays(
        self, pulses: list["SIFAST"], x_matrix: np.ndarray, y_matrix: np.ndarray, n_omega: int
    ) -> dict:
        """Create and fill merged data arrays using fully vectorized operations."""
        ny, nx = x_matrix.shape

        # Initialize arrays
        merged_data = {
            "Sw_unknown": np.full((ny, nx, n_omega), np.nan),
            "time_interval": np.full((ny, nx), np.nan),
            "pulse_front": np.full((ny, nx), np.nan),
        }

        # Fully vectorized filling from each pulse
        for pulse in pulses:
            # Get flat indices for the pulse data
            pulse_shape = pulse.x_matrix.shape
            flat_indices = np.arange(pulse.x_matrix.size)
            row_indices, col_indices = np.unravel_index(flat_indices, pulse_shape)

            # Get coordinates
            x_pulse = pulse.x_matrix[row_indices, col_indices]
            y_pulse = pulse.y_matrix[row_indices, col_indices]

            # Find indices in merged array
            x_indices = np.searchsorted(x_matrix[0, :], x_pulse)
            y_indices = np.searchsorted(y_matrix[:, 0], y_pulse)

            # Direct assignment without loops
            merged_data["Sw_unknown"][y_indices, x_indices, :] = pulse.Sw_unknown[row_indices, col_indices, :]
            merged_data["time_interval"][y_indices, x_indices] = pulse.time_interval[row_indices, col_indices]
            merged_data["pulse_front"][y_indices, x_indices] = pulse.pulse_front[row_indices, col_indices]

        return merged_data

    def _get_valid_indices(self, time_interval: np.ndarray) -> tuple[list[int], list[int]]:
        """Get row and col indices for valid (non-NaN) measurements."""
        valid_indices = np.where(~np.isnan(time_interval))
        return list(valid_indices[0]), list(valid_indices[1])

    def _merge_phase_with_calibration(
        self,
        pulses: list["SIFAST"],
        x_matrix: np.ndarray,
        y_matrix: np.ndarray,
        Sw_unknown_merged: np.ndarray,
        calibration_index: tuple[int, int] | None,
        calibration_point: tuple[float, float] | None,
    ) -> np.ndarray:
        """Merge phase arrays with spatial interpolation calibration."""
        ny, nx, n_omega = Sw_unknown_merged.shape
        phase_merged = np.full((ny, nx, n_omega), np.nan)
        center_freq_idx = n_omega // 2

        # Process first pulse as reference
        pulse_ref = pulses[0]
        phase_ref = self._prepare_phase(pulse_ref.phase, pulse_ref.Sw_unknown)

        # Fully vectorized copy of reference phase
        pulse_shape = pulse_ref.x_matrix.shape
        flat_indices = np.arange(pulse_ref.x_matrix.size)
        row_indices, col_indices = np.unravel_index(flat_indices, pulse_shape)

        x_pulse = pulse_ref.x_matrix[row_indices, col_indices]
        y_pulse = pulse_ref.y_matrix[row_indices, col_indices]

        x_indices = np.searchsorted(x_matrix[0, :], x_pulse)
        y_indices = np.searchsorted(y_matrix[:, 0], y_pulse)

        phase_merged[y_indices, x_indices, :] = phase_ref[row_indices, col_indices, :]

        # Process remaining pulses with calibration
        for pulse_idx, pulse_offset in enumerate(pulses[1:], 1):
            phase_offset = self._prepare_phase(pulse_offset.phase, pulse_offset.Sw_unknown)

            # Determine calibration position for this pulse_offset
            if calibration_index is not None:
                # Use specified index from pulse_offset
                r0, c0 = calibration_index
                calib_x = pulse_offset.x_matrix[r0, c0]
                calib_y = pulse_offset.y_matrix[r0, c0]
            elif calibration_point is not None:
                # Use specified spatial point
                calib_x, calib_y = calibration_point
                ix_closest = np.argmin(np.abs(pulse_offset.x_axis - calib_x))
                iy_closest = np.argmin(np.abs(pulse_offset.y_axis - calib_y))
                calib_x = pulse_offset.x_axis[ix_closest]
                calib_y = pulse_offset.y_axis[iy_closest]
            else:
                # Use center of pulse_offset
                calib_x = pulse_offset.x_axis[pulse_offset.x_axis.size // 2]
                calib_y = pulse_offset.y_axis[pulse_offset.y_axis.size // 2]

            # Find phase offset using spatial interpolation
            offset = self._calculate_phase_offset_interpolated(
                pulse_ref,
                phase_ref,
                phase_merged,
                pulse_offset,
                phase_offset,
                calib_x,
                calib_y,
                center_freq_idx,
                x_matrix,
                y_matrix,
            )

            # Fully vectorized application of offset and copy to merged array
            pulse_shape = pulse_offset.x_matrix.shape
            flat_indices = np.arange(pulse_offset.x_matrix.size)
            row_indices, col_indices = np.unravel_index(flat_indices, pulse_shape)

            x_pulse = pulse_offset.x_matrix[row_indices, col_indices]
            y_pulse = pulse_offset.y_matrix[row_indices, col_indices]

            x_indices = np.searchsorted(x_matrix[0, :], x_pulse)
            y_indices = np.searchsorted(y_matrix[:, 0], y_pulse)

            # Apply offset to all points and copy
            phase_merged[y_indices, x_indices, :] = phase_offset[row_indices, col_indices, :] + offset

        return phase_merged

    def _prepare_phase(self, phase: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Prepare phase array with optional 2D unwrapping."""
        phase_prep = phase.copy()

        if self.unwrap_before_merge:
            from skimage.restoration import unwrap_phase

            # Apply 2D unwrapping for each frequency
            for freq_idx in range(phase.shape[2]):
                # Only unwrap where we have valid intensity
                mask = ~np.isnan(intensity[:, :, freq_idx]) & (intensity[:, :, freq_idx] > 0)
                if np.any(mask):
                    phase_slice = phase_prep[:, :, freq_idx]
                    phase_slice[mask] = unwrap_phase(phase_slice[mask].astype(np.float32))
                    phase_prep[:, :, freq_idx] = phase_slice

        return phase_prep

    def _calculate_phase_offset_interpolated(
        self,
        pulse_ref: "SIFAST",
        phase_ref: np.ndarray,
        phase_merged: np.ndarray,
        pulse_offset: "SIFAST",
        phase_offset: np.ndarray,
        calib_x: float,
        calib_y: float,
        freq_idx: int,
        x_matrix: np.ndarray,
        y_matrix: np.ndarray,
    ) -> float:
        """
        Calculate phase offset using spatial interpolation.

        Uses k-nearest neighbors in the already merged data to estimate the phase
        at the calibration point, then calculates offset for the new pulse.
        """
        # Get phase value from pulse_offset at calibration position
        # Find nearest point in pulse_offset to calibration position
        distances = np.sqrt((pulse_offset.x_matrix - calib_x) ** 2 + (pulse_offset.y_matrix - calib_y) ** 2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        phase_offset_at_calib = phase_offset[min_idx[0], min_idx[1], freq_idx]

        # Find k nearest neighbors in merged data to estimate phase at calibration point
        valid_mask = ~np.isnan(phase_merged[:, :, freq_idx])
        if not np.any(valid_mask):
            # No valid points in merged data yet, use reference pulse
            return self._calculate_from_reference_only(
                pulse_ref, phase_ref, calib_x, calib_y, freq_idx, phase_offset_at_calib
            )

        # Get valid points from merged data
        valid_indices = np.where(valid_mask)
        x_valid = x_matrix[valid_indices]
        y_valid = y_matrix[valid_indices]
        phase_valid = phase_merged[valid_indices[0], valid_indices[1], freq_idx]

        # Find k nearest neighbors
        points = np.column_stack([x_valid, y_valid])
        tree = cKDTree(points)
        k = min(self.n_neighbors, len(points))
        distances, indices = tree.query([calib_x, calib_y], k=k)

        if isinstance(distances, float):  # Only one point
            distances = np.array([distances])
            indices = np.array([indices])

        # Calculate weighted average phase at calibration point
        weights = 1 / (distances + 1e-10)
        weights = weights / np.sum(weights)
        phase_merged_at_calib = np.sum(phase_valid[indices] * weights)

        # Calculate offset
        return phase_merged_at_calib - phase_offset_at_calib

    def _calculate_from_reference_only(
        self,
        pulse_ref: "SIFAST",
        phase_ref: np.ndarray,
        calib_x: float,
        calib_y: float,
        freq_idx: int,
        phase_offset_at_calib: float,
    ) -> float:
        """Fallback calculation using only reference pulse."""
        valid_mask = ~np.isnan(pulse_ref.time_interval)
        if not np.any(valid_mask):
            return 0.0

        x_valid = pulse_ref.x_matrix[valid_mask]
        y_valid = pulse_ref.y_matrix[valid_mask]
        phase_valid = phase_ref[valid_mask, freq_idx]

        # Find nearest neighbors
        points = np.column_stack([x_valid, y_valid])
        tree = cKDTree(points)
        k = min(self.n_neighbors, len(points))
        distances, indices = tree.query([calib_x, calib_y], k=k)

        if isinstance(distances, float):
            distances = np.array([distances])
            indices = np.array([indices])

        # Weighted average
        weights = 1 / (distances + 1e-10)
        weights = weights / np.sum(weights)
        phase_ref_at_calib = np.sum(phase_valid[indices] * weights)

        return phase_ref_at_calib - phase_offset_at_calib

    def _create_merged_instance(
        self,
        reference_pulse: "SIFAST",
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        x_matrix: np.ndarray,
        y_matrix: np.ndarray,
        row: list[int],
        col: list[int],
        merged_data: dict,
        phase: np.ndarray,
    ) -> "SIFAST":
        """Create merged SIFAST instance."""
        # Create new instance by copying reference
        merged = type(reference_pulse).__new__(type(reference_pulse))

        # Copy non-spatial attributes
        for attr in [
            "omega_center",
            "n_omega",
            "n_fft",
            "wavelength_axis",
            "omega_axis",
            "t_axis",
            "wavelength",
            "rp",
            "SPEED_OF_LIGHT",
        ]:
            if hasattr(reference_pulse, attr):
                setattr(merged, attr, getattr(reference_pulse, attr))

        # Set spatial attributes
        merged.x_axis = x_axis
        merged.y_axis = y_axis
        merged.x_matrix = x_matrix
        merged.y_matrix = y_matrix
        merged.number_x = len(x_axis)
        merged.number_y = len(y_axis)
        merged.shape = (merged.number_y, merged.number_x)

        # Set fiber indices
        merged.row = row
        merged.col = col

        # Set data arrays
        merged.Sw_unknown = merged_data["Sw_unknown"]
        merged.time_interval = merged_data["time_interval"]
        merged.pulse_front = merged_data["pulse_front"]
        merged.phase = phase

        # Set other required attributes with NaN
        merged.Sw_interference = np.full_like(merged.Sw_unknown, np.nan)
        merged.phase_diff_with_sphere = np.full_like(merged.phase, np.nan)
        merged.phase_diff = np.full_like(merged.phase, np.nan)
        merged.pulse_front_reference = np.full_like(merged.pulse_front, np.nan)

        # Update params to reflect merging
        merged.params = reference_pulse.params.copy()
        merged.params["spatial_scan"] = True
        merged.params["n_merged_pulses"] = len(row)

        return merged


def merge_spatial_scans(
    pulses: list["SIFAST"],
    calibration_index: tuple[int, int] | None = None,
    calibration_point: tuple[float, float] | None = None,
    unwrap_before_merge: bool = False,
    n_neighbors: int = 3,
) -> "SIFAST":
    """
    Convenience function to merge multiple SIFAST spatial scans.

    Parameters
    ----------
    pulses : List[SIFAST]
        List of SIFAST instances from different positions
    calibration_index : Tuple[int, int], optional
        (row, col) index in first pulse for phase calibration
    calibration_point : Tuple[float, float], optional
        (x, y) spatial position for phase calibration
    unwrap_before_merge : bool
        Whether to apply 2D phase unwrapping before merging
    n_neighbors : int
        Number of nearest neighbors for phase interpolation

    Returns
    -------
    SIFAST
        Merged SIFAST instance with improved spatial resolution

    Notes
    -----
    Phase calibration method:
    1. For the first pulse, phase is used as reference
    2. For subsequent pulses:
       - Find k nearest neighbors in reference pulse to calibration point
       - Calculate weighted average phase at that point (weighted by intensity/distance)
       - Find phase in new pulse at nearest point to calibration position
       - Calculate offset = reference_phase - new_pulse_phase
       - Apply this offset to all points in the new pulse
    """
    scanner = SpatialScanner(unwrap_before_merge=unwrap_before_merge, n_neighbors=n_neighbors)
    return scanner.merge_sifast_measurements(pulses, calibration_index, calibration_point)
