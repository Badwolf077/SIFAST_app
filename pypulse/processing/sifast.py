"""Spatially resolved Interferometric Field Autocorrelation Scan Technique (SIFAST)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from ..core.pulse import PulseBase
from ..fiber.registry import get_fiber_array, get_fiber_array_config
from ..io.logging import update_processing_log
from ..io.readers import SpectrumReader
from ..io.writers import DataWriter
from ..visualization.plotting import SIFASTVisualizer
from .srsi import SRSI


class SIFAST(PulseBase):
    """SIFAST pulse characterization processor."""

    def __init__(
        self,
        mode_input: str,
        mode_acquire: str,
        gate_noise_intensity: float,
        wavelength_center: float,
        wavelength_width: float,
        n_omega: int,
        n_fft: int,
        mode_fiber_position: str = "calibration",
        method: str = "linear",
        fiber_array_id: str = "default_14x14",
        dx: float = 0,
        dy: float = 0,
        reference_pulse: SRSI | None = None,
        as_calibration: bool = False,
        config_folder_path: str | Path | None = None,
        delay_min: float | None = None,
        **kwargs,
    ):
        """
        Initialize SIFAST processor.

        Parameters
        ----------
        mode_input : str
            Input mode ('read' or 'acquire')
        mode_acquire : str
            Acquisition mode ('single', 'double', 'triple')
        gate_noise_intensity : float
            Noise gate threshold
        wavelength_center : float
            Center wavelength (nm)
        wavelength_width : float
            Wavelength range (nm)
        n_omega : int
            Number of frequency points
        n_fft : int
            FFT size
        mode_fiber_position : str
            Fiber position mode ('calibration' or 'calculation')
        method_interpolation : str
            Interpolation method
        fiber_array_id : str
            Fiber array configuration ID
        dx, dy : float
            Fiber array position offsets
        reference_pulse : SRSI, optional
            Reference pulse for phase compensation
        as_calibration : bool
            Whether to perform calibration
        config_folder_path : str or Path, optional
            Configuration folder path
        delay_min : float, optional
            Minimum delay for peak detection
        **kwargs
            Additional arguments for data input
        """
        super().__init__()

        # Store all parameters
        self.params = self._collect_parameters(locals())
        try:
            fiber_array_config = get_fiber_array_config(fiber_array_id)
            self.params["fiber_array_config"] = fiber_array_config
        except:  # noqa: E722
            # If we can't get the config, at least keep the ID
            self.params["fiber_array_config"] = {"id": fiber_array_id}

        try:
            self._validate_inputs(mode_input, mode_acquire, mode_fiber_position, method)

            # Initialize core parameters
            self.omega_center = 2 * np.pi * self.SPEED_OF_LIGHT / wavelength_center
            self.n_omega = n_omega
            self.n_fft = n_fft

            # Initialize fiber array
            fiber_array = get_fiber_array(fiber_array_id, dx, dy)
            self._apply_fiber_array_properties(fiber_array)

            # Process based on input mode
            if mode_input == "read":
                self._process_read_mode(kwargs, mode_acquire, config_folder_path)
            else:  # acquire
                self._process_acquire_mode(kwargs, mode_acquire, config_folder_path)

            # Common processing steps
            self._process_fiber_positions(
                gate_noise_intensity, mode_fiber_position, mode_acquire, self.final_config_path
            )
            self._resample_and_process_spectra(wavelength_center, wavelength_width, method, mode_acquire)
            self._perform_interferometry(
                n_omega,
                n_fft,
                delay_min,
                mode_acquire,
                as_calibration,
                reference_pulse,
                method,
                wavelength_center,
            )

            # Log success for read mode
            if mode_input == "read" and hasattr(self, "_folder_path"):
                if hasattr(self, "_copy_config_after_processing") and self._copy_config_after_processing:
                    import shutil

                    local_config_path = Path(self._folder_path) / "config"
                    shutil.copytree(config_folder_path, local_config_path)

                update_processing_log(
                    self._folder_path,
                    "SUCCESS",
                    self.params,
                    f"Data processed using config from '{self.final_config_path}'",
                )

        except Exception as e:
            if mode_input == "read" and hasattr(self, "_folder_path"):
                update_processing_log(self._folder_path, "FAILURE", self.params, str(e))
            raise

    def _collect_parameters(self, local_vars: dict[str, Any]) -> dict[str, Any]:
        """Collect and clean parameters."""
        params = local_vars.copy()
        params.pop("self")
        params.pop("__class__")
        if "kwargs" in params:
            kw = params.pop("kwargs")
            params.update(kw)
        return params

    def _validate_inputs(
        self, mode_input: str, mode_acquire: str, mode_fiber_position: str, method_interpolation: str
    ) -> None:
        """Validate input parameters."""
        if mode_input not in ["read", "acquire"]:
            raise ValueError("mode_input must be 'read' or 'acquire'")
        if mode_acquire not in ["single", "double", "triple"]:
            raise ValueError("mode_acquire must be 'single', 'double', or 'triple'")
        if mode_fiber_position not in ["calibration", "calculation"]:
            raise ValueError("mode_fiber_position must be 'calibration' or 'calculation'")
        if method_interpolation not in ["linear", "slinear", "quadratic", "cubic"]:
            raise ValueError(f"Invalid interpolation method: {method_interpolation}")

    def _apply_fiber_array_properties(self, fiber_array) -> None:
        """Apply fiber array properties to instance."""
        properties = fiber_array.get_properties()
        for key, value in properties.items():
            if key != "config":  # Don't store internal config
                setattr(self, key, value)

    def _process_read_mode(
        self, kwargs: dict[str, Any], mode_acquire: str, config_folder_path: str | Path | None
    ) -> None:
        """Process read mode input."""
        folder_path = kwargs.pop("folder_path", None)
        if folder_path is None:
            raise ValueError("folder_path must be provided when mode_input is 'read'")

        self._folder_path = folder_path

        # Read data
        reader = SpectrumReader()
        data = reader.read_sifast_data(folder_path, mode_acquire)
        self.image_interference = data["interference"]
        self.wavelength = data["wavelength"]

        if mode_acquire in ["double", "triple"]:
            self.image_unknown = data.get("unknown")
        if mode_acquire == "triple":
            self.image_reference = data.get("reference")

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        # Handle configuration
        self._setup_read_config(folder_path, config_folder_path)

    def _setup_read_config(self, folder_path: str | Path, config_folder_path: str | Path | None) -> None:
        """Set up configuration for read mode."""
        folder_path = Path(folder_path)
        local_config_path = folder_path / "config"

        if config_folder_path is not None:
            config_folder_path = Path(config_folder_path)
            if local_config_path.exists():
                raise ValueError(f"Configuration conflict in {folder_path}")
            if not config_folder_path.exists():
                raise FileNotFoundError(f"External config path does not exist: {config_folder_path}")
            self.final_config_path = config_folder_path
            self._copy_config_after_processing = True
        else:
            if local_config_path.exists():
                self.final_config_path = local_config_path
            else:
                raise FileNotFoundError(f"Missing config in {folder_path} and no external path provided")

    def _process_acquire_mode(
        self, kwargs: dict[str, Any], mode_acquire: str, config_folder_path: str | Path | None
    ) -> None:
        """Process acquire mode input."""
        # Validate required parameters
        self.image_interference = kwargs.pop("image_interference", None)
        if self.image_interference is None:
            raise ValueError("image_interference must be provided when mode_input is 'acquire'")

        self.wavelength = kwargs.pop("wavelength", None)
        if self.wavelength is None:
            raise ValueError("wavelength must be provided when mode_input is 'acquire'")

        # Get mode-specific images
        if mode_acquire == "double":
            self.image_unknown = kwargs.pop("image_unknown", None)
            if self.image_unknown is None:
                raise ValueError("image_unknown must be provided when mode_acquire is 'double'")
        elif mode_acquire == "triple":
            self.image_unknown = kwargs.pop("image_unknown", None)
            if self.image_unknown is None:
                raise ValueError("image_unknown must be provided when mode_acquire is 'triple'")
            self.image_reference = kwargs.pop("image_reference", None)
            if self.image_reference is None:
                raise ValueError("image_reference must be provided when mode_acquire is 'triple'")

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        # Set up configuration
        if config_folder_path:
            self.final_config_path = Path(config_folder_path)
            if not self.final_config_path.exists():
                raise FileNotFoundError(f"Config path does not exist: {config_folder_path}")
        else:
            # Use default config location
            self.final_config_path = Path.cwd() / "config" / "device"

    def _process_fiber_positions(
        self, gate_noise_intensity: float, mode_fiber_position: str, mode_acquire: str, config_path: Path
    ) -> None:
        """Process fiber positions."""
        # Select image for fiber detection
        if mode_acquire == "single":
            image = self.image_interference
        else:
            image = self.image_unknown

        if mode_fiber_position == "calibration":
            calibration_file = config_path / "setting_fiber_calibration.csv"
            pixel_positions = np.loadtxt(calibration_file, delimiter=",", dtype=int) - 1
            self._get_fiber_position_from_calibration(image, pixel_positions, gate_noise_intensity)
        else:  # calculation
            settings = np.loadtxt(config_path / "setting_fiber_calibration.csv", delimiter=",")
            first_fiber_position = settings[0]
            fiber_spacing = settings[1]
            self._get_fiber_position_from_calculation(image, first_fiber_position, fiber_spacing, gate_noise_intensity)

    def _get_fiber_position_from_calibration(
        self, image: npt.NDArray[np.float64], pixel_positions: npt.NDArray[np.int64], gate_noise_intensity: float
    ) -> None:
        """Get fiber positions from calibration data."""
        max_intensity = np.max(image, axis=1)
        valid_fibers = max_intensity[pixel_positions] > gate_noise_intensity
        signal_indices = np.where(valid_fibers)[0]

        _, self.row, self.col = np.where(signal_indices[:, np.newaxis, np.newaxis] == self.fiber_number)
        self.pixel_of_signal = pixel_positions[signal_indices]

    def _get_fiber_position_from_calculation(
        self, image: npt.NDArray[np.float64], first_position: float, spacing: float, gate_noise_intensity: float
    ) -> None:
        """Calculate fiber positions from parameters."""
        from scipy.signal import find_peaks

        max_intensity = np.max(image, axis=1)
        peaks, _ = find_peaks(max_intensity, height=gate_noise_intensity, distance=spacing / 2)

        # Calculate fiber numbers
        fiber_numbers = np.round((peaks - first_position) / spacing)

        # Filter valid fibers
        max_fiber = self.fiber_number[-1, -1]
        valid_mask = (fiber_numbers >= 0) & (fiber_numbers <= max_fiber)

        self.pixel_of_signal = peaks[valid_mask]
        fiber_numbers = fiber_numbers[valid_mask].astype(int)

        _, self.row, self.col = np.where(fiber_numbers[:, np.newaxis, np.newaxis] == self.fiber_number)

    def _resample_and_process_spectra(
        self, wavelength_center: float, wavelength_width: float, method: str, mode_acquire: str
    ) -> None:
        """Resample spectra to common grid."""
        n_fibers = len(self.row)

        # Initialize arrays
        Sw_interference = np.zeros((self.number_y, self.number_x, self.n_omega))
        if mode_acquire in ["double", "triple"]:
            Sw_unknown = np.zeros((self.number_y, self.number_x, self.n_omega))
        if mode_acquire == "triple":
            Sw_reference = np.zeros((self.number_y, self.number_x, self.n_omega))

        # Resample each fiber's spectrum
        for i in range(n_fibers):
            r, c = self.row[i], self.col[i]
            pixel = self.pixel_of_signal[i]

            Sw_interference[r, c, :] = self.resample_spectrum(
                self.image_interference[pixel, :], wavelength_center, self.n_omega, wavelength_width, method
            )

            if mode_acquire in ["double", "triple"]:
                Sw_unknown[r, c, :] = self.resample_spectrum(
                    self.image_unknown[pixel, :], wavelength_center, self.n_omega, wavelength_width, method
                )

            if mode_acquire == "triple":
                Sw_reference[r, c, :] = self.resample_spectrum(
                    self.image_reference[pixel, :], wavelength_center, self.n_omega, wavelength_width, method
                )

        self.Sw_interference = Sw_interference
        if mode_acquire in ["double", "triple"]:
            self.Sw_unknown = Sw_unknown
        if mode_acquire == "triple":
            self.Sw_reference = Sw_reference

    def _perform_interferometry(
        self,
        n_omega: int,
        n_fft: int,
        delay_min: float | None,
        mode_acquire: str,
        as_calibration: bool,
        reference_pulse: SRSI | None,
        method: str,
        wavelength_center: float,
    ) -> None:
        """Perform spectral interferometry analysis."""
        # FTSI
        self.phase_diff_with_sphere, self.time_interval, Su = self.fourier_transform_spectral_interferometry(
            n_omega, n_fft, delay_min
        )

        if mode_acquire == "single":
            self.Sw_unknown = Su

        # Get reference parameters
        if as_calibration:
            self.rp = self._fit_reference_parameters()
        else:
            rp_path = self.final_config_path / "reference_parameters.json"
            with open(rp_path) as f:
                self.rp = json.load(f)

        # Calculate pulse fronts
        self._calculate_pulse_fronts(wavelength_center)

        # Apply reference pulse compensation if provided
        if reference_pulse is not None:
            if not isinstance(reference_pulse, SRSI):
                raise TypeError("reference_pulse must be an instance of SRSI")
            self.compensate_phase(reference_pulse, method)
        else:
            self.phase = self.phase_diff.copy()

    def _calculate_pulse_fronts(self, wavelength_center: float) -> None:
        """Calculate pulse front timing."""
        # Reference pulse front
        distance = np.sqrt(
            (self.x_matrix - self.rp["x0"]) ** 2 + (self.y_matrix - self.rp["y0"]) ** 2 + self.rp["L"] ** 2
        )
        self.pulse_front_reference = (distance - self.rp["L"]) / (self.SPEED_OF_LIGHT * 1e-6) + self.rp["tau0"]

        # Unknown pulse front
        self.pulse_front = self.pulse_front_reference - self.time_interval

        # Phase difference
        omega_broadcast = self.omega_axis.reshape(1, 1, -1)
        phase_diff = self.phase_diff_with_sphere + self.pulse_front[:, :, np.newaxis] * omega_broadcast

        # Spherical phase correction
        phase_sph = 2 * np.pi * distance / wavelength_center * 1e6
        self.phase_diff = np.angle(np.exp(1j * (phase_diff - phase_sph[:, :, np.newaxis])))

    def _fit_reference_parameters(self) -> dict[str, float]:
        """Fit reference sphere parameters."""

        def sphere_equation(M, x0, y0, L, tau0):
            x, y = M
            return (np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + L**2) - L) / (self.SPEED_OF_LIGHT * 1e-6) + tau0

        # Initial guess
        p0 = (0, 0, 1000, np.nanmin(self.time_interval))

        # Flatten data for fitting
        x_flat = self.x_matrix.ravel()
        y_flat = self.y_matrix.ravel()
        time_flat = self.time_interval.ravel()

        # Fit
        popt, _ = curve_fit(sphere_equation, (x_flat, y_flat), time_flat, p0=p0, nan_policy="omit")

        return {"x0": popt[0], "y0": popt[1], "L": popt[2], "tau0": popt[3]}

    def compensate_phase(self, reference_pulse: SRSI, method: str) -> None:
        """Compensate phase using reference pulse."""
        from scipy.interpolate import interp1d

        # Interpolate reference phase to current frequency axis
        phase_interp = interp1d(
            reference_pulse.omega_axis, reference_pulse.phase.squeeze(), fill_value=0, bounds_error=False, kind=method
        )
        phase_reference = phase_interp(self.omega_axis)

        # Apply compensation
        self.phase = self.phase_diff.copy()
        self.phase[self.row, self.col, :] = np.angle(np.exp(1j * (self.phase[self.row, self.col, :] + phase_reference)))

    @property
    def Et(self) -> npt.NDArray[np.complex128]:
        """Electric field in time domain."""
        phase = self.phase.copy()
        phase[np.isnan(phase)] = 0
        Et = self.iFt(np.sqrt(self.Sw_unknown) * np.exp(-1j * phase), self.n_omega, self.n_fft)
        Et[np.isnan(Et)] = 0
        return Et

    def save_data_to_file(self, folder_path: str | Path, **kwargs) -> None:
        """Save data to files."""
        writer = DataWriter()
        writer.save_sifast_data(
            folder_path,
            self.wavelength,
            self.image_interference,
            getattr(self, "image_unknown", None),
            getattr(self, "image_reference", None),
        )

        # Copy configuration
        config_folder_path = kwargs.pop("config_folder_path", None)
        if config_folder_path is None:
            config_folder_path = Path.cwd() / "config" / "device"

        import shutil

        shutil.copytree(config_folder_path, Path(folder_path) / "config")

        # Update processing log
        params = self.params.copy()
        for key in ["wavelength", "image_interference", "image_unknown", "image_reference"]:
            params.pop(key, None)
        params["folder_path"] = str(folder_path)

        update_processing_log(folder_path, "SUCCESS", params, "Data saved successfully.")

    def plot_scatter(self, values: npt.NDArray[np.float64], scene_model=None, backend: str = "mayavi") -> None:
        """Plot 3D scatter visualization."""
        visualizer = SIFASTVisualizer(self, backend=backend)
        visualizer.plot_scatter(values, scene_model)

    def plot_isosurface(
        self,
        t_min: float,
        t_max: float,
        frequency_scale: float,
        isovalue: float,
        indexing: str = "xy",
        scene_model=None,
        backend: str = "mayavi",
        **kwargs,
    ) -> None:
        """Plot 3D isosurface visualization."""
        visualizer = SIFASTVisualizer(self, backend=backend)
        visualizer.plot_isosurface(t_min, t_max, frequency_scale, isovalue, indexing, scene_model, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Export parameters as dictionary."""
        return self.params.copy()
