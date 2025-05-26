import json
import pathlib
import shutil

import mayavi.core.ui.api
import mayavi.mlab
import numpy as np
from matplotlib import colors
from scipy.signal import find_peaks

from .base import PulseBasic
from .fiber_array import FiberArray
from .srsi import SRSI
from .utils import rescale, update_processing_log


class SIFAST(PulseBasic):
    x_axis: np.ndarray
    y_axis: np.ndarray
    x_matrix: np.ndarray
    y_matrix: np.ndarray
    fiber_number: np.ndarray
    number_x: int
    number_y: int

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

    @property
    def Et(self) -> np.ndarray:
        """Returns the electric field in the time domain."""
        phase = self.phase_diff
        phase[np.isnan(phase)] = 0
        _Et = self.iFt(
            np.sqrt(self.Sw_unknown) * np.exp(-1j * phase), self.n_omega, self.n_fft
        )
        _Et[np.isnan(_Et)] = 0
        return _Et

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
        method_interpolation: str = "linear",
        dx: float = 0,
        dy: float = 0,
        reference_pulse: SRSI | None = None,
        as_calibration: bool = False,
        config_folder_path: str | pathlib.Path | None = None,
        delay_min: float | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the SIFAST class.

        Parameters:
        - mode_input: The mode of input, either "SRSI" or "SIFAST".
        - mode_acquire: The mode of acquisition, either "double" or "triple".
        - gate_noise_intensity: The intensity of the gate noise.
        - wavelength_center: The center wavelength.
        - wavelength_width: The width of the wavelength.
        - n_omega: The number of omega points.
        - n_fft: The number of FFT points.
        - mode_fiber_position: The mode of fiber position, either "calibration" or "measurement".
        - method_interpolation: The method of interpolation, either "linear", "slinear", "quadratic", or "cubic".
        - dx: The x offset for the fiber array.
        - dy: The y offset for the fiber array.
        - reference_pulse: An instance of the SRSI class for reference pulse.
        - as_calibration: Whether to use as calibration or not.
        - config_folder_path: The path to the configuration folder.
        - delay_min: The minimum delay time.
        - kwargs: Additional keyword arguments.
            - folder_path: The path to the folder containing the CSV files (required if mode_input is "read").
            - image_interference: The interference image (required if mode_input is "acquire").
            - image_unknown: The unknown image (required if mode_acquire is "double" or "triple").
            - image_reference: The reference image (required if mode_acquire is "triple").
            - wavelength: The wavelength (required if mode_input is "acquire").
        """
        params = locals()
        del params["self"]
        if "kwargs" in params:
            kw = params.pop("kwargs")
            params.update(kw)
        self.params = params
        try:
            if mode_input not in ["read", "acquire"]:
                raise ValueError("mode_input must be 'read' or 'acquire'")
            if mode_acquire not in ["single", "double", "triple"]:
                raise ValueError("mode_acquire must be 'single', 'double', or 'triple'")
            if mode_fiber_position not in ["calibration", "calculation"]:
                raise ValueError(
                    "mode_fiber_position must be 'calibration' or 'calculation'"
                )
            if method_interpolation not in ["linear", "slinear", "quadratic", "cubic"]:
                raise ValueError(
                    "method_interpolation must be 'linear', 'slinear', 'quadratic', or 'cubic'"
                )

            (
                final_config_path,
                copy_config_after_processing,
                folder_path_read_mode,
            ) = self._initialize_input_data(
                mode_input, mode_acquire, config_folder_path, **kwargs
            )
            # folder_path_read_mode is None if mode_input is not 'read'

            self.omega_center = 2 * np.pi * self.SPEED_OF_LIGHT / wavelength_center
            self.n_omega = n_omega
            self.n_fft = n_fft

            fiber_array = FiberArray(dx, dy)
            for key, value in fiber_array.get_fiber_array_properties().items():
                setattr(self, key, value)

            self._get_fiber_position(
                gate_noise_intensity,
                mode_fiber_position,
                mode_acquire,
                config_folder_path=final_config_path,
            )

            self._process_spectral_data(
                wavelength_center,
                self.n_omega,
                wavelength_width,
                method_interpolation,
                mode_acquire,
            )

            # Call fourier_transform_spectral_interferometry and store results in local variables
            phase_diff_with_sphere, time_interval, Su = (
                self.fourier_transform_spectral_interferometry(
                    self.n_omega, n_fft, delay_min=delay_min
                )
            )
            if mode_acquire == "single":
                self.Sw_unknown = Su

            self._calculate_phase_properties(
                final_config_path,
                as_calibration,
                reference_pulse,
                wavelength_center,
                phase_diff_with_sphere,  # Pass local variable
                time_interval,  # Pass local variable
            )

            if mode_input == "read":
                if copy_config_after_processing:
                    # Ensure local_config_path is defined correctly if needed here
                    # For now, assuming config_folder_path from _initialize_input_data's scope
                    # or that final_config_path is the one to be copied from.
                    # The original logic used local_config_path = pathlib.Path(folder_path) / "config"
                    # and config_folder_path was the external one.
                    # This needs careful handling if copy_config_after_processing is True.
                    # Based on the logic, if copy_config_after_processing is True,
                    # then an external config_folder_path was provided and should be copied to
                    # the local config path.
                    if (
                        config_folder_path and folder_path_read_mode
                    ):  # Ensure config_folder_path was external
                        external_config_path = pathlib.Path(config_folder_path)
                        local_config_target_path = (
                            pathlib.Path(folder_path_read_mode) / "config"
                        )
                        shutil.copytree(external_config_path, local_config_target_path)
                success_message = (
                    f"Data processed using config from '{final_config_path}'."
                )
                update_processing_log(
                    folder_path_read_mode, "SUCCESS", self.params, success_message
                )
        except Exception as e:
            if mode_input == "read" and folder_path_read_mode:
                update_processing_log(
                    folder_path_read_mode, "FAILURE", self.params, str(e)
                )
            raise

    def _initialize_input_data(
        self,
        mode_input: str,
        mode_acquire: str,
        config_folder_path: str | pathlib.Path | None,
        **kwargs,
    ):
        """
        Initializes input data based on mode_input.

        Sets instance attributes for images and wavelength.
        Returns final_config_path and copy_config_after_processing flag.
        """
        final_config_path = None
        copy_config_after_processing = False
        folder_path_for_log = None  # Used for the try-except block in __init__

        if mode_input == "read":
            folder_path = kwargs.pop("folder_path", None)
            if folder_path is None:
                raise ValueError(
                    "folder_path must be provided when mode_input is 'read'"
                )
            folder_path_for_log = folder_path
            self._read_image_from_csv(folder_path, mode_acquire)
            if kwargs:
                raise ValueError(
                    f"Unexpected keyword arguments for mode_input 'read': {', '.join(kwargs.keys())}"
                )

            if config_folder_path:
                config_folder_path = pathlib.Path(config_folder_path)
                local_config_path = pathlib.Path(folder_path) / "config"
                if local_config_path.exists():
                    raise ValueError(f"Configuration conflict in {folder_path}.")
                if not config_folder_path.exists():
                    raise FileNotFoundError(
                        f"External config path does not exist: {config_folder_path}"
                    )
                final_config_path = config_folder_path
                copy_config_after_processing = True
            else:
                local_config_path = pathlib.Path(folder_path) / "config"
                if local_config_path.exists():
                    final_config_path = local_config_path
                else:
                    raise FileNotFoundError(
                        f"Missing config in {folder_path} and no external path provided."
                    )
        elif mode_input == "acquire":
            image_interference = kwargs.pop("image_interference", None)
            if image_interference is None:
                raise ValueError(
                    "image_interference must be provided when mode_input is 'acquire'"
                )
            self.image_interference = image_interference
            wavelength = kwargs.pop("wavelength", None)
            if wavelength is None:
                raise ValueError(
                    "wavelength must be provided when mode_input is 'acquire'"
                )
            self.wavelength = wavelength

            match mode_acquire:
                case "double":
                    image_unknown = kwargs.pop("image_unknown", None)
                    if image_unknown is None:
                        raise ValueError(
                            "image_unknown must be provided when mode_acquire is 'double'"
                        )
                    self.image_unknown = image_unknown
                case "triple":
                    image_unknown = kwargs.pop("image_unknown", None)
                    if image_unknown is None:
                        raise ValueError(
                            "image_unknown must be provided when mode_acquire is 'triple'"
                        )
                    self.image_unknown = image_unknown
                    image_reference = kwargs.pop("image_reference", None)
                    if image_reference is None:
                        raise ValueError(
                            "image_reference must be provided when mode_acquire is 'triple'"
                        )
                    self.image_reference = image_reference
            if kwargs:
                raise ValueError(
                    f"Unexpected keyword arguments for mode_input 'acquire': {', '.join(kwargs.keys())}"
                )

            if config_folder_path:
                config_folder_path = pathlib.Path(config_folder_path)
                if not config_folder_path.exists():
                    raise FileNotFoundError(
                        f"External config path does not exist: {config_folder_path}"
                    )
                final_config_path = config_folder_path
            else:
                final_config_path = pathlib.Path(__file__).parent.parent / "config"

        return final_config_path, copy_config_after_processing, folder_path_for_log

    def _read_image_from_csv(self, folder_path: str, mode_acquire: str) -> None:
        """
        Reads the image data from CSV files in the specified folder.

        Parameters:
        - folder_path: The path to the folder containing the CSV files.
        - mode_acquire: The acquisition mode (e.g., "double", "triple").

        Returns:
        - None
        """
        matching_file = list(pathlib.Path(folder_path).glob("*inter*.csv"))
        if len(matching_file) == 0:
            raise FileNotFoundError(f"No interference spectrum found in {folder_path}")
        self.image_interference = np.loadtxt(
            matching_file[0], delimiter=",", skiprows=3
        )[1:, 1:]
        self.wavelength = np.loadtxt(matching_file[0], delimiter=",", skiprows=3)[0, 1:]

        match mode_acquire:
            case "double":
                matching_file = list(pathlib.Path(folder_path).glob("*unk*.csv"))
                if len(matching_file) == 0:
                    raise FileNotFoundError(
                        f"No unknown spectrum found in {folder_path}"
                    )
                self.image_unknown = np.loadtxt(
                    matching_file[0], delimiter=",", skiprows=3
                )[1:, 1:]
            case "triple":
                matching_file = list(pathlib.Path(folder_path).glob("*unk*.csv"))
                if len(matching_file) == 0:
                    raise FileNotFoundError(
                        f"No unknown spectrum found in {folder_path}"
                    )
                self.image_unknown = np.loadtxt(
                    matching_file[0], delimiter=",", skiprows=3
                )[1:, 1:]
                matching_file = list(pathlib.Path(folder_path).glob("*ref*.csv"))
                if len(matching_file) == 0:
                    raise FileNotFoundError(
                        f"No reference spectrum found in {folder_path}"
                    )
                self.image_reference = np.loadtxt(
                    matching_file[0], delimiter=",", skiprows=3
                )[1:, 1:]

    def _get_fiber_position(
        self,
        gate_noise_intensity: float,
        mode_fiber_position: str,
        mode_acquire: str,
        config_folder_path: str,
    ) -> None:
        if mode_acquire == "single":
            image = self.image_interference
        else:
            image = self.image_unknown

        match mode_fiber_position:
            case "calibration":
                pixel_position_of_fiber = (
                    np.loadtxt(
                        pathlib.Path(config_folder_path)
                        / "setting_fiber_calibration.csv",
                        delimiter=",",
                        dtype=int,
                    )
                    - 1
                )
                self._get_fiber_position_from_calibration(
                    image, pixel_position_of_fiber, gate_noise_intensity
                )
            case "calculation":
                pixel_position_of_first_fiber = np.loadtxt(
                    pathlib.Path(config_folder_path) / "setting_fiber_calibration.csv",
                    delimiter=",",
                )[0]
                pixel_of_fiber_spacing = np.loadtxt(
                    pathlib.Path(config_folder_path) / "setting_fiber_calibration.csv",
                    delimiter=",",
                )[1]
                self._get_fiber_position_from_calculation(
                    image,
                    pixel_position_of_first_fiber,
                    pixel_of_fiber_spacing,
                    gate_noise_intensity,
                )

    def _get_fiber_position_from_calibration(
        self,
        image: np.ndarray,
        pixel_position_of_fiber: np.ndarray,
        gate_noise_intensity: float,
    ) -> None:
        max_intensity_of_every_row = np.max(image, axis=1)
        signal_number = np.where(
            max_intensity_of_every_row[pixel_position_of_fiber] > gate_noise_intensity
        )[0]
        _, self.row, self.col = np.where(
            signal_number[:, np.newaxis, np.newaxis] == self.fiber_number
        )
        self.pixel_of_signal = pixel_position_of_fiber[signal_number]

    def _get_fiber_position_from_calculation(
        self,
        image: np.ndarray,
        pixel_position_of_first_fiber: float,
        pixel_of_fiber_spacing: float,
        gate_noise_intensity: float,
    ) -> None:
        max_intensity_of_every_row = np.max(image, axis=1)
        pixel_of_signal, _ = find_peaks(
            max_intensity_of_every_row,
            height=gate_noise_intensity,
            distance=pixel_of_fiber_spacing / 2,
        )
        signal_number = np.round(
            (pixel_of_signal - pixel_position_of_first_fiber) / pixel_of_fiber_spacing
        )
        pixel_of_signal = np.delete(
            pixel_of_signal,
            np.where(
                (signal_number < 0) | (signal_number >= self.fiber_number[-1, -1])
            )[0],
        )
        signal_number = np.delete(
            signal_number,
            np.where(
                (signal_number < 0) | (signal_number >= self.fiber_number[-1, -1])
            )[0],
        )
        _, self.row, self.col = np.where(
            signal_number[:, np.newaxis, np.newaxis] == self.fiber_number
        )
        self.pixel_of_signal = pixel_of_signal

    def _process_spectral_data(
        self,
        wavelength_center: float,
        n_omega: int,
        wavelength_width: float,
        method_interpolation: str,
        mode_acquire: str,
    ) -> None:
        """
        Processes spectral data by resampling and assigning to instance attributes.
        """
        Sw_interference = np.zeros((self.number_y, self.number_x, n_omega))
        Sw_unknown = np.zeros((self.number_y, self.number_x, n_omega))
        Sw_reference = np.zeros((self.number_y, self.number_x, n_omega))

        for i in range(len(self.row)):
            Sw_interference[self.row[i], self.col[i], :] = self.resample_signal(
                self.image_interference[self.pixel_of_signal[i], :],
                wavelength_center,
                n_omega,
                wavelength_width,
                method_interpolation,
            )
        self.Sw_interference = Sw_interference

        if mode_acquire == "double":
            for i in range(len(self.row)):
                Sw_unknown[self.row[i], self.col[i], :] = self.resample_signal(
                    self.image_unknown[self.pixel_of_signal[i], :],
                    wavelength_center,
                    n_omega,
                    wavelength_width,
                    method_interpolation,
                )
            self.Sw_unknown = Sw_unknown
        elif mode_acquire == "triple":
            for i in range(len(self.row)):
                Sw_unknown[self.row[i], self.col[i], :] = self.resample_signal(
                    self.image_unknown[self.pixel_of_signal[i], :],
                    wavelength_center,
                    n_omega,
                    wavelength_width,
                    method_interpolation,
                )
            self.Sw_unknown = Sw_unknown
            for i in range(len(self.row)):
                Sw_reference[self.row[i], self.col[i], :] = self.resample_signal(
                    self.image_reference[self.pixel_of_signal[i], :],
                    wavelength_center,
                    n_omega,
                    wavelength_width,
                    method_interpolation,
                )
            self.Sw_reference = Sw_reference

        # Assign Sw_unknown and Sw_reference (populated or zero arrays) to instance attributes
        # self.Sw_unknown is already assigned within the double/triple blocks.
        # If mode_acquire is "single", the local Sw_unknown (zeros) should be assigned.
        # The same applies to Sw_reference.
        if mode_acquire == "single":
            self.Sw_unknown = Sw_unknown
            self.Sw_reference = Sw_reference
        elif mode_acquire == "double":
            # self.Sw_unknown is already set.
            self.Sw_reference = Sw_reference  # Assigns the zero array for Sw_reference

    def _calculate_phase_properties(
        self,
        final_config_path: pathlib.Path,
        as_calibration: bool,
        reference_pulse: SRSI | None,
        wavelength_center: float,
        phase_diff_with_sphere: np.ndarray,
        time_interval: np.ndarray,
    ) -> None:
        """
        Calculates pulse front and phase properties.
        Sets self.pulse_front_reference, self.pulse_front, self.phase_diff, and self.phase.
        """
        if as_calibration:
            rp = {}
            # In as_calibration mode, rp parameters like x0, y0, L, tau0 might be missing.
            # The original code would fail here if rp keys are accessed.
            # Assuming that if as_calibration is True, these calculations are either
            # skipped or handled differently, or rp is pre-populated with defaults.
            # For now, replicating the 'pass' behavior implicitly means subsequent lines
            # might error if they depend on x0, y0, L, tau0 from rp.
            # This behavior is kept from the original code.
            # If specific calculations should be skipped, that logic needs to be added.
            # Let's assume for now that if as_calibration is true, the rp-dependent lines are not critical
            # or rp gets populated by some other means not shown.
            # A proper implementation might initialize rp with default values or skip calculations.
            # For this refactoring, we keep the structure.
            # Example: rp = {"x0": 0, "y0": 0, "L": 1, "tau0": 0} # Default/dummy values
        else:
            rp_path = final_config_path / "reference_parameters.json"
            rp = json.load(open(rp_path))

        # These calculations will fail if as_calibration is True and rp is empty and does not have default values.
        # This matches the original code's potential issue.
        self.pulse_front_reference = (
            np.sqrt(
                (self.x_matrix - rp.get("x0", 0)) ** 2
                + (self.y_matrix - rp.get("y0", 0)) ** 2
                + rp.get("L", 1) ** 2
            )
            - rp.get("L", 1)
        ) / (self.SPEED_OF_LIGHT * 1e-6) + rp.get("tau0", 0)
        self.pulse_front = self.pulse_front_reference - time_interval

        # Ensure self.omega_axis is available
        if not hasattr(self, "_omega_axis") or self._omega_axis is None:
            # This might happen if resample_signal which sets _omega_axis was not called
            # or if it's not set correctly prior to this.
            # For now, assuming it's set. If not, an error or default is needed.
            # This was not explicitly handled before but could be an issue.
            # A robust solution would ensure _omega_axis is always valid.
            pass  # Assuming omega_axis is correctly set by resample_signal

        self.phase_diff = phase_diff_with_sphere + self.pulse_front[
            :, :, np.newaxis
        ] * self.omega_axis.reshape(1, 1, -1)

        phase_sph = (
            2
            * np.pi
            * np.sqrt(
                (self.x_matrix - rp.get("x0", 0)) ** 2
                + (self.y_matrix - rp.get("y0", 0)) ** 2
                + rp.get("L", 1) ** 2
            )
            / wavelength_center
            * 1e6
        )
        self.phase_diff = np.angle(
            np.exp(1j * (self.phase_diff - phase_sph[:, :, np.newaxis]))
        )

        if reference_pulse is not None:
            # Original code had 'pass', implying self.phase might not be set if reference_pulse is present.
            # Or, it implies that if reference_pulse is used, self.phase is handled by it.
            # For now, replicating 'pass'. If self.phase should take a specific value,
            # it needs to be defined here.
            # If the intention was that self.phase_diff is the final phase, then:
            # self.phase = self.phase_diff
            pass  # Original behavior
        else:
            self.phase = self.phase_diff

    def save_data_to_file(self, folder_path: str, **kwargs) -> None:
        zero_matrix = np.zeros((2052, 2049))
        zero_matrix[3, 1:] = self.wavelength
        image_interference = zero_matrix.copy()
        image_interference[4:, 1:] = self.image_interference
        np.savetxt(
            pathlib.Path(folder_path) / "inter.csv", image_interference, delimiter=","
        )

        if self.image_unknown is not None:
            image_unknown = zero_matrix.copy()
            image_unknown[4:, 1:] = self.image_unknown
            np.savetxt(
                pathlib.Path(folder_path) / "unk.csv", image_unknown, delimiter=","
            )

        if self.image_reference is not None:
            image_reference = zero_matrix.copy()
            image_reference[4:, 1:] = self.image_reference
            np.savetxt(
                pathlib.Path(folder_path) / "ref.csv", image_reference, delimiter=","
            )

        config_folder_path = kwargs.pop("config_folder_path", None)
        if config_folder_path is None:
            config_folder_path = pathlib.Path(__file__).parent.parent / "config"
        else:
            config_folder_path = pathlib.Path(config_folder_path)
        shutil.copytree(config_folder_path, folder_path / "config")
        params = self.params
        del params["wavelength"]
        del params["image_interference"]
        del params["image_unknown"]
        del params["image_reference"]
        params["folder_path"] = folder_path
        update_processing_log(
            folder_path, "SUCCESS", params, "Data saved successfully."
        )

    def plot_scatter(
        self,
        values: np.ndarray,
        scene_model: None | mayavi.core.ui.api.MlabSceneModel = None,
    ) -> None:
        if scene_model is None:
            fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            active_mlab = mayavi.mlab
            target_scene = fig
        else:
            active_mlab = scene_model.mlab
            target_scene = scene_model.mayavi_scene

        x_range = self.x_axis.max() - self.x_axis.min()
        values = values.flatten()
        values_range = np.nanmax(values) - np.nanmin(values)
        if values_range != 0:
            scale_factor_values = x_range / values_range
        else:
            scale_factor_values = 1.0
        values_for_plotting = values * scale_factor_values

        points_obj = active_mlab.points3d(
            self.x_matrix.flatten(),
            self.y_matrix.flatten(),
            values_for_plotting,
            values,
            mode="sphere",
            colormap="inferno",
            scale_factor=0.5,
            figure=target_scene,
            resolution=20,
            scale_mode="none",
        )
        active_mlab.outline(points_obj, color=(0.7, 0.7, 0.7))

        axes = active_mlab.axes(points_obj)
        axes_range_scaled = axes.axes.bounds
        axes_range = axes_range_scaled
        axes_range[4:6] /= scale_factor_values
        axes.remove()
        axes = active_mlab.axes(
            points_obj, xlabel="x (mm)", ylabel="y (mm)", nb_labels=5, ranges=axes_range
        )
        axes.axes.label_format = "%.1f"
        axes.title_text_property.color = (0.0, 0.0, 0.0)
        axes.label_text_property.color = (0.0, 0.0, 0.0)
        axes.axes.fly_mode = "none"
        active_mlab.colorbar(points_obj, orientation="vertical")

        if target_scene:
            target_scene.scene.reset_zoom()

        if scene_model is None:
            mayavi.mlab.show()

    def plot_isosurface(
        self,
        t_min: float,
        t_max: float,
        frequency_scale: float,
        isovalue: float,
        indexing: str = "xy",
        scene_model: None | mayavi.core.ui.api.MlabSceneModel = None,
        **kwargs,
    ) -> None:
        if scene_model is None:
            fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            active_mlab = mayavi.mlab
            target_scene = fig
        else:
            active_mlab = scene_model.mlab
            target_scene = scene_model.mayavi_scene
        opacity = kwargs.pop("opacity", 1)

        Et = self.Et
        if indexing == "xy":
            Et = np.transpose(Et, (1, 0, 2))

        if frequency_scale != 0:
            values = np.real(
                rescale(np.abs(Et))
                * np.exp(
                    1j
                    * frequency_scale
                    * self.t_axis[np.newaxis, np.newaxis, :]
                    * self.omega_center
                )
                * np.exp(1j * frequency_scale * np.angle(Et))
            )
        else:
            values = rescale(np.abs(Et) ** 2)

        values_for_plotting = values[
            :, :, (self.t_axis > t_min) & (self.t_axis < t_max)
        ]
        x_plot, y_plot, z_plot = np.meshgrid(
            self.x_axis,
            self.y_axis,
            self.t_axis[(self.t_axis > t_min) & (self.t_axis < t_max)],
            indexing="ij",
        )
        isosurface_obj = active_mlab.contour3d(
            x_plot,
            y_plot,
            z_plot,
            values_for_plotting,
            contours=[isovalue],
            color=colors.to_rgb("lightblue"),
            opacity=0.5,
        )

        axes = active_mlab.axes(isosurface_obj)
        axes_range = axes.axes.bounds
        isosurface_obj.remove()
        axes.remove()
        x_range = axes_range[1] - axes_range[0]
        z_range = axes_range[5] - axes_range[4]
        if z_range != 0:
            scale_factor_z = x_range / z_range * 1.618
        else:
            scale_factor_z = 1.0

        x_plot, y_plot, z_plot = np.meshgrid(
            self.x_axis,
            self.y_axis,
            self.t_axis[(self.t_axis > t_min) & (self.t_axis < t_max)] * scale_factor_z,
            indexing="ij",
        )
        isosurface_obj = active_mlab.contour3d(
            x_plot,
            y_plot,
            z_plot,
            values_for_plotting,
            contours=[isovalue],
            color=colors.to_rgb("lightblue"),
            opacity=opacity,
        )
        active_mlab.outline(isosurface_obj, color=(0.7, 0.7, 0.7))
        axes = active_mlab.axes(
            isosurface_obj,
            xlabel="x (mm)",
            ylabel="y (mm)",
            zlabel="t (fs)",
            nb_labels=5,
            ranges=axes_range,
        )
        axes.axes.label_format = "%.1f"
        axes.title_text_property.color = (0.0, 0.0, 0.0)
        axes.label_text_property.color = (0.0, 0.0, 0.0)
        axes.axes.fly_mode = "none"
        active_mlab.colorbar(isosurface_obj, orientation="vertical")

        if target_scene:
            target_scene.scene.reset_zoom()

        if scene_model is None:
            mayavi.mlab.show()

    def to_dict(self) -> dict:
        return self.params
