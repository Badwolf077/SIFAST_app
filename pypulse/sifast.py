from .base import PulseBasic, FiberArrayBasic
from .srsi import SRSI
from .fiber_array import FiberArray
import numpy as np
import os
import glob
from scipy.signal import find_peaks

class SIFAST(PulseBasic): 
    x_axis: np.ndarray
    y_axis: np.ndarray
    x_matrix: np.ndarray
    y_matrix: np.ndarray
    fiber_number: np.ndarray
    number_x: int
    number_y: int
    @PulseBasic.t_axis.getter
    def t_axis(self) -> np.ndarray:
        """Returns the time axis of the pulse."""
        return self._t_axis
    @PulseBasic.omega_axis.getter
    def omega_axis(self) -> np.ndarray:
        """Returns the frequency axis of the pulse."""
        return self._omega_axis
    @PulseBasic.wavelength_axis.getter
    def wavelength_axis(self) -> np.ndarray:
        """Returns the wavelength axis of the pulse."""
        return self._wavelength_axis
    
    def __init__(self, 
                 mode_input: str,
                 mode_acquire: str,
                 gate_noise_intensity: float,
                 wavelength_center: float,
                 wavelgth_width: float,
                 n_omega: int,
                 n_fft: int,
                 mode_fiber_position: str = "calibration",
                 method_interpolation: str = "linear",
                 dx: float = 0,
                 dy: float = 0,
                 reference_pulse: SRSI | None = None,
                 as_calibration: bool = False,
                 **kwargs) -> None:
        """
        Initializes the SIFAST class.

        Parameters:
        - mode_input: The mode of input, either "SRSI" or "SIFAST".
        - mode_acquire: The mode of acquisition, either "double" or "triple".
        - gate_noise_intensity: The intensity of the gate noise.
        - wavelength_center: The center wavelength.
        - wavelgth_width: The width of the wavelength.
        - n_omega: The number of omega points.
        - n_fft: The number of FFT points.
        - mode_fiber_position: The mode of fiber position, either "calibration" or "measurement".
        - method_interpolation: The method of interpolation, either "linear", "slinear", "quadratic", or "cubic".
        - dx: The x offset for the fiber array.
        - dy: The y offset for the fiber array.
        - reference_pulse: An instance of the SRSI class for reference pulse.
        - as_calibration: Whether to use as calibration or not.
        - kwargs: Additional keyword arguments.
        """
        if mode_input not in ["read", "acquire"]:
            raise ValueError("mode_input must be 'read' or 'acquire'")
        if mode_acquire not in ["single", "double", "triple"]:
            raise ValueError("mode_acquire must be 'single', 'double', or 'triple'")
        if mode_fiber_position not in ["calibration", "calculation"]:
            raise ValueError("mode_fiber_position must be 'calibration' or 'calculation'")
        if method_interpolation not in ["linear", "slinear", "quadratic", "cubic"]:
            raise ValueError("method_interpolation must be 'linear', 'slinear', 'quadratic', or 'cubic'")
        
        if not as_calibration:
            config_folder_path = kwargs.pop("config_folder_path", None)
            if config_folder_path is None:
                raise ValueError("config_folder_path must be provided when as_calibration is False")
        
        if mode_input == "read":
            folder_path = kwargs.pop("folder_path", None)
            if folder_path is None:
                raise ValueError("folder_path must be provided when mode_input is 'read'")
            self.__read_image_from_csv(folder_path, mode_acquire)
            if kwargs:
                raise ValueError(f"Unexpected keyword arguments for mode_input 'read': {', '.join(kwargs.keys())}")
        elif mode_input == "acquire":
            image_interference = kwargs.pop("image_interference", None)
            if image_interference is None:
                raise ValueError("image_interference must be provided when mode_input is 'acquire'")
            self.Sw_interference = image_interference
            wavelength = kwargs.pop("wavelength", None)
            if wavelength is None:
                raise ValueError("wavelength must be provided when mode_input is 'acquire'")
            self.wavelength = wavelength
            
            match mode_acquire:
                case "double":
                    image_unknown = kwargs.pop("image_unknown", None)
                    if image_unknown is None:
                        raise ValueError("image_unknown must be provided when mode_acquire is 'double'")
                    self.Sw_unknown = image_unknown
                case "triple":
                    image_unknown = kwargs.pop("image_unknown", None)
                    if image_unknown is None:
                        raise ValueError("image_unknown must be provided when mode_acquire is 'triple'")
                    self.Sw_unknown = image_unknown
                    image_reference = kwargs.pop("image_reference", None)
                    if image_reference is None:
                        raise ValueError("image_reference must be provided when mode_acquire is 'triple'")
                    self.Sw_reference = image_reference
                    
            if kwargs:
                raise ValueError(f"Unexpected keyword arguments for mode_input 'acquire': {', '.join(kwargs.keys())}")
        
        self.omega_center = 2 * np.pi * self.SPEED_OF_LIGHT / wavelength_center
        self.n_omega = n_omega
        self.n_fft = n_fft
        
        fiber_array = FiberArray(dx, dy)
        for key, value in fiber_array.get_fiber_array_properties().items():
            setattr(self, key, value)
        
        
            
    def __read_image_from_csv(self, folder_path: str, mode_acquire: str) -> None:
        """
        Reads the image data from CSV files in the specified folder.

        Parameters:
        - folder_path: The path to the folder containing the CSV files.
        - mode_acquire: The acquisition mode (e.g., "double", "triple").

        Returns:
        - None
        """
        matching_file = glob.glob(os.path.join(folder_path, '*inter*.txt'))
        if len(matching_file) == 0:
            raise FileNotFoundError(f"No interference spectrum found in {folder_path}")
        self.Sw_interference = np.loadtxt(matching_file[0], delimiter=',', skiprows=3)[1:, 1:]
        self.wavelength = np.loadtxt(matching_file[0], delimiter=',', skiprows=3)[0, 1:]
        
        match mode_acquire:
            case "double":
                matching_file = glob.glob(os.path.join(folder_path, '*unk*.txt'))
                if len(matching_file) == 0:
                    raise FileNotFoundError(f"No unknown spectrum found in {folder_path}")
                self.Sw_unknown = np.loadtxt(matching_file[0], delimiter=',', skiprows=3)[1:, 1:]
            case "triple":
                matching_file = glob.glob(os.path.join(folder_path, '*unk*.txt'))
                if len(matching_file) == 0:
                    raise FileNotFoundError(f"No unknown spectrum found in {folder_path}")
                self.Sw_unknown = np.loadtxt(matching_file[0], delimiter=',', skiprows=3)[1:, 1:]
                matching_file = glob.glob(os.path.join(folder_path, '*ref*.txt'))
                if len(matching_file) == 0:
                    raise FileNotFoundError(f"No reference spectrum found in {folder_path}")
                self.Sw_reference = np.loadtxt(matching_file[0], delimiter=',', skiprows=3)[1:, 1:]

    def __get_fiber_position(self, gate_noise_intensity: float, 
                             mode_fiber_position: str, 
                             mode_acquire: str, 
                             config_folder_path: str) -> None:
        if mode_acquire == "single":
            image = self.Sw_interference
        else:
            image = self.Sw_unknown
            
        match mode_fiber_position:
            case "calibration":
                pixel_position_of_fiber = np.loadtxt(os.path.join(config_folder_path, 'fiber_position.txt'), delimiter=',')
                self.__get_fiber_position_from_calibration(image, pixel_position_of_fiber, gate_noise_intensity)
            case "calculation":
                pass

    def __get_fiber_position_from_calibration(self, image: np.ndarray, 
                                              pixel_position_of_fiber: np.ndarray,
                                              gate_noise_intensity: float) -> None:
        max_intensity_of_every_row = np.max(image, axis=1)
        signal_number = np.where(max_intensity_of_every_row[pixel_position_of_fiber-1] > gate_noise_intensity)[0]
        _, self.row, self.col = np.where(signal_number[:, np.newaxis, np.newaxis] == self.fiber_number)
        self.pixel_of_signal = pixel_position_of_fiber[signal_number]
        
    def __get_fiber_position_from_calculation(self, image: np.ndarray,
                                              pixel_position_of_first_fiber: float,
                                              pixel_of_fiber_spacing: float,
                                              gate_noise_intensity: float) -> None:
        max_intensity_of_every_row = np.max(image, axis=1)
        pixel_of_signal, _ = find_peaks(max_intensity_of_every_row, height=gate_noise_intensity, 
                                  distance=pixel_of_fiber_spacing/2)
        signal_number = np.round((pixel_of_signal-pixel_position_of_first_fiber)/pixel_of_fiber_spacing)
        pixel_of_signal = np.delete(pixel_of_signal, np.where(signal_number < 0 or signal_number >= self.fiber_number)[0])