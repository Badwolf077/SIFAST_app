from abc import ABC, abstractmethod
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2
import scipy.interpolate as interp
from scipy.signal import find_peaks
from .utils import rescale

class PulseBasic(ABC):
    @property
    def SPEED_OF_LIGHT(self) -> float:
        """Returns the speed of light in vacuum."""
        return 299792458*1e9*1e-15  # in fs/nm
    @property
    @abstractmethod
    def t_axis(self) -> np.ndarray:
        """Returns the time axis of the pulse."""
        pass
    @t_axis.setter
    def t_axis(self, value: np.ndarray) -> None:
        """Sets the time axis of the pulse."""
        self._t_axis = value
    @property
    @abstractmethod
    def omega_axis(self) -> np.ndarray:
        """Returns the frequency axis of the pulse."""
        pass
    @omega_axis.setter
    def omega_axis(self, value: np.ndarray) -> None:
        """Sets the frequency axis of the pulse."""
        self._omega_axis = value
    @property
    @abstractmethod
    def wavelength_axis(self) -> np.ndarray:
        """Returns the wavelength axis of the pulse."""
        pass
    @wavelength_axis.setter
    def wavelength_axis(self, value: np.ndarray) -> None:
        """Sets the wavelength axis of the pulse."""
        self._wavelength_axis = value
        
    def get_property(self, property_name: str):
        return getattr(self, property_name)
    
    @staticmethod
    def Ft(Et: np.ndarray, n_omega: int, n_fft: int) -> np.ndarray:
        Ew = fftshift(fft(ifftshift(Et, axes=2), axis=2), axes=2)
        Ew = Ew[:, :, int((n_fft-n_omega)/2):int((n_fft+n_omega)/2)]
        return Ew
    @staticmethod
    def iFt(Ew: np.ndarray, n_omega: int, n_fft: int) -> np.ndarray:
        zero_padding = np.zeros((Ew.shape[0], Ew.shape[1], (n_fft-n_omega)//2))
        Ew = np.concatenate((zero_padding, Ew, zero_padding), axis=2)
        Et = fftshift(ifft(ifftshift(Ew, axes=2), axis=2), axes=2)
        return Et
    @staticmethod
    def F(Exy: np.ndarray) -> np.ndarray:
        EXY = ifftshift(ifftshift(
            fft2(fftshift(fftshift(Exy, axes=1), axes=0)), axes=1), axes=0)
        return EXY
    @staticmethod
    def iF(EXY: np.ndarray) -> np.ndarray:
        Exy = ifftshift(ifftshift(
            ifft2(fftshift(fftshift(EXY, axes=1), axes=0)), axes=1), axes=0)
        return Exy
    
    def resample_signal(self, spectrum: np.ndarray, wavelength_center: float, 
                        n_omega: int, wavelength_width: float, 
                        method: str = "linear") -> np.ndarray:
        """
        Resamples the spectrum to a new wavelength axis using interpolation.

        Parameters:
        - spectrum: The spectrum to be resampled.
        - wavelength: The original wavelength axis.
        - n_omega: The number of points in the new wavelength axis.
        - wavelength_width: The width of the new wavelength axis.
        - method: The interpolation method (default is "linear").

        Returns:
        - resampled_spectrum: The resampled spectrum.
        """
        for attr in ['wavelength', 'omega_center']:
            if not hasattr(self, attr):
                raise ValueError(f"{attr} is not defined in the pulse object.")
            
        idx = np.where(np.abs(
            self.get_property('wavelength') - wavelength_center) > wavelength_width / 2)[0]
        spectrum = spectrum - np.mean(spectrum[idx])
        omega = (2 * np.pi * self.SPEED_OF_LIGHT / self.get_property('wavelength') - self.get_property('omega_center'))
        delta_omega = (2 * np.pi * self.SPEED_OF_LIGHT * 
                        (1/(wavelength_center-wavelength_width/2)-1/wavelength_center))
        self.omega_axis = np.linspace(-delta_omega, delta_omega, n_omega)
        self.wavelength_axis = (2 * np.pi * self.SPEED_OF_LIGHT / 
                                (self.omega_axis+self.get_property('omega_center')))
        spectrum_resampled = interp.interp1d(omega, spectrum, kind=method, 
                                                bounds_error=False, fill_value=0)(self.omega_axis)
        return spectrum_resampled
    
    def fourier_transform_spectral_interferometry(self, n_omega: int, n_fft: int, 
                                                  delay_min: float | None = None,
                                                  filter_order: int=8) -> tuple[
                                                      np.ndarray, np.ndarray, np.ndarray]:
        for attr in ['Sw_interference', 'row', 'col']:
            if not hasattr(self, attr):
                raise ValueError(f"{attr} is not defined in the pulse object.")
        if filter_order % 2 != 0:
            raise ValueError("Filter order must be even.")
        
        f = (np.max(self.omega_axis) + 
             (n_fft-n_omega)/2*np.abs(self.omega_axis[1]-self.omega_axis[0]))/np.pi
        self.t_axis = (np.arange(n_fft) - (n_fft-1)/2) / f
        
        St = self.iFt(self.get_property('Sw_interference'), n_omega, n_fft)
        delay = np.full((St.shape[0], St.shape[1]), np.nan)
        
        if delay_min is None:
            t_axis_temp = self.t_axis[n_fft//2:]
            for i in range(len(self.get_property('row'))):
                signal = rescale(np.abs(St[self.get_property('row')[i], self.get_property('col')[i], :]))
                signal = signal[n_fft//2:]
                idx_peaks, _ = find_peaks(signal, height=0.01)
                if len(idx_peaks) > 1:
                    delay[self.get_property('row')[i], self.get_property('col')[i]] = t_axis_temp[idx_peaks[np.argsort(signal[idx_peaks])[-1]]]
        else:
            t_axis_temp = self.t_axis[self.t_axis > delay_min/2]
            for i in range(len(self.get_property('row'))):
                signal = rescale(np.abs(St[self.get_property('row')[i], self.get_property('col')[i], :]))
                signal = signal[self.t_axis > delay_min]
                idx_peaks, _ = find_peaks(signal, height=0.01)
                if len(idx_peaks) > 1:
                    delay[self.get_property('row')[i], self.get_property('col')[i]] = t_axis_temp[idx_peaks[np.argsort(signal[idx_peaks])[-1]]]
        
        filter_width = (-np.log(0.001))**(-1/filter_order) * delay/2
        filter_AC = np.exp(-((np.reshape(self.t_axis, (1,1,-1)) - delay)/filter_width)**filter_order)
        filter_DC = np.exp(-(np.reshape(self.t_axis, (1,1,-1))/filter_width)**filter_order)

        St_AC = St * filter_AC
        St_DC = St * filter_DC
        
        Sw_AC = self.Ft(St_AC, n_omega, n_fft)
        Sw_DC = self.Ft(St_DC, n_omega, n_fft)
        
        phase = np.angle(Sw_AC * np.exp(1j*self.omega_axis.reshape(1,1,-1) * delay))
        
        a = np.abs(Sw_DC)-2*np.abs(Sw_AC)
        a[a<0] = 0
        Su = (0.5*(np.sqrt(np.abs(Sw_DC)+2*np.abs(Sw_AC))+np.sqrt(a)))**2

        return phase, delay, Su


class FiberArrayBasic(ABC):
    @property
    @abstractmethod
    def x_axis(self) -> np.ndarray:
        """Returns the x-axis of the fiber array."""
        pass
    @property
    @abstractmethod
    def y_axis(self) -> np.ndarray:
        """Returns the y-axis of the fiber array."""
        pass
    @property
    @abstractmethod
    def x_matrix(self) -> np.ndarray:
        """Returns the x matrix of the fiber array."""
        pass
    @property
    @abstractmethod
    def y_matrix(self) -> np.ndarray:
        """Returns the y matrix of the fiber array."""
        pass
    @property
    @abstractmethod
    def fiber_number(self) -> np.ndarray:
        """Returns the number of fibers in the array."""
        pass
    @property
    @abstractmethod
    def number_x(self) -> int:
        """Returns the number of fibers in the x direction."""
        pass
    @property
    @abstractmethod
    def number_y(self) -> int:
        """Returns the number of fibers in the y direction."""
        pass
    
    @abstractmethod
    def __init__(self, dx: float, dy: float) -> None:
        """Initializes the fiber array with the offset dx and dy."""
        pass
    
    def get_fiber_array_properties(self):
        """Returns the properties of the fiber array."""
        return {
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "x_matrix": self.x_matrix,
            "y_matrix": self.y_matrix,
            "fiber_number": self.fiber_number,
            "number_x": self.number_x,
            "number_y": self.number_y
        }