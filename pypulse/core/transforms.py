"""Fourier transform utilities."""

import numpy as np
import numpy.typing as npt
from numpy.fft import fft, fft2, fftshift, ifft, ifft2, ifftshift


class FourierTransforms:
    """Mixin class providing Fourier transform methods."""

    @staticmethod
    def Ft(Et: npt.NDArray[np.complex128], n_omega: int, n_fft: int) -> npt.NDArray[np.complex128]:
        """Time to frequency domain transform."""
        Ew = fftshift(fft(fftshift(Et, axes=2), axis=2), axes=2)
        start = (n_fft - n_omega) // 2
        end = (n_fft + n_omega) // 2
        return Ew[:, :, start:end]

    @staticmethod
    def iFt(Ew: npt.NDArray[np.complex128], n_omega: int, n_fft: int) -> npt.NDArray[np.complex128]:
        """Frequency to time domain transform."""
        padding_size = (n_fft - n_omega) // 2
        padding = np.zeros((Ew.shape[0], Ew.shape[1], padding_size), dtype=Ew.dtype)
        Ew_padded = np.concatenate((padding, Ew, padding), axis=2)
        return fftshift(ifft(fftshift(Ew_padded, axes=2), axis=2), axes=2)

    @staticmethod
    def F(Exy: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """2D spatial Fourier transform."""
        return ifftshift(ifftshift(fft2(fftshift(fftshift(Exy, axes=1), axes=0)), axes=1), axes=0)

    @staticmethod
    def iF(EXY: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """2D inverse spatial Fourier transform."""
        return ifftshift(ifftshift(ifft2(fftshift(fftshift(EXY, axes=1), axes=0)), axes=1), axes=0)
