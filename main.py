import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

import pypulse

if __name__ == "__main__":
    mode_input: str = "read"
    mode_acquire: str = "triple"
    gate_noise_intensity: float = 200
    wavelength_center: float = 793
    wavelength_width: float = 100
    n_omega: int = 2048
    n_fft: int = 65536
    # config_folder_path: str = "/Users/xuyilin/科研/时空耦合测量/code_20241227/config/20250421"
    delay_min: float | None = 3000
    folder_path = "/Users/xuyilin/科研/时空耦合测量/20241212/l=1/low resolution"
    mode_fiber_position: str = "calibration"

    pulse = pypulse.SIFAST(
        mode_input,
        mode_acquire,
        gate_noise_intensity,
        wavelength_center,
        wavelength_width,
        n_omega,
        n_fft,
        # config_folder_path=config_folder_path,
        delay_min=delay_min,
        folder_path=folder_path,
        mode_fiber_position=mode_fiber_position,
    )
    # pulse.plot_scatter(pulse.pulse_front)
    pulse.plot_isosurface(-250, 700, 0, 0.05, opacity=1)
