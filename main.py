from pathlib import Path

import pypulse


def process_from_log():
    log_dir = Path("./data/SIFAST/20241212/l=1/low resolution/")
    entry_id = 1  # Specify the entry ID to reproduce
    pulse = pypulse.reproduce_from_log(log_dir, entry_id)
    return pulse


def process_from_dir():
    pypulse.register_fiber_array(
        "Fiber_array_14x14_1.1",
        {
            "type": "rectangular_14x14",
            "nx": 14,
            "ny": 14,
            "spacing": 1.1,
            "description": "Default 14x14 rectangular array",
        },
    )

    config = pypulse.ProcessingConfig(
        mode_input="read",
        mode_acquire="triple",
        gate_noise_intensity=200.0,
        wavelength_center=793.0,
        wavelength_width=100.0,
        n_omega=2048,
        n_fft=65536,
        mode_fiber_position="calibration",
        method="linear",
        fiber_array_id="Fiber_array_14x14_1.1",
        dx=0.0,
        dy=0.0,
        as_calibration=False,
        delay_min=3000,
    )
    n_iteration: int = 30
    folder_path_SIFAST = Path("./data/SIFAST/20241212/l=1/low resolution/")
    folder_path_SRSI = Path("data/SRSI/20231226/参考标定")

    reference_pulse = pypulse.SRSI(
        folder_path=folder_path_SRSI,
        mode_acquire=config.mode_acquire,
        wavelength_center=config.wavelength_center,
        wavelength_width=config.wavelength_width,
        n_omega=config.n_omega,
        n_fft=config.n_fft,
        method=config.method,
        n_iteration=n_iteration,
    )

    pulse = pypulse.SIFAST(
        folder_path=folder_path_SIFAST,
        reference_pulse=reference_pulse,
        **config.to_dict(),
    )
    return pulse


if __name__ == "__main__":
    from_log = True
    if from_log:
        pulse = process_from_log()
    else:
        pulse = process_from_dir()
    # pulse.plot_scatter(pulse.pulse_front)
    pulse.plot_isosurface(-250, 700, 0, 0.05, opacity=0.9, zoom=None)
