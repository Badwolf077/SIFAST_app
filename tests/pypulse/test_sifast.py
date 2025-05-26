import unittest
import numpy as np
import pathlib

# Adjust the import path if necessary, assuming pypulse is in the python path
from pypulse.sifast import SIFAST
from pypulse.srsi import SRSI # SIFAST can take an SRSI object

# Helper to get project root, assuming this test file is in tests/pypulse/
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

class TestSIFASTInstantiation(unittest.TestCase):
    def setUp(self):
        """
        Set up dummy configuration files if they don't exist.
        SIFAST (mode='acquire', config_folder_path=None) expects:
        <PROJECT_ROOT>/config/setting_fiber_calibration.csv
        """
        self.config_dir = PROJECT_ROOT / "config"
        self.config_dir.mkdir(exist_ok=True) # Ensure config dir exists

        self.fiber_calib_file = self.config_dir / "setting_fiber_calibration.csv"
        if not self.fiber_calib_file.exists():
            # Create a dummy file if it wasn't created by a previous step or user
            with open(self.fiber_calib_file, "w") as f:
                f.write("10\n") # Dummy pixel_position_of_fiber or similar
                f.write("20\n")
                f.write("30\n")
        
        # For _calculate_phase_properties, it might try to load reference_parameters.json
        self.ref_params_file = self.config_dir / "reference_parameters.json"
        if not self.ref_params_file.exists():
            import json
            dummy_ref_params = {"x0": 0, "y0": 0, "L": 1, "tau0": 0}
            with open(self.ref_params_file, "w") as f:
                json.dump(dummy_ref_params, f)


    def test_sifast_instantiation_acquire_mode_double_calib(self):
        """
        Test SIFAST instantiation with mode_input='acquire', mode_acquire='double',
        and mode_fiber_position='calibration'.
        """
        params = {
            "mode_input": "acquire",
            "mode_acquire": "double", # Requires image_unknown
            "gate_noise_intensity": 0.1,
            "wavelength_center": 800e-9,
            "wavelength_width": 100e-9,
            "n_omega": 128, # Number of points for spectral axis
            "n_fft": 256,   # Number of points for FFT
            "mode_fiber_position": "calibration", # Uses setting_fiber_calibration.csv
            "method_interpolation": "linear",
            "dx": 0.1, # fiber array property
            "dy": 0.1, # fiber array property
            "reference_pulse": None,
            "as_calibration": False, # Will try to load reference_parameters.json
            "config_folder_path": None, # Uses default <project_root>/config
            "delay_min": None,
            # Kwargs for mode_input='acquire'
            "image_interference": np.random.rand(50, 10), # Dummy: num_pixels_y, num_spectral_points
            "image_unknown": np.random.rand(50, 10),      # Dummy: num_pixels_y, num_spectral_points
            "wavelength": np.linspace(750e-9, 850e-9, 10) # Dummy: 1D array, num_spectral_points
        }

        try:
            sifast_instance = SIFAST(**params)
            self.assertIsInstance(sifast_instance, SIFAST)
        except Exception as e:
            self.fail(f"SIFAST instantiation failed with mode_input='acquire' (double, calib): {e}")

    def test_sifast_instantiation_acquire_mode_triple_calib(self):
        """
        Test SIFAST instantiation with mode_input='acquire', mode_acquire='triple',
        and mode_fiber_position='calibration'.
        """
        params = {
            "mode_input": "acquire",
            "mode_acquire": "triple", # Requires image_unknown and image_reference
            "gate_noise_intensity": 0.1,
            "wavelength_center": 800e-9,
            "wavelength_width": 100e-9,
            "n_omega": 128,
            "n_fft": 256,
            "mode_fiber_position": "calibration",
            "method_interpolation": "linear",
            "dx": 0.1,
            "dy": 0.1,
            "reference_pulse": None,
            "as_calibration": False,
            "config_folder_path": None,
            "delay_min": None,
            "image_interference": np.random.rand(50, 10),
            "image_unknown": np.random.rand(50, 10),
            "image_reference": np.random.rand(50, 10), # Added for triple mode
            "wavelength": np.linspace(750e-9, 850e-9, 10)
        }

        try:
            sifast_instance = SIFAST(**params)
            self.assertIsInstance(sifast_instance, SIFAST)
        except Exception as e:
            self.fail(f"SIFAST instantiation failed with mode_input='acquire' (triple, calib): {e}")
            
    def test_sifast_instantiation_acquire_mode_single_calib(self):
        """
        Test SIFAST instantiation with mode_input='acquire', mode_acquire='single',
        and mode_fiber_position='calibration'.
        """
        params = {
            "mode_input": "acquire",
            "mode_acquire": "single", # Does not require image_unknown or image_reference
            "gate_noise_intensity": 0.1,
            "wavelength_center": 800e-9,
            "wavelength_width": 100e-9,
            "n_omega": 128,
            "n_fft": 256,
            "mode_fiber_position": "calibration",
            "method_interpolation": "linear",
            "dx": 0.1,
            "dy": 0.1,
            "reference_pulse": None,
            "as_calibration": False,
            "config_folder_path": None,
            "delay_min": None,
            "image_interference": np.random.rand(50, 10), 
            # No image_unknown or image_reference for single mode
            "wavelength": np.linspace(750e-9, 850e-9, 10)
        }

        try:
            sifast_instance = SIFAST(**params)
            self.assertIsInstance(sifast_instance, SIFAST)
        except Exception as e:
            self.fail(f"SIFAST instantiation failed with mode_input='acquire' (single, calib): {e}")


if __name__ == '__main__':
    unittest.main()
