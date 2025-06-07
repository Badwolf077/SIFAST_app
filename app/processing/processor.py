"""
Processing thread for PyPulse calculations.
"""

from typing import Any

from PySide6.QtCore import QThread, Signal

import pypulse


class ProcessingThread(QThread):
    """Thread for running pypulse processing without blocking UI."""

    progress = Signal(int)
    status = Signal(str, str)  # message, level
    error = Signal(str)
    finished = Signal(object)  # Returns processed pulse object

    def __init__(self, params: dict[str, Any], mode: str = "single"):
        super().__init__()
        self.params = params
        self.mode = mode
        self.pulse = None

    def run(self):
        """Run the processing in a separate thread."""
        try:
            if self.mode == "single":
                self._process_single()
            elif self.mode == "scan":
                self._process_scan()

            self.finished.emit(self.pulse)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)

    def _process_single(self):
        """Process single measurement."""
        self.status.emit("Processing single measurement...", "INFO")

        # Create processing config
        config_params = {k: v for k, v in self.params.items() if k not in ["folder_path", "reference_pulse"]}

        # Always set mode_input to "read" for folder processing
        config_params["mode_input"] = "read"

        # Check if we need reference pulse
        if self.params.get("mode_acquire") == "triple":
            # TODO: Handle reference pulse loading
            reference_pulse = None
        else:
            reference_pulse = None

        # Process
        folder_path = self.params.get("folder_path")
        if folder_path:
            self.pulse = pypulse.SIFAST(folder_path=folder_path, reference_pulse=reference_pulse, **config_params)
            self.status.emit("Processing completed successfully", "SUCCESS")
        else:
            raise ValueError("No folder path provided")

    def _process_scan(self):
        """Process scan data with spatial merging."""
        self.status.emit("Processing scan data...", "INFO")

        # TODO: Implement scan processing
        # This would involve:
        # 1. Finding all measurement folders in scan directory
        # 2. Processing each one individually
        # 3. Using pypulse.merge_spatial_scans to combine them

        raise NotImplementedError("Scan processing not yet implemented")
