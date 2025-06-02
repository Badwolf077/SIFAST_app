"""Fiber array registry for managing configurations."""

import json
from pathlib import Path
from typing import Any

from .array import FiberArray


class FiberArrayRegistry:
    """Registry for fiber array configurations."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize registry.

        Parameters
        ----------
        config_dir : Path, optional
            Directory containing fiber array configurations
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config" / "fiber_array"
        self._arrays: dict[str, dict[str, Any]] = {}
        self._load_default_arrays()
        if config_dir:
            self._load_custom_arrays()

    def _load_default_arrays(self) -> None:
        """Load default array configurations."""
        # Default 14x14 array
        self._arrays["default_14x14"] = {
            "type": "rectangular_14x14",
            "nx": 14,
            "ny": 14,
            "spacing": 1.1,
            "description": "Default 14x14 rectangular array",
        }

    def _load_custom_arrays(self) -> None:
        """Load custom array configurations from config directory."""
        if not self.config_dir.exists():
            return

        array_files = self.config_dir.glob("*.json")
        for file in array_files:
            try:
                with open(file) as f:
                    config = json.load(f)
                    array_id = file.stem
                    self._arrays[array_id] = config
            except Exception as e:
                print(f"Warning: Could not load array config {file}: {e}")

    def register_array(self, array_id: str, config: dict[str, Any], auto_save: bool = True) -> None:
        """
        Register a new array configuration.

        Parameters
        ----------
        array_id : str
            Array identifier
        config : dict
            Array configuration
        auto_save : bool
            Automatically save to config directory
        """
        self._arrays[array_id] = config

        # Auto-save if config directory is set
        if auto_save and self.config_dir:
            try:
                self.save_array_config(array_id, config)
            except Exception as e:
                print(f"Warning: Could not auto-save array config: {e}")

    def get_array(self, array_id: str, dx: float = 0, dy: float = 0) -> FiberArray:
        """
        Get fiber array instance.

        Parameters
        ----------
        array_id : str
            Array identifier
        dx, dy : float
            Position offsets

        Returns
        -------
        FiberArray
            Configured fiber array instance
        """
        if array_id not in self._arrays:
            raise ValueError(f"Unknown array ID: {array_id}")

        config = self._arrays[array_id].copy()
        return FiberArray(config, dx, dy)

    def get_array_config(self, array_id: str) -> dict[str, Any]:
        """
        Get array configuration.

        Parameters
        ----------
        array_id : str
            Array identifier

        Returns
        -------
        dict
            Array configuration
        """
        if array_id not in self._arrays:
            raise ValueError(f"Unknown array ID: {array_id}")

        return self._arrays[array_id].copy()

    def list_arrays(self) -> dict[str, str]:
        """List available array configurations."""
        return {array_id: config.get("description", "No description") for array_id, config in self._arrays.items()}

    def save_array_config(self, array_id: str, config: dict[str, Any]) -> None:
        """Save array configuration to file."""
        if not self.config_dir:
            raise ValueError("No config directory specified")

        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.config_dir / f"{array_id}.json"

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        self._arrays[array_id] = config


# Global registry instance
_registry = FiberArrayRegistry()


def get_fiber_array(array_id: str = "default_14x14", dx: float = 0, dy: float = 0) -> FiberArray:
    """Get fiber array from global registry."""
    return _registry.get_array(array_id, dx, dy)


def get_fiber_array_config(array_id: str) -> dict[str, Any]:
    """Get fiber array configuration from global registry."""
    return _registry.get_array_config(array_id)


def register_fiber_array(array_id: str, config: dict[str, Any], auto_save: bool = True) -> None:
    """Register array in global registry with auto-save option."""
    _registry.register_array(array_id, config, auto_save)


def set_fiber_array_config_dir(config_dir: Path) -> None:
    """Set the configuration directory for the global registry."""
    global _registry
    _registry.config_dir = config_dir
