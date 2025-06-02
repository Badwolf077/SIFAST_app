"""Configuration management."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.serialization import load_json, save_json


@dataclass
class ProcessingConfig:
    """Processing configuration parameters."""

    # Acquisition settings
    mode_input: str = "read"
    mode_acquire: str = "triple"
    gate_noise_intensity: float = 200.0

    # Wavelength settings
    wavelength_center: float = 800.0
    wavelength_width: float = 100.0

    # FFT settings
    n_omega: int = 2048
    n_fft: int = 65536

    # Fiber array settings
    fiber_array_id: str = "default_14x14"
    mode_fiber_position: str = "calibration"
    dx: float = 0.0
    dy: float = 0.0

    # Processing settings
    method: str = "linear"
    delay_min: float | None = None
    as_calibration: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingConfig":
        """Create from dictionary."""
        return cls(**data)

    def save(self, filepath: Path) -> None:
        """Save configuration to file."""
        save_json(self.to_dict(), filepath)

    @classmethod
    def load(cls, filepath: Path) -> "ProcessingConfig":
        """Load configuration from file."""
        data = load_json(filepath)
        return cls.from_dict(data)


class ConfigManager:
    """Manage processing configurations."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize config manager.

        Parameters
        ----------
        config_dir : Path, optional
            Configuration directory
        """
        self.config_dir = config_dir or Path.home() / ".pypulse" / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, name: str, config: ProcessingConfig) -> None:
        """Save named configuration."""
        filepath = self.config_dir / f"{name}.json"
        config.save(filepath)

    def load_config(self, name: str) -> ProcessingConfig:
        """Load named configuration."""
        filepath = self.config_dir / f"{name}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")
        return ProcessingConfig.load(filepath)

    def list_configs(self) -> list[str]:
        """List available configurations."""
        return [f.stem for f in self.config_dir.glob("*.json")]
