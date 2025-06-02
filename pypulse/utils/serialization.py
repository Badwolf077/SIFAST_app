"""Serialization utilities."""

import json
from pathlib import Path
from typing import Any


class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for custom types."""

    def default(self, obj: Any) -> Any:
        # Handle objects with to_dict method
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return {"__class__": obj.__class__.__name__, **obj.to_dict()}

        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)

        # Handle numpy arrays
        if hasattr(obj, "tolist"):
            return obj.tolist()

        return super().default(obj)


def save_json(data: Any, filepath: Path, **kwargs) -> None:
    """Save data to JSON file with enhanced encoding."""
    with open(filepath, "w") as f:
        json.dump(data, f, cls=EnhancedJSONEncoder, indent=2, **kwargs)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath) as f:
        return json.load(f)
