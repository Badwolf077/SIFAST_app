"""Processing history logging functionality."""

import datetime
import json
import re
from pathlib import Path
from typing import Any


class SerializableEncoder(json.JSONEncoder):
    """JSON encoder for custom objects."""

    def default(self, obj):
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _get_next_entry_id(log_filepath: str | Path) -> int:
    """Get next available entry ID."""
    log_path = Path(log_filepath)
    if not log_path.exists():
        return 1

    content = log_path.read_text(encoding="utf-8")
    ids = re.findall(r"^- \*\*Entry ID\*\*: (\d+)", content, re.MULTILINE)

    return max(int(id_str) for id_str in ids) + 1 if ids else 1


def update_processing_log(folder_path: str | Path, status: str, params: dict[str, Any], message: str = "") -> None:
    """
    Update processing history log.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing log file
    status : str
        Processing status ('SUCCESS' or 'FAILURE')
    params : dict
        Processing parameters
    message : str
        Status message
    """
    log_filepath = Path(folder_path) / "processing_history.md"
    new_id = _get_next_entry_id(log_filepath)
    timestamp = datetime.datetime.now().isoformat()

    # Serialize parameters
    params_str = json.dumps(params, indent=4, cls=SerializableEncoder, ensure_ascii=False)

    # Build log entry
    log_entry = f"""
---
## Processing Entry (ID: {new_id}) - {status}

- **Entry ID**: {new_id}
- **Timestamp**: `{timestamp}`

### Parameters Used
```json
{params_str}
```
"""

    # Add status-specific information
    if status.upper() == "SUCCESS":
        log_entry += f"""
### Result
- **Details**: {message if message else "Processing completed successfully."}
"""
    elif status.upper() == "FAILURE":
        log_entry += f"""
### Error Details
```
{message}
```
"""

    # Append to log file
    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)


def reproduce_from_log(log_dir: str | Path, entry_id: int | None = None) -> Any:
    """
    Reproduce processing from log entry.

    Parameters
    ----------
    log_file_path : str or Path
        Path to log file
    entry_id : int
        Entry ID to reproduce

    Returns
    -------
    SIFAST or SRSI instance
    """
    from ..fiber.registry import register_fiber_array
    from ..processing.sifast import SIFAST
    from ..processing.srsi import SRSI

    log_path = Path(log_dir) / "processing_history.md"
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if entry_id is None:
        # Get the ID of the last entry
        last_id = _get_next_entry_id(log_path) - 1
        if last_id < 1:
            raise ValueError("No entries found in the log file to reproduce.")
        entry_id = last_id

    content = log_path.read_text(encoding="utf-8")
    entries = content.split("---")

    # Find entry with matching ID
    target_entry = None
    id_pattern = f"- \\*\\*Entry ID\\*\\*: {entry_id}\\b"

    for entry in entries:
        if re.search(id_pattern, entry):
            target_entry = entry
            break

    if target_entry is None:
        raise ValueError(f"Entry with ID {entry_id} not found")

    # Extract JSON parameters
    match = re.search(r"```json\s*(\{.*?\})\s*```", target_entry, re.DOTALL)
    if not match:
        raise ValueError("Could not parse parameters from log entry")

    params = json.loads(match.group(1))

    # Handle fiber array configuration
    if "fiber_array_config" in params:
        fiber_config = params.pop("fiber_array_config")
        fiber_array_id = params.get("fiber_array_id", "custom_from_log")
        # If it's a full config (not just an ID), register it
        if isinstance(fiber_config, dict) and "nx" in fiber_config:
            # Register the array configuration
            register_fiber_array(fiber_array_id, fiber_config, auto_save=False)

    # Handle reference pulse reconstruction
    if "reference_pulse" in params and isinstance(params["reference_pulse"], dict):
        ref_params = params["reference_pulse"]
        params["reference_pulse"] = SRSI(**ref_params)

    # Create instance
    return SIFAST(**params)
