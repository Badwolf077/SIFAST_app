import datetime
import json
import pathlib
import re

import numpy as np


class SerializableEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing objects with a `to_dict` method and
    `pathlib.Path` objects.
    """

    def default(self, obj):
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if isinstance(obj, pathlib.Path):
            return str(obj)
        return super().default(obj)


def _get_next_entry_id(log_filepath: str | pathlib.Path) -> int:
    """
    Get the next entry ID for the processing log.

    Parameters
    ----------
    log_filepath : str | pathlib.Path
        The path to the processing log file.

    Returns
    -------
    int
        The next entry ID.
    """
    log_path = pathlib.Path(log_filepath)
    if not log_path.exists():
        return 1
    content = log_path.read_text(encoding="utf-8")
    ids_found = re.findall(r"^- \*\*Entry ID\*\*: (\d+)", content, re.MULTILINE)
    if not ids_found:
        return 1
    return max(int(id_str) for id_str in ids_found) + 1


def update_processing_log(folder_path: str | pathlib.Path, status: str, params: dict, message: str = "") -> None:
    """
    Update the processing log with the current status and parameters.

    Parameters
    ----------
    folder_path : str | pathlib.Path
        The path to the folder where the log file is located.
    status : str
        The current status of the processing.
    params : dict
        The parameters used for processing.
    message : str, optional
        An optional message to include in the log entry.

    Returns
    -------
    None
    """
    log_filepath = pathlib.Path(folder_path) / "processing_history.md"
    new_id = _get_next_entry_id(log_filepath)
    timestamp = datetime.datetime.now().isoformat()
    params_str = json.dumps(params, indent=4, cls=SerializableEncoder)
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
    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)


def reproduce_from_log(log_file_path: str | pathlib.Path, entry_id: int):
    """
    Reproduce the instance of the SIFAST class from the log file.
    """
    from .sifast import SIFAST
    from .srsi import SRSI

    log_path = pathlib.Path(log_file_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    content = log_path.read_text(encoding="utf-8")
    entries = content.split("---")
    target_entry_text = None
    id_pattern = f"- \\*\\*Entry ID\\*\\*: {entry_id}\\b"

    for entry in entries:
        if re.search(id_pattern, entry):
            target_entry_text = entry
            break

    if target_entry_text is None:
        raise ValueError(f"Entry with ID {entry_id} not found in the log file.")

    match = re.search(r"```json\s*(\{.*?\})\s*```", target_entry_text, re.DOTALL)
    if not match:
        raise ValueError("Could not parse parameters from log entry.")

    params_json_str = match.group(1)
    try:
        params = json.loads(params_json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in log entry parameters.")

    if "reference_pulse" in params and isinstance(params["reference_pulse"], dict):
        ref_params = params["reference_pulse"]
        params["reference_pulse"] = SRSI(**ref_params)

    pulse = SIFAST(**params)
    return pulse


def rescale(x: np.ndarray, new_min: float = 0, new_max: float = 1) -> np.ndarray:
    """
    Rescale the input array x to the range [new_min, new_max].

    Parameters
    ----------
    x : np.ndarray
        The input array to be rescaled.
    new_min : float
        The new minimum value of the rescaled array.
    new_max : float
        The new maximum value of the rescaled array.

    Returns
    -------
    np.ndarray
        The rescaled array.
    """
    old_min = np.nanmin(x)
    old_max = np.nanmax(x)

    if old_max - old_min == 0:
        return np.full_like(x, new_min)

    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
