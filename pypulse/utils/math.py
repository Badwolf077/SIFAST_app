"""Mathematical utilities."""

import numpy as np
import numpy.typing as npt


def rescale(x: npt.NDArray[np.float64], new_min: float = 0, new_max: float = 1) -> npt.NDArray[np.float64]:
    """
    Rescale array to new range.

    Parameters
    ----------
    x : array_like
        Input array
    new_min : float
        New minimum value
    new_max : float
        New maximum value

    Returns
    -------
    array_like
        Rescaled array
    """
    x = np.asarray(x, dtype=np.float64)
    old_min = np.nanmin(x)
    old_max = np.nanmax(x)

    if old_max - old_min == 0:
        return np.full_like(x, new_min)

    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
