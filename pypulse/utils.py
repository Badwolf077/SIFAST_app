import numpy as np

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
    old_min = np.min(x)
    old_max = np.max(x)
    
    if old_max - old_min == 0:
        return np.full_like(x, new_min)
    
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min