"""Historical optional causal-EMA view, retained for archive equality only."""

import numpy as np

def _as_dxdy(dxdy: np.ndarray) -> np.ndarray:
    arr = np.asarray(dxdy)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("dxdy must have shape (T, 2)")
    return arr.astype(np.float64, copy=False)

def causal_ema_dxdy(dxdy: np.ndarray, *, alpha: float = 0.25) -> np.ndarray:
    """A cheap strictly causal smoothed-prefix view.

    This is stored as an input ablation alongside the untouched raw prefix.  It
    is not used to select B and is not interchangeable with the centered
    offline smoother used to construct planner targets.
    """

    arr = _as_dxdy(dxdy)
    a = float(alpha)
    if not 0.0 < a <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    if len(arr) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    # Closed form of y[t] = a*x[t] + (1-a)*y[t-1], y[0] = x[0].
    # These two short convolutions run in NumPy's compiled loop (~10-15 us
    # for a 160-tick prefix) instead of spending hundreds of microseconds in
    # Python. This keeps the causal-smoothed planner-input arm viable in the
    # real-time budget without changing its numerical contract.
    decay = 1.0 - a
    n = len(arr)
    kernel = a * np.power(decay, np.arange(n, dtype=np.float64))
    initial = np.power(decay, np.arange(1, n + 1, dtype=np.float64))
    out = np.empty_like(arr, dtype=np.float64)
    for axis in range(2):
        out[:, axis] = (
            np.convolve(arr[:, axis], kernel, mode="full")[:n]
            + initial * arr[0, axis]
        )
    return out.astype(np.float32)
