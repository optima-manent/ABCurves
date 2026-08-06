"""Loading the AIM-session dataset.

An ABCurves dataset is an ``.npz`` of recorded aim-trainer movements, each cut
into an A->B prefix and a B->C future, all at 1 kHz in mouse **count** space
(hardware dx/dy, before any cursor sensitivity is applied).  The example
dataset under ``examples/`` ships in this format; the columns below are all the
shipped training and inference code reads.

Per-event columns (``N`` events):

===============================  ==============================  =================================
column                           shape                           meaning
===============================  ==============================  =================================
``prefix_raw_dxdy``              ``[N, P, 2]`` float32           A->B counts (1 per ms), left-padded
``prefix_mask``                  ``[N, P]``    float32           1 where the prefix tick is valid
``future_raw_dxdy``             ``[N, H, 2]`` float32           B->C counts (the ground truth)
``future_mask``                  ``[N, H]``    float32           1 where the future tick is valid
``future_smooth_dxdy``          ``[N, H, 2]`` float32           smoothed B->C (planner targets)
``target_rel_x_at_B`` / ``_y``   ``[N]``       float32           target center minus cursor at B
``target_radius``                ``[N]``       float32           target radius (counts)
``progress``                     ``[N]``       float32           fraction of A->target covered at B
``speed_at_B`` / ``accel_at_B``  ``[N]``       float32           kinematic state at the cut
``user_id`` / ``outcome``        ``[N]``       str               which user; hit_click / hit_dwell
===============================  ==============================  =================================

The prefix is right-aligned inside its ``P`` window (the cut B is always the
last valid tick), so ``prefix_raw_dxdy[i][prefix_mask[i] > 0.5]`` is the true
variable-length A->B stream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUCCESS_OUTCOMES = ("hit_click", "hit_dwell")


def load_dataset(path: str | Path, *, success_only: bool = False) -> dict[str, np.ndarray]:
    """Load an ABCurves ``.npz`` dataset into a plain dict of arrays.

    ``success_only`` keeps only successful (hit) events -- the planner is
    trained on these (a failed/timed-out movement is not a movement you want to
    imitate).
    """

    z = dict(np.load(Path(path), allow_pickle=False))
    if success_only and "outcome" not in z:
        raise ValueError(
            f"{path} has no canonical 'outcome' column; refusing to pretend "
            "success_only filtering was applied"
        )
    if success_only:
        keep = np.isin(z["outcome"].astype(str), SUCCESS_OUTCOMES)
        z = subset(z, keep)
    return z


def subset(arrays: dict[str, np.ndarray], mask_or_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Select a subset of events, leaving non-event metadata columns intact."""

    n = len(arrays["future_mask"])
    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[mask_or_idx]
        else:
            out[key] = arr
    return out


def prefix_of(arrays: dict[str, np.ndarray], idx: int) -> np.ndarray:
    """The variable-length A->B stream for event ``idx`` (``[len, 2]``)."""

    prefix = arrays["prefix_raw_dxdy"][idx]
    mask = arrays.get("prefix_mask")
    if mask is not None:
        prefix = prefix[mask[idx] > 0.5]
    return np.asarray(prefix, dtype=np.float32)


def future_of(arrays: dict[str, np.ndarray], idx: int, *, smooth: bool = False) -> np.ndarray:
    """The variable-length B->C stream for event ``idx`` (``[len, 2]``)."""

    key = "future_smooth_dxdy" if smooth else "future_raw_dxdy"
    n = int(arrays["future_mask"][idx].sum())
    return np.asarray(arrays[key][idx, :n], dtype=np.float32)


def target_rel_at_B(arrays: dict[str, np.ndarray], idx: int) -> tuple[float, float]:
    return (float(arrays["target_rel_x_at_B"][idx]), float(arrays["target_rel_y_at_B"][idx]))


__all__ = ["load_dataset", "subset", "prefix_of", "future_of", "target_rel_at_B", "SUCCESS_OUTCOMES"]
