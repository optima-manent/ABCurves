"""Continuous target-directed movement in common angular coordinates."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np

from ._continuous.runtime import (
    COMMON_RADIANS_PER_COUNT, MovementRuntime, RuntimeFailure,
)
from .model_store import default_model_dir


@dataclass(frozen=True)
class HumanStart:
    """Causal intent history, its filtered anchor, and the actual observed anchor."""
    initial_xy: np.ndarray
    history: np.ndarray
    observed_xy: np.ndarray


def prepare_history(raw_displacements, *, current_xy) -> HumanStart:
    """Convert 164 real 1 ms common-unit displacements to a 160-sample start.

    The selected training representation filters positions causally with W5
    weights [1,2,3,2,1]/9, then differences them. Four extra genuine samples
    avoid inventing stationary history. ``current_xy`` is the observed position
    after the last input interval; the filtered anchor generally differs.
    Input and output axes are x right / y up.
    """
    raw = np.asarray(raw_displacements, dtype=np.float64)
    current = MovementRuntime._check_position(current_xy)
    if raw.shape != (164, 2) or not np.isfinite(raw).all():
        raise ValueError("raw_displacements must contain exactly 164 finite 1 ms pairs")
    weights = np.array([1., 2., 3., 2., 1.]) / 9.
    # Filtering deltas directly preserves small motion at large global origins.
    # The anchor lag is the same causal position filter, expressed locally.
    with np.errstate(over="ignore", invalid="ignore"):
        history = sum(weights[k] * raw[k:k + 160] for k in range(5))
        lag = (8. * raw[-1] + 6. * raw[-2] + 3. * raw[-3] + raw[-4]) / 9.
        anchor = current - lag
    if not np.isfinite(history).all() or not np.isfinite(anchor).all():
        raise ValueError("History filtering exceeds finite coordinate range")
    return HumanStart(anchor, history, current)


class ContinuousPlanner(MovementRuntime):
    """Load the selected motor, movement selector, brake and stop/restart system.

    Positions use x-right/y-up common angular counts. ``history`` contains 160
    chronological 1 ms displacement samples ending at ``initial_xy``; it is
    optional. Each ``advance`` returns absolute positions at closed endpoints.
    Targets are timestamped receipts, not future path samples. See docs/INTEGRATION.md
    for causal preprocessing and the distinction between receipts and output.
    """

    def __init__(self, assets: str | Path | None = None, *, seed=7,
                 initial_xy=(0., 0.), history=None, backend="native",
                 kernel_backend="numba", prewarm=True, diagnostics=False,
                 skip_unused=True, allow_custom_assets=False):
        from ._continuous.planner import Planner
        from ._continuous.backend import Assets
        if backend not in ("native", "onnx"):
            raise ValueError("backend must be 'native' or 'onnx'")
        bundle = Assets(assets or default_model_dir() / "continuous", allow_custom=allow_custom_assets)
        policy = Planner(bundle,
                         backend=backend, kernel_backend=kernel_backend,
                         diagnostics=diagnostics, skip_unused=skip_unused)
        if prewarm:
            policy.prewarm()
        super().__init__(policy, seed=seed, initial_xy=initial_xy, history=history)

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls(**kwargs)


def load(**kwargs) -> ContinuousPlanner:
    """Load ABCurves' default Continuous Planner, once before the live clock."""
    return ContinuousPlanner(**kwargs)


__all__ = ["ContinuousPlanner", "load", "prepare_history", "HumanStart",
           "RuntimeFailure", "COMMON_RADIANS_PER_COUNT"]
