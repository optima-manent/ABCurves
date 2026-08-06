"""Causal session-state hand-off for the Renderer C adapter.

The C adapter consumes three standardized Stage-0 texture scores, named
``C``, ``M`` and ``H``.  For a human event at position *t*, the state used by
the next event is half the mean of the ten preceding human scores in the same
contiguous semantic run, clipped to ``[-2.5, 2.5]``.  The current event is
never part of its own state.

This module owns only the temporal contract.  A project-specific Stage-0
scorer may call :meth:`CausalStyleState.observe_human`; applications without
that scorer should leave the state at its exact-zero fallback.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
import threading

import numpy as np


STATE_FEATURE_NAMES = ("block_state_C", "block_state_M", "block_state_H")
SUPPORT_EVENTS = 10
STATE_SHRINKAGE = 0.5
STATE_CLIP = 2.5


def zero_causal_state() -> np.ndarray:
    """Return the byte-stable fallback used before ten human observations."""

    return np.zeros(3, dtype=np.float32)


def validate_causal_state(value: Iterable[float] | np.ndarray) -> np.ndarray:
    """Copy and validate a precomputed C/M/H state for one event."""

    state = np.array(value, dtype=np.float32, order="C", copy=True)
    if state.shape != (3,) or not np.all(np.isfinite(state)):
        raise ValueError("causal Renderer state must be three finite C/M/H values")
    if float(np.max(np.abs(state))) > STATE_CLIP:
        raise ValueError(f"causal Renderer state must lie within +/-{STATE_CLIP}")
    return state


class CausalStyleState:
    """Track the exact trailing-N10, same-run Renderer state contract.

    Call :meth:`before_event` before generating an event.  After a genuine
    human event has completed and a Stage-0 scorer has produced its three
    standardized scores, call :meth:`observe_human`.  Generated events must
    never be fed back into this buffer.
    """

    def __init__(self) -> None:
        self._run_id: Hashable | None = None
        self._scores: deque[np.ndarray] = deque(maxlen=SUPPORT_EVENTS)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._run_id = None
            self._scores.clear()

    def _select_run(self, run_id: Hashable) -> None:
        if self._run_id != run_id:
            self._run_id = run_id
            self._scores.clear()

    def before_event(self, run_id: Hashable) -> np.ndarray:
        """Return the state based only on earlier human events in ``run_id``."""

        with self._lock:
            self._select_run(run_id)
            if len(self._scores) < SUPPORT_EVENTS:
                return zero_causal_state()
            mean = np.mean(np.stack(tuple(self._scores)), axis=0, dtype=np.float64)
            state = np.clip(STATE_SHRINKAGE * mean, -STATE_CLIP, STATE_CLIP)
            return np.asarray(state, dtype=np.float32)

    def observe_human(
        self,
        run_id: Hashable,
        stage0_scores: Iterable[float] | np.ndarray,
    ) -> None:
        """Append one completed human event's standardized C/M/H scores."""

        scores = np.array(stage0_scores, dtype=np.float64, order="C", copy=True)
        if scores.shape != (3,) or not np.all(np.isfinite(scores)):
            raise ValueError("Stage-0 scores must be three finite C/M/H values")
        with self._lock:
            self._select_run(run_id)
            self._scores.append(scores)

    @property
    def support(self) -> int:
        with self._lock:
            return len(self._scores)


__all__ = [
    "CausalStyleState",
    "STATE_CLIP",
    "STATE_FEATURE_NAMES",
    "STATE_SHRINKAGE",
    "SUPPORT_EVENTS",
    "validate_causal_state",
    "zero_causal_state",
]
