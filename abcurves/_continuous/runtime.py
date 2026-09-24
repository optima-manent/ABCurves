"""Causal, batch-one B2-23 streaming without per-sample tensor allocation.

Coordinates are common angular counts; time is an integer microsecond offset.
The runtime returns absolute positions at closed 1 ms endpoints. A newly
received target can affect the next uncommitted 32 ms plan, never an existing
commitment. There is no future-target interpolation or wall-clock sleeping.
"""
from __future__ import annotations

from collections import deque
import time
from typing import Protocol

import numpy as np


COMMON_RADIANS_PER_COUNT = 0.0003509487083280618
SAMPLE_US = 1000
COMMIT_SAMPLES = 32
HISTORY_SAMPLES = 640
WARM_HISTORY_SAMPLES = 160


class Planner(Protocol):
    """Internal boundary between streaming and the stateful B2-23 planner.

    Inputs are chronological, borrowed NumPy views that remain valid only for
    the duration of ``plan`` and must not be mutated or retained. Physical
    inputs use float64, availability uses int64, and masks use bool. ``plan``
    returns exactly 32 one-millisecond displacements with shape ``(32, 2)``.
    The planner owns hybrid policy state and all seeded random generators.
    """

    def reset(self, *, seed: int) -> None: ...

    def plan(
        self,
        history: np.ndarray,
        position: np.ndarray,
        target: np.ndarray,
        available: np.ndarray,
        valid: np.ndarray,
        motion_known: np.ndarray,
        now_us: int,
    ) -> np.ndarray: ...


class RuntimeFailure(FloatingPointError):
    """A failed plan plus the complete output committed before the failure.

    ``at_us`` is the numerical time of the failure. ``partial`` has the same
    ``time_us`` and ``xy`` arrays as a successful ``advance`` call. The runtime
    refuses further advances until explicitly reset.
    """

    def __init__(self, message: str, at_us: int, partial: dict[str, np.ndarray]):
        super().__init__(message)
        self.at_us = at_us
        self.partial = partial


class MovementRuntime:
    """Serve a loaded planner as causal 1 ms position samples.

    ``history`` is an optional chronological ``(160, 2)`` array of prior
    one-millisecond displacements, ending at ``initial_xy``. These 160 samples
    are known, including zero samples; earlier motion is unknown, matching the
    reference warm-start contract. No target history is fabricated.

    ``update_target`` queues a receipt at its stated timestamp. Receipts must
    be nondecreasing and cannot precede committed numerical movement. ``advance``
    returns every newly closed endpoint up to its timestamp; a fractional
    millisecond remainder carries into the next call. Omitting a timestamp
    uses elapsed monotonic time since construction/reset. A target receipt
    between endpoints becomes available at the next endpoint. Queued future
    targets are never exposed to the planner early.

    ``reset()`` repeats the most recent initialization and seed. Explicit
    ``initial_xy``, ``history``, or ``seed`` replace those reset defaults.
    Pass an explicit zero history to discard a previous warm start. Reset
    clears all receipts, pending movement, clocks, policy state, and failure.

    The instance is single-stream and not thread-safe. Internal motion buffers
    have fixed size. Queued receipts and a returned output array scale with the
    caller's queued events and requested output span, respectively. This API
    performs numerical inference; it does not pace output or inject device IO.
    """

    def __init__(
        self,
        planner: Planner,
        *,
        initial_xy=(0.0, 0.0),
        history=None,
        seed: int = 7,
        clock_ns=time.monotonic_ns,
    ):
        self.planner = planner
        self.clock_ns = clock_ns
        self._initial_xy = self._check_position(initial_xy)
        self._initial_history = self._check_history(history)
        self.seed = self._check_seed(seed)
        # Mirrored rings make any chronological 640-sample window contiguous.
        # Each tick writes one slot and its mirror, rather than shifting history.
        self._history = np.empty((2 * HISTORY_SAMPLES, 2), np.float64)
        self._target_history = np.empty_like(self._history)
        self._available = np.empty(2 * HISTORY_SAMPLES, np.int64)
        self._valid = np.empty(2 * HISTORY_SAMPLES, np.bool_)
        self._motion_known = np.empty(2 * HISTORY_SAMPLES, np.bool_)
        self._actions = np.empty((COMMIT_SAMPLES, 2), np.float64)
        self._action_points = np.empty_like(self._actions)
        self._position = np.empty(2, np.float64)
        self._events: deque[tuple[int, np.ndarray]] = deque()
        self.reset()

    @staticmethod
    def _check_position(value) -> np.ndarray:
        point = np.asarray(value, np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Position must be a finite pair in common counts")
        return point.copy()

    @staticmethod
    def _check_history(value) -> np.ndarray:
        history = (
            np.zeros((WARM_HISTORY_SAMPLES, 2), np.float64)
            if value is None
            else np.asarray(value, np.float64)
        )
        if history.shape != (WARM_HISTORY_SAMPLES, 2) or not np.isfinite(history).all():
            raise ValueError("History must contain 160 finite displacement pairs")
        return history.copy()

    @staticmethod
    def _check_seed(value) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("Seed must be an integer")
        value = int(value)
        if not -(1 << 63) <= value < (1 << 64):
            raise ValueError("Seed must be in [-2**63, 2**64 - 1]")
        return value

    def reset(self, *, initial_xy=None, history=None, seed=None) -> None:
        """Restart numerical time at zero and repeat/reset the random stream."""
        position = self._initial_xy if initial_xy is None else self._check_position(initial_xy)
        warm_history = self._initial_history if history is None else self._check_history(history)
        reset_seed = self.seed if seed is None else self._check_seed(seed)
        self.planner.reset(seed=reset_seed)
        self._initial_xy = position.copy()
        self._initial_history = warm_history.copy()
        self.seed = reset_seed
        self.clock_origin = self.clock_ns()
        self.now_us = 0
        self.requested_us = 0
        self.last_receipt_us = -1
        self.receipt_us = 0
        self._events.clear()
        self._target = None
        self._position[:] = position
        self._cursor = 0
        self._history.fill(0)
        self._history[HISTORY_SAMPLES - WARM_HISTORY_SAMPLES : HISTORY_SAMPLES] = warm_history
        self._history[HISTORY_SAMPLES:] = self._history[:HISTORY_SAMPLES]
        self._target_history.fill(0)
        self._available.fill(0)
        self._valid.fill(False)
        self._motion_known.fill(False)
        self._motion_known[HISTORY_SAMPLES - WARM_HISTORY_SAMPLES : HISTORY_SAMPLES] = True
        self._motion_known[HISTORY_SAMPLES:] = self._motion_known[:HISTORY_SAMPLES]
        self._pending_index = COMMIT_SAMPLES
        self._observed = False
        self.fresh_decisions = 0
        self.failed = False

    def _time(self, value) -> int:
        if value is None:
            value = (self.clock_ns() - self.clock_origin) // 1000
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
            or value > np.iinfo(np.int64).max
        ):
            raise ValueError("Time must be a nonnegative int64 microsecond offset")
        return int(value)

    def update_target(self, xy, timestamp_us=None) -> None:
        """Queue a target and its receipt time, without advancing movement."""
        at = self._time(timestamp_us)
        point = np.asarray(xy, np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Target must be a finite pair in common counts")
        if at < max(self.now_us, self.last_receipt_us):
            raise ValueError("Target receipts must be monotonic and not precede committed movement")
        self._events.append((at, point.copy()))
        self.last_receipt_us = at

    def _arrive(self) -> None:
        if not self._events or self._events[0][0] > self.now_us:
            return
        while self._events and self._events[0][0] <= self.now_us:
            self.receipt_us, self._target = self._events.popleft()
        at = (self._cursor + HISTORY_SAMPLES - 1) % HISTORY_SAMPLES
        self._target_history[at] = self._target_history[at + HISTORY_SAMPLES] = self._target
        self._available[at] = self._available[at + HISTORY_SAMPLES] = self.receipt_us
        self._valid[at] = self._valid[at + HISTORY_SAMPLES] = True

    def _append(self, delta: np.ndarray | None) -> None:
        at = self._cursor
        mirror = at + HISTORY_SAMPLES
        self._history[at] = self._history[mirror] = 0.0 if delta is None else delta
        if self._target is None:
            self._target_history[at] = self._target_history[mirror] = 0.0
            self._available[at] = self._available[mirror] = 0
            self._valid[at] = self._valid[mirror] = False
        else:
            self._target_history[at] = self._target_history[mirror] = self._target
            self._available[at] = self._available[mirror] = self.receipt_us
            self._valid[at] = self._valid[mirror] = True
        self._motion_known[at] = self._motion_known[mirror] = True
        self._cursor = (at + 1) % HISTORY_SAMPLES

    def _initialize_observation(self) -> None:
        # The research adapter creates its 640 ms state on the first commanded
        # call, even if an arbitrary uncommanded interval preceded that call.
        older = (self._cursor + np.arange(HISTORY_SAMPLES - WARM_HISTORY_SAMPLES)) % HISTORY_SAMPLES
        for ring in (self._history, self._target_history, self._available, self._valid, self._motion_known):
            ring[older] = 0
            ring[older + HISTORY_SAMPLES] = 0
        self._observed = True

    def _plan(self) -> None:
        if not self._observed:
            self._initialize_observation()
        start, end = self._cursor, self._cursor + HISTORY_SAMPLES
        action = np.asarray(
            self.planner.plan(
                self._history[start:end],
                self._position,
                self._target_history[start:end],
                self._available[start:end],
                self._valid[start:end],
                self._motion_known[start:end],
                self.now_us,
            ),
            dtype=np.float64,
        )
        if action.shape != (COMMIT_SAMPLES, 2) or not np.isfinite(action).all():
            raise FloatingPointError("Policy returned a nonfinite or incomplete action")
        self._actions[:] = action
        # The reference rebases absolute positions on each 8 ms public slice.
        # Retaining that summation boundary avoids needless numerical drift.
        origin = self._position
        with np.errstate(over="ignore", invalid="ignore"):
            for offset in range(0, COMMIT_SAMPLES, 8):
                block = self._action_points[offset : offset + 8]
                np.cumsum(self._actions[offset : offset + 8], axis=0, out=block)
                block += origin
                origin = block[-1]
        if not np.isfinite(self._action_points).all():
            raise FloatingPointError("Nonfinite absolute action path")
        self._pending_index = 0
        self.fresh_decisions += 1

    def advance(self, timestamp_us=None) -> dict[str, np.ndarray]:
        """Return newly closed 1 ms endpoints as ``time_us[int64]``, ``xy[N,2]``.

        Each returned position is float64 in common counts. A call needing a
        fresh plan synchronously computes it before returning its first point.
        There is no extra output-delay queue beyond the existing commitment.
        """
        if self.failed:
            raise RuntimeError("Runtime has failed; inspect retained state or reset before advancing")
        until = self._time(timestamp_us)
        if until < self.requested_us:
            raise ValueError("Advance clock must be monotonic")
        self.requested_us = until
        count = (until - self.now_us) // SAMPLE_US
        points = np.empty((count, 2), np.float64)
        times = self.now_us + np.arange(1, count + 1, dtype=np.int64) * SAMPLE_US
        self._arrive()
        for output_at in range(count):
            if self._pending_index == COMMIT_SAMPLES:
                if self._target is None:
                    # With no target, preserve numerical position and append a
                    # known zero displacement. Do not consume policy randomness.
                    self._append(None)
                    self.now_us += SAMPLE_US
                    self._arrive()
                    points[output_at] = self._position
                    continue
                try:
                    self._plan()
                except Exception as exc:
                    self.failed = True
                    partial = dict(time_us=times[:output_at].copy(), xy=points[:output_at].copy())
                    raise RuntimeFailure(f"Policy action failed: {exc}", self.now_us, partial) from exc
            at = self._pending_index
            self._position[:] = self._action_points[at]
            self._append(self._actions[at])
            self._pending_index += 1
            self.now_us += SAMPLE_US
            self._arrive()
            points[output_at] = self._position
        return dict(time_us=times, xy=points)

    @property
    def current_xy(self) -> np.ndarray:
        """An independent copy of the latest committed absolute position."""
        return self._position.copy()

    @property
    def needs_plan(self) -> bool:
        """Whether the next emitted point requires a fresh policy decision.

        A queued receipt at the current numerical endpoint is also considered,
        even if ``advance`` has not yet installed it into target history.
        """
        has_target = self._target is not None or bool(self._events and self._events[0][0] <= self.now_us)
        return self._pending_index == COMMIT_SAMPLES and has_target

    @property
    def decisions(self) -> int:
        """Alias for ``fresh_decisions`` (32 ms plans, excluding buffer reads)."""
        return self.fresh_decisions
