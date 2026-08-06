"""Causal movement-onset and B-trigger state machines.

These are the live counterparts of :mod:`abcurves.capture_preprocess`.  They
operate on closed 1 ms count bins and use the exact contract embedded in the
release Planner: sustained target-aligned onset, then B at 80% of the distance
to the target edge.  The Planner receives center-referenced progress even
though the trigger itself is edge-referenced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .capture_preprocess import CausalBConfig, CausalOnsetConfig


@dataclass(frozen=True)
class OnsetEvent:
    index: int
    reason: str
    threshold: float
    speed_median: float
    speed_mad: float


@dataclass(frozen=True)
class _OnsetTick:
    index: int
    speed: float
    aligned: bool


class OnsetDetector:
    """Detect A causally from 1 kHz counts and the current target vector."""

    def __init__(self, config: CausalOnsetConfig | None = None) -> None:
        self.config = config or CausalOnsetConfig()
        self.reset()

    @classmethod
    def from_seam_contract(cls, contract: Mapping[str, Any]) -> "OnsetDetector":
        record = dict(contract.get("onset", {}))
        return cls(CausalOnsetConfig(**record))

    def reset(self) -> None:
        self._baseline_speeds: list[float] = []
        self._pending: list[_OnsetTick] = []
        self._threshold: float | None = None
        self._median = 0.0
        self._mad = 0.0
        self._run = 0
        self._tick_count = 0
        self._done = False

    def push(
        self,
        dx: float,
        dy: float,
        target_rel: tuple[float, float] | np.ndarray | None,
    ) -> OnsetEvent | None:
        """Consume one closed 1 ms bin; return A once, then remain latched."""

        if self._done:
            return None
        dx_value, dy_value = float(dx), float(dy)
        speed = float(math.hypot(dx_value, dy_value))
        if target_rel is None:
            aligned = True
        else:
            target = np.asarray(target_rel, dtype=np.float64).reshape(-1)
            if target.shape != (2,) or not np.all(np.isfinite(target)):
                raise ValueError("target_rel must contain two finite values")
            target_norm = float(np.linalg.norm(target))
            denom = speed * target_norm + 1e-6
            aligned = (
                (dx_value * float(target[0]) + dy_value * float(target[1])) / denom
                >= float(self.config.alignment_min)
            )
        tick = _OnsetTick(self._tick_count, speed, bool(aligned))
        self._tick_count += 1

        if self._threshold is None:
            self._baseline_speeds.append(speed)
            self._pending.append(tick)
            if len(self._baseline_speeds) < self.config.noise_window_ms:
                return None
            baseline = np.asarray(self._baseline_speeds, dtype=np.float64)
            self._median = float(np.median(baseline))
            self._mad = float(np.median(np.abs(baseline - self._median)))
            if self.config.quiet_baseline_fallback and self._median > self.config.speed_floor:
                self._threshold = float(self.config.speed_floor)
            else:
                self._threshold = max(
                    float(self.config.speed_floor),
                    self._median
                    + float(self.config.threshold_mad_multiplier) * self._mad,
                )
            pending, self._pending = self._pending, []
            for buffered in pending:
                event = self._scan(buffered)
                if event is not None:
                    return event
            return None
        return self._scan(tick)

    def _scan(self, tick: _OnsetTick) -> OnsetEvent | None:
        preferred = tick.speed > float(self._threshold or 0.0) and tick.aligned
        self._run = self._run + 1 if preferred else 0
        run_length = int(self.config.consecutive_ticks)
        if self._run < run_length:
            return None
        self._done = True
        onset_start = tick.index - run_length + 1
        return OnsetEvent(
            index=max(0, onset_start - int(self.config.backtrack_ms)),
            reason="sustained_aligned_movement",
            threshold=float(self._threshold or 0.0),
            speed_median=self._median,
            speed_mad=self._mad,
        )

    @property
    def fired(self) -> bool:
        return self._done


def _unit2(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-9 else np.zeros(2, dtype=np.float64)


@dataclass(frozen=True)
class BFire:
    t_ms: int
    progress_edge: float
    progress_center: float
    movement_counts: tuple[float, float]
    target_rel_at_B: tuple[float, float]
    target_radius: float
    remaining_counts: float


@dataclass(frozen=True)
class BReject:
    t_ms: int
    reason: str
    progress_center: float


class BTrigger:
    """Track one A->B movement and fire at the frozen edge-progress seam."""

    def __init__(self, config: CausalBConfig | None = None) -> None:
        self.config = config or CausalBConfig(threshold=0.8)
        self._armed = False
        self._movement = np.zeros(2, dtype=np.float64)
        self._target_at_A = np.zeros(2, dtype=np.float64)
        self._radius_at_A = 0.0
        self._t_ms = 0
        self._max_progress_center = 0.0

    @classmethod
    def from_seam_contract(cls, contract: Mapping[str, Any]) -> "BTrigger":
        trigger = dict(contract.get("trigger", {}))
        thresholds = tuple(float(v) for v in trigger.pop("thresholds", ()))
        if len(thresholds) != 1:
            raise ValueError("runtime seam contract must contain one B threshold")
        config = CausalBConfig(
            threshold=thresholds[0],
            trigger_reference=str(trigger.get("reference", "edge")),
            planner_progress_reference=str(
                contract.get("planner_context", {}).get("progress_reference", "center")
            ),
            require_outside_target=bool(trigger.get("require_outside_target", True)),
            max_ab_ms=int(trigger.get("max_ab_ms", 1500)),
            min_remaining_counts=float(trigger.get("min_remaining_counts", 8.0)),
            progress_regression_reset=bool(
                trigger.get("progress_regression_reset", True)
            ),
            progress_regression_threshold=float(
                trigger.get("progress_regression_threshold", 0.18)
            ),
            max_center_progress=float(trigger.get("max_center_progress", 0.92)),
            max_realized_progress=trigger.get("max_realized_progress"),
        )
        return cls(config)

    def arm(
        self,
        target_rel_at_A: tuple[float, float] | np.ndarray,
        target_radius: float,
    ) -> None:
        target = np.asarray(target_rel_at_A, dtype=np.float64).reshape(-1)
        radius = float(target_radius)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError("target_rel_at_A must contain two finite values")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("target_radius must be positive")
        self._armed = True
        self._movement[:] = 0.0
        self._target_at_A = target.copy()
        self._radius_at_A = radius
        self._t_ms = 0
        self._max_progress_center = 0.0

    def disarm(self) -> None:
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def movement_counts(self) -> np.ndarray:
        return self._movement.copy()

    def progress(self, *, reference: str = "edge") -> float:
        distance = float(np.linalg.norm(self._target_at_A))
        denominator = distance
        if reference == "edge":
            denominator -= max(self._radius_at_A, 0.0)
        elif reference != "center":
            raise ValueError("progress reference must be 'edge' or 'center'")
        denominator = max(denominator, 1e-9)
        return float(
            np.dot(self._movement, _unit2(self._target_at_A)) / denominator
        )

    def push_tick(
        self,
        dx: float,
        dy: float,
        *,
        target_rel_now: tuple[float, float] | np.ndarray,
        target_radius_now: float,
    ) -> BFire | BReject | None:
        if not self._armed:
            return None
        target = np.asarray(target_rel_now, dtype=np.float64).reshape(-1)
        radius = float(target_radius_now)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError("target_rel_now must contain two finite values")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("target_radius_now must be positive")

        self._movement += (float(dx), float(dy))
        self._t_ms += 1
        progress_center = self.progress(reference="center")
        progress_edge = self.progress(reference="edge")
        cfg = self.config

        if self._t_ms > cfg.max_ab_ms:
            return self._reject("max_ab_ms", progress_center)
        if progress_center > self._max_progress_center:
            self._max_progress_center = progress_center
        elif (
            cfg.progress_regression_reset
            and self._max_progress_center - progress_center
            > cfg.progress_regression_threshold
        ):
            return self._reject("progress_regression", progress_center)
        if progress_edge < cfg.threshold:
            return None
        if progress_center > cfg.max_center_progress:
            return self._reject("max_center_progress", progress_center)
        remaining = float(np.linalg.norm(target))
        if remaining < cfg.min_remaining_counts:
            return self._reject("min_remaining_counts", progress_center)
        if cfg.require_outside_target and remaining <= radius:
            return self._reject("inside_target", progress_center)

        self._armed = False
        return BFire(
            t_ms=self._t_ms,
            progress_edge=progress_edge,
            progress_center=progress_center,
            movement_counts=(float(self._movement[0]), float(self._movement[1])),
            target_rel_at_B=(float(target[0]), float(target[1])),
            target_radius=radius,
            remaining_counts=remaining,
        )

    def _reject(self, reason: str, progress_center: float) -> BReject:
        self._armed = False
        return BReject(self._t_ms, reason, float(progress_center))


__all__ = ["BFire", "BReject", "BTrigger", "OnsetDetector", "OnsetEvent"]
