"""Preprocessing primitives for ABCurves hardware-capture exports.

The capture trainer intentionally exports an interchange view rather than
model-ready A->B/B->C tensors.  This module defines the downstream choices that
must be explicit and reproducible:

* how event timestamps select dense 1 ms bins;
* how progress is measured relative to the *target edge*;
* how a causal B seam is selected;
* which event-level diagnostics describe a clean shot; and
* how multiple B cuts from one source trial are weighted.

The functions are deliberately independent of pandas and filesystem layout so
their geometry and boundary semantics can be unit tested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


EPS = 1e-9
SUCCESS_OUTCOMES = frozenset({"hit_click", "hit_dwell"})
CAUSAL_SEAM_CONTRACT_SCHEMA = "abcurves.causal_onset_b.v1"
B_TRIGGER_REFERENCE = "edge"
PLANNER_PROGRESS_REFERENCE = "center"


@dataclass(frozen=True)
class CausalOnsetConfig:
    """Versioned, deployable movement-onset settings.

    These defaults mirror the live hardware path.  They are a reproducible
    study baseline, not a claim that the values are scientifically final.
    """

    noise_window_ms: int = 24
    consecutive_ticks: int = 12
    backtrack_ms: int = 4
    speed_floor: float = 0.35
    threshold_mad_multiplier: float = 6.0
    alignment_min: float = 0.15
    quiet_baseline_fallback: bool = True

    def __post_init__(self) -> None:
        if self.noise_window_ms < 1 or self.consecutive_ticks < 1:
            raise ValueError("onset window and consecutive ticks must be positive")
        if self.backtrack_ms < 0:
            raise ValueError("onset backtrack must be non-negative")
        if self.speed_floor < 0.0 or self.threshold_mad_multiplier < 0.0:
            raise ValueError("onset speed floor and MAD multiplier must be non-negative")
        if not -1.0 <= self.alignment_min <= 1.0:
            raise ValueError("onset alignment_min must lie in [-1, 1]")


OnsetConfig = CausalOnsetConfig


def causal_onset_contract_record(onset: OnsetConfig) -> dict[str, object]:
    """Serialize the release onset policy."""

    if not isinstance(onset, CausalOnsetConfig):
        raise TypeError(f"unsupported onset config: {type(onset).__name__}")
    return asdict(onset)


def onset_config_from_record(record: dict[str, object]) -> OnsetConfig:
    """Parse and validate a serialized release onset policy."""

    return CausalOnsetConfig(**dict(record))


@dataclass(frozen=True)
class CausalBConfig:
    """The causal A->B trigger contract for one threshold."""

    threshold: float
    trigger_reference: str = B_TRIGGER_REFERENCE
    planner_progress_reference: str = PLANNER_PROGRESS_REFERENCE
    require_outside_target: bool = True
    max_ab_ms: int = 1_500
    min_remaining_counts: float = 8.0
    progress_regression_reset: bool = True
    progress_regression_threshold: float = 0.18
    max_center_progress: float = 0.92
    max_realized_progress: float | None = None

    def __post_init__(self) -> None:
        if self.trigger_reference != B_TRIGGER_REFERENCE:
            raise ValueError(
                f"causal B trigger reference must be {B_TRIGGER_REFERENCE!r}"
            )
        if self.planner_progress_reference != PLANNER_PROGRESS_REFERENCE:
            raise ValueError(
                "planner progress reference must be "
                f"{PLANNER_PROGRESS_REFERENCE!r}"
            )
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("B threshold must be strictly between zero and one")
        if self.max_ab_ms < 1:
            raise ValueError("max_ab_ms must be positive")
        if self.min_remaining_counts < 0.0:
            raise ValueError("min_remaining_counts must be non-negative")
        if self.progress_regression_threshold < 0.0:
            raise ValueError("progress regression threshold must be non-negative")


@dataclass(frozen=True)
class SeamEligibility:
    """Post-B requirements for admitting a frozen seam to a training set."""

    min_prefix_ms: int = 24
    min_future_ms: int = 12

    def __post_init__(self) -> None:
        if self.min_prefix_ms < 0 or self.min_future_ms < 0:
            raise ValueError("minimum prefix/future lengths must be non-negative")


def causal_seam_contract_record(
    onset: OnsetConfig,
    b_configs: Iterable[CausalBConfig],
    eligibility: SeamEligibility,
) -> dict[str, object]:
    """Return the JSON/NPZ-safe contract shared by profiling and deployment."""

    configs = tuple(b_configs)
    if not configs:
        raise ValueError("at least one B threshold must be supplied")
    trigger_shapes = {
        (
            item.trigger_reference,
            item.planner_progress_reference,
            item.require_outside_target,
            item.max_ab_ms,
            item.min_remaining_counts,
            item.progress_regression_reset,
            item.progress_regression_threshold,
            item.max_center_progress,
            item.max_realized_progress,
        )
        for item in configs
    }
    if len(trigger_shapes) != 1:
        raise ValueError("all threshold arms must share one causal trigger shape")
    exemplar = configs[0]
    return {
        "schema": CAUSAL_SEAM_CONTRACT_SCHEMA,
        "onset": causal_onset_contract_record(onset),
        "trigger": {
            "reference": exemplar.trigger_reference,
            "thresholds": sorted({float(item.threshold) for item in configs}),
            "require_outside_target": bool(exemplar.require_outside_target),
            "max_ab_ms": int(exemplar.max_ab_ms),
            "min_remaining_counts": float(exemplar.min_remaining_counts),
            "progress_regression_reset": bool(
                exemplar.progress_regression_reset
            ),
            "progress_regression_threshold": float(
                exemplar.progress_regression_threshold
            ),
            "max_center_progress": float(exemplar.max_center_progress),
            "max_realized_progress": exemplar.max_realized_progress,
        },
        "planner_context": {
            "progress_reference": exemplar.planner_progress_reference,
        },
        "training_eligibility": asdict(eligibility),
    }


@dataclass(frozen=True)
class DenseSlice:
    """Half-open row range selected from a consecutive dense 1 ms grid."""

    start: int
    stop: int
    policy: str

    @property
    def length(self) -> int:
        return max(0, self.stop - self.start)


@dataclass(frozen=True)
class EdgeProgressSeam:
    """A causal A->B/B->C split.

    ``split_index`` is the number of deltas in A->B.  Consequently,
    ``raw[:split_index]`` is the prefix and ``raw[split_index:]`` is the
    future.  The B position is the post-delta position of the last prefix tick.
    """

    split_index: int
    threshold: float
    realized_progress: float
    center_progress: float
    target_rel_at_b_x: float
    target_rel_at_b_y: float
    initial_center_distance: float
    initial_edge_distance: float


@dataclass(frozen=True)
class EdgeProgressDecision:
    """Result of replaying the deployable B-trigger state machine."""

    seam: EdgeProgressSeam | None
    reason: str
    eligibility_reasons: tuple[str, ...] = ()

    @property
    def training_eligible(self) -> bool:
        return self.seam is not None and not self.eligibility_reasons


@dataclass(frozen=True)
class MovementOnset:
    """Detected beginning A of the committed target-directed movement."""

    index: int
    threshold: float
    speed_median: float
    speed_mad: float
    reason: str
    detection_index: int | None = None


@dataclass(frozen=True)
class ShotQuality:
    """Continuous event-level diagnostics.

    These values are measurements, not a hidden learned score.  A
    :class:`ShotFilterPolicy` turns them into named rejection reasons.
    """

    duration_ms: int
    initial_distance: float
    target_radius: float
    endpoint_distance: float
    endpoint_radius_fraction: float
    minimum_distance: float
    minimum_radius_fraction: float
    path_length: float
    path_efficiency: float
    radial_return_distance: float
    radial_return_fraction: float
    direction_reversals: int
    total_abs_turn_degrees: float
    first_inside_tick: int
    final_inside_run_ms: int
    trailing_quiet_ms: int
    tail_active_fraction_16ms: float
    tail_speed_mean_16ms: float
    tail_speed_max_16ms: float
    max_speed: float
    finite: bool


@dataclass(frozen=True)
class ShotFilterPolicy:
    """Auditable clean-shot cohort definition.

    The defaults are the final release cohort. They are explicit policy
    choices rather than learned scores or universal truths.
    """

    min_duration_ms: int = 24
    max_duration_ms: int = 2_500
    max_endpoint_radius_fraction: float = 0.75
    min_final_inside_run_ms: int = 8
    min_trailing_quiet_ms: int = 4
    max_tail_active_fraction_16ms: float = 0.50
    max_path_efficiency: float = 2.0
    max_radial_return_fraction: float = 0.35
    max_direction_reversals: int = 8
    max_total_abs_turn_degrees: float = 900.0


def _as_dxdy(dxdy: np.ndarray) -> np.ndarray:
    arr = np.asarray(dxdy)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("dxdy must have shape (T, 2)")
    return arr.astype(np.float64, copy=False)


def dense_slice_indices(
    begin_unix_ns: np.ndarray,
    end_unix_ns: np.ndarray,
    event_start_unix_ns: int,
    event_end_unix_ns: int,
    *,
    policy: str = "fully_contained",
) -> DenseSlice:
    """Select bins for one event from a sorted, consecutive dense grid.

    Policies:

    ``fully_contained``
        Include only bins whose complete half-open interval lies inside the
        event.  This can discard at most one partial bin at each boundary and
        cannot import movement from the neighboring event.

    ``midpoint``
        Include bins whose midpoint lies in the event. This alternate import
        policy is explicit so callers cannot change boundary handling silently.
    """

    begin = np.asarray(begin_unix_ns, dtype=np.int64)
    end = np.asarray(end_unix_ns, dtype=np.int64)
    if begin.ndim != 1 or end.ndim != 1 or len(begin) != len(end):
        raise ValueError("begin_unix_ns and end_unix_ns must be equal-length vectors")
    if event_end_unix_ns <= event_start_unix_ns:
        raise ValueError("event end must be later than event start")
    if len(begin) and (np.any(end <= begin) or np.any(begin[1:] < begin[:-1])):
        raise ValueError("dense bin boundaries must be ordered and non-empty")

    if policy == "fully_contained":
        start = int(np.searchsorted(begin, int(event_start_unix_ns), side="left"))
        stop = int(np.searchsorted(end, int(event_end_unix_ns), side="right"))
    elif policy == "midpoint":
        midpoint = begin + (end - begin) // 2
        start = int(np.searchsorted(midpoint, int(event_start_unix_ns), side="left"))
        stop = int(np.searchsorted(midpoint, int(event_end_unix_ns), side="left"))
    else:
        raise ValueError(f"unknown dense slicing policy: {policy}")

    start = min(max(start, 0), len(begin))
    stop = min(max(stop, start), len(begin))
    return DenseSlice(start=start, stop=stop, policy=policy)


def post_delta_path(dxdy: np.ndarray) -> np.ndarray:
    """Return the relative post-delta cursor position for every tick."""

    arr = _as_dxdy(dxdy)
    return np.cumsum(arr, axis=0)


def causal_ema_dxdy(dxdy: np.ndarray, *, alpha: float = 0.25) -> np.ndarray:
    """A cheap strictly causal smoothed-prefix view.

    This optional checkpoint-declared view is not used to select B and is not
    interchangeable with the centered offline smoother used for Planner
    targets.
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
    # Python. This keeps the causal-smoothed Planner input viable in the
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


def detect_movement_onset(
    dxdy: np.ndarray,
    target_rel_at_presentation: np.ndarray | tuple[float, float],
    *,
    noise_window_ms: int = 24,
    consecutive_ticks: int = 12,
    backtrack_ms: int = 4,
    speed_floor: float = 0.35,
    threshold_mad_multiplier: float = 6.0,
    alignment_min: float = 0.15,
    quiet_baseline_fallback: bool = True,
    config: OnsetConfig | None = None,
) -> MovementOnset | None:
    """Detect A with the same causal rule used by the live release.

    The threshold is estimated from the first ``noise_window_ms`` ticks.  A is
    the first sustained run above that threshold whose motion is aligned with
    the current target-relative vector, backtracked a few ticks. Unrelated
    bursts never arm the detector.
    """

    if config is not None:
        noise_window_ms = config.noise_window_ms
        backtrack_ms = config.backtrack_ms
        speed_floor = config.speed_floor
        threshold_mad_multiplier = config.threshold_mad_multiplier
        alignment_min = config.alignment_min
        quiet_baseline_fallback = config.quiet_baseline_fallback
        consecutive_ticks = config.consecutive_ticks

    arr = _as_dxdy(dxdy)
    if len(arr) == 0:
        return None
    target = np.asarray(target_rel_at_presentation, dtype=np.float64).reshape(2)
    if not np.isfinite(target).all():
        return None
    window = max(1, int(noise_window_ms))
    if len(arr) < window:
        # The live detector cannot freeze its baseline before this many
        # observed ticks; offline preprocessing must not gain an early look.
        return None
    speed = np.linalg.norm(arr, axis=1)
    baseline = speed[:window]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    threshold = (
        float(speed_floor)
        if quiet_baseline_fallback and median > float(speed_floor)
        else max(
            float(speed_floor),
            median + float(threshold_mad_multiplier) * mad,
        )
    )
    pre_path = np.zeros_like(arr)
    if len(arr) > 1:
        pre_path[1:] = np.cumsum(arr[:-1], axis=0)
    target_now = target[None, :] - pre_path
    denom = speed * np.linalg.norm(target_now, axis=1) + 1e-6
    alignment = np.sum(arr * target_now, axis=1) / denom
    moving = speed > threshold
    preferred = moving & (alignment >= float(alignment_min))

    def first_run(mask: np.ndarray) -> tuple[int, int] | None:
        run = max(1, int(consecutive_ticks))
        if len(mask) < run:
            return None
        # Convolution keeps the definition exact while avoiding a Python scan
        # across tens of thousands of events.
        hits = np.flatnonzero(np.convolve(mask.astype(np.int16), np.ones(run, dtype=np.int16), mode="valid") == run)
        if not len(hits):
            return None
        start = int(hits[0])
        return start, start + run - 1

    decision = first_run(preferred)
    if decision is None:
        return None
    onset, detection = decision
    return MovementOnset(
        index=max(0, onset - int(backtrack_ms)),
        threshold=threshold,
        speed_median=median,
        speed_mad=mad,
        reason="sustained_aligned_movement",
        detection_index=max(int(noise_window_ms) - 1, detection),
    )


def edge_progress(
    post_positions: np.ndarray,
    target_rel_at_start: np.ndarray | tuple[float, float],
    target_radius: float,
) -> np.ndarray:
    """Measure deployed A->target progress using the target *edge* as 100%.

    Let ``D0`` be the center distance at A and ``r`` the target radius.  The
    distance along the onset-to-target axis to the near edge is ``D0-r``.  At
    each post-delta cursor position ``p``:

    ``progress = dot(p, unit(target)) / (D0-r)``.

    This exactly matches the live ``BTrigger`` contract.  Progress is causal,
    may decrease when the user moves away, and is intentionally not clipped.
    Lateral error is checked independently by the pre-entry gate.
    """

    path = np.asarray(post_positions, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("post_positions must have shape (T, 2)")
    target = np.asarray(target_rel_at_start, dtype=np.float64).reshape(2)
    radius = float(target_radius)
    if not np.isfinite(target).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError("target and positive target_radius must be finite")
    initial_center = float(np.linalg.norm(target))
    initial_edge = initial_center - radius
    if initial_edge <= EPS:
        raise ValueError("event starts inside or on the target edge")

    target_direction = target / initial_center
    return (path @ target_direction) / initial_edge


def select_edge_progress_seam(
    dxdy: np.ndarray,
    target_rel_at_start: np.ndarray | tuple[float, float],
    target_radius: float,
    *,
    threshold: float | None = None,
    b_config: CausalBConfig | None = None,
    min_prefix_ms: int = 24,
    min_future_ms: int = 12,
    require_outside_target: bool = True,
    max_ab_ms: int = 1_500,
    min_remaining_counts: float = 8.0,
    progress_regression_reset: bool = True,
    progress_regression_threshold: float = 0.18,
    max_center_progress: float = 0.92,
    max_realized_progress: float | None = None,
) -> EdgeProgressSeam | None:
    """Return the first deployable edge-progress crossing, or ``None``.

    Only the observed prefix and known target geometry are used.  No future
    peak, deceleration, closest-approach, entry, or terminal state selects B.
    """

    decision = edge_progress_decision(
        dxdy,
        target_rel_at_start,
        target_radius,
        threshold=threshold,
        b_config=b_config,
        min_prefix_ms=min_prefix_ms,
        min_future_ms=min_future_ms,
        require_outside_target=require_outside_target,
        max_ab_ms=max_ab_ms,
        min_remaining_counts=min_remaining_counts,
        progress_regression_reset=progress_regression_reset,
        progress_regression_threshold=progress_regression_threshold,
        max_center_progress=max_center_progress,
        max_realized_progress=max_realized_progress,
    )
    return decision.seam if decision.training_eligible else None


def edge_progress_decision(
    dxdy: np.ndarray,
    target_rel_at_start: np.ndarray | tuple[float, float],
    target_radius: float,
    *,
    threshold: float | None = None,
    b_config: CausalBConfig | None = None,
    min_prefix_ms: int = 24,
    min_future_ms: int = 12,
    require_outside_target: bool = True,
    max_ab_ms: int = 1_500,
    min_remaining_counts: float = 8.0,
    progress_regression_reset: bool = True,
    progress_regression_threshold: float = 0.18,
    max_center_progress: float = 0.92,
    max_realized_progress: float | None = None,
) -> EdgeProgressDecision:
    """Replay the causal trigger, then label training-only eligibility.

    The first causal threshold crossing is immutable.  Prefix and future length
    requirements are evaluated only after that seam has been frozen; they can
    exclude an example but can never move B to a later tick.
    """

    arr = _as_dxdy(dxdy)
    if b_config is not None:
        if threshold is not None and not math.isclose(
            float(threshold), float(b_config.threshold), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("threshold and b_config.threshold disagree")
        threshold = float(b_config.threshold)
        require_outside_target = b_config.require_outside_target
        max_ab_ms = b_config.max_ab_ms
        min_remaining_counts = b_config.min_remaining_counts
        progress_regression_reset = b_config.progress_regression_reset
        progress_regression_threshold = b_config.progress_regression_threshold
        max_center_progress = b_config.max_center_progress
        max_realized_progress = b_config.max_realized_progress
    if threshold is None:
        raise ValueError("an explicit B threshold or b_config is required")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    eligibility = SeamEligibility(
        min_prefix_ms=int(min_prefix_ms),
        min_future_ms=int(min_future_ms),
    )

    target = np.asarray(target_rel_at_start, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(target)) <= float(target_radius) + EPS:
        return EdgeProgressDecision(None, "starts_inside_target")
    path = post_delta_path(arr)
    progress = edge_progress(path, target, float(target_radius))
    initial_center = float(np.linalg.norm(target))
    target_direction = target / initial_center
    center_progress = (path @ target_direction) / initial_center
    max_center_seen = 0.0
    for tick_index in range(len(arr)):
        split_index = int(tick_index) + 1
        if split_index > int(max_ab_ms):
            return EdgeProgressDecision(None, "max_ab_ms")
        current_center_progress = float(center_progress[tick_index])
        if current_center_progress > max_center_seen:
            max_center_seen = current_center_progress
        elif (
            progress_regression_reset
            and max_center_seen - current_center_progress
            > float(progress_regression_threshold)
        ):
            return EdgeProgressDecision(None, "progress_regression")
        if float(progress[tick_index]) < float(threshold):
            continue
        if current_center_progress > float(max_center_progress):
            return EdgeProgressDecision(None, "max_center_progress")
        if (
            max_realized_progress is not None
            and float(progress[tick_index]) > float(max_realized_progress)
        ):
            return EdgeProgressDecision(None, "late_progress_jump")
        target_b = target - path[tick_index]
        remaining = float(np.linalg.norm(target_b))
        if remaining < float(min_remaining_counts):
            return EdgeProgressDecision(None, "min_b_remaining")
        if require_outside_target and float(np.linalg.norm(target_b)) <= float(target_radius):
            # The live trigger rejects a seam that has already entered the
            # target.  Do not silently choose a later re-crossing.
            return EdgeProgressDecision(None, "inside_target")
        seam = EdgeProgressSeam(
            split_index=split_index,
            threshold=float(threshold),
            realized_progress=float(progress[tick_index]),
            center_progress=current_center_progress,
            target_rel_at_b_x=float(target_b[0]),
            target_rel_at_b_y=float(target_b[1]),
            initial_center_distance=initial_center,
            initial_edge_distance=initial_center - float(target_radius),
        )
        eligibility_reasons: list[str] = []
        if split_index < eligibility.min_prefix_ms:
            eligibility_reasons.append("insufficient_prefix")
        if len(arr) - split_index < eligibility.min_future_ms:
            eligibility_reasons.append("insufficient_future")
        return EdgeProgressDecision(
            seam,
            "fired",
            tuple(eligibility_reasons),
        )
    return EdgeProgressDecision(None, "threshold_not_reached")


def _trailing_true(values: np.ndarray) -> int:
    count = 0
    for value in np.asarray(values, dtype=bool)[::-1]:
        if not bool(value):
            break
        count += 1
    return count


def _coarse_motion(dxdy: np.ndarray, bin_ms: int) -> np.ndarray:
    """Aggregate raw count texture before measuring behavioral shape."""

    arr = np.asarray(dxdy, dtype=np.float64)
    width = max(int(bin_ms), 1)
    if width == 1 or len(arr) == 0:
        return arr
    n_bins = (len(arr) + width - 1) // width
    padded = np.zeros((n_bins * width, 2), dtype=np.float64)
    padded[: len(arr)] = arr
    return padded.reshape(n_bins, width, 2).sum(axis=1)


def _direction_reversal_count(dxdy: np.ndarray, target_direction: np.ndarray) -> int:
    along = np.asarray(dxdy, dtype=np.float64) @ target_direction
    sign = np.sign(along[np.abs(along) > EPS])
    return int(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0


def _total_abs_turn_degrees(dxdy: np.ndarray) -> float:
    arr = np.asarray(dxdy, dtype=np.float64)
    speed = np.linalg.norm(arr, axis=1)
    active = arr[speed > EPS]
    if len(active) < 2:
        return 0.0
    unit = active / np.linalg.norm(active, axis=1, keepdims=True)
    dot = np.sum(unit[1:] * unit[:-1], axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)).sum())


def shot_quality(
    dxdy: np.ndarray,
    target_rel_at_start: np.ndarray | tuple[float, float],
    target_radius: float,
    *,
    quiet_count_threshold: float = 1.0,
    behavioral_bin_ms: int = 8,
) -> ShotQuality:
    """Compute clean-shot diagnostics for an event ending at its chosen C."""

    arr = _as_dxdy(dxdy)
    target = np.asarray(target_rel_at_start, dtype=np.float64).reshape(2)
    radius = float(target_radius)
    finite = bool(np.isfinite(arr).all() and np.isfinite(target).all() and math.isfinite(radius))
    if not finite or radius <= 0:
        return ShotQuality(
            duration_ms=len(arr),
            initial_distance=float("nan"),
            target_radius=radius,
            endpoint_distance=float("nan"),
            endpoint_radius_fraction=float("inf"),
            minimum_distance=float("nan"),
            minimum_radius_fraction=float("inf"),
            path_length=float("nan"),
            path_efficiency=float("inf"),
            radial_return_distance=float("inf"),
            radial_return_fraction=float("inf"),
            direction_reversals=0,
            total_abs_turn_degrees=float("inf"),
            first_inside_tick=-1,
            final_inside_run_ms=0,
            trailing_quiet_ms=0,
            tail_active_fraction_16ms=1.0,
            tail_speed_mean_16ms=float("inf"),
            tail_speed_max_16ms=float("inf"),
            max_speed=float("inf"),
            finite=False,
        )

    initial_distance = float(np.linalg.norm(target))
    if len(arr):
        path = post_delta_path(arr)
        center_distance = np.linalg.norm(target[None, :] - path, axis=1)
        endpoint_distance = float(center_distance[-1])
        minimum_distance = float(np.min(center_distance))
        speed = np.linalg.norm(arr, axis=1)
        path_length = float(speed.sum())
        radial_delta = np.diff(np.concatenate([[initial_distance], center_distance]))
        radial_return = float(np.maximum(radial_delta, 0.0).sum())
        inside = center_distance <= radius
        first_inside = int(np.flatnonzero(inside)[0]) if bool(np.any(inside)) else -1
        final_inside_run = _trailing_true(inside)
        quiet = np.max(np.abs(arr), axis=1) <= float(quiet_count_threshold)
        trailing_quiet = _trailing_true(quiet)
        tail = speed[-min(16, len(speed)) :]
        tail_active = float(np.mean(tail > float(quiet_count_threshold))) if len(tail) else 0.0
        target_direction = target / max(initial_distance, EPS)
        behavioral_motion = _coarse_motion(arr, behavioral_bin_ms)
        behavioral_speed = np.linalg.norm(behavioral_motion, axis=1)
        if len(behavioral_speed):
            # Ignore tiny aggregate packets when measuring path topology.  They
            # are renderer texture, not evidence of a behavioral reversal.
            motion_floor = max(1.0, 0.02 * float(np.max(behavioral_speed)))
            behavioral_motion = behavioral_motion[behavioral_speed >= motion_floor]
        reversals = _direction_reversal_count(behavioral_motion, target_direction)
        total_turn = _total_abs_turn_degrees(behavioral_motion)
        max_speed = float(np.max(speed))
        tail_mean = float(np.mean(tail)) if len(tail) else 0.0
        tail_max = float(np.max(tail)) if len(tail) else 0.0
    else:
        endpoint_distance = initial_distance
        minimum_distance = initial_distance
        path_length = 0.0
        radial_return = 0.0
        first_inside = -1
        final_inside_run = 0
        trailing_quiet = 0
        tail_active = 0.0
        reversals = 0
        total_turn = 0.0
        max_speed = 0.0
        tail_mean = 0.0
        tail_max = 0.0

    return ShotQuality(
        duration_ms=int(len(arr)),
        initial_distance=initial_distance,
        target_radius=radius,
        endpoint_distance=endpoint_distance,
        endpoint_radius_fraction=endpoint_distance / max(radius, EPS),
        minimum_distance=minimum_distance,
        minimum_radius_fraction=minimum_distance / max(radius, EPS),
        path_length=path_length,
        path_efficiency=path_length / max(initial_distance, EPS),
        radial_return_distance=radial_return,
        radial_return_fraction=radial_return / max(initial_distance, EPS),
        direction_reversals=reversals,
        total_abs_turn_degrees=total_turn,
        first_inside_tick=first_inside,
        final_inside_run_ms=final_inside_run,
        trailing_quiet_ms=trailing_quiet,
        tail_active_fraction_16ms=tail_active,
        tail_speed_mean_16ms=tail_mean,
        tail_speed_max_16ms=tail_max,
        max_speed=max_speed,
        finite=True,
    )


def shot_filter_reasons(
    quality: ShotQuality,
    *,
    outcome: str,
    technical_outcome: str = "none",
    policy: ShotFilterPolicy | None = None,
) -> tuple[str, ...]:
    """Return stable, machine-readable reasons an event misses a cohort."""

    p = policy or ShotFilterPolicy()
    reasons: list[str] = []
    if outcome not in SUCCESS_OUTCOMES:
        reasons.append("not_success")
    if technical_outcome not in {"", "none"}:
        reasons.append("technical_outcome")
    if not quality.finite:
        reasons.append("nonfinite")
        return tuple(reasons)
    if quality.duration_ms < p.min_duration_ms:
        reasons.append("too_short")
    if quality.duration_ms > p.max_duration_ms:
        reasons.append("too_long")
    if quality.endpoint_radius_fraction > p.max_endpoint_radius_fraction:
        reasons.append("endpoint_not_inner")
    if quality.final_inside_run_ms < p.min_final_inside_run_ms:
        reasons.append("insufficient_inside_settle")
    if quality.trailing_quiet_ms < p.min_trailing_quiet_ms:
        reasons.append("clicked_while_moving")
    if quality.tail_active_fraction_16ms > p.max_tail_active_fraction_16ms:
        reasons.append("active_tail")
    if quality.path_efficiency > p.max_path_efficiency:
        reasons.append("path_inefficient")
    if quality.radial_return_fraction > p.max_radial_return_fraction:
        reasons.append("large_return")
    if quality.direction_reversals > p.max_direction_reversals:
        reasons.append("many_reversals")
    if quality.total_abs_turn_degrees > p.max_total_abs_turn_degrees:
        reasons.append("excessive_turning")
    return tuple(reasons)


def post_c_tail_metrics(
    dxdy: np.ndarray,
    target_rel_at_c: np.ndarray | tuple[float, float],
    movement_direction: np.ndarray | tuple[float, float],
    target_radius: float,
    *,
    windows_ms: tuple[int, ...] = (16, 32, 64),
) -> dict[str, float | int]:
    """Describe motion after the chosen C without deciding cohort membership."""

    arr = _as_dxdy(dxdy)
    remaining = np.asarray(target_rel_at_c, dtype=np.float64).reshape(2)
    direction = np.asarray(movement_direction, dtype=np.float64).reshape(2)
    radius = float(target_radius)
    norm = float(np.linalg.norm(direction))
    finite = bool(
        np.isfinite(arr).all()
        and np.isfinite(remaining).all()
        and np.isfinite(direction).all()
        and math.isfinite(radius)
        and radius > 0.0
        and norm > EPS
    )
    result: dict[str, float | int] = {
        "post_c_available_ms": int(len(arr)),
        "post_c_finite": int(finite),
    }
    if not finite:
        for window in windows_ms:
            prefix = f"post_c_{int(window)}ms"
            result.update(
                {
                    f"{prefix}_observed_ms": 0,
                    f"{prefix}_path_length": float("nan"),
                    f"{prefix}_displacement": float("nan"),
                    f"{prefix}_max_excursion": float("nan"),
                    f"{prefix}_backtrack_toward_A": float("nan"),
                    f"{prefix}_active_fraction": float("nan"),
                    f"{prefix}_endpoint_radius_fraction": float("nan"),
                }
            )
        return result

    unit = direction / norm
    for window in windows_ms:
        requested = max(int(window), 0)
        observed = arr[:requested]
        prefix = f"post_c_{requested}ms"
        if len(observed):
            path = np.cumsum(observed, axis=0)
            speed = np.linalg.norm(observed, axis=1)
            displacement = path[-1]
            along = path @ unit
            max_excursion = float(np.max(np.linalg.norm(path, axis=1)))
            backtrack = float(max(0.0, -float(np.min(along))))
            target_after = remaining - displacement
            endpoint_fraction = float(np.linalg.norm(target_after) / radius)
            active_fraction = float(np.mean(np.any(observed != 0.0, axis=1)))
            path_length = float(speed.sum())
            displacement_norm = float(np.linalg.norm(displacement))
        else:
            max_excursion = 0.0
            backtrack = 0.0
            endpoint_fraction = float(np.linalg.norm(remaining) / radius)
            active_fraction = 0.0
            path_length = 0.0
            displacement_norm = 0.0
        result.update(
            {
                f"{prefix}_observed_ms": int(len(observed)),
                f"{prefix}_path_length": path_length,
                f"{prefix}_displacement": displacement_norm,
                f"{prefix}_max_excursion": max_excursion,
                f"{prefix}_backtrack_toward_A": backtrack,
                f"{prefix}_active_fraction": active_fraction,
                f"{prefix}_endpoint_radius_fraction": endpoint_fraction,
            }
        )
    return result


def per_source_trial_weights(source_trial_ids: Iterable[str]) -> np.ndarray:
    """Give every source trial total weight one across all augmented cuts."""

    ids = np.asarray(list(source_trial_ids), dtype=str)
    if ids.ndim != 1:
        raise ValueError("source_trial_ids must be one-dimensional")
    if len(ids) == 0:
        return np.zeros(0, dtype=np.float32)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float32)


def quality_to_dict(quality: ShotQuality) -> dict[str, object]:
    """JSON-friendly representation used by preprocessing reports."""

    return asdict(quality)


__all__ = [
    "SUCCESS_OUTCOMES",
    "CAUSAL_SEAM_CONTRACT_SCHEMA",
    "B_TRIGGER_REFERENCE",
    "PLANNER_PROGRESS_REFERENCE",
    "CausalOnsetConfig",
    "OnsetConfig",
    "causal_onset_contract_record",
    "onset_config_from_record",
    "CausalBConfig",
    "SeamEligibility",
    "causal_seam_contract_record",
    "DenseSlice",
    "EdgeProgressSeam",
    "EdgeProgressDecision",
    "MovementOnset",
    "ShotQuality",
    "ShotFilterPolicy",
    "dense_slice_indices",
    "post_delta_path",
    "causal_ema_dxdy",
    "detect_movement_onset",
    "edge_progress",
    "edge_progress_decision",
    "select_edge_progress_seam",
    "shot_quality",
    "shot_filter_reasons",
    "post_c_tail_metrics",
    "per_source_trial_weights",
    "quality_to_dict",
]
