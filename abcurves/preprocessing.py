"""Portable, auditable event preparation for the ABCurves Planner.

The native ABCurves-Capture ZIP format is intentionally not decoded here.  The
input is either a validated ``abcurves.research_export.v1`` directory produced
by Capture, or :data:`PORTABLE_EVENT_SCHEMA`.  Both routes yield dense 1 ms
mouse-count events from causal movement onset A through fixed terminal event C.
This module then makes the Planner-specific choices:

* Planner: a dense, adaptive B78--B90/B92 seam family, clean-shot and
  trajectory-shape filtering, physical-event vetoes for signature-like
  motion, source-balanced weights, and an optional tiny-target curriculum.
The output stores each accepted physical event once and represents training
rows as seam indices into those events.  Dense cuts therefore do not duplicate
raw motion in the file and, more importantly, cannot multiply a source's
optimizer mass.

The global Renderer deliberately does not use these A-to-C event crops.  Its
dense whole-session builder lives in :mod:`abcurves.global_data`.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .capture_preprocess import (
    CausalBConfig,
    CausalOnsetConfig,
    SeamEligibility,
    ShotFilterPolicy,
    causal_seam_contract_record,
    edge_progress_decision,
    per_source_trial_weights,
    shot_filter_reasons,
    shot_quality,
)
from .smoothing import smooth_dxdy


PORTABLE_EVENT_SCHEMA = "abcurves.portable_events.v1"
PREPARED_CUT_SCHEMA = "abcurves.prepared_cuts.v1"
PREPARATION_MANIFEST_SCHEMA = "abcurves.dataset_preparation_manifest.v1"


class DatasetPreparationError(ValueError):
    """Raised when an input cannot satisfy the public preparation contract."""


class NativeCaptureExporterUnavailable(DatasetPreparationError):
    """Raised when a raw capture archive is supplied to the portable tool."""


@dataclass(frozen=True)
class PortableEvent:
    """One physical A-to-C event on a dense 1 ms hardware-count grid."""

    source_trial_id: str
    dxdy: np.ndarray
    target_rel_at_a: np.ndarray
    target_radius: float
    outcome: str = "hit_click"
    technical_outcome: str = "none"
    user_id: str = "unknown"
    session_id: str = "unknown"
    split: str = "train"

    def __post_init__(self) -> None:
        raw = np.asarray(self.dxdy)
        target = np.asarray(self.target_rel_at_a)
        if not self.source_trial_id:
            raise DatasetPreparationError("source_trial_id must not be empty")
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise DatasetPreparationError("event dxdy must have shape (T, 2)")
        if target.shape != (2,):
            raise DatasetPreparationError("target_rel_at_a must have shape (2,)")
        if not np.isfinite(raw).all() or not np.isfinite(target).all():
            raise DatasetPreparationError("event geometry must be finite")
        if not math.isfinite(float(self.target_radius)) or self.target_radius <= 0:
            raise DatasetPreparationError("target_radius must be finite and positive")


@dataclass(frozen=True)
class ResearchExportConversion:
    """Portable events plus accounting from validated research exports."""

    events: tuple[PortableEvent, ...]
    export_directories: tuple[str, ...]
    profiled_event_count: int
    excluded_reason_counts: Mapping[str, int]


@dataclass(frozen=True)
class SeamPreparationConfig:
    """Shared causal B-trigger and training-eligibility settings."""

    min_prefix_ms: int = 24
    min_future_ms: int = 12
    max_future_ms: int = 1_000
    max_ab_ms: int = 1_500
    min_remaining_counts: float = 8.0
    min_edge_margin_counts: float = 8.0
    max_center_progress: float = 0.92
    progress_regression_threshold: float = 0.18

    def __post_init__(self) -> None:
        if self.min_prefix_ms < 0 or self.min_future_ms < 0:
            raise DatasetPreparationError("prefix/future minima must be non-negative")
        if self.max_future_ms < self.min_future_ms:
            raise DatasetPreparationError("max_future_ms must cover min_future_ms")
        if self.max_ab_ms < 1:
            raise DatasetPreparationError("max_ab_ms must be positive")
        if self.min_remaining_counts < 0 or self.min_edge_margin_counts < 0:
            raise DatasetPreparationError("remaining-distance gates must be non-negative")


@dataclass(frozen=True)
class PlannerShapeConfig:
    """Interpretable count-scale gates for Planner intent labels."""

    extreme_arc_chord: float = 1.80
    backtrack_arc_chord: float = 1.45
    axial_regression_over_edge: float = 0.10
    lateral_over_radius: float = 1.50
    detour_path_over_edge: float = 1.60
    lateral_prominence_floor: float = 3.0
    lateral_prominence_radius_fraction: float = 0.12
    axial_prominence_floor: float = 4.0
    axial_prominence_edge_fraction: float = 0.08
    reversal_count: int = 2
    preentry_turn_degrees: float = 540.0
    post_b_turn_degrees: float = 600.0
    turn_bin_ms: int = 8
    turn_minimum_displacement: float = 2.0
    stationary_pause_ms: int = 160
    repeated_target_exits: int = 3
    far_reentry_exits: int = 2
    far_reentry_radius_fraction: float = 1.75
    postentry_turn_degrees: float = 600.0


@dataclass(frozen=True)
class TinyTargetConfig:
    """Low-dose, train-only, fixed-center and shrink-only augmentation."""

    enabled: bool = False
    radius_quotas: tuple[tuple[float, int], ...] = (
        (2.0, 64),
        (3.0, 96),
        (4.0, 128),
        (5.0, 160),
        (6.0, 192),
        (7.0, 224),
        (8.0, 256),
        (9.0, 288),
        (10.0, 320),
        (12.0, 400),
        (15.0, 500),
        (20.0, 640),
    )
    parent_radius_ratio_min: float = 1.25
    endpoint_radius_fraction_max: float = 0.50
    final_inside_ms: int = 8
    train_split: str = "train"
    selection_salt: str = "abcurves.final_v2.tiny"

    def __post_init__(self) -> None:
        seen: set[float] = set()
        for radius, quota in self.radius_quotas:
            if radius <= 0 or quota < 0 or radius in seen:
                raise DatasetPreparationError("tiny-target radii/quotas are invalid")
            seen.add(radius)
        if self.parent_radius_ratio_min < 1.0:
            raise DatasetPreparationError("tiny targets must be shrink-only")
        if self.final_inside_ms < 1:
            raise DatasetPreparationError("final_inside_ms must be positive")


@dataclass(frozen=True)
class MaterializationConfig:
    """Dense arrays consumed directly by the public training scripts."""

    prefix_ticks: int = 160
    future_horizon_ms: int = 1_000
    smoothing_spec: str = "triangular_moving_average_path:window=5"
    onset: CausalOnsetConfig = field(default_factory=CausalOnsetConfig)

    def __post_init__(self) -> None:
        if self.prefix_ticks < 1 or self.future_horizon_ms < 1:
            raise DatasetPreparationError("materialized prefix/horizon must be positive")
        # Parse and execute a zero-length-safe sample now, so a misspelled
        # method fails at configuration load rather than hours into a run.
        try:
            smooth_dxdy(np.zeros((1, 2), dtype=np.float32), spec=self.smoothing_spec)
        except ValueError as exc:
            raise DatasetPreparationError(f"invalid smoothing_spec: {exc}") from exc


@dataclass(frozen=True)
class PlannerPreparationConfig:
    """Typed Planner preparation preset used for the final release."""

    b_min: float = 0.78
    b_short_max: float = 0.90
    b_long_max: float = 0.92
    b_count: int = 21
    long_edge_distance: float = 150.0
    dense_train_split: str = "train"
    runtime_b_threshold: float = 0.80
    seam: SeamPreparationConfig = field(default_factory=SeamPreparationConfig)
    materialization: MaterializationConfig = field(default_factory=MaterializationConfig)
    apply_quality_filter: bool = True
    quality: ShotFilterPolicy = field(default_factory=ShotFilterPolicy)
    recover_turn_only_if_path_efficiency_at_most: float = 1.20
    recover_turn_only_if_radial_return_at_most: float = 0.05
    shape: PlannerShapeConfig = field(default_factory=PlannerShapeConfig)
    tiny_target: TinyTargetConfig = field(default_factory=TinyTargetConfig)

    def __post_init__(self) -> None:
        if not 0 < self.b_min <= self.b_short_max <= self.b_long_max < 1:
            raise DatasetPreparationError("Planner B thresholds must increase inside (0, 1)")
        if self.b_count < 1:
            raise DatasetPreparationError("Planner b_count must be positive")
        if not 0 < self.runtime_b_threshold < 1:
            raise DatasetPreparationError("runtime_b_threshold must lie inside (0, 1)")
        if self.materialization.future_horizon_ms < self.seam.max_future_ms:
            raise DatasetPreparationError("materialized horizon must cover max_future_ms")
        if self.tiny_target.train_split != self.dense_train_split:
            raise DatasetPreparationError("dense and tiny-target train split names must agree")


@dataclass(frozen=True)
class PreparedCut:
    """A model row referring to one seam in one physical source event."""

    event_index: int
    source_trial_id: str
    variant_id: str
    requested_index: int
    split_index: int
    threshold: float
    realized_progress: float
    center_progress: float
    target_rel_at_b_x: float
    target_rel_at_b_y: float
    target_radius: float
    synthetic: bool = False
    borderline: bool = False
    example_weight: float = 0.0


@dataclass(frozen=True)
class PreparationResult:
    branch: str
    events: tuple[PortableEvent, ...]
    cuts: tuple[PreparedCut, ...]
    rejected: Mapping[str, tuple[str, ...]]
    config: PlannerPreparationConfig

    def source_weight_error(self) -> float:
        totals: dict[str, float] = {}
        for cut in self.cuts:
            totals[cut.source_trial_id] = totals.get(cut.source_trial_id, 0.0) + float(
                cut.example_weight
            )
        return max((abs(value - 1.0) for value in totals.values()), default=0.0)


def _text_array(values: Sequence[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _read_scalar_text(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise DatasetPreparationError("schema field must contain one string")
    return str(arr.reshape(-1)[0])


def load_portable_events(path: str | Path) -> tuple[PortableEvent, ...]:
    """Load a safe, pickle-free portable event NPZ.

    A directory is accepted when it contains ``events.npz``.  Raw Capture ZIP
    archives are refused because reproducing their native clock/session audit
    requires the corresponding ABCurves-Capture exporter.
    """

    source = Path(path)
    if source.is_dir():
        source = source / "events.npz"
    if source.suffix.lower() == ".zip":
        raise NativeCaptureExporterUnavailable(
            "raw ABCurves-Capture ZIP import is not bundled: export it to "
            f"{PORTABLE_EVENT_SCHEMA!r} with the Capture exporter first"
        )
    if source.suffix.lower() != ".npz":
        raise DatasetPreparationError("input must be a portable .npz file or its directory")
    if not source.is_file():
        raise DatasetPreparationError(f"portable event file does not exist: {source}")

    with np.load(source, allow_pickle=False) as data:
        required = {
            "schema",
            "dxdy",
            "event_offsets",
            "source_trial_id",
            "target_rel_at_a",
            "target_radius",
            "outcome",
            "technical_outcome",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise DatasetPreparationError(f"portable event NPZ is missing: {', '.join(missing)}")
        schema = _read_scalar_text(data["schema"])
        if schema != PORTABLE_EVENT_SCHEMA:
            raise DatasetPreparationError(f"unsupported portable event schema: {schema!r}")
        dxdy = np.asarray(data["dxdy"], dtype=np.float32)
        offsets = np.asarray(data["event_offsets"], dtype=np.int64)
        source_ids = np.asarray(data["source_trial_id"]).astype(str)
        targets = np.asarray(data["target_rel_at_a"], dtype=np.float64)
        radii = np.asarray(data["target_radius"], dtype=np.float64)
        outcomes = np.asarray(data["outcome"]).astype(str)
        technical = np.asarray(data["technical_outcome"]).astype(str)
        count = len(source_ids)

        def optional_text(name: str, default: str) -> np.ndarray:
            if name not in data.files:
                return np.full(count, default, dtype=f"<U{max(1, len(default))}")
            return np.asarray(data[name]).astype(str)

        users = optional_text("user_id", "unknown")
        sessions = optional_text("session_id", "unknown")
        splits = optional_text("split", "train")

    if dxdy.ndim != 2 or dxdy.shape[1] != 2:
        raise DatasetPreparationError("dxdy must have shape (sum(T), 2)")
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(dxdy):
        raise DatasetPreparationError("event_offsets must start at zero and end at len(dxdy)")
    if np.any(np.diff(offsets) < 0):
        raise DatasetPreparationError("event_offsets must be monotonic")
    per_event = (targets, radii, outcomes, technical, users, sessions, splits)
    if targets.shape != (count, 2) or any(len(values) != count for values in per_event[1:]):
        raise DatasetPreparationError("per-event arrays do not agree with source_trial_id")
    if len(set(source_ids.tolist())) != count:
        raise DatasetPreparationError("source_trial_id must uniquely identify physical events")

    events: list[PortableEvent] = []
    for index in range(count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        events.append(
            PortableEvent(
                source_trial_id=str(source_ids[index]),
                dxdy=dxdy[start:stop],
                target_rel_at_a=targets[index],
                target_radius=float(radii[index]),
                outcome=str(outcomes[index]),
                technical_outcome=str(technical[index]),
                user_id=str(users[index]),
                session_id=str(sessions[index]),
                split=str(splits[index]),
            )
        )
    return tuple(events)


def write_portable_events(path: str | Path, events: Iterable[PortableEvent]) -> Path:
    """Write the documented interchange representation used by this module."""

    rows = tuple(events)
    _validate_unique_sources(rows)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, event in enumerate(rows):
        offsets[index + 1] = offsets[index] + len(event.dxdy)
    dxdy = (
        np.concatenate([np.asarray(event.dxdy, dtype=np.float32) for event in rows])
        if rows
        else np.zeros((0, 2), dtype=np.float32)
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(PORTABLE_EVENT_SCHEMA),
            dxdy=dxdy,
            event_offsets=offsets,
            source_trial_id=_text_array([event.source_trial_id for event in rows]),
            target_rel_at_a=np.asarray([event.target_rel_at_a for event in rows], dtype=np.float32).reshape(-1, 2),
            target_radius=np.asarray([event.target_radius for event in rows], dtype=np.float32),
            outcome=_text_array([event.outcome for event in rows]),
            technical_outcome=_text_array([event.technical_outcome for event in rows]),
            user_id=_text_array([event.user_id for event in rows]),
            session_id=_text_array([event.session_id for event in rows]),
            split=_text_array([event.split for event in rows]),
        )
    return destination


_BRANCH_DECIDABLE_PROFILE_REASONS = frozenset(
    {
        "not_success",
        "technical_outcome",
        "too_short",
        "too_long",
        "endpoint_not_inner",
        "path_inefficient",
        "large_return",
        "many_reversals",
        "excessive_turning",
    }
)


def _validated_export_directories(root: Path) -> tuple[Path, ...]:
    if (root / "export_manifest.json").is_file():
        return (root,)
    exports = tuple(sorted(path.parent for path in root.rglob("export_manifest.json")))
    if not exports:
        if any(root.rglob("*.zip")):
            raise NativeCaptureExporterUnavailable(
                "folder contains sealed/raw Capture ZIPs but no validated research exports; "
                "run the ABCurves-Capture validator/exporter first"
            )
        raise DatasetPreparationError(
            "directory contains neither events.npz nor an abcurves.research_export.v1 export"
        )
    return exports


def load_research_export_events(
    root: str | Path,
    *,
    onset_config: CausalOnsetConfig | None = None,
    seam_eligibility: SeamEligibility | None = None,
    filter_policy: ShotFilterPolicy | None = None,
    validation_fraction: float = 0.15,
    split_seed: str = "abcurves.final_v2.user_split",
) -> ResearchExportConversion:
    """Convert one or more validated research-export directories to events.

    The existing :mod:`abcurves.capture_exports` profiler remains the sole
    authority for clocks, dense slicing, USB click resolution and causal A.
    This bridge only extracts its accepted A-to-C interval.  Planner-only path
    quality reasons are preserved for the branch-specific stage; capture,
    clock, onset, and ambiguous-terminal failures are excluded here.

    Users, never individual events, are assigned to validation.  With fewer
    than two user/session identities the safe result is train-only; the tool
    reports that no validation file could be formed.
    """

    source_root = Path(root)
    if not source_root.is_dir():
        raise DatasetPreparationError(f"research export root is not a directory: {source_root}")
    if not 0.0 <= float(validation_fraction) < 1.0:
        raise DatasetPreparationError("validation_fraction must lie in [0, 1)")
    try:
        from .capture_exports import (
            load_export_manifest,
            load_export_tables,
            profile_export_session,
        )
    except ModuleNotFoundError as exc:
        raise DatasetPreparationError(
            "research-export folders require the data extra: pip install -e '.[data]'"
        ) from exc

    export_dirs = _validated_export_directories(source_root)
    events: list[PortableEvent] = []
    excluded: Counter[str] = Counter()
    profiled_count = 0
    onset = onset_config or CausalOnsetConfig()
    eligibility = seam_eligibility or SeamEligibility()
    policy = filter_policy or ShotFilterPolicy()
    for export_dir in export_dirs:
        manifest = load_export_manifest(export_dir)
        artifacts = {
            str(item.get("relative_path")): str(item.get("sha256", ""))
            for item in manifest.get("artifacts", [])
        }
        for relative_path in ("mouse_1ms.csv", "trainer_events.csv"):
            expected = artifacts.get(relative_path)
            artifact = export_dir / relative_path
            if expected is None or len(expected) != 64 or not artifact.is_file():
                raise DatasetPreparationError(
                    f"export manifest does not bind {relative_path}: {export_dir}"
                )
            if _file_sha256(artifact) != expected:
                raise DatasetPreparationError(
                    f"research export hash mismatch for {artifact}"
                )
        frame, _ = profile_export_session(
            export_dir,
            filter_policy=policy,
            progress_thresholds=(0.80,),
            onset_config=onset,
            seam_eligibility=eligibility,
        )
        _, _, mouse = load_export_tables(export_dir)
        raw_all = mouse[["canonical_dx", "canonical_dy"]].to_numpy(np.float32)
        profiled_count += len(frame)
        for record in frame.to_dict(orient="records"):
            raw_reasons = str(record.get("rejection_reasons") or "")
            reasons = {reason for reason in raw_reasons.split("|") if reason and reason != "nan"}
            blocking = sorted(reasons - _BRANCH_DECIDABLE_PROFILE_REASONS)
            start = record.get("dense_model_start_index")
            stop = record.get("dense_stop_index")
            try:
                finite_interval = math.isfinite(float(start)) and math.isfinite(float(stop))
            except (TypeError, ValueError):
                finite_interval = False
            if not finite_interval:
                blocking.append("missing_profiled_a_to_c_interval")
            if blocking:
                excluded.update(sorted(set(blocking)))
                continue
            begin_index, stop_index = int(start), int(stop)
            if not 0 <= begin_index < stop_index <= len(raw_all):
                excluded["invalid_profiled_a_to_c_interval"] += 1
                continue
            try:
                event = PortableEvent(
                    source_trial_id=str(record["source_trial_id"]),
                    dxdy=np.asarray(raw_all[begin_index:stop_index], dtype=np.float32),
                    target_rel_at_a=np.asarray(
                        [record["target_rel_at_A_x"], record["target_rel_at_A_y"]],
                        dtype=np.float64,
                    ),
                    target_radius=float(record["target_radius_counts"]),
                    outcome=str(record["natural_outcome"]),
                    technical_outcome=str(record["technical_outcome"]),
                    user_id=str(record["user_id"]),
                    session_id=str(record["session_id"]),
                    split="train",
                )
            except (DatasetPreparationError, KeyError, TypeError, ValueError) as exc:
                excluded[f"portable_event_error:{type(exc).__name__}"] += 1
                continue
            events.append(event)

    _validate_unique_sources(events)
    identities = sorted(
        {
            event.user_id if event.user_id != "unknown" else event.session_id
            for event in events
        },
        key=lambda identity: _stable_key(split_seed, identity),
    )
    validation_identities: set[str] = set()
    if validation_fraction > 0.0 and len(identities) >= 2:
        validation_count = max(
            1,
            min(len(identities) - 1, int(round(len(identities) * validation_fraction))),
        )
        validation_identities = set(identities[:validation_count])
    assigned = tuple(
        replace(
            event,
            split=(
                "val"
                if (event.user_id if event.user_id != "unknown" else event.session_id)
                in validation_identities
                else "train"
            ),
        )
        for event in events
    )
    return ResearchExportConversion(
        events=assigned,
        export_directories=tuple(str(path) for path in export_dirs),
        profiled_event_count=profiled_count,
        excluded_reason_counts=dict(sorted(excluded.items())),
    )


def _validate_unique_sources(events: Sequence[PortableEvent]) -> None:
    source_ids = [event.source_trial_id for event in events]
    duplicates = [key for key, count in Counter(source_ids).items() if count > 1]
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:3])
        raise DatasetPreparationError(f"duplicate physical source_trial_id values: {preview}")


def adaptive_planner_thresholds(
    target_rel_at_a: np.ndarray,
    target_radius: float,
    config: PlannerPreparationConfig,
) -> np.ndarray:
    """Return B78--B90, extending to B92 only for long edge distances."""

    edge_distance = float(np.linalg.norm(target_rel_at_a) - target_radius)
    cap = config.b_long_max if edge_distance >= config.long_edge_distance else config.b_short_max
    return np.linspace(config.b_min, cap, config.b_count, dtype=np.float64)


def _coarse_turn_degrees(
    dxdy: np.ndarray,
    *,
    bin_ms: int,
    minimum_bin_displacement: float,
) -> float:
    arr = np.asarray(dxdy, dtype=np.float64)
    if not len(arr):
        return 0.0
    width = max(int(bin_ms), 1)
    n_bins = (len(arr) + width - 1) // width
    padded = np.zeros((n_bins * width, 2), dtype=np.float64)
    padded[: len(arr)] = arr
    coarse = padded.reshape(n_bins, width, 2).sum(axis=1)
    speed = np.linalg.norm(coarse, axis=1)
    floor = max(float(minimum_bin_displacement), 0.02 * float(np.max(speed)))
    coarse = coarse[speed >= floor]
    if len(coarse) < 2:
        return 0.0
    unit = coarse / np.linalg.norm(coarse, axis=1, keepdims=True)
    dot = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)).sum())


def _prominent_reversal_count(values: np.ndarray, prominence: float) -> int:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(series) < 2:
        return 0
    anchor = extreme = float(series[0])
    trend = reversals = 0
    threshold = max(float(prominence), 1e-9)
    for raw_value in series[1:]:
        value = float(raw_value)
        if trend == 0:
            if value - anchor >= threshold:
                trend, extreme = 1, value
            elif anchor - value >= threshold:
                trend, extreme = -1, value
        elif trend > 0:
            if value > extreme:
                extreme = value
            elif extreme - value >= threshold:
                reversals += 1
                trend, extreme = -1, value
        else:
            if value < extreme:
                extreme = value
            elif value - extreme >= threshold:
                reversals += 1
                trend, extreme = 1, value
    return reversals


def _max_resumed_stationary_run_ms(dxdy: np.ndarray) -> int:
    moving = np.flatnonzero(np.any(np.asarray(dxdy) != 0, axis=1))
    if not len(moving):
        return 0
    leading = int(moving[0])
    internal = int(np.max(np.diff(moving) - 1)) if len(moving) >= 2 else 0
    return max(leading, internal)


def planner_shape_metrics(
    event_dxdy: np.ndarray,
    target_rel_at_a: np.ndarray,
    target_radius: float,
    split_index: int,
    config: PlannerShapeConfig | None = None,
) -> dict[str, float | int]:
    """Measure cut geometry with count-scale, auditable quantities."""

    shape = config or PlannerShapeConfig()
    raw = np.asarray(event_dxdy, dtype=np.float64)
    target = np.asarray(target_rel_at_a, dtype=np.float64).reshape(2)
    b_position = np.sum(raw[:split_index], axis=0)
    target_b = target - b_position
    future = raw[split_index:]
    future_path = np.cumsum(future, axis=0)
    distance = np.linalg.norm(target_b[None, :] - future_path, axis=1)
    inside = distance <= float(target_radius)
    hits = np.flatnonzero(inside)
    first_entry = int(hits[0]) if len(hits) else max(len(future) - 1, 0)
    pre = future[: first_entry + 1]
    pre_path = np.cumsum(pre, axis=0) if len(pre) else np.zeros((0, 2))
    pre_speed = np.linalg.norm(pre, axis=1) if len(pre) else np.zeros(0)
    chord = float(np.linalg.norm(pre_path[-1])) if len(pre_path) else 0.0
    path_length = float(pre_speed.sum())
    edge = max(float(np.linalg.norm(target_b) - target_radius), 1e-9)
    direction = target_b / max(float(np.linalg.norm(target_b)), 1e-9)
    normal = np.asarray([-direction[1], direction[0]])
    if len(pre_path):
        axial = pre_path @ direction
        running_max = np.maximum.accumulate(np.concatenate([[0.0], axial]))[:-1]
        axial_regression = float(np.max(np.maximum(running_max - axial, 0.0)))
        lateral = float(np.max(np.abs(pre_path @ normal)))
    else:
        axial_regression = lateral = 0.0
    entries = int(bool(len(inside)) and bool(inside[0]))
    exits = int(np.sum(inside[:-1] & (~inside[1:]))) if len(inside) > 1 else 0
    if len(inside) > 1:
        entries += int(np.sum((~inside[:-1]) & inside[1:]))
    post_distance = distance[first_entry:] if len(distance) else np.zeros(0)
    post_future = future[first_entry + 1 :] if first_entry + 1 < len(future) else np.zeros((0, 2))
    all_speed = np.linalg.norm(future, axis=1) if len(future) else np.zeros(0)
    future_chord = float(np.linalg.norm(future_path[-1])) if len(future_path) else 0.0
    pre_lateral = np.concatenate([[0.0], pre_path @ normal if len(pre_path) else np.zeros(0)])
    pre_axial = np.concatenate([[0.0], pre_path @ direction if len(pre_path) else np.zeros(0)])
    post_lateral = np.concatenate([[0.0], future_path @ normal if len(future_path) else np.zeros(0)])
    post_axial = np.concatenate([[0.0], future_path @ direction if len(future_path) else np.zeros(0)])
    lateral_prominence = max(
        shape.lateral_prominence_floor,
        shape.lateral_prominence_radius_fraction * float(target_radius),
    )
    axial_prominence = max(
        shape.axial_prominence_floor,
        shape.axial_prominence_edge_fraction * edge,
    )
    return {
        "first_entry_ms": first_entry + 1,
        "preentry_arc_chord": path_length / max(chord, 1e-9),
        "preentry_path_over_edge": path_length / edge,
        "preentry_axial_regression_over_edge": axial_regression / edge,
        "preentry_lateral_over_radius": lateral / max(float(target_radius), 1e-9),
        "preentry_substantial_turn_degrees": _coarse_turn_degrees(
            pre,
            bin_ms=shape.turn_bin_ms,
            minimum_bin_displacement=shape.turn_minimum_displacement,
        ),
        "preentry_lateral_reversals": _prominent_reversal_count(pre_lateral, lateral_prominence),
        "preentry_axial_reversals": _prominent_reversal_count(pre_axial, axial_prominence),
        "preentry_max_stationary_run_ms": _max_resumed_stationary_run_ms(pre),
        "postentry_entries": entries,
        "postentry_exits": exits,
        "postentry_max_radius_fraction": (
            float(np.max(post_distance) / max(target_radius, 1e-9)) if len(post_distance) else 0.0
        ),
        "postentry_turn_degrees": _coarse_turn_degrees(
            post_future, bin_ms=shape.turn_bin_ms, minimum_bin_displacement=1.0
        ),
        "post_b_arc_chord": float(all_speed.sum()) / max(future_chord, 1e-9),
        "post_b_substantial_turn_degrees": _coarse_turn_degrees(
            future,
            bin_ms=shape.turn_bin_ms,
            minimum_bin_displacement=shape.turn_minimum_displacement,
        ),
        "post_b_lateral_reversals": _prominent_reversal_count(post_lateral, lateral_prominence),
        "post_b_axial_reversals": _prominent_reversal_count(post_axial, axial_prominence),
    }


def planner_shape_reasons(
    metrics: Mapping[str, float | int],
    config: PlannerShapeConfig | None = None,
) -> tuple[str, ...]:
    """Turn Planner shape measurements into stable rejection reason names."""

    shape = config or PlannerShapeConfig()
    reasons: list[str] = []
    arc = float(metrics["preentry_arc_chord"])
    regression = float(metrics["preentry_axial_regression_over_edge"])
    lateral = float(metrics["preentry_lateral_over_radius"])
    path_edge = float(metrics["preentry_path_over_edge"])
    pre_turn = float(metrics["preentry_substantial_turn_degrees"])
    pre_lat = int(metrics["preentry_lateral_reversals"])
    pre_ax = int(metrics["preentry_axial_reversals"])
    stationary = int(metrics["preentry_max_stationary_run_ms"])
    exits = int(metrics["postentry_exits"])
    post_max = float(metrics["postentry_max_radius_fraction"])
    post_turn = float(metrics["postentry_turn_degrees"])
    post_b_turn = float(metrics["post_b_substantial_turn_degrees"])
    post_lat = int(metrics["post_b_lateral_reversals"])
    post_ax = int(metrics["post_b_axial_reversals"])
    if arc > shape.extreme_arc_chord:
        reasons.append("preentry_arc_chord_extreme")
    if arc > shape.backtrack_arc_chord and regression > shape.axial_regression_over_edge:
        reasons.append("preentry_backtrack")
    if lateral > shape.lateral_over_radius and path_edge > shape.detour_path_over_edge:
        reasons.append("preentry_wide_detour")
    if pre_lat >= shape.reversal_count:
        reasons.append("preentry_multiple_lateral_swings")
    if pre_ax >= shape.reversal_count:
        reasons.append("preentry_multiple_axial_reversals")
    if pre_turn > shape.preentry_turn_degrees and (pre_lat >= 1 or pre_ax >= 1):
        reasons.append("preentry_substantial_turn_signature")
    if post_lat >= shape.reversal_count:
        reasons.append("post_b_multiple_lateral_swings")
    if post_ax >= shape.reversal_count:
        reasons.append("post_b_multiple_axial_reversals")
    if post_b_turn > shape.post_b_turn_degrees:
        reasons.append("post_b_substantial_turn_signature")
    if stationary >= shape.stationary_pause_ms:
        reasons.append("long_preentry_stationary_pause")
    if exits >= shape.repeated_target_exits:
        reasons.append("repeated_target_reentry")
    if exits >= shape.far_reentry_exits and post_max > shape.far_reentry_radius_fraction:
        reasons.append("far_reentry_signature")
    if exits >= shape.far_reentry_exits and post_turn > shape.postentry_turn_degrees:
        reasons.append("postentry_signature")
    return tuple(reasons)


EVENT_LEVEL_SHAPE_REASONS = frozenset(
    {
        "preentry_multiple_lateral_swings",
        "preentry_multiple_axial_reversals",
        "preentry_substantial_turn_signature",
        "post_b_multiple_lateral_swings",
        "post_b_multiple_axial_reversals",
        "post_b_substantial_turn_signature",
        "long_preentry_stationary_pause",
    }
)


def _is_borderline(metrics: Mapping[str, float | int]) -> bool:
    return bool(
        float(metrics["preentry_arc_chord"]) >= 1.60
        or (
            float(metrics["preentry_arc_chord"]) >= 1.35
            and float(metrics["preentry_axial_regression_over_edge"]) >= 0.075
        )
        or (
            float(metrics["preentry_lateral_over_radius"]) >= 1.25
            and float(metrics["preentry_path_over_edge"]) >= 1.40
        )
        or int(metrics["preentry_axial_reversals"]) >= 1
        or float(metrics["preentry_substantial_turn_degrees"]) >= 360.0
        or float(metrics["post_b_substantial_turn_degrees"]) >= 480.0
        or int(metrics["preentry_max_stationary_run_ms"]) >= 96
        or int(metrics["postentry_exits"]) >= 2
    )


def _seam_decision(
    event: PortableEvent,
    radius: float,
    threshold: float,
    seam: SeamPreparationConfig,
):
    return edge_progress_decision(
        event.dxdy,
        event.target_rel_at_a,
        radius,
        b_config=CausalBConfig(
            threshold=float(threshold),
            max_ab_ms=seam.max_ab_ms,
            min_remaining_counts=seam.min_remaining_counts,
            progress_regression_threshold=seam.progress_regression_threshold,
            max_center_progress=seam.max_center_progress,
        ),
        min_prefix_ms=seam.min_prefix_ms,
        min_future_ms=seam.min_future_ms,
    )


def _planner_cuts_for_radius(
    event_index: int,
    event: PortableEvent,
    radius: float,
    config: PlannerPreparationConfig,
    *,
    variant_id: str,
    synthetic: bool,
    thresholds: Sequence[float] | None = None,
) -> tuple[list[PreparedCut], set[str]]:
    threshold_values = (
        adaptive_planner_thresholds(event.target_rel_at_a, radius, config)
        if thresholds is None
        else np.asarray(tuple(thresholds), dtype=np.float64)
    )
    candidates: list[tuple[PreparedCut, tuple[str, ...], Mapping[str, float | int]]] = []
    failures: set[str] = set()
    for requested_index, threshold in enumerate(threshold_values):
        decision = _seam_decision(event, radius, float(threshold), config.seam)
        if not decision.training_eligible or decision.seam is None:
            if decision.seam is None:
                failures.add(decision.reason)
            failures.update(decision.eligibility_reasons)
            continue
        seam = decision.seam
        future_ms = len(event.dxdy) - seam.split_index
        edge_margin = math.hypot(seam.target_rel_at_b_x, seam.target_rel_at_b_y) - radius
        if future_ms > config.seam.max_future_ms:
            failures.add("future_exceeds_horizon")
            continue
        if edge_margin < config.seam.min_edge_margin_counts:
            failures.add("edge_margin_under_minimum")
            continue
        metrics = planner_shape_metrics(
            event.dxdy, event.target_rel_at_a, radius, seam.split_index, config.shape
        )
        reasons = planner_shape_reasons(metrics, config.shape)
        candidates.append(
            (
                PreparedCut(
                    event_index=event_index,
                    source_trial_id=event.source_trial_id,
                    variant_id=variant_id,
                    requested_index=requested_index,
                    split_index=seam.split_index,
                    threshold=float(threshold),
                    realized_progress=seam.realized_progress,
                    center_progress=seam.center_progress,
                    target_rel_at_b_x=seam.target_rel_at_b_x,
                    target_rel_at_b_y=seam.target_rel_at_b_y,
                    target_radius=float(radius),
                    synthetic=synthetic,
                    borderline=_is_borderline(metrics),
                ),
                reasons,
                metrics,
            )
        )

    # A finite 1 ms grid can make neighboring thresholds fire on the same
    # physical tick.  Keep the nominal threshold closest to realized progress.
    deduplicated: dict[int, tuple[PreparedCut, tuple[str, ...], Mapping[str, float | int]]] = {}
    for candidate in candidates:
        cut = candidate[0]
        previous = deduplicated.get(cut.split_index)
        if previous is None or (
            abs(cut.threshold - cut.realized_progress), -cut.threshold
        ) < (
            abs(previous[0].threshold - previous[0].realized_progress),
            -previous[0].threshold,
        ):
            deduplicated[cut.split_index] = candidate
    ordered = [deduplicated[key] for key in sorted(deduplicated)]
    event_veto = {
        reason
        for _, reasons, _ in ordered
        for reason in reasons
        if reason in EVENT_LEVEL_SHAPE_REASONS
    }
    retained = [cut for cut, reasons, _ in ordered if not reasons]
    for _, reasons, _ in ordered:
        failures.update(reasons)
    return retained, event_veto or failures


def _planner_base_reasons(event: PortableEvent, config: PlannerPreparationConfig) -> tuple[str, ...]:
    if not config.apply_quality_filter:
        return ()
    quality = shot_quality(event.dxdy, event.target_rel_at_a, event.target_radius)
    reasons = list(
        shot_filter_reasons(
            quality,
            outcome=event.outcome,
            technical_outcome=event.technical_outcome,
            policy=config.quality,
        )
    )
    # A clean, efficient correction is retained here. The count-scale signature
    # gates decide separately whether it is a repeated behavioral pattern.
    if reasons == ["excessive_turning"] and (
        quality.path_efficiency <= config.recover_turn_only_if_path_efficiency_at_most
        and quality.radial_return_fraction <= config.recover_turn_only_if_radial_return_at_most
    ):
        return ()
    return tuple(reasons)


def _stable_key(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_robin_by_user(
    candidates: Sequence[tuple[int, PortableEvent]],
    *,
    radius: float,
    salt: str,
) -> list[tuple[int, PortableEvent]]:
    groups: dict[str, list[tuple[int, PortableEvent]]] = {}
    for item in candidates:
        identity = item[1].user_id if item[1].user_id != "unknown" else item[1].session_id
        groups.setdefault(identity, []).append(item)
    for identity, values in groups.items():
        values.sort(key=lambda item: _stable_key(salt, radius, identity, item[1].source_trial_id))
    users = sorted(groups, key=lambda user: _stable_key(salt, radius, "users", user))
    output: list[tuple[int, PortableEvent]] = []
    index = 0
    while users:
        remaining: list[str] = []
        for user in users:
            values = groups[user]
            if index < len(values):
                output.append(values[index])
            if index + 1 < len(values):
                remaining.append(user)
        users = remaining
        index += 1
    return output


def _tiny_eligible(event: PortableEvent, radius: float, tiny: TinyTargetConfig) -> bool:
    if event.split != tiny.train_split:
        return False
    if event.target_radius < tiny.parent_radius_ratio_min * radius:
        return False
    path = np.cumsum(np.asarray(event.dxdy, dtype=np.float64), axis=0)
    if len(path) < tiny.final_inside_ms:
        return False
    distance = np.linalg.norm(np.asarray(event.target_rel_at_a)[None, :] - path, axis=1)
    return bool(
        distance[-1] <= tiny.endpoint_radius_fraction_max * radius
        and np.all(distance[-tiny.final_inside_ms :] <= radius)
    )


def prepare_planner(
    events: Iterable[PortableEvent],
    config: PlannerPreparationConfig | None = None,
) -> PreparationResult:
    """Prepare source-balanced dense Planner seams from portable events."""

    cfg = config or PlannerPreparationConfig()
    rows = tuple(events)
    _validate_unique_sources(rows)
    native_by_source: dict[str, list[PreparedCut]] = {}
    rejected: dict[str, tuple[str, ...]] = {}
    eligible_for_tiny: list[tuple[int, PortableEvent]] = []
    for event_index, event in enumerate(rows):
        base_reasons = _planner_base_reasons(event, cfg)
        if base_reasons:
            rejected[event.source_trial_id] = base_reasons
            continue
        cuts, observed_reasons = _planner_cuts_for_radius(
            event_index,
            event,
            event.target_radius,
            cfg,
            variant_id="native",
            synthetic=False,
            thresholds=(
                None
                if event.split == cfg.dense_train_split
                else (cfg.runtime_b_threshold,)
            ),
        )
        event_veto = sorted(EVENT_LEVEL_SHAPE_REASONS.intersection(observed_reasons))
        if event_veto:
            rejected[event.source_trial_id] = tuple(f"event_veto:{reason}" for reason in event_veto)
            continue
        if not cuts:
            rejected[event.source_trial_id] = tuple(sorted(observed_reasons)) or ("no_planner_cut",)
            continue
        native_by_source[event.source_trial_id] = cuts
        eligible_for_tiny.append((event_index, event))

    synthetic_by_source: dict[str, list[PreparedCut]] = {}
    tiny = cfg.tiny_target
    if tiny.enabled:
        used: set[str] = set()
        for radius, quota in sorted(tiny.radius_quotas):
            candidates = [
                item
                for item in eligible_for_tiny
                if item[1].source_trial_id not in used and _tiny_eligible(item[1], radius, tiny)
            ]
            selected = 0
            for event_index, event in _round_robin_by_user(
                candidates, radius=radius, salt=tiny.selection_salt
            ):
                if selected >= quota:
                    break
                synthetic, _ = _planner_cuts_for_radius(
                    event_index,
                    event,
                    radius,
                    cfg,
                    variant_id=f"synthetic_r{radius:g}",
                    synthetic=True,
                )
                native = native_by_source[event.source_trial_id]
                available = {cut.requested_index: cut for cut in synthetic}
                kept: list[PreparedCut] = []
                # Match the trusted native B indices and cap synthetic rows at
                # the native count, hence synthetic share <= one half.
                for native_cut in sorted(native, key=lambda cut: cut.requested_index):
                    if not available:
                        break
                    requested = min(
                        available,
                        key=lambda value: (abs(value - native_cut.requested_index), value),
                    )
                    kept.append(available.pop(requested))
                if not kept:
                    continue
                synthetic_by_source[event.source_trial_id] = kept
                used.add(event.source_trial_id)
                selected += 1

    cuts: list[PreparedCut] = []
    for source_id, native in native_by_source.items():
        cuts.extend(native)
        cuts.extend(synthetic_by_source.get(source_id, ()))
    weights = per_source_trial_weights(cut.source_trial_id for cut in cuts)
    weighted = tuple(replace(cut, example_weight=float(weight)) for cut, weight in zip(cuts, weights))
    result = PreparationResult("planner", rows, weighted, rejected, cfg)
    if result.source_weight_error() > 1e-6:
        raise DatasetPreparationError("Planner source weights do not sum to one")
    return result


def subset_preparation_result(result: PreparationResult, split: str) -> PreparationResult:
    """Select one named split and remap event indices for standalone training."""

    selected_old = [
        index for index, event in enumerate(result.events) if event.split == split
    ]
    index_map = {old: new for new, old in enumerate(selected_old)}
    events = tuple(result.events[index] for index in selected_old)
    cuts = tuple(
        replace(cut, event_index=index_map[cut.event_index])
        for cut in result.cuts
        if cut.event_index in index_map
    )
    source_ids = {event.source_trial_id for event in events}
    rejected = {
        source_id: reasons
        for source_id, reasons in result.rejected.items()
        if source_id in source_ids
    }
    return PreparationResult(result.branch, events, cuts, rejected, result.config)


def _jsonable_config(config: Any) -> dict[str, Any]:
    result = asdict(config)
    if isinstance(config, PlannerPreparationConfig):
        result["tiny_target"]["radius_quotas"] = {
            f"{radius:g}": quota for radius, quota in config.tiny_target.radius_quotas
        }
    return result


def preparation_manifest(result: PreparationResult) -> dict[str, Any]:
    """Build the compact provenance/accounting manifest for one branch."""

    rejection_counts = Counter(reason for values in result.rejected.values() for reason in values)
    accepted_sources = {cut.source_trial_id for cut in result.cuts}
    materialization = result.config.materialization
    return {
        "schema": PREPARATION_MANIFEST_SCHEMA,
        "branch": result.branch,
        "input_event_count": len(result.events),
        "accepted_source_count": len(accepted_sources),
        "rejected_source_count": len(result.rejected),
        "row_count": len(result.cuts),
        "native_row_count": sum(not cut.synthetic for cut in result.cuts),
        "synthetic_row_count": sum(cut.synthetic for cut in result.cuts),
        "splits": sorted({event.split for event in result.events}),
        "source_weight_max_abs_error": result.source_weight_error(),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "training_contract": {
            "directly_trainable": True,
            "prefix_ticks": materialization.prefix_ticks,
            "future_horizon_ms": materialization.future_horizon_ms,
            "future_smoothing": materialization.smoothing_spec,
            "required_arrays": [
                "prefix_raw_dxdy",
                "prefix_mask",
                "future_raw_dxdy",
                "future_mask",
                "future_smooth_dxdy",
                "target_rel_x_at_B",
                "target_rel_y_at_B",
                "target_radius",
                "progress",
                "speed_at_B",
                "accel_at_B",
                "outcome",
                "source_trial_id",
                "event_weight",
            ],
        },
        "config": _jsonable_config(result.config),
        "limitations": [
            "input is already cropped to causal A and fixed C by the Capture exporter",
            "native Capture ZIP clock/session auditing is not reimplemented here",
            "tiny-target augmentation is metadata/label coverage for Planner only",
        ],
    }


def _causal_contract_for_result(result: PreparationResult) -> dict[str, object]:
    config = result.config
    thresholds = sorted({round(float(cut.threshold), 12) for cut in result.cuts})
    if not thresholds:
        thresholds = [
            config.runtime_b_threshold
            if isinstance(config, PlannerPreparationConfig)
            else config.b_threshold
        ]
    b_configs = [
        CausalBConfig(
            threshold=threshold,
            max_ab_ms=config.seam.max_ab_ms,
            min_remaining_counts=config.seam.min_remaining_counts,
            progress_regression_threshold=config.seam.progress_regression_threshold,
            max_center_progress=config.seam.max_center_progress,
        )
        for threshold in thresholds
    ]
    return causal_seam_contract_record(
        config.materialization.onset,
        b_configs,
        SeamEligibility(
            min_prefix_ms=config.seam.min_prefix_ms,
            min_future_ms=config.seam.min_future_ms,
        ),
    )


def _trainable_arrays(
    result: PreparationResult,
) -> dict[str, np.ndarray]:
    """Materialize the exact padded fields consumed by public trainers."""

    materialization = result.config.materialization
    prefix_ticks = int(materialization.prefix_ticks)
    horizon = int(materialization.future_horizon_ms)
    row_count = len(result.cuts)
    prefix_raw = np.zeros((row_count, prefix_ticks, 2), dtype=np.float32)
    prefix_mask = np.zeros((row_count, prefix_ticks), dtype=np.float32)
    future_raw = np.zeros((row_count, horizon, 2), dtype=np.float32)
    future_smooth = np.zeros((row_count, horizon, 2), dtype=np.float32)
    future_mask = np.zeros((row_count, horizon), dtype=np.float32)
    target_b = np.zeros((row_count, 2), dtype=np.float32)
    speed_at_b = np.zeros(row_count, dtype=np.float32)
    accel_at_b = np.zeros(row_count, dtype=np.float32)
    duration = np.zeros(row_count, dtype=np.int32)
    smooth_cache: dict[int, np.ndarray] = {}

    # ``result.cuts`` still indexes result.events, while accepted_events is a
    # compact audit view.  Smoothing is cached by the original event index so
    # dense siblings share one exact A-to-C smoothing pass.
    for row_index, cut in enumerate(result.cuts):
        event = result.events[cut.event_index]
        raw = np.asarray(event.dxdy, dtype=np.float32)
        split = int(cut.split_index)
        if not 0 < split < len(raw):
            raise DatasetPreparationError(
                f"invalid retained split for {event.source_trial_id}: {split}"
            )
        prefix = raw[:split]
        prefix_take = min(prefix_ticks, len(prefix))
        if prefix_take:
            prefix_raw[row_index, -prefix_take:] = prefix[-prefix_take:]
            prefix_mask[row_index, -prefix_take:] = 1.0
        future = raw[split:]
        if len(future) > horizon:
            raise DatasetPreparationError(
                f"future exceeds materialized horizon for {event.source_trial_id}"
            )
        duration[row_index] = len(future)
        future_raw[row_index, : len(future)] = future
        future_mask[row_index, : len(future)] = 1.0
        event_smooth = smooth_cache.get(cut.event_index)
        if event_smooth is None:
            event_smooth = smooth_dxdy(
                raw,
                spec=materialization.smoothing_spec,
            ).astype(np.float32)
            smooth_cache[cut.event_index] = event_smooth
        future_smooth[row_index, : len(future)] = event_smooth[split:]
        target_b[row_index] = (cut.target_rel_at_b_x, cut.target_rel_at_b_y)
        last = prefix[-1]
        previous = prefix[-2] if len(prefix) >= 2 else last
        speed_at_b[row_index] = float(np.linalg.norm(last))
        accel_at_b[row_index] = float(np.linalg.norm(last - previous))

    event_by_index = [result.events[cut.event_index] for cut in result.cuts]
    distances = np.linalg.norm(target_b.astype(np.float64), axis=1).astype(np.float32)
    thresholds = np.asarray([cut.threshold for cut in result.cuts], dtype=np.float32)
    variant_ids = [cut.variant_id for cut in result.cuts]
    cut_ids = [
        f"{cut.variant_id}:k{cut.requested_index + 1:02d}_b{100.0 * cut.threshold:.3f}"
        for cut in result.cuts
    ]
    row_ids = [
        f"{cut.source_trial_id}:{cut_ids[index]}" for index, cut in enumerate(result.cuts)
    ]
    contract = _causal_contract_for_result(result)
    split_names = sorted({event.split for event in event_by_index})
    return {
        "prefix_raw_dxdy": prefix_raw,
        "prefix_mask": prefix_mask,
        "future_raw_dxdy": future_raw,
        "future_mask": future_mask,
        "future_smooth_dxdy": future_smooth,
        "target_rel_x_at_B": target_b[:, 0],
        "target_rel_y_at_B": target_b[:, 1],
        "target_rel_at_B": target_b,
        "target_radius": np.asarray([cut.target_radius for cut in result.cuts], dtype=np.float32),
        "target_distance_at_B": distances,
        "distance_over_radius_at_B": distances
        / np.maximum(
            np.asarray([cut.target_radius for cut in result.cuts], dtype=np.float32),
            np.float32(1e-9),
        ),
        "progress": np.asarray([cut.center_progress for cut in result.cuts], dtype=np.float32),
        "edge_trigger_progress": thresholds,
        "realized_edge_progress": np.asarray(
            [cut.realized_progress for cut in result.cuts], dtype=np.float32
        ),
        "speed_at_B": speed_at_b,
        "accel_at_B": accel_at_b,
        "event_weight": np.asarray(
            [cut.example_weight for cut in result.cuts], dtype=np.float32
        ),
        "source_trial_id": _text_array([cut.source_trial_id for cut in result.cuts]),
        "cut_id": _text_array(cut_ids),
        "event_id": _text_array(row_ids),
        "variant_id": _text_array(variant_ids),
        "outcome": _text_array([event.outcome for event in event_by_index]),
        "technical_outcome": _text_array(
            [event.technical_outcome for event in event_by_index]
        ),
        "user_id": _text_array([event.user_id for event in event_by_index]),
        "session_id": _text_array([event.session_id for event in event_by_index]),
        "split": _text_array([event.split for event in event_by_index]),
        "b_index": np.asarray([cut.split_index for cut in result.cuts], dtype=np.int32),
        "future_duration_ms": duration,
        "schema": np.asarray(PREPARED_CUT_SCHEMA),
        "cohort": np.asarray(
            f"{result.branch}_{split_names[0] if len(split_names) == 1 else 'mixed'}"
        ),
        "smoothing": np.asarray(materialization.smoothing_spec),
        "causal_seam_contract_schema": np.asarray(str(contract["schema"])),
        "causal_seam_contract_json": np.asarray(
            json.dumps(contract, sort_keys=True, separators=(",", ":"))
        ),
    }


def save_prepared_dataset(result: PreparationResult, path: str | Path) -> tuple[Path, Path]:
    """Write one self-contained, directly trainable NPZ plus audit manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    accepted_indices = sorted({cut.event_index for cut in result.cuts})
    index_map = {old: new for new, old in enumerate(accepted_indices)}
    accepted_events = [result.events[index] for index in accepted_indices]
    offsets = np.zeros(len(accepted_events) + 1, dtype=np.int64)
    for index, event in enumerate(accepted_events):
        offsets[index + 1] = offsets[index] + len(event.dxdy)
    raw = (
        np.concatenate([np.asarray(event.dxdy, dtype=np.float32) for event in accepted_events])
        if accepted_events
        else np.zeros((0, 2), dtype=np.float32)
    )
    arrays = _trainable_arrays(result)
    arrays.update(
        {
            # Compact physical-event audit representation.  Row-level names
            # intentionally remain trainer-compatible; event-level identity
            # is prefixed to avoid shape ambiguity in ``abcurves.data.subset``.
            "branch": np.asarray(result.branch),
            "dxdy": raw,
            "event_offsets": offsets,
            "event_source_trial_id": _text_array(
                [event.source_trial_id for event in accepted_events]
            ),
            "target_rel_at_a": np.asarray(
                [event.target_rel_at_a for event in accepted_events], dtype=np.float32
            ).reshape(-1, 2),
            "original_target_radius": np.asarray(
                [event.target_radius for event in accepted_events], dtype=np.float32
            ),
            "event_user_id": _text_array([event.user_id for event in accepted_events]),
            "event_session_id": _text_array(
                [event.session_id for event in accepted_events]
            ),
            "event_split": _text_array([event.split for event in accepted_events]),
            "row_event_index": np.asarray(
                [index_map[cut.event_index] for cut in result.cuts], dtype=np.int64
            ),
            "requested_index": np.asarray(
                [cut.requested_index for cut in result.cuts], dtype=np.int16
            ),
            "split_index": np.asarray(
                [cut.split_index for cut in result.cuts], dtype=np.int32
            ),
            "b_threshold": np.asarray(
                [cut.threshold for cut in result.cuts], dtype=np.float32
            ),
            "realized_progress": np.asarray(
                [cut.realized_progress for cut in result.cuts], dtype=np.float32
            ),
            "center_progress": np.asarray(
                [cut.center_progress for cut in result.cuts], dtype=np.float32
            ),
            "target_rel_at_b": np.asarray(
                [[cut.target_rel_at_b_x, cut.target_rel_at_b_y] for cut in result.cuts],
                dtype=np.float32,
            ).reshape(-1, 2),
            "example_weight": np.asarray(
                [cut.example_weight for cut in result.cuts], dtype=np.float32
            ),
            "synthetic": np.asarray([cut.synthetic for cut in result.cuts], dtype=np.bool_),
            "borderline": np.asarray([cut.borderline for cut in result.cuts], dtype=np.bool_),
        }
    )
    with destination.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    digest = _file_sha256(destination)
    rejection_path = destination.with_suffix(".rejections.csv")
    with rejection_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_trial_id", "reasons"))
        for source_id, reasons in sorted(result.rejected.items()):
            writer.writerow((source_id, "|".join(reasons)))
    rejection_digest = _file_sha256(rejection_path)
    manifest = preparation_manifest(result)
    manifest["prepared_npz"] = destination.name
    manifest["prepared_npz_sha256"] = digest
    manifest["rejected_sources_csv"] = rejection_path.name
    manifest["rejected_sources_csv_sha256"] = rejection_digest
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination, manifest_path


__all__ = [
    "DatasetPreparationError",
    "EVENT_LEVEL_SHAPE_REASONS",
    "NativeCaptureExporterUnavailable",
    "MaterializationConfig",
    "PORTABLE_EVENT_SCHEMA",
    "PREPARATION_MANIFEST_SCHEMA",
    "PREPARED_CUT_SCHEMA",
    "PlannerPreparationConfig",
    "PlannerShapeConfig",
    "PortableEvent",
    "PreparationResult",
    "ResearchExportConversion",
    "SeamPreparationConfig",
    "TinyTargetConfig",
    "adaptive_planner_thresholds",
    "load_portable_events",
    "load_research_export_events",
    "planner_shape_metrics",
    "planner_shape_reasons",
    "preparation_manifest",
    "prepare_planner",
    "save_prepared_dataset",
    "subset_preparation_result",
    "write_portable_events",
]
