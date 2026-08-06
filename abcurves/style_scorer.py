"""Frozen human-event scorer for the Renderer C/M/H style adapter.

The scorer is deliberately small and deterministic.  A completed human B->C
count stream is reduced to the public texture19 panel, nuisance-adjusted by
the frozen all-training ridge, standardized by its frozen residual scales,
and projected onto the three safe C (cadence), M (magnitude), and H
(high-frequency) directions.

These event scores are observations, not the state supplied to the Renderer.
Feed completed *human* scores to :class:`abcurves.personalization.CausalStyleState`;
that class applies the trailing-ten, half-shrinkage, same-run contract.  Never
feed generated events back into the state.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .features import target_frame_basis
from .model_store import default_model_dir
from .personalization import CausalStyleState
from .smoothing import SmoothingSpec, smooth_dxdy
from .texture import TEXTURE_FEATURE_NAMES, texture_features


SCHEMA = "abcurves.style_scorer.v1"
STYLE_SCORE_NAMES = ("C", "M", "H")

CAUSAL_CONTEXT_NAMES = (
    "target_radius",
    "target_distance_at_A",
    "target_distance_at_B",
    "progress",
    "edge_trigger_progress",
    "edge_realized_progress",
    "prefix_duration_ms",
    "prefix_path_length",
    "prefix_net_distance",
    "prefix_straightness",
    "prefix_speed_mean",
    "prefix_speed_std",
    "prefix_speed_max",
    "prefix_tail_speed_mean",
    "prefix_recent_speed_slope",
    "prefix_zero_rate",
    "prefix_sign_flip_rate",
    "prefix_approach_cos",
    "prefix_approach_sin",
    "prefix_recent_lateral_fraction",
)

PLANNED_TRAJECTORY_NAMES = (
    "log_duration",
    "speed_mean",
    "speed_max",
    "peak_speed_fraction",
    "tail_speed_ratio",
    "decel_tail_slope",
    "jerk_roughness_log1p",
)

TASK_LABELS = (
    "accuracy_precision_miss_sensitive",
    "chain_micro_reacquire",
    "default_static_flick",
    "dwell_stabilize_static",
    "fast_flick_timed",
    "microadjust_close_static",
    "overshoot_recovery_static",
    "precision_big_timed",
    "precision_small_timed",
    "reacceleration_precision_switch",
)

TARGET_ROLE_LABELS = ("general", "tickle_reset", "tickle_small")

CONTEXT_FEATURE_NAMES = (
    *CAUSAL_CONTEXT_NAMES,
    *(f"planned::{name}" for name in PLANNED_TRAJECTORY_NAMES),
    *(f"task::{name}" for name in TASK_LABELS),
    *(f"role::{name}" for name in TARGET_ROLE_LABELS),
)

HUMAN_SHAPE_SMOOTHING = SmoothingSpec(
    "triangular_moving_average_path",
    window=5,
    preserve_endpoint=False,
)


def default_style_scorer_path() -> Path:
    """Return the repository's path-free frozen transform artifact."""

    return default_model_dir() / "style_scorer.json"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _finite_vector(value: Sequence[float] | np.ndarray, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {width} finite values")
    return result


def _valid_stream(
    dxdy: Sequence[Sequence[float]] | np.ndarray,
    mask: Sequence[float] | np.ndarray | None,
    name: str,
) -> np.ndarray:
    values = np.asarray(dxdy, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must have shape (ticks, 2) with finite values")
    if mask is None:
        return values
    valid = np.asarray(mask, dtype=np.float64)
    if valid.shape != (len(values),) or not np.all(np.isfinite(valid)):
        raise ValueError(f"{name}_mask must have shape ({len(values)},)")
    return values[valid > 0.5]


def _linear_slope(values: np.ndarray) -> float:
    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(sequence) < 2:
        return 0.0
    x = np.arange(len(sequence), dtype=np.float64)
    x -= np.mean(x)
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(x * (sequence - np.mean(sequence))) / denominator)


def _prefix_sign_flip_rate(prefix: np.ndarray) -> float:
    flips = 0
    comparisons = 0
    for axis in range(2):
        active = prefix[:, axis][np.abs(prefix[:, axis]) > 0.0]
        signs = np.sign(active)
        if len(signs) >= 2:
            flips += int(np.sum(signs[1:] != signs[:-1]))
            comparisons += len(signs) - 1
    return float(flips / comparisons) if comparisons else 0.0


def causal_context_features(
    prefix_dxdy: Sequence[Sequence[float]] | np.ndarray,
    target_rel_at_b: Sequence[float] | np.ndarray,
    target_radius: float,
    progress: float,
    *,
    prefix_mask: Sequence[float] | np.ndarray | None = None,
    target_distance_at_a: float | None = None,
    edge_trigger_progress: float | None = None,
    edge_realized_progress: float | None = None,
) -> dict[str, float]:
    """Compute the exact frozen 20-value target/prefix nuisance context.

    The clock is 1 kHz, so the number of valid prefix ticks is also its
    duration in milliseconds.  If ``target_distance_at_a`` is unavailable,
    the frozen fallback ``distance_at_B / (1 - progress)`` is used.
    """

    prefix = _valid_stream(prefix_dxdy, prefix_mask, "prefix_dxdy")
    target = _finite_vector(target_rel_at_b, 2, "target_rel_at_b")
    radius = float(target_radius)
    event_progress = float(progress)
    if not math.isfinite(radius) or not math.isfinite(event_progress):
        raise ValueError("target_radius and progress must be finite")

    distance_b = float(np.linalg.norm(target))
    distance_a = (
        distance_b / max(1.0 - event_progress, 1e-6)
        if target_distance_at_a is None
        else float(target_distance_at_a)
    )
    edge_trigger = event_progress if edge_trigger_progress is None else float(edge_trigger_progress)
    edge_realized = edge_trigger if edge_realized_progress is None else float(edge_realized_progress)
    if not all(math.isfinite(value) for value in (distance_a, edge_trigger, edge_realized)):
        raise ValueError("target and edge progress values must be finite")

    speed = np.linalg.norm(prefix, axis=1) if len(prefix) else np.zeros(0, dtype=np.float64)
    path_length = float(np.sum(speed))
    displacement = np.sum(prefix, axis=0) if len(prefix) else np.zeros(2, dtype=np.float64)
    net_distance = float(np.linalg.norm(displacement))
    toward, tangent = target_frame_basis(target)
    approach = np.sum(prefix[-10:], axis=0) if len(prefix) else np.zeros(2, dtype=np.float64)
    approach_norm = float(np.linalg.norm(approach))
    approach_cos = float(np.dot(approach, toward) / approach_norm) if approach_norm else 0.0
    approach_sin = float(np.dot(approach, tangent) / approach_norm) if approach_norm else 0.0
    recent = prefix[-16:]
    if len(recent):
        recent_target_frame = np.stack([recent @ toward, recent @ tangent], axis=1)
        along_energy = float(np.sum(np.abs(recent_target_frame[:, 0])))
        lateral_energy = float(np.sum(np.abs(recent_target_frame[:, 1])))
        lateral_fraction = lateral_energy / max(along_energy + lateral_energy, 1e-9)
    else:
        lateral_fraction = 0.0

    values = (
        radius,
        distance_a,
        distance_b,
        event_progress,
        edge_trigger,
        edge_realized,
        float(len(prefix)),
        path_length,
        net_distance,
        net_distance / max(path_length, 1e-9),
        float(np.mean(speed)) if len(speed) else 0.0,
        float(np.std(speed)) if len(speed) else 0.0,
        float(np.max(speed)) if len(speed) else 0.0,
        float(np.mean(speed[-16:])) if len(speed) else 0.0,
        _linear_slope(speed[-32:]),
        float(np.mean(speed <= 0.0)) if len(speed) else 1.0,
        _prefix_sign_flip_rate(prefix),
        approach_cos,
        approach_sin,
        lateral_fraction,
    )
    return dict(zip(CAUSAL_CONTEXT_NAMES, values, strict=True))


def planned_trajectory_features(
    planned_dxdy: Sequence[Sequence[float]] | np.ndarray,
    target_rel_at_b: Sequence[float] | np.ndarray,
    target_radius: float,
    *,
    planned_mask: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the seven frozen shape values available to the Renderer.

    For a Stage-0 human observation, ``planned_dxdy`` is the B->C slice of the
    canonical triangular-path-w5 smoothing of the concatenated prefix and that
    same completed human stream.  It is not a shadow Planner rollout, and B->C
    must not be smoothed in isolation.  Prefer :func:`completed_human_context`,
    which enforces this boundary contract.  The ``planned::`` prefix is kept
    because it is the frozen coefficient artifact's declared field name.
    """

    movement = _valid_stream(planned_dxdy, planned_mask, "planned_dxdy")
    target = _finite_vector(target_rel_at_b, 2, "target_rel_at_b")
    radius = float(target_radius)
    if not math.isfinite(radius):
        raise ValueError("target_radius must be finite")
    duration = len(movement)
    if duration < 2:
        values = np.zeros(len(PLANNED_TRAJECTORY_NAMES), dtype=np.float64)
    else:
        speed = np.linalg.norm(movement, axis=1)
        peak_index = int(np.argmax(speed))
        tail = speed[int(0.75 * duration) :]
        tail_ratio = (
            float(np.mean(tail) / max(np.mean(speed), 1e-6)) if len(tail) else 0.0
        )
        deceleration_segment = speed[int(0.66 * duration) :]
        deceleration = (
            float(
                np.polyfit(
                    np.arange(len(deceleration_segment), dtype=np.float64),
                    deceleration_segment,
                    1,
                )[0]
            )
            if len(deceleration_segment) >= 2
            else 0.0
        )
        jerk = (
            np.linalg.norm(np.diff(movement, n=2, axis=0), axis=1)
            if duration >= 3
            else np.zeros(1, dtype=np.float64)
        )
        values = np.asarray(
            [
                math.log(max(float(duration), 1.0)),
                float(np.mean(speed)),
                float(np.max(speed)),
                peak_index / max(duration - 1, 1),
                tail_ratio,
                deceleration,
                math.log1p(max(float(np.mean(jerk)), 0.0)),
            ],
            dtype=np.float64,
        )
    return {
        f"planned::{name}": float(value)
        for name, value in zip(PLANNED_TRAJECTORY_NAMES, values, strict=True)
    }


def _component_vector(
    values: Mapping[str, float] | Sequence[float] | np.ndarray,
    names: tuple[str, ...],
    label: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"{label} lacks required fields: {', '.join(missing)}")
        return _finite_vector([values[name] for name in names], len(names), label)
    return _finite_vector(values, len(names), label)


def renderer_context(
    causal: Mapping[str, float] | Sequence[float] | np.ndarray,
    planned: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    task_type: str,
    target_role: str,
) -> np.ndarray:
    """Assemble the exact ordered 40-value nuisance vector.

    ``causal`` may be the result of :func:`causal_context_features` or an
    ordered 20-vector.  ``planned`` may use either base names such as
    ``speed_mean`` or the explicit ``planned::speed_mean`` keys.
    """

    causal_vector = _component_vector(causal, CAUSAL_CONTEXT_NAMES, "causal context")
    if isinstance(planned, Mapping):
        normalized = {
            name: planned[name]
            if name in planned
            else planned.get(f"planned::{name}")
            for name in PLANNED_TRAJECTORY_NAMES
        }
        if any(value is None for value in normalized.values()):
            missing = [name for name, value in normalized.items() if value is None]
            raise ValueError(f"planned context lacks required fields: {', '.join(missing)}")
        planned_vector = _finite_vector(
            [normalized[name] for name in PLANNED_TRAJECTORY_NAMES],
            len(PLANNED_TRAJECTORY_NAMES),
            "planned context",
        )
    else:
        planned_vector = _finite_vector(
            planned, len(PLANNED_TRAJECTORY_NAMES), "planned context"
        )
    task = str(task_type)
    role = str(target_role)
    if task not in TASK_LABELS:
        raise ValueError(f"task_type is outside the frozen vocabulary: {task!r}")
    if role not in TARGET_ROLE_LABELS:
        raise ValueError(f"target_role is outside the frozen vocabulary: {role!r}")
    task_one_hot = np.asarray([task == label for label in TASK_LABELS], dtype=np.float64)
    role_one_hot = np.asarray([role == label for label in TARGET_ROLE_LABELS], dtype=np.float64)
    result = np.concatenate([causal_vector, planned_vector, task_one_hot, role_one_hot])
    return _finite_vector(result, len(CONTEXT_FEATURE_NAMES), "renderer context")


def completed_human_context(
    prefix_dxdy: Sequence[Sequence[float]] | np.ndarray,
    completed_raw_dxdy: Sequence[Sequence[float]] | np.ndarray,
    target_rel_at_b: Sequence[float] | np.ndarray,
    target_radius: float,
    progress: float,
    *,
    task_type: str,
    target_role: str,
    prefix_mask: Sequence[float] | np.ndarray | None = None,
    completed_mask: Sequence[float] | np.ndarray | None = None,
    target_distance_at_a: float | None = None,
    edge_trigger_progress: float | None = None,
    edge_realized_progress: float | None = None,
) -> np.ndarray:
    """Build all 40 frozen context values for a completed human event.

    This convenience path applies the exact triangular-path-w5 transform used
    to create the frozen human reference caches.  Smoothing is performed on
    the concatenated observed prefix and completion and only then sliced at B;
    smoothing B->C in isolation would change the seam boundary.  The raw
    completion remains untouched for texture scoring.
    """

    prefix = _valid_stream(prefix_dxdy, prefix_mask, "prefix_dxdy")
    completed = _valid_stream(
        completed_raw_dxdy, completed_mask, "completed_raw_dxdy"
    )
    full_event = np.concatenate([prefix, completed], axis=0)
    smoothed = smooth_dxdy(full_event, HUMAN_SHAPE_SMOOTHING)[len(prefix) :]
    causal = causal_context_features(
        prefix,
        target_rel_at_b,
        target_radius,
        progress,
        target_distance_at_a=target_distance_at_a,
        edge_trigger_progress=edge_trigger_progress,
        edge_realized_progress=edge_realized_progress,
    )
    planned = planned_trajectory_features(
        smoothed,
        target_rel_at_b,
        target_radius,
    )
    return renderer_context(
        causal,
        planned,
        task_type=task_type,
        target_role=target_role,
    )


def ordered_context(
    value: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Copy a complete named or already ordered frozen context vector."""

    return _component_vector(value, CONTEXT_FEATURE_NAMES, "renderer context")


class FrozenStyleScorer:
    """Load and execute the exact frozen all-training deployment transform."""

    def __init__(self, artifact: str | Path | None = None) -> None:
        self.path = default_style_scorer_path() if artifact is None else Path(artifact)
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read frozen style transform: {self.path}") from exc
        self._validate_record(record)
        nuisance = record["nuisance_transform"]
        self._mean = np.asarray(nuisance["x_mean"], dtype=np.float64)
        self._scale = np.asarray(nuisance["x_scale"], dtype=np.float64)
        self._beta = np.asarray(nuisance["beta"], dtype=np.float64)
        self._residual_scale = np.asarray(nuisance["residual_scale"], dtype=np.float64)
        self._groups = {
            group: np.asarray(
                [TEXTURE_FEATURE_NAMES.index(name) for name in record["safe_groups"][group]],
                dtype=np.int64,
            )
            for group in STYLE_SCORE_NAMES
        }
        self._directions = {
            group: np.asarray(record["directions"][group], dtype=np.float64)
            for group in STYLE_SCORE_NAMES
        }
        self.contract_sha256 = str(record["contract_sha256"])

    @staticmethod
    def _validate_record(record: Mapping[str, Any]) -> None:
        if record.get("schema") != SCHEMA:
            raise ValueError("unsupported style transform schema")
        expected_contract = str(record.get("contract_sha256", ""))
        payload = dict(record)
        payload.pop("contract_sha256", None)
        observed_contract = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        if observed_contract != expected_contract:
            raise ValueError("style transform contract hash differs")
        if tuple(record.get("score_names", ())) != STYLE_SCORE_NAMES:
            raise ValueError("style score order differs")
        if tuple(record.get("texture_feature_names", ())) != TEXTURE_FEATURE_NAMES:
            raise ValueError("texture feature contract differs")
        if tuple(record.get("context_feature_names", ())) != CONTEXT_FEATURE_NAMES:
            raise ValueError("style context contract differs")
        nuisance = record.get("nuisance_transform", {})
        arrays = {
            "x_mean": (nuisance.get("x_mean"), (40,)),
            "x_scale": (nuisance.get("x_scale"), (40,)),
            "beta": (nuisance.get("beta"), (41, 19)),
            "residual_scale": (nuisance.get("residual_scale"), (19,)),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"style transform {name} contract differs")
            if name in {"x_scale", "residual_scale"} and np.any(array <= 0.0):
                raise ValueError(f"style transform {name} must be positive")
        for group in STYLE_SCORE_NAMES:
            names = tuple(record.get("safe_groups", {}).get(group, ()))
            direction = np.asarray(record.get("directions", {}).get(group), dtype=np.float64)
            if not names or direction.shape != (len(names),) or not np.all(np.isfinite(direction)):
                raise ValueError(f"style direction {group} contract differs")

    def score_texture(
        self,
        texture19: Sequence[float] | np.ndarray,
        context: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Score one event or a batch of events from texture and context."""

        texture = np.asarray(texture19, dtype=np.float64)
        squeeze = texture.ndim == 1
        if squeeze:
            texture = texture[None, :]
        if texture.ndim != 2 or texture.shape[1] != len(TEXTURE_FEATURE_NAMES):
            raise ValueError("texture19 must have shape (19,) or (events, 19)")
        if not np.all(np.isfinite(texture)):
            raise ValueError("texture19 must be finite")

        if isinstance(context, Mapping):
            contexts = ordered_context(context)[None, :]
        else:
            contexts = np.asarray(context, dtype=np.float64)
            if contexts.ndim == 1:
                contexts = ordered_context(contexts)[None, :]
            elif contexts.ndim != 2 or contexts.shape[1] != len(CONTEXT_FEATURE_NAMES):
                raise ValueError("context must have shape (40,) or (events, 40)")
        if not np.all(np.isfinite(contexts)):
            raise ValueError("context must be finite")
        if len(contexts) == 1 and len(texture) != 1:
            contexts = np.repeat(contexts, len(texture), axis=0)
        if len(contexts) != len(texture):
            raise ValueError("texture and context batch lengths differ")

        standardized_context = (contexts - self._mean) / self._scale
        design = np.concatenate(
            [np.ones((len(contexts), 1), dtype=np.float64), standardized_context], axis=1
        )
        predicted_texture = design @ self._beta
        residual = (texture - predicted_texture) / self._residual_scale
        scores = np.zeros((len(texture), len(STYLE_SCORE_NAMES)), dtype=np.float64)
        for column, group in enumerate(STYLE_SCORE_NAMES):
            scores[:, column] = residual[:, self._groups[group]] @ self._directions[group]
        if not np.all(np.isfinite(scores)):
            raise ValueError("style transform produced non-finite scores")
        return scores[0] if squeeze else scores

    def score_completed_event(
        self,
        raw_dxdy: Sequence[Sequence[float]] | np.ndarray,
        context: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        mask: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Score one completed human B->C raw-count stream."""

        movement = np.asarray(raw_dxdy, dtype=np.float64)
        if (
            movement.ndim != 2
            or movement.shape[1] != 2
            or not np.all(np.isfinite(movement))
        ):
            raise ValueError("raw_dxdy must have shape (ticks, 2) with finite values")
        valid = (
            np.ones(len(movement), dtype=np.float64)
            if mask is None
            else np.asarray(mask, dtype=np.float64)
        )
        if valid.shape != (len(movement),) or not np.all(np.isfinite(valid)):
            raise ValueError(f"mask must have shape ({len(movement)},)")
        descriptors = texture_features(movement[None, :, :], valid[None, :])[0]
        return self.score_texture(descriptors, context)

    def observe_completed_human(
        self,
        state: CausalStyleState,
        run_id: Hashable,
        raw_dxdy: Sequence[Sequence[float]] | np.ndarray,
        context: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        mask: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Score and append one completed human event after it has finished."""

        if not isinstance(state, CausalStyleState):
            raise TypeError("state must be a CausalStyleState")
        scores = self.score_completed_event(raw_dxdy, context, mask=mask)
        state.observe_human(run_id, scores)
        return scores


__all__ = [
    "CAUSAL_CONTEXT_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "FrozenStyleScorer",
    "HUMAN_SHAPE_SMOOTHING",
    "PLANNED_TRAJECTORY_NAMES",
    "STYLE_SCORE_NAMES",
    "TARGET_ROLE_LABELS",
    "TASK_LABELS",
    "causal_context_features",
    "completed_human_context",
    "default_style_scorer_path",
    "ordered_context",
    "planned_trajectory_features",
    "renderer_context",
]
