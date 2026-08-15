"""Causal summary features of an A->B prefix (what the planner conditions on).

At the moment of the cut (B) the planner sees two things: the raw 1 kHz prefix
stream itself (fed to a small temporal conv-net) and a vector of ~60 summary
numbers describing the situation at B -- how fast the hand is moving, where the
target is, how straight the approach was, whether the hand is decelerating,
recent micro-texture statistics, and so on.  Everything here is **causal**: it
is computed only from the A->B counts and the target position/radius known at
B.  Nothing ever peeks at the future.

The public entry point is :func:`summary_features`, which returns the exact
feature dict the shipped planner checkpoint was trained with (the checkpoint
stores the feature-name list and normalizers; the planner assembles its input
vector from this dict by name).

The three layers below mirror the research code that produced the checkpoint,
verbatim, so a prefix produces bit-identical features:

1. :func:`causal_context_arrays` -- windowed kinematics recomputed from the
   prefix (recent speed slope, zero rate, sign flips, heading vs target, ...).
2. :func:`primitive_features` -- the situation-at-B scalars (target geometry,
   speed/acceleration state, deceleration flags, prefix path shape).
3. :func:`prefix_shape_features` -- curvature/approach-direction descriptors of
   the last few tens of milliseconds before B.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------------------
# Target-frame helpers
# ---------------------------------------------------------------------------
def target_frame_basis(target_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return target-axis and tangent unit vectors for coordinates at B."""

    target = np.asarray(target_rel, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(target))
    if norm <= EPS:
        toward = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        toward = target / norm
    tangent = np.asarray([-toward[1], toward[0]], dtype=np.float64)
    return toward, tangent


def transform_to_target_frame(values: np.ndarray, target_rel: np.ndarray) -> np.ndarray:
    """Project ``[..., 2]`` points or deltas onto target-aligned x/y axes."""

    arr = np.asarray(values, dtype=np.float64)
    toward, tangent = target_frame_basis(target_rel)
    return np.stack([arr @ toward, arr @ tangent], axis=-1)


def direction_angle_error_deg(a: np.ndarray, b: np.ndarray, *, default: float = 180.0) -> float:
    """Unsigned angle error in degrees between two movement vectors."""

    va = np.asarray(a, dtype=np.float64).reshape(2)
    vb = np.asarray(b, dtype=np.float64).reshape(2)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= EPS and nb <= EPS:
        return 0.0
    if na <= EPS or nb <= EPS:
        return float(default)
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# Small numeric helpers (verbatim from the research code)
# ---------------------------------------------------------------------------
def _valid_prefix(arrays: dict[str, np.ndarray], idx: int) -> np.ndarray:
    prefix = arrays.get("prefix_raw_dxdy")
    if prefix is None:
        return np.zeros((0, 2), dtype=np.float64)
    mask = arrays.get("prefix_mask", np.ones(prefix.shape[:2], dtype=np.float32))[idx].astype(bool)
    valid = np.asarray(prefix[idx], dtype=np.float64)[mask]
    valid[~np.isfinite(valid)] = 0.0
    return valid


def _target_rel(arrays: dict[str, np.ndarray], idx: int) -> np.ndarray:
    n = len(arrays["future_mask"])
    return np.asarray(
        [
            arrays.get("target_rel_x_at_B", np.zeros(n, dtype=np.float32))[idx],
            arrays.get("target_rel_y_at_B", np.zeros(n, dtype=np.float32))[idx],
        ],
        dtype=np.float64,
    )


def _last_velocity(arrays: dict[str, np.ndarray], idx: int, prefix: np.ndarray) -> np.ndarray:
    n = len(arrays["future_mask"])
    last = np.asarray(
        [
            arrays.get("last_velocity_x", np.zeros(n, dtype=np.float32))[idx],
            arrays.get("last_velocity_y", np.zeros(n, dtype=np.float32))[idx],
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(last)) <= EPS and len(prefix):
        return prefix[-1].astype(np.float64)
    return last


def _last_accel(arrays: dict[str, np.ndarray], idx: int, prefix: np.ndarray) -> np.ndarray:
    n = len(arrays["future_mask"])
    last = np.asarray(
        [
            arrays.get("last_accel_x", np.zeros(n, dtype=np.float32))[idx],
            arrays.get("last_accel_y", np.zeros(n, dtype=np.float32))[idx],
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(last)) <= EPS and len(prefix) >= 2:
        return prefix[-1] - prefix[-2]
    return last


def _last_jerk(prefix: np.ndarray) -> float:
    if len(prefix) < 4:
        return 0.0
    accel = np.diff(prefix[-4:], axis=0)
    jerk = accel[-1] - accel[-2]
    return float(np.linalg.norm(jerk))


def _prefix_path_at_b(prefix: np.ndarray) -> np.ndarray:
    if len(prefix) == 0:
        return np.zeros((1, 2), dtype=np.float64)
    path = np.cumsum(prefix, axis=0)
    return path - path[-1][None, :]


def _scalar(arrays: dict[str, np.ndarray], key: str, idx: int, default: float) -> float:
    value = arrays.get(key)
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.shape[0] <= idx:
        return float(default)
    try:
        out = float(arr[idx])
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _linear_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) <= 1:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)
    x -= np.mean(x)
    denom = float(np.sum(x * x))
    if denom <= EPS:
        return 0.0
    return float(np.sum(x * (arr - np.mean(arr))) / denom)


def _sign_flip_rate(dxdy: np.ndarray) -> float:
    arr = np.asarray(dxdy, dtype=np.float64)
    if len(arr) <= 1:
        return 0.0
    signs = np.sign(arr[np.linalg.norm(arr, axis=1) > EPS])
    if len(signs) <= 1:
        return 0.0
    flips = np.any(signs[1:] != signs[:-1], axis=1)
    return float(np.mean(flips))


def _direction_change_rate(dxdy: np.ndarray) -> float:
    arr = np.asarray(dxdy, dtype=np.float64)
    if len(arr) <= 2:
        return 0.0
    valid = arr[np.linalg.norm(arr, axis=1) > EPS]
    if len(valid) <= 2:
        return 0.0
    cos = np.sum(valid[1:] * valid[:-1], axis=1) / np.maximum(
        np.linalg.norm(valid[1:], axis=1) * np.linalg.norm(valid[:-1], axis=1), EPS
    )
    return float(np.mean(np.arccos(np.clip(cos, -1.0, 1.0)) > np.deg2rad(35.0)))


def _resample_bins(values: np.ndarray, bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) == 0:
        return np.zeros(int(bins), dtype=np.float64)
    edges = np.linspace(0, len(arr), int(bins) + 1)
    out = np.zeros(int(bins), dtype=np.float64)
    for i in range(int(bins)):
        lo = int(np.floor(edges[i]))
        hi = int(np.floor(edges[i + 1]))
        if hi <= lo:
            hi = min(lo + 1, len(arr))
        out[i] = float(np.mean(arr[lo:hi])) if lo < len(arr) else float(arr[-1])
    return out


def _tail_run_length(mask: np.ndarray, value: bool) -> int:
    count = 0
    for item in mask[::-1]:
        if bool(item) != bool(value):
            break
        count += 1
    return count


def _recent_sign_flip_rate(dxdy: np.ndarray) -> float:
    flips = 0
    denom = 0
    for axis in range(2):
        signs = np.sign(dxdy[:, axis][np.abs(dxdy[:, axis]) > 0])
        if len(signs) > 1:
            flips += int(np.sum(signs[1:] != signs[:-1]))
            denom += len(signs) - 1
    return float(flips / denom) if denom else 0.0


def _recent_direction_change_rate(dxdy: np.ndarray) -> float:
    if len(dxdy) <= 2:
        return 0.0
    mag = np.linalg.norm(dxdy, axis=1)
    valid = mag > 1e-6
    unit = np.zeros_like(dxdy, dtype=np.float64)
    unit[valid] = dxdy[valid] / mag[valid, None]
    dots = np.sum(unit[1:] * unit[:-1], axis=1)
    active = valid[1:] & valid[:-1]
    if not np.any(active):
        return 0.0
    return float(np.mean(dots[active] < 0.7071))


def _slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)
    x = x - np.mean(x)
    denom = float(np.sum(x * x))
    if denom <= 1e-9:
        return 0.0
    return float(np.sum(x * (arr - np.mean(arr))) / denom)


# ---------------------------------------------------------------------------
# Layer 1: windowed causal kinematics recomputed from the prefix
# ---------------------------------------------------------------------------
def causal_context_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fill the recomputable causal scalars for a batch of prefix rows."""

    prefix = arrays["prefix_raw_dxdy"].astype(np.float32)
    pmask = arrays["prefix_mask"].astype(bool)
    n = len(prefix)
    recomputed = {
        "distance_normalized_by_radius": np.zeros(n, dtype=np.float32),
        "inside_target_at_B": np.zeros(n, dtype=np.float32),
        "prefix_speed_mean": np.zeros(n, dtype=np.float32),
        "prefix_speed_last": np.zeros(n, dtype=np.float32),
        "prefix_speed_std": np.zeros(n, dtype=np.float32),
        "prefix_speed_max": np.zeros(n, dtype=np.float32),
        "prefix_accel_mean": np.zeros(n, dtype=np.float32),
        "prefix_accel_last": np.zeros(n, dtype=np.float32),
        "last_velocity_x": np.zeros(n, dtype=np.float32),
        "last_velocity_y": np.zeros(n, dtype=np.float32),
        "last_accel_x": np.zeros(n, dtype=np.float32),
        "last_accel_y": np.zeros(n, dtype=np.float32),
        "prefix_cumulative_dx": np.zeros(n, dtype=np.float32),
        "prefix_cumulative_dy": np.zeros(n, dtype=np.float32),
        "prefix_net_displacement": np.zeros(n, dtype=np.float32),
        "prefix_path_length": np.zeros(n, dtype=np.float32),
        "prefix_straightness": np.zeros(n, dtype=np.float32),
        "heading_to_target_cos": np.zeros(n, dtype=np.float32),
        "heading_to_target_sin": np.zeros(n, dtype=np.float32),
        "velocity_toward_target": np.zeros(n, dtype=np.float32),
        "velocity_tangential_to_target": np.zeros(n, dtype=np.float32),
        "recent_zero_rate": np.zeros(n, dtype=np.float32),
        "recent_nonzero_run_length": np.zeros(n, dtype=np.float32),
        "recent_zero_run_length": np.zeros(n, dtype=np.float32),
        "recent_sign_flip_rate": np.zeros(n, dtype=np.float32),
        "recent_count_magnitude_mean": np.zeros(n, dtype=np.float32),
        "recent_count_magnitude_max": np.zeros(n, dtype=np.float32),
        "recent_direction_change_rate": np.zeros(n, dtype=np.float32),
        "recent_curvature_proxy": np.zeros(n, dtype=np.float32),
        "recent_speed_slope": np.zeros(n, dtype=np.float32),
        "recent_accel_slope": np.zeros(n, dtype=np.float32),
        "prefix_duration_ms": np.zeros(n, dtype=np.float32),
        "B_index_or_prefix_len_ms": np.zeros(n, dtype=np.float32),
    }
    target_rel = np.column_stack(
        [
            arrays.get("target_rel_x_at_B", np.zeros(n, dtype=np.float32)).astype(np.float32),
            arrays.get("target_rel_y_at_B", np.zeros(n, dtype=np.float32)).astype(np.float32),
        ]
    )
    radius = arrays.get("target_radius", np.ones(n, dtype=np.float32)).astype(np.float32)
    for idx in range(n):
        valid = prefix[idx][pmask[idx]]
        if len(valid) == 0:
            continue
        speed = np.linalg.norm(valid, axis=1)
        accel_vec = np.diff(valid, axis=0, prepend=valid[:1])
        accel_mag = np.linalg.norm(accel_vec, axis=1)
        last = valid[-1].astype(np.float64)
        last_accel = accel_vec[-1].astype(np.float64)
        cumulative = np.sum(valid, axis=0).astype(np.float64)
        path_len = float(np.sum(speed))
        net = float(np.linalg.norm(cumulative))
        target = target_rel[idx].astype(np.float64)
        target_norm = float(np.linalg.norm(target))
        target_unit = target / target_norm if target_norm > 1e-9 else np.asarray([1.0, 0.0], dtype=np.float64)
        last_norm = float(np.linalg.norm(last))
        last_unit = last / last_norm if last_norm > 1e-9 else np.zeros(2, dtype=np.float64)
        recent = valid[-min(40, len(valid)) :]
        recent_mag = np.linalg.norm(recent, axis=1)
        recent_nonzero = recent_mag > 0
        recent_accel = np.diff(recent, axis=0, prepend=recent[:1])
        recent_accel_mag = np.linalg.norm(recent_accel, axis=1)
        prefix_len = int(np.sum(pmask[idx]))
        recomputed["distance_normalized_by_radius"][idx] = float(target_norm / max(float(radius[idx]), 1e-6))
        recomputed["inside_target_at_B"][idx] = float(target_norm <= max(float(radius[idx]), 0.0))
        recomputed["prefix_speed_mean"][idx] = float(np.mean(speed))
        recomputed["prefix_speed_last"][idx] = float(speed[-1])
        recomputed["prefix_speed_std"][idx] = float(np.std(speed))
        recomputed["prefix_speed_max"][idx] = float(np.max(speed))
        recomputed["prefix_accel_mean"][idx] = float(np.mean(accel_mag))
        recomputed["prefix_accel_last"][idx] = float(accel_mag[-1])
        recomputed["last_velocity_x"][idx] = float(last[0])
        recomputed["last_velocity_y"][idx] = float(last[1])
        recomputed["last_accel_x"][idx] = float(last_accel[0])
        recomputed["last_accel_y"][idx] = float(last_accel[1])
        recomputed["prefix_cumulative_dx"][idx] = float(cumulative[0])
        recomputed["prefix_cumulative_dy"][idx] = float(cumulative[1])
        recomputed["prefix_net_displacement"][idx] = net
        recomputed["prefix_path_length"][idx] = path_len
        recomputed["prefix_straightness"][idx] = float(net / max(path_len, 1e-6))
        recomputed["heading_to_target_cos"][idx] = float(np.dot(last_unit, target_unit))
        recomputed["heading_to_target_sin"][idx] = float(
            last_unit[0] * target_unit[1] - last_unit[1] * target_unit[0]
        )
        recomputed["velocity_toward_target"][idx] = float(np.dot(last, target_unit))
        recomputed["velocity_tangential_to_target"][idx] = float(
            last[0] * target_unit[1] - last[1] * target_unit[0]
        )
        recomputed["recent_zero_rate"][idx] = float(np.mean(~recent_nonzero))
        recomputed["recent_nonzero_run_length"][idx] = float(_tail_run_length(recent_nonzero, True))
        recomputed["recent_zero_run_length"][idx] = float(_tail_run_length(recent_nonzero, False))
        recomputed["recent_sign_flip_rate"][idx] = float(_recent_sign_flip_rate(recent))
        recomputed["recent_count_magnitude_mean"][idx] = float(np.mean(recent_mag))
        recomputed["recent_count_magnitude_max"][idx] = float(np.max(recent_mag))
        recomputed["recent_direction_change_rate"][idx] = float(_recent_direction_change_rate(recent))
        recomputed["recent_curvature_proxy"][idx] = float(
            np.mean(recent_accel_mag / np.maximum(recent_mag, 1.0))
        )
        recomputed["recent_speed_slope"][idx] = float(_slope(recent_mag))
        recomputed["recent_accel_slope"][idx] = float(_slope(recent_accel_mag))
        recomputed["prefix_duration_ms"][idx] = float(prefix_len)
        recomputed["B_index_or_prefix_len_ms"][idx] = float(
            arrays.get("b_index", np.full(n, prefix_len))[idx] if "b_index" in arrays else prefix_len
        )
    for value in recomputed.values():
        value[~np.isfinite(value)] = 0.0
    return recomputed


# ---------------------------------------------------------------------------
# Layer 2: the situation-at-B scalar features
# ---------------------------------------------------------------------------
def primitive_features(arrays: dict[str, np.ndarray], idx: int) -> dict[str, float]:
    """The core situation-at-B feature dict for one event row."""

    prefix = _valid_prefix(arrays, idx)
    speeds = np.linalg.norm(prefix, axis=1) if len(prefix) else np.zeros(0, dtype=np.float64)
    target = _target_rel(arrays, idx)
    radius = max(_scalar(arrays, "target_radius", idx, 1.0), EPS)
    target_distance = max(float(np.linalg.norm(target)), EPS)
    toward, tangent = target_frame_basis(target)
    last_v = _last_velocity(arrays, idx, prefix)
    last_a = _last_accel(arrays, idx, prefix)
    speed_at_b = _scalar(arrays, "speed_at_B", idx, float(np.linalg.norm(last_v)))
    accel_at_b = _scalar(
        arrays, "accel_at_B", idx, _scalar(arrays, "prefix_accel_last", idx, float(np.linalg.norm(last_a)))
    )
    prefix_sum = np.sum(prefix, axis=0) if len(prefix) else np.zeros(2, dtype=np.float64)
    prefix_path_length = float(np.sum(speeds)) if len(speeds) else 0.0
    prefix_distance = float(np.linalg.norm(prefix_sum))
    prefix_tf = transform_to_target_frame(prefix_sum.reshape(1, 2), target)[0]
    last_v_tf = transform_to_target_frame(last_v.reshape(1, 2), target)[0]
    last_a_tf = transform_to_target_frame(last_a.reshape(1, 2), target)[0]
    path = _prefix_path_at_b(prefix)
    if len(path):
        dist_to_target = np.linalg.norm(target[None, :] - path, axis=1)
        path_tf = transform_to_target_frame(path, target)
        crossed = float(np.any(dist_to_target <= radius))
        near_rate = float(np.mean(dist_to_target <= max(radius * 2.0, radius + 4.0)))
        min_prefix_dist = float(np.min(dist_to_target))
        overshot = float(np.max(path_tf[:, 0]) > target_distance * 1.03)
    else:
        crossed = 0.0
        near_rate = 0.0
        min_prefix_dist = target_distance
        overshot = 0.0
    mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
    peak_speed = float(np.max(speeds)) if len(speeds) else 0.0
    speed_std = float(np.std(speeds)) if len(speeds) else 0.0
    recent = speeds[-20:] if len(speeds) else np.zeros(0, dtype=np.float64)
    early = speeds[:20] if len(speeds) else np.zeros(0, dtype=np.float64)
    recent_mean = float(np.mean(recent)) if len(recent) else 0.0
    early_mean = float(np.mean(early)) if len(early) else 0.0
    speed_drop_recent = early_mean - recent_mean
    movement_norm = max(float(np.linalg.norm(last_v)), EPS)
    alignment_cos = float(np.dot(last_v, toward) / movement_norm)
    alignment_sin = float(np.dot(last_v, tangent) / movement_norm)
    direction_error = direction_angle_error_deg(last_v, target, default=90.0)
    jerk = _last_jerk(prefix)
    discontinuity = abs(speed_at_b - mean_speed)
    distance_over_radius = _scalar(
        arrays,
        "distance_over_radius_at_B",
        idx,
        _scalar(arrays, "distance_normalized_by_radius", idx, target_distance / radius),
    )
    progress = _scalar(arrays, "progress", idx, 0.0)
    inside = _scalar(arrays, "inside_target_at_B", idx, 0.0)
    velocity_toward = _scalar(arrays, "velocity_toward_target", idx, float(last_v_tf[0]))
    velocity_lateral = _scalar(arrays, "velocity_tangential_to_target", idx, float(last_v_tf[1]))
    recent_speed_slope = _scalar(arrays, "recent_speed_slope", idx, _linear_slope(speeds[-32:]))
    recent_accel_slope = _scalar(arrays, "recent_accel_slope", idx, 0.0)
    recent_zero_rate = _scalar(
        arrays, "recent_zero_rate", idx, float(np.mean(recent <= EPS)) if len(recent) else 1.0
    )
    recent_sign_flip_rate = _scalar(arrays, "recent_sign_flip_rate", idx, _sign_flip_rate(prefix[-32:]))
    recent_direction_change = _scalar(
        arrays, "recent_direction_change_rate", idx, _direction_change_rate(prefix[-32:])
    )
    prefix_duration = _scalar(arrays, "prefix_duration_ms", idx, float(len(prefix)))
    active_motion = float(speed_at_b > 0.5 or recent_mean > 0.5)
    stabilization_like = float(distance_over_radius <= 1.5 and speed_at_b < max(1.0, mean_speed * 0.5))
    return {
        "prefix_duration_ms": float(prefix_duration),
        "prefix_dx": float(prefix_sum[0]),
        "prefix_dy": float(prefix_sum[1]),
        "prefix_distance": prefix_distance,
        "prefix_path_length": prefix_path_length,
        "prefix_straightness": float(prefix_distance / max(prefix_path_length, EPS)),
        "prefix_target_axis_progress": float(prefix_tf[0]),
        "prefix_lateral_error": float(prefix_tf[1] / radius),
        "target_rel_x": float(target[0]),
        "target_rel_y": float(target[1]),
        "target_distance": float(target_distance),
        "target_radius": float(radius),
        "distance_over_radius": float(distance_over_radius),
        "target_unit_x": float(toward[0]),
        "target_unit_y": float(toward[1]),
        "progress_context": float(progress),
        "inside_target_at_B": float(inside),
        "near_target_flag": float(distance_over_radius <= 2.0 or inside > 0.5),
        "prefix_crossing_state": crossed,
        "prefix_overshoot_state": overshot,
        "prefix_min_target_distance_over_radius": float(min_prefix_dist / radius),
        "prefix_near_target_rate": near_rate,
        "last_velocity_x": float(last_v[0]),
        "last_velocity_y": float(last_v[1]),
        "movement_dir_x": float(last_v[0] / movement_norm),
        "movement_dir_y": float(last_v[1] / movement_norm),
        "speed_at_B": float(speed_at_b),
        "mean_prefix_speed": mean_speed,
        "peak_prefix_speed": peak_speed,
        "prefix_speed_std": speed_std,
        "relative_speed_distance": float(speed_at_b / target_distance),
        "relative_speed_radius": float(speed_at_b / radius),
        "velocity_toward_target": float(velocity_toward),
        "velocity_tangential_to_target": float(velocity_lateral),
        "direction_alignment_cos": alignment_cos,
        "direction_alignment_sin": alignment_sin,
        "direction_error_deg": float(direction_error),
        "accel_at_B": float(accel_at_b),
        "accel_toward_target": float(last_a_tf[0]),
        "accel_lateral": float(last_a_tf[1]),
        "recent_speed_slope": float(recent_speed_slope),
        "recent_accel_slope": float(recent_accel_slope),
        "deceleration_indicator": float(recent_speed_slope < -0.03 or accel_at_b < -0.03),
        "speed_drop_recent": float(speed_drop_recent),
        "jerk_at_B": float(jerk),
        "discontinuity_speed_jump": float(discontinuity),
        "recent_zero_rate": float(recent_zero_rate),
        "recent_sign_flip_rate": float(recent_sign_flip_rate),
        "recent_direction_change_rate": float(recent_direction_change),
        "active_motion_flag": active_motion,
        "stabilization_like_flag": stabilization_like,
    }


# ---------------------------------------------------------------------------
# Layer 3: prefix shape features (curvature / approach direction near B)
# ---------------------------------------------------------------------------
def prefix_shape_features(arrays: dict[str, np.ndarray], idx: int) -> dict[str, float]:
    prefix = _valid_prefix(arrays, idx)
    target = _target_rel(arrays, idx)
    toward, tangent = target_frame_basis(target)
    recent = prefix[-16:] if len(prefix) else np.zeros((0, 2), dtype=np.float64)
    # Curvature near B from cross products of consecutive recent velocity vectors.
    curv_signed = 0.0
    curv_abs = 0.0
    turn_angles: list[float] = []
    if len(recent) >= 2:
        v0 = recent[:-1]
        v1 = recent[1:]
        n0 = np.linalg.norm(v0, axis=1)
        n1 = np.linalg.norm(v1, axis=1)
        active = (n0 > 1e-6) & (n1 > 1e-6)
        if np.any(active):
            cross = v0[active, 0] * v1[active, 1] - v0[active, 1] * v1[active, 0]
            denom = np.maximum(n0[active] * n1[active], 1e-9)
            sin_t = cross / denom
            curv_signed = float(np.mean(sin_t))
            curv_abs = float(np.mean(np.abs(sin_t)))
            dot = np.sum(v0[active] * v1[active], axis=1) / denom
            turn_angles = list(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
    dir_change_mean = float(np.mean(turn_angles)) if turn_angles else 0.0
    dir_change_slope = _linear_slope(np.asarray(turn_angles, dtype=np.float64)) if len(turn_angles) >= 2 else 0.0
    # Approach direction (mean of last ~10 deltas) vs target direction.
    approach = np.sum(prefix[-10:], axis=0) if len(prefix) else np.zeros(2, dtype=np.float64)
    a_norm = float(np.linalg.norm(approach))
    if a_norm > 1e-6:
        approach_cos = float(np.dot(approach, toward) / a_norm)
        approach_sin = float(np.dot(approach, tangent) / a_norm)
    else:
        approach_cos = 0.0
        approach_sin = 0.0
    # Speed profile bins over the whole prefix (normalized by mean prefix speed).
    speeds = np.linalg.norm(prefix, axis=1) if len(prefix) else np.zeros(0, dtype=np.float64)
    speed_bins = _resample_bins(speeds, 4)
    mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
    speed_bins = speed_bins / max(mean_speed, 1e-6)
    # Recent lateral drift in target frame (tangential vs along-target energy).
    rec_tf = transform_to_target_frame(recent, target) if len(recent) else np.zeros((0, 2), dtype=np.float64)
    lateral_drift = 0.0
    if len(rec_tf):
        along = float(np.sum(np.abs(rec_tf[:, 0])))
        lat = float(np.sum(np.abs(rec_tf[:, 1])))
        lateral_drift = lat / max(along + lat, 1e-6)
    return {
        "prefix_shape_curvature_signed": curv_signed,
        "prefix_shape_curvature_abs": curv_abs,
        "prefix_shape_dir_change_mean_deg": dir_change_mean,
        "prefix_shape_dir_change_slope": dir_change_slope,
        "prefix_shape_approach_cos": approach_cos,
        "prefix_shape_approach_sin": approach_sin,
        "prefix_shape_speed_bin_0": float(speed_bins[0]),
        "prefix_shape_speed_bin_1": float(speed_bins[1]),
        "prefix_shape_speed_bin_2": float(speed_bins[2]),
        "prefix_shape_speed_bin_3": float(speed_bins[3]),
        "prefix_shape_recent_lateral_drift": lateral_drift,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def summary_features_for_row(
    row: dict[str, np.ndarray], prefix: np.ndarray, *, horizon: int = 1000
) -> dict[str, float]:
    """Build the full summary dict for a one-event arrays row (research shape)."""

    required = {
        "target_rel_x_at_B",
        "target_rel_y_at_B",
        "target_radius",
        "progress",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            "summary rows require target geometry and progress; missing "
            + ", ".join(missing)
        )

    live = dict(row)
    live["future_mask"] = np.ones((1, int(horizon)), dtype=np.float32)
    p = np.asarray(prefix, dtype=np.float32)
    if len(p):
        live["last_velocity_x"] = np.asarray([float(p[-1, 0])], dtype=np.float32)
        live["last_velocity_y"] = np.asarray([float(p[-1, 1])], dtype=np.float32)
    feat = primitive_features(live, 0)
    feat.update(prefix_shape_features(live, 0))
    return {str(k): float(v) for k, v in feat.items()}


def summary_features(
    prefix: np.ndarray,
    target_rel_at_B: tuple[float, float],
    target_radius: float,
    progress: float,
    *,
    horizon: int = 1000,
    b_index_ms: float | None = None,
) -> dict[str, float]:
    """The live summary feature dict for one A->B prefix.

    ``prefix`` is the raw 1 kHz A->B count stream ``[P, 2]``.
    ``target_rel_at_B`` is the target center relative to the cursor at B, in
    counts. ``progress`` is the fraction of the initial A-to-target distance
    already covered at B. It is required because zero is a learned value, not
    a neutral substitute for missing context.
    """

    p = np.asarray(prefix, dtype=np.float32)
    if p.ndim != 2 or (p.size and p.shape[1] != 2):
        raise ValueError("prefix must have shape (P, 2)")
    p = p[np.isfinite(p).all(axis=1)] if len(p) else p.reshape(0, 2)

    tx, ty = float(target_rel_at_B[0]), float(target_rel_at_B[1])
    radius = float(target_radius)
    distance = float(np.hypot(tx, ty))
    speed_at_b = float(np.linalg.norm(p[-1])) if len(p) else 0.0
    accel_at_b = float(np.linalg.norm(p[-1] - p[-2])) if len(p) >= 2 else 0.0

    def f32(*values: float) -> np.ndarray:
        return np.asarray(list(values), dtype=np.float32)

    row: dict[str, np.ndarray] = {
        "prefix_raw_dxdy": p[None, :, :],
        "prefix_mask": np.ones((1, len(p)), dtype=np.float32),
        "future_mask": np.ones((1, 1), dtype=np.float32),
        "target_rel_x_at_B": f32(tx),
        "target_rel_y_at_B": f32(ty),
        "target_radius": f32(radius),
        "target_distance_at_B": f32(distance),
        "progress": f32(float(progress)),
        "speed_at_B": f32(speed_at_b),
        "accel_at_B": f32(accel_at_b),
        "distance_over_radius_at_B": f32(distance / max(radius, 1e-6)),
        "b_index": f32(float(b_index_ms) if b_index_ms is not None else float(len(p))),
    }
    row.update({k: v for k, v in causal_context_arrays(row).items()})
    return summary_features_for_row(row, p, horizon=horizon)


def build_feature_table(
    arrays: dict[str, np.ndarray], *, profile_bins: int = 8
) -> dict[str, object]:
    """Batch feature table for training (primitive + prefix-shape features).

    Returns ``{"feature_names": [...], "features": [N, F] float64}`` in the
    exact order the planner checkpoint expects.
    """

    n = len(arrays["future_mask"])
    dicts = []
    for idx in range(n):
        feat = primitive_features(arrays, idx)
        feat.update(prefix_shape_features(arrays, idx))
        dicts.append(feat)
    feature_names = list(dicts[0].keys()) if dicts else []
    features = np.asarray(
        [[float(d[name]) for name in feature_names] for d in dicts], dtype=np.float64
    )
    features[~np.isfinite(features)] = 0.0
    return {"feature_names": feature_names, "features": features}


__all__ = [
    "summary_features",
    "summary_features_for_row",
    "build_feature_table",
    "causal_context_arrays",
    "primitive_features",
    "prefix_shape_features",
    "target_frame_basis",
    "transform_to_target_frame",
    "direction_angle_error_deg",
]
