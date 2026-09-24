"""Historical 20-column matching context; generation-known fields only."""

import numpy as np

from abcurves.features import target_frame_basis

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

def _masked_prefix(
    prefix: np.ndarray,
    mask: np.ndarray,
    index: int,
) -> np.ndarray:
    return np.asarray(prefix[index][np.asarray(mask[index]) > 0.5], dtype=np.float64)

def _linear_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)
    x -= np.mean(x)
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(x * (arr - np.mean(arr))) / denominator)

def _prefix_sign_flip_rate(prefix: np.ndarray) -> float:
    if len(prefix) < 2:
        return 0.0
    flips = 0
    comparisons = 0
    for axis in range(2):
        active = prefix[:, axis][np.abs(prefix[:, axis]) > 0.0]
        signs = np.sign(active)
        if len(signs) >= 2:
            flips += int(np.sum(signs[1:] != signs[:-1]))
            comparisons += len(signs) - 1
    return float(flips / comparisons) if comparisons else 0.0

def _target_distance_at_a(arrays: dict[str, np.ndarray], index: int) -> float:
    if "target_distance_at_A" in arrays:
        value = float(arrays["target_distance_at_A"][index])
        if np.isfinite(value):
            return value
    if "target_rel_x_at_A" in arrays and "target_rel_y_at_A" in arrays:
        return float(
            np.hypot(
                float(arrays["target_rel_x_at_A"][index]),
                float(arrays["target_rel_y_at_A"][index]),
            )
        )
    distance_b = float(
        np.hypot(
            float(arrays["target_rel_x_at_B"][index]),
            float(arrays["target_rel_y_at_B"][index]),
        )
    )
    progress = float(arrays["progress"][index])
    return float(distance_b / max(1.0 - progress, 1e-6))

def causal_context_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return the documented target/prefix-only matching context.

    This function intentionally never indexes a future or outcome field.
    """

    prefix = np.asarray(arrays["prefix_raw_dxdy"])
    prefix_mask = np.asarray(arrays["prefix_mask"])
    n = len(prefix)
    out = np.zeros((n, len(CAUSAL_CONTEXT_NAMES)), dtype=np.float64)
    for index in range(n):
        row = _masked_prefix(prefix, prefix_mask, index)
        target = np.asarray(
            [
                float(arrays["target_rel_x_at_B"][index]),
                float(arrays["target_rel_y_at_B"][index]),
            ],
            dtype=np.float64,
        )
        radius = float(arrays["target_radius"][index])
        distance_b = float(np.linalg.norm(target))
        speed = np.linalg.norm(row, axis=1) if len(row) else np.zeros(0)
        path_length = float(np.sum(speed))
        displacement = np.sum(row, axis=0) if len(row) else np.zeros(2)
        net_distance = float(np.linalg.norm(displacement))
        toward, tangent = target_frame_basis(target)
        approach = np.sum(row[-10:], axis=0) if len(row) else np.zeros(2)
        approach_norm = float(np.linalg.norm(approach))
        approach_cos = float(np.dot(approach, toward) / approach_norm) if approach_norm else 0.0
        approach_sin = float(np.dot(approach, tangent) / approach_norm) if approach_norm else 0.0
        recent = row[-16:]
        if len(recent):
            recent_tf = np.stack([recent @ toward, recent @ tangent], axis=1)
            along_energy = float(np.sum(np.abs(recent_tf[:, 0])))
            lateral_energy = float(np.sum(np.abs(recent_tf[:, 1])))
            lateral_fraction = lateral_energy / max(along_energy + lateral_energy, 1e-9)
        else:
            lateral_fraction = 0.0
        edge_trigger = (
            float(arrays["edge_trigger_progress"][index])
            if "edge_trigger_progress" in arrays
            else float(arrays["progress"][index])
        )
        edge_realized = (
            float(arrays["edge_realized_progress"][index])
            if "edge_realized_progress" in arrays
            else edge_trigger
        )
        out[index] = (
            radius,
            _target_distance_at_a(arrays, index),
            distance_b,
            float(arrays["progress"][index]),
            edge_trigger,
            edge_realized,
            float(len(row)),
            path_length,
            net_distance,
            net_distance / max(path_length, 1e-9),
            float(np.mean(speed)) if len(speed) else 0.0,
            float(np.std(speed)) if len(speed) else 0.0,
            float(np.max(speed)) if len(speed) else 0.0,
            float(np.mean(speed[-16:])) if len(speed) else 0.0,
            _linear_slope(speed[-32:]),
            float(np.mean(speed <= 0.0)) if len(speed) else 1.0,
            _prefix_sign_flip_rate(row),
            approach_cos,
            approach_sin,
            lateral_fraction,
        )
    if not np.all(np.isfinite(out)):
        raise ValueError("causal matching context contains non-finite values")
    return out
