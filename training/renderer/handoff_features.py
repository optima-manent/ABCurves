"""Frozen genuinely-continuous online-handoff feature semantics.

The observer has no advance knowledge of ``begin``.  From physical-session
reset it updates the retained H80 once for every raw report.  Centered w5 core
features are emitted with a four-report delay; the first four core vectors are
zero placeholders.  The last four unavailable core vectors and the exact
final canonical regime are supplied once to a small begin adapter.  No GRU
step and no raw-prefix replay occurs at begin.
"""

from __future__ import annotations

import numpy as np


SCHEMA = "abcurves.online_handoff_features.v1"
EPS = 1.0e-9
RECENT = 60
DELAY = 4
HF_PROXY_SCALE = np.float32(0.69508773)

ONLINE_SUMMARY_NAMES = (
    "prefix_to_date_active_rate",
    "prefix_to_date_log1p_active_mean_magnitude",
    "prefix_to_date_log1p_running_max_magnitude",
    "prefix_to_date_sign_flip_rate",
    "prefix_to_date_scaled_log1p_magnitude_delta_energy",
)

CONTRACT = {
    "schema": SCHEMA,
    "raw_magnitude": "max(abs(dx),abs(dy))",
    "core": (
        "whole physical session, centered triangular-path w5 core15 delayed four; "
        "observer ticks 0..3 use zero core placeholders and tick t>=4 uses core[t-4]"
    ),
    "summary_order": list(ONLINE_SUMMARY_NAMES),
    "summary_snapshot": "update with current raw tick, then feed current five-vector",
    "hf_proxy_scale": float(HF_PROXY_SCALE),
    "begin_input": "online_h80[80] + exact unconsumed old-target core15[4] + exact final old regime[5]",
    "begin": "rank16 residual adapter; no GRU step and no raw-prefix replay",
}


def _box5(values: np.ndarray) -> np.ndarray:
    padded = np.concatenate(
        [values[:, :1], values[:, :1], values, values[:, -1:], values[:, -1:]],
        axis=1,
    )
    output = np.zeros_like(values, dtype=np.float64)
    for offset in range(5):
        output += padded[:, offset : offset + values.shape[1]] * 0.2
    return output


def smooth_w5_batch(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32).astype(np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[1] < 1:
        raise ValueError("raw must be [batch,ticks,2]")
    path = np.cumsum(values, axis=1, dtype=np.float64)
    twice = _box5(_box5(path))
    previous = np.concatenate([np.zeros_like(twice[:, :1]), twice[:, :-1]], axis=1)
    return (twice - previous).astype(np.float32).astype(np.float64)


def core15_batch(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32).astype(np.float64)
    smooth = smooth_w5_batch(values)
    batch, ticks, _ = values.shape
    speed = np.sqrt(np.sum(smooth * smooth, axis=2))
    accel = np.zeros_like(speed)
    accel[:, 1:] = speed[:, 1:] - speed[:, :-1]
    curvature = np.zeros_like(speed)
    if ticks > 1:
        denominator = speed[:, :-1] * speed[:, 1:]
        valid = denominator > EPS
        cosine = np.zeros_like(denominator)
        numerator = np.sum(smooth[:, :-1] * smooth[:, 1:], axis=2)
        cosine[valid] = np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)
        curvature[:, 1:] = np.where(valid, 1.0 - cosine, 0.0)
    tangent = np.empty_like(smooth)
    last = np.zeros((batch, 2), np.float64)
    last[:, 0] = 1.0
    for tick in range(ticks):
        moving = speed[:, tick] > EPS
        last[moving] = smooth[moving, tick]
        norm = np.sqrt(np.sum(last * last, axis=1))
        norm[norm <= EPS] = 1.0
        tangent[:, tick] = last / norm[:, None]
    normal = np.stack([-tangent[:, :, 1], tangent[:, :, 0]], axis=2)
    desired = np.cumsum(smooth, axis=1, dtype=np.float64)
    desired -= np.cumsum(values, axis=1, dtype=np.float64) - values
    previous_emit = np.zeros_like(values)
    previous_emit[:, 1:] = values[:, :-1]
    active = np.any(values != 0.0, axis=2)
    last_nonzero = np.zeros_like(values)
    current_last = np.zeros((batch, 2), np.float64)
    run_active = np.zeros((batch, ticks), np.float64)
    run_length = np.zeros((batch, ticks), np.float64)
    state_active = np.zeros(batch, bool)
    state_length = np.zeros(batch, np.int64)
    for tick in range(ticks):
        last_nonzero[:, tick] = current_last
        current_last[active[:, tick]] = values[active[:, tick], tick]
        run_active[:, tick] = state_active
        run_length[:, tick] = state_length
        reset = (state_length == 0) | (active[:, tick] != state_active)
        state_length = np.where(reset, 1, state_length + 1)
        state_active = active[:, tick]
    inactive = (~active).astype(np.float64)
    cs = np.concatenate(
        [np.zeros((batch, 1), np.float64), np.cumsum(inactive, axis=1)], axis=1
    )
    recent_zero = np.empty((batch, ticks), np.float64)
    recent_zero[:, 0] = 1.0
    for tick in range(1, ticks):
        low = max(0, tick - RECENT)
        recent_zero[:, tick] = (cs[:, tick] - cs[:, low]) / float(tick - low)
    dot = lambda a, b: np.sum(a * b, axis=2)
    return np.stack(
        [
            speed / 4.0,
            np.clip(accel, -3.0, 3.0),
            curvature,
            tangent[:, :, 0],
            tangent[:, :, 1],
            np.clip(dot(desired, tangent), -4.0, 4.0),
            np.clip(dot(desired, normal), -4.0, 4.0),
            np.clip(dot(previous_emit, tangent) / 4.0, -4.0, 4.0),
            np.clip(dot(previous_emit, normal) / 4.0, -4.0, 4.0),
            run_active,
            np.minimum(run_length, 64.0) / 64.0,
            recent_zero,
            np.log1p(speed),
            np.clip(dot(last_nonzero, tangent) / 4.0, -4.0, 4.0),
            np.clip(dot(last_nonzero, normal) / 4.0, -4.0, 4.0),
        ],
        axis=2,
    ).astype(np.float32)


def online_summary_batch(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("raw must be [batch,ticks,2]")
    batch, ticks, _ = values.shape
    magnitude = np.maximum(np.abs(values[:, :, 0]), np.abs(values[:, :, 1]))
    active = magnitude > 0.0
    active_count = np.cumsum(active, axis=1, dtype=np.int32)
    active_sum = np.cumsum(np.where(active, magnitude, 0.0), axis=1, dtype=np.float64)
    denominator = np.arange(1, ticks + 1, dtype=np.float64)[None]
    active_rate = active_count / denominator
    active_mean = np.divide(
        active_sum,
        active_count,
        out=np.zeros_like(active_sum),
        where=active_count > 0,
    )
    upper_tail = np.maximum.accumulate(magnitude, axis=1)

    flips = np.zeros((batch, ticks), np.float64)
    comparisons = np.zeros((batch, ticks), np.float64)
    previous = np.zeros((batch, 2), np.int8)
    flip_count = np.zeros(batch, np.int32)
    comparison_count = np.zeros(batch, np.int32)
    for tick in range(ticks):
        signs = np.sign(values[:, tick]).astype(np.int8)
        for axis in range(2):
            current_nonzero = signs[:, axis] != 0
            comparable = current_nonzero & (previous[:, axis] != 0)
            flip_count += comparable & (signs[:, axis] != previous[:, axis])
            comparison_count += comparable
            previous[current_nonzero, axis] = signs[current_nonzero, axis]
        flips[:, tick] = flip_count
        comparisons[:, tick] = comparison_count
    sign_flip = np.divide(
        flips,
        comparisons,
        out=np.zeros_like(flips),
        where=comparisons > 0,
    )
    delta = np.zeros_like(magnitude, dtype=np.float64)
    delta[:, 1:] = magnitude[:, 1:] - magnitude[:, :-1]
    delta_energy = np.cumsum(delta * delta, axis=1, dtype=np.float64)
    return np.stack(
        [
            active_rate,
            np.log1p(active_mean),
            np.log1p(upper_tail),
            sign_flip,
            float(HF_PROXY_SCALE) * np.log1p(delta_energy),
        ],
        axis=2,
    ).astype(np.float32)


def continuous_observer_features(raw: np.ndarray) -> np.ndarray:
    """Return the deployable every-tick observer inputs for a raw session."""

    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (256, 2):
        raise ValueError("raw must be [batch,256,2]")
    core = core15_batch(values)
    output = np.zeros((len(values), 256, 20), np.float32)
    output[:, 4:, :15] = core[:, :-4]
    output[:, :, 15:] = online_summary_batch(values)
    return output


def begin_input(
    online_hidden: np.ndarray, target_core_tail4: np.ndarray, final_regime: np.ndarray
) -> np.ndarray:
    h = np.asarray(online_hidden, np.float32)
    tail = np.asarray(target_core_tail4, np.float32)
    regime = np.asarray(final_regime, np.float32)
    if h.ndim != 2 or h.shape[1] != 80 or tail.shape != (len(h), 4, 15):
        raise ValueError("hidden/tail shape differs")
    if regime.shape != (len(h), 5):
        raise ValueError("regime shape differs")
    return np.concatenate([h, tail.reshape(len(h), 60), regime], axis=1).astype(np.float32)


__all__ = [
    "CONTRACT",
    "DELAY",
    "HF_PROXY_SCALE",
    "SCHEMA",
    "begin_input",
    "core15_batch",
    "continuous_observer_features",
    "online_summary_batch",
    "smooth_w5_batch",
]
