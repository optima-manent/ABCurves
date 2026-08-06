"""Compiled live summary-vector path for the retained raw-prefix Planner.

The authority remains ``ABCurves.abcurves.features.summary_features`` followed
by ``Planner._summary_vector``.  This module is a packed, allocation-light
translation of that one-event path.  It deliberately preserves the public
builder's float32 hand-offs (live row fields and recomputed causal scalars)
rather than silently defining a cleaner, different feature law.

The compiled kernel emits the 62 features in ``CANONICAL_FEATURE_NAMES``
order.  ``CompiledSummaryVector`` binds that order to the names and
normalizers embedded in a loaded Planner checkpoint.  Unknown checkpoint
features are rejected instead of being filled optimistically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit


EPS = 1e-6


CANONICAL_FEATURE_NAMES = (
    "prefix_duration_ms",
    "prefix_dx",
    "prefix_dy",
    "prefix_distance",
    "prefix_path_length",
    "prefix_straightness",
    "prefix_target_axis_progress",
    "prefix_lateral_error",
    "target_rel_x",
    "target_rel_y",
    "target_distance",
    "target_radius",
    "distance_over_radius",
    "target_unit_x",
    "target_unit_y",
    "progress_context",
    "inside_target_at_B",
    "near_target_flag",
    "prefix_crossing_state",
    "prefix_overshoot_state",
    "prefix_min_target_distance_over_radius",
    "prefix_near_target_rate",
    "last_velocity_x",
    "last_velocity_y",
    "movement_dir_x",
    "movement_dir_y",
    "speed_at_B",
    "mean_prefix_speed",
    "peak_prefix_speed",
    "prefix_speed_std",
    "relative_speed_distance",
    "relative_speed_radius",
    "velocity_toward_target",
    "velocity_tangential_to_target",
    "direction_alignment_cos",
    "direction_alignment_sin",
    "direction_error_deg",
    "accel_at_B",
    "accel_toward_target",
    "accel_lateral",
    "recent_speed_slope",
    "recent_accel_slope",
    "deceleration_indicator",
    "speed_drop_recent",
    "jerk_at_B",
    "discontinuity_speed_jump",
    "recent_zero_rate",
    "recent_sign_flip_rate",
    "recent_direction_change_rate",
    "active_motion_flag",
    "stabilization_like_flag",
    "prefix_shape_curvature_signed",
    "prefix_shape_curvature_abs",
    "prefix_shape_dir_change_mean_deg",
    "prefix_shape_dir_change_slope",
    "prefix_shape_approach_cos",
    "prefix_shape_approach_sin",
    "prefix_shape_speed_bin_0",
    "prefix_shape_speed_bin_1",
    "prefix_shape_speed_bin_2",
    "prefix_shape_speed_bin_3",
    "prefix_shape_recent_lateral_drift",
)


@njit(cache=True, nogil=True)
def _norm2_f32(x: np.float32, y: np.float32) -> np.float32:
    return np.float32(np.sqrt(np.float32(x * x + y * y)))


@njit(cache=True, nogil=True)
def _norm2(x: float, y: float) -> float:
    # ``float(np.float32)`` does not force widening inside Numba.  Widen here
    # so primitive/prefix-shape math follows _valid_prefix's float64 array.
    xd = np.float64(x)
    yd = np.float64(y)
    return np.sqrt(xd * xd + yd * yd)


@njit(cache=True, nogil=True)
def _pairwise_sum_f32(values: np.ndarray, length: int) -> np.float32:
    """NumPy's contiguous float32 pairwise reduction for lengths below 128."""

    if length < 8:
        result = np.float32(-0.0)
        for i in range(length):
            result = np.float32(result + values[i])
        return result
    r0 = values[0]
    r1 = values[1]
    r2 = values[2]
    r3 = values[3]
    r4 = values[4]
    r5 = values[5]
    r6 = values[6]
    r7 = values[7]
    i = 8
    while i <= length - 8:
        r0 = np.float32(r0 + values[i])
        r1 = np.float32(r1 + values[i + 1])
        r2 = np.float32(r2 + values[i + 2])
        r3 = np.float32(r3 + values[i + 3])
        r4 = np.float32(r4 + values[i + 4])
        r5 = np.float32(r5 + values[i + 5])
        r6 = np.float32(r6 + values[i + 6])
        r7 = np.float32(r7 + values[i + 7])
        i += 8
    left = np.float32(np.float32(r0 + r1) + np.float32(r2 + r3))
    right = np.float32(np.float32(r4 + r5) + np.float32(r6 + r7))
    result = np.float32(left + right)
    while i < length:
        result = np.float32(result + values[i])
        i += 1
    return result


@njit(cache=True, nogil=True)
def _pairwise_sum_f64(values: np.ndarray, length: int) -> float:
    """NumPy's contiguous float64 pairwise reduction for lengths below 128."""

    if length < 8:
        result = -0.0
        for i in range(length):
            result += values[i]
        return result
    r0 = values[0]
    r1 = values[1]
    r2 = values[2]
    r3 = values[3]
    r4 = values[4]
    r5 = values[5]
    r6 = values[6]
    r7 = values[7]
    i = 8
    while i <= length - 8:
        r0 += values[i]
        r1 += values[i + 1]
        r2 += values[i + 2]
        r3 += values[i + 3]
        r4 += values[i + 4]
        r5 += values[i + 5]
        r6 += values[i + 6]
        r7 += values[i + 7]
        i += 8
    result = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < length:
        result += values[i]
        i += 1
    return result


@njit(cache=True, nogil=True)
def _slope_f32_values(values: np.ndarray, length: int) -> float:
    """Mirror features._slope, which widens its float32 input to float64."""

    if length <= 1:
        return 0.0
    widened = np.empty(length, dtype=np.float64)
    for i in range(length):
        widened[i] = np.float64(values[i])
    mean = _pairwise_sum_f64(widened, length) / float(length)
    x_mean = 0.5 * float(length - 1)
    denom_values = np.empty(length, dtype=np.float64)
    numer_values = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = float(i) - x_mean
        denom_values[i] = x * x
        numer_values[i] = x * (widened[i] - mean)
    denom = _pairwise_sum_f64(denom_values, length)
    numer = _pairwise_sum_f64(numer_values, length)
    if denom <= 1e-9:
        return 0.0
    return numer / denom


@njit(cache=True, nogil=True)
def _linear_slope(values: np.ndarray, length: int) -> float:
    """Mirror features._linear_slope on float64 values."""

    if length <= 1:
        return 0.0
    mean = 0.0
    for i in range(length):
        mean += values[i]
    mean /= float(length)
    x_mean = 0.5 * float(length - 1)
    denom = 0.0
    numer = 0.0
    for i in range(length):
        x = float(i) - x_mean
        denom += x * x
        numer += x * (values[i] - mean)
    if denom <= EPS:
        return 0.0
    return numer / denom


@njit(cache=True, nogil=True)
def _raw_summary62(
    prefix: np.ndarray,
    target_x_f32: np.float32,
    target_y_f32: np.float32,
    radius_f32: np.float32,
    progress_f32: np.float32,
    distance_over_radius_f32: np.float32,
) -> np.ndarray:
    """Return the current public live feature law in canonical order."""

    n = prefix.shape[0]
    out = np.zeros(62, dtype=np.float64)

    # The live builder first writes target geometry into a float32 one-row
    # research record, then the primitive layer reads it back as float64.
    tx = np.float64(target_x_f32)
    ty = np.float64(target_y_f32)
    radius = max(np.float64(radius_f32), EPS)
    target_norm_unclipped = _norm2(tx, ty)
    target_distance = max(target_norm_unclipped, EPS)
    if target_norm_unclipped <= EPS:
        toward_x = 1.0
        toward_y = 0.0
    else:
        toward_x = tx / target_norm_unclipped
        toward_y = ty / target_norm_unclipped
    tangent_x = -toward_y
    tangent_y = toward_x

    # Primitive-layer prefix calculations are float64 because _valid_prefix
    # explicitly converts the float32 live row before doing any reductions.
    speeds = np.empty(n, dtype=np.float64)
    prefix_x = 0.0
    prefix_y = 0.0
    path_length = 0.0
    peak_speed = 0.0
    for i in range(n):
        x = np.float64(prefix[i, 0])
        y = np.float64(prefix[i, 1])
        prefix_x += x
        prefix_y += y
        speed = _norm2(x, y)
        speeds[i] = speed
        path_length += speed
        if i == 0 or speed > peak_speed:
            peak_speed = speed
    prefix_distance = _norm2(prefix_x, prefix_y)
    mean_speed = path_length / float(n) if n else 0.0
    speed_variance = 0.0
    for i in range(n):
        d = speeds[i] - mean_speed
        speed_variance += d * d
    speed_std = np.sqrt(speed_variance / float(n)) if n else 0.0

    # Live speed/acceleration fields are calculated from float32 rows and then
    # stored as float32.  Keep these distinct from primitive float64 speeds.
    if n:
        lvx = np.float64(prefix[n - 1, 0])
        lvy = np.float64(prefix[n - 1, 1])
        speed_at_b32 = _norm2_f32(prefix[n - 1, 0], prefix[n - 1, 1])
    else:
        lvx = 0.0
        lvy = 0.0
        speed_at_b32 = np.float32(0.0)
    if n >= 2:
        lax32 = np.float32(prefix[n - 1, 0] - prefix[n - 2, 0])
        lay32 = np.float32(prefix[n - 1, 1] - prefix[n - 2, 1])
        accel_at_b32 = _norm2_f32(lax32, lay32)
    else:
        lax32 = np.float32(0.0)
        lay32 = np.float32(0.0)
        accel_at_b32 = np.float32(0.0)
    lax = np.float64(lax32)
    lay = np.float64(lay32)

    # causal_context_arrays values consumed by primitive_features.  These are
    # computed on float32 arrays and rounded into float32 output columns.
    inside32 = np.float32(0.0)
    velocity_toward32 = np.float32(0.0)
    velocity_lateral32 = np.float32(0.0)
    recent_zero32 = np.float32(0.0)
    recent_sign_flip32 = np.float32(0.0)
    recent_direction_change32 = np.float32(0.0)
    recent_speed_slope32 = np.float32(0.0)
    recent_accel_slope32 = np.float32(0.0)
    duration32 = np.float32(0.0)
    if n:
        duration32 = np.float32(n)
        inside32 = np.float32(
            1.0 if target_norm_unclipped <= max(np.float64(radius_f32), 0.0) else 0.0
        )
        # causal_context_arrays uses a tighter 1e-9 target-unit threshold.
        if target_norm_unclipped > 1e-9:
            causal_tx = tx / target_norm_unclipped
            causal_ty = ty / target_norm_unclipped
        else:
            causal_tx = 1.0
            causal_ty = 0.0
        velocity_toward32 = np.float32(lvx * causal_tx + lvy * causal_ty)
        velocity_lateral32 = np.float32(lvx * causal_ty - lvy * causal_tx)

        rlen = min(40, n)
        start = n - rlen
        recent_mag32 = np.empty(rlen, dtype=np.float32)
        recent_accel_mag32 = np.empty(rlen, dtype=np.float32)
        zeros = 0
        for j in range(rlen):
            i = start + j
            mx = prefix[i, 0]
            my = prefix[i, 1]
            mag = _norm2_f32(mx, my)
            recent_mag32[j] = mag
            if mag <= np.float32(0.0):
                zeros += 1
            if j == 0:
                ax = np.float32(0.0)
                ay = np.float32(0.0)
            else:
                ax = np.float32(prefix[i, 0] - prefix[i - 1, 0])
                ay = np.float32(prefix[i, 1] - prefix[i - 1, 1])
            recent_accel_mag32[j] = _norm2_f32(ax, ay)
        recent_zero32 = np.float32(float(zeros) / float(rlen))
        recent_speed_slope32 = np.float32(_slope_f32_values(recent_mag32, rlen))
        recent_accel_slope32 = np.float32(_slope_f32_values(recent_accel_mag32, rlen))

        flips = 0
        flip_denom = 0
        for axis in range(2):
            have_previous = False
            previous_sign = 0
            for i in range(start, n):
                value = prefix[i, axis]
                if abs(float(value)) > 0.0:
                    sign = 1 if value > 0 else -1
                    if have_previous:
                        flip_denom += 1
                        if sign != previous_sign:
                            flips += 1
                    previous_sign = sign
                    have_previous = True
        if flip_denom:
            recent_sign_flip32 = np.float32(float(flips) / float(flip_denom))

        if rlen > 2:
            direction_changes = 0
            direction_denom = 0
            previous_valid = False
            previous_ux = 0.0
            previous_uy = 0.0
            for i in range(start, n):
                # The source expression divides float32 arrays before assigning
                # into its float64 unit array, hence the explicit float32 quotient.
                mag = recent_mag32[i - start]
                valid = mag > np.float32(1e-6)
                if valid:
                    ux = np.float64(np.float32(prefix[i, 0] / mag))
                    uy = np.float64(np.float32(prefix[i, 1] / mag))
                    if previous_valid:
                        direction_denom += 1
                        if previous_ux * ux + previous_uy * uy < 0.7071:
                            direction_changes += 1
                    previous_ux = ux
                    previous_uy = uy
                previous_valid = valid
            if direction_denom:
                recent_direction_change32 = np.float32(
                    float(direction_changes) / float(direction_denom)
                )

    # Path relative to B.  For an empty prefix the authority deliberately
    # supplies one zero point via _prefix_path_at_b.
    crossed = 0.0
    overshot = 0.0
    min_prefix_dist = target_distance
    near_count = 0
    path_points = n if n else 1
    cx = 0.0
    cy = 0.0
    near_threshold = max(radius * 2.0, radius + 4.0)
    max_axis = -np.inf
    for i in range(path_points):
        if n:
            cx += np.float64(prefix[i, 0])
            cy += np.float64(prefix[i, 1])
            px = cx - prefix_x
            py = cy - prefix_y
        else:
            px = 0.0
            py = 0.0
        dx = tx - px
        dy = ty - py
        distance = _norm2(dx, dy)
        if i == 0 or distance < min_prefix_dist:
            min_prefix_dist = distance
        if distance <= radius:
            crossed = 1.0
        if distance <= near_threshold:
            near_count += 1
        axis = px * toward_x + py * toward_y
        if axis > max_axis:
            max_axis = axis
    if max_axis > target_distance * 1.03:
        overshot = 1.0
    near_rate = float(near_count) / float(path_points)

    early_len = min(20, n)
    recent20_len = min(20, n)
    early_mean = 0.0
    recent_mean = 0.0
    for i in range(early_len):
        early_mean += speeds[i]
    for i in range(n - recent20_len, n):
        recent_mean += speeds[i]
    if early_len:
        early_mean /= float(early_len)
    if recent20_len:
        recent_mean /= float(recent20_len)
    speed_drop_recent = early_mean - recent_mean

    movement_norm = max(_norm2(lvx, lvy), EPS)
    alignment_cos = (lvx * toward_x + lvy * toward_y) / movement_norm
    alignment_sin = (lvx * tangent_x + lvy * tangent_y) / movement_norm
    last_norm = _norm2(lvx, lvy)
    if last_norm <= EPS and target_norm_unclipped <= EPS:
        direction_error = 0.0
    elif last_norm <= EPS or target_norm_unclipped <= EPS:
        direction_error = 90.0
    else:
        cosine = (lvx * tx + lvy * ty) / (last_norm * target_norm_unclipped)
        cosine = min(1.0, max(-1.0, cosine))
        direction_error = np.degrees(np.arccos(cosine))

    jerk = 0.0
    if n >= 4:
        a0x = np.float64(prefix[n - 2, 0]) - np.float64(prefix[n - 3, 0])
        a0y = np.float64(prefix[n - 2, 1]) - np.float64(prefix[n - 3, 1])
        a1x = np.float64(prefix[n - 1, 0]) - np.float64(prefix[n - 2, 0])
        a1y = np.float64(prefix[n - 1, 1]) - np.float64(prefix[n - 2, 1])
        jerk = _norm2(a1x - a0x, a1y - a0y)

    prefix_target_axis = prefix_x * toward_x + prefix_y * toward_y
    prefix_lateral = (prefix_x * tangent_x + prefix_y * tangent_y) / radius
    accel_toward = lax * toward_x + lay * toward_y
    accel_lateral = lax * tangent_x + lay * tangent_y
    distance_over_radius = float(distance_over_radius_f32)
    recent_speed_slope = float(recent_speed_slope32)
    recent_accel_slope = float(recent_accel_slope32)
    recent_zero = float(recent_zero32)
    recent_sign_flip = float(recent_sign_flip32)
    recent_direction_change = float(recent_direction_change32)
    speed_at_b = float(speed_at_b32)
    accel_at_b = float(accel_at_b32)

    out[0] = float(duration32)
    out[1] = prefix_x
    out[2] = prefix_y
    out[3] = prefix_distance
    out[4] = path_length
    out[5] = prefix_distance / max(path_length, EPS)
    out[6] = prefix_target_axis
    out[7] = prefix_lateral
    out[8] = tx
    out[9] = ty
    out[10] = target_distance
    out[11] = radius
    out[12] = distance_over_radius
    out[13] = toward_x
    out[14] = toward_y
    out[15] = float(progress_f32)
    out[16] = float(inside32)
    out[17] = 1.0 if distance_over_radius <= 2.0 or inside32 > 0.5 else 0.0
    out[18] = crossed
    out[19] = overshot
    out[20] = min_prefix_dist / radius
    out[21] = near_rate
    out[22] = lvx
    out[23] = lvy
    out[24] = lvx / movement_norm
    out[25] = lvy / movement_norm
    out[26] = speed_at_b
    out[27] = mean_speed
    out[28] = peak_speed
    out[29] = speed_std
    out[30] = speed_at_b / target_distance
    out[31] = speed_at_b / radius
    out[32] = float(velocity_toward32)
    out[33] = float(velocity_lateral32)
    out[34] = alignment_cos
    out[35] = alignment_sin
    out[36] = direction_error
    out[37] = accel_at_b
    out[38] = accel_toward
    out[39] = accel_lateral
    out[40] = recent_speed_slope
    out[41] = recent_accel_slope
    out[42] = 1.0 if recent_speed_slope < -0.03 or accel_at_b < -0.03 else 0.0
    out[43] = speed_drop_recent
    out[44] = jerk
    out[45] = abs(speed_at_b - mean_speed)
    out[46] = recent_zero
    out[47] = recent_sign_flip
    out[48] = recent_direction_change
    out[49] = 1.0 if speed_at_b > 0.5 or recent_mean > 0.5 else 0.0
    out[50] = (
        1.0
        if distance_over_radius <= 1.5
        and speed_at_b < max(1.0, mean_speed * 0.5)
        else 0.0
    )

    # Prefix-shape layer: recent curvature, approach, four speed-profile bins,
    # and target-frame lateral drift.
    shape_start = max(0, n - 16)
    curv_signed_sum = 0.0
    curv_abs_sum = 0.0
    turn_angles = np.empty(15, dtype=np.float64)
    turn_count = 0
    if n - shape_start >= 2:
        for i in range(shape_start, n - 1):
            x0 = np.float64(prefix[i, 0])
            y0 = np.float64(prefix[i, 1])
            x1 = np.float64(prefix[i + 1, 0])
            y1 = np.float64(prefix[i + 1, 1])
            n0 = _norm2(x0, y0)
            n1 = _norm2(x1, y1)
            if n0 > 1e-6 and n1 > 1e-6:
                denom = max(n0 * n1, 1e-9)
                sine = (x0 * y1 - y0 * x1) / denom
                curv_signed_sum += sine
                curv_abs_sum += abs(sine)
                dot = (x0 * x1 + y0 * y1) / denom
                dot = min(1.0, max(-1.0, dot))
                turn_angles[turn_count] = np.degrees(np.arccos(dot))
                turn_count += 1
    if turn_count:
        curv_signed = curv_signed_sum / float(turn_count)
        curv_abs = curv_abs_sum / float(turn_count)
        direction_change_mean = 0.0
        for i in range(turn_count):
            direction_change_mean += turn_angles[i]
        direction_change_mean /= float(turn_count)
    else:
        curv_signed = 0.0
        curv_abs = 0.0
        direction_change_mean = 0.0
    direction_change_slope = (
        _linear_slope(turn_angles, turn_count) if turn_count >= 2 else 0.0
    )

    approach_x = 0.0
    approach_y = 0.0
    for i in range(max(0, n - 10), n):
        approach_x += np.float64(prefix[i, 0])
        approach_y += np.float64(prefix[i, 1])
    approach_norm = _norm2(approach_x, approach_y)
    if approach_norm > 1e-6:
        approach_cos = (approach_x * toward_x + approach_y * toward_y) / approach_norm
        approach_sin = (approach_x * tangent_x + approach_y * tangent_y) / approach_norm
    else:
        approach_cos = 0.0
        approach_sin = 0.0

    for b in range(4):
        if n == 0:
            bin_mean = 0.0
        else:
            lo = int(np.floor(float(n) * float(b) / 4.0))
            hi = int(np.floor(float(n) * float(b + 1) / 4.0))
            if hi <= lo:
                hi = min(lo + 1, n)
            if lo < n:
                total = 0.0
                for i in range(lo, hi):
                    total += speeds[i]
                bin_mean = total / float(hi - lo)
            else:
                bin_mean = speeds[n - 1]
        out[57 + b] = bin_mean / max(mean_speed, 1e-6)

    along = 0.0
    lateral = 0.0
    for i in range(shape_start, n):
        x = np.float64(prefix[i, 0])
        y = np.float64(prefix[i, 1])
        along += abs(x * toward_x + y * toward_y)
        lateral += abs(x * tangent_x + y * tangent_y)
    lateral_drift = lateral / max(along + lateral, 1e-6) if n else 0.0

    out[51] = curv_signed
    out[52] = curv_abs
    out[53] = direction_change_mean
    out[54] = direction_change_slope
    out[55] = approach_cos
    out[56] = approach_sin
    out[61] = lateral_drift
    return out


@njit(cache=True, nogil=True)
def _normalized_summary(
    prefix: np.ndarray,
    target_x_f32: np.float32,
    target_y_f32: np.float32,
    radius_f32: np.float32,
    progress_f32: np.float32,
    distance_over_radius_f32: np.float32,
    reorder: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    raw = _raw_summary62(
        prefix,
        target_x_f32,
        target_y_f32,
        radius_f32,
        progress_f32,
        distance_over_radius_f32,
    )
    out = np.empty((1, reorder.shape[0]), dtype=np.float32)
    for j in range(reorder.shape[0]):
        value = np.float32(raw[reorder[j]])
        if not np.isfinite(value):
            value = np.float32(0.0)
        normalized = np.float32(
            np.float32(value - mean[j]) / std[j]
        )
        out[0, j] = normalized if np.isfinite(normalized) else np.float32(0.0)
    return out


@dataclass(frozen=True)
class SummaryInput:
    """Sanitized scalar hand-off shared by raw and normalized entry points."""

    prefix: np.ndarray
    target_x: np.float32
    target_y: np.float32
    radius: np.float32
    progress: np.float32
    distance_over_radius: np.float32


class CompiledSummaryVector:
    """Checkpoint-bound compiled replacement for one live summary vector."""

    def __init__(self, planner: Any) -> None:
        names = tuple(str(name) for name in planner.summary_names)
        index = {name: i for i, name in enumerate(CANONICAL_FEATURE_NAMES)}
        unknown = [name for name in names if name not in index]
        if unknown:
            raise ValueError(f"compiled summary does not know checkpoint features: {unknown!r}")
        if len(set(names)) != len(names):
            raise ValueError("checkpoint summary feature names contain duplicates")
        self.names = names
        self.reorder = np.asarray([index[name] for name in names], dtype=np.int64)
        self.mean = np.ascontiguousarray(
            np.asarray(planner._summary_mean, dtype=np.float32).reshape(-1)
        )
        self.std = np.ascontiguousarray(
            np.asarray(planner._summary_std, dtype=np.float32).reshape(-1)
        )
        if self.mean.shape != (len(names),) or self.std.shape != (len(names),):
            raise ValueError("checkpoint summary normalizer shape differs from feature names")
        self.std = self.std.copy()
        self.std[self.std < np.float32(1e-6)] = np.float32(1.0)

    @staticmethod
    def _prepare(
        prefix: np.ndarray,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress: float,
        *,
        assume_finite_counts: bool,
    ) -> SummaryInput:
        p = np.asarray(prefix, dtype=np.float32)
        if p.ndim != 2 or (p.size and p.shape[1] != 2):
            raise ValueError("prefix must have shape (P, 2)")
        if not p.size:
            p = p.reshape(0, 2)
        elif not assume_finite_counts:
            p = p[np.isfinite(p).all(axis=1)]
        if not p.flags.c_contiguous:
            p = np.ascontiguousarray(p)

        # Match summary_features: distance/radius is evaluated before fields
        # are packed into float32 research-row arrays.
        tx = float(target_rel_at_B[0])
        ty = float(target_rel_at_B[1])
        radius = float(target_radius)
        distance = float(np.hypot(tx, ty))
        distance_over_radius = np.float32(distance / max(radius, 1e-6))
        return SummaryInput(
            prefix=p,
            target_x=np.float32(tx),
            target_y=np.float32(ty),
            radius=np.float32(radius),
            progress=np.float32(float(progress)),
            distance_over_radius=distance_over_radius,
        )

    def raw(
        self,
        prefix: np.ndarray,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress: float = 0.0,
        *,
        b_index_ms: float | None = None,
        assume_finite_counts: bool = False,
    ) -> np.ndarray:
        """Return all 62 canonical raw features.

        ``b_index_ms`` is accepted to retain the public call shape.  The
        current 62-feature law does not consume ``B_index_or_prefix_len_ms``;
        changing this argument therefore intentionally leaves output intact.
        """

        del b_index_ms
        item = self._prepare(
            prefix,
            target_rel_at_B,
            target_radius,
            progress,
            assume_finite_counts=assume_finite_counts,
        )
        return _raw_summary62(
            item.prefix,
            item.target_x,
            item.target_y,
            item.radius,
            item.progress,
            item.distance_over_radius,
        )

    def vector(
        self,
        prefix: np.ndarray,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress: float = 0.0,
        *,
        b_index_ms: float | None = None,
        assume_finite_counts: bool = False,
    ) -> np.ndarray:
        """Return the checkpoint-ordered normalized ``[1, F]`` float32 vector."""

        del b_index_ms
        item = self._prepare(
            prefix,
            target_rel_at_B,
            target_radius,
            progress,
            assume_finite_counts=assume_finite_counts,
        )
        return _normalized_summary(
            item.prefix,
            item.target_x,
            item.target_y,
            item.radius,
            item.progress,
            item.distance_over_radius,
            self.reorder,
            self.mean,
            self.std,
        )

    def prewarm(self) -> None:
        """Compile both raw and normalized kernels before the live path."""

        prefix = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
        self.raw(prefix, (40.0, -12.0), 18.0, 0.25, assume_finite_counts=True)
        self.vector(prefix, (40.0, -12.0), 18.0, 0.25, assume_finite_counts=True)


__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "CompiledSummaryVector",
    "SummaryInput",
]
