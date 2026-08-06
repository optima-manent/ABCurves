"""Interpretable and raw-sequence judges for generated mouse movement.

The functions in this module answer controlled, labeled two-sample questions:
given human/generated labels and leakage-aware groups, how much held-out
separation remains?  AUC 0.5 is chance and AUC 1.0 is complete separation.
Wasserstein tables show which marginal descriptors differ, while the raw CNN
can find temporal structure outside the hand-designed panel.

These scores are intentionally separate from unknown-identity attribution. A
labeled classifier may expose a population signature without supplying a
false-positive-safe threshold for a collection key absent from fitting and
calibration.
The complete-key cold protocol lives in :mod:`evaluation.cold`; known-match
human distance floors live in :mod:`evaluation.floors`.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from .texture import TEXTURE_FEATURE_NAMES, texture_features
from .features import target_frame_basis

# ---------------------------------------------------------------------------
# Trajectory-shape descriptors (the "does the curve move like a hand?" panel)
# ---------------------------------------------------------------------------
TRAJ_FEATURE_NAMES = (
    "log_duration",
    "along_endpoint_frac",
    "lateral_endpoint_over_radius",
    "endpoint_to_target_over_radius",
    "endpoint_angle_abs_deg",
    "speed_mean",
    "speed_max",
    "peak_speed_fraction",
    "tail_speed_ratio",
    "decel_tail_slope",
    "reversal_count",
    "jerk_roughness_log1p",
    "straightness",
    "overshoot",
)


def _traj_descriptor(dxdy: np.ndarray, mask: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    v = np.asarray(dxdy, dtype=np.float64)[np.asarray(mask) > 0.5]
    d = len(v)
    if d < 2:
        return np.zeros(len(TRAJ_FEATURE_NAMES), dtype=np.float64)
    speed = np.linalg.norm(v, axis=1)
    path = np.cumsum(v, axis=0)
    endpoint = path[-1]
    toward, tangent = target_frame_basis(target)
    end_along = float(endpoint @ toward)
    end_lat = float(endpoint @ tangent)
    target_dist = max(float(np.linalg.norm(target)), 1.0)
    r = max(float(radius), 1.0)
    endpoint_to_target = float(np.linalg.norm(endpoint - target))
    endpoint_angle = math.degrees(math.atan2(abs(end_lat), max(end_along, 1e-6)))
    peak_idx = int(np.argmax(speed))
    peak_frac = peak_idx / max(d - 1, 1)
    tail = speed[int(0.75 * d) :]
    tail_ratio = float(np.mean(tail) / max(np.mean(speed), 1e-6)) if len(tail) else 0.0
    # deceleration slope over the last third
    seg = speed[int(0.66 * d) :]
    if len(seg) >= 2:
        xs = np.arange(len(seg), dtype=np.float64)
        decel = float(np.polyfit(xs, seg, 1)[0])
    else:
        decel = 0.0
    # reversals: sign changes of the along-target velocity component
    along_v = v @ toward
    sign = np.sign(along_v[np.abs(along_v) > 1e-6])
    reversals = float(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0.0
    jerk = np.linalg.norm(np.diff(v, n=2, axis=0), axis=1) if d >= 3 else np.zeros(1)
    jerk_rough = float(np.mean(jerk))
    straightness = float(np.linalg.norm(endpoint) / max(float(np.sum(speed)), 1e-6))
    overshoot = float(max(0.0, float(np.max(path @ toward)) - target_dist) / target_dist)
    return np.asarray(
        [
            math.log(max(float(d), 1.0)),
            end_along / target_dist,
            abs(end_lat) / r,
            endpoint_to_target / r,
            abs(endpoint_angle),
            float(np.mean(speed)),
            float(np.max(speed)),
            peak_frac,
            tail_ratio,
            decel,
            reversals,
            math.log1p(max(jerk_rough, 0.0)),
            straightness,
            overshoot,
        ],
        dtype=np.float64,
    )


def trajectory_features(streams: list[np.ndarray], targets: list[np.ndarray], radii: list[float]) -> np.ndarray:
    """Shape descriptors ``[N, 14]`` for a list of variable-length B->C streams."""

    out = np.zeros((len(streams), len(TRAJ_FEATURE_NAMES)), dtype=np.float64)
    for i, v in enumerate(streams):
        v = np.asarray(v, dtype=np.float64)
        mask = np.ones(len(v), dtype=np.float32)
        out[i] = _traj_descriptor(v, mask, np.asarray(targets[i], dtype=np.float64), float(radii[i]))
    out[~np.isfinite(out)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Target-entry, landing, stable-stop, and A->B | B->C seam descriptors
# ---------------------------------------------------------------------------
TARGET_EVENT_FEATURE_NAMES = (
    "seam_vector_jump_over_local_speed",
    "seam_speed_log_ratio_abs",
    "seam_direction_change_deg",
    "target_ever_entered",
    "first_target_entry_fraction",
    "target_entry_count",
    "target_exit_count",
    "inside_target_fraction",
    "tail_inside_target_fraction",
    "endpoint_inside_target",
    "endpoint_inside_inner_target",
    "tail_speed_mean",
    "tail_speed_over_mean",
    "tail_path_over_radius",
    "tail_excursion_over_radius",
    "stable_stop_flag",
)

FULL_SYSTEM_FEATURE_NAMES = TRAJ_FEATURE_NAMES + TEXTURE_FEATURE_NAMES + TARGET_EVENT_FEATURE_NAMES


def _validate_masked_stream_batch(
    dxdy: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(dxdy, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError(f"{name} must have shape [N, T, 2]")
    if valid.shape != values.shape[:2]:
        raise ValueError(f"{name.replace('dxdy', 'mask')} must have shape {values.shape[:2]}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(valid)):
        raise ValueError(f"{name} and its mask must be finite")
    return values, valid


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 and nb <= 1e-9:
        return 0.0
    if na <= 1e-9 or nb <= 1e-9:
        return 90.0
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def target_event_features(
    dxdy: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    radii: np.ndarray,
    *,
    prefix_dxdy: np.ndarray | None = None,
    prefix_mask: np.ndarray | None = None,
    seam_window: int = 8,
    stop_window: int = 32,
    inner_radius_fraction: float = 0.5,
    stable_inside_fraction: float = 0.95,
    stable_excursion_fraction: float = 0.25,
    stable_excursion_floor: float = 2.0,
) -> np.ndarray:
    """Mask-aware target/seam descriptors for raw or smooth continuations.

    Targets are relative to B. Target entry and stop behavior are measured from
    the complete valid B->C path. Seam continuity compares short vector-mean
    windows immediately before and after B, which is robust to isolated zero
    packets while remaining sensitive to a direction or speed discontinuity.

    ``stable_stop_flag`` is a transparent proxy: the endpoint must be inside,
    at least ``stable_inside_fraction`` of the final window must remain inside,
    and its maximum excursion from the endpoint must be no more than the larger
    of ``stable_excursion_fraction * radius`` and ``stable_excursion_floor``.
    Continuous tail descriptors are returned alongside it so callers need not
    rely on this thresholded summary.
    """

    values, valid = _validate_masked_stream_batch(dxdy, mask, name="dxdy")
    n = len(values)
    target_arr = np.asarray(targets, dtype=np.float64)
    radius_arr = np.asarray(radii, dtype=np.float64).reshape(-1)
    if target_arr.shape != (n, 2):
        raise ValueError(f"targets must have shape {(n, 2)}")
    if radius_arr.shape != (n,):
        raise ValueError(f"radii must have shape {(n,)}")
    if not np.all(np.isfinite(target_arr)) or not np.all(np.isfinite(radius_arr)):
        raise ValueError("targets and radii must be finite")
    if np.any(radius_arr <= 0.0):
        raise ValueError("radii must be positive")
    if int(seam_window) < 1 or int(stop_window) < 1:
        raise ValueError("seam_window and stop_window must be positive")
    if not 0.0 < float(inner_radius_fraction) <= 1.0:
        raise ValueError("inner_radius_fraction must be in (0, 1]")
    if not 0.0 <= float(stable_inside_fraction) <= 1.0:
        raise ValueError("stable_inside_fraction must be in [0, 1]")
    if float(stable_excursion_fraction) < 0.0 or float(stable_excursion_floor) < 0.0:
        raise ValueError("stable excursion limits must be non-negative")

    if prefix_dxdy is None:
        prefix_values = np.zeros((n, 0, 2), dtype=np.float64)
        prefix_valid = np.zeros((n, 0), dtype=np.float32)
    else:
        if prefix_mask is None:
            raise ValueError("prefix_mask is required when prefix_dxdy is supplied")
        prefix_values, prefix_valid = _validate_masked_stream_batch(
            prefix_dxdy, prefix_mask, name="prefix_dxdy"
        )
        if len(prefix_values) != n:
            raise ValueError("prefix and future batches must have the same N")

    out = np.zeros((n, len(TARGET_EVENT_FEATURE_NAMES)), dtype=np.float64)
    seam_n = int(seam_window)
    stop_n = int(stop_window)
    for i in range(n):
        future = values[i][valid[i] > 0.5]
        prefix = prefix_values[i][prefix_valid[i] > 0.5]
        if len(future) == 0:
            continue

        # A->B | B->C seam.
        before = prefix[-seam_n:]
        after = future[:seam_n]
        if len(before):
            before_vec = np.mean(before, axis=0)
            after_vec = np.mean(after, axis=0)
            before_speed = float(np.linalg.norm(before_vec))
            after_speed = float(np.linalg.norm(after_vec))
            local_speed = 0.5 * (
                float(np.mean(np.linalg.norm(before, axis=1)))
                + float(np.mean(np.linalg.norm(after, axis=1)))
            )
            seam_jump = float(np.linalg.norm(after_vec - before_vec) / max(local_speed, 1e-6))
            seam_speed_ratio = abs(math.log1p(after_speed) - math.log1p(before_speed))
            seam_direction = _angle_between_deg(before_vec, after_vec)
        else:
            # A missing prefix means seam continuity is unavailable, not a
            # ninety-degree discontinuity from an imaginary zero vector.
            seam_jump = seam_speed_ratio = seam_direction = 0.0

        # B-relative target entry and landing.
        path = np.cumsum(future, axis=0)
        radius = max(float(radius_arr[i]), 1e-6)
        distance = np.linalg.norm(path - target_arr[i][None, :], axis=1)
        inside = distance <= radius
        inner = distance <= radius * float(inner_radius_fraction)
        at_b_inside = bool(np.linalg.norm(target_arr[i]) <= radius)
        previous_inside = np.concatenate([[at_b_inside], inside[:-1]])
        entries = inside & ~previous_inside
        exits = ~inside & previous_inside
        ever_entered = at_b_inside or bool(np.any(inside))
        if at_b_inside:
            first_entry_fraction = 0.0
        elif np.any(entries):
            first_entry_fraction = int(np.argmax(entries)) / max(len(future) - 1, 1)
        else:
            # The companion ``target_ever_entered`` bit disambiguates this
            # right-censored sentinel from a genuine entry at the final tick.
            first_entry_fraction = 1.0

        tail_len = min(stop_n, len(future))
        tail_speed = np.linalg.norm(future[-tail_len:], axis=1)
        tail_path = float(np.sum(tail_speed))
        mean_speed = float(np.mean(np.linalg.norm(future, axis=1)))
        tail_start = len(path) - tail_len
        tail_anchor = np.zeros(2) if tail_start == 0 else path[tail_start - 1]
        tail_positions = np.concatenate([tail_anchor[None, :], path[-tail_len:]], axis=0)
        endpoint = path[-1]
        tail_excursion = float(np.max(np.linalg.norm(tail_positions - endpoint[None, :], axis=1)))
        tail_inside_fraction = float(np.mean(inside[-tail_len:]))
        endpoint_inside = bool(inside[-1])
        endpoint_inner = bool(inner[-1])
        excursion_limit = max(
            float(stable_excursion_fraction) * radius,
            float(stable_excursion_floor),
        )
        stable_stop = (
            endpoint_inside
            and tail_inside_fraction >= float(stable_inside_fraction)
            and tail_excursion <= excursion_limit
        )

        out[i] = np.asarray(
            [
                seam_jump,
                seam_speed_ratio,
                seam_direction,
                float(ever_entered),
                float(first_entry_fraction),
                float(np.sum(entries)),
                float(np.sum(exits)),
                float(np.mean(inside)),
                tail_inside_fraction,
                float(endpoint_inside),
                float(endpoint_inner),
                float(np.mean(tail_speed)),
                float(np.mean(tail_speed) / max(mean_speed, 1e-6)),
                tail_path / radius,
                tail_excursion / radius,
                float(stable_stop),
            ],
            dtype=np.float64,
        )
    out[~np.isfinite(out)] = 0.0
    return out


def full_system_features(
    dxdy: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    radii: np.ndarray,
    *,
    prefix_dxdy: np.ndarray | None = None,
    prefix_mask: np.ndarray | None = None,
    seam_window: int = 8,
    stop_window: int = 32,
) -> np.ndarray:
    """Combined shape + hardware texture + target/seam descriptors.

    This is the target-aware descriptor panel for a full Planner->Renderer
    comparison. It respects arbitrary masks rather than treating padded rows as
    motion and can also be applied to oracle-renderer stages using the exact
    same contexts.
    """

    values, valid = _validate_masked_stream_batch(dxdy, mask, name="dxdy")
    n = len(values)
    target_arr = np.asarray(targets, dtype=np.float64)
    radius_arr = np.asarray(radii, dtype=np.float64).reshape(-1)
    if target_arr.shape != (n, 2):
        raise ValueError(f"targets must have shape {(n, 2)}")
    if radius_arr.shape != (n,):
        raise ValueError(f"radii must have shape {(n,)}")
    if not np.all(np.isfinite(target_arr)) or not np.all(np.isfinite(radius_arr)):
        raise ValueError("targets and radii must be finite")
    if np.any(radius_arr <= 0.0):
        raise ValueError("radii must be positive")
    streams = [values[i][valid[i] > 0.5] for i in range(n)]
    trajectory = trajectory_features(
        streams,
        [target_arr[i] for i in range(n)],
        [float(radius_arr[i]) for i in range(n)],
    )
    texture = texture_features(values, valid)
    target_event = target_event_features(
        values,
        valid,
        target_arr,
        radius_arr,
        prefix_dxdy=prefix_dxdy,
        prefix_mask=prefix_mask,
        seam_window=seam_window,
        stop_window=stop_window,
    )
    out = np.concatenate([trajectory, texture, target_event], axis=1)
    if out.shape[1] != len(FULL_SYSTEM_FEATURE_NAMES):
        raise RuntimeError("full-system feature-name and matrix widths disagree")
    return out


# ---------------------------------------------------------------------------
# The two-sample tests
# ---------------------------------------------------------------------------
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("AUC scores must be finite")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # Ranks are one-based. All members of a tie receive their average rank.
        sorted_ranks[start:stop] = 0.5 * ((start + 1) + stop)
        start = stop
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = sorted_ranks
    pos = labels > 0.5
    n_pos = int(np.sum(pos))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _canonical_groups(values, expected: int, name: str) -> np.ndarray:
    vals = list(values)
    if len(vals) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(vals)}")

    def canonical(value) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return f"{type(value).__name__}:{value!r}"

    return np.asarray([canonical(value) for value in vals], dtype=object)


def _resolve_groups(
    n_real: int,
    n_fake: int,
    *,
    groups=None,
    real_groups=None,
    fake_groups=None,
) -> np.ndarray:
    """Resolve group IDs for the concatenated real/fake sample.

    ``groups`` is the convenient paired form: it supplies one source-trial ID
    per row and is reused for both equally sized bags. Explicit
    ``real_groups`` / ``fake_groups`` support unpaired comparisons. Equal
    explicit IDs intentionally join samples across bags into one fold.
    """

    if groups is not None:
        if real_groups is not None or fake_groups is not None:
            raise ValueError("pass either groups or real_groups/fake_groups, not both")
        if n_real != n_fake:
            raise ValueError("paired groups require equally sized real and fake bags")
        common = _canonical_groups(groups, n_real, "groups")
        return np.concatenate([common, common])

    if real_groups is None:
        rg = np.asarray([f"__real_sample_{i}" for i in range(n_real)], dtype=object)
    else:
        rg = _canonical_groups(real_groups, n_real, "real_groups")
    if fake_groups is None:
        fg = np.asarray([f"__fake_sample_{i}" for i in range(n_fake)], dtype=object)
    else:
        fg = _canonical_groups(fake_groups, n_fake, "fake_groups")
    return np.concatenate([rg, fg])


def _stratified_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic greedy stratified-group folds.

    Every source group is assigned wholly to one test fold. The requested fold
    count is reduced only when the number of groups carrying either class makes
    it mathematically impossible to put both classes in every test fold.
    """

    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    g = np.asarray(groups, dtype=object).reshape(-1)
    if len(y) != len(g):
        raise ValueError("labels and groups must have the same length")
    if len(y) < 4:
        raise ValueError("at least four samples are required for cross-validation")
    if int(folds) < 2:
        raise ValueError("folds must be at least 2")
    classes = (y > 0.5).astype(np.int64)
    if len(np.unique(classes)) != 2:
        raise ValueError("both classes are required")

    unique_groups, inverse = np.unique(g.astype(str), return_inverse=True)
    counts = np.zeros((len(unique_groups), 2), dtype=np.int64)
    np.add.at(counts, (inverse, classes), 1)
    groups_per_class = np.sum(counts > 0, axis=0)
    n_folds = min(int(folds), len(unique_groups), int(groups_per_class.min()))
    if n_folds < 2:
        raise ValueError("at least two independent groups from each class are required")

    totals = counts.sum(axis=0).astype(np.float64)
    group_sizes = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_score = float("inf")

    # Greedy balancing is deterministic for a seed. Several seeded restarts
    # avoid pathological allocations when group sizes or class mixes vary.
    for _ in range(64):
        order = rng.permutation(len(unique_groups))
        order = order[np.argsort(-group_sizes[order], kind="stable")]
        assignment = np.full(len(unique_groups), -1, dtype=np.int64)
        fold_counts = np.zeros((n_folds, 2), dtype=np.float64)
        fold_group_counts = np.zeros(n_folds, dtype=np.int64)

        for step, group_idx in enumerate(order):
            empty = np.where(fold_group_counts == 0)[0]
            candidates = empty if step < n_folds and len(empty) else np.arange(n_folds)
            candidate_scores = []
            for fold_idx in candidates:
                trial = fold_counts.copy()
                trial[fold_idx] += counts[group_idx]
                label_balance = float(np.mean(np.std(trial / totals[None, :], axis=0)))
                size_balance = float(np.std(trial.sum(axis=1) / max(float(group_sizes.sum()), 1.0)))
                group_balance = float(
                    np.std(
                        (fold_group_counts + (np.arange(n_folds) == fold_idx))
                        / max(float(len(unique_groups)), 1.0)
                    )
                )
                candidate_scores.append(label_balance + 0.2 * size_balance + 0.05 * group_balance)
            chosen = int(candidates[int(np.argmin(candidate_scores))])
            assignment[group_idx] = chosen
            fold_counts[chosen] += counts[group_idx]
            fold_group_counts[chosen] += 1

        valid = bool(np.all(fold_counts > 0))
        for fold_idx in range(n_folds):
            train_counts = totals - fold_counts[fold_idx]
            valid = valid and bool(np.all(train_counts > 0))
        score = (
            float(np.mean(np.std(fold_counts / totals[None, :], axis=0)))
            + 0.2 * float(np.std(fold_counts.sum(axis=1) / max(float(group_sizes.sum()), 1.0)))
        )
        if valid and score < best_score:
            best_assignment = assignment.copy()
            best_score = score

    if best_assignment is None:
        raise ValueError("could not construct grouped folds with both classes in every split")

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_folds):
        test_group_mask = best_assignment == fold_idx
        test_mask = test_group_mask[inverse]
        test_idx = np.flatnonzero(test_mask)
        train_idx = np.flatnonzero(~test_mask)
        out.append((train_idx, test_idx))
    return out


def _split_plans(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    repeats: int,
    seed: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    if int(repeats) < 1:
        raise ValueError("repeats must be at least 1")
    plans = []
    for repeat in range(int(repeats)):
        repeat_seed = int(
            np.random.SeedSequence([int(seed) & 0xFFFFFFFF, repeat, 0xC2A5]).generate_state(1)[0]
        )
        plans.append(_stratified_group_folds(labels, groups, folds=folds, seed=repeat_seed))
    return plans


def _aggregate_fold_aucs(aucs: list[float], weights: list[float]) -> float:
    if not aucs:
        return 0.5
    return float(np.average(np.asarray(aucs, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def _run_cv(
    labels: np.ndarray,
    groups: np.ndarray,
    plans: list[list[tuple[np.ndarray, np.ndarray]]],
    fit_predict: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    *,
    seed: int,
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    y = np.asarray(labels, dtype=np.float32)
    repeat_aucs: list[float] = []
    repeat_records: list[list[dict[str, Any]]] = []
    for repeat, splits in enumerate(plans):
        fold_aucs: list[float] = []
        fold_weights: list[float] = []
        records: list[dict[str, Any]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_seed = int(
                np.random.SeedSequence(
                    [int(seed) & 0xFFFFFFFF, repeat, fold_idx, 0xF01D]
                ).generate_state(1)[0]
            )
            scores = np.asarray(fit_predict(train_idx, test_idx, fold_seed), dtype=np.float64)
            test_y = y[test_idx]
            auc = _auc(scores, test_y)
            n_pos = int(np.sum(test_y > 0.5))
            n_neg = len(test_y) - n_pos
            weight = float(n_pos * n_neg)
            fold_aucs.append(auc)
            fold_weights.append(weight)
            records.append(
                {
                    "auc": auc,
                    "weight": weight,
                    "scores": scores,
                    "labels": test_y.copy(),
                    "groups": np.asarray(groups, dtype=object)[test_idx].copy(),
                }
            )
        repeat_aucs.append(_aggregate_fold_aucs(fold_aucs, fold_weights))
        repeat_records.append(records)
    return repeat_aucs, repeat_records


def _descriptor_cv(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    plans: list[list[tuple[np.ndarray, np.ndarray]]],
    *,
    seed: int,
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray, fold_seed: int) -> np.ndarray:
        # The scaler is part of the judge and must be fit only on judge-train.
        mean = x[train_idx].mean(axis=0, keepdims=True)
        std = x[train_idx].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        x_train = ((x[train_idx] - mean) / std).astype(np.float32)
        x_test = ((x[test_idx] - mean) / std).astype(np.float32)
        y_train = labels[train_idx].astype(np.float32)

        torch.manual_seed(fold_seed)
        model = torch.nn.Linear(x.shape[1], 1)
        # A convex logistic fit does not need a random start. Zero init also
        # makes the identical-constant-bag null exactly reproducible.
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
        xt = torch.from_numpy(x_train)
        yt = torch.from_numpy(y_train)
        opt = torch.optim.LBFGS(model.parameters(), max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt).squeeze(-1), yt)
            loss = loss + 1e-3 * sum((p * p).sum() for p in model.parameters())
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return model(torch.from_numpy(x_test)).squeeze(-1).numpy()

    return _run_cv(labels, groups, plans, fit_predict, seed=seed)


def _bootstrap_auc(
    records: list[list[dict[str, Any]]],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float] | None:
    """Cluster bootstrap held-out source groups within a randomly chosen repeat."""

    if int(iterations) <= 0:
        return None
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    max_attempts = max(int(iterations) * 10, 100)
    while len(values) < int(iterations) and attempts < max_attempts:
        attempts += 1
        repeat = records[int(rng.integers(0, len(records)))]
        fold_aucs: list[float] = []
        fold_weights: list[float] = []
        for rec in repeat:
            fold_groups = np.asarray(rec["groups"], dtype=object)
            unique = np.unique(fold_groups.astype(str))
            sampled = rng.choice(unique, size=len(unique), replace=True)
            indices = [np.flatnonzero(fold_groups.astype(str) == group) for group in sampled]
            idx = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
            labels = np.asarray(rec["labels"])[idx]
            n_pos = int(np.sum(labels > 0.5))
            n_neg = len(labels) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            fold_aucs.append(_auc(np.asarray(rec["scores"])[idx], labels))
            fold_weights.append(float(n_pos * n_neg))
        if fold_aucs:
            values.append(_aggregate_fold_aucs(fold_aucs, fold_weights))
    if not values:
        return (float("nan"), float("nan"))
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _permuted_labels(labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exchange labels while respecting source-trial clustering.

    Mixed groups (the paired real/generated case) are permuted within group.
    If every group is class-pure (an unpaired comparison), group labels are
    shuffled across whole groups.
    """

    y = np.asarray(labels, dtype=np.float32)
    g = np.asarray(groups, dtype=object).astype(str)
    unique = np.unique(g)
    group_indices = [np.flatnonzero(g == group) for group in unique]
    pure = [bool(np.all(y[idx] == y[idx][0])) for idx in group_indices]
    out = y.copy()
    if all(pure):
        shuffled_labels = rng.permutation(np.asarray([y[idx][0] for idx in group_indices]))
        for idx, label in zip(group_indices, shuffled_labels):
            out[idx] = label
        return out
    for idx in group_indices:
        n_pos = int(np.sum(out[idx] > 0.5))
        n_neg = len(idx) - n_pos
        if n_pos == n_neg:
            # Paired source trial: swap the two populations as one cluster,
            # preserving every augmented cut and within-trial dependency.
            if bool(rng.integers(0, 2)):
                out[idx] = 1.0 - out[idx]
        else:
            out[idx] = rng.permutation(out[idx])
    return out


def c2st_report(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    *,
    folds: int = 5,
    repeats: int = 5,
    seed: int = 7,
    groups=None,
    real_groups=None,
    fake_groups=None,
    bootstrap: int = 200,
    confidence: float = 0.95,
    permutations: int = 0,
) -> dict[str, Any]:
    """A grouped, repeated logistic C2ST report with uncertainty.

    ``real_feats`` / ``fake_feats`` are descriptor matrices (e.g. from
    :func:`trajectory_features` and/or :func:`abcurves.texture.texture_features`).
    Use ``groups`` for paired bags from the same source trials, or explicit
    ``real_groups`` / ``fake_groups`` for unpaired bags. All variants of a
    source group stay in one fold. Scaling is fit on judge-train only, AUC is
    computed within each fold, and fold AUCs are pair-count weighted.

    ``bootstrap`` performs a held-out source-group bootstrap for a confidence
    interval. ``permutations`` optionally retrains the judge under clustered
    label permutations and reports a two-sided permutation p-value.
    """

    real = np.asarray(real_feats, dtype=np.float64)
    fake = np.asarray(fake_feats, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError("real_feats and fake_feats must be 2D with matching feature counts")
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("each bag must contain at least two samples")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(fake)):
        raise ValueError("feature matrices must be finite")
    x = np.concatenate([real, fake]).astype(np.float32)
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))]).astype(np.float32)
    resolved_groups = _resolve_groups(
        len(real),
        len(fake),
        groups=groups,
        real_groups=real_groups,
        fake_groups=fake_groups,
    )
    plans = _split_plans(y, resolved_groups, folds=folds, repeats=repeats, seed=seed)
    repeat_aucs, records = _descriptor_cv(x, y, resolved_groups, plans, seed=seed)
    auc = float(np.mean(repeat_aucs))
    ci = _bootstrap_auc(
        records,
        iterations=bootstrap,
        confidence=confidence,
        seed=int(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0xB007]).generate_state(1)[0]),
    )

    null_aucs: list[float] = []
    rng = np.random.default_rng(
        int(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x9E37]).generate_state(1)[0])
    )
    for permutation in range(max(int(permutations), 0)):
        # Retry rare invalid unpaired permutations which leave a fold with one
        # class; paired source-trial permutations remain balanced by design.
        for _ in range(50):
            permuted = _permuted_labels(y, resolved_groups, rng)
            valid = all(
                len(np.unique(permuted[idx] > 0.5)) == 2
                for repeat_plan in plans
                for train_test in repeat_plan
                for idx in train_test
            )
            if valid:
                break
        else:
            raise ValueError("could not construct a valid grouped label permutation")
        perm_seed = int(
            np.random.SeedSequence(
                [int(seed) & 0xFFFFFFFF, permutation, 0xA11C]
            ).generate_state(1)[0]
        )
        perm_repeat, _ = _descriptor_cv(x, permuted, resolved_groups, plans, seed=perm_seed)
        null_aucs.append(float(np.mean(perm_repeat)))

    pvalue = None
    if null_aucs:
        observed_stat = abs(auc - 0.5)
        null_stat = np.abs(np.asarray(null_aucs, dtype=np.float64) - 0.5)
        pvalue = float((1 + np.sum(null_stat >= observed_stat)) / (len(null_stat) + 1))

    return {
        "auc": auc,
        "auc_ci": ci,
        "confidence": float(confidence),
        "repeat_auc": [float(value) for value in repeat_aucs],
        "fold_auc": [[float(rec["auc"]) for rec in repeat] for repeat in records],
        "folds": len(plans[0]),
        "repeats": int(repeats),
        "n_real": len(real),
        "n_fake": len(fake),
        "n_groups": int(len(np.unique(resolved_groups.astype(str)))),
        "permutation_pvalue": pvalue,
        "permutation_auc": null_aucs,
    }


def c2st_auc(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 7,
    groups=None,
    real_groups=None,
    fake_groups=None,
    repeats: int = 1,
) -> float:
    """Backward-compatible scalar grouped logistic C2ST AUC.

    For confidence intervals and permutation nulls use :func:`c2st_report`.
    """

    return float(
        c2st_report(
            real_feats,
            fake_feats,
            folds=folds,
            repeats=repeats,
            seed=seed,
            groups=groups,
            real_groups=real_groups,
            fake_groups=fake_groups,
            bootstrap=0,
            permutations=0,
        )["auc"]
    )


class _CnnJudge(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(3, 16, kernel_size=9, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
        )
        self.drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.fc(self.drop(pooled)).squeeze(-1)


def raw_crops(streams: list[np.ndarray], crop: int = 256) -> np.ndarray:
    """Fixed-length ``[N, 3, crop]`` crops (dx, dy, active flag) for the CNN judge."""

    out = np.zeros((len(streams), 3, crop), dtype=np.float32)
    for i, v in enumerate(streams):
        v = np.asarray(v, dtype=np.float32)
        t = min(len(v), crop)
        if t == 0:
            continue
        clipped = np.clip(v[:t], -16.0, 16.0) / 8.0
        out[i, 0, :t] = clipped[:, 0]
        out[i, 1, :t] = clipped[:, 1]
        out[i, 2, :t] = (np.abs(v[:t]).max(axis=1) > 0).astype(np.float32)
    return out


def cnn_c2st(
    real_crops: np.ndarray,
    fake_crops: np.ndarray,
    *,
    seed: int = 7,
    epochs: int = 20,
    folds: int = 5,
    repeats: int = 1,
    device: str = "cpu",
    groups=None,
    real_groups=None,
    fake_groups=None,
) -> float:
    """Grouped held-out AUC of a small 1D CNN reading raw count crops.

    AUC is computed inside each fold and then pair-count weighted; logits from
    independently trained fold models are never pooled. Use ``groups`` for
    paired real/generated crops from the same source trials.
    """

    dev = torch.device(device)
    x = np.concatenate([real_crops, fake_crops]).astype(np.float32)
    y = np.concatenate([np.zeros(len(real_crops)), np.ones(len(fake_crops))]).astype(np.float32)
    resolved_groups = _resolve_groups(
        len(real_crops),
        len(fake_crops),
        groups=groups,
        real_groups=real_groups,
        fake_groups=fake_groups,
    )
    plans = _split_plans(y, resolved_groups, folds=folds, repeats=repeats, seed=seed)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray, fold_seed: int) -> np.ndarray:
        torch.manual_seed(fold_seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(fold_seed)
        model = _CnnJudge().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.from_numpy(x[train_idx]).to(dev)
        yt = torch.from_numpy(y[train_idx]).to(dev)
        n_train = len(train_idx)
        model.train()
        for _ in range(epochs):
            order = torch.randperm(n_train, device=dev)
            for bstart in range(0, n_train, 128):
                idx = order[bstart : bstart + 128]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(x[test_idx]).to(dev)).cpu().numpy()

    repeat_aucs, _ = _run_cv(y, resolved_groups, plans, fit_predict, seed=seed)
    return float(np.mean(repeat_aucs))


def _wasserstein1(x: np.ndarray, y: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 201)[1:-1]
    return float(np.mean(np.abs(np.quantile(x, grid) - np.quantile(y, grid))))


def w1_table(real_feats: np.ndarray, fake_feats: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    """Per-descriptor Wasserstein-1 gap (standardized by the real spread).

    Sorted high-to-low, this reads out *which* statistics separate the fake bag
    from the real bag -- the diagnostic that told us what to fix next.
    """

    real = np.asarray(real_feats, dtype=np.float64)
    fake = np.asarray(fake_feats, dtype=np.float64)
    scale = real.std(axis=0)
    scale[scale < 1e-6] = 1.0
    table = {names[j]: float(_wasserstein1(fake[:, j] / scale[j], real[:, j] / scale[j])) for j in range(len(names))}
    return dict(sorted(table.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Convenience: a full raw-level panel on two bags of streams
# ---------------------------------------------------------------------------
def raw_texture_panel(
    real_streams: list[np.ndarray],
    fake_streams: list[np.ndarray],
    *,
    seed: int = 7,
    device: str = "cpu",
    run_cnn: bool = True,
    groups=None,
    repeats: int = 1,
) -> dict[str, object]:
    """Judge two bags of raw B->C count streams end to end.

    Returns descriptor-C2ST AUC, CNN-C2ST AUC (optional), and the top W1 gaps.
    Feed real human streams and generated streams of the *same events* (so any
    difference is texture, not situation).
    """

    def pad(streams: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        h = max(len(s) for s in streams)
        arr = np.zeros((len(streams), h, 2), dtype=np.float32)
        msk = np.zeros((len(streams), h), dtype=np.float32)
        for i, s in enumerate(streams):
            arr[i, : len(s)] = s
            msk[i, : len(s)] = 1.0
        return arr, msk

    real_arr, real_msk = pad(real_streams)
    fake_arr, fake_msk = pad(fake_streams)
    real_feats = texture_features(real_arr, real_msk)
    fake_feats = texture_features(fake_arr, fake_msk)
    result: dict[str, object] = {
        "descriptor_c2st_auc": c2st_auc(
            real_feats, fake_feats, seed=seed, groups=groups, repeats=repeats
        ),
        "top_w1": dict(list(w1_table(real_feats, fake_feats, TEXTURE_FEATURE_NAMES).items())[:6]),
        "n_real": len(real_streams),
        "n_fake": len(fake_streams),
    }
    if run_cnn:
        result["cnn_c2st_auc"] = cnn_c2st(
            raw_crops(real_streams),
            raw_crops(fake_streams),
            seed=seed,
            device=device,
            groups=groups,
            repeats=repeats,
        )
    return result


__all__ = [
    "trajectory_features",
    "texture_features",
    "target_event_features",
    "full_system_features",
    "TRAJ_FEATURE_NAMES",
    "TEXTURE_FEATURE_NAMES",
    "TARGET_EVENT_FEATURE_NAMES",
    "FULL_SYSTEM_FEATURE_NAMES",
    "c2st_auc",
    "c2st_report",
    "cnn_c2st",
    "raw_crops",
    "w1_table",
    "raw_texture_panel",
]
