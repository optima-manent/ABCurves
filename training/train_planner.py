"""Train the Planner (RWTA-16) and export a contracted checkpoint.

This is the exact recipe behind the bundled Planner. Read it top to bottom to
see how the 16 heads are trained to *cover the distribution* of human
continuations instead of collapsing to the average.

The idea in three lines:

1. Fit each real B->C future to a ProDMP weight vector + log-duration (the
   trajectory representation).  Standardize -> these are the head targets.
2. Train a TCN -> 16 heads.  Per example, decode all 16 to smooth curves,
   score each against the realized future with a differentiable trajectory
   distance, put ~all the gradient on the closest head (relaxed
   winner-takes-all), a crumb on the rest so none die.  Add a turn-rate hinge
   so no head learns the non-human zigzag caricature.
3. Export weights + normalizers + feature list + ProDMP metadata.

Usage:

    python training/train_planner.py \
        --train prepared/planner_train.npz \
        --val prepared/planner_val.npz \
        --out runs/planner_retrained.pt --epochs 260

Inputs must come from ``tools/prepare_dataset.py`` so train and validation
carry the same explicit causal seam contract. The trainer refuses uncontracted
fixtures instead of producing a checkpoint that the release runtime cannot use.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abcurves.data import load_dataset
from abcurves.capture_preprocess import (
    B_TRIGGER_REFERENCE,
    CAUSAL_SEAM_CONTRACT_SCHEMA,
    PLANNER_PROGRESS_REFERENCE,
)
from abcurves.features import build_feature_table, causal_context_arrays
from abcurves.planner import (
    CONTRACTED_PLANNER_SCHEMA,
    SPLIT_PREFIX_PLANNER_SCHEMA,
    MultiHeadModel,
    PlannerConfig,
)
from abcurves.prefix_representation import (
    DEFAULT_PREFIX_EMA_ALPHA,
    RAW_PREFIX_REPRESENTATION,
    SUPPORTED_PREFIX_REPRESENTATIONS,
    PlannerPrefixContract,
    PrefixRepresentationSpec,
    apply_prefix_representation,
    planner_prefix_contract_from_config,
)
from abcurves.preprocessing import PREPARED_CUT_SCHEMA
from abcurves.prodmp import ProDMP, ProDMPConfig

DUR_BINS = ((0.0, 150.0), (150.0, 250.0), (250.0, np.inf))
CUT_SAMPLING_MODES = ("all_weighted", "one_per_source_per_epoch")
MULTIB_AUGMENTATION_SCHEMA = "abcurves.planner_multib_augmentation.v1"
SOURCE_BALANCED_CUT_SCHEDULE = "source_specific_shuffled_cycle.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_text(arrays: dict, key: str) -> str | None:
    """Read singleton metadata without confusing it for an event column."""

    if key not in arrays:
        return None
    values = np.asarray(arrays[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{key} must contain exactly one dataset-level value")
    value = values[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def require_prepared_branch(arrays: dict, *, branch: str, label: str) -> None:
    """Refuse ambiguous inputs that bypass the release dataset builder."""

    schema = _metadata_text(arrays, "schema")
    cohort = _metadata_text(arrays, "cohort")
    if schema != PREPARED_CUT_SCHEMA or not str(cohort).startswith(f"{branch}_"):
        raise ValueError(
            f"{label} must be a {branch} split emitted by "
            "tools/prepare_dataset.py"
        )


def planner_seam_contract(train_arrays: dict, val_arrays: dict) -> dict:
    """Validate the causal A/B contract shared by the train and validation sets.

    Prepared datasets must share every causal onset/trigger/context field
    except the list of training B thresholds. A multi-B training set may
    therefore be judged on a fixed single-B validation set. In that case the
    exported runtime contract is the validation contract (the deployable B),
    while the training thresholds remain explicit augmentation provenance.
    """

    require_prepared_branch(train_arrays, branch="planner", label="train dataset")
    require_prepared_branch(
        val_arrays,
        branch="planner",
        label="validation dataset",
    )
    parsed: list[dict | None] = []
    for label, arrays in (("train", train_arrays), ("validation", val_arrays)):
        schema = _metadata_text(arrays, "causal_seam_contract_schema")
        raw = _metadata_text(arrays, "causal_seam_contract_json")
        if schema is None and raw is None:
            parsed.append(None)
            continue
        if schema is None or raw is None:
            raise ValueError(
                f"{label} dataset has an incomplete causal seam contract"
            )
        try:
            contract = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} dataset has invalid causal_seam_contract_json"
            ) from exc
        if schema != CAUSAL_SEAM_CONTRACT_SCHEMA or contract.get("schema") != schema:
            raise ValueError(
                f"{label} dataset uses unsupported causal seam schema {schema!r}"
            )
        trigger = contract.get("trigger")
        if not isinstance(trigger, dict):
            raise ValueError(f"{label} seam contract has no B trigger definition")
        if trigger.get("reference") != B_TRIGGER_REFERENCE:
            raise ValueError(
                f"{label} B trigger must use {B_TRIGGER_REFERENCE!r} progress"
            )
        raw_thresholds = trigger.get("thresholds")
        if not isinstance(raw_thresholds, list) or not raw_thresholds:
            raise ValueError(f"{label} seam contract has no B thresholds")
        thresholds: list[float] = []
        for raw_threshold in raw_thresholds:
            threshold = float(raw_threshold)
            if not 0.0 < threshold < 1.0:
                raise ValueError(f"{label} B threshold must be in (0, 1)")
            thresholds.append(threshold)
        if (
            contract.get("planner_context", {}).get("progress_reference")
            != PLANNER_PROGRESS_REFERENCE
        ):
            raise ValueError(
                f"{label} planner context must use "
                f"{PLANNER_PROGRESS_REFERENCE!r} progress"
            )
        # Store the thresholds explicitly in a stable, convenient runtime field.
        contract["b_thresholds"] = sorted(set(thresholds))
        parsed.append(contract)

    if parsed[0] is None or parsed[1] is None:
        raise ValueError(
            "train and validation datasets must both declare the causal seam "
            "contract emitted by tools/prepare_dataset.py"
        )
    if parsed[0] == parsed[1]:
        return parsed[0]

    train_contract = parsed[0]
    validation_contract = parsed[1]
    assert train_contract is not None and validation_contract is not None
    train_thresholds = list(train_contract["b_thresholds"])
    validation_thresholds = list(validation_contract["b_thresholds"])

    def without_thresholds(contract: dict) -> dict:
        comparable = json.loads(json.dumps(contract))
        comparable.pop("b_thresholds", None)
        comparable["trigger"].pop("thresholds", None)
        return comparable

    if without_thresholds(train_contract) != without_thresholds(validation_contract):
        raise ValueError(
            "train and validation causal seam contracts differ outside B thresholds"
        )
    if len(validation_thresholds) != 1:
        raise ValueError(
            "multi-B training requires one fixed validation/runtime B threshold"
        )
    if not math.isclose(
        float(validation_thresholds[0]),
        0.80,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "train and validation causal seam contracts differ: "
            "multi-B augmentation requires fixed B80 validation/runtime"
        )

    # Runtime must reproduce the validation/evaluation seam, not every
    # augmentation seam observed during training. The latter remain auditable
    # but must not broaden deployment silently.
    exported = json.loads(json.dumps(validation_contract))
    exported["training_augmentation"] = {
        "schema": MULTIB_AUGMENTATION_SCHEMA,
        "train_b_thresholds": train_thresholds,
        "validation_runtime_b_thresholds": validation_thresholds,
        "policy": "training_thresholds_may_differ_runtime_is_fixed_validation_B",
    }
    return exported


def event_weights(arrays: dict) -> np.ndarray:
    n = len(arrays["future_mask"])
    weights = np.asarray(
        arrays.get("event_weight", np.ones(n, dtype=np.float32)),
        dtype=np.float64,
    ).reshape(-1)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("event_weight must be a finite positive vector of length N")
    return weights


def source_trial_weight_summary(arrays: dict) -> dict[str, float | int]:
    """Validate that augmented siblings give each source trial total weight one."""

    if "source_trial_id" not in arrays:
        raise ValueError("source-balanced cut training requires source_trial_id")
    ids = np.asarray(arrays["source_trial_id"]).astype(str).reshape(-1)
    weights = event_weights(arrays)
    if ids.shape != weights.shape:
        raise ValueError("source_trial_id must be a vector matching event_weight")
    unique, inverse = np.unique(ids, return_inverse=True)
    totals = np.bincount(inverse, weights=weights, minlength=len(unique))
    error = np.abs(totals - 1.0)
    if not np.all(error <= 1e-6):
        worst = int(np.argmax(error))
        raise ValueError(
            "event_weight must sum to one per source_trial_id; "
            f"{unique[worst]!r} sums to {totals[worst]:.9g}"
        )
    counts = np.bincount(inverse, minlength=len(unique))
    return {
        "source_trials": int(len(unique)),
        "rows": int(len(ids)),
        "cuts_per_source_min": int(counts.min()) if len(counts) else 0,
        "cuts_per_source_max": int(counts.max()) if len(counts) else 0,
        "max_abs_weight_sum_error": float(error.max()) if len(error) else 0.0,
    }


def source_balanced_epoch_indices(
    arrays: dict,
    *,
    epoch: int,
    seed: int,
) -> np.ndarray:
    """Choose one cyclically balanced cut per source trial, then shuffle sources.

    Each source gets its own deterministic, seeded cut permutation for every
    complete cycle. Epochs walk that permutation without replacement, so every
    cut appears exactly once in each ``k``-epoch cycle. The final source-order
    shuffle uses a separate RNG that depends only on seed, epoch, and the sorted
    source-ID set. Consequently a singleton B80 dataset and a multi-B dataset
    with the same sources see the same per-epoch source order.
    """

    if int(epoch) < 1:
        raise ValueError("epoch must be one-based and positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    source_trial_weight_summary(arrays)
    ids = np.asarray(arrays["source_trial_id"]).astype(str).reshape(-1)
    groups: dict[str, list[int]] = {}
    for index, source_id in enumerate(ids):
        groups.setdefault(str(source_id), []).append(int(index))

    cut_ids = (
        np.asarray(arrays["cut_id"]).astype(str).reshape(-1)
        if "cut_id" in arrays
        else None
    )
    thresholds = (
        np.asarray(arrays["edge_trigger_progress"], dtype=np.float64).reshape(-1)
        if "edge_trigger_progress" in arrays
        else None
    )

    def canonical_indices(indices: list[int]) -> list[int]:
        return sorted(
            indices,
            key=lambda index: (
                float(thresholds[index]) if thresholds is not None else 0.0,
                str(cut_ids[index]) if cut_ids is not None else "",
                int(index),
            ),
        )

    epoch_index = int(epoch) - 1
    ordered_sources = sorted(groups)
    selected_by_source: dict[str, int] = {}
    for source_id in ordered_sources:
        indices = canonical_indices(groups[source_id])
        cut_count = len(indices)
        cycle_index, cycle_position = divmod(epoch_index, cut_count)
        source_digest = hashlib.sha256(source_id.encode("utf-8")).digest()
        source_words = [
            int.from_bytes(source_digest[offset : offset + 4], "little")
            for offset in range(0, 16, 4)
        ]
        cut_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(seed),
                    int(cycle_index),
                    0xC17,
                    *source_words,
                ]
            )
        )
        cycle_order = cut_rng.permutation(cut_count)
        selected_by_source[source_id] = indices[
            int(cycle_order[cycle_position])
        ]

    order_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), 0xB80, 0x50A])
    )
    source_order = order_rng.permutation(len(ordered_sources))
    return np.asarray(
        [
            selected_by_source[ordered_sources[int(position)]]
            for position in source_order
        ],
        dtype=np.int64,
    )


def represented_planner_arrays(
    arrays: dict,
    spec: PrefixRepresentationSpec,
    *,
    prefix_len: int | None = None,
) -> dict:
    """Return the exact prefix view used by every planner training input.

    ``raw`` returns the input mapping unchanged when it already fits the
    deployable window.
    Otherwise the same right-aligned window used at inference is enforced.
    ``causal_ema`` recomputes the representation from raw valid ticks and
    refreshes all prefix-derived context columns so the summary table, TCN
    tensor, ProDMP fit, and surrogate boundary velocity see one signal.
    """

    raw = np.asarray(arrays["prefix_raw_dxdy"], dtype=np.float32)
    mask = np.asarray(arrays["prefix_mask"], dtype=bool)
    if raw.ndim != 3 or raw.shape[2] != 2 or mask.shape != raw.shape[:2]:
        raise ValueError(
            "prefix_raw_dxdy/prefix_mask must have shapes [N, P, 2]/[N, P]"
        )
    window = raw.shape[1] if prefix_len is None else max(1, int(prefix_len))
    needs_truncation = raw.shape[1] > window
    if spec.name == RAW_PREFIX_REPRESENTATION and not needs_truncation:
        return arrays
    effective_mask = mask.copy()
    if needs_truncation:
        effective_mask[:, : raw.shape[1] - window] = False
    represented = np.zeros_like(raw, dtype=np.float32)
    for index in range(len(raw)):
        valid = raw[index, effective_mask[index]]
        represented[index, effective_mask[index]] = apply_prefix_representation(
            valid,
            spec,
        )

    out = dict(arrays)
    out["prefix_raw_dxdy"] = represented
    out["prefix_mask"] = effective_mask.astype(np.float32)
    out.update(causal_context_arrays(out))

    speed_at_b = np.zeros(len(raw), dtype=np.float32)
    accel_at_b = np.zeros(len(raw), dtype=np.float32)
    for index in range(len(raw)):
        valid = represented[index, effective_mask[index]]
        if len(valid):
            speed_at_b[index] = float(np.linalg.norm(valid[-1]))
        if len(valid) >= 2:
            accel_at_b[index] = float(np.linalg.norm(valid[-1] - valid[-2]))
    out["speed_at_B"] = speed_at_b
    out["accel_at_B"] = accel_at_b
    return out


def planner_training_views(
    arrays: dict,
    contract: PlannerPrefixContract,
    *,
    prefix_len: int | None = None,
) -> tuple[dict, dict]:
    """Return encoder/summary and ProDMP-boundary training views.

    Both are constructed from the same raw right-aligned window. Matching
    declarations reuse the same mapping.
    """

    encoder = represented_planner_arrays(
        arrays,
        contract.encoder,
        prefix_len=prefix_len,
    )
    if contract.views_match:
        return encoder, encoder
    boundary = represented_planner_arrays(
        arrays,
        contract.boundary_velocity,
        prefix_len=prefix_len,
    )
    if not np.array_equal(
        np.asarray(encoder["prefix_mask"]),
        np.asarray(boundary["prefix_mask"]),
    ):
        raise RuntimeError(
            "encoder and boundary prefix views produced different windows"
        )
    return encoder, boundary


def weighted_mean_std(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(x) != len(w):
        raise ValueError("weights and values must have the same first dimension")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total")
    mean = np.sum(x * w.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0) / total
    centered = x - mean
    variance = np.sum(
        centered * centered * w.reshape((-1,) + (1,) * (x.ndim - 1)),
        axis=0,
    ) / total
    return mean, np.sqrt(np.maximum(variance, 0.0))


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, percentile: float
) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(x) != len(w) or len(x) == 0:
        raise ValueError("weighted percentile requires equal non-empty vectors")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cumulative = np.cumsum(w)
    cutoff = float(np.clip(percentile, 0.0, 100.0)) / 100.0 * float(w.sum())
    return float(x[min(int(np.searchsorted(cumulative, cutoff, side="left")), len(x) - 1)])


# ---------------------------------------------------------------------------
# Step 1: ProDMP weight labels + input tensors
# ---------------------------------------------------------------------------
def _future_len(mask_row: np.ndarray) -> int:
    return int(np.sum(mask_row > 0.5))


def fit_weight_labels(prodmp: ProDMP, arrays: dict, *, eps_scale: float, ridge: float) -> dict:
    """Fit each real smooth future to normalized ProDMP weights + duration."""

    n = len(arrays["future_mask"])
    nw = prodmp.n_weights
    w_norm = np.zeros((n, nw, 2), dtype=np.float64)
    scale = np.full(n, float(eps_scale), dtype=np.float64)
    durations = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d = _future_len(arrays["future_mask"][i])
        durations[i] = float(d)
        if d <= 0:
            continue
        smooth = arrays["future_smooth_dxdy"][i, :d].astype(np.float64)
        prefix = arrays["prefix_raw_dxdy"][i][arrays["prefix_mask"][i] > 0.5].astype(np.float64)
        ydot_b = prefix[-1] if len(prefix) else np.zeros(2)
        path = np.cumsum(smooth, axis=0)
        s = max(float(np.linalg.norm(path[-1])), float(eps_scale))
        scale[i] = s
        w_norm[i] = prodmp.fit_weights(path / s, ydot_b / s, d, goal_mode="learned", ridge=ridge)
    return {"w_norm": w_norm, "scale": scale, "durations": durations}


def build_targets(labels: dict) -> np.ndarray:
    w_raw = labels["w_norm"] * labels["scale"][:, None, None]
    log_duration = np.log(np.clip(labels["durations"], 1.0, None))[:, None]
    return np.concatenate([w_raw.reshape(len(w_raw), -1), log_duration], axis=1).astype(np.float32)


def prefix_tensor(arrays: dict, mean: np.ndarray, std: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["prefix_raw_dxdy"], dtype=np.float32)
    mask = np.asarray(arrays["prefix_mask"], dtype=np.float32)
    n = raw.shape[0]
    out = np.zeros((n, length, 3), dtype=np.float32)
    out_mask = np.zeros((n, length), dtype=np.float32)
    take = min(length, raw.shape[1])
    raw_take = raw[:, raw.shape[1] - take :, :]
    mask_take = mask[:, mask.shape[1] - take :]
    norm = (raw_take - mean.reshape(1, 1, 2)) / std.reshape(1, 1, 2)
    norm[mask_take < 0.5] = 0.0
    out[:, -take:, :2] = norm
    out[:, -take:, 2] = mask_take
    out_mask[:, -take:] = mask_take
    return out, out_mask


def prefix_stats(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.asarray(arrays["prefix_raw_dxdy"], dtype=np.float64)
    mask = np.asarray(arrays["prefix_mask"], dtype=bool)
    valid = prefix[mask]
    example_weights = event_weights(arrays)
    tick_weights = np.broadcast_to(example_weights[:, None], mask.shape)[mask]
    mean, std = weighted_mean_std(valid, tick_weights)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def grid_targets(arrays: dict, s_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real smooth futures resampled onto the normalized-time grid + boundary velocity."""

    n = len(arrays["future_mask"])
    paths = np.zeros((n, len(s_grid), 2), dtype=np.float64)
    taus = np.zeros(n, dtype=np.float64)
    ydot = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        d = _future_len(arrays["future_mask"][i])
        if d > 0:
            path = np.cumsum(arrays["future_smooth_dxdy"][i, :d].astype(np.float64), axis=0)
            t = np.concatenate([[0.0], np.arange(1, d + 1, dtype=np.float64)])
            px = np.concatenate([[0.0], path[:, 0]])
            py = np.concatenate([[0.0], path[:, 1]])
            ts = s_grid * float(d)
            paths[i] = np.stack([np.interp(ts, t, px), np.interp(ts, t, py)], axis=1)
            taus[i] = float(d)
        prefix = arrays["prefix_raw_dxdy"][i][arrays["prefix_mask"][i] > 0.5].astype(np.float64)
        ydot[i] = prefix[-1] if len(prefix) else np.zeros(2)
    return paths, np.maximum(taus, 1.0), ydot


# ---------------------------------------------------------------------------
# Step 2: differentiable decode + surrogate distance (the WTA scoring)
# ---------------------------------------------------------------------------
def decode_grid(y_norm, ydot_b, aux):
    y_raw = y_norm * aux["y_std"] + aux["y_mean"]
    nw = aux["n_weights"]
    b, k, d = y_raw.shape
    w = y_raw[..., : nw * 2].reshape(b, k, nw, 2)
    log_tau = y_raw[..., -1].clamp(0.0, math.log(float(aux["horizon"])))
    tau = torch.exp(log_tau)
    s = aux["s_grid"]
    boundary_profile = s * torch.exp(-aux["half_alpha"] * s)
    boundary = tau[..., None, None] * boundary_profile[None, None, :, None] * ydot_b[:, None, None, :]
    shape = torch.einsum("sn,bknd->bksd", aux["H"], w)
    return boundary + shape, tau


def total_turn(steps):
    v1, v2 = steps[..., :-1, :], steps[..., 1:, :]
    n1 = torch.linalg.vector_norm(v1, dim=-1)
    n2 = torch.linalg.vector_norm(v2, dim=-1)
    cos = (v1 * v2).sum(dim=-1) / (n1 * n2).clamp(min=1e-9)
    turn = 1.0 - cos.clamp(-1.0, 1.0)
    mean_step = torch.linalg.vector_norm(steps, dim=-1).mean(dim=-1, keepdim=True)
    gate = (torch.minimum(n1, n2) / (0.25 * mean_step).clamp(min=1e-9)).clamp(max=1.0)
    return (turn * gate).sum(dim=-1)


def surrogate_distance(pos, tau, target_pos, target_tau, cfg, aux):
    endpoint = torch.linalg.vector_norm(pos[..., -1, :] - target_pos[..., -1, :], dim=-1) / 50.0
    path = torch.linalg.vector_norm(pos - target_pos, dim=-1).mean(dim=-1) / 50.0
    ds = 1.0 / float(pos.shape[-2])
    pred_steps = torch.diff(pos, dim=-2, prepend=torch.zeros_like(pos[..., :1, :]))
    tgt_steps = torch.diff(target_pos, dim=-2, prepend=torch.zeros_like(target_pos[..., :1, :]))
    pred_speed = torch.linalg.vector_norm(pred_steps, dim=-1) / (ds * tau[..., None].clamp(min=1.0))
    tgt_speed = torch.linalg.vector_norm(tgt_steps, dim=-1) / (ds * target_tau[..., None].clamp(min=1.0))
    speed = torch.abs(pred_speed - tgt_speed).mean(dim=-1) / 0.35
    duration = torch.abs(tau - target_tau) / 80.0
    j0 = aux["dir_index"]
    u_pred = pos[..., j0, :] / torch.linalg.vector_norm(pos[..., j0, :], dim=-1, keepdim=True).clamp(min=1e-6)
    u_tgt = target_pos[..., j0, :] / torch.linalg.vector_norm(target_pos[..., j0, :], dim=-1, keepdim=True).clamp(min=1e-6)
    direction = torch.linalg.vector_norm(u_pred - u_tgt, dim=-1)
    total_w = cfg.w_endpoint + cfg.w_path + cfg.w_speed + cfg.w_duration + cfg.w_dir
    out = (
        cfg.w_endpoint * endpoint
        + cfg.w_path * path
        + cfg.w_speed * speed
        + cfg.w_duration * duration
        + cfg.w_dir * direction
    )
    if cfg.w_turn_hinge > 0.0:
        if len(cfg.turn_hinge_thresholds) != 3:
            raise ValueError("w_turn_hinge requires 3 turn_hinge_thresholds")
        rate = total_turn(pred_steps) / tau.clamp(min=1.0) * 100.0
        t0, t1, t2 = (float(v) for v in cfg.turn_hinge_thresholds)
        thr = torch.where(tau < 150.0, torch.full_like(tau, t0), torch.where(tau < 250.0, torch.full_like(tau, t1), torch.full_like(tau, t2)))
        hinge = torch.relu(rate - thr) / max(cfg.turn_hinge_norm, 1e-9)
        out = out + cfg.w_turn_hinge * hinge
        total_w = total_w + cfg.w_turn_hinge
    return out / total_w


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    train_arrays,
    val_arrays,
    cfg: PlannerConfig,
    device: str,
    *,
    cut_sampling: str = "all_weighted",
    model_selection_interval: int | None = None,
):
    if cut_sampling not in CUT_SAMPLING_MODES:
        raise ValueError(f"cut_sampling must be one of {CUT_SAMPLING_MODES}")
    if model_selection_interval is None:
        model_selection_interval = int(cfg.epochs)
    if int(model_selection_interval) < 1:
        raise ValueError("model_selection_interval must be positive")
    if int(cfg.epochs) % int(model_selection_interval) != 0:
        raise ValueError(
            "epochs must be divisible by model_selection_interval so the final "
            "epoch is eligible for checkpoint selection"
        )
    seam_contract = planner_seam_contract(train_arrays, val_arrays)
    source_summary = None
    if cut_sampling == "one_per_source_per_epoch":
        source_summary = source_trial_weight_summary(train_arrays)
    train_source_grid_raw = _metadata_text(
        train_arrays,
        "complete_source_grid_thresholds_json",
    )
    val_source_grid_raw = _metadata_text(
        val_arrays,
        "complete_source_grid_thresholds_json",
    )
    if (train_source_grid_raw is None) != (val_source_grid_raw is None):
        raise ValueError(
            "train and validation must both declare the complete source grid"
        )
    complete_source_grid = None
    if train_source_grid_raw is not None:
        try:
            train_source_grid = sorted(
                float(value) for value in json.loads(train_source_grid_raw)
            )
            val_source_grid = sorted(
                float(value) for value in json.loads(val_source_grid_raw)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid complete source grid metadata") from error
        if train_source_grid != val_source_grid or not train_source_grid:
            raise ValueError(
                "train and validation complete source grids must match and be non-empty"
            )
        complete_source_grid = train_source_grid
    prefix_contract = planner_prefix_contract_from_config(cfg)
    train_encoder_inputs, train_boundary_inputs = planner_training_views(
        train_arrays,
        prefix_contract,
        prefix_len=cfg.prefix_len,
    )
    val_encoder_inputs, val_boundary_inputs = planner_training_views(
        val_arrays,
        prefix_contract,
        prefix_len=cfg.prefix_len,
    )
    prodmp = ProDMP(ProDMPConfig(n_basis=cfg.n_basis, alpha=cfg.alpha, alpha_phase=cfg.alpha_phase, ridge=cfg.weight_ridge))

    # summary features (train-fit normalizer)
    train_pack = build_feature_table(
        train_encoder_inputs,
        profile_bins=cfg.profile_bins,
    )
    val_pack = build_feature_table(
        val_encoder_inputs,
        profile_bins=cfg.profile_bins,
    )
    feature_names = train_pack["feature_names"]
    x_train = np.asarray(train_pack["features"], dtype=np.float64)
    tr_weights = event_weights(train_arrays)
    va_weights = event_weights(val_arrays)
    summ_mean, summ_std = weighted_mean_std(x_train, tr_weights)
    summ_std[summ_std < 1e-6] = 1.0

    def summ(pack):
        x = (np.asarray(pack["features"], dtype=np.float64) - summ_mean) / summ_std
        x[~np.isfinite(x)] = 0.0
        return x.astype(np.float32)

    # targets
    tr_labels = fit_weight_labels(
        prodmp,
        train_boundary_inputs,
        eps_scale=cfg.eps_scale,
        ridge=cfg.weight_ridge,
    )
    va_labels = fit_weight_labels(
        prodmp,
        val_boundary_inputs,
        eps_scale=cfg.eps_scale,
        ridge=cfg.weight_ridge,
    )
    y_tr_raw = build_targets(tr_labels)
    y_va_raw = build_targets(va_labels)
    y_mean_1d, y_std_1d = weighted_mean_std(y_tr_raw, tr_weights)
    y_mean = y_mean_1d[None, :]
    y_std = y_std_1d[None, :]
    y_std[y_std < 1e-6] = 1.0

    pmean, pstd = prefix_stats(train_encoder_inputs)
    horizon = int(train_arrays["future_mask"].shape[1])
    tr_prefix, tr_mask = prefix_tensor(
        train_encoder_inputs,
        pmean,
        pstd,
        cfg.prefix_len,
    )
    va_prefix, va_mask = prefix_tensor(
        val_encoder_inputs,
        pmean,
        pstd,
        cfg.prefix_len,
    )

    # normalized-time grid + aux tensors (targets for the surrogate distance)
    s_grid = (np.arange(1, cfg.grid_size + 1, dtype=np.float64)) / float(cfg.grid_size)
    phi, _ = prodmp._basis_at(s_grid)
    aux = {
        "s_grid": torch.as_tensor(s_grid, dtype=torch.float32, device=device),
        "H": torch.as_tensor(phi, dtype=torch.float32, device=device),
        "half_alpha": float(prodmp.alpha) / 2.0,
        "n_weights": int(prodmp.n_weights),
        "horizon": horizon,
        "y_mean": torch.as_tensor(y_mean.reshape(-1), dtype=torch.float32, device=device),
        "y_std": torch.as_tensor(y_std.reshape(-1), dtype=torch.float32, device=device),
        "dir_index": int(np.clip(round(cfg.dir_s * cfg.grid_size) - 1, 0, cfg.grid_size - 1)),
    }
    tr_paths, tr_taus, tr_ydot = grid_targets(
        train_boundary_inputs,
        s_grid,
    )
    va_paths, va_taus, va_ydot = grid_targets(
        val_boundary_inputs,
        s_grid,
    )

    # Duration-conditional p95 of the dimensionless 1-cos turn surrogate,
    # accumulated and normalized to a 100 ms interval.
    tp = torch.as_tensor(tr_paths, dtype=torch.float32, device=device)
    tt = torch.as_tensor(tr_taus, dtype=torch.float32, device=device)
    steps = torch.diff(tp, dim=-2, prepend=torch.zeros_like(tp[..., :1, :]))
    rate = (total_turn(steps) / tt.clamp(min=1.0) * 100.0).cpu().numpy()
    thresholds = []
    for lo, hi in DUR_BINS:
        mask = (tr_taus >= lo) & (tr_taus < hi)
        thresholds.append(
            weighted_percentile(rate[mask], tr_weights[mask], 95.0)
            if bool(np.any(mask))
            else float("inf")
        )
    cfg = PlannerConfig(**{**cfg.__dict__, "turn_hinge_thresholds": tuple(thresholds)})
    print(
        "turn-surrogate thresholds (per 100 ms, by duration bin): "
        f"{[round(t, 3) for t in thresholds]}"
    )

    # tensors
    def tt_(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    tr = {
        "prefix": tt_(tr_prefix), "mask": tt_(tr_mask), "summary": tt_(summ(train_pack)),
        "target": tt_((y_tr_raw - y_mean) / y_std),
        "path": tt_(tr_paths), "tau": tt_(tr_taus), "ydot": tt_(tr_ydot),
        "weight": tt_(tr_weights),
    }
    va = {
        "prefix": tt_(va_prefix), "mask": tt_(va_mask), "summary": tt_(summ(val_pack)),
        "target": tt_((y_va_raw - y_mean) / y_std),
        "path": tt_(va_paths), "tau": tt_(va_taus), "ydot": tt_(va_ydot),
        "weight": tt_(va_weights),
    }

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = MultiHeadModel(tr["summary"].shape[1], y_tr_raw.shape[1], cfg.heads, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)
    anneal_epochs = max(1, int(cfg.wta_anneal_epochs))
    n = tr["target"].shape[0]
    optimizer_examples_per_epoch = (
        int(source_summary["source_trials"])
        if source_summary is not None
        else int(n)
    )

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cut_sampling == "one_per_source_per_epoch":
            perm = torch.as_tensor(
                source_balanced_epoch_indices(
                    train_arrays,
                    epoch=epoch,
                    seed=cfg.seed,
                ),
                dtype=torch.long,
                device=device,
            )
        else:
            perm = torch.randperm(n, generator=gen).to(device)
        eps = cfg.wta_eps_start + (cfg.wta_eps_end - cfg.wta_eps_start) * min(
            1.0, (epoch - 1) / float(anneal_epochs)
        )
        losses = []
        for start in range(0, len(perm), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            y = model(tr["prefix"][idx], tr["mask"][idx], tr["summary"][idx])
            pos, tau = decode_grid(y, tr["ydot"][idx], aux)
            d = surrogate_distance(pos, tau, tr["path"][idx][:, None], tr["tau"][idx][:, None], cfg, aux)
            k = d.shape[1]
            winner = torch.argmin(d.detach(), dim=1)
            q = torch.full_like(d, eps / max(k - 1, 1))
            q.scatter_(1, winner[:, None], 1.0 - eps)
            per_example = (q * d).sum(dim=1)
            batch_weights = (
                torch.ones_like(tr["weight"][idx])
                if cut_sampling == "one_per_source_per_epoch"
                else tr["weight"][idx]
            )
            loss = (per_example * batch_weights).sum() / batch_weights.sum().clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            y = model(va["prefix"], va["mask"], va["summary"])
            pos, tau = decode_grid(y, va["ydot"], aux)
            d = surrogate_distance(pos, tau, va["path"][:, None], va["tau"][:, None], cfg, aux)
            best = d.min(dim=1).values
            val_metric = float(
                (best * va["weight"]).sum() / va["weight"].sum().clamp_min(1e-9)
            )
        selection_epoch = epoch % int(model_selection_interval) == 0
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train {np.mean(losses):.4f}  val best-of-16 {val_metric:.4f}  eps {eps:.3f}")
        if selection_epoch:
            if val_metric < best_val - 1e-6:
                best_val, stale = val_metric, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch} (best val {best_val:.4f})")
                break
    model.load_state_dict(best_state)
    return model, cfg, dict(
        feature_names=feature_names, summ_mean=summ_mean, summ_std=summ_std,
        pmean=pmean, pstd=pstd, y_mean=y_mean, y_std=y_std, horizon=horizon,
        thresholds=thresholds, best_val=best_val,
        train_event_weight_sum=float(tr_weights.sum()),
        val_event_weight_sum=float(va_weights.sum()),
        seam_contract=seam_contract,
        cut_sampling=cut_sampling,
        cut_schedule=(
            SOURCE_BALANCED_CUT_SCHEDULE
            if cut_sampling == "one_per_source_per_epoch"
            else None
        ),
        model_selection_interval=int(model_selection_interval),
        complete_source_grid=complete_source_grid,
        train_source_trial_summary=source_summary,
        optimizer_examples_per_epoch=optimizer_examples_per_epoch,
    )


def export(model, cfg, meta, out_path: Path):
    seam_contract = meta.get("seam_contract")
    if not isinstance(seam_contract, dict):
        raise ValueError("Planner export requires a causal seam contract")
    prefix_contract = planner_prefix_contract_from_config(cfg)
    split_contract = cfg.boundary_prefix_representation is not None
    planner_config = {
        **cfg.__dict__,
        "turn_hinge_thresholds": list(cfg.turn_hinge_thresholds),
    }
    if split_contract:
        planner_config["boundary_prefix_representation"] = (
            prefix_contract.boundary_velocity.name
        )
        planner_config["boundary_prefix_ema_alpha"] = (
            prefix_contract.boundary_velocity.causal_ema_alpha
        )
    else:
        # A one-view export must not contain orphaned split-contract fields.
        planner_config.pop("boundary_prefix_representation", None)
        planner_config.pop("boundary_prefix_ema_alpha", None)
    payload = {
        "schema": (
            SPLIT_PREFIX_PLANNER_SCHEMA
            if split_contract
            else CONTRACTED_PLANNER_SCHEMA
        ),
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": "ABCurves planner (RWTA-16). Trained by training/train_planner.py.",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "planner_config": planner_config,
        "heads": int(cfg.heads),
        "target_dim": int(meta["y_mean"].shape[1]),
        "summary_dim": int(len(meta["feature_names"])),
        "summary_feature_names": list(meta["feature_names"]),
        "summary_mean": torch.as_tensor(meta["summ_mean"], dtype=torch.float32),
        "summary_std": torch.as_tensor(meta["summ_std"], dtype=torch.float32),
        "prefix_mean": torch.as_tensor(meta["pmean"], dtype=torch.float32),
        "prefix_std": torch.as_tensor(meta["pstd"], dtype=torch.float32),
        "y_mean": torch.as_tensor(meta["y_mean"].reshape(-1), dtype=torch.float32),
        "y_std": torch.as_tensor(meta["y_std"].reshape(-1), dtype=torch.float32),
        "horizon": int(meta["horizon"]),
        "hinge_thresholds": meta["thresholds"],
        "train_event_weight_sum": meta["train_event_weight_sum"],
        "val_event_weight_sum": meta["val_event_weight_sum"],
        "cut_sampling": meta.get("cut_sampling", "all_weighted"),
        "cut_schedule": meta.get("cut_schedule"),
        "model_selection_interval": int(meta.get("model_selection_interval", 1)),
        "complete_source_grid": meta.get("complete_source_grid"),
        "train_source_trial_summary": meta.get("train_source_trial_summary"),
        "optimizer_examples_per_epoch": meta.get("optimizer_examples_per_epoch"),
        "seam_contract": seam_contract,
        "train_dataset": meta.get("train_dataset"),
        "train_dataset_sha256": meta.get("train_dataset_sha256"),
        "val_dataset": meta.get("val_dataset"),
        "val_dataset_sha256": meta.get("val_dataset_sha256"),
        "prodmp": {"n_basis": cfg.n_basis, "alpha": cfg.alpha, "alpha_phase": cfg.alpha_phase, "ridge": cfg.weight_ridge},
    }
    if split_contract:
        payload["planner_prefix_contract"] = prefix_contract.metadata()
    else:
        payload["prefix_representation"] = prefix_contract.encoder.metadata()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train",
        required=True,
        help="prepared planner_train.npz emitted by tools/prepare_dataset.py",
    )
    ap.add_argument(
        "--val",
        required=True,
        help="prepared planner_val.npz emitted by tools/prepare_dataset.py",
    )
    ap.add_argument("--out", default="runs/planner_retrained.pt")
    ap.add_argument("--epochs", type=int, default=260)
    ap.add_argument(
        "--wta-anneal-epochs",
        type=int,
        default=45,
        help="intervals used to anneal RWTA epsilon from 0.50 to 0.05",
    )
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--prefix-representation",
        choices=sorted(SUPPORTED_PREFIX_REPRESENTATIONS),
        default=RAW_PREFIX_REPRESENTATION,
        help=(
            "planner encoder/summary representation (default: raw); unless "
            "--boundary-prefix-representation is provided, it also controls "
            "the ProDMP boundary"
        ),
    )
    ap.add_argument(
        "--prefix-ema-alpha",
        type=float,
        default=DEFAULT_PREFIX_EMA_ALPHA,
        help="causal EMA alpha for the encoder/summary representation",
    )
    ap.add_argument(
        "--boundary-prefix-representation",
        choices=sorted(SUPPORTED_PREFIX_REPRESENTATIONS),
        default=None,
        help=(
            "explicit ProDMP boundary-velocity view; omitted maps the encoder "
            "view to the boundary"
        ),
    )
    ap.add_argument(
        "--boundary-prefix-ema-alpha",
        type=float,
        default=None,
        help=(
            "causal EMA alpha for an explicit boundary view "
            f"(default when needed: {DEFAULT_PREFIX_EMA_ALPHA:g})"
        ),
    )
    ap.add_argument(
        "--cut-sampling",
        choices=CUT_SAMPLING_MODES,
        default="one_per_source_per_epoch",
        help=(
            "all_weighted visits every materialized cut; "
            "one_per_source_per_epoch samples one available B cut per source "
            "trial so multi-B arms have baseline-comparable optimizer steps"
        ),
    )
    ap.add_argument(
        "--model-selection-interval",
        type=int,
        default=0,
        help=(
            "only epochs divisible by this interval may become the exported "
            "checkpoint; 0 means terminal epoch only (the release recipe)"
        ),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing planner: {out_path}")
    train_arrays = load_dataset(train_path, success_only=True)
    val_arrays = load_dataset(val_path, success_only=True)
    print(f"train {len(train_arrays['future_mask'])} events, val {len(val_arrays['future_mask'])} events, device {args.device}")

    cfg = PlannerConfig(
        epochs=args.epochs,
        heads=args.heads,
        seed=args.seed,
        wta_anneal_epochs=args.wta_anneal_epochs,
        prefix_representation=args.prefix_representation,
        prefix_ema_alpha=args.prefix_ema_alpha,
        boundary_prefix_representation=args.boundary_prefix_representation,
        boundary_prefix_ema_alpha=args.boundary_prefix_ema_alpha,
    )
    t0 = time.time()
    model, cfg, meta = train(
        train_arrays,
        val_arrays,
        cfg,
        args.device,
        cut_sampling=args.cut_sampling,
        model_selection_interval=(
            args.epochs if args.model_selection_interval == 0
            else args.model_selection_interval
        ),
    )
    meta.update(
        {
            "train_dataset": str(train_path.resolve()),
            "train_dataset_sha256": _sha256(train_path),
            "val_dataset": str(val_path.resolve()),
            "val_dataset_sha256": _sha256(val_path),
        }
    )
    export(model, cfg, meta, out_path)
    print(f"wrote {args.out}  (best val best-of-16 {meta['best_val']:.4f}, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
