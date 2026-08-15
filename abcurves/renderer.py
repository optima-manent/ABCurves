"""Float training reference for the global count-texture Renderer.

The default production Renderer is the authenticated compact C artifact exposed
by ``abcurves.portable_renderer``.  This module documents and trains the learned
law behind it, and exposes the direct float-checkpoint backend used for retraining:
a phase-free H80 GRU that can texture smooth movement in the learned 1 kHz
count-space regimes, not a model tied to a B-to-C event crop.
Training examples are blind windows
from dense 1 ms physical sessions: 256 observed reports followed by 800 future
reports.  That includes ordinary motion before A, between events and after C.

The future is smoothed with either the w3 or w5 teacher, and the raw future is
the self-supervised target.  Teacher-forced loss is useful diagnostics, but it
does not choose a release checkpoint; promotion is based on sampled texture
with state carried over complete held-out sessions.

How it works
------------
A tiny **GRU** wrapped around a **hysteretic delta-sigma accumulator**:

* the accumulator integrates the smooth intent and only "releases" an integer
  count once enough movement mass has built up -- this reproduces the bursty
  zero/nonzero cadence and, because the accumulator reclaims whatever it
  emits, keeps residual endpoint debt small and bounded rather than letting
  independent rounding wander;
* per tick, the GRU consumes causal features (smooth kinematics, accumulator
  state in the movement frame, previous emission, run state, recent zero rate,
  and a summary of the observed 256-report count regime) and samples: an
  ``emit`` decision (report now?) and a small integer ``offset`` around the
  accumulator's rounded base;
* the observed context warms the recurrent state before any generated report.

The float graph stores one recurrent matrix set, for 34,362 learned scalars.
The selected runtime then post-training-quantizes the hot recurrence and adds a
small rank-16 prefix handoff. That handoff compresses observer state; it does
not encode a user identity.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from threading import Lock
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .portable_renderer import (
    CONTEXT_TICKS,
    LATERAL_OFFSET_PENALTY,
    RendererRuntimeError,
    validate_physical_context,
    validate_smooth_future,
)
from .smoothing import parse_smoothing_spec, smooth_dxdy

EPS = 1e-9
RENDERER_SCHEMA = "abcurves.renderer_global_float.v2"
SUPPORTED_RENDERER_SCHEMAS = {RENDERER_SCHEMA}
RECENT_WINDOW = 60

BASE_FEATURE_NAMES = (
    "speed_scaled",
    "accel",
    "curvature",
    "tangent_x",
    "tangent_y",
    "acc_tangent",
    "acc_normal",
    "prev_emit_tangent",
    "prev_emit_normal",
    "run_active",
    "run_length_norm",
    "recent_zero_rate",
    "log1p_speed",
    "last_nonzero_emit_tangent",
    "last_nonzero_emit_normal",
)
REGIME_FEATURE_NAMES = (
    "regime_active_rate",
    "regime_log1p_active_mag_mean",
    "regime_log1p_active_mag_p95",
    "regime_sign_flip_rate",
    "regime_hf_power_log1p",
)


@dataclass(frozen=True)
class RendererConfig:
    hidden: int = 80
    offset_radius: int = 5
    context_ticks: int = 256
    recurrent_warm_ticks: int = 128
    future_ticks: int = 800
    presentation_budget: int = 118_345
    batch_size: int = 256
    sampler: str = "frozen_epoch_view_sampler_v1"
    deterministic_algorithms: bool = True
    lr: float = 2e-3
    weight_decay: float = 1e-5
    seed: int = 7
    temperature: float = 1.3  # emit head temperature (cadence)
    offset_temperature: float = 1.0  # fallback joint temperature
    offset_magnitude_temperature: float | None = 0.75
    offset_direction_temperature: float | None = 0.15
    teacher_base_hysteresis: float = 1.0
    base_hysteresis: float = 0.5  # deadband against accumulator sign crossings
    force_release: float = 32.0  # |accumulator| safety release (drift guard)
    max_abs_count: int = 127
    prefix_smoothing_spec: str | None = None
    zero_intent_gate: bool = True
    zero_intent_threshold: float = 1e-7
    zero_accumulator_threshold: float = 0.5
    emit_logit_bias: float = 1.5
    regime_window: int = 256
    prefix_loss_weight: float = 0.0
    regime_feature_names: tuple[str, ...] = REGIME_FEATURE_NAMES
    smooth_methods: tuple[str, ...] = (
        "triangular_moving_average_path:window=3",
        "triangular_moving_average_path:window=5",
    )


def _feature_names_for_config(config: RendererConfig) -> tuple[str, ...]:
    return BASE_FEATURE_NAMES + tuple(config.regime_feature_names)


def _schema_for_config(config: RendererConfig) -> str:
    _feature_names_for_config(config)
    return RENDERER_SCHEMA


def _seed_training(seed: int) -> None:
    """Bind model initialization and kernels to the promoted run's seed rule."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _epoch_view_schedule(
    windows: int, smoothing_views: int, seed: int, epoch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen per-epoch source order and one teacher per source."""

    generator = np.random.default_rng([int(seed), int(epoch)])
    choices = generator.integers(0, int(smoothing_views), size=int(windows), dtype=np.int64)
    order = generator.permutation(int(windows))
    return order, choices[order]


def renderer_config_from_dict(raw: dict[str, Any]) -> RendererConfig:
    known = {f.name for f in fields(RendererConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    for k in ("smooth_methods", "regime_feature_names"):
        if k in kwargs and kwargs[k] is not None:
            kwargs[k] = tuple(kwargs[k])
    return RendererConfig(**kwargs)


class CountTextureModel(torch.nn.Module):
    """GRU with an emit head and a joint (x, y) integer-offset head."""

    def __init__(self, config: RendererConfig = RendererConfig()) -> None:
        super().__init__()
        self.config = config
        self.feature_names = _feature_names_for_config(config)
        self.n_features = len(self.feature_names)
        self.regime_feature_names = tuple(config.regime_feature_names)
        k = 2 * config.offset_radius + 1
        self.gru = torch.nn.GRU(self.n_features, config.hidden, batch_first=True)
        # The frozen seed-7 initialization law consumes one GRUCell-shaped
        # random draw between the GRU and the two heads. Constructing and
        # immediately discarding it preserves the selected head initialization
        # without adding an inactive duplicate weight set to checkpoints.
        _frozen_initialization_draw = torch.nn.GRUCell(
            self.n_features, config.hidden
        )
        del _frozen_initialization_draw
        self.head_emit = torch.nn.Linear(config.hidden, 1)
        self.head_off = torch.nn.Linear(config.hidden, k * k)

    def step_cell(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Apply one GRU tick without registering a duplicate ``GRUCell``."""

        input_gates = torch.nn.functional.linear(
            x, self.gru.weight_ih_l0, self.gru.bias_ih_l0
        )
        hidden_gates = torch.nn.functional.linear(
            hidden, self.gru.weight_hh_l0, self.gru.bias_hh_l0
        )
        input_reset, input_update, input_new = input_gates.chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        return candidate + update * (hidden - candidate)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out, hn = self.gru(x, h0)
        return {
            "emit_logit": self.head_emit(out).squeeze(-1),
            "off_logits": self.head_off(out),
            "hidden": hn,
        }


# ---------------------------------------------------------------------------
# Causal feature helpers (shared by training-time teacher forcing and sampling)
# ---------------------------------------------------------------------------
def _ffill_directions(smooth: np.ndarray, seed_dir: np.ndarray) -> np.ndarray:
    n = len(smooth)
    dirs = np.zeros((n, 2), dtype=np.float64)
    speed = np.linalg.norm(smooth, axis=1)
    last = np.asarray(seed_dir, dtype=np.float64)
    if float(np.linalg.norm(last)) <= EPS:
        last = np.asarray([1.0, 0.0])
    for t in range(n):
        if speed[t] > EPS:
            last = smooth[t]
        dirs[t] = last
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms <= EPS] = 1.0
    return dirs / norms


def _kinematics(
    smooth: np.ndarray,
    left_context: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.linalg.norm(smooth, axis=1)
    accel = np.diff(speed, prepend=speed[:1])
    n = len(smooth)
    curvature = np.zeros(n, dtype=np.float64)
    if n and left_context is not None:
        left = np.asarray(left_context, dtype=np.float64).reshape(2)
        left_speed = float(np.linalg.norm(left))
        accel[0] = speed[0] - left_speed
        if left_speed > EPS and speed[0] > EPS:
            cosine = float(np.clip(np.dot(left, smooth[0]) / (left_speed * speed[0]), -1.0, 1.0))
            curvature[0] = 1.0 - cosine
    if n > 1:
        a = smooth[:-1]
        b = smooth[1:]
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        ok = (na > EPS) & (nb > EPS)
        cos = np.zeros(n - 1)
        cos[ok] = np.clip(np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok]), -1.0, 1.0)
        curvature[1:] = np.where(ok, 1.0 - cos, 0.0)
    return speed, accel, curvature


def _run_state_before(active: np.ndarray, init_active: bool, init_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(active)
    run_active = np.zeros(n, dtype=np.float64)
    run_len = np.zeros(n, dtype=np.float64)
    ra, rl = bool(init_active), int(init_len)
    for t in range(n):
        run_active[t] = 1.0 if ra else 0.0
        run_len[t] = rl
        if t == 0 and rl == 0:
            ra, rl = bool(active[t]), 1
        elif bool(active[t]) != ra:
            ra, rl = bool(active[t]), 1
        else:
            rl += 1
    return run_active, run_len


def _recent_zero_rate_excl(active: np.ndarray, init_zero_rate: float) -> np.ndarray:
    n = len(active)
    out = np.zeros(n, dtype=np.float64)
    inact = (~active.astype(bool)).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(inact)])
    for t in range(n):
        lo = max(0, t - RECENT_WINDOW)
        out[t] = init_zero_rate if t == 0 else (cs[t] - cs[lo]) / float(t - lo)
    return out


def _sign_flip_rate(dxdy: np.ndarray) -> float:
    flips = 0
    denom = 0
    for axis in range(2):
        values = dxdy[:, axis]
        signs = np.sign(values[np.abs(values) > 0.0])
        if len(signs) > 1:
            flips += int(np.sum(signs[1:] != signs[:-1]))
            denom += int(len(signs) - 1)
    return float(flips / denom) if denom else 0.0


def _prefix_from_arrays(arrays: dict[str, np.ndarray], idx: int) -> np.ndarray:
    prefix = arrays["prefix_raw_dxdy"][idx].astype(np.float64)
    pmask = arrays.get("prefix_mask")
    if pmask is not None:
        prefix = prefix[pmask[idx] > 0.5]
    return prefix[np.isfinite(prefix).all(axis=1)]


def _truncate_prefix(prefix: np.ndarray, prefix_ticks: int) -> np.ndarray:
    ticks = max(int(prefix_ticks), 0)
    if ticks == 0:
        return prefix[:0]
    return prefix[-ticks:] if len(prefix) > ticks else prefix


def _regime_features_from_prefix(prefix: np.ndarray, *, window: int) -> np.ndarray:
    prefix = np.asarray(prefix, dtype=np.float64)
    if prefix.ndim != 2 or prefix.shape[1] != 2 or len(prefix) == 0:
        return np.zeros(len(REGIME_FEATURE_NAMES), dtype=np.float32)
    w = max(int(window), 1)
    recent = prefix[-w:]
    mag = np.max(np.abs(recent), axis=1)
    active = mag > 0.0
    active_mag = mag[active]
    active_rate = float(np.mean(active))
    mean_mag = float(np.mean(active_mag)) if len(active_mag) else 0.0
    p95_mag = float(np.percentile(active_mag, 95)) if len(active_mag) else 0.0
    sign_flip = _sign_flip_rate(recent)
    if len(mag) >= 8:
        x = mag - float(np.mean(mag))
        power = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(mag), d=1e-3)
        hf = float(np.sum(power[(freqs >= 30.0) & (freqs < 500.0)])) / float(len(mag))
    else:
        hf = 0.0
    out = np.asarray(
        [active_rate, np.log1p(mean_mag), np.log1p(p95_mag), sign_flip, np.log1p(hf)],
        dtype=np.float32,
    )
    out[~np.isfinite(out)] = 0.0
    return out


def _regime_features(prefix: np.ndarray, names: tuple[str, ...], *, window: int) -> np.ndarray:
    base = _regime_features_from_prefix(prefix, window=window)
    if tuple(names) == REGIME_FEATURE_NAMES:
        return base
    out = np.zeros(len(names), dtype=np.float32)
    base_by_name = dict(zip(REGIME_FEATURE_NAMES, base))
    for j, name in enumerate(names):
        out[j] = float(base_by_name.get(str(name), 0.0))
    return out


def _last_axis_nonzero_before(raw: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    out = np.zeros_like(raw, dtype=np.float64)
    cur = np.zeros(2, dtype=np.float64) if seed is None else np.asarray(seed, dtype=np.float64).copy()
    for t in range(len(raw)):
        out[t] = cur
        nz = raw[t] != 0.0
        cur[nz] = raw[t, nz]
    return out


def _hysteretic_base_np(acc: np.ndarray, last_axis_nonzero: np.ndarray, threshold: float) -> np.ndarray:
    base = np.rint(np.asarray(acc, dtype=np.float64))
    threshold = float(threshold)
    if threshold <= 0.0:
        return base
    last_sign = np.sign(np.asarray(last_axis_nonzero, dtype=np.float64))
    base_sign = np.sign(base)
    crosses = (last_sign != 0.0) & (base_sign != 0.0) & (last_sign * base_sign < 0.0)
    blocked = crosses & (np.abs(acc) < threshold)
    return np.where(blocked, 0.0, base)


def _hysteretic_base_torch(acc: torch.Tensor, last_axis_nonzero: torch.Tensor, threshold: float) -> torch.Tensor:
    base = torch.round(acc)
    threshold = float(threshold)
    if threshold <= 0.0:
        return base
    last_sign = torch.sign(last_axis_nonzero)
    base_sign = torch.sign(base)
    crosses = (last_sign != 0.0) & (base_sign != 0.0) & (last_sign * base_sign < 0.0)
    blocked = crosses & (acc.abs() < threshold)
    return torch.where(blocked, torch.zeros_like(base), base)


def _effective_temperature_any(primary, fallback: float):
    if primary is None:
        return max(float(fallback), 1e-3)
    if isinstance(primary, torch.Tensor):
        return primary.clamp_min(1e-3).reshape(-1, 1)
    return max(float(primary), 1e-3)


def _compose_offset_logits(
    logits: torch.Tensor,
    offset_magnitude: torch.Tensor,
    *,
    joint_temperature: float,
    magnitude_temperature=None,
    direction_temperature=None,
) -> torch.Tensor:
    """Apply separable magnitude/direction temperatures to the joint offset head."""

    tau_joint = max(float(joint_temperature), 1e-3)
    tau_mag = _effective_temperature_any(magnitude_temperature, tau_joint)
    tau_dir = _effective_temperature_any(direction_temperature, tau_joint)
    if (
        not isinstance(tau_mag, torch.Tensor)
        and not isinstance(tau_dir, torch.Tensor)
        and abs(tau_mag - tau_joint) < 1e-9
        and abs(tau_dir - tau_joint) < 1e-9
    ):
        return logits / tau_joint
    out = torch.empty_like(logits)
    for mag in torch.unique(offset_magnitude):
        ring = offset_magnitude == mag
        ring_logits = logits[:, ring]
        ring_lse = torch.logsumexp(ring_logits, dim=1, keepdim=True)
        out[:, ring] = ring_lse / tau_mag + (ring_logits - ring_lse) / tau_dir
    return out


# ---------------------------------------------------------------------------
# Training-time teacher-forced example construction
# ---------------------------------------------------------------------------
def build_teacher_example(
    smooth: np.ndarray,
    raw: np.ndarray,
    *,
    reset_idx: int,
    offset_radius: int,
    seed_dir: np.ndarray | None = None,
    seed_last_nonzero: np.ndarray | None = None,
    seed_last_axis_nonzero: np.ndarray | None = None,
    regime_features: np.ndarray | None = None,
    regime_feature_names: tuple[str, ...] = REGIME_FEATURE_NAMES,
    prefix_loss_weight: float = 0.0,
    base_hysteresis: float = 0.0,
) -> dict[str, np.ndarray]:
    """Teacher-forced features/labels over a ``[prefix; future]`` stream.

    ``reset_idx`` marks B: the delta-sigma accumulator resets there (the prefix
    part only warms the recurrent state; loss is masked to ``t >= reset_idx``
    unless ``prefix_loss_weight`` supervises it too).
    """

    smooth = np.asarray(smooth, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    n = len(smooth)
    if raw.shape != smooth.shape:
        raise ValueError("smooth and raw must have matching (T, 2) shapes")
    speed, accel, curvature = _kinematics(smooth)
    dirs = _ffill_directions(smooth, seed_dir if seed_dir is not None else np.asarray([1.0, 0.0]))
    tangent = dirs
    normal = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

    desired = np.zeros((n, 2), dtype=np.float64)
    for lo, hi in ((0, reset_idx), (reset_idx, n)):
        if hi <= lo:
            continue
        cs = np.cumsum(smooth[lo:hi], axis=0)
        cr = np.cumsum(raw[lo:hi], axis=0) - raw[lo:hi]
        desired[lo:hi] = cs - cr

    active = np.any(raw != 0.0, axis=1)
    last_axis_nz = _last_axis_nonzero_before(
        raw, seed_last_axis_nonzero if seed_last_axis_nonzero is not None else seed_last_nonzero
    )
    base = _hysteretic_base_np(desired, last_axis_nz, base_hysteresis)
    off = raw - base
    valid_off = active & np.all(np.abs(off) <= offset_radius, axis=1)

    prev_emit = np.zeros((n, 2), dtype=np.float64)
    prev_emit[1:] = raw[:-1]
    last_nz = np.zeros((n, 2), dtype=np.float64)
    cur_nz = np.zeros(2) if seed_last_nonzero is None else np.asarray(seed_last_nonzero, dtype=np.float64)
    for t in range(n):
        last_nz[t] = cur_nz
        if active[t]:
            cur_nz = raw[t]
    run_active, run_len = _run_state_before(active, False, 0)
    rz = _recent_zero_rate_excl(active, 1.0)

    acc_t = np.sum(desired * tangent, axis=1)
    acc_n = np.sum(desired * normal, axis=1)
    prev_t = np.sum(prev_emit * tangent, axis=1)
    prev_n = np.sum(prev_emit * normal, axis=1)
    lnz_t = np.sum(last_nz * tangent, axis=1)
    lnz_n = np.sum(last_nz * normal, axis=1)
    regime_names = tuple(regime_feature_names)
    regime = np.zeros(len(regime_names), dtype=np.float32)
    if regime_features is not None:
        supplied = np.asarray(regime_features, dtype=np.float32).reshape(-1)
        if len(supplied) != len(regime_names):
            raise ValueError(f"regime_features must have {len(regime_names)} values")
        regime[:] = supplied
    regime_cols = np.tile(regime[None, :], (n, 1))

    base_columns = [
            speed / 4.0,
            np.clip(accel, -3.0, 3.0),
            curvature,
            tangent[:, 0],
            tangent[:, 1],
            np.clip(acc_t, -4.0, 4.0),
            np.clip(acc_n, -4.0, 4.0),
            np.clip(prev_t / 4.0, -4.0, 4.0),
            np.clip(prev_n / 4.0, -4.0, 4.0),
            run_active,
            np.minimum(run_len, 64.0) / 64.0,
            rz,
            np.log1p(speed),
            np.clip(lnz_t / 4.0, -4.0, 4.0),
            np.clip(lnz_n / 4.0, -4.0, 4.0),
        ]
    feats = np.stack(base_columns, axis=1).astype(np.float32)
    feats = np.concatenate([feats, regime_cols], axis=1).astype(np.float32)

    k = 2 * offset_radius + 1
    off_clipped = np.clip(off, -offset_radius, offset_radius).astype(np.int64) + offset_radius
    loss_mask = np.zeros(n, dtype=np.float32)
    if reset_idx > 0 and float(prefix_loss_weight) > 0.0:
        loss_mask[:reset_idx] = float(prefix_loss_weight)
    loss_mask[reset_idx:] = 1.0
    return {
        "features": feats,
        "emit": active.astype(np.float32),
        "off_joint": off_clipped[:, 0] * k + off_clipped[:, 1],
        "valid_off": valid_off.astype(np.float32),
        "loss_mask": loss_mask,
        "clipped_off": (active & ~valid_off).astype(np.float32),
    }


def _example_streams(
    arrays: dict[str, np.ndarray], idx: int, spec_key: str, *, config: RendererConfig
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray] | None:
    if "future_mask" in arrays:
        mask = arrays["future_mask"][idx].astype(bool)
        n = int(np.sum(mask))
    else:
        n = int(arrays["future_raw_dxdy"].shape[1])
    if n < 4:
        return None
    raw_f = arrays["future_raw_dxdy"][idx, :n].astype(np.float64)
    prefix_full = _prefix_from_arrays(arrays, idx)
    regime = _regime_features(prefix_full, tuple(config.regime_feature_names), window=config.regime_window)
    prefix = _truncate_prefix(prefix_full, config.recurrent_warm_ticks)
    raw = np.concatenate([prefix, raw_f], axis=0)
    # Deployment warms the recurrent state from the observed prefix alone, so
    # its smoothed prefix features must not use future samples.  The planner's
    # future target, however, is smoothed with its preceding trajectory context.
    # Preserve both facts: causal/available prefix warm-up plus the future side
    # of a context-aware joint smoothing pass.  Any remaining B mismatch is the
    # real deployment seam the renderer must handle, not a leak to hide.
    future_spec = parse_smoothing_spec(spec_key)
    prefix_spec = parse_smoothing_spec(config.prefix_smoothing_spec or spec_key)
    joint_smooth = smooth_dxdy(
        raw.astype(np.float32),
        spec=future_spec,
    ).astype(np.float64)
    if len(prefix):
        smooth_prefix = smooth_dxdy(
            prefix.astype(np.float32),
            spec=prefix_spec,
        ).astype(np.float64)
    else:
        smooth_prefix = np.zeros((0, 2), dtype=np.float64)
    smooth = np.concatenate(
        [smooth_prefix, joint_smooth[len(prefix) :]],
        axis=0,
    )
    return smooth, raw, len(prefix), regime


def _stack_examples(
    examples: list[dict[str, np.ndarray]], *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Stack one fixed-size global-window batch without materializing the corpus."""

    if not examples:
        raise ValueError("cannot stack an empty Renderer batch")
    keys = ("features", "emit", "off_joint", "valid_off", "loss_mask")
    arrays = {key: np.stack([row[key] for row in examples], axis=0) for key in keys}
    return {
        "features": torch.from_numpy(arrays["features"]).to(device),
        "emit": torch.from_numpy(arrays["emit"]).to(device),
        "off_joint": torch.from_numpy(arrays["off_joint"]).to(device),
        "valid_off": torch.from_numpy(arrays["valid_off"]).to(device),
        "loss_mask": torch.from_numpy(arrays["loss_mask"]).to(device),
    }


def _require_global_arrays(
    arrays: dict[str, np.ndarray], config: RendererConfig
) -> tuple[np.ndarray, np.ndarray]:
    try:
        prefix = arrays["prefix_raw_dxdy"]
        future = arrays["future_raw_dxdy"]
    except KeyError as exc:
        raise ValueError(
            "global Renderer data needs prefix_raw_dxdy and future_raw_dxdy"
        ) from exc
    expected_prefix = (len(prefix), config.context_ticks, 2)
    expected_future = (len(prefix), config.future_ticks, 2)
    if tuple(prefix.shape) != expected_prefix:
        raise ValueError(f"prefix_raw_dxdy must have shape {expected_prefix}")
    if tuple(future.shape) != expected_future:
        raise ValueError(f"future_raw_dxdy must have shape {expected_future}")
    if len(prefix) < 1:
        raise ValueError("global Renderer data is empty")
    if not np.issubdtype(prefix.dtype, np.number) or not np.issubdtype(
        future.dtype, np.number
    ):
        raise ValueError("global Renderer reports must be numeric")
    return prefix, future


def _teacher_row(
    arrays: dict[str, np.ndarray],
    index: int,
    spec_key: str,
    config: RendererConfig,
) -> dict[str, np.ndarray]:
    streams = _example_streams(arrays, int(index), spec_key, config=config)
    if streams is None:
        raise ValueError(f"window {index} has no usable future")
    smooth, raw, reset_idx, regime = streams
    return build_teacher_example(
        smooth,
        raw,
        reset_idx=reset_idx,
        offset_radius=config.offset_radius,
        regime_features=regime,
        regime_feature_names=tuple(config.regime_feature_names),
        prefix_loss_weight=0.0,
        base_hysteresis=config.teacher_base_hysteresis,
    )


def _loss_terms(
    model: CountTextureModel, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(batch["features"])
    mask = batch["loss_mask"]
    emit = torch.nn.functional.binary_cross_entropy_with_logits(
        output["emit_logit"], batch["emit"], reduction="none"
    )
    emit_loss = (emit * mask).sum() / mask.sum().clamp_min(1.0)
    offset_mask = mask * batch["valid_off"]
    offset = torch.nn.functional.cross_entropy(
        output["off_logits"].flatten(0, 1),
        batch["off_joint"].flatten(),
        reduction="none",
    ).view_as(offset_mask)
    offset_loss = (offset * offset_mask).sum() / offset_mask.sum().clamp_min(1.0)
    return emit_loss + offset_loss, emit_loss, offset_loss


def _loss_sums(
    model: CountTextureModel, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact numerators/denominators for partition-invariant reporting."""

    output = model(batch["features"])
    mask = batch["loss_mask"]
    emit_values = torch.nn.functional.binary_cross_entropy_with_logits(
        output["emit_logit"], batch["emit"], reduction="none"
    )
    offset_mask = mask * batch["valid_off"]
    offset_values = torch.nn.functional.cross_entropy(
        output["off_logits"].flatten(0, 1),
        batch["off_joint"].flatten(),
        reduction="none",
    ).view_as(offset_mask)
    return (
        (emit_values * mask).sum(),
        mask.sum(),
        (offset_values * offset_mask).sum(),
        offset_mask.sum(),
    )


def teacher_forced_loss(
    model: CountTextureModel,
    arrays: dict[str, np.ndarray],
    *,
    spec_key: str = "triangular_moving_average_path:window=5",
    device: str | torch.device = "cpu",
    max_presentations: int | None = None,
) -> dict[str, float]:
    """Measure diagnostic teacher-forced loss without selecting a checkpoint."""

    config = model.config
    prefix, _ = _require_global_arrays(arrays, config)
    count = len(prefix)
    if max_presentations is not None:
        count = min(count, max(int(max_presentations), 0))
    if count < 1:
        raise ValueError("teacher-forced evaluation needs at least one presentation")
    target = torch.device(device)
    model = model.to(target).eval()
    totals = np.zeros(4, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, count, config.batch_size):
            indices = range(start, min(start + config.batch_size, count))
            batch = _stack_examples(
                [_teacher_row(arrays, index, spec_key, config) for index in indices],
                device=target,
            )
            sums = _loss_sums(model, batch)
            totals += [float(value) for value in sums]
    emit_loss = totals[0] / max(totals[1], 1.0)
    offset_loss = totals[2] / max(totals[3], 1.0)
    return {
        "loss": float(emit_loss + offset_loss),
        "emit_bce": float(emit_loss),
        "offset_ce": float(offset_loss),
        "presentations": int(count),
        "smoothing": str(spec_key),
        "checkpoint_selection": False,
    }


def train_count_texture_model(
    arrays: dict[str, np.ndarray],
    config: RendererConfig = RendererConfig(),
    device: str | torch.device = "cpu",
    log_every: int = 25,
) -> tuple[CountTextureModel, dict[str, Any]]:
    """Train to an exact presentation budget using natural window weighting.

    A presentation is one source window shown once under one randomly chosen
    w3/w5 teacher.  The counter, rather than an epoch label, is the transferable
    training budget.  For the selected corpus, 118,345 presentations means one
    complete pass over 81,737 windows plus 36,608 windows of the next pass.
    """

    prefix, _ = _require_global_arrays(arrays, config)
    if config.hidden != 80 or config.offset_radius != 5:
        raise ValueError("the final global architecture is H80 with radius 5")
    if config.prefix_loss_weight != 0.0:
        raise ValueError("the final global law is phase-free with future-only loss")
    if config.teacher_base_hysteresis != 1.0:
        raise ValueError("the frozen training teacher uses base hysteresis 1.0")
    if config.base_hysteresis != 0.5:
        raise ValueError("the selected deployment calibration uses hysteresis 0.5")
    budget = int(config.presentation_budget)
    if budget < 1:
        raise ValueError("presentation_budget must be positive")
    methods = tuple(config.smooth_methods)
    if methods != (
        "triangular_moving_average_path:window=3",
        "triangular_moving_average_path:window=5",
    ):
        raise ValueError("the final training teachers are exactly w3 and w5")
    if config.sampler != "frozen_epoch_view_sampler_v1":
        raise ValueError("the final training route uses the frozen epoch-view sampler")
    if config.deterministic_algorithms is not True:
        raise ValueError("the final training route enables deterministic algorithms")

    _seed_training(config.seed)
    target = torch.device(device)
    model = CountTextureModel(config).to(target)
    learned = sum(parameter.numel() for parameter in model.parameters())
    if learned != 34_362:
        raise RuntimeError(f"global Renderer architecture has {learned} scalars, expected 34362")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    n_windows = len(prefix)
    epoch = 0
    order, epoch_views = _epoch_view_schedule(
        n_windows, len(methods), config.seed, epoch
    )
    cursor = 0
    presentations = 0
    optimizer_steps = 0
    history: list[dict[str, float | int]] = []
    clip_total = 0.0
    active_total = 0.0
    model.train()

    while presentations < budget:
        # Epoch boundaries are optimizer boundaries. The promoted run used a
        # 73-row final batch at the end of P0 before beginning epoch two; never
        # fill that batch with rows from the next permutation.
        batch_count = min(
            config.batch_size,
            budget - presentations,
            n_windows - cursor,
        )
        chosen = order[cursor : cursor + batch_count]
        views = epoch_views[cursor : cursor + batch_count]
        examples = [
            _teacher_row(arrays, index, methods[int(view)], config)
            for index, view in zip(chosen, views)
        ]
        for row in examples:
            mask = row["loss_mask"]
            clip_total += float(np.sum(row["clipped_off"] * mask))
            active_total += float(np.sum(row["emit"] * mask))
        batch = _stack_examples(examples, device=target)
        loss, emit_loss, offset_loss = _loss_terms(model, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        presentations += batch_count
        optimizer_steps += 1
        cursor += batch_count
        if cursor == n_windows:
            epoch += 1
            cursor = 0
            if presentations < budget:
                order, epoch_views = _epoch_view_schedule(
                    n_windows, len(methods), config.seed, epoch
                )
        if (
            optimizer_steps == 1
            or presentations == budget
            or (log_every and optimizer_steps % int(log_every) == 0)
        ):
            record: dict[str, float | int] = {
                "optimizer_step": optimizer_steps,
                "presentations": presentations,
                "passes": presentations / float(n_windows),
                "loss": float(loss.detach()),
                "emit_bce": float(emit_loss.detach()),
                "offset_ce": float(offset_loss.detach()),
            }
            history.append(record)
            if log_every:
                print(
                    "  renderer "
                    f"p{presentations}: loss {record['loss']:.4f} "
                    f"(emit {record['emit_bce']:.4f}, off {record['offset_ce']:.4f})"
                )

    model.eval()
    report = {
        "schema": _schema_for_config(config),
        "config": asdict(config),
        "feature_names": list(model.feature_names),
        "n_features": int(model.n_features),
        "architecture": "phase_free_h80_gru_emit_joint_offset_r5",
        "initialization_rng": "frozen_gru_then_discarded_grucell_then_heads",
        "training_smooth_methods": list(methods),
        "source_windows": int(n_windows),
        "presentations": int(presentations),
        "optimizer_steps": int(optimizer_steps),
        "passes": float(presentations / n_windows),
        "completed_full_passes": int(epoch),
        "next_epoch_offset": int(cursor),
        "sampler": config.sampler,
        "deterministic_algorithms": config.deterministic_algorithms,
        "natural_window_weighting": True,
        "teacher_offset_clip_fraction": float(clip_total / max(active_total, 1.0)),
        "n_parameters": int(learned),
        "history": history,
        "selection": {
            "performed": False,
            "reason": (
                "teacher-forced loss is diagnostic; promote only after sampled "
                "texture evaluation with state carried across held-out sessions"
            ),
        },
    }
    return model, report


def save_count_model(
    model: CountTextureModel,
    report: dict[str, Any],
    path: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically save a float checkpoint.

    Programmatic callers retain the default replacement behavior. Long-running
    CLI jobs pass ``overwrite=False`` so a file created after their initial check
    wins the race instead of being silently replaced.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save({"state_dict": model.state_dict(), "report": report}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite checkpoint: {path}"
                ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _restricted_checkpoint_load(path: str | Path) -> dict[str, Any]:
    """Load a clean public checkpoint without permitting arbitrary pickle code."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("renderer checkpoint payload is not a mapping")
    if not isinstance(payload.get("report"), dict):
        raise RuntimeError("renderer checkpoint report is missing or invalid")
    if not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("renderer checkpoint state_dict is missing or invalid")
    return payload


def load_count_model(path: str | Path) -> tuple[CountTextureModel, dict[str, Any]]:
    payload = _restricted_checkpoint_load(path)
    report = payload["report"]
    schema = report.get("schema")
    if schema not in SUPPORTED_RENDERER_SCHEMAS:
        raise RuntimeError(
            f"incompatible renderer schema {schema!r}; "
            f"expected one of {sorted(SUPPORTED_RENDERER_SCHEMAS)!r}"
        )
    if not isinstance(report.get("config"), dict):
        raise RuntimeError("renderer checkpoint config is missing or invalid")
    if not isinstance(report.get("feature_names"), (list, tuple)):
        raise RuntimeError("renderer checkpoint feature_names are missing or invalid")
    reported_names = tuple(report["feature_names"])
    if not reported_names:
        raise RuntimeError("renderer checkpoint feature_names are empty")

    cfg = renderer_config_from_dict(dict(report["config"]))
    model = CountTextureModel(cfg)
    if "n_features" not in report or int(report["n_features"]) != model.n_features:
        raise RuntimeError(f"incompatible renderer feature width {report.get('n_features')}")
    if reported_names != tuple(model.feature_names):
        raise RuntimeError("incompatible renderer feature names; retrain the renderer")
    expected_keys = set(model.state_dict())
    actual_keys = set(payload["state_dict"])
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(f"incompatible renderer state keys; missing={missing}, extra={extra}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, report


# ---------------------------------------------------------------------------
# Sampling (inference)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_count_streams(
    model: CountTextureModel,
    arrays: dict[str, np.ndarray],
    dxdy: np.ndarray,
    mask: np.ndarray,
    *,
    spec_key: str,
    seed: int = 7,
    device: str | torch.device = "cpu",
    offset_temperature: float | None = None,
    offset_magnitude_temperature: float | None = None,
    offset_direction_temperature: float | None = None,
    offset_magnitude_temperature_map: np.ndarray | None = None,
    offset_direction_temperature_map: np.ndarray | None = None,
    base_hysteresis: float | None = None,
    sign_crossing_penalty: float | None = None,
    lateral_offset_penalty: float | None = 1.5,
    lateral_offset_free: float = 1.0,
) -> np.ndarray:
    """Sample integer count streams for a batch of smooth intents.

    ``dxdy``/``mask`` are the smooth intents to render, ``[N, H, 2]`` / ``[N, H]``.
    ``arrays`` supplies each example's exact 256-report physical context. All
    256 reports define the regime and the most recent 128 warm the float GRU;
    the raw future is never read. AF1.5 is the release law and therefore the
    default. ``offset_*_temperature_map`` are optional per-tick ``(N, H)`` maps
    (used by controlled sampling studies); scalar temperatures otherwise.
    """

    cfg = model.config
    dxdy = np.asarray(dxdy, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if dxdy.ndim != 3 or dxdy.shape[2] != 2 or mask.shape != dxdy.shape[:2]:
        raise ValueError("smooth intents and masks must have shapes [N,H,2] and [N,H]")
    if not np.all(np.isfinite(dxdy)) or not np.all(np.isfinite(mask)):
        raise ValueError("smooth intents and masks must be finite")
    active_mask = mask > 0.5
    for row in active_mask:
        duration = int(np.sum(row))
        if duration < 1 or not np.all(row[:duration]) or np.any(row[duration:]):
            raise ValueError("each future mask must be one contiguous active prefix")
    context = np.asarray(arrays.get("prefix_raw_dxdy"))
    expected_context = (len(dxdy), cfg.context_ticks, 2)
    if context.shape != expected_context:
        raise ValueError(f"prefix_raw_dxdy must have shape {expected_context}")
    if not np.issubdtype(context.dtype, np.number) or not np.all(np.isfinite(context)):
        raise ValueError("prefix_raw_dxdy must contain finite numeric reports")
    rounded_context = np.rint(context.astype(np.float64, copy=False))
    if not np.array_equal(context, rounded_context):
        raise ValueError("prefix_raw_dxdy must contain integer hardware counts")
    if np.any(rounded_context < -32768.0) or np.any(rounded_context > 32767.0):
        raise ValueError("prefix_raw_dxdy exceeds signed int16 count range")
    if "prefix_mask" in arrays:
        prefix_mask = np.asarray(arrays["prefix_mask"])
        if prefix_mask.shape != (len(dxdy), cfg.context_ticks) or not np.all(
            prefix_mask > 0.5
        ):
            raise ValueError("global Renderer context cannot be padded or masked")
    device = torch.device(device)
    model = model.to(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    n_ex = len(mask)
    horizon = int(mask.shape[1])
    durations = np.sum(active_mask, axis=1).astype(int)
    out = np.zeros((n_ex, horizon, 2), dtype=np.float32)

    feats_static = np.zeros((n_ex, horizon, 4), dtype=np.float32)
    tangents = np.zeros((n_ex, horizon, 2), dtype=np.float32)
    normals = np.zeros((n_ex, horizon, 2), dtype=np.float32)
    smooth_all = np.zeros((n_ex, horizon, 2), dtype=np.float32)
    prefix_states: list[dict[str, Any]] = []

    prefix_spec = parse_smoothing_spec(cfg.prefix_smoothing_spec or spec_key)
    for i in range(n_ex):
        d = durations[i]
        if d < 1:
            prefix_states.append({"features": np.zeros((0, model.n_features), dtype=np.float32)})
            continue
        smooth = dxdy[i, :d].astype(np.float64)
        smooth_all[i, :d] = smooth.astype(np.float32)
        prefix_full = _prefix_from_arrays(arrays, i)
        regime = _regime_features(prefix_full, tuple(cfg.regime_feature_names), window=cfg.regime_window)
        prefix = _truncate_prefix(prefix_full, cfg.recurrent_warm_ticks)
        if len(prefix):
            smooth_p = smooth_dxdy(prefix.astype(np.float32), spec=prefix_spec).astype(np.float64)
            pex = build_teacher_example(
                smooth_p,
                prefix,
                reset_idx=len(prefix),
                offset_radius=cfg.offset_radius,
                regime_features=regime,
                regime_feature_names=tuple(cfg.regime_feature_names),
                base_hysteresis=cfg.base_hysteresis,
            )
            last_smooth = smooth_p[-1]
            last_dir = last_smooth if float(np.linalg.norm(last_smooth)) > EPS else np.asarray([1.0, 0.0])
            active_p = np.any(prefix != 0.0, axis=1)
            ra, rl = False, 0
            for t in range(len(active_p)):
                if t == 0 or bool(active_p[t]) != ra:
                    ra, rl = bool(active_p[t]), 1
                else:
                    rl += 1
            tail = active_p[-RECENT_WINDOW:]
            nz_rows = prefix[active_p]
            last_axis = np.zeros(2, dtype=np.float64)
            for row in prefix:
                nz_axis = row != 0.0
                last_axis[nz_axis] = row[nz_axis]
            prefix_states.append(
                {
                    "features": pex["features"],
                    "last_dir": last_dir,
                    "last_smooth": last_smooth,
                    "prev_emit": prefix[-1],
                    "last_nz": nz_rows[-1] if len(nz_rows) else np.zeros(2),
                    "last_axis_nz": last_axis,
                    "run_active": ra,
                    "run_length": rl,
                    "recent_zero": float(np.mean(~tail)) if len(tail) else 1.0,
                    "active_tail": tail.astype(np.float32),
                    "regime_features": regime,
                }
            )
        else:
            prefix_states.append(
                {
                    "features": np.zeros((0, model.n_features), dtype=np.float32),
                    "last_dir": np.asarray([1.0, 0.0]),
                    "last_smooth": None,
                    "prev_emit": np.zeros(2),
                    "last_nz": np.zeros(2),
                    "last_axis_nz": np.zeros(2),
                    "run_active": False,
                    "run_length": 0,
                    "recent_zero": 1.0,
                    "regime_features": regime,
                }
            )

        speed, accel, curvature = _kinematics(smooth, left_context=prefix_states[-1]["last_smooth"])
        dirs = _ffill_directions(smooth, prefix_states[-1]["last_dir"])
        tangents[i, :d] = dirs.astype(np.float32)
        normals[i, :d] = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1).astype(np.float32)
        feats_static[i, :d, 0] = (speed / 4.0).astype(np.float32)
        feats_static[i, :d, 1] = np.clip(accel, -3.0, 3.0).astype(np.float32)
        feats_static[i, :d, 2] = curvature.astype(np.float32)
        feats_static[i, :d, 3] = np.log1p(speed).astype(np.float32)

    # prefix warm-up (padded batch through the GRUCell, right-aligned)
    plens = np.asarray([len(ps["features"]) for ps in prefix_states], dtype=int)
    pmax = int(plens.max()) if len(plens) else 0
    h = torch.zeros((n_ex, cfg.hidden), dtype=torch.float32, device=device)
    if pmax > 0:
        pf = np.zeros((n_ex, pmax, model.n_features), dtype=np.float32)
        for i, ps in enumerate(prefix_states):
            m = len(ps["features"])
            if m:
                pf[i, pmax - m :] = ps["features"]
        pf_t = torch.from_numpy(pf).to(device)
        hcur = torch.zeros((n_ex, cfg.hidden), dtype=torch.float32, device=device)
        for t in range(pmax):
            xt = pf_t[:, t]
            valid = torch.from_numpy((pmax - plens <= t).astype(np.float32)).to(device).unsqueeze(1)
            hnew = model.step_cell(xt, hcur)
            hcur = valid * hnew + (1.0 - valid) * hcur
        h = hcur

    smooth_t = torch.from_numpy(smooth_all).to(device)
    tang_t = torch.from_numpy(tangents).to(device)
    norm_t = torch.from_numpy(normals).to(device)
    stat_t = torch.from_numpy(feats_static).to(device)
    dur_t = torch.from_numpy(durations).to(device)

    acc = torch.zeros((n_ex, 2), dtype=torch.float32, device=device)
    prev_emit = torch.from_numpy(np.stack([ps.get("prev_emit", np.zeros(2)) for ps in prefix_states]).astype(np.float32)).to(device)
    last_nz = torch.from_numpy(np.stack([ps.get("last_nz", np.zeros(2)) for ps in prefix_states]).astype(np.float32)).to(device)
    last_axis_nz = torch.from_numpy(np.stack([ps.get("last_axis_nz", np.zeros(2)) for ps in prefix_states]).astype(np.float32)).to(device)
    run_active = torch.from_numpy(np.asarray([1.0 if ps.get("run_active") else 0.0 for ps in prefix_states], dtype=np.float32)).to(device)
    run_len = torch.from_numpy(np.asarray([float(ps.get("run_length", 0)) for ps in prefix_states], dtype=np.float32)).to(device)
    ring_np = np.zeros((n_ex, RECENT_WINDOW), dtype=np.float32)
    ring_n_np = np.zeros(n_ex, dtype=np.float32)
    ptr_np = np.zeros(n_ex, dtype=np.int64)
    for i, ps in enumerate(prefix_states):
        tail = ps.get("active_tail", np.zeros(0, dtype=np.float32))
        k = len(tail)
        if k:
            ring_np[i, :k] = tail
            ring_n_np[i] = k
            ptr_np[i] = k % RECENT_WINDOW
    ring = torch.from_numpy(ring_np).to(device)
    ring_n = torch.from_numpy(ring_n_np).to(device)
    ring_ptr = torch.from_numpy(ptr_np).to(device)
    recent_zero = torch.from_numpy(np.asarray([float(ps.get("recent_zero", 1.0)) for ps in prefix_states], dtype=np.float32)).to(device)
    regime_dim = len(cfg.regime_feature_names)
    regime_t = torch.from_numpy(
        np.stack(
            [np.asarray(ps.get("regime_features", np.zeros(regime_dim, dtype=np.float32)), dtype=np.float32) for ps in prefix_states]
        )
        if n_ex
        else np.zeros((0, regime_dim), dtype=np.float32)
    ).to(device)

    offsets = torch.arange(-cfg.offset_radius, cfg.offset_radius + 1, dtype=torch.float32, device=device)
    offset_grid = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), dim=-1).reshape(-1, 2)
    offset_magnitude = offset_grid.abs().amax(dim=1)
    out_t = torch.zeros((n_ex, horizon, 2), dtype=torch.float32, device=device)
    tau = max(float(cfg.temperature), 1e-3)
    tau_off = max(float(cfg.offset_temperature if offset_temperature is None else offset_temperature), 1e-3)
    tau_mag = cfg.offset_magnitude_temperature if offset_magnitude_temperature is None else offset_magnitude_temperature
    tau_dir = cfg.offset_direction_temperature if offset_direction_temperature is None else offset_direction_temperature
    tau_mag_map_t = None
    if offset_magnitude_temperature_map is not None:
        if offset_magnitude_temperature_map.shape != (n_ex, horizon):
            raise ValueError(f"offset_magnitude_temperature_map must have shape {(n_ex, horizon)}")
        tau_mag_map_t = torch.from_numpy(offset_magnitude_temperature_map.astype(np.float32)).to(device)
    tau_dir_map_t = None
    if offset_direction_temperature_map is not None:
        if offset_direction_temperature_map.shape != (n_ex, horizon):
            raise ValueError(f"offset_direction_temperature_map must have shape {(n_ex, horizon)}")
        tau_dir_map_t = torch.from_numpy(offset_direction_temperature_map.astype(np.float32)).to(device)
    hysteresis = float(cfg.base_hysteresis if base_hysteresis is None else base_hysteresis)
    crossing_penalty = max(float(0.0 if sign_crossing_penalty is None else sign_crossing_penalty), 0.0)
    lateral_penalty = max(float(1.5 if lateral_offset_penalty is None else lateral_offset_penalty), 0.0)
    lateral_free = max(float(lateral_offset_free), 0.0)

    tmax = int(durations.max()) if n_ex else 0
    for t in range(tmax):
        alive = dur_t > t
        alive_f = alive.float().unsqueeze(1)
        acc = acc + smooth_t[:, t] * alive_f
        tangent = tang_t[:, t]
        normal = norm_t[:, t]
        acc_t = (acc * tangent).sum(dim=1)
        acc_n = (acc * normal).sum(dim=1)
        prev_t = (prev_emit * tangent).sum(dim=1)
        prev_n = (prev_emit * normal).sum(dim=1)
        lnz_t = (last_nz * tangent).sum(dim=1)
        lnz_n = (last_nz * normal).sum(dim=1)
        rz = torch.where(ring_n >= 1.0, 1.0 - ring.sum(dim=1) / ring_n.clamp_min(1.0), recent_zero)
        base_columns = [
                stat_t[:, t, 0],
                stat_t[:, t, 1],
                stat_t[:, t, 2],
                tangent[:, 0],
                tangent[:, 1],
                acc_t.clamp(-4.0, 4.0),
                acc_n.clamp(-4.0, 4.0),
                (prev_t / 4.0).clamp(-4.0, 4.0),
                (prev_n / 4.0).clamp(-4.0, 4.0),
                run_active,
                torch.minimum(run_len, torch.tensor(64.0, device=device)) / 64.0,
                rz,
                stat_t[:, t, 3],
                (lnz_t / 4.0).clamp(-4.0, 4.0),
                (lnz_n / 4.0).clamp(-4.0, 4.0),
            ]
        base_x = torch.stack(base_columns, dim=1)
        x = torch.cat([base_x, regime_t], dim=1)
        h_new = model.step_cell(x, h)
        h = torch.where(alive.unsqueeze(1), h_new, h)

        emit_logit = model.head_emit(h).squeeze(-1) + float(cfg.emit_logit_bias)
        p_emit = torch.sigmoid(emit_logit / tau)
        emit = (torch.rand(n_ex, generator=gen, device=device) < p_emit).float()
        forced = acc.abs().amax(dim=1) >= cfg.force_release
        if cfg.zero_intent_gate:
            quiet_intent = smooth_t[:, t].abs().amax(dim=1) <= float(cfg.zero_intent_threshold)
            low_debt = acc.abs().amax(dim=1) < float(cfg.zero_accumulator_threshold)
            emit = torch.where(quiet_intent & low_debt, torch.zeros_like(emit), emit)
        # The safety release wins over the quiet gate, while duration ending
        # never invents a terminal packet.
        emit = torch.where(forced, torch.ones_like(emit), emit) * alive.float()

        k = 2 * cfg.offset_radius + 1
        off_logits = _compose_offset_logits(
            model.head_off(h),
            offset_magnitude,
            joint_temperature=tau_off,
            magnitude_temperature=tau_mag if tau_mag_map_t is None else tau_mag_map_t[:, t],
            direction_temperature=tau_dir if tau_dir_map_t is None else tau_dir_map_t[:, t],
        )
        base_for_guard = _hysteretic_base_torch(acc, last_axis_nz, hysteresis)
        candidates = base_for_guard.unsqueeze(1) + offset_grid.unsqueeze(0)
        if crossing_penalty > 0.0:
            last_sign = torch.sign(last_axis_nz).unsqueeze(1)
            cand_sign = torch.sign(candidates)
            crosses = (last_sign != 0.0) & (cand_sign != 0.0) & (last_sign * cand_sign < 0.0)
            off_logits = off_logits - crossing_penalty * crosses.any(dim=2).float()
        if lateral_penalty > 0.0:
            off_lat = (offset_grid.unsqueeze(0) * normal.unsqueeze(1)).sum(dim=2).abs()
            off_logits = off_logits - lateral_penalty * (off_lat - lateral_free).clamp_min(0.0)
        off_p = torch.softmax(off_logits, dim=1)
        joint = torch.multinomial(off_p, 1, generator=gen).squeeze(1)
        ox = offsets[joint // k]
        oy = offsets[joint % k]
        mark = base_for_guard + torch.stack([ox, oy], dim=1)
        mark = mark.clamp(-float(cfg.max_abs_count), float(cfg.max_abs_count))
        mark = mark * emit.unsqueeze(1)
        acc = acc - mark

        out_t[:, t] = mark
        is_active = (mark.abs().sum(dim=1) > 0).float()
        changed = (is_active != run_active) | (run_len == 0)
        run_len = torch.where(alive, torch.where(changed, torch.ones_like(run_len), run_len + 1.0), run_len)
        run_active = torch.where(alive, is_active, run_active)
        prev_emit = torch.where(alive.unsqueeze(1), mark, prev_emit)
        nz_now = (alive & (is_active > 0.5)).unsqueeze(1)
        last_nz = torch.where(nz_now, mark, last_nz)
        axis_nz_now = alive.unsqueeze(1) & (mark != 0.0)
        last_axis_nz = torch.where(axis_nz_now, mark, last_axis_nz)
        old = ring.gather(1, ring_ptr.unsqueeze(1)).squeeze(1)
        ring.scatter_(1, ring_ptr.unsqueeze(1), torch.where(alive, is_active, old).unsqueeze(1))
        ring_ptr = torch.where(alive, (ring_ptr + 1) % RECENT_WINDOW, ring_ptr)
        ring_n = torch.where(alive, torch.minimum(ring_n + 1.0, torch.tensor(float(RECENT_WINDOW), device=device)), ring_n)

    out[:] = out_t.cpu().numpy()
    for i in range(n_ex):
        out[i, durations[i] :] = 0.0
    return out


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FloatRendererReceipt:
    """Identity and execution boundary of a user-supplied float checkpoint."""

    backend: str
    artifact: str
    artifact_bytes: int
    artifact_sha256: str
    context_ticks: int
    device: str
    prefix_smoothing_spec: str
    lateral_offset_penalty: float
    native_online_handoff: bool


class FloatRendererModel:
    """Use a safely loaded float Renderer checkpoint behind the Pipeline API.

    This is the direct research path produced by ``training/train_renderer.py``.
    It replays the observed context through the float GRU and samples the full
    continuation when an event begins.  It is intentionally not presented as
    the authenticated constant-time native handoff.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        prefix_smoothing_spec: str | None = None,
    ) -> None:
        self.artifact = Path(checkpoint).expanduser().resolve()
        if not self.artifact.is_file():
            raise RendererRuntimeError(
                f"float Renderer checkpoint is missing: {self.artifact}"
            )
        artifact_bytes = self.artifact.stat().st_size
        artifact_sha256 = _checkpoint_sha256(self.artifact)
        try:
            model, report = load_count_model(self.artifact)
        except (OSError, RuntimeError, ValueError) as error:
            raise RendererRuntimeError(
                f"float Renderer checkpoint is incompatible: {error}"
            ) from error
        if (
            self.artifact.stat().st_size != artifact_bytes
            or _checkpoint_sha256(self.artifact) != artifact_sha256
        ):
            raise RendererRuntimeError(
                "float Renderer checkpoint changed while it was loaded"
            )
        if model.config.context_ticks != CONTEXT_TICKS or model.n_features != 20:
            raise RendererRuntimeError(
                "float Renderer must use the phase-free 20-feature, 256-context contract"
            )
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RendererRuntimeError("requested CUDA device is unavailable")
        smoothing = (
            prefix_smoothing_spec
            or model.config.prefix_smoothing_spec
            or "triangular_moving_average_path:window=5"
        )
        try:
            parse_smoothing_spec(smoothing)
        except ValueError as error:
            raise RendererRuntimeError(
                f"float Renderer prefix smoothing is invalid: {error}"
            ) from error
        if smoothing not in model.config.smooth_methods:
            raise RendererRuntimeError(
                "float Renderer prefix smoothing must be one of its trained w3/w5 views"
            )
        self.model = model.eval()
        self.report = report
        self.device = resolved_device
        self.prefix_smoothing_spec = smoothing
        self._render_lock = Lock()
        self.receipt = FloatRendererReceipt(
            backend="pytorch_float_checkpoint",
            artifact=self.artifact.name,
            artifact_bytes=int(artifact_bytes),
            artifact_sha256=artifact_sha256,
            context_ticks=CONTEXT_TICKS,
            device=str(resolved_device),
            prefix_smoothing_spec=smoothing,
            lateral_offset_penalty=LATERAL_OFFSET_PENALTY,
            native_online_handoff=False,
        )

    def prepare_context(self, raw_dxdy: np.ndarray) -> "PreparedFloatRendererContext":
        return PreparedFloatRendererContext(
            self,
            validate_physical_context(raw_dxdy),
        )

    def render(
        self,
        context: np.ndarray,
        smooth: np.ndarray,
        *,
        event_seed: int,
    ) -> np.ndarray:
        mask = np.ones((1, len(smooth)), dtype=np.float32)
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            )
        with self._render_lock, torch.random.fork_rng(devices=cuda_devices):
            deterministic_before = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                sampled = sample_count_streams(
                    self.model,
                    {"prefix_raw_dxdy": context.astype(np.float32, copy=False)[None]},
                    smooth.astype(np.float32, copy=False)[None],
                    mask,
                    spec_key=self.prefix_smoothing_spec,
                    seed=event_seed,
                    device=self.device,
                    base_hysteresis=self.model.config.base_hysteresis,
                    lateral_offset_penalty=LATERAL_OFFSET_PENALTY,
                )[0]
            finally:
                torch.use_deterministic_algorithms(deterministic_before)
        rounded = np.rint(sampled)
        if not np.array_equal(sampled, rounded):
            raise RendererRuntimeError("float Renderer emitted non-integer reports")
        if np.any(rounded < -32768.0) or np.any(rounded > 32767.0):
            raise RendererRuntimeError("float Renderer emitted reports outside int16")
        return np.ascontiguousarray(rounded, dtype=np.int16)


class PreparedFloatRendererContext:
    """One exact float-GRU context, ready for a single continuation."""

    def __init__(self, model: FloatRendererModel, context: np.ndarray) -> None:
        self._model = model
        self._context = context
        self._used = False

    def begin(
        self,
        smooth_dxdy: np.ndarray,
        future_mask: np.ndarray,
        *,
        event_seed: int,
    ) -> "FloatRendererEvent":
        if self._used:
            raise RendererRuntimeError("prepared Renderer context may only be used once")
        if type(event_seed) is not int or not 0 <= event_seed < 2**64:
            raise RendererRuntimeError(
                "event_seed must be a Python integer in uint64 range"
            )
        smooth = validate_smooth_future(smooth_dxdy, future_mask)
        output = self._model.render(
            self._context,
            smooth,
            event_seed=event_seed,
        )
        self._used = True
        return FloatRendererEvent(output)


class FloatRendererEvent:
    """Step through a continuation pre-rendered by the float reference graph."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = np.ascontiguousarray(output, dtype=np.int16)
        self._tick = 0

    @property
    def duration(self) -> int:
        return len(self._output)

    @property
    def complete(self) -> bool:
        return self._tick >= self.duration

    def step(self) -> np.ndarray:
        if self.complete:
            raise RendererRuntimeError("Renderer event is complete")
        output = self._output[self._tick].copy()
        self._tick += 1
        return output

    def render_remaining(self) -> np.ndarray:
        output = self._output[self._tick :].copy()
        self._tick = self.duration
        return output


def benchmark_tick_latency(model: CountTextureModel, n_ticks: int = 2000) -> dict[str, float]:
    """Single-stream CPU per-tick latency of the sampling step (1 kHz budget check)."""

    model = model.to("cpu")
    model.eval()
    x = torch.randn(1, model.n_features)
    h = torch.zeros(1, model.config.hidden)
    with torch.no_grad():
        for _ in range(50):
            h = model.step_cell(x, h)
        t0 = time.perf_counter()
        for _ in range(n_ticks):
            h = model.step_cell(x, h)
            _ = torch.sigmoid(model.head_emit(h))
            p = torch.softmax(model.head_off(h), dim=1)
            _ = torch.multinomial(p, 1)
        dt = time.perf_counter() - t0
    return {"per_tick_us": float(dt / n_ticks * 1e6), "ticks_per_second": float(1.0 / (dt / n_ticks))}


__all__ = [
    "RendererConfig",
    "CountTextureModel",
    "FloatRendererEvent",
    "FloatRendererModel",
    "FloatRendererReceipt",
    "PreparedFloatRendererContext",
    "renderer_config_from_dict",
    "train_count_texture_model",
    "save_count_model",
    "load_count_model",
    "sample_count_streams",
    "build_teacher_example",
    "benchmark_tick_latency",
    "RENDERER_SCHEMA",
]
