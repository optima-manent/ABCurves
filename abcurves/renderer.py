"""The Renderer: turn a smooth plan into real 1 kHz hardware texture.

The planner outputs a *smooth* B->C curve.  Real hardware never emits a smooth
curve -- it emits quantized, bursty, integer ``dx/dy`` packets: mostly zeros,
occasional bursts, a characteristic negative autocorrelation (the "delta-sigma"
quantization texture), poll-skip gaps, and a specific spectral fingerprint.
The renderer's job is to add exactly that texture on top of the smooth intent,
consistently with the A->B prefix the movement came from.

How it works
------------
A tiny **GRU** wrapped around a **hysteretic delta-sigma accumulator**:

* the accumulator integrates the smooth intent and only "releases" an integer
  count once enough movement mass has built up -- this reproduces the bursty
  zero/nonzero cadence and, because the accumulator reclaims whatever it
  emits, keeps the endpoint unbiased *by construction* (no drift);
* per tick, the GRU consumes causal features (smooth kinematics, accumulator
  state in the movement frame, previous emission, run state, recent zero rate,
  and a summary of the A->B prefix's own count regime) and samples: an
  ``emit`` decision (report now?) and a small integer ``offset`` around the
  accumulator's rounded base;
* the A->B prefix warms the GRU hidden state and accumulator so the B seam is
  continuous.

It is trained self-supervised: each retained successful B80 event supplies a
``(smooth(raw), raw)`` pair.  The final model has 80,378 parameters and never
sees the held-out raw future at inference; it receives only the smooth plan,
the causal A->B prefix, and an optional three-number causal style state built
from earlier human events in the same contiguous run.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .smoothing import parse_smoothing_spec, smooth_dxdy

EPS = 1e-9
RENDERER_SCHEMA = "abcurves.renderer.v1"
SUPPORTED_RENDERER_SCHEMAS = {RENDERER_SCHEMA}
RECENT_WINDOW = 60

BASE_FEATURE_NAMES = (
    "speed_scaled",
    "accel",
    "curvature",
    "phase",
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
    hidden: int = 96
    offset_radius: int = 5
    prefix_ticks: int = 128
    epochs: int = 12
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    seed: int = 7
    temperature: float = 1.0  # emit head temperature (cadence)
    offset_temperature: float = 1.0  # joint offset head temperature
    offset_magnitude_temperature: float | None = 1.0  # radial texture temperature
    offset_direction_temperature: float | None = 0.3  # within-radius direction temperature
    base_hysteresis: float = 1.0  # deadband against accumulator sign crossings
    force_release: float = 48.0  # |accumulator| safety release (drift guard)
    max_abs_count: int = 127
    regime_window: int = 256
    prefix_loss_weight: float = 1.0  # supervise the prefix warm-up ticks too
    regime_feature_names: tuple[str, ...] = REGIME_FEATURE_NAMES
    smooth_methods: tuple[str, ...] = (
        "triangular_moving_average_path:window=5",
        "triangular_moving_average_path:window=9",
    )


def _feature_names_for_config(config: RendererConfig) -> tuple[str, ...]:
    return BASE_FEATURE_NAMES + tuple(config.regime_feature_names)


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
        self.cell = torch.nn.GRUCell(self.n_features, config.hidden)
        self.head_emit = torch.nn.Linear(config.hidden, 1)
        self.head_off = torch.nn.Linear(config.hidden, k * k)

    def sync_cell(self) -> None:
        """Copy GRU weights into the GRUCell used for stepwise sampling."""

        sd = self.gru.state_dict()
        self.cell.load_state_dict(
            {
                "weight_ih": sd["weight_ih_l0"],
                "weight_hh": sd["weight_hh_l0"],
                "bias_ih": sd["bias_ih_l0"],
                "bias_hh": sd["bias_hh_l0"],
            }
        )

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


def _kinematics(smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.linalg.norm(smooth, axis=1)
    accel = np.diff(speed, prepend=speed[:1])
    n = len(smooth)
    curvature = np.zeros(n, dtype=np.float64)
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

    phase_den = float(max(n - reset_idx - 1, 1))
    phase = np.zeros(n, dtype=np.float64)
    phase[reset_idx:] = (np.arange(n - reset_idx, dtype=np.float64)) / phase_den
    phase[:reset_idx] = 1.0

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

    feats = np.stack(
        [
            speed / 4.0,
            np.clip(accel, -3.0, 3.0),
            curvature,
            phase,
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
        ],
        axis=1,
    ).astype(np.float32)
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
    mask = arrays["future_mask"][idx].astype(bool)
    n = int(np.sum(mask))
    if n < 4:
        return None
    raw_f = arrays["future_raw_dxdy"][idx, :n].astype(np.float64)
    prefix_full = _prefix_from_arrays(arrays, idx)
    regime = _regime_features(prefix_full, tuple(config.regime_feature_names), window=config.regime_window)
    prefix = _truncate_prefix(prefix_full, config.prefix_ticks)
    raw = np.concatenate([prefix, raw_f], axis=0)
    # Deployment warms the recurrent state from the observed prefix alone, so
    # its smoothed prefix features must not use future samples.  The planner's
    # future target, however, is smoothed with its preceding trajectory context.
    # Preserve both facts: causal/available prefix warm-up plus the future side
    # of a context-aware joint smoothing pass.  Any remaining B mismatch is the
    # real deployment seam the renderer must handle, not a leak to hide.
    joint_smooth = smooth_dxdy(
        raw.astype(np.float32),
        spec=parse_smoothing_spec(spec_key),
    ).astype(np.float64)
    if len(prefix):
        smooth_prefix = smooth_dxdy(
            prefix.astype(np.float32),
            spec=parse_smoothing_spec(spec_key),
        ).astype(np.float64)
    else:
        smooth_prefix = np.zeros((0, 2), dtype=np.float64)
    smooth = np.concatenate(
        [smooth_prefix, joint_smooth[len(prefix) :]],
        axis=0,
    )
    return smooth, raw, len(prefix), regime


def _pad_variants(variants: list[list[dict[str, np.ndarray]]]) -> tuple[list[dict[str, torch.Tensor]], int]:
    if not variants or not all(variants):
        raise ValueError("at least one non-empty variant rowset is required")
    n_usable = min(len(rowset) for rowset in variants)
    max_len = max(len(ex["features"]) for rowset in variants for ex in rowset[:n_usable])
    n_features = int(variants[0][0]["features"].shape[1])

    def pad_stack(rowset: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        f = np.zeros((n_usable, max_len, n_features), dtype=np.float32)
        emit = np.zeros((n_usable, max_len), dtype=np.float32)
        off_joint = np.zeros((n_usable, max_len), dtype=np.int64)
        voff = np.zeros((n_usable, max_len), dtype=np.float32)
        lmask = np.zeros((n_usable, max_len), dtype=np.float32)
        for j, ex in enumerate(rowset[:n_usable]):
            m = len(ex["features"])
            f[j, :m] = ex["features"]
            emit[j, :m] = ex["emit"]
            off_joint[j, :m] = ex["off_joint"]
            voff[j, :m] = ex["valid_off"]
            lmask[j, :m] = ex["loss_mask"]
        return {
            "features": torch.from_numpy(f),
            "emit": torch.from_numpy(emit),
            "off_joint": torch.from_numpy(off_joint),
            "valid_off": torch.from_numpy(voff),
            "loss_mask": torch.from_numpy(lmask),
        }

    return [pad_stack(rowset) for rowset in variants], n_usable


def _fit_variants(
    model: CountTextureModel,
    variants: list[list[dict[str, np.ndarray]]],
    config: RendererConfig,
    *,
    device: torch.device,
    rng: np.random.Generator,
    opt: torch.optim.Optimizer,
    epochs: int,
    log_every: int,
) -> list[dict[str, float]]:
    tensors, n_usable = _pad_variants(variants)
    history: list[dict[str, float]] = []
    for local_epoch in range(int(epochs)):
        order = rng.permutation(n_usable)
        choice = rng.integers(0, len(tensors), size=n_usable)
        total, total_emit, total_off, batches = 0.0, 0.0, 0.0, 0
        model.train()
        for start in range(0, n_usable, config.batch_size):
            idx = order[start : start + config.batch_size]
            parts = {k: [] for k in ("features", "emit", "off_joint", "valid_off", "loss_mask")}
            for j in idx:
                src = tensors[int(choice[j])]
                for k in parts:
                    parts[k].append(src[k][j])
            batch = {k: torch.stack(v).to(device) for k, v in parts.items()}
            out = model(batch["features"])
            lm = batch["loss_mask"]
            emit_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out["emit_logit"], batch["emit"], reduction="none"
            )
            emit_loss = (emit_loss * lm).sum() / lm.sum().clamp_min(1.0)
            om = lm * batch["valid_off"]
            ce = torch.nn.functional.cross_entropy(
                out["off_logits"].flatten(0, 1), batch["off_joint"].flatten(), reduction="none"
            ).view_as(om)
            off_loss = (ce * om).sum() / om.sum().clamp_min(1.0)
            loss = emit_loss + off_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            total_emit += float(emit_loss.item())
            total_off += float(off_loss.item())
            batches += 1
        rec = {
            "epoch": int(local_epoch),
            "loss": total / max(batches, 1),
            "emit_bce": total_emit / max(batches, 1),
            "offset_ce": total_off / max(batches, 1),
        }
        history.append(rec)
        if log_every and (local_epoch % log_every == 0 or local_epoch == int(epochs) - 1):
            print(
                f"  renderer epoch {local_epoch}: loss {rec['loss']:.4f} "
                f"(emit {rec['emit_bce']:.4f}, off {rec['offset_ce']:.4f})"
            )
    return history


def train_count_texture_model(
    arrays: dict[str, np.ndarray],
    config: RendererConfig = RendererConfig(),
    device: str | torch.device = "cpu",
    log_every: int = 5,
) -> tuple[CountTextureModel, dict[str, Any]]:
    """Teacher-forced training on ``(smooth(raw), raw)`` pairs from raw streams.

    Each example is built once for both release smoothing teachers in
    ``config.smooth_methods`` (w5 for the Planner-target regime and w9 for a
    smoother oracle regime). Epochs sample the two views uniformly so the
    Renderer is robust to the intent smoothness it sees at deployment. No
    external labels are used: the raw stream is its own supervision.
    """

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(device)
    event_weights = np.asarray(
        arrays.get(
            "event_weight",
            np.ones(len(arrays["future_mask"]), dtype=np.float32),
        ),
        dtype=np.float64,
    ).reshape(-1)
    if (
        event_weights.shape != (len(arrays["future_mask"]),)
        or not np.isfinite(event_weights).all()
        or np.any(event_weights <= 0.0)
    ):
        raise ValueError("event_weight must be a finite positive vector of length N")

    teacher_methods = tuple(config.smooth_methods)
    if not teacher_methods:
        raise ValueError("at least one training smoothing method is required")
    variants: list[list[dict[str, np.ndarray]]] = []
    for spec_key in teacher_methods:
        rowset: list[dict[str, np.ndarray]] = []
        for i in range(len(arrays["future_mask"])):
            streams = _example_streams(arrays, i, spec_key, config=config)
            if streams is None:
                continue
            smooth, raw, reset_idx, regime = streams
            example = build_teacher_example(
                    smooth,
                    raw,
                    reset_idx=reset_idx,
                    offset_radius=config.offset_radius,
                    regime_features=regime,
                    regime_feature_names=tuple(config.regime_feature_names),
                    prefix_loss_weight=config.prefix_loss_weight,
                    base_hysteresis=config.base_hysteresis,
                )
            example["loss_mask"] = example["loss_mask"] * float(event_weights[i])
            rowset.append(example)
        variants.append(rowset)

    n_usable = min(len(rowset) for rowset in variants)
    clip_total = sum(float(np.sum(ex["clipped_off"] * ex["loss_mask"])) for rs in variants for ex in rs)
    active_total = sum(float(np.sum(ex["emit"] * ex["loss_mask"])) for rs in variants for ex in rs)
    clip_fraction = clip_total / max(active_total, 1.0)

    model = CountTextureModel(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = _fit_variants(
        model, variants, config, device=device, rng=rng, opt=opt, epochs=config.epochs, log_every=log_every
    )
    model.eval()
    model.sync_cell()
    report = {
        "schema": RENDERER_SCHEMA,
        "config": asdict(config),
        "feature_names": list(model.feature_names),
        "n_features": int(model.n_features),
        "architecture": "gru_emit_joint_offset_frame_regime_hysteretic",
        "training_smooth_methods": list(teacher_methods),
        "n_events": int(n_usable),
        "event_weight_sum": float(event_weights.sum()),
        "teacher_offset_clip_fraction": float(clip_fraction),
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "history": history,
    }
    return model, report


def save_count_model(model: CountTextureModel, report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "report": report}, path)


def load_count_model(path: str | Path) -> tuple[CountTextureModel, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = payload.get("report", {})
    if report.get("schema") not in SUPPORTED_RENDERER_SCHEMAS:
        raise RuntimeError(
            f"incompatible renderer schema {report.get('schema')!r}; "
            f"expected one of {sorted(SUPPORTED_RENDERER_SCHEMAS)!r}"
        )
    cfg = renderer_config_from_dict(dict(report.get("config", {})))
    model = CountTextureModel(cfg)
    if int(report.get("n_features", model.n_features)) != model.n_features:
        raise RuntimeError(f"incompatible renderer feature width {report.get('n_features')}")
    feature_names = tuple(report.get("feature_names", model.feature_names))
    if feature_names != tuple(model.feature_names):
        raise RuntimeError("incompatible renderer feature names; retrain the renderer")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.sync_cell()
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
    lateral_offset_penalty: float | None = None,
    lateral_offset_free: float = 1.0,
) -> np.ndarray:
    """Sample integer count streams for a batch of smooth intents.

    ``dxdy``/``mask`` are the smooth intents to render, ``[N, H, 2]`` / ``[N, H]``.
    ``arrays`` supplies each example's causal A->B prefix (``prefix_raw_dxdy`` /
    ``prefix_mask``) used to warm the GRU + accumulator; the raw future is never
    read.  ``offset_*_temperature_map`` are optional per-tick ``(N, H)`` maps
    (used by controlled sampling studies); scalar temperatures otherwise.
    """

    cfg = model.config
    device = torch.device(device)
    model = model.to(device)
    model.sync_cell()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    n_ex = len(mask)
    horizon = int(mask.shape[1])
    durations = np.sum(mask > 0.5, axis=1).astype(int)
    out = np.zeros((n_ex, horizon, 2), dtype=np.float32)

    feats_static = np.zeros((n_ex, horizon, 5), dtype=np.float32)
    tangents = np.zeros((n_ex, horizon, 2), dtype=np.float32)
    normals = np.zeros((n_ex, horizon, 2), dtype=np.float32)
    smooth_all = np.zeros((n_ex, horizon, 2), dtype=np.float32)
    prefix_states: list[dict[str, Any]] = []

    spec = parse_smoothing_spec(spec_key)
    for i in range(n_ex):
        d = durations[i]
        if d < 1:
            prefix_states.append({"features": np.zeros((0, model.n_features), dtype=np.float32)})
            continue
        smooth = dxdy[i, :d].astype(np.float64)
        smooth_all[i, :d] = smooth.astype(np.float32)
        prefix_full = _prefix_from_arrays(arrays, i)
        regime = _regime_features(prefix_full, tuple(cfg.regime_feature_names), window=cfg.regime_window)
        prefix = _truncate_prefix(prefix_full, cfg.prefix_ticks)
        if len(prefix):
            smooth_p = smooth_dxdy(prefix.astype(np.float32), spec=spec).astype(np.float64)
            pex = build_teacher_example(
                smooth_p,
                prefix,
                reset_idx=len(prefix),
                offset_radius=cfg.offset_radius,
                regime_features=regime,
                regime_feature_names=tuple(cfg.regime_feature_names),
                base_hysteresis=cfg.base_hysteresis,
            )
            last_dir = smooth_p[-1] if float(np.linalg.norm(smooth_p[-1])) > EPS else np.asarray([1.0, 0.0])
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
                    "prev_emit": np.zeros(2),
                    "last_nz": np.zeros(2),
                    "last_axis_nz": np.zeros(2),
                    "run_active": False,
                    "run_length": 0,
                    "recent_zero": 1.0,
                    "regime_features": regime,
                }
            )

        speed, accel, curvature = _kinematics(smooth)
        dirs = _ffill_directions(smooth, prefix_states[-1]["last_dir"])
        tangents[i, :d] = dirs.astype(np.float32)
        normals[i, :d] = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1).astype(np.float32)
        phase = np.arange(d, dtype=np.float64) / float(max(d - 1, 1))
        feats_static[i, :d, 0] = (speed / 4.0).astype(np.float32)
        feats_static[i, :d, 1] = np.clip(accel, -3.0, 3.0).astype(np.float32)
        feats_static[i, :d, 2] = curvature.astype(np.float32)
        feats_static[i, :d, 3] = phase.astype(np.float32)
        feats_static[i, :d, 4] = np.log1p(speed).astype(np.float32)

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
            hnew = model.cell(xt, hcur)
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
    lateral_penalty = max(float(0.0 if lateral_offset_penalty is None else lateral_offset_penalty), 0.0)
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
        base_x = torch.stack(
            [
                stat_t[:, t, 0],
                stat_t[:, t, 1],
                stat_t[:, t, 2],
                stat_t[:, t, 3],
                tangent[:, 0],
                tangent[:, 1],
                acc_t.clamp(-4.0, 4.0),
                acc_n.clamp(-4.0, 4.0),
                (prev_t / 4.0).clamp(-4.0, 4.0),
                (prev_n / 4.0).clamp(-4.0, 4.0),
                run_active,
                torch.minimum(run_len, torch.tensor(64.0, device=device)) / 64.0,
                rz,
                stat_t[:, t, 4],
                (lnz_t / 4.0).clamp(-4.0, 4.0),
                (lnz_n / 4.0).clamp(-4.0, 4.0),
            ],
            dim=1,
        )
        x = torch.cat([base_x, regime_t], dim=1)
        h_new = model.cell(x, h)
        h = torch.where(alive.unsqueeze(1), h_new, h)

        emit_logit = model.head_emit(h).squeeze(-1)
        p_emit = torch.sigmoid(emit_logit / tau)
        emit = (torch.rand(n_ex, generator=gen, device=device) < p_emit).float()
        forced = (acc.abs().amax(dim=1) >= cfg.force_release) | (dur_t == t + 1)
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


def benchmark_tick_latency(model: CountTextureModel, n_ticks: int = 2000) -> dict[str, float]:
    """Single-stream CPU per-tick latency of the sampling step (1 kHz budget check)."""

    model = model.to("cpu")
    model.eval()
    model.sync_cell()
    x = torch.randn(1, model.n_features)
    h = torch.zeros(1, model.config.hidden)
    with torch.no_grad():
        for _ in range(50):
            h = model.cell(x, h)
        t0 = time.perf_counter()
        for _ in range(n_ticks):
            h = model.cell(x, h)
            _ = torch.sigmoid(model.head_emit(h))
            p = torch.softmax(model.head_off(h), dim=1)
            _ = torch.multinomial(p, 1)
        dt = time.perf_counter() - t0
    return {"per_tick_us": float(dt / n_ticks * 1e6), "ticks_per_second": float(1.0 / (dt / n_ticks))}


__all__ = [
    "RendererConfig",
    "CountTextureModel",
    "renderer_config_from_dict",
    "train_count_texture_model",
    "save_count_model",
    "load_count_model",
    "sample_count_streams",
    "build_teacher_example",
    "benchmark_tick_latency",
    "RENDERER_SCHEMA",
]
