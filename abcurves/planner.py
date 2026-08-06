"""The Planner: 16 hypotheses for how a human finishes the movement.

Given the causal A->B evidence, a checkpoint-selected raw or causal-EMA view
feeds the encoder and matching summaries. A separately declared causal view
provides the ProDMP boundary velocity. A small temporal conv-net emits **16
complete candidate continuations at once** -- each one a ProDMP weight vector
plus a log-duration.

Why 16 heads instead of one prediction: humans are not deterministic.  From
the same A->B, many different-but-valid B->C continuations exist, and a model
trained to minimize error against the one realized future collapses onto the
*average* movement -- statistically the most detectable thing you can produce.
The heads are trained with **relaxed winner-takes-all** (see
``training/train_planner.py``): for every training example, the head closest
to the realized human future gets almost all of the gradient, and the other 15
are left nearly alone.  Over the dataset the heads spread out and tile the
space of plausible human continuations -- 16 "personalities" instead of one
compromise.

The frozen deployment rule is deliberately simple: **sample one head uniformly
at random** per request. This keeps the full learned distribution in play
without searching for a preferred-looking answer.

Everything the Planner needs at inference lives inside one checkpoint
export: model weights, input normalizers, feature-name list, and ProDMP decode
metadata.  ``Planner.plan()`` is the entry point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import summary_features
from .prefix_representation import (
    DEFAULT_PREFIX_EMA_ALPHA,
    RAW_PREFIX_REPRESENTATION,
    PlannerPrefixContract,
    PrefixRepresentationSpec,
    apply_prefix_representation,
    planner_prefix_contract_from_config,
    planner_prefix_contract_from_payload,
)
from .prodmp import ProDMP, ProDMPConfig

CONTRACTED_PLANNER_SCHEMA = "abcurves.planner.v2"
SPLIT_PREFIX_PLANNER_SCHEMA = "abcurves.planner.v3"
SUPPORTED_PLANNER_SCHEMAS = {
    CONTRACTED_PLANNER_SCHEMA,
    SPLIT_PREFIX_PLANNER_SCHEMA,
}


# ---------------------------------------------------------------------------
# Config (the subset of training knobs the architecture depends on)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannerConfig:
    seed: int = 7
    profile_bins: int = 8
    n_basis: int = 20
    alpha: float = 25.0
    alpha_phase: float = 3.0
    weight_ridge: float = 1e-3
    eps_scale: float = 1.0
    prefix_len: int = 160
    # Input ablation. Public and live APIs still receive raw count ticks; this
    # checkpointed choice controls the view consumed by summaries, the TCN,
    # and the ProDMP boundary velocity.
    prefix_representation: str = RAW_PREFIX_REPRESENTATION
    prefix_ema_alpha: float = DEFAULT_PREFIX_EMA_ALPHA
    # Optional explicit ProDMP boundary view. ``None`` uses the encoder view.
    boundary_prefix_representation: str | None = None
    boundary_prefix_ema_alpha: float | None = None
    use_summary_features: bool = True
    hidden: int = 96
    blocks: int = 3
    dropout: float = 0.15
    batch_size: int = 128
    epochs: int = 260
    patience: int = 14
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Differentiable surrogate trajectory loss (normalized-time grid).
    grid_size: int = 48
    dir_s: float = 0.08
    w_endpoint: float = 1.0
    w_path: float = 0.6
    w_speed: float = 0.5
    w_duration: float = 0.75
    w_dir: float = 0.8
    # Relaxed winner-takes-all schedule.
    heads: int = 16
    wta_eps_start: float = 0.5
    wta_eps_end: float = 0.05
    # The final E260 recipe reaches the RWTA floor at epoch 46, then leaves
    # the sixteen modes stable for the remainder of training.
    wta_anneal_epochs: int = 45
    # Hinge-at-the-human-envelope turn penalty (kills the zigzag caricature
    # without over-straightening; thresholds are duration-conditional p95 turn
    # rates measured on REAL train curves, computed by the training script).
    w_turn_hinge: float = 0.3
    turn_hinge_thresholds: tuple[float, ...] = ()
    turn_hinge_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.wta_anneal_epochs < 1:
            raise ValueError("wta_anneal_epochs must be positive")
        planner_prefix_contract_from_config(self)


def planner_config_from_dict(raw: dict[str, Any]) -> PlannerConfig:
    """Build a PlannerConfig from a stored export dict, ignoring unknown keys."""

    known = {f.name for f in fields(PlannerConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if "turn_hinge_thresholds" in kwargs:
        kwargs["turn_hinge_thresholds"] = tuple(kwargs["turn_hinge_thresholds"])
    return PlannerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Model (a small causal TCN encoder + 16 linear heads)
# ---------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (4, 0))
        y = self.conv1(y).transpose(1, 2)
        y = self.dropout(F.relu(self.norm1(y))).transpose(1, 2)
        y = F.pad(y, (4, 0))
        y = self.conv2(y).transpose(1, 2)
        y = self.dropout(F.relu(self.norm2(y))).transpose(1, 2)
        return x + y


class TCNEncoder(nn.Module):
    def __init__(self, in_channels: int, summary_dim: int, hidden: int, blocks: int, dropout: float):
        super().__init__()
        self.input = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([TemporalBlock(hidden, dropout=dropout) for _ in range(int(blocks))])
        self.summary = (
            nn.Sequential(nn.Linear(summary_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            if summary_dim > 0
            else None
        )
        self.out_dim = hidden + (hidden if summary_dim > 0 else 0)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        x = prefix.transpose(1, 2)
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        # Prefix tensors are left-padded and right-aligned, so the B-adjacent
        # timestep is always the final position.
        last = h[:, -1, :]
        if self.summary is not None:
            last = torch.cat([last, self.summary(summary)], dim=-1)
        return last


class MultiHeadModel(nn.Module):
    """16 candidate (ProDMP weights + log-duration) vectors from one forward."""

    def __init__(self, summary_dim: int, out_dim: int, heads: int, config: PlannerConfig):
        super().__init__()
        self.heads = int(heads)
        self.out_dim = int(out_dim)
        self.encoder = TCNEncoder(3, summary_dim, config.hidden, config.blocks, config.dropout)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.out_dim, config.hidden), nn.ReLU(), nn.Dropout(config.dropout)
        )
        self.out = nn.Linear(config.hidden, self.heads * self.out_dim)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self.encoder(prefix, mask, summary))
        return self.out(h).view(-1, self.heads, self.out_dim)


# ---------------------------------------------------------------------------
# Decode: normalized head outputs -> smooth delta streams via ProDMP
# ---------------------------------------------------------------------------
def decode_heads(
    prodmp: ProDMP,
    y_norm: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    ydot_b: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``[N, K, D]`` normalized head outputs to delta streams + masks.

    Returns ``(dxdy [N, K, horizon, 2], mask [N, K, horizon])``.  ``ydot_b`` is
    the boundary velocity per example, ``[N, 2]``.
    """

    y_raw = np.asarray(y_norm, dtype=np.float64) * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    n, k, d = y_raw.shape
    coeff_dim = d - 1
    w = y_raw[:, :, :coeff_dim].reshape(n, k, prodmp.n_weights, 2)
    duration = np.exp(np.clip(y_raw[:, :, -1], 0.0, math.log(float(horizon))))
    out = np.zeros((n, k, horizon, 2), dtype=np.float32)
    masks = np.zeros((n, k, horizon), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            dur = int(np.clip(round(float(duration[i, j])), 1, horizon))
            deltas = prodmp.generate_deltas(w[i, j], np.asarray(ydot_b[i], dtype=np.float64), dur, goal_mode="learned")
            out[i, j, :dur] = deltas.astype(np.float32)
            masks[i, j, :dur] = 1.0
    return out, masks


class _CachedProDMP(ProDMP):
    """A ProDMP whose boundary components are memoized per duration (exact).

    The decode path always evaluates the basis at ``t = 1..d`` with
    ``tau = d, t_b = 0`` -- a pure function of ``d``.  Reusing the arrays is
    bit-exact by construction and turns a ~0.4 ms interpolation into a lookup.
    """

    def __init__(self, cfg: ProDMPConfig):
        super().__init__(cfg)
        self._components_cache: dict[int, tuple] = {}

    def _components(self, t, tau, t_b):
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
        d = len(t_arr)
        canonical = (
            d > 0
            and float(t_b) == 0.0
            and float(tau) == float(d)
            and t_arr[0] == 1.0
            and t_arr[-1] == float(d)
            and (d < 3 or bool(np.array_equal(t_arr, np.arange(1, d + 1, dtype=np.float64))))
        )
        if not canonical:
            return super()._components(t, tau, t_b)
        hit = self._components_cache.get(d)
        if hit is None:
            hit = super()._components(t_arr, tau, t_b)
            self._components_cache[d] = hit
        return hit

    def prewarm(self, max_duration: int) -> None:
        for d in range(1, int(max_duration) + 1):
            self._components(np.arange(1, d + 1, dtype=np.float64), float(d), 0.0)


# ---------------------------------------------------------------------------
# The deployable planner
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    """One sampled smooth B->C continuation."""

    smooth_dxdy: np.ndarray  # [H, 2] float32 smooth deltas (1 per ms)
    mask: np.ndarray  # [H] float32 validity mask
    duration_ms: int
    head: int


class Planner:
    """Load-once wrapper around a contracted Planner export.

    >>> planner = Planner("models/planner_seed7.pt")
    >>> intent = planner.plan(prefix_counts, target_rel_at_B=(120.0, -35.0),
    ...                       target_radius=18.0, seed=7)
    >>> intent.smooth_dxdy[: intent.duration_ms]  # the smooth B->C plan
    """

    def __init__(
        self,
        export_path: str | Path,
        device: str = "cpu",
        *,
        prewarm_decode: bool = False,
    ):
        self.path = Path(export_path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        schema = payload.get("schema")
        if schema not in SUPPORTED_PLANNER_SCHEMAS:
            raise RuntimeError(
                f"unsupported planner schema {schema!r} in {self.path}; "
                f"expected one of {sorted(SUPPORTED_PLANNER_SCHEMAS)!r}"
            )
        self.payload: dict[str, Any] = payload
        # Every supported export declares the exact causal onset/B contract
        # expected by deployment.
        self.seam_contract: dict[str, Any] | None = payload.get("seam_contract")
        if not isinstance(self.seam_contract, dict):
            raise RuntimeError(
                f"contracted planner {self.path} has no causal seam contract"
            )
        if schema == SPLIT_PREFIX_PLANNER_SCHEMA:
            if payload.get("planner_prefix_contract") is None:
                raise RuntimeError(
                    f"split-prefix planner {self.path} has no prefix contract"
                )
        elif payload.get("planner_prefix_contract") is not None:
            raise RuntimeError(
                "planner_prefix_contract requires planner checkpoint schema "
                f"{SPLIT_PREFIX_PLANNER_SCHEMA!r}"
            )
        self.planner_prefix_contract = planner_prefix_contract_from_payload(
            payload
        )
        self.encoder_prefix_representation_spec = (
            self.planner_prefix_contract.encoder
        )
        self.boundary_prefix_representation_spec = (
            self.planner_prefix_contract.boundary_velocity
        )
        # Concise alias used by callers that precompute summaries.
        self.prefix_representation_spec = (
            self.encoder_prefix_representation_spec
        )
        self.device = torch.device(device)
        config_record = dict(payload["planner_config"])
        config_record.setdefault(
            "prefix_representation", self.prefix_representation_spec.name
        )
        config_record.setdefault(
            "prefix_ema_alpha",
            self.prefix_representation_spec.causal_ema_alpha,
        )
        if schema == SPLIT_PREFIX_PLANNER_SCHEMA:
            config_record.setdefault(
                "boundary_prefix_representation",
                self.boundary_prefix_representation_spec.name,
            )
            config_record.setdefault(
                "boundary_prefix_ema_alpha",
                self.boundary_prefix_representation_spec.causal_ema_alpha,
            )
        cfg = planner_config_from_dict(config_record)
        self.config = cfg
        self.heads = int(payload["heads"])
        self.horizon = int(payload["horizon"])
        self.prefix_len = int(payload["planner_config"]["prefix_len"])
        self.summary_names: list[str] = list(payload["summary_feature_names"])

        self.model = MultiHeadModel(
            int(payload["summary_dim"]), int(payload["target_dim"]), self.heads, cfg
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.prodmp = _CachedProDMP(
            ProDMPConfig(
                n_basis=int(payload["prodmp"]["n_basis"]),
                alpha=float(payload["prodmp"]["alpha"]),
                alpha_phase=float(payload["prodmp"]["alpha_phase"]),
                ridge=float(payload["prodmp"]["ridge"]),
            )
        )
        if prewarm_decode:
            # One-time startup cost (~1 s, ~90 MB) so no live decode ever pays
            # the cold basis-interpolation price. Optional for offline use.
            self.prodmp.prewarm(self.horizon)

        # Normalizers, materialized once.
        self._prefix_mean = self._payload_array("prefix_mean").reshape(1, 2)
        self._prefix_std = self._payload_array("prefix_std").reshape(1, 2)
        self._prefix_std[self._prefix_std < 1e-6] = 1.0
        self._summary_mean = self._payload_array("summary_mean").reshape(1, -1)
        self._summary_std = self._payload_array("summary_std").reshape(1, -1)
        self._summary_std[self._summary_std < 1e-6] = 1.0
        self.y_mean = self._payload_array("y_mean")
        self.y_std = self._payload_array("y_std")

    def _payload_array(self, key: str) -> np.ndarray:
        value = self.payload[key]
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Input encoding
    # ------------------------------------------------------------------ #
    def represented_prefix(self, prefix: np.ndarray) -> np.ndarray:
        """Return the encoder/summary view of the deployable raw window."""

        return self.represented_prefix_views(prefix)[0]

    def represented_prefix_views(
        self,
        prefix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(encoder, boundary_velocity)`` views from one raw window.

        Right alignment happens before either representation. Matching views
        reuse the same array, preserving the old/default runtime path.
        """

        raw = np.asarray(prefix)
        if raw.ndim != 2 or raw.shape[1:] != (2,):
            raise ValueError("planner prefix must have shape (P, 2)")
        window = raw[-self.prefix_len :]
        encoder_spec = getattr(
            self,
            "encoder_prefix_representation_spec",
            self.prefix_representation_spec,
        )
        boundary_spec = getattr(
            self,
            "boundary_prefix_representation_spec",
            encoder_spec,
        )
        encoder = apply_prefix_representation(window, encoder_spec)
        if encoder_spec.metadata() == boundary_spec.metadata():
            return encoder, encoder
        boundary = apply_prefix_representation(window, boundary_spec)
        return encoder, boundary

    def boundary_prefix(self, prefix: np.ndarray) -> np.ndarray:
        """Return the checkpoint-declared ProDMP boundary view."""

        return self.represented_prefix_views(prefix)[1]

    def _prefix_tensor(self, prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = self.prefix_len
        raw = np.asarray(prefix, dtype=np.float32)
        take = min(length, raw.shape[0])
        out = np.zeros((1, length, 3), dtype=np.float32)
        mask = np.zeros((1, length), dtype=np.float32)
        if take <= 0:
            return out, mask
        norm = (raw[-take:] - self._prefix_mean) / self._prefix_std
        out[0, -take:, :2] = norm
        out[0, -take:, 2] = 1.0
        mask[0, -take:] = 1.0
        return out, mask

    def _summary_vector(self, features: dict[str, float]) -> np.ndarray:
        x = np.asarray(
            [[float(features.get(name, 0.0)) for name in self.summary_names]], dtype=np.float32
        )
        x[~np.isfinite(x)] = 0.0
        out = (x - self._summary_mean) / self._summary_std
        out[~np.isfinite(out)] = 0.0
        return out.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Forward + decode
    # ------------------------------------------------------------------ #
    def forward_all_heads(
        self, prefix: np.ndarray, features: dict[str, float]
    ) -> np.ndarray:
        """One model forward -> normalized head outputs ``[1, heads, D]``.

        ``prefix`` is raw at this public boundary. ``features`` must describe
        the checkpoint-selected representation; :meth:`plan` and
        :meth:`plan_all_heads` construct that feature dict automatically.
        """

        p = self.represented_prefix(prefix)
        return self._forward_represented_prefix(p, features)

    def _forward_represented_prefix(
        self, prefix: np.ndarray, features: dict[str, float]
    ) -> np.ndarray:
        """Internal forward after the representation has been applied once."""

        summary = self._summary_vector(features)
        prefix_tensor, prefix_mask = self._prefix_tensor(prefix)
        with torch.no_grad():
            y_all = (
                self.model(
                    torch.as_tensor(prefix_tensor, dtype=torch.float32, device=self.device),
                    torch.as_tensor(prefix_mask, dtype=torch.float32, device=self.device),
                    torch.as_tensor(summary, dtype=torch.float32, device=self.device),
                )
                .detach()
                .cpu()
                .numpy()
            )
        return y_all

    def decode_head(self, y_all: np.ndarray, head: int, ydot_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode ONE head at full 1 kHz resolution."""

        y = y_all[:, head : head + 1, :]
        dxdy, mask = decode_heads(
            self.prodmp, y, self.y_mean, self.y_std, ydot_b.reshape(1, 2), horizon=self.horizon
        )
        return dxdy[0, 0].astype(np.float32), mask[0, 0].astype(np.float32)

    # ------------------------------------------------------------------ #
    # The planner entry point (at B)
    # ------------------------------------------------------------------ #
    def plan(
        self,
        prefix: np.ndarray,
        *,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress: float = 0.0,
        seed: int = 7,
        head: int | None = None,
        b_index_ms: float | None = None,
    ) -> Intent:
        """Sample one human-plausible smooth B->C continuation.

        ``prefix`` is always the raw A->B count stream ``[P, 2]`` at 1 kHz.
        The checkpoint's encoder view supplies summaries and the TCN; its
        boundary view supplies ProDMP velocity. The frozen deployment rule
        samples one of the 16 heads uniformly; pass ``head`` to force a
        specific hypothesis (for inspection / A-B tests).
        """

        represented, boundary = self.represented_prefix_views(prefix)
        feats = summary_features(
            represented,
            target_rel_at_B,
            target_radius,
            progress,
            horizon=self.horizon,
            b_index_ms=b_index_ms,
        )
        y_all = self._forward_represented_prefix(represented, feats)
        if head is None:
            rng = np.random.default_rng(seed)
            chosen = int(rng.integers(0, self.heads))
        else:
            chosen = int(np.clip(int(head), 0, self.heads - 1))
        ydot_b = (
            boundary[-1].astype(np.float64)
            if len(boundary)
            else np.zeros(2)
        )
        smooth, mask = self.decode_head(y_all, chosen, ydot_b)
        return Intent(
            smooth_dxdy=smooth,
            mask=mask,
            duration_ms=int(np.sum(mask > 0.5)),
            head=chosen,
        )

    def plan_all_heads(
        self,
        prefix: np.ndarray,
        *,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress: float = 0.0,
        b_index_ms: float | None = None,
    ) -> list[Intent]:
        """Decode all 16 hypotheses (visualization / best-of-K evaluation)."""

        represented, boundary = self.represented_prefix_views(prefix)
        feats = summary_features(
            represented,
            target_rel_at_B,
            target_radius,
            progress,
            horizon=self.horizon,
            b_index_ms=b_index_ms,
        )
        y_all = self._forward_represented_prefix(represented, feats)
        ydot_b = (
            boundary[-1].astype(np.float64)
            if len(boundary)
            else np.zeros(2)
        )
        intents = []
        for h in range(self.heads):
            smooth, mask = self.decode_head(y_all, h, ydot_b)
            intents.append(
                Intent(smooth_dxdy=smooth, mask=mask, duration_ms=int(np.sum(mask > 0.5)), head=h)
            )
        return intents


__all__ = [
    "Planner",
    "Intent",
    "PlannerConfig",
    "planner_config_from_dict",
    "MultiHeadModel",
    "TCNEncoder",
    "TemporalBlock",
    "decode_heads",
    "CONTRACTED_PLANNER_SCHEMA",
    "SPLIT_PREFIX_PLANNER_SCHEMA",
    "SUPPORTED_PLANNER_SCHEMAS",
    "PrefixRepresentationSpec",
    "PlannerPrefixContract",
]
