"""The frozen raw-prefix contract used by every released Planner.

Planner inputs are ordinary chronological count reports.  The right-aligned
160-tick window is passed unchanged to the summary table, causal TCN and ProDMP
boundary velocity.  The small metadata parser below exists to authenticate that
contract in a checkpoint; it is not an ablation switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PREFIX_REPRESENTATION_SCHEMA = "abcurves.planner_prefix.v1"
RAW_PREFIX_REPRESENTATION = "raw"
DEFAULT_PREFIX_EMA_ALPHA = 0.25
PREFIX_WINDOW_POLICY = "right_aligned_prefix_len_before_representation"
CAUSAL_EMA_STATE_POLICY = "reset_to_first_window_tick"


@dataclass(frozen=True)
class PrefixRepresentationSpec:
    """Authenticated metadata for the one supported raw-prefix view."""

    name: str = RAW_PREFIX_REPRESENTATION
    # Stored release checkpoints include this compatibility field even though
    # the raw representation does not use it. Keeping and checking the value
    # preserves exact container authentication without exposing an EMA path.
    causal_ema_alpha: float = DEFAULT_PREFIX_EMA_ALPHA

    def __post_init__(self) -> None:
        if self.name != RAW_PREFIX_REPRESENTATION:
            raise ValueError("the final Planner supports only raw prefix reports")
        alpha = float(self.causal_ema_alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("planner prefix metadata alpha must lie in (0, 1]")

    def metadata(self) -> dict[str, object]:
        return {
            "schema": PREFIX_REPRESENTATION_SCHEMA,
            "name": RAW_PREFIX_REPRESENTATION,
            "causal_ema_alpha": float(self.causal_ema_alpha),
            "window_policy": PREFIX_WINDOW_POLICY,
            "causal_ema_state_policy": CAUSAL_EMA_STATE_POLICY,
        }


def apply_prefix_representation(
    dxdy: np.ndarray,
    spec: PrefixRepresentationSpec,
) -> np.ndarray:
    """Validate the raw contract and return an owned float32 count stream."""

    if spec.name != RAW_PREFIX_REPRESENTATION:
        raise ValueError("the final Planner supports only raw prefix reports")
    values = np.asarray(dxdy)
    if values.ndim != 2 or values.shape[1:] != (2,):
        raise ValueError("planner prefix must have shape (P, 2)")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("planner prefix reports must be finite numbers")
    return np.array(values, dtype=np.float32, order="C", copy=True)


def prefix_representation_from_config(config: Any) -> PrefixRepresentationSpec:
    """Read and validate the frozen raw-prefix fields from a config object."""

    if getattr(config, "boundary_prefix_representation", None) is not None or getattr(
        config, "boundary_prefix_ema_alpha", None
    ) is not None:
        raise ValueError("split Planner prefix views are not part of the final recipe")
    return PrefixRepresentationSpec(
        name=str(
            getattr(config, "prefix_representation", RAW_PREFIX_REPRESENTATION)
        ),
        causal_ema_alpha=float(
            getattr(config, "prefix_ema_alpha", DEFAULT_PREFIX_EMA_ALPHA)
        ),
    )


def prefix_representation_from_payload(
    payload: Mapping[str, Any],
) -> PrefixRepresentationSpec:
    """Authenticate a release checkpoint's raw-prefix declaration."""

    if payload.get("planner_prefix_contract") is not None:
        raise RuntimeError("split Planner prefix checkpoints are not supported")
    raw = payload.get("prefix_representation")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            "planner checkpoint has no versioned prefix_representation metadata"
        )
    expected = {
        "schema": PREFIX_REPRESENTATION_SCHEMA,
        "name": RAW_PREFIX_REPRESENTATION,
        "window_policy": PREFIX_WINDOW_POLICY,
        "causal_ema_state_policy": CAUSAL_EMA_STATE_POLICY,
    }
    for name, value in expected.items():
        if raw.get(name) != value:
            raise RuntimeError(
                f"planner prefix metadata {name!r} differs from the raw contract"
            )
    try:
        spec = PrefixRepresentationSpec(
            name=str(raw["name"]),
            causal_ema_alpha=float(raw["causal_ema_alpha"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("planner prefix metadata is malformed") from error
    config = payload.get("planner_config")
    if not isinstance(config, Mapping):
        raise RuntimeError("planner_config must be a mapping")
    if str(config.get("prefix_representation")) != spec.name:
        raise RuntimeError("planner config and prefix metadata disagree")
    try:
        config_alpha = float(config["prefix_ema_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("planner config prefix metadata is malformed") from error
    if not np.isclose(
        config_alpha,
        spec.causal_ema_alpha,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("planner config and prefix metadata disagree")
    return spec


__all__ = [
    "DEFAULT_PREFIX_EMA_ALPHA",
    "PREFIX_REPRESENTATION_SCHEMA",
    "PrefixRepresentationSpec",
    "RAW_PREFIX_REPRESENTATION",
    "apply_prefix_representation",
    "prefix_representation_from_config",
    "prefix_representation_from_payload",
]
