"""Versioned Planner-prefix representations and split-view contracts.

The Planner always receives the causal raw A->B count stream at its public
boundary. A checkpoint independently declares the view consumed by its
encoder/summary path and the view used as the ProDMP boundary velocity. Each
view may be raw or a strictly causal EMA. A single-view contract maps the same
declared representation to both consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .capture_preprocess import causal_ema_dxdy


PREFIX_REPRESENTATION_SCHEMA = "abcurves.planner_prefix.v1"
PLANNER_PREFIX_CONTRACT_SCHEMA = "abcurves.planner_prefix_contract.v2"
RAW_PREFIX_REPRESENTATION = "raw"
CAUSAL_EMA_PREFIX_REPRESENTATION = "causal_ema"
SUPPORTED_PREFIX_REPRESENTATIONS = frozenset(
    {RAW_PREFIX_REPRESENTATION, CAUSAL_EMA_PREFIX_REPRESENTATION}
)
DEFAULT_PREFIX_EMA_ALPHA = 0.25
PREFIX_WINDOW_POLICY = "right_aligned_prefix_len_before_representation"
CAUSAL_EMA_STATE_POLICY = "reset_to_first_window_tick"
ENCODER_CONSUMER_POLICY = "encoder_tensor_and_summary_features"
BOUNDARY_VELOCITY_CONSUMER_POLICY = (
    "last_tick_for_prodmp_label_fit_surrogate_and_runtime_decode"
)


@dataclass(frozen=True)
class PrefixRepresentationSpec:
    """The checkpoint-controlled planner view of one causal prefix."""

    name: str = RAW_PREFIX_REPRESENTATION
    causal_ema_alpha: float = DEFAULT_PREFIX_EMA_ALPHA

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_PREFIX_REPRESENTATIONS:
            raise ValueError(
                f"unsupported planner prefix representation {self.name!r}; "
                f"expected one of {sorted(SUPPORTED_PREFIX_REPRESENTATIONS)!r}"
            )
        alpha = float(self.causal_ema_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("planner prefix causal_ema_alpha must lie in (0, 1]")

    def metadata(self) -> dict[str, object]:
        """Return the canonical checkpoint record."""

        return {
            "schema": PREFIX_REPRESENTATION_SCHEMA,
            "name": self.name,
            "causal_ema_alpha": float(self.causal_ema_alpha),
            "window_policy": PREFIX_WINDOW_POLICY,
            "causal_ema_state_policy": CAUSAL_EMA_STATE_POLICY,
        }


@dataclass(frozen=True)
class PlannerPrefixContract:
    """Checkpoint contract for the two causally available planner views."""

    encoder: PrefixRepresentationSpec = field(
        default_factory=PrefixRepresentationSpec
    )
    boundary_velocity: PrefixRepresentationSpec = field(
        default_factory=PrefixRepresentationSpec
    )

    def metadata(self) -> dict[str, object]:
        """Return the canonical split-view checkpoint record."""

        return {
            "schema": PLANNER_PREFIX_CONTRACT_SCHEMA,
            "encoder": self.encoder.metadata(),
            "boundary_velocity": self.boundary_velocity.metadata(),
            "encoder_consumer_policy": ENCODER_CONSUMER_POLICY,
            "boundary_velocity_consumer_policy": (
                BOUNDARY_VELOCITY_CONSUMER_POLICY
            ),
        }

    @property
    def views_match(self) -> bool:
        return self.encoder.metadata() == self.boundary_velocity.metadata()


def apply_prefix_representation(
    prefix: np.ndarray,
    spec: PrefixRepresentationSpec,
) -> np.ndarray:
    """Return the checkpoint-selected ``[P, 2]`` planner input.

    Both arms are pure NumPy and return float32.  The EMA is the recurrence
    ``y[0] = x[0]`` and
    ``y[t] = alpha*x[t] + (1-alpha)*y[t-1]``; it never reads a future tick.
    """

    raw = np.asarray(prefix, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1:] != (2,):
        raise ValueError("planner prefix must have shape (P, 2)")
    clean = raw.copy()
    clean[~np.isfinite(clean)] = 0.0
    if spec.name == RAW_PREFIX_REPRESENTATION:
        return clean
    return causal_ema_dxdy(clean, alpha=float(spec.causal_ema_alpha))


def prefix_representation_from_config(config: Any) -> PrefixRepresentationSpec:
    """Resolve and validate a PlannerConfig-like object."""

    return PrefixRepresentationSpec(
        name=str(
            getattr(config, "prefix_representation", RAW_PREFIX_REPRESENTATION)
        ),
        causal_ema_alpha=float(
            getattr(config, "prefix_ema_alpha", DEFAULT_PREFIX_EMA_ALPHA)
        ),
    )


def planner_prefix_contract_from_config(config: Any) -> PlannerPrefixContract:
    """Resolve a PlannerConfig-like object into the two-view contract.

    ``prefix_*`` declares the encoder view. Omitting ``boundary_prefix_*``
    maps that same representation to the ProDMP boundary.
    """

    encoder = prefix_representation_from_config(config)
    boundary_name = getattr(config, "boundary_prefix_representation", None)
    boundary_alpha = getattr(config, "boundary_prefix_ema_alpha", None)
    if boundary_name is None:
        if boundary_alpha is not None:
            raise ValueError(
                "boundary_prefix_ema_alpha requires "
                "boundary_prefix_representation"
            )
        return PlannerPrefixContract(
            encoder=encoder,
            boundary_velocity=encoder,
        )
    boundary = PrefixRepresentationSpec(
        name=str(boundary_name),
        causal_ema_alpha=(
            DEFAULT_PREFIX_EMA_ALPHA
            if boundary_alpha is None
            else float(boundary_alpha)
        ),
    )
    return PlannerPrefixContract(
        encoder=encoder,
        boundary_velocity=boundary,
    )


def _spec_from_metadata(
    raw_metadata: Mapping[str, Any],
    *,
    label: str,
) -> PrefixRepresentationSpec:
    if raw_metadata.get("schema") != PREFIX_REPRESENTATION_SCHEMA:
        raise RuntimeError(
            f"unsupported {label} prefix representation schema "
            f"{raw_metadata.get('schema')!r}"
        )
    if raw_metadata.get("window_policy") != PREFIX_WINDOW_POLICY:
        raise RuntimeError(
            f"{label} prefix representation uses an unsupported window policy"
        )
    if (
        raw_metadata.get("causal_ema_state_policy")
        != CAUSAL_EMA_STATE_POLICY
    ):
        raise RuntimeError(
            f"{label} prefix representation uses an unsupported EMA state policy"
        )
    try:
        return PrefixRepresentationSpec(
            name=str(raw_metadata["name"]),
            causal_ema_alpha=float(raw_metadata["causal_ema_alpha"]),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"{label} prefix representation metadata is missing {exc.args[0]!r}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid {label} prefix representation metadata: {exc}"
        ) from exc


def _assert_config_matches_spec(
    planner_config: Mapping[str, Any],
    spec: PrefixRepresentationSpec,
    *,
    name_key: str,
    alpha_key: str,
    label: str,
    required: bool,
) -> None:
    if required and (name_key not in planner_config or alpha_key not in planner_config):
        raise RuntimeError(
            f"split planner checkpoint must expose {name_key} and {alpha_key}"
        )
    if name_key in planner_config and (
        str(planner_config[name_key]) != spec.name
    ):
        raise RuntimeError(
            f"planner_config.{name_key} disagrees with {label} metadata"
        )
    if alpha_key in planner_config:
        try:
            config_alpha = float(planner_config[alpha_key])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"planner_config.{alpha_key} is invalid") from exc
        if not np.isclose(
            config_alpha,
            spec.causal_ema_alpha,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"planner_config.{alpha_key} disagrees with {label} metadata"
            )


def prefix_representation_from_payload(
    payload: Mapping[str, Any],
) -> PrefixRepresentationSpec:
    """Resolve and cross-check a single-view checkpoint declaration."""

    planner_config = payload.get("planner_config", {})
    if not isinstance(planner_config, Mapping):
        raise RuntimeError("planner_config must be a mapping")
    raw_metadata = payload.get("prefix_representation")
    if raw_metadata is None:
        raise RuntimeError(
            "planner checkpoint has no versioned prefix_representation metadata"
        )
    if not isinstance(raw_metadata, Mapping):
        raise RuntimeError("planner prefix_representation metadata must be a mapping")
    spec = _spec_from_metadata(raw_metadata, label="planner")
    _assert_config_matches_spec(
        planner_config,
        spec,
        name_key="prefix_representation",
        alpha_key="prefix_ema_alpha",
        label="checkpoint",
        required=True,
    )
    return spec


def planner_prefix_contract_from_payload(
    payload: Mapping[str, Any],
) -> PlannerPrefixContract:
    """Load an explicit split-view or single-view checkpoint contract."""

    planner_config = payload.get("planner_config", {})
    if not isinstance(planner_config, Mapping):
        raise RuntimeError("planner_config must be a mapping")
    raw_contract = payload.get("planner_prefix_contract")
    if raw_contract is None:
        if any(
            key in planner_config
            for key in (
                "boundary_prefix_representation",
                "boundary_prefix_ema_alpha",
            )
        ):
            raise RuntimeError(
                "planner checkpoint has boundary-prefix config fields but no "
                "versioned planner_prefix_contract metadata"
            )
        spec = prefix_representation_from_payload(payload)
        return PlannerPrefixContract(
            encoder=spec,
            boundary_velocity=spec,
        )
    if payload.get("prefix_representation") is not None:
        raise RuntimeError(
            "split planner checkpoint may not also carry ambiguous "
            "prefix_representation metadata"
        )
    if not isinstance(raw_contract, Mapping):
        raise RuntimeError("planner_prefix_contract metadata must be a mapping")
    if raw_contract.get("schema") != PLANNER_PREFIX_CONTRACT_SCHEMA:
        raise RuntimeError(
            "unsupported planner prefix contract schema "
            f"{raw_contract.get('schema')!r}"
        )
    if raw_contract.get("encoder_consumer_policy") != ENCODER_CONSUMER_POLICY:
        raise RuntimeError("unsupported planner encoder consumer policy")
    if (
        raw_contract.get("boundary_velocity_consumer_policy")
        != BOUNDARY_VELOCITY_CONSUMER_POLICY
    ):
        raise RuntimeError(
            "unsupported planner boundary-velocity consumer policy"
        )
    try:
        raw_encoder = raw_contract["encoder"]
        raw_boundary = raw_contract["boundary_velocity"]
    except KeyError as exc:
        raise RuntimeError(
            f"planner_prefix_contract metadata is missing {exc.args[0]!r}"
        ) from exc
    if not isinstance(raw_encoder, Mapping) or not isinstance(
        raw_boundary,
        Mapping,
    ):
        raise RuntimeError(
            "planner prefix contract views must be representation mappings"
        )
    encoder = _spec_from_metadata(raw_encoder, label="encoder")
    boundary = _spec_from_metadata(raw_boundary, label="boundary-velocity")
    _assert_config_matches_spec(
        planner_config,
        encoder,
        name_key="prefix_representation",
        alpha_key="prefix_ema_alpha",
        label="encoder",
        required=True,
    )
    _assert_config_matches_spec(
        planner_config,
        boundary,
        name_key="boundary_prefix_representation",
        alpha_key="boundary_prefix_ema_alpha",
        label="boundary-velocity",
        required=True,
    )
    return PlannerPrefixContract(
        encoder=encoder,
        boundary_velocity=boundary,
    )


__all__ = [
    "PREFIX_REPRESENTATION_SCHEMA",
    "PLANNER_PREFIX_CONTRACT_SCHEMA",
    "RAW_PREFIX_REPRESENTATION",
    "CAUSAL_EMA_PREFIX_REPRESENTATION",
    "SUPPORTED_PREFIX_REPRESENTATIONS",
    "DEFAULT_PREFIX_EMA_ALPHA",
    "PREFIX_WINDOW_POLICY",
    "CAUSAL_EMA_STATE_POLICY",
    "ENCODER_CONSUMER_POLICY",
    "BOUNDARY_VELOCITY_CONSUMER_POLICY",
    "PrefixRepresentationSpec",
    "PlannerPrefixContract",
    "apply_prefix_representation",
    "prefix_representation_from_config",
    "prefix_representation_from_payload",
    "planner_prefix_contract_from_config",
    "planner_prefix_contract_from_payload",
]
