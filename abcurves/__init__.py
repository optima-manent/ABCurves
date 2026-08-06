"""ABCurves: human-conditioned A -> B -> C movement continuation."""

from . import judges
from .capture_preprocess import (
    CausalBConfig,
    CausalOnsetConfig,
    SeamEligibility,
    ShotFilterPolicy,
)
from .model_store import ModelIntegrityError
from .personalization import CausalStyleState, zero_causal_state
from .pipeline import (
    InferenceContractError,
    PendingB,
    Pipeline,
    PreparedStream,
)
from .planner import Intent, Planner
from .prodmp import ProDMP, ProDMPConfig
from .seam import BFire, BReject, BTrigger, OnsetDetector, OnsetEvent
from .style_scorer import (
    FrozenStyleScorer,
    causal_context_features,
    completed_human_context,
    planned_trajectory_features,
    renderer_context,
)

__version__ = "1.0.0"

__all__ = [
    "BFire",
    "BReject",
    "BTrigger",
    "CausalBConfig",
    "CausalOnsetConfig",
    "CausalStyleState",
    "FrozenStyleScorer",
    "InferenceContractError",
    "Intent",
    "ModelIntegrityError",
    "OnsetDetector",
    "OnsetEvent",
    "PendingB",
    "Pipeline",
    "Planner",
    "PreparedStream",
    "ProDMP",
    "ProDMPConfig",
    "SeamEligibility",
    "ShotFilterPolicy",
    "judges",
    "causal_context_features",
    "completed_human_context",
    "planned_trajectory_features",
    "renderer_context",
    "zero_causal_state",
]
