"""ABCurves: plan a human continuation, then restore hardware texture."""

from . import judges
from .capture_preprocess import (
    CausalBConfig,
    CausalOnsetConfig,
    SeamEligibility,
    ShotFilterPolicy,
)
from .model_store import ModelIntegrityError
from .pipeline import (
    InferenceContractError,
    PendingB,
    Pipeline,
    PreparedStream,
    RendererProfile,
)
from .planner import Intent, Planner
from .portable_renderer import (
    PortableRendererEvent,
    PortableRendererModel,
    PreparedRendererContext,
    RendererRuntimeError,
)
from .renderer import (
    FloatRendererEvent,
    FloatRendererModel,
    PreparedFloatRendererContext,
)
from .prodmp import ProDMP, ProDMPConfig
from .seam import BFire, BReject, BTrigger, OnsetDetector, OnsetEvent

__version__ = "1.5.1"

__all__ = [
    "BFire",
    "BReject",
    "BTrigger",
    "CausalBConfig",
    "CausalOnsetConfig",
    "InferenceContractError",
    "Intent",
    "FloatRendererEvent",
    "FloatRendererModel",
    "ModelIntegrityError",
    "OnsetDetector",
    "OnsetEvent",
    "PendingB",
    "Pipeline",
    "Planner",
    "PortableRendererEvent",
    "PortableRendererModel",
    "PreparedRendererContext",
    "PreparedFloatRendererContext",
    "PreparedStream",
    "RendererProfile",
    "ProDMP",
    "ProDMPConfig",
    "SeamEligibility",
    "ShotFilterPolicy",
    "RendererRuntimeError",
    "judges",
]
