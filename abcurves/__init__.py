"""ABCurves: continuous movement, observed-prefix continuation and mouse texture.

Components load lazily so Continuous inference needs no training framework.
"""
from importlib import import_module

__version__ = "2.0.0"

_EXPORTS = {
    "load": ("continuous", "load"),
    "prepare_history": ("continuous", "prepare_history"),
    "HumanStart": ("continuous", "HumanStart"),
    "ContinuousPlanner": ("continuous", "ContinuousPlanner"),
    "ContinuousPipeline": ("continuous_pipeline", "ContinuousPipeline"),
    "Pipeline": ("continuous_pipeline", "ContinuousPipeline"),
    "CountTransform": ("continuous_pipeline", "CountTransform"),
    "ContinuousPipelineFailure": ("continuous_pipeline", "ContinuousPipelineFailure"),
    "StaticPipeline": ("pipeline", "Pipeline"),
    "StaticPlanner": ("pipeline", "FastPlanner"),
    "Planner": ("planner", "Planner"),
    "Intent": ("planner", "Intent"),
    "ProDMP": ("prodmp", "ProDMP"),
    "ProDMPConfig": ("prodmp", "ProDMPConfig"),
    "ModelIntegrityError": ("model_store", "ModelIntegrityError"),
    **{n: ("pipeline", n) for n in ("InferenceContractError", "PendingB", "PreparedStream", "RendererProfile")},
    **{n: ("capture_preprocess", n) for n in ("CausalBConfig", "CausalOnsetConfig", "SeamEligibility", "ShotFilterPolicy")},
    **{n: ("seam", n) for n in ("BFire", "BReject", "BTrigger", "OnsetDetector", "OnsetEvent")},
    **{n: ("portable_renderer", n) for n in ("PortableRendererEvent", "PortableRendererModel", "PortableRendererStream", "PreparedRendererContext", "RendererRuntimeError")},
    **{n: ("renderer", n) for n in ("FloatRendererEvent", "FloatRendererModel", "PreparedFloatRendererContext")},
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), attribute)
    globals()[name] = value
    return value
