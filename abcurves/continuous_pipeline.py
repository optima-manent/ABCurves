"""Compose angular planning with a persistent native-count renderer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .continuous import COMMON_RADIANS_PER_COUNT, ContinuousPlanner, RuntimeFailure
from .model_store import default_model_dir
from .portable_renderer import PortableRendererModel


@dataclass(frozen=True)
class CountTransform:
    """Fixed per-axis conversion between native and common angular counts.

    ``radians_per_count`` is the applied angular gain on each hardware axis.
    The planner uses x right / y up; ``y_down=True`` selects downward-positive
    hardware Y. Application/world projection and nonlinear control mappings
    belong to the caller; this helper represents a constant diagonal mapping.
    """
    radians_per_count: float | tuple[float, float]
    y_down: bool = True

    def __post_init__(self):
        scale = np.broadcast_to(np.asarray(self.radians_per_count, dtype=float), (2,))
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("radians_per_count must be one or two finite positive scales")
        with np.errstate(over="ignore", under="ignore"):
            factor = scale / COMMON_RADIANS_PER_COUNT
        if not np.isfinite(factor).all() or np.any(factor <= 0):
            raise ValueError("Angular conversion exceeds finite coordinate range")
        object.__setattr__(self, "radians_per_count", tuple(map(float, scale)))

    @property
    def common_per_native(self):
        scale = np.asarray(self.radians_per_count) / COMMON_RADIANS_PER_COUNT
        return scale * np.array([1., -1. if self.y_down else 1.])

    def to_common(self, reports):
        return np.asarray(reports, dtype=np.float64) * self.common_per_native

    def to_native(self, displacement):
        return np.asarray(displacement, dtype=np.float64) / self.common_per_native


class ContinuousPipelineFailure(RuntimeError):
    """Failure with all successfully rendered output in ``partial``.

    Reset starts a new stream after the caller handles the partial output.
    """
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


class ContinuousPipeline:
    """Load once; timestamp targets and obtain 1 kHz integer hardware reports.

    ``profile_reports`` is exactly 256 chronological genuine integer hardware
    reports. Planner history and initial position, if supplied, are in common
    angular counts. ``CountTransform`` supplies a fixed angular gain per axis.
    Model state follows its generated trajectory; ``rendered_xy`` accumulates
    emitted reports under that transform, without observing the application.
    """
    def __init__(self, profile_reports, *, transform: CountTransform,
                 renderer_seed=7, model_dir: str | Path | None = None,
                 renderer_library=None, renderer_model=None, observed_xy=None, **planner_options):
        if not isinstance(transform, CountTransform):
            raise TypeError("transform must be a CountTransform")
        root = default_model_dir() if model_dir is None else Path(model_dir)
        self.transform = transform
        self.movement = ContinuousPlanner(root / "continuous", **planner_options)
        if renderer_model is not None and renderer_library is not None:
            raise ValueError("Pass a renderer model or a library, not both")
        if renderer_model is not None and not isinstance(renderer_model, PortableRendererModel):
            raise TypeError("renderer_model must be a PortableRendererModel")
        self.renderer_model = renderer_model if renderer_model is not None else PortableRendererModel(
            root / "renderer_global_h80.bin", library=renderer_library)
        # Prepare the learned texture state in canonical x-right/y-up counts.
        canonical = np.asarray(profile_reports, dtype=np.float64).copy()
        if transform.y_down and canonical.ndim == 2 and canonical.shape[1] == 2:
            canonical[:, 1] *= -1
        self.profile = self.renderer_model.prepare_context(canonical)
        self.renderer_seed = renderer_seed
        self._observed_offset = np.zeros(2)
        self.reset(observed_xy=observed_xy)

    @classmethod
    def from_pretrained(cls, profile_reports, **kwargs):
        return cls(profile_reports, **kwargs)

    def reset(self, *, renderer_seed=None, observed_xy=None, **planner_options):
        """Reset both state machines; retain profile, sensitivity and seed defaults."""
        seed = self.renderer_seed if renderer_seed is None else renderer_seed
        observed = None if observed_xy is None else self.movement._check_position(observed_xy)
        stream = self.profile.begin_stream(event_seed=seed)
        self.movement.reset(**planner_options)
        self.renderer = stream
        self.renderer_seed = seed
        self._previous = self.movement.current_xy
        if observed is not None:
            self._observed_offset = observed - self._previous
        self._rendered = self._previous + self._observed_offset
        self.failed = False

    def update_target(self, xy, timestamp_us=None):
        if self.failed:
            raise RuntimeError("Pipeline failed; reset before updating targets")
        self.movement.update_target(xy, timestamp_us)

    @property
    def needs_plan(self):
        return self.movement.needs_plan

    @property
    def current_xy(self):
        return self.movement.current_xy

    @property
    def rendered_xy(self):
        return self._rendered.copy()

    def advance(self, timestamp_us=None):
        if self.failed:
            raise RuntimeError("Pipeline failed; inspect partial output and reset")
        failure = None
        try:
            planned = self.movement.advance(timestamp_us)
        except RuntimeFailure as exc:
            planned, failure = exc.partial, exc
        reports = np.empty((len(planned["xy"]), 2), dtype=np.int16)
        rendered = np.empty_like(planned["xy"])
        emitted = 0
        try:
            for point in planned["xy"]:
                delta = (point - self._previous) / np.abs(self.transform.common_per_native)
                report = self.renderer.step(delta)
                if self.transform.y_down:
                    report[1] = -report[1]
                self._previous = point.copy()
                self._rendered += self.transform.to_common(report)
                reports[emitted] = report
                rendered[emitted] = self._rendered
                emitted += 1
        except Exception as exc:
            failure = exc
        result = dict(time_us=planned["time_us"][:emitted],
                      xy=planned["xy"][:emitted], reports=reports[:emitted],
                      rendered_xy=rendered[:emitted])
        if failure is not None:
            self.failed = True
            raise ContinuousPipelineFailure(str(failure), result) from failure
        return result


__all__ = ["ContinuousPipeline", "CountTransform", "ContinuousPipelineFailure"]
