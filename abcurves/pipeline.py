"""The final Planner -> global Renderer streaming pipeline.

The Planner reads the human A->B movement and chooses one smooth finish.  In
parallel, the global Renderer observes exactly 256 chronological hardware
reports.  The default fixed-point C core performs its bounded online handoff
and emits one integer report per millisecond.  A checkpoint produced by the
public trainer can instead be supplied explicitly; that research backend
replays the same context through the float graph and pre-renders the event.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .fast_planner import NumbaPlannerForward, _tcn_forward
from .fast_summary import CompiledSummaryVector
from .model_store import resolve_model_files
from .planner import Intent, Planner
from .portable_renderer import (
    CONTEXT_TICKS,
    PortableRendererEvent,
    PortableRendererModel,
    PreparedRendererContext,
)
from .renderer import (
    FloatRendererEvent,
    FloatRendererModel,
    PreparedFloatRendererContext,
)


class InferenceContractError(RuntimeError):
    """Raised when a model or live input violates the release contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceContractError(message)


class _SelectedHeadForward:
    """Run the shared Planner graph and only one preselected output slice."""

    def __init__(self, all_heads: NumbaPlannerForward) -> None:
        self.base = all_heads
        _require(
            all_heads.heads == 16 and all_heads.out_dim == 43,
            "release Planner output geometry differs",
        )
        self.weights = np.ascontiguousarray(
            all_heads.w_out_t.reshape(
                all_heads.hidden, all_heads.heads, all_heads.out_dim
            ).transpose(1, 0, 2),
            dtype=np.float32,
        )
        self.biases = np.ascontiguousarray(
            all_heads.b_out.reshape(all_heads.heads, all_heads.out_dim),
            dtype=np.float32,
        )

    def forward(
        self,
        prefix_tensor: np.ndarray,
        summary: np.ndarray,
        head: int,
    ) -> np.ndarray:
        chosen = int(head)
        _require(0 <= chosen < self.base.heads, "Planner head is out of range")
        window = np.ascontiguousarray(
            prefix_tensor[0, -self.base.window :, :], dtype=np.float32
        )
        output = _tcn_forward(
            window,
            np.ascontiguousarray(summary[0], dtype=np.float32),
            self.base.w_in_t,
            self.base.b_in,
            self.base.conv_w,
            self.base.conv_b,
            self.base.ln_w,
            self.base.ln_b,
            self.base.ln_eps,
            self.base.w_sum_t,
            self.base.b_sum,
            self.base.w_trunk_t,
            self.base.b_trunk,
            self.weights[chosen],
            self.biases[chosen],
        )
        return output.reshape(1, 1, self.base.out_dim)


@dataclass(frozen=True)
class PlannerTiming:
    encode_us: float
    forward_us: float
    decode_us: float
    total_us: float


@dataclass(frozen=True)
class PlannedIntent:
    intent: Intent
    timing: PlannerTiming
    backend: str = "numba_selected_head_float32"


class FastPlanner:
    """Checkpoint-exact input/decode with the compiled selected-head TCN."""

    def __init__(self, path: str | Path, *, prewarm: bool = True) -> None:
        self.planner = Planner(path, device="cpu", prewarm_decode=prewarm)
        _require(self.planner.heads == 16, "release Planner must have 16 heads")
        _require(self.planner.horizon == 1000, "release Planner horizon differs")
        _require(self.planner.prefix_len == 160, "release Planner prefix differs")
        contract = self.planner.seam_contract or {}
        trigger = contract.get("trigger", {})
        _require(
            contract.get("schema") == "abcurves.causal_onset_b.v1"
            and trigger.get("reference") == "edge"
            and tuple(trigger.get("thresholds", ())) == (0.8,),
            "Planner causal B contract differs from the release",
        )
        self.summary = CompiledSummaryVector(self.planner)
        self.all_heads = NumbaPlannerForward(self.planner.model)
        self.selected_head = _SelectedHeadForward(self.all_heads)
        if prewarm:
            self.summary.prewarm()
            prefix = np.zeros((1, self.planner.prefix_len, 3), dtype=np.float32)
            summary = np.zeros((1, len(self.planner.summary_names)), dtype=np.float32)
            self.selected_head.forward(prefix, summary, 0)

    @property
    def prefix_len(self) -> int:
        return self.planner.prefix_len

    def plan(
        self,
        prefix_raw_dxdy: np.ndarray,
        *,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress_center: float,
        seed: int,
        head: int | None = None,
        b_index_ms: float | None = None,
    ) -> PlannedIntent:
        started = time.perf_counter_ns()
        prefix = np.asarray(prefix_raw_dxdy)
        _require(
            prefix.ndim == 2
            and prefix.shape[1] == 2
            and len(prefix) > 0
            and bool(np.all(np.isfinite(prefix))),
            "Planner prefix must be a non-empty finite array with shape (P, 2)",
        )
        target = np.asarray(target_rel_at_B, dtype=np.float64).reshape(-1)
        _require(
            target.shape == (2,) and bool(np.all(np.isfinite(target))),
            "target_rel_at_B must contain two finite count-space values",
        )
        radius = float(target_radius)
        progress = float(progress_center)
        _require(np.isfinite(radius) and radius > 0.0, "target_radius must be positive")
        _require(
            np.isfinite(progress) and 0.0 <= progress <= 1.0,
            "progress_center must lie in [0, 1]",
        )
        if b_index_ms is not None:
            _require(
                np.isfinite(float(b_index_ms)) and float(b_index_ms) >= 0.0,
                "b_index_ms must be finite and non-negative",
            )

        represented, boundary = self.planner.represented_prefix_views(prefix)
        summary = self.summary.vector(
            represented,
            (float(target[0]), float(target[1])),
            radius,
            progress,
            b_index_ms=b_index_ms,
            assume_finite_counts=True,
        )
        prefix_tensor, _ = self.planner._prefix_tensor(represented)
        boundary_velocity = (
            boundary[-1].astype(np.float64)
            if len(boundary)
            else np.zeros(2, dtype=np.float64)
        )
        encoded = time.perf_counter_ns()

        if head is None:
            chosen = int(np.random.default_rng(int(seed)).integers(0, 16))
        else:
            chosen = int(head)
            _require(0 <= chosen < 16, "Planner head is out of range")
        prediction = self.selected_head.forward(prefix_tensor, summary, chosen)
        forwarded = time.perf_counter_ns()
        smooth, mask = self.planner.decode_head(prediction, 0, boundary_velocity)
        finished = time.perf_counter_ns()

        intent = Intent(
            smooth_dxdy=np.asarray(smooth, dtype=np.float32),
            mask=np.asarray(mask, dtype=np.float32),
            duration_ms=int(np.sum(mask > 0.5)),
            head=chosen,
        )
        return PlannedIntent(
            intent=intent,
            timing=PlannerTiming(
                encode_us=(encoded - started) / 1_000.0,
                forward_us=(forwarded - encoded) / 1_000.0,
                decode_us=(finished - forwarded) / 1_000.0,
                total_us=(finished - started) / 1_000.0,
            ),
        )


@dataclass(frozen=True)
class PipelineTiming:
    b_to_context_submit_us: float
    finish_planner_us: float
    wait_for_renderer_context_us: float
    renderer_begin_us: float
    finish_total_us: float


@dataclass
class PreparedStream:
    """A state-owning, one-tick-at-a-time Renderer stream."""

    renderer: PortableRendererEvent | FloatRendererEvent
    planned: PlannedIntent
    timing: PipelineTiming

    @property
    def duration_ms(self) -> int:
        return self.renderer.duration

    @property
    def complete(self) -> bool:
        return self.renderer.complete

    def step(self) -> np.ndarray:
        return self.renderer.step()

    def render_remaining(self) -> np.ndarray:
        return self.renderer.render_remaining().copy()


class PendingB:
    """Planner work overlapped with the 256-report Renderer observation."""

    def __init__(
        self,
        owner: "Pipeline",
        prefix: np.ndarray,
        context_future: Future[
            PreparedRendererContext | PreparedFloatRendererContext
        ],
        started_ns: int,
        submitted_ns: int,
    ) -> None:
        self.owner = owner
        self.prefix = prefix
        self.context_future = context_future
        self.started_ns = int(started_ns)
        self.submitted_ns = int(submitted_ns)
        self._finished = False

    def finish(
        self,
        *,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress_center: float,
        planner_seed: int,
        renderer_event_seed_u64: int,
        planner_head: int | None = None,
        b_index_ms: float | None = None,
    ) -> PreparedStream:
        _require(not self._finished, "PendingB.finish() may only be called once")
        self._finished = True
        planned = self.owner.planner.plan(
            self.prefix,
            target_rel_at_B=target_rel_at_B,
            target_radius=target_radius,
            progress_center=progress_center,
            seed=planner_seed,
            head=planner_head,
            b_index_ms=b_index_ms,
        )
        planner_done = time.perf_counter_ns()
        prepared = self.context_future.result()
        context_done = time.perf_counter_ns()
        event = prepared.begin(
            planned.intent.smooth_dxdy,
            planned.intent.mask,
            event_seed=int(renderer_event_seed_u64),
        )
        finished = time.perf_counter_ns()
        return PreparedStream(
            renderer=event,
            planned=planned,
            timing=PipelineTiming(
                b_to_context_submit_us=(self.submitted_ns - self.started_ns) / 1_000.0,
                finish_planner_us=(planner_done - self.submitted_ns) / 1_000.0,
                wait_for_renderer_context_us=(context_done - planner_done) / 1_000.0,
                renderer_begin_us=(finished - context_done) / 1_000.0,
                finish_total_us=(finished - self.started_ns) / 1_000.0,
            ),
        )


class Pipeline:
    """Load-once final pipeline.

    ``model_seed`` chooses one of the two independently trained Planners.  The
    selected global Renderer is shared by both Planner cells.  Pass
    ``float_renderer_checkpoint`` to use a newly trained PyTorch checkpoint
    through the same API without claiming it is the authenticated native image.
    Either Renderer context is exactly 256 chronological hardware reports; when
    the Planner prefix is shorter, pass it separately with
    ``renderer_context_raw_dxdy``.
    """

    def __init__(
        self,
        model_seed: int = 7,
        *,
        model_dir: str | Path | None = None,
        renderer_library: str | Path | None = None,
        float_renderer_checkpoint: str | Path | None = None,
        float_renderer_device: str = "cpu",
        float_renderer_prefix_smoothing_spec: str | None = None,
        verify_models: bool = True,
        prewarm: bool = True,
        torch_threads: int | None = 1,
    ) -> None:
        if torch_threads is not None:
            _require(int(torch_threads) >= 1, "torch_threads must be positive")
            torch.set_num_threads(int(torch_threads))
        self.model_files = resolve_model_files(
            model_seed, model_dir=model_dir, verify=verify_models
        )
        self.model_seed = self.model_files.seed
        self.planner = FastPlanner(self.model_files.planner, prewarm=prewarm)
        if float_renderer_checkpoint is None:
            self.renderer = PortableRendererModel(
                self.model_files.renderer,
                library=renderer_library,
                verify=verify_models,
            )
        else:
            _require(
                renderer_library is None,
                "renderer_library cannot be combined with a float Renderer checkpoint",
            )
            self.renderer = FloatRendererModel(
                float_renderer_checkpoint,
                device=float_renderer_device,
                prefix_smoothing_spec=float_renderer_prefix_smoothing_spec,
            )
        self.renderer_receipt = asdict(self.renderer.receipt)
        self._context_worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="abcurves-renderer-context",
        )
        self._closed = False
        if prewarm:
            context = np.zeros((CONTEXT_TICKS, 2), dtype=np.int16)
            self._context_worker.submit(self.renderer.prepare_context, context).result()

    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> "Pipeline":
        return cls(model_seed=7, **kwargs)

    def begin_at_b(
        self,
        prefix_raw_dxdy: np.ndarray,
        *,
        renderer_context_raw_dxdy: np.ndarray | None = None,
    ) -> PendingB:
        """Freeze A->B and start the exact 256-report Renderer observation."""

        _require(not self._closed, "Pipeline is closed")
        started = time.perf_counter_ns()
        prefix = np.array(prefix_raw_dxdy, dtype=np.float32, order="C", copy=True)
        _require(
            prefix.ndim == 2
            and prefix.shape[1] == 2
            and len(prefix) > 0
            and bool(np.all(np.isfinite(prefix))),
            "B prefix must be a non-empty finite array with shape (P, 2)",
        )
        if renderer_context_raw_dxdy is None:
            _require(
                len(prefix) == CONTEXT_TICKS,
                "global Renderer needs exactly 256 chronological reports; pass "
                "renderer_context_raw_dxdy when the Planner prefix has another length",
            )
            context = np.array(prefix, copy=True)
        else:
            context = np.array(renderer_context_raw_dxdy, copy=True)
        prefix.setflags(write=False)
        context.setflags(write=False)
        prepared = self._context_worker.submit(self.renderer.prepare_context, context)
        submitted = time.perf_counter_ns()
        return PendingB(self, prefix, prepared, started, submitted)

    def prepare(
        self,
        prefix_raw_dxdy: np.ndarray,
        *,
        renderer_context_raw_dxdy: np.ndarray | None = None,
        **finish_kwargs: Any,
    ) -> PreparedStream:
        return self.begin_at_b(
            prefix_raw_dxdy,
            renderer_context_raw_dxdy=renderer_context_raw_dxdy,
        ).finish(**finish_kwargs)

    def generate(
        self,
        prefix_raw_dxdy: np.ndarray,
        *,
        target_rel_at_B: tuple[float, float],
        target_radius: float,
        progress_center: float,
        seed: int = 7,
        renderer_context_raw_dxdy: np.ndarray | None = None,
        planner_head: int | None = None,
        b_index_ms: float | None = None,
    ) -> np.ndarray:
        event_seed = int(seed)
        _require(0 <= event_seed < 2**64, "seed must fit in uint64")
        stream = self.prepare(
            prefix_raw_dxdy,
            renderer_context_raw_dxdy=renderer_context_raw_dxdy,
            target_rel_at_B=target_rel_at_B,
            target_radius=target_radius,
            progress_center=progress_center,
            planner_seed=event_seed,
            renderer_event_seed_u64=event_seed,
            planner_head=planner_head,
            b_index_ms=b_index_ms,
        )
        return stream.render_remaining()

    def close(self) -> None:
        if not self._closed:
            self._context_worker.shutdown(wait=True, cancel_futures=False)
            self._closed = True

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


__all__ = [
    "FastPlanner",
    "InferenceContractError",
    "PendingB",
    "Pipeline",
    "PipelineTiming",
    "PlannedIntent",
    "PlannerTiming",
    "PreparedStream",
]
