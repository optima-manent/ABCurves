from __future__ import annotations

import numpy as np
import pytest

from abcurves import InferenceContractError, Pipeline, RendererRuntimeError
from abcurves.features import summary_features_for_row
from abcurves.planner import Planner


def _kwargs(row: dict[str, object], seed: int = 9123) -> dict[str, object]:
    return {
        "target_rel_at_B": row["target"],
        "target_radius": row["radius"],
        "progress_center": row["progress"],
        "renderer_context_raw_dxdy": row["renderer_context"],
        "seed": seed,
    }


def test_generation_is_integer_deterministic_and_nonempty(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    first = pipeline7.generate(example_row["prefix"], **_kwargs(example_row))
    second = pipeline7.generate(example_row["prefix"], **_kwargs(example_row))
    assert first.dtype == np.int16
    assert first.ndim == 2 and first.shape[1] == 2 and len(first) > 0
    assert np.array_equal(first, second)


def test_streaming_matches_bulk_output(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    seed = 44
    pending = pipeline7.begin_at_b(
        example_row["prefix"],
        renderer_context_raw_dxdy=example_row["renderer_context"],
    )
    stream = pending.finish(
        target_rel_at_B=example_row["target"],
        target_radius=example_row["radius"],
        progress_center=example_row["progress"],
        planner_seed=seed,
        renderer_event_seed_u64=seed,
    )
    ticks = np.stack([stream.step() for _ in range(stream.duration_ms)])
    expected = pipeline7.generate(example_row["prefix"], **_kwargs(example_row, seed))
    assert stream.complete
    assert np.array_equal(ticks, expected)
    with pytest.raises(RendererRuntimeError, match="complete"):
        stream.step()


def test_inputs_are_owned_at_event_boundary(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    prefix = np.array(example_row["prefix"], copy=True)
    context = np.array(example_row["renderer_context"], copy=True)
    pending = pipeline7.begin_at_b(
        prefix, renderer_context_raw_dxdy=context
    )
    prefix[:] = 12345.0
    context[:] = 12345
    stream = pending.finish(
        target_rel_at_B=example_row["target"],
        target_radius=example_row["radius"],
        progress_center=example_row["progress"],
        planner_seed=71,
        renderer_event_seed_u64=71,
    )
    observed = stream.render_remaining()
    expected = pipeline7.generate(example_row["prefix"], **_kwargs(example_row, 71))
    assert np.array_equal(observed, expected)


def test_renderer_context_is_exact_and_never_implicitly_sliced(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    with pytest.raises(RendererRuntimeError, match="exactly"):
        pipeline7.begin_at_b(
            example_row["prefix"],
            renderer_context_raw_dxdy=np.zeros((255, 2), dtype=np.int16),
        ).context_future.result()
    with pytest.raises(RendererRuntimeError, match="exactly"):
        pipeline7.begin_at_b(
            example_row["prefix"],
            renderer_context_raw_dxdy=np.zeros((257, 2), dtype=np.int16),
        ).context_future.result()
    with pytest.raises(InferenceContractError, match="exactly 256"):
        pipeline7.begin_at_b(example_row["prefix"])


def test_invalid_live_inputs_fail_closed(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    with pytest.raises(InferenceContractError):
        pipeline7.generate(
            np.zeros((0, 2), dtype=np.float32), **_kwargs(example_row)
        )
    with pytest.raises(InferenceContractError):
        pipeline7.generate(
            example_row["prefix"],
            target_rel_at_B=example_row["target"],
            target_radius=-1.0,
            progress_center=example_row["progress"],
            renderer_context_raw_dxdy=example_row["renderer_context"],
        )


def test_reference_planner_rejects_an_invalid_head_before_inference() -> None:
    planner = Planner.__new__(Planner)
    planner.heads = 16
    with pytest.raises(ValueError, match="head must be"):
        planner.plan(
            np.zeros((1, 2), dtype=np.float32),
            target_rel_at_B=(1.0, 0.0),
            target_radius=1.0,
            progress=0.0,
            head=99,
        )


@pytest.mark.parametrize(
    ("prefix", "target", "radius", "progress"),
    [
        (np.zeros((0, 2)), (1.0, 0.0), 1.0, 0.0),
        (np.array([[np.nan, 0.0]]), (1.0, 0.0), 1.0, 0.0),
        (np.zeros((1, 2)), (np.nan, 0.0), 1.0, 0.0),
        (np.zeros((1, 2)), (1.0, 0.0), -1.0, 0.0),
        (np.zeros((1, 2)), (1.0, 0.0), 1.0, 2.0),
    ],
)
def test_reference_planner_rejects_invalid_inputs_before_inference(
    prefix: np.ndarray,
    target: tuple[float, float],
    radius: float,
    progress: float,
) -> None:
    planner = Planner.__new__(Planner)
    planner.heads = 16
    with pytest.raises(ValueError):
        planner.plan(
            prefix,
            target_rel_at_B=target,
            target_radius=radius,
            progress=progress,
            head=0,
        )


def test_summary_row_cannot_substitute_zero_for_missing_geometry() -> None:
    with pytest.raises(ValueError, match="target geometry and progress"):
        summary_features_for_row({}, np.zeros((1, 2), dtype=np.float32))


def test_seed23_replication_shares_the_renderer(
    example_row: dict[str, object]
) -> None:
    with Pipeline(model_seed=23, prewarm=True) as runtime:
        output = runtime.generate(example_row["prefix"], **_kwargs(example_row, 23))
        renderer_hash = runtime.renderer_receipt["artifact_sha256"]
    assert output.dtype == np.int16 and output.shape[1] == 2
    assert renderer_hash == (
        "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b"
    )
