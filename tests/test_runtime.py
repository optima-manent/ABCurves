from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from abcurves import (
    InferenceContractError,
    Pipeline,
    RendererProfile,
    RendererRuntimeError,
)
from abcurves.features import summary_features_for_row
from abcurves.planner import Planner


def _kwargs(
    row: dict[str, object],
    seed: int = 9123,
    *,
    profile: RendererProfile | None = None,
) -> dict[str, object]:
    return {
        "target_rel_at_B": row["target"],
        "target_radius": row["radius"],
        "progress_center": row["progress"],
        **(
            {"renderer_context_raw_dxdy": row["renderer_context"]}
            if profile is None
            else {"renderer_profile": profile}
        ),
        "seed": seed,
    }


def test_generation_is_integer_deterministic_and_nonempty(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    kwargs = _kwargs(example_row, profile=renderer_profile7)
    first = pipeline7.generate(example_row["prefix"], **kwargs)
    second = pipeline7.generate(example_row["prefix"], **kwargs)
    assert first.dtype == np.int16
    assert first.ndim == 2 and first.shape[1] == 2 and len(first) > 0
    assert np.array_equal(first, second)


def test_streaming_matches_bulk_output(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    seed = 44
    pending = pipeline7.begin_at_b(
        example_row["prefix"],
        renderer_profile=renderer_profile7,
    )
    stream = pending.finish(
        target_rel_at_B=example_row["target"],
        target_radius=example_row["radius"],
        progress_center=example_row["progress"],
        planner_seed=seed,
        renderer_event_seed_u64=seed,
    )
    ticks = np.stack([stream.step() for _ in range(stream.duration_ms)])
    expected = pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, seed, profile=renderer_profile7),
    )
    assert stream.complete
    assert np.array_equal(ticks, expected)
    with pytest.raises(RendererRuntimeError, match="complete"):
        stream.step()


def test_inputs_are_owned_at_event_boundary(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    prefix = np.array(example_row["prefix"], copy=True)
    context = np.array(example_row["renderer_context"], copy=True)
    profile = pipeline7.prepare_renderer_profile(context)
    pending = pipeline7.begin_at_b(prefix, renderer_profile=profile)
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
    expected_profile = pipeline7.prepare_renderer_profile(
        example_row["renderer_context"]
    )
    expected = pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, 71, profile=expected_profile),
    )
    assert np.array_equal(observed, expected)


def test_renderer_context_is_exact_and_never_implicitly_sliced(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    with pytest.raises(RendererRuntimeError, match="exactly"):
        pipeline7.prepare_renderer_profile(np.zeros((255, 2), dtype=np.int16))
    with pytest.raises(RendererRuntimeError, match="exactly"):
        pipeline7.prepare_renderer_profile(np.zeros((257, 2), dtype=np.int16))
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
    with pytest.raises(InferenceContractError, match="prepared 256-report profile"):
        pipeline7.begin_at_b(example_row["prefix"])

    implicit_context = np.asarray(example_row["renderer_context"], dtype=np.float32)
    implicit = pipeline7.generate(
        implicit_context,
        target_rel_at_B=example_row["target"],
        target_radius=example_row["radius"],
        progress_center=example_row["progress"],
        seed=441,
    )
    exact = pipeline7.generate(
        implicit_context,
        renderer_context_raw_dxdy=implicit_context,
        target_rel_at_B=example_row["target"],
        target_radius=example_row["radius"],
        progress_center=example_row["progress"],
        seed=441,
    )
    assert np.array_equal(implicit, exact)


def test_profile_matches_exact_context_and_remains_reusable(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    seed = 604
    profiled = pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, seed, profile=renderer_profile7),
    )
    exact = pipeline7.generate(example_row["prefix"], **_kwargs(example_row, seed))
    repeated = pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, seed, profile=renderer_profile7),
    )
    assert np.array_equal(profiled, exact)
    assert np.array_equal(profiled, repeated)


def test_profile_refresh_creates_an_independent_snapshot(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    replacement_window = np.array(example_row["renderer_context"], copy=True)
    replacement_window[0] = (7, -5)
    replacement = pipeline7.prepare_renderer_profile(replacement_window)
    assert replacement is not renderer_profile7

    seed = 730
    before = pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, seed, profile=renderer_profile7),
    )
    pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, seed, profile=replacement),
    )
    after = pipeline7.generate(
        example_row["prefix"],
        **_kwargs(example_row, seed, profile=renderer_profile7),
    )
    assert np.array_equal(before, after)


def test_profile_and_exact_context_are_mutually_exclusive(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    with pytest.raises(InferenceContractError, match="mutually exclusive"):
        pipeline7.begin_at_b(
            example_row["prefix"],
            renderer_profile=renderer_profile7,
            renderer_context_raw_dxdy=example_row["renderer_context"],
        )


def test_profile_cannot_cross_pipeline_instances(
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    with Pipeline(model_seed=23, prewarm=False) as other:
        with pytest.raises(InferenceContractError, match="another Pipeline"):
            other.begin_at_b(
                example_row["prefix"], renderer_profile=renderer_profile7
            )


def test_malformed_profile_fails_before_planning(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    malformed = RendererProfile(
        _owner=pipeline7,
        _prepared=None,  # type: ignore[arg-type]
    )
    with pytest.raises(InferenceContractError, match="invalid"):
        pipeline7.begin_at_b(example_row["prefix"], renderer_profile=malformed)


def test_renderer_profile_supports_concurrent_independent_pipeline_events(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    def render(seed: int) -> np.ndarray:
        return pipeline7.generate(
            example_row["prefix"],
            **_kwargs(example_row, seed, profile=renderer_profile7),
        )

    with ThreadPoolExecutor(max_workers=4) as workers:
        same = list(workers.map(render, [91, 91, 91, 91]))
    assert all(np.array_equal(same[0], value) for value in same[1:])
    assert np.array_equal(render(91), same[0])


def test_invalid_live_inputs_fail_closed(
    pipeline7: Pipeline,
    renderer_profile7: RendererProfile,
    example_row: dict[str, object],
) -> None:
    with pytest.raises(InferenceContractError):
        pipeline7.generate(
            np.zeros((0, 2), dtype=np.float32),
            **_kwargs(example_row, profile=renderer_profile7),
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
        profile = runtime.prepare_renderer_profile(example_row["renderer_context"])
        output = runtime.generate(
            example_row["prefix"],
            **_kwargs(example_row, 23, profile=profile),
        )
        renderer_hash = runtime.renderer_receipt["artifact_sha256"]
    assert output.dtype == np.int16 and output.shape[1] == 2
    assert renderer_hash == (
        "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b"
    )
