from __future__ import annotations

import numpy as np
import pytest

from abcurves import InferenceContractError, Pipeline


def _kwargs(row: dict[str, object], seed: int = 9123) -> dict[str, object]:
    return {
        "target_rel_at_B": row["target"],
        "target_radius": row["radius"],
        "progress_center": row["progress"],
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
    pending = pipeline7.begin_at_b(example_row["prefix"])
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
    with pytest.raises(Exception, match="complete"):
        stream.step()


def test_inputs_are_owned_at_event_boundary(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    prefix = np.array(example_row["prefix"], copy=True)
    pending = pipeline7.begin_at_b(prefix)
    prefix[:] = 12345.0
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


def test_causal_state_is_copied_and_clipped(
    pipeline7: Pipeline, example_row: dict[str, object]
) -> None:
    state = np.asarray([0.2, -0.4, 0.7], dtype=np.float32)
    pending = pipeline7.begin_at_b(example_row["prefix"])
    stream = pending.finish(
        target_rel_at_B=example_row["target"],
        target_radius=example_row["radius"],
        progress_center=example_row["progress"],
        planner_seed=5,
        renderer_event_seed_u64=5,
        causal_c_state=state,
    )
    state[:] = 0.0
    assert np.array_equal(
        stream.renderer.causal_c_state,
        np.asarray([0.2, -0.4, 0.7], dtype=np.float32),
    )
    with pytest.raises(Exception, match="clip"):
        pipeline7.generate(
            example_row["prefix"],
            **_kwargs(example_row, 5),
            causal_c_state=np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
        )


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
        )


def test_seed23_replication_loads(example_row: dict[str, object]) -> None:
    with Pipeline(model_seed=23, prewarm=True) as runtime:
        output = runtime.generate(example_row["prefix"], **_kwargs(example_row, 23))
    assert output.dtype == np.int16 and output.shape[1] == 2
