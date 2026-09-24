from __future__ import annotations

import gc
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from abcurves import StaticPipeline as Pipeline, RendererRuntimeError
from abcurves.portable_renderer import PortableRendererEvent, PortableRendererModel, _q16_pair
from abcurves.renderer import FloatRendererEvent


def _array_q16(value: np.ndarray) -> tuple[int, int]:
    """Independent vector formulation of the float32 -> signed Q16 law."""
    pair = np.asarray(value, dtype=np.float32).reshape(-1)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise RendererRuntimeError("smooth intent tick must contain two finite values")
    scaled = np.rint(pair.astype(np.float64) * 65536.0)
    if np.any(scaled < -(2**31)) or np.any(scaled > 2**31 - 1):
        raise RendererRuntimeError("smooth intent tick exceeds signed Q16 range")
    return int(scaled[0]), int(scaled[1])


def test_q16_rounding_staging_and_range_match_the_conversion_law() -> None:
    generator = np.random.default_rng(923)
    random_bits = generator.integers(0, 2**32, (12_000, 2), dtype=np.uint32).view(np.float32)
    halfway = (np.arange(-2048, 2049, dtype=np.float64) + 0.5) / 65536.0
    edges = np.array([
        0.0, -0.0, -32768.0, 32768.0,
        np.nextafter(np.float32(32768), np.float32(0)),
        np.nextafter(np.float32(-32768), np.float32(-np.inf)),
        np.nextafter(np.float32(0), np.float32(1)),
        np.inf, -np.inf, np.nan,
    ], np.float32)
    pairs = np.concatenate((random_bits, np.column_stack((halfway, -halfway)).astype(np.float32),
                            np.column_stack((edges, edges))))
    for pair in pairs:
        try:
            expected = _array_q16(pair)
        except RendererRuntimeError as error:
            with pytest.raises(RendererRuntimeError, match=str(error)):
                _q16_pair(pair)
        else:
            assert _q16_pair(pair) == expected
    # Rounding must follow float32 staging even for double/list inputs.
    value = [(0.5 + 1e-9) / 65536, -(0.5 + 1e-9) / 65536]
    assert _q16_pair(value) == (0, 0)
    assert _q16_pair(np.array([[1.5, -2.5]]) / 65536) == (2, -2)


@pytest.mark.parametrize("value", [[], [1.0], [1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]]])
def test_q16_rejects_wrong_tick_size(value: object) -> None:
    with pytest.raises(RendererRuntimeError, match="two finite values"):
        _q16_pair(value)


def test_event_owns_its_model_and_returns_independent_reports() -> None:
    artifact = Path(__file__).resolve().parents[1] / "models/renderer_global_h80.bin"
    model = PortableRendererModel(artifact)
    model_ref = weakref.ref(model)
    context = np.zeros((256, 2), np.int16)
    context[1::2, 0] = 1
    profile = model.prepare_context(context)
    event = profile.begin(np.tile((1., .5), (16, 1)), np.ones(16), event_seed=123)
    del model, profile
    gc.collect()
    assert model_ref() is not None
    first = event.step()
    expected = first.copy()
    event.render_remaining()
    assert np.array_equal(first, expected)
    assert np.array_equal(first, [1, 0])
    del event
    gc.collect()
    assert model_ref() is None


@pytest.mark.parametrize("backend", ["native", "float"])
def test_prewarm_disposes_the_event_and_preserves_seeded_behavior(
    backend: str, example_row: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = {}
    event_type = PortableRendererEvent
    if backend == "float":
        kwargs["float_renderer_checkpoint"] = (
            Path(__file__).resolve().parents[1] / "models/renderer_global_h80_float.pt"
        )
        event_type = FloatRendererEvent
    stepped = []
    original_step = event_type.step

    def step(self):
        stepped.append(weakref.ref(self))
        return original_step(self)

    monkeypatch.setattr(event_type, "step", step)
    outputs = []
    rng_states = []
    for prewarm in (False, True):
        torch.manual_seed(319)
        np.random.seed(319)
        stepped.clear()
        with Pipeline(prewarm=prewarm, **kwargs) as pipeline:
            rng_states.append((torch.random.get_rng_state().clone(), np.random.get_state()))
            assert len(stepped) == int(prewarm)
            gc.collect()
            assert all(ref() is None for ref in stepped)
            profile = pipeline.prepare_renderer_profile(example_row["renderer_context"])
            outputs.append(pipeline.generate(
                example_row["prefix"], renderer_profile=profile,
                target_rel_at_B=example_row["target"], target_radius=example_row["radius"],
                progress_center=example_row["progress"], seed=0,
            ))
    assert np.array_equal(outputs[0], outputs[1])
    assert torch.equal(rng_states[0][0], rng_states[1][0])
    for left, right in zip(rng_states[0][1], rng_states[1][1]):
        assert np.array_equal(left, right)


def test_prewarm_failure_shuts_down_its_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []
    close = Pipeline.close

    def fail(self):
        raise RuntimeError("prewarm step failed")

    def record_close(self):
        close(self)
        closed.append(self)

    monkeypatch.setattr(PortableRendererEvent, "step", fail)
    monkeypatch.setattr(Pipeline, "close", record_close)
    with pytest.raises(RuntimeError, match="prewarm step failed"):
        Pipeline(prewarm=True)
    assert len(closed) == 1 and closed[0]._closed
    assert all(not thread.is_alive() for thread in closed[0]._context_worker._threads)
