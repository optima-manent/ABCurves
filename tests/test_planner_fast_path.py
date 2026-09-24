from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import numpy as np
import pytest

from abcurves.pipeline import FastPlanner
from abcurves.planner import Planner, _CachedProDMP, decode_heads
from abcurves.prodmp import ProDMP, ProDMPConfig


ROOT = Path(__file__).resolve().parents[1]


def assert_same_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_canonical_cache_matches_general_components_for_every_release_duration() -> None:
    cached = _CachedProDMP(ProDMPConfig())
    reference = ProDMP(ProDMPConfig())
    cached.prewarm(1000)
    for duration in range(1, 1001):
        times = np.arange(1, duration + 1, dtype=np.float64)
        expected = reference._components(times, float(duration), 0.0)
        actual = cached._components(times, float(duration), 0.0)
        for a, e in zip(actual, expected):
            assert_same_bits(a, e)


@pytest.mark.parametrize("config", [
    ProDMPConfig(n_basis=2, alpha=1.0, alpha_phase=0.5, grid_points=16),
    ProDMPConfig(n_basis=7, alpha=40.0, alpha_phase=5.0, grid_points=257),
])
def test_canonical_cache_uses_configured_basis(config: ProDMPConfig) -> None:
    cached, reference = _CachedProDMP(config), ProDMP(config)
    for duration in (1, 2, 17, 128, 1000):
        times = np.arange(1, duration + 1, dtype=np.float64)
        for a, e in zip(cached._components(times, duration, 0),
                        reference._components(times, duration, 0)):
            assert_same_bits(a, e)


def test_noncanonical_times_and_boundaries_keep_general_evaluation() -> None:
    cached, reference = _CachedProDMP(ProDMPConfig()), ProDMP()
    weights = np.random.default_rng(7).normal(size=(21, 2))
    for times, tau, boundary in (
        (np.array([1., 1.5, 3.]), 3., 0.),
        (np.array([0., 0.5, 4.]), 3., 0.),
        (np.array([1., 2., 3.]), 4., 0.),
        (np.array([1., 2., 3.]), 3., 0.5),
    ):
        kw = dict(tau=tau, ydot_b=np.array([2., -1.]), y_b=np.array([3., 4.]), t_b=boundary)
        assert_same_bits(cached.evaluate(times, weights, **kw),
                         reference.evaluate(times, weights, **kw))


def test_lazy_cache_supports_concurrent_independent_calls() -> None:
    cached, reference = _CachedProDMP(ProDMPConfig()), ProDMP()
    rng = np.random.default_rng(71)
    cases = [(rng.normal(size=(21, 2)), rng.normal(size=2), d)
             for d in (1, 37, 1000, 37, 1000, 1, 37, 160)]
    expected = [reference.generate_deltas(*case) for case in cases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda case: cached.generate_deltas(*case), cases))
    for case, a, e in zip(cases, actual, expected):
        assert_same_bits(a, e)
        a[:] = 123
        assert_same_bits(cached.generate_deltas(*case), e)


@pytest.mark.parametrize("duration", [1, 2, 3, 25, 160, 999, 1000])
@pytest.mark.parametrize("dof", [1, 2, 3])
def test_cached_deltas_preserve_general_generation(duration: int, dof: int) -> None:
    cached, reference = _CachedProDMP(ProDMPConfig()), ProDMP()
    rng = np.random.default_rng(23)
    # A strided input and a scalar boundary exercise conversion and broadcast.
    weights = rng.normal(size=(21, dof * 2))[:, ::2]
    before = weights.copy()
    for boundary in (np.array(-0.0), np.arange(dof, dtype=np.float64) - 1.5):
        for options in ({}, {"y_b": np.arange(dof, dtype=np.float64)},
                        {"goal_mode": "target", "goal": np.ones(dof)},
                        {"goal_mode": "other"}):
            expected = reference.generate_deltas(weights, boundary, duration, **options)
            actual = cached.generate_deltas(weights, boundary, duration, **options)
            assert_same_bits(actual, expected)
            assert_same_bits(cached.generate_deltas(weights, boundary, duration, **options), expected)
    assert_same_bits(weights, before)


@pytest.mark.parametrize("weights,velocity,options", [
    (np.zeros((20, 2)), np.zeros(2), {}),
    (np.zeros((21, 2)), np.zeros(3), {}),
    (np.zeros((21, 2)), np.zeros(2), {"goal_mode": "target"}),
    (np.zeros((21, 2)), np.zeros(2), {"goal_mode": "target", "goal": np.zeros(3)}),
])
def test_cached_generation_preserves_validation(weights, velocity, options) -> None:
    errors = []
    for prodmp in (ProDMP(), _CachedProDMP(ProDMPConfig())):
        with pytest.raises(ValueError) as error:
            prodmp.generate_deltas(weights, velocity, 16, **options)
        errors.append(str(error.value))
    assert errors[0] == errors[1]


def test_decode_clipping_rounding_float32_masks_and_endpoints() -> None:
    cached, reference = _CachedProDMP(ProDMPConfig()), ProDMP()
    rng = np.random.default_rng(9123)
    logs = [-np.inf, -1.0, 0.0, np.inf, math.log(1000), math.log(1001)]
    for duration in (1.5, 2.5, 25.5, 159.5, 998.5, 999.5):
        value = math.log(duration)
        logs.extend((np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf)))
    y = rng.normal(size=(2, len(logs), 43))
    y[:, :, -1] = logs
    mean, std = np.zeros(43, np.float32), np.ones(43, np.float32)
    velocity = np.array([[0., -0.], [32767., -32768.]])
    actual, mask = decode_heads(cached, y, mean, std, velocity, horizon=1000)
    expected = np.zeros_like(actual)
    for i in range(len(y)):
        for head, log_duration in enumerate(logs):
            duration = int(np.clip(round(float(np.exp(np.clip(log_duration, 0, math.log(1000))))), 1, 1000))
            positions = reference.generate(y[i, head, :-1].reshape(21, 2), velocity[i], duration)
            expected[i, head, :duration] = np.diff(positions, axis=0, prepend=np.zeros((1, 2)))
            assert np.all(mask[i, head, :duration] == 1)
            assert np.all(mask[i, head, duration:] == 0)
            assert np.all(actual[i, head, duration:] == 0)
            assert np.all(np.isfinite(actual[i, head]))
    assert_same_bits(actual, expected)
    y[0, 0, -1] = np.nan
    with pytest.raises(ValueError):
        decode_heads(cached, y, mean, std, velocity, horizon=1000)


@pytest.mark.parametrize("length", [0, 1, 2, 24, 25, 26, 159, 160, 161, 256])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_short_normalized_window_matches_full_tensor_tail(length, dtype) -> None:
    planner = Planner.__new__(Planner)
    planner.prefix_len = 160
    planner._prefix_mean = np.array([[2.25, -1.75]], np.float32)
    planner._prefix_std = np.array([[3.75, 0.0625]], np.float32)
    raw = (np.arange(length * 4).reshape(length, 4) - 7).astype(dtype)[:, ::2]
    raw.setflags(write=False)
    expected = np.zeros((1, 160, 3), np.float32)
    if length:
        take = min(160, length)
        expected[0, -take:, :2] = (raw[-take:].astype(np.float32) - planner._prefix_mean) / planner._prefix_std
        expected[0, -take:, 2] = 1
    full, mask = planner._prefix_tensor(raw)
    short = planner._normalized_prefix_window(raw, 25)
    assert_same_bits(full, expected)
    assert_same_bits(mask, expected[:, :, 2])
    assert_same_bits(short, expected[:, -25:, :])
    mask[:] = 9
    assert_same_bits(full, expected)


@pytest.fixture(scope="module", params=[7, 23])
def fast_planner(request) -> FastPlanner:
    return FastPlanner(ROOT / "models" / f"planner_seed{request.param}.pt", prewarm=True)


def test_summary_keeps_older_history_and_prefix_truncation(fast_planner: FastPlanner) -> None:
    fast = fast_planner
    prefix = np.zeros((160, 2), np.float32)
    changed = prefix.copy()
    changed[:135] = (4, -2)
    kw = dict(target_rel_at_B=(50., -10.), target_radius=8., progress_center=.7, seed=23)
    first = fast.plan(prefix, **kw).intent
    second = fast.plan(changed, **kw).intent
    assert not np.array_equal(first.smooth_dxdy, second.smooth_dxdy)
    padded = np.concatenate((np.full((100, 2), 1234, np.float32), changed))
    again = fast.plan(padded, **kw).intent
    assert_same_bits(again.smooth_dxdy, second.smooth_dxdy)
    assert_same_bits(again.mask, second.mask)


def test_concurrent_seeded_calls_and_outputs_are_independent(fast_planner: FastPlanner) -> None:
    fast = fast_planner
    prefix = np.random.default_rng(10).normal(size=(160, 2)).astype(np.float32)
    kw = dict(target_rel_at_B=(40., 20.), target_radius=12., progress_center=.6)
    seeds = [0, 7, 23, 2**64 - 1] * 4
    def plan(seed):
        return fast.plan(prefix, seed=seed, **kw).intent
    expected = [plan(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(plan, seeds))
    for seed, a, e in zip(seeds, actual, expected):
        assert a.head == e.head == int(np.random.default_rng(seed).integers(0, 16))
        assert_same_bits(a.smooth_dxdy, e.smooth_dxdy)
        assert_same_bits(a.mask, e.mask)
        a.smooth_dxdy[:] = 9
        a.mask[:] = 0
    again = plan(seeds[0])
    assert_same_bits(again.smooth_dxdy, expected[0].smooth_dxdy)
    assert_same_bits(again.mask, expected[0].mask)
