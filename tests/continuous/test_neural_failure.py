"""Neural numerical failures must not silently become ordinary hold output."""
import numpy as np
import pytest

from abcurves._continuous.planner import Planner
from abcurves._continuous.rng import softmax
from abcurves._continuous.runtime import MovementRuntime, RuntimeFailure


@pytest.mark.parametrize("logits", [[np.nan, 0.0], [np.inf, 0.0], [-np.inf, -np.inf]])
def test_nonfinite_choice_logits_are_rejected(logits):
    with np.errstate(invalid="ignore"):
        with pytest.raises(FloatingPointError, match="Invalid categorical probabilities"):
            softmax(logits)


def test_neural_hazard_failure_is_latched_and_reset_starts_a_new_stream():
    class HazardFixture:
        broken = True

        def hazard(self, context):
            assert context.shape == (1, 32)
            return np.full(2, np.nan if self.broken else -40.0, np.float32)

    # Isolate the full production hazard/hold path without loading a checkpoint:
    # invalid event output must fail before a discarded motor proposal is needed.
    planner = object.__new__(Planner)
    planner.engine = HazardFixture()
    planner.diagnostics = False
    planner.skip_unused = True
    planner.kernel_backend = "numpy"
    planner._recent_offsets = np.arange(-31, 1, dtype=np.int64) * 1000
    planner._times = np.arange(33, dtype=np.float64)
    runtime = MovementRuntime(planner, seed=23)
    runtime.update_target((160.0, 40.0), timestamp_us=0)
    with pytest.raises(RuntimeFailure, match="Nonfinite event probabilities") as failure:
        runtime.advance(1000)
    assert failure.value.at_us == 0
    assert failure.value.partial["xy"].shape == (0, 2)
    assert runtime.failed
    assert runtime.fresh_decisions == 0
    np.testing.assert_array_equal(runtime.current_xy, (0.0, 0.0))
    with pytest.raises(RuntimeError, match="Runtime has failed"):
        runtime.advance(2000)
    planner.engine.broken = False
    runtime.reset()
    runtime.update_target((160.0, 40.0), timestamp_us=0)
    result = runtime.advance(32000)
    np.testing.assert_array_equal(result["time_us"], np.arange(1, 33) * 1000)
    np.testing.assert_array_equal(result["xy"], np.zeros((32, 2)))
    assert not runtime.failed
