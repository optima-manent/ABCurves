"""Public streaming-contract checks using a cheap, state-observing planner."""
from __future__ import annotations

import unittest

import numpy as np

from abcurves._continuous.runtime import MovementRuntime, RuntimeFailure


class RecordingPlanner:
    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def plan(self, history, position, target, available, valid, motion_known, now_us):
        self.calls.append(
            dict(
                history=history.copy(), position=position.copy(), target=target.copy(),
                available=available.copy(), valid=valid.copy(), motion_known=motion_known.copy(),
                now_us=now_us,
            )
        )
        velocity = (target[-1] - position) / 100 + history[-1] / 10
        noise = self.rng.normal(0, 0.001, (32, 2))
        return velocity[None] + noise


class ConstantPlanner(RecordingPlanner):
    def plan(self, *args):
        super().plan(*args)
        return np.ones((32, 2), np.float64)


class StreamingTests(unittest.TestCase):
    def test_uncommanded_position_and_closed_endpoints(self):
        planner = RecordingPlanner()
        runtime = MovementRuntime(planner, initial_xy=(12, -3))
        first = runtime.advance(999)
        self.assertEqual(first["xy"].shape, (0, 2))
        self.assertEqual(first["time_us"].dtype, np.int64)
        result = runtime.advance(3500)
        np.testing.assert_array_equal(result["time_us"], [1000, 2000, 3000])
        np.testing.assert_array_equal(result["xy"], [[12, -3]] * 3)
        np.testing.assert_array_equal(runtime.advance(4000)["time_us"], [4000])
        self.assertEqual(planner.calls, [])
        self.assertEqual(runtime.fresh_decisions, 0)

    def test_fractional_receipts_same_time_last_wins(self):
        planner = ConstantPlanner()
        runtime = MovementRuntime(planner)
        runtime.update_target((90, 30), 1501)
        runtime.update_target((100, 40), 1501)
        runtime.update_target((900, 400), 9000)
        result = runtime.advance(3000)
        np.testing.assert_array_equal(result["xy"], [[0, 0], [0, 0], [1, 1]])
        call = planner.calls[0]
        self.assertEqual(call["now_us"], 2000)
        np.testing.assert_array_equal(call["target"][-1], [100, 40])
        self.assertEqual(call["available"][-1], 1501)
        self.assertEqual(np.count_nonzero(call["valid"]), 1)
        self.assertLessEqual(call["available"].max(), call["now_us"])

    def test_receipt_does_not_change_existing_commitment(self):
        planner = ConstantPlanner()
        runtime = MovementRuntime(planner)
        runtime.update_target((100, 0), 0)
        self.assertTrue(runtime.needs_plan)
        runtime.advance(1000)
        self.assertFalse(runtime.needs_plan)
        runtime.update_target((-100, 0), 1000)
        runtime.advance(32000)
        self.assertEqual(len(planner.calls), 1)
        self.assertTrue(runtime.needs_plan)
        runtime.advance(33000)
        self.assertEqual([call["now_us"] for call in planner.calls], [0, 32000])
        np.testing.assert_array_equal(planner.calls[1]["target"][-1], [-100, 0])
        self.assertEqual(runtime.fresh_decisions, 2)

    def test_arbitrary_chunking_seed_and_receipt_causality(self):
        events = [(0, (100, 10)), (17500, (100, 100)), (33101, (-50, 100)), (101500, (1, 2))]

        def run(chunks):
            planner = RecordingPlanner()
            runtime = MovementRuntime(planner, initial_xy=(3, 4), seed=123)
            for at, target in events:
                runtime.update_target(target, at)
            output = [runtime.advance(at) for at in chunks]
            return (
                np.concatenate([part["time_us"] for part in output]),
                np.concatenate([part["xy"] for part in output]),
                planner.calls,
            )

        once = run([140000])
        chunked = run([0, 37, 999, 1000, 17501, 33101, 33400, 50000, 50999, 101501, 140000])
        np.testing.assert_array_equal(once[0], chunked[0])
        np.testing.assert_array_equal(once[1], chunked[1])
        for left, right in zip(once[2], chunked[2], strict=True):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
            causal_cut = left["now_us"] + np.arange(-639, 1) * 1000
            self.assertTrue(np.all(left["available"][left["valid"]] <= causal_cut[left["valid"]]))

    def test_warm_start_and_delayed_first_target(self):
        history = np.arange(320, dtype=np.float64).reshape(160, 2) / 100
        for delay in (0, 17000, 900000):
            with self.subTest(delay=delay):
                planner = RecordingPlanner()
                runtime = MovementRuntime(planner, initial_xy=(13, 14), history=history)
                runtime.update_target((100, 200), delay)
                runtime.advance(delay + 1000)
                call = planner.calls[0]
                expected = np.concatenate((history, np.zeros((delay // 1000, 2))))[-160:]
                np.testing.assert_array_equal(call["history"][:480], 0)
                np.testing.assert_array_equal(call["history"][-160:], expected)
                np.testing.assert_array_equal(call["motion_known"][:480], False)
                np.testing.assert_array_equal(call["motion_known"][-160:], True)
                np.testing.assert_array_equal(call["position"], [13, 14])
                self.assertEqual(np.count_nonzero(call["valid"]), 1)

    def test_history_remains_complete_after_multiple_wraps(self):
        planner = ConstantPlanner()
        runtime = MovementRuntime(planner)
        runtime.update_target((100, 200), 0)
        runtime.advance(2001000)
        for call in planner.calls:
            elapsed = call["now_us"] // 1000
            known = min(640, 160 + elapsed)
            np.testing.assert_array_equal(call["motion_known"][-known:], True)
            if known < 640:
                np.testing.assert_array_equal(call["motion_known"][:-known], False)
            if elapsed >= 640:
                np.testing.assert_array_equal(call["history"], 1)
                np.testing.assert_array_equal(call["target"], np.tile([100, 200], (640, 1)))
                np.testing.assert_array_equal(call["available"], 0)
                np.testing.assert_array_equal(call["valid"], True)
        np.testing.assert_array_equal(runtime.current_xy, [2001, 2001])

    def test_reset_repeats_initialization_and_seed(self):
        planner = RecordingPlanner()
        history = np.full((160, 2), 0.1)
        runtime = MovementRuntime(planner, initial_xy=(3, 4), history=history, seed=7)
        runtime.update_target((100, 50), 0)
        expected = runtime.advance(100000)["xy"]
        runtime.reset()
        self.assertEqual(runtime.now_us, 0)
        self.assertFalse(runtime.needs_plan)
        runtime.update_target((100, 50), 0)
        np.testing.assert_array_equal(runtime.advance(100000)["xy"], expected)
        runtime.reset(seed=19, initial_xy=(1, 2), history=np.zeros((160, 2)))
        runtime.update_target((100, 50), 0)
        changed = runtime.advance(100000)["xy"]
        self.assertFalse(np.array_equal(changed, expected))
        runtime.reset()
        np.testing.assert_array_equal(runtime.current_xy, [1, 2])
        runtime.update_target((100, 50), 0)
        np.testing.assert_array_equal(runtime.advance(100000)["xy"], changed)

    def test_output_arrays_and_supplied_inputs_are_not_aliased(self):
        initial = np.array([1.0, 2.0])
        history = np.zeros((160, 2))
        target = np.array([100.0, 200.0])
        planner = ConstantPlanner()
        runtime = MovementRuntime(planner, initial_xy=initial, history=history)
        runtime.update_target(target, 0)
        initial[:] = history[:] = target[:] = -999
        output = runtime.advance(1000)
        np.testing.assert_array_equal(output["xy"], [[2, 3]])
        output["xy"][:] = -100
        copy = runtime.current_xy
        copy[:] = -200
        np.testing.assert_array_equal(runtime.current_xy, [2, 3])
        np.testing.assert_array_equal(runtime.advance(2000)["xy"], [[3, 4]])
        np.testing.assert_array_equal(planner.calls[0]["target"][-1], [100, 200])
        np.testing.assert_array_equal(planner.calls[0]["history"], 0)

    def test_failure_returns_complete_partial_output_and_requires_reset(self):
        class FailingPlanner(ConstantPlanner):
            def plan(self, *args):
                action = super().plan(*args)
                if len(self.calls) == 2:
                    action[17, 0] = np.nan
                return action

        runtime = MovementRuntime(FailingPlanner())
        runtime.update_target((100, 200), 0)
        with self.assertRaises(RuntimeFailure) as context:
            runtime.advance(80000)
        failure = context.exception
        self.assertEqual(failure.at_us, 32000)
        np.testing.assert_array_equal(failure.partial["time_us"], np.arange(1, 33) * 1000)
        np.testing.assert_array_equal(failure.partial["xy"], np.repeat(np.arange(1, 33)[:, None], 2, axis=1))
        np.testing.assert_array_equal(runtime.current_xy, [32, 32])
        with self.assertRaises(RuntimeError):
            runtime.advance(81000)
        runtime.reset()
        runtime.update_target((1, 2), 0)
        self.assertEqual(runtime.advance(1000)["xy"].shape, (1, 2))

    def test_incomplete_action_exception_and_overflow_are_failures(self):
        class BadPlanner(RecordingPlanner):
            def plan(self, *args):
                if isinstance(self.response, Exception):
                    raise self.response
                return self.response

        for response in (np.zeros((31, 2)), np.full((32, 2), np.inf), ValueError("broken"), np.full((32, 2), 1e308)):
            with self.subTest(response=str(response)[:40]):
                planner = BadPlanner()
                planner.response = response
                runtime = MovementRuntime(planner)
                runtime.update_target((100, 200), 0)
                with self.assertRaises(RuntimeFailure) as context:
                    runtime.advance(1000)
                self.assertEqual(context.exception.at_us, 0)
                self.assertEqual(context.exception.partial["xy"].shape, (0, 2))
                self.assertEqual(runtime.fresh_decisions, 0)

    def test_validation_and_requested_time_monotonicity(self):
        for invalid in (True, np.bool_(False), -1, 1.0, "1", 1 << 63):
            with self.subTest(invalid=invalid):
                runtime = MovementRuntime(RecordingPlanner())
                with self.assertRaises(ValueError):
                    runtime.advance(invalid)
                with self.assertRaises(ValueError):
                    runtime.update_target((0, 0), invalid)
        runtime = MovementRuntime(RecordingPlanner())
        runtime.advance(1500)
        with self.assertRaises(ValueError):
            runtime.advance(1499)
        with self.assertRaises(ValueError):
            runtime.update_target((0, 0), 999)
        runtime.update_target((0, 0), 2000)
        with self.assertRaises(ValueError):
            runtime.update_target((0, 0), 1999)
        for target in ([np.nan, 0], [np.inf, 0], [0], [[0, 0]]):
            with self.assertRaises(ValueError):
                runtime.update_target(target, 3000)
        for kwargs in (dict(initial_xy=(np.nan, 0)), dict(history=np.zeros((640, 2))), dict(seed=1.2)):
            with self.assertRaises(ValueError):
                MovementRuntime(RecordingPlanner(), **kwargs)

    def test_8ms_absolute_rebasing(self):
        class SmallPlanner(RecordingPlanner):
            def plan(self, *args):
                return np.full((32, 2), 0.125)

        runtime = MovementRuntime(SmallPlanner(), initial_xy=(1e16, 1e16))
        runtime.update_target((1e16 + 100, 1e16 + 100), 0)
        result = runtime.advance(32000)["xy"]
        expected = np.empty((32, 2))
        origin = np.full(2, 1e16)
        for at in range(0, 32, 8):
            expected[at : at + 8] = origin + np.cumsum(np.full((8, 2), 0.125), axis=0)
            origin = expected[at + 7]
        np.testing.assert_array_equal(result, expected)
        self.assertNotEqual(result[-1, 0], 1e16 + 4)

    def test_elapsed_clock_is_relative_to_each_reset(self):
        clock = [10_000_000]
        runtime = MovementRuntime(RecordingPlanner(), clock_ns=lambda: clock[0])
        clock[0] += 1_500_000
        np.testing.assert_array_equal(runtime.advance()["time_us"], [1000])
        runtime.reset()
        clock[0] += 1_000_000
        np.testing.assert_array_equal(runtime.advance()["time_us"], [1000])


if __name__ == "__main__":
    unittest.main()
