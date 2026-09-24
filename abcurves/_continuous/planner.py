"""Frozen B2-23 motor/selector/brake composition, without research scaffolding."""
from __future__ import annotations

import math

import numpy as np

from .backend import Assets, OnnxBackend, NativeBackend
from . import kernels
from .rng import RandomStream, softmax


class Planner:
    """One mutable B2-23 stream; use separate instances for concurrent streams.

    ``diagnostics`` retains reference-compatible decision/event records for
    qualification. Leave it off for bounded-memory deployment. ``skip_unused``
    omits discarded motor proposals, preserving their random draw consumption.
    """
    def __init__(self, assets, *, diagnostics=False, backend="native",
                 kernel_backend="numba", skip_unused=True):
        self.assets = assets if isinstance(assets, Assets) else Assets(assets)
        if backend not in ("onnx", "onnx-split", "native-hybrid", "native", "native-pade7"):
            raise ValueError("Unknown neural backend: " + backend)
        if kernel_backend not in ("numpy", "numba"):
            raise ValueError("Unknown physical backend: " + kernel_backend)
        if backend in ("native", "native-pade7"):
            self.engine = NativeBackend(self.assets, motor_variant=("pade9_vector" if backend == "native" else "pade7_vector"))
        else:
            self.engine = OnnxBackend(self.assets, split_choice=backend != "onnx")
        if backend == "native-hybrid":
            from .neural_native import NativeMotor
            self.native_motor = NativeMotor(self.assets, "pade7_vector")
            self.engine.motor = self.native_motor.motor
        self.kernel_backend = kernel_backend
        self.diagnostics = bool(diagnostics)
        self.skip_unused = bool(skip_unused)
        self.velocity_basis = self.assets.arrays["motor.velocity_basis"]
        self.carry_velocity = self.assets.arrays["motor.carry_velocity"]
        self.mean_basis, self.mean_carry = kernels.decoder_geometry(self.velocity_basis, self.carry_velocity)
        self._times = np.arange(33, dtype=np.float64)
        self._recent_offsets = np.arange(-31, 1, dtype=np.int64) * 1000
        self.reset(seed=7)

    def reset(self, *, seed=7):
        self.motor_rng = RandomStream(seed)
        self.selector_rng = RandomStream(self.motor_rng.seed ^ 0x5397A1)
        self.brake_rng = RandomStream(self.motor_rng.seed ^ 0x1763B4)
        self.mode = None
        self.previous = None
        self.origin = None
        self.previous_valid = False
        self.events = []
        self.movement_choices = []
        self.last_sample = None
        self.decision_count = 0
        self.motor_evaluations = 0

    def _initialize(self, history, position, goal, now_us):
        self.mode = 2 if np.abs(history[-32:]).max() <= 1e-12 else 0
        self.age = 160.
        self.duration = 64.
        self.brake_v = np.zeros(2, np.float64)
        self.brake_a = np.zeros(2, np.float64)
        self.coeff = np.zeros((5, 2), np.float64)
        self.hold_goal = goal.copy()
        self.hold_position = position.copy()
        self.hold_error = float(np.linalg.norm(goal - position))
        self.changed = self.hold_error > 1e-7
        self.innov_at = now_us / 1000 if self.changed else math.nan
        self.resume_integral = 0.
        self.stop_integral = 0.
        # Reference budget arrays are float32, including subsequent assignments.
        self.resume_budget = -np.log(np.maximum(self.selector_rng.uniform(1), np.float32(1e-8)))[0]
        self.stop_budget = -np.log(np.maximum(self.brake_rng.uniform(1), np.float32(1e-8)))[0]

    def _motor(self, history, position, target, available, valid, motion_known, now_us):
        coarse, fine, dynamics = kernels.motor_features(history, position, target,
            available, valid, motion_known, now_us, backend=self.kernel_backend)
        encoded, coefficients = self.engine.motor(coarse, fine, dynamics)
        heads = kernels.decode_geometry(coefficients, history[-1], self.mean_basis, self.mean_carry)
        previous = self.previous if self.previous_valid else None
        actual_step = position - self.origin if self.origin is not None else np.zeros(2)
        unary, pairs = kernels.selector_inputs(encoded, heads, previous, actual_step, backend=self.kernel_backend)
        logits = self.engine.choice(encoded, np.ascontiguousarray(unary[:, :, 96:]), pairs, self.previous_valid)
        probabilities = softmax(logits)
        chosen = self.motor_rng.categorical(probabilities)
        self.previous = heads[chosen].copy()
        self.origin = position.copy()
        self.previous_valid = True
        self.motor_evaluations += 1
        action = kernels.decode_selected(coefficients[chosen], history[-1], self.velocity_basis[:32], self.carry_velocity[:32])
        return action, chosen, probabilities

    def plan(self, history, position, target, available, valid, motion_known, now_us):
        known = bool(valid[-1] and available[-1] <= now_us)
        goal = target[-1] if known else position
        if self.mode is None:
            self._initialize(history, position, goal, now_us)
        old = self.mode
        if old != 0:
            self.previous_valid = False
            recent_known = valid[-32:] & (available[-32:] <= now_us + self._recent_offsets)
            change = np.flatnonzero(recent_known & (np.linalg.norm(target[-32:] - self.hold_goal, axis=-1) > 1e-7))
            if len(change):
                self.changed = True
                if math.isnan(self.innov_at):
                    self.innov_at = now_us / 1000 - 31 + int(change[0])
        innovation_age = 0. if math.isnan(self.innov_at) else max(0., now_us / 1000 - self.innov_at)
        raw, context, basis = kernels.event_features(history, position, target, available,
            valid, motion_known, now_us, hold_age=self.age, hold_target=self.hold_goal,
            hold_position=self.hold_position, initial_error=self.hold_error,
            innovation_age=innovation_age, mode=old, backend=self.kernel_backend)
        innovation = old == 2 and (self.changed or np.linalg.norm(position - self.hold_position) > 1e-5 or raw[0] > 10)
        full_events = self.diagnostics or not self.skip_unused
        need_hazard = old == 0 or innovation
        if full_events:
            duration, coefficients, frequency, hazard = self.engine.events(context)
        elif need_hazard:
            hazard = self.engine.hazard(context)
        else:
            hazard = None
        probabilities = (np.float32(1.) / (np.float32(1.) + np.exp(-np.clip(hazard, -40, 40)))
                         if hazard is not None else np.zeros(2, np.float32))
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Nonfinite event probabilities")
        draws = self.selector_rng.uniform(2)
        brake_head = self.brake_rng.categorical(softmax(frequency)) if full_events else None
        event = None
        if old == 0:
            if known:
                self.stop_integral += -math.log(max(1e-8, 1 - float(probabilities[0])))
            if self.stop_integral >= self.stop_budget and known:
                if not full_events:
                    duration, coefficients, frequency = self.engine.brake(context)
                    brake_head = self.brake_rng.categorical(softmax(frequency))
                self.mode = 1
                self.age = 0.
                self.brake_v = history[-1].copy()
                self.brake_a = (history[-8:].mean(0) - history[-16:-8].mean(0)) / 8
                self.duration = float(duration[brake_head])
                self.coeff = coefficients[brake_head] @ basis.T
                self.hold_goal = goal.copy()
                self.hold_error = float(raw[0])
                self.changed = False
                self.innov_at = math.nan
                event = dict(head=brake_head, endpoint=self.coeff[0].tolist(),
                    incoming_acceleration=self.brake_a.tolist(), shape=self.coeff.tolist(), geometry="C2")
        elif old == 2:
            if innovation and known:
                self.resume_integral += -math.log(max(1e-8, 1 - float(probabilities[1])))
            if innovation and self.resume_integral >= self.resume_budget and known:
                self.mode = 0
                self.age = 0.
                self.changed = False
                self.innov_at = math.nan
                self.stop_integral = 0.
                self.stop_budget = np.float32(-math.log(max(float(draws[0]), 1e-8)))

        if not full_events and brake_head is None:
            self.brake_rng.discard_categorical()

        # Separate RNG streams allow event computation before the motor. The
        # discarded proposal never enters physical state or valid predecessor
        # geometry. Consuming its raw draw words keeps future seeds aligned.
        if self.mode == 0 or self.diagnostics or not self.skip_unused:
            latent, movement_head, movement_probabilities = self._motor(
                history, position, target, available, valid, motion_known, now_us)
        else:
            self.motor_rng.discard_categorical()
            latent = None
            movement_head = -1
            movement_probabilities = None

        if self.mode == 1:
            curve = kernels.c2_path(self.brake_v, self.brake_a, self.duration, self.coeff,
                                   self.age + self._times, backend=self.kernel_backend)
            actual = np.diff(curve, axis=0)
            self.age += 32
            if self.age >= self.duration:
                self.mode = 2
                self.age = max(0., self.age - self.duration)
                self.hold_position = position + actual.sum(0)
                self.hold_goal = goal.copy()
                self.hold_error = float(np.linalg.norm(goal - self.hold_position))
                self.changed = False
                self.innov_at = math.nan
                self.resume_integral = 0.
                self.resume_budget = np.float32(-math.log(max(float(draws[1]), 1e-8)))
        elif self.mode == 2:
            actual = np.zeros((32, 2), np.float64)
            self.age += 32
        else:
            actual = latent
            self.age += 32
        if self.mode != 0:
            self.previous_valid = False
        if not np.isfinite(actual).all() or np.abs(actual).max() > 1e5:
            raise FloatingPointError("Invalid finite primitive")
        if self.diagnostics:
            if self.mode != old or event is not None:
                self.events.append(dict(batch=0, at_ms=now_us / 1000, previous=old,
                    mode=self.mode, distance=float(raw[0]), speed=float(raw[1]),
                    target_speed=float(raw[5]), duration=self.duration,
                    probabilities=probabilities.tolist(), **(event or {})))
            self.movement_choices.append(dict(at_ms=[now_us / 1000], mode=[self.mode],
                head=[movement_head], probabilities=[movement_probabilities.tolist()]))
            self.last_sample = dict(head=[movement_head], mode=[self.mode],
                probabilities=[probabilities.tolist()], movement_probabilities=[movement_probabilities.tolist()],
                brake_acceleration=[self.brake_a.tolist()])
        self.decision_count += 1
        return actual

    def prewarm(self):
        """Compile/load kernels and exercise every numerical path, then reset."""
        history = np.zeros((640, 2), np.float64)
        position = np.zeros(2, np.float64)
        target = np.full((640, 2), 100., np.float64)
        available = np.full(640, -640000, np.int64)
        known = np.ones(640, bool)
        self._motor(history, position, target, available, known, known, 0)
        self.plan(history, position, target, available, known, known, 0)
        context = np.zeros((1,32), np.float32)
        self.engine.events(context)
        self.engine.hazard(context)
        self.engine.brake(context)
        kernels.c2_path(np.zeros(2), np.zeros(2), 64., np.zeros((5,2)), self._times,
                        backend=self.kernel_backend)
        self.reset(seed=7)
