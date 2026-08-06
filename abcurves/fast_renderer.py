"""Compiled streaming Renderer used by the released ABCurves pipeline.

The model law is the frozen F0/TP1/C/AF1.5 Renderer: a 96-wide GRU, TP1's
threshold-only terminal policy, an optional rank-four causal session-state
adapter, event-local SplitMix64 sampling, and a 1.5 normal-frame anti-flip
penalty.  Numba moves the recurrent hot loop out of Python while leaving the
checkpoint, random coordinates, and state transitions unchanged.

``FastRendererKernel`` is loaded once.  ``prepare_prefix`` may run in parallel
with Planner inference, and ``FastRendererEvent.step`` emits exactly one
integer ``(dx, dy)`` report for each 1 ms tick.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numba import njit

from .fast_kernels import (
    _begin_precompute,
    _gru_gates,
    _gru_walk,
    _matvec,
    _regime_features_nb,
    _smooth_w5,
    _warm_features,
)


HORIZON = 1000
RECENT_WINDOW = 60
STATE_WIDTH = 3
LATERAL_PENALTY = 1.5
LATERAL_FREE = 1.0

GAMMA_U64 = np.uint64(0x9E3779B97F4A7C15)
MIX1_U64 = np.uint64(0xBF58476D1CE4E5B9)
MIX2_U64 = np.uint64(0x94D049BB133111EB)
U23_DENOM = np.float32(8388608.0)


class RuntimeContractError(RuntimeError):
    """Raised when a live input differs from the retained runtime contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _as_f32(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))


@njit(cache=True, nogil=True)
def _splitmix64_value(seed: np.uint64, ordinal: np.uint64) -> np.uint64:
    """Frozen counter mapping used by the current stateless sampler."""

    value = seed + GAMMA_U64 * ordinal
    value = (value ^ (value >> np.uint64(30))) * MIX1_U64
    value = (value ^ (value >> np.uint64(27))) * MIX2_U64
    return value ^ (value >> np.uint64(31))


@njit(cache=True, nogil=True)
def _uniform_pair(seed: np.uint64, tick: np.int64) -> tuple[np.float32, np.float32]:
    first = _splitmix64_value(seed, np.uint64(2 * tick + 1))
    second = _splitmix64_value(seed, np.uint64(2 * tick + 2))
    u0 = (np.float32(first >> np.uint64(41)) + np.float32(0.5)) / U23_DENOM
    u1 = (np.float32(second >> np.uint64(41)) + np.float32(0.5)) / U23_DENOM
    return u0, u1


@njit(cache=True, nogil=True)
def _current_tick(
    tick: np.int64,
    smooth: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    static: np.ndarray,
    regime: np.ndarray,
    # Base recurrent and output weights.
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
    w_emit: np.ndarray,
    b_emit: np.float32,
    w_off: np.ndarray,
    b_off: np.ndarray,
    # C adapter. ``has_c`` is false for the exact-zero route.
    has_c: np.bool_,
    c_state: np.ndarray,
    c_wh: np.ndarray,
    c_ws: np.ndarray,
    c_u: np.ndarray,
    # Offset layout and temperatures.
    offsets: np.ndarray,
    ring_id: np.ndarray,
    n_rings: np.int64,
    tau_emit: np.float32,
    tau_mag: np.float32,
    tau_dir: np.float32,
    hysteresis: np.float32,
    force_release: np.float32,
    max_abs_count: np.float32,
    lateral_penalty: np.float32,
    lateral_free: np.float32,
    event_seed: np.uint64,
    # Mutable event state and preallocated scratch.
    hidden: np.ndarray,
    state: np.ndarray,
    active_ring: np.ndarray,
    ring_meta: np.ndarray,
    feature_buf: np.ndarray,
    adapted_hidden: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    hidden_buf: np.ndarray,
    input_gate_buf: np.ndarray,
    hidden_gate_buf: np.ndarray,
    c_q: np.ndarray,
    c_hproj: np.ndarray,
    c_sproj: np.ndarray,
) -> tuple[np.int64, np.int64]:
    """One current TP1+C+AF1.5 recurrent tick.

    State layout is ``acc[2], prev[2], last_nonzero[2], last_axis[2],
    run_active, run_length, recent_zero``.  The C adapter never touches the
    recurrence; it adapts only the hidden vector presented to both heads.
    """

    f0 = np.float32(0.0)
    f1 = np.float32(1.0)
    four = np.float32(4.0)
    hidden_width = hidden.shape[0]
    grid = ring_id.shape[0]
    k = offsets.shape[0]

    state[0] = state[0] + smooth[tick, 0]
    state[1] = state[1] + smooth[tick, 1]
    tx, ty = tangents[tick, 0], tangents[tick, 1]
    nx, ny = normals[tick, 0], normals[tick, 1]

    acc_t = state[0] * tx + state[1] * ty
    acc_n = state[0] * nx + state[1] * ny
    prev_t = state[2] * tx + state[3] * ty
    prev_n = state[2] * nx + state[3] * ny
    last_t = state[4] * tx + state[5] * ty
    last_n = state[4] * nx + state[5] * ny

    if ring_meta[1] >= 1:
        active_sum = f0
        for index in range(active_ring.shape[0]):
            active_sum += active_ring[index]
        recent_zero = f1 - active_sum / np.float32(ring_meta[1])
    else:
        recent_zero = state[10]

    feature_buf[0] = static[tick, 0]
    feature_buf[1] = static[tick, 1]
    feature_buf[2] = static[tick, 2]
    feature_buf[3] = static[tick, 3]
    feature_buf[4] = tx
    feature_buf[5] = ty
    feature_buf[6] = min(max(acc_t, -four), four)
    feature_buf[7] = min(max(acc_n, -four), four)
    feature_buf[8] = min(max(prev_t / four, -four), four)
    feature_buf[9] = min(max(prev_n / four, -four), four)
    feature_buf[10] = state[8]
    feature_buf[11] = min(state[9], np.float32(64.0)) / np.float32(64.0)
    feature_buf[12] = recent_zero
    feature_buf[13] = static[tick, 4]
    feature_buf[14] = min(max(last_t / four, -four), four)
    feature_buf[15] = min(max(last_n / four, -four), four)
    for index in range(regime.shape[0]):
        feature_buf[16 + index] = regime[index]

    _matvec(w_ih, feature_buf, input_gate_buf)
    _matvec(w_hh, hidden, hidden_gate_buf)
    for index in range(input_gate_buf.shape[0]):
        input_gate_buf[index] += b_ih[index]
        hidden_gate_buf[index] += b_hh[index]
    _gru_gates(
        input_gate_buf, hidden_gate_buf, hidden, hidden_width, hidden_buf
    )
    for index in range(hidden_width):
        hidden[index] = hidden_buf[index]
        adapted_hidden[index] = hidden[index]

    # q = tanh(Wh h) * tanh(Ws s); h' = h + U q.
    if has_c:
        for rank_index in range(c_q.shape[0]):
            hp = f0
            sp = f0
            for index in range(hidden_width):
                hp += c_wh[rank_index, index] * hidden[index]
            for index in range(c_state.shape[0]):
                sp += c_ws[rank_index, index] * c_state[index]
            c_hproj[rank_index] = np.float32(math.tanh(np.float64(hp)))
            c_sproj[rank_index] = np.float32(math.tanh(np.float64(sp)))
            c_q[rank_index] = c_hproj[rank_index] * c_sproj[rank_index]
        for index in range(hidden_width):
            residual = f0
            for rank_index in range(c_q.shape[0]):
                residual += c_u[index, rank_index] * c_q[rank_index]
            adapted_hidden[index] = hidden[index] + residual

    emit_logit = b_emit
    for index in range(hidden_width):
        emit_logit += w_emit[index] * adapted_hidden[index]
    emit_probability = np.float32(
        1.0 / (1.0 + math.exp(-np.float64(emit_logit / tau_emit)))
    )
    uniform_emit, uniform_offset = _uniform_pair(event_seed, tick)
    emit = uniform_emit < emit_probability
    abs0 = abs(state[0])
    abs1 = abs(state[1])
    if max(abs0, abs1) >= force_release:
        emit = True

    _matvec(w_off, adapted_hidden, logits)
    for index in range(grid):
        logits[index] += b_off[index]

    # Exact current magnitude/direction temperature decomposition.
    for radius_index in range(n_rings):
        maximum = np.float32(-3.4e38)
        for index in range(grid):
            if ring_id[index] == radius_index and logits[index] > maximum:
                maximum = logits[index]
        total64 = 0.0
        for index in range(grid):
            if ring_id[index] == radius_index:
                total64 += math.exp(np.float64(logits[index] - maximum))
        lse = np.float32(np.float64(maximum) + math.log(total64))
        for index in range(grid):
            if ring_id[index] == radius_index:
                logits[index] = lse / tau_mag + (logits[index] - lse) / tau_dir

    # Retained anti-flip: lambda * relu(abs(offset dot normal) - 1 count).
    if lateral_penalty > f0:
        for index in range(grid):
            ox = offsets[index // k]
            oy = offsets[index % k]
            lateral = abs(ox * nx + oy * ny)
            excess = lateral - lateral_free
            if excess > f0:
                logits[index] -= lateral_penalty * excess

    maximum = logits[0]
    for index in range(1, grid):
        if logits[index] > maximum:
            maximum = logits[index]
    total = f0
    for index in range(grid):
        value = np.float32(math.exp(np.float64(logits[index] - maximum)))
        probabilities[index] = value
        total += value
    cumulative = f0
    joint = grid - 1
    for index in range(grid):
        cumulative += probabilities[index] / total
        if uniform_offset <= cumulative:
            joint = index
            break

    base0 = np.float32(np.rint(state[0]))
    base1 = np.float32(np.rint(state[1]))
    if hysteresis > f0:
        last_sign0 = f1 if state[6] > f0 else (-f1 if state[6] < f0 else f0)
        last_sign1 = f1 if state[7] > f0 else (-f1 if state[7] < f0 else f0)
        base_sign0 = f1 if base0 > f0 else (-f1 if base0 < f0 else f0)
        base_sign1 = f1 if base1 > f0 else (-f1 if base1 < f0 else f0)
        if (
            last_sign0 != f0
            and base_sign0 != f0
            and last_sign0 * base_sign0 < f0
            and abs0 < hysteresis
        ):
            base0 = f0
        if (
            last_sign1 != f0
            and base_sign1 != f0
            and last_sign1 * base_sign1 < f0
            and abs1 < hysteresis
        ):
            base1 = f0

    mark0 = base0 + offsets[joint // k]
    mark1 = base1 + offsets[joint % k]
    mark0 = min(max(mark0, -max_abs_count), max_abs_count)
    mark1 = min(max(mark1, -max_abs_count), max_abs_count)
    if not emit:
        mark0 = f0
        mark1 = f0

    state[0] -= mark0
    state[1] -= mark1
    active_now = mark0 != f0 or mark1 != f0
    active_float = f1 if active_now else f0
    if active_float != state[8] or state[9] == f0:
        state[9] = f1
    else:
        state[9] += f1
    state[8] = active_float
    state[2] = mark0
    state[3] = mark1
    if active_now:
        state[4] = mark0
        state[5] = mark1
    if mark0 != f0:
        state[6] = mark0
    if mark1 != f0:
        state[7] = mark1
    active_ring[ring_meta[0]] = active_float
    ring_meta[0] = (ring_meta[0] + 1) % active_ring.shape[0]
    if ring_meta[1] < active_ring.shape[0]:
        ring_meta[1] += 1

    return np.int64(mark0), np.int64(mark1)


@dataclass(frozen=True)
class PrefixWarmState:
    hidden: np.ndarray
    last_direction: np.ndarray
    previous_emit: np.ndarray
    last_nonzero: np.ndarray
    last_axis_nonzero: np.ndarray
    run_active: float
    run_length: float
    recent_zero: float
    active_tail: np.ndarray
    regime: np.ndarray
    warm_ticks: int


@dataclass(frozen=True)
class FutureTables:
    smooth: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    static: np.ndarray
    duration: int


class FastRendererKernel:
    """Load-once weight pack shared by consecutive live events."""

    def __init__(self, base_model: Any, c_adapter: Any | None) -> None:
        base_model = base_model.to("cpu").eval()
        base_model.sync_cell()
        cfg = base_model.config
        require(int(cfg.hidden) == 96, "Renderer hidden width differs")
        require(int(cfg.offset_radius) == 5, "Renderer offset radius differs")
        require(int(base_model.n_features) == 21, "Renderer feature width differs")
        require(int(cfg.prefix_ticks) == 128, "Renderer prefix window differs")
        require(tuple(cfg.regime_feature_names) and len(cfg.regime_feature_names) == 5,
                "Renderer regime layout differs")
        self.base_model = base_model
        self.c_adapter = c_adapter
        self.cfg = cfg

        cell = base_model.cell
        self.w_ih = _as_f32(cell.weight_ih)
        self.w_hh = _as_f32(cell.weight_hh)
        self.w_iht = np.ascontiguousarray(self.w_ih.T)
        self.b_ih = _as_f32(cell.bias_ih)
        self.b_hh = _as_f32(cell.bias_hh)
        self.w_emit = _as_f32(base_model.head_emit.weight)[0]
        self.b_emit = np.float32(_as_f32(base_model.head_emit.bias)[0])
        self.w_off = _as_f32(base_model.head_off.weight)
        self.b_off = _as_f32(base_model.head_off.bias)

        if c_adapter is None:
            self.c_wh = np.zeros((4, 96), dtype=np.float32)
            self.c_ws = np.zeros((4, STATE_WIDTH), dtype=np.float32)
            self.c_u = np.zeros((96, 4), dtype=np.float32)
        else:
            require(getattr(c_adapter, "base", None) is base_model,
                    "C adapter is not bound to the base model")
            self.c_wh = _as_f32(c_adapter.Wh)
            self.c_ws = _as_f32(c_adapter.Ws)
            self.c_u = _as_f32(c_adapter.U)
        require(self.c_wh.shape == (4, 96), "C Wh shape differs")
        require(self.c_ws.shape == (4, STATE_WIDTH), "C Ws shape differs")
        require(self.c_u.shape == (96, 4), "C U shape differs")

        radius = int(cfg.offset_radius)
        self.offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        k = len(self.offsets)
        magnitude = np.maximum(
            np.abs(np.repeat(self.offsets, k)),
            np.abs(np.tile(self.offsets, k)),
        )
        unique = np.unique(magnitude)
        self.ring_id = np.searchsorted(unique, magnitude).astype(np.int64)
        self.n_rings = np.int64(len(unique))
        self.tau_emit = np.float32(max(float(cfg.temperature), 1e-3))
        self.tau_mag = np.float32(max(float(cfg.offset_magnitude_temperature), 1e-3))
        self.tau_dir = np.float32(max(float(cfg.offset_direction_temperature), 1e-3))
        self.hysteresis = np.float32(cfg.base_hysteresis)
        self.force_release = np.float32(cfg.force_release)
        self.max_abs_count = np.float32(cfg.max_abs_count)

    def prepare_prefix(self, prefix_raw_dxdy: np.ndarray) -> PrefixWarmState:
        """Causally prepare and recurrently warm the last 128 prefix ticks."""

        prefix_full = np.asarray(prefix_raw_dxdy, dtype=np.float64)
        require(prefix_full.ndim == 2 and prefix_full.shape[1] == 2,
                "raw prefix shape differs")
        require(len(prefix_full) > 0 and np.all(np.isfinite(prefix_full)),
                "raw prefix is empty or nonfinite")
        used = np.ascontiguousarray(prefix_full[-int(self.cfg.prefix_ticks):])
        regime = _regime_features_nb(
            np.ascontiguousarray(prefix_full), np.int64(self.cfg.regime_window)
        )
        smooth_prefix = _smooth_w5(used)
        features = _warm_features(smooth_prefix, used, regime)
        hidden = _gru_walk(
            features, self.w_iht, self.w_hh, self.b_ih, self.b_hh
        )

        last_direction = (
            smooth_prefix[-1]
            if float(np.linalg.norm(smooth_prefix[-1])) > 1e-9
            else np.asarray([1.0, 0.0], dtype=np.float64)
        )
        active = np.any(used != 0.0, axis=1)
        run_active = bool(active[-1])
        changes = np.flatnonzero(active[1:] != active[:-1])
        run_length = int(len(active) - 1 - changes[-1]) if len(changes) else len(active)
        tail = active[-RECENT_WINDOW:]
        nonzero_rows = used[active]
        last_axis = np.zeros(2, dtype=np.float64)
        for axis in range(2):
            indices = np.flatnonzero(used[:, axis] != 0.0)
            if len(indices):
                last_axis[axis] = used[indices[-1], axis]
        return PrefixWarmState(
            hidden=np.ascontiguousarray(hidden, dtype=np.float32),
            last_direction=np.ascontiguousarray(last_direction, dtype=np.float64),
            previous_emit=np.ascontiguousarray(used[-1], dtype=np.float32),
            last_nonzero=np.ascontiguousarray(
                nonzero_rows[-1] if len(nonzero_rows) else np.zeros(2),
                dtype=np.float32,
            ),
            last_axis_nonzero=np.ascontiguousarray(last_axis, dtype=np.float32),
            run_active=float(run_active),
            run_length=float(run_length),
            recent_zero=float(np.mean(~tail)) if len(tail) else 1.0,
            active_tail=np.ascontiguousarray(tail, dtype=np.float32),
            regime=np.ascontiguousarray(regime, dtype=np.float32),
            warm_ticks=int(len(used)),
        )

    def prepare_future(
        self,
        smooth_dxdy: np.ndarray,
        future_mask: np.ndarray,
        last_direction: np.ndarray,
    ) -> FutureTables:
        smooth = np.ascontiguousarray(np.asarray(smooth_dxdy, dtype=np.float32))
        mask = np.ascontiguousarray(np.asarray(future_mask, dtype=np.float32))
        require(smooth.shape == (HORIZON, 2) and mask.shape == (HORIZON,),
                "future layout differs")
        duration = int(np.sum(mask > 0.5))
        require(1 <= duration <= HORIZON, "future duration differs")
        require(np.all(mask[:duration] > 0.5) and np.all(mask[duration:] < 0.5),
                "future mask is not contiguous")
        require(np.array_equal(smooth[duration:], np.zeros_like(smooth[duration:])),
                "smooth intent is nonzero after mask")
        smooth_all = np.zeros((HORIZON, 2), dtype=np.float32)
        tangents = np.zeros((HORIZON, 2), dtype=np.float32)
        normals = np.zeros((HORIZON, 2), dtype=np.float32)
        static = np.zeros((HORIZON, 5), dtype=np.float32)
        _begin_precompute(
            np.ascontiguousarray(smooth[:duration], dtype=np.float64),
            np.ascontiguousarray(last_direction, dtype=np.float64),
            smooth_all,
            tangents,
            normals,
            static,
        )
        return FutureTables(smooth_all, tangents, normals, static, duration)

    def begin_event(
        self,
        warm: PrefixWarmState,
        future: FutureTables,
        causal_c_state: np.ndarray,
        event_seed_u64: int,
        *,
        lateral_penalty: float = LATERAL_PENALTY,
    ) -> "FastRendererEvent":
        # C is fixed at the event boundary.  Always own the three floats so a
        # caller reusing its session-state buffer cannot mutate an in-flight
        # stream (or leak the query outcome into later ticks).
        state = np.array(
            causal_c_state, dtype=np.float32, order="C", copy=True
        )
        require(state.shape == (STATE_WIDTH,) and np.all(np.isfinite(state)),
                "causal C state differs")
        require(float(np.max(np.abs(state))) <= 2.5, "causal C state exceeds clip")
        require(type(event_seed_u64) is int and 0 <= event_seed_u64 < 2**64,
                "event seed is not uint64")
        require(self.c_adapter is not None or np.array_equal(state, np.zeros_like(state)),
                "nonzero C state has no adapter")
        state.setflags(write=False)
        return FastRendererEvent(
            kernel=self,
            warm=warm,
            future=future,
            causal_c_state=state,
            event_seed_u64=event_seed_u64,
            lateral_penalty=float(lateral_penalty),
        )

    def prewarm(self) -> None:
        """Compile every hot kernel without consuming a real event."""

        prefix = np.zeros((160, 2), dtype=np.float32)
        prefix[-1, 0] = 1.0
        warm = self.prepare_prefix(prefix)
        smooth = np.zeros((HORIZON, 2), dtype=np.float32)
        smooth[:2, 0] = (0.4, 0.4)
        mask = np.zeros(HORIZON, dtype=np.float32)
        mask[:2] = 1.0
        future = self.prepare_future(smooth, mask, warm.last_direction)
        event = self.begin_event(warm, future, np.zeros(STATE_WIDTH, np.float32), 0)
        event.step()


class FastRendererEvent:
    """Explicit state-carry stream; `step()` emits exactly one 1 ms mark."""

    def __init__(
        self,
        *,
        kernel: FastRendererKernel,
        warm: PrefixWarmState,
        future: FutureTables,
        causal_c_state: np.ndarray,
        event_seed_u64: int,
        lateral_penalty: float,
    ) -> None:
        self.kernel = kernel
        self.future = future
        self.causal_c_state = causal_c_state
        self.event_seed_u64 = int(event_seed_u64)
        self.lateral_penalty = np.float32(max(lateral_penalty, 0.0))
        self.tick_index = 0
        self.hidden = warm.hidden.copy()
        self.state = np.zeros(11, dtype=np.float32)
        self.state[2:4] = warm.previous_emit
        self.state[4:6] = warm.last_nonzero
        self.state[6:8] = warm.last_axis_nonzero
        self.state[8] = warm.run_active
        self.state[9] = warm.run_length
        self.state[10] = warm.recent_zero
        self.active_ring = np.zeros(RECENT_WINDOW, dtype=np.float32)
        length = len(warm.active_tail)
        if length:
            self.active_ring[:length] = warm.active_tail
        self.ring_meta = np.asarray(
            [length % RECENT_WINDOW, length], dtype=np.int64
        )
        self.regime = warm.regime.copy()
        self.output = np.zeros((HORIZON, 2), dtype=np.int16)

        width = int(kernel.w_ih.shape[1])
        hidden = int(kernel.w_hh.shape[1])
        grid = int(kernel.w_off.shape[0])
        rank = int(kernel.c_wh.shape[0])
        self.feature_buf = np.zeros(width, dtype=np.float32)
        self.adapted_hidden = np.zeros(hidden, dtype=np.float32)
        self.logits = np.zeros(grid, dtype=np.float32)
        self.probabilities = np.zeros(grid, dtype=np.float32)
        self.hidden_buf = np.zeros(hidden, dtype=np.float32)
        self.input_gate_buf = np.zeros(3 * hidden, dtype=np.float32)
        self.hidden_gate_buf = np.zeros(3 * hidden, dtype=np.float32)
        self.c_q = np.zeros(rank, dtype=np.float32)
        self.c_hproj = np.zeros(rank, dtype=np.float32)
        self.c_sproj = np.zeros(rank, dtype=np.float32)
        self.has_c = np.bool_(not np.array_equal(
            causal_c_state, np.zeros_like(causal_c_state)
        ))

    @property
    def duration(self) -> int:
        return int(self.future.duration)

    @property
    def complete(self) -> bool:
        return self.tick_index >= self.duration

    def step(self) -> np.ndarray:
        require(not self.complete, "Renderer event is complete")
        kernel = self.kernel
        tick = self.tick_index
        mark0, mark1 = _current_tick(
            np.int64(tick),
            self.future.smooth,
            self.future.tangents,
            self.future.normals,
            self.future.static,
            self.regime,
            kernel.w_ih,
            kernel.w_hh,
            kernel.b_ih,
            kernel.b_hh,
            kernel.w_emit,
            kernel.b_emit,
            kernel.w_off,
            kernel.b_off,
            self.has_c,
            self.causal_c_state,
            kernel.c_wh,
            kernel.c_ws,
            kernel.c_u,
            kernel.offsets,
            kernel.ring_id,
            kernel.n_rings,
            kernel.tau_emit,
            kernel.tau_mag,
            kernel.tau_dir,
            kernel.hysteresis,
            kernel.force_release,
            kernel.max_abs_count,
            self.lateral_penalty,
            np.float32(LATERAL_FREE),
            np.uint64(self.event_seed_u64),
            self.hidden,
            self.state,
            self.active_ring,
            self.ring_meta,
            self.feature_buf,
            self.adapted_hidden,
            self.logits,
            self.probabilities,
            self.hidden_buf,
            self.input_gate_buf,
            self.hidden_gate_buf,
            self.c_q,
            self.c_hproj,
            self.c_sproj,
        )
        self.output[tick, 0] = np.int16(mark0)
        self.output[tick, 1] = np.int16(mark1)
        self.tick_index += 1
        return self.output[tick].copy()

    def render_remaining(self) -> np.ndarray:
        while not self.complete:
            self.step()
        return self.output.copy()


__all__ = [
    "FastRendererEvent",
    "FastRendererKernel",
    "FutureTables",
    "LATERAL_FREE",
    "LATERAL_PENALTY",
    "PrefixWarmState",
    "RuntimeContractError",
]
