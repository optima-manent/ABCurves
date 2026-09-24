"""Batch-one B2 physical preprocessing, separate from learned neural layers.

Physical observations and decoder geometry stay float64.  The deliberate
float32 boundaries match the frozen motor, movement selector, and event model.
No research modules or Torch are imported by this module.  ``backend='numba'``
selects the optional cached, strict-arithmetic implementation lazily.
"""
from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np

VELOCITY_SCALE = 1.1982087601807259
POSITION_SCALE = 81.73350722609987
_INDEX = np.r_[np.arange(15, 480, 16), np.arange(483, 640, 4)]
_OFFSETS_US = np.arange(-639, 1, dtype=np.int64) * 1000
_SCALES = np.array([10, .1, .1, .1, 1, .1, .1, .1, .1, .1, .01, 10,
                    128, 1, 1, 1, 128, 10, 10, .1, 1, 10], np.float64)
_ZERO_HEAD = np.zeros((16, 2), np.float32)
_ZERO_STEP = np.zeros(2, np.float32)


@lru_cache(maxsize=1)
def _native():
    try:
        return importlib.import_module('.native', __package__)
    except ImportError as exc:
        raise ImportError("The numba backend requires the optional numba dependency") from exc


def motor_features(history, position, target, available, valid, motion_known,
                   cut_us, *, backend='numpy'):
    """Return ``coarse[1,70,9], fine[1,54], dynamics[1,8]`` float32.

    Inputs are chronological, unbatched 640 ms arrays; the last slot is at
    ``cut_us``.  Availability is checked against each slot's own physical time.
    """
    if backend == 'numba':
        return _native().motor_features(history, position, target, available,
                                        valid, motion_known, cut_us)
    if backend != 'numpy':
        raise ValueError('Unknown physical-kernel backend')
    known = valid & (available <= cut_us + _OFFSETS_US)
    motion = np.where(motion_known[:, None], history, 0.)
    goal = np.where(known[:, None], target, 0.)
    pair = known[32:] & known[:-32]
    tv = np.zeros((640, 2), np.float64)
    tv[32:] = np.where(pair[:, None], (goal[32:] - goal[:-32]) / 32., 0.)
    points = position + motion.cumsum(0) - motion.sum(0)
    rel = np.where((known & motion_known)[:, None],
                   np.arcsinh((goal - points) / POSITION_SCALE), 0.)
    c = np.empty((1, 70, 9), np.float32)
    c[0, :30, :2] = motion[:480].reshape(30, 16, 2).mean(1) / VELOCITY_SCALE
    c[0, 30:, :2] = motion[480:].reshape(40, 4, 2).mean(1) / VELOCITY_SCALE
    c[0, :, 2:4] = rel[_INDEX]
    c[0, :, 4:6] = tv[_INDEX] / VELOCITY_SCALE
    c[0, :, 6] = known[_INDEX]
    c[0, :30, 7] = motion_known[:480].reshape(30, 16).mean(1)
    c[0, 30:, 7] = motion_known[480:].reshape(40, 4).mean(1)
    c[0, :30, 8] = 1.
    c[0, 30:, 8] = .25
    f = np.empty((1, 54), np.float32)
    f[0, :32] = (motion[-16:] / VELOCITY_SCALE).reshape(32)
    f[0, 32:48] = motion_known[-16:]
    f[0, 48:50] = np.arcsinh((goal[-1] - position) / POSITION_SCALE) if known[-1] else 0.
    f[0, 50:52] = tv[-1] / VELOCITY_SCALE
    f[0, 52] = known[-1]
    f[0, 53] = motion_known.mean()
    return c, f, motor_dynamics(f)


def motor_dynamics(fine):
    """MC-A's smooth physical-direction feature, including float32 boundaries."""
    f = np.asarray(fine, np.float32).reshape(54)
    velocity = f[:32].reshape(16, 2)
    current = velocity[-1]
    acc = (velocity[-4:].mean(0) - velocity[-8:-4].mean(0))
    acc = acc * np.float32(VELOCITY_SCALE) / np.float32(4.)
    if not np.all(f[40:48] > .5):
        acc[:] = 0.
    acc = acc / np.float32(.25)
    error = np.sinh(f[48:50].astype(np.float64)) * POSITION_SCALE
    direction = (error / np.sqrt(np.sum(error * error) + 100.)).astype(np.float32)
    if f[52] <= .5:
        direction[:] = 0.
    normal = np.array([-direction[1], direction[0]], np.float32)
    relative = current - f[50:52]
    out = np.empty((1, 8), np.float32)
    out[0, :2], out[0, 2:4] = current, acc
    out[0, 4:] = ((relative * direction).sum(), (relative * normal).sum(),
                  (acc * direction).sum(), (acc * normal).sum())
    return out


def decoder_geometry(velocity_basis, carry_velocity):
    """Precompute the selector's sixteen 4 ms means from unchanged geometry."""
    return (np.ascontiguousarray(velocity_basis[:64].reshape(16, 4, 21).mean(1)),
            np.ascontiguousarray(carry_velocity[:64].reshape(16, 4).mean(1)))


def decode(coefficients, incoming, velocity_basis, carry_velocity):
    """Decode all supplied heads/times without unused diagnostic transforms."""
    q = np.asarray(coefficients, np.float64)
    return np.matmul(velocity_basis, q) + carry_velocity[:, None] * incoming


def decode_geometry(coefficients, incoming, mean_basis, mean_carry):
    """Compute all head geometry directly; only double reduction order changes."""
    return decode(coefficients, incoming, mean_basis, mean_carry).astype(np.float32)


def decode_selected(coefficients, incoming, velocity_basis, carry_velocity):
    """Decode one head for exactly the number of supplied physical rows."""
    return decode(coefficients, incoming, velocity_basis, carry_velocity)


def selector_geometry(forecasts):
    return forecasts[:, :64].reshape(16, 16, 4, 2).mean(2).astype(np.float32)


def selector_inputs(encoded, heads, previous=None, actual_step=None, *, backend='numpy'):
    """Return unary ``[1,16,112]`` and pair ``[1,16,20]`` neural inputs.

    A missing previous head returns zero pair features.  The caller must then
    use only unary logits, as for invalidated HOLD/BRAKE predecessor state.
    """
    if backend == 'numba':
        return _native().selector_inputs(np.asarray(encoded, np.float32).reshape(96),
            heads, _ZERO_HEAD if previous is None else np.asarray(previous, np.float32).reshape(16, 2),
            _ZERO_STEP if actual_step is None else np.asarray(actual_step, np.float32).reshape(2),
            previous is not None)
    if backend != 'numpy':
        raise ValueError('Unknown physical-kernel backend')
    unary = np.empty((1, 16, 112), np.float32)
    unary[0, :, :96] = np.asarray(encoded).reshape(96)
    unary[0, :, 96:] = np.arcsinh(heads[:, :8].reshape(16, 16))
    pairs = np.zeros((1, 16, 20), np.float32)
    if previous is not None:
        previous = np.asarray(previous, np.float32).reshape(16, 2)
        actual = np.asarray(actual_step, np.float32).reshape(2)
        pairs[0, :, :16] = (previous[None, 8:] - heads[:, :8]).reshape(16, 16)
        pairs[0, :, 16:18] = ((previous.sum(0) * np.float32(4.) - actual)
                                - heads[:, :8].sum(1) * np.float32(4.)) / np.float32(32.)
        pairs[0, :, 18:20] = (previous[:8].sum(0) * np.float32(4.) - actual) / np.float32(32.)
        np.arcsinh(pairs, out=pairs)
    return unary, pairs


def event_features(history, position, target, available, valid, motion_known,
                   cut_us, *, hold_age, hold_target, hold_position, initial_error,
                   innovation_age, mode, backend='numpy'):
    """Return physical raw[22], event input[1,32], and local basis[2,2]."""
    if backend == 'numba':
        return _native().event_features(history, position, target, available,
            valid, motion_known, cut_us, hold_age, hold_target, hold_position,
            initial_error, innovation_age, mode)
    if backend != 'numpy':
        raise ValueError('Unknown physical-kernel backend')
    h, p, g = history, position, target
    known_context = valid & (available <= cut_us + _OFFSETS_US)
    known = known_context & np.isfinite(g).all(-1)
    latest = bool(known[-1])
    goal = g[-1] if latest else p
    error = goal - p
    distance = np.linalg.norm(error)
    axis = error / max(distance, 1e-8)
    v1, v8, v32 = h[-1], h[-8:].mean(0), h[-32:].mean(0)
    s1, s8, s32 = np.linalg.norm(v1), np.linalg.norm(v8), np.linalg.norm(v32)
    tv = {k: (goal-g[-1-k])/k if latest and known[-1-k] else np.zeros(2)
          for k in (8, 32, 128)}
    ts8, ts32, ts128 = (np.linalg.norm(tv[k]) for k in (8, 32, 128))
    previous_position = p-h[-32:].sum(0)
    growth = (distance-np.linalg.norm(g[-33]-previous_position))/32 if latest and known[-33] else 0.
    acceleration = (v8-h[-16:-8].mean(0))/8
    excursions = np.linalg.norm(g[-128:] - goal, axis=-1)
    changes = (excursions > 1e-8) | ~known[-128:]
    change_indices = np.flatnonzero(changes)
    age = 127-int(change_indices[-1]) if len(change_indices) else 128
    excursion = np.max(np.where(known[-128:], excursions, 0.))
    raw = np.array([distance, s1, s8, s32, s8/max(s32, .005), ts32, ts128,
        np.dot(v8, axis), abs(v8[0]*axis[1]-v8[1]*axis[0]), growth,
        -np.dot(acceleration, v8)/max(s8, .005), excursion, age,
        known[-128:].mean(), latest, motion_known.mean(), min(hold_age, 4096.),
        np.linalg.norm(goal-hold_target), np.linalg.norm(p-hold_position), ts8,
        np.dot(v8, tv[32])/max(s8*ts32, 1e-8), initial_error], np.float32)
    if mode == 0:
        raw[17:19] = 0.
        raw[21] = 0.
    context_raw = raw.copy()
    context_raw[16] = min(context_raw[16], 256.)
    context_goal = g[-1] if known_context[-1] else p
    error = context_goal-p
    distance = np.linalg.norm(error)
    direction = error/max(distance, 1e-9) if distance > 1e-6 else v8/max(s8, 1e-9)
    if distance <= 1e-6 and s8 <= 1e-6:
        direction = np.array([1., 0.])
    basis = np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    context_tv = [(context_goal-g[-1-k])/k if known_context[-1] and known_context[-1-k]
                  else np.zeros(2) for k in (32, 128)]
    extra = np.r_[v1 @ basis, (acceleration @ basis)*8, context_tv[0] @ basis,
                  context_tv[1] @ basis, (v8 @ basis)[1], min(2048., innovation_age)/128.]
    x = np.empty((1, 32), np.float32)
    x[0, :22] = np.arcsinh(context_raw/_SCALES)
    x[0, 22:] = np.arcsinh(extra)
    return raw, x, basis


def c2_path(velocity, acceleration, duration, coefficients, times, *, backend='numpy'):
    """The finite C2 brake, including its fixed incoming acceleration."""
    if backend == 'numba':
        return _native().c2_path(velocity, acceleration, duration, coefficients, times)
    if backend != 'numpy':
        raise ValueError('Unknown physical-kernel backend')
    s = np.clip(np.asarray(times)/duration, 0., 1.)
    z = 2*s-1
    envelope = 64*s**3*(1-s)**3
    residual = np.stack((envelope, envelope*z, envelope*(3*z*z-1)/2,
                         envelope*(5*z*z*z-3*z)/2), -1)
    carry = s-6*s**3+8*s**4-3*s**5
    acc = .5*s*s*(1-s)**3
    goal = 10*s**3-15*s**4+6*s**5
    return (carry[:, None]*duration*velocity + acc[:, None]*duration**2*acceleration
            + goal[:, None]*coefficients[0] + residual @ coefficients[1:])
