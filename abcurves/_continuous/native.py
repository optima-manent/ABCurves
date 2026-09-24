"""Optional Numba kernels.  Cached compilation, no fastmath, no thread pool.

Import this module only after selecting the numba backend.  Physical arithmetic
stays double precision; neural-input assignments establish float32 boundaries.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .kernels import POSITION_SCALE, VELOCITY_SCALE, _SCALES


@njit(cache=True)
def motor_features(history, position, target, available, valid, motion_known, cut_us):
    known = np.empty(640, np.bool_)
    total_x = 0.
    total_y = 0.
    coverage = 0
    for i in range(640):
        known[i] = valid[i] and available[i] <= cut_us+(i-639)*1000
        if motion_known[i]:
            total_x += history[i, 0]
            total_y += history[i, 1]
            coverage += 1
    coarse = np.empty((1, 70, 9), np.float32)
    fine = np.empty((1, 54), np.float32)
    dynamics = np.empty((1, 8), np.float32)
    cumulative_x = 0.
    cumulative_y = 0.
    end = 0
    for token in range(70):
        width = 16 if token < 30 else 4
        begin = end
        end += width
        sum_x = 0.
        sum_y = 0.
        count = 0
        for j in range(begin, end):
            if motion_known[j]:
                sum_x += history[j, 0]
                sum_y += history[j, 1]
                cumulative_x += history[j, 0]
                cumulative_y += history[j, 1]
                count += 1
        coarse[0, token, 0] = sum_x/width/VELOCITY_SCALE
        coarse[0, token, 1] = sum_y/width/VELOCITY_SCALE
        i = end-1
        for d in range(2):
            rel = 0.
            if known[i] and motion_known[i]:
                point = position[d]+(cumulative_x if d == 0 else cumulative_y)-(total_x if d == 0 else total_y)
                rel = math.asinh((target[i, d]-point)/POSITION_SCALE)
            coarse[0, token, 2+d] = rel
            tv = 0.
            if i >= 32 and known[i] and known[i-32]:
                tv = (target[i, d]-target[i-32, d])/32.
            coarse[0, token, 4+d] = tv/VELOCITY_SCALE
        coarse[0, token, 6] = known[i]
        coarse[0, token, 7] = count/width
        coarse[0, token, 8] = 1. if token < 30 else .25
    for i in range(16):
        for d in range(2):
            fine[0, i*2+d] = history[624+i, d]/VELOCITY_SCALE if motion_known[624+i] else 0.
        fine[0, 32+i] = motion_known[624+i]
    for d in range(2):
        fine[0, 48+d] = math.asinh((target[639, d]-position[d])/POSITION_SCALE) if known[639] else 0.
        fine[0, 50+d] = (target[639, d]-target[607, d])/32./VELOCITY_SCALE if known[639] and known[607] else 0.
    fine[0, 52] = known[639]
    fine[0, 53] = coverage/640.
    valid_acc = True
    for i in range(40, 48):
        valid_acc = valid_acc and fine[0, i] > .5
    for d in range(2):
        a = np.float32(0.)
        b = np.float32(0.)
        for i in range(4):
            a = np.float32(a+fine[0, (12+i)*2+d])
            b = np.float32(b+fine[0, (8+i)*2+d])
        a = np.float32(a/np.float32(4.))
        b = np.float32(b/np.float32(4.))
        acc = np.float32(a-b)
        acc = np.float32(acc*np.float32(VELOCITY_SCALE))
        acc = np.float32(acc/np.float32(4.))
        acc = np.float32(acc/np.float32(.25)) if valid_acc else np.float32(0.)
        dynamics[0, d] = fine[0, 30+d]
        dynamics[0, 2+d] = acc
    ex = math.sinh(np.float64(fine[0, 48]))*POSITION_SCALE
    ey = math.sinh(np.float64(fine[0, 49]))*POSITION_SCALE
    magnitude = math.sqrt(ex*ex+ey*ey+100.)
    dx = np.float32(ex/magnitude) if fine[0, 52] > .5 else np.float32(0.)
    dy = np.float32(ey/magnitude) if fine[0, 52] > .5 else np.float32(0.)
    vx = np.float32(fine[0, 30]-fine[0, 50])
    vy = np.float32(fine[0, 31]-fine[0, 51])
    ax, ay = dynamics[0, 2], dynamics[0, 3]
    dynamics[0, 4] = np.float32(np.float32(vx*dx)+np.float32(vy*dy))
    dynamics[0, 5] = np.float32(np.float32(vx*(-dy))+np.float32(vy*dx))
    dynamics[0, 6] = np.float32(np.float32(ax*dx)+np.float32(ay*dy))
    dynamics[0, 7] = np.float32(np.float32(ax*(-dy))+np.float32(ay*dx))
    return coarse, fine, dynamics


@njit(cache=True)
def _norm(x, y):
    return math.sqrt(x*x+y*y)


@njit(cache=True)
def _mean(history, start, end):
    x, y = 0., 0.
    for i in range(start, end):
        x += history[i, 0]
        y += history[i, 1]
    return x/(end-start), y/(end-start)


@njit(cache=True)
def event_features(h, p, g, available, valid, motion_known, cut_us,
                   hold_age, hold_target, hold_position, initial_error,
                   innovation_age, mode):
    known = np.empty(640, np.bool_)
    known_context = np.empty(640, np.bool_)
    motion_count = 0
    for i in range(640):
        known_context[i] = valid[i] and available[i] <= cut_us+(i-639)*1000
        known[i] = known_context[i] and math.isfinite(g[i, 0]) and math.isfinite(g[i, 1])
        motion_count += int(motion_known[i])
    latest = known[639]
    gx = g[639, 0] if latest else p[0]
    gy = g[639, 1] if latest else p[1]
    ex, ey = gx-p[0], gy-p[1]
    distance = _norm(ex, ey)
    axis_x, axis_y = ex/max(distance, 1e-8), ey/max(distance, 1e-8)
    v1x, v1y = h[639, 0], h[639, 1]
    v8x, v8y = _mean(h, 632, 640)
    v32x, v32y = _mean(h, 608, 640)
    previous_x, previous_y = _mean(h, 624, 632)
    acc_x, acc_y = (v8x-previous_x)/8., (v8y-previous_y)/8.
    s1, s8, s32 = _norm(v1x, v1y), _norm(v8x, v8y), _norm(v32x, v32y)
    tv = np.zeros((3, 2), np.float64)
    periods = (8, 32, 128)
    for j in range(3):
        k = periods[j]
        if latest and known[639-k]:
            tv[j, 0] = (gx-g[639-k, 0])/k
            tv[j, 1] = (gy-g[639-k, 1])/k
    ts8, ts32, ts128 = _norm(tv[0, 0], tv[0, 1]), _norm(tv[1, 0], tv[1, 1]), _norm(tv[2, 0], tv[2, 1])
    growth = 0.
    if latest and known[607]:
        old_distance = _norm(g[607, 0]-(p[0]-v32x*32.), g[607, 1]-(p[1]-v32y*32.))
        growth = (distance-old_distance)/32.
    last_change = -1
    excursion = 0.
    target_count = 0
    for i in range(512, 640):
        d = _norm(g[i, 0]-gx, g[i, 1]-gy)
        if d > 1e-8 or not known[i]:
            last_change = i-512
        if known[i]:
            excursion = max(excursion, d)
            target_count += 1
    raw = np.empty(22, np.float32)
    raw[0], raw[1], raw[2], raw[3] = distance, s1, s8, s32
    raw[4] = s8/max(s32, .005)
    raw[5], raw[6] = ts32, ts128
    raw[7] = v8x*axis_x+v8y*axis_y
    raw[8] = abs(v8x*axis_y-v8y*axis_x)
    raw[9] = growth
    raw[10] = -(acc_x*v8x+acc_y*v8y)/max(s8, .005)
    raw[11], raw[12] = excursion, 127-last_change
    raw[13], raw[14], raw[15] = target_count/128., (1. if latest else 0.), motion_count/640.
    raw[16] = min(hold_age, 4096.)
    raw[17] = _norm(gx-hold_target[0], gy-hold_target[1]) if mode != 0 else 0.
    raw[18] = _norm(p[0]-hold_position[0], p[1]-hold_position[1]) if mode != 0 else 0.
    raw[19] = ts8
    raw[20] = (v8x*tv[1, 0]+v8y*tv[1, 1])/max(s8*ts32, 1e-8)
    raw[21] = initial_error if mode != 0 else 0.
    cgx = g[639, 0] if known_context[639] else p[0]
    cgy = g[639, 1] if known_context[639] else p[1]
    cex, cey = cgx-p[0], cgy-p[1]
    cd = _norm(cex, cey)
    if cd > 1e-6:
        axis_x, axis_y = cex/max(cd, 1e-9), cey/max(cd, 1e-9)
    elif s8 > 1e-6:
        axis_x, axis_y = v8x/max(s8, 1e-9), v8y/max(s8, 1e-9)
    else:
        axis_x, axis_y = 1., 0.
    basis = np.empty((2, 2), np.float64)
    basis[0, 0], basis[1, 0] = axis_x, axis_y
    basis[0, 1], basis[1, 1] = -axis_y, axis_x
    extra = np.empty(10, np.float64)
    extra[0], extra[1] = v1x*axis_x+v1y*axis_y, -v1x*axis_y+v1y*axis_x
    extra[2], extra[3] = (acc_x*axis_x+acc_y*axis_y)*8., (-acc_x*axis_y+acc_y*axis_x)*8.
    for j in range(2):
        k = 32 if j == 0 else 128
        tx, ty = 0., 0.
        if known_context[639] and known_context[639-k]:
            tx, ty = (cgx-g[639-k, 0])/k, (cgy-g[639-k, 1])/k
        extra[4+j*2] = tx*axis_x+ty*axis_y
        extra[5+j*2] = -tx*axis_y+ty*axis_x
    extra[8] = -v8x*axis_y+v8y*axis_x
    extra[9] = min(2048., innovation_age)/128.
    x = np.empty((1, 32), np.float32)
    for j in range(22):
        value = min(np.float64(raw[j]), 256.) if j == 16 else np.float64(raw[j])
        x[0, j] = math.asinh(value/_SCALES[j])
    for j in range(10):
        x[0, 22+j] = math.asinh(extra[j])
    return raw, x, basis


@njit(cache=True)
def c2_path(velocity, acceleration, duration, coefficients, times):
    out = np.empty((len(times), 2), np.float64)
    for i in range(len(times)):
        s = min(1., max(0., times[i]/duration))
        z = 2*s-1
        e = 64*s**3*(1-s)**3
        carry = s-6*s**3+8*s**4-3*s**5
        acc = .5*s*s*(1-s)**3
        goal = 10*s**3-15*s**4+6*s**5
        r0, r1, r2, r3 = e, e*z, e*(3*z*z-1)/2, e*(5*z*z*z-3*z)/2
        for d in range(2):
            res = r0*coefficients[1, d]+r1*coefficients[2, d]+r2*coefficients[3, d]+r3*coefficients[4, d]
            out[i, d] = (carry*duration*velocity[d]+acc*duration**2*acceleration[d]
                         +goal*coefficients[0, d]+res)
    return out


@njit(cache=True)
def selector_inputs(encoded, heads, previous, actual, has_previous):
    unary = np.empty((1, 16, 112), np.float32)
    pairs = np.zeros((1, 16, 20), np.float32)
    previous_sum = np.zeros(2, np.float32)
    previous_prefix_sum = np.zeros(2, np.float32)
    if has_previous:
        for d in range(2):
            for i in range(16):
                previous_sum[d] = np.float32(previous_sum[d]+previous[i, d])
                if i < 8:
                    previous_prefix_sum[d] = np.float32(previous_prefix_sum[d]+previous[i, d])
    for head in range(16):
        for i in range(96):
            unary[0, head, i] = encoded[i]
        for i in range(8):
            for d in range(2):
                unary[0, head, 96+2*i+d] = np.arcsinh(heads[head, i, d])
                if has_previous:
                    pairs[0, head, 2*i+d] = np.arcsinh(np.float32(previous[8+i, d]-heads[head, i, d]))
        if has_previous:
            for d in range(2):
                total = np.float32(0.)
                for i in range(8):
                    total = np.float32(total+heads[head, i, d])
                endpoint = np.float32(np.float32(previous_sum[d]*np.float32(4.))-actual[d])
                endpoint = np.float32(endpoint-np.float32(total*np.float32(4.)))
                endpoint = np.float32(endpoint/np.float32(32.))
                feedback = np.float32(np.float32(previous_prefix_sum[d]*np.float32(4.))-actual[d])
                feedback = np.float32(feedback/np.float32(32.))
                pairs[0, head, 16+d] = np.arcsinh(endpoint)
                pairs[0, head, 18+d] = np.arcsinh(feedback)
    return unary, pairs
