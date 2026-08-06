"""Numba kernels for the final Planner and Renderer streaming path.

The kernels preserve the frozen float32 state transitions while moving the
recurrent and TCN hot loops outside Python. They are deterministic, cached
after first compilation on a machine, and release the GIL so prefix warming
can overlap the Planner forward pass.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=True)
def _matvec(w: np.ndarray, x: np.ndarray, out: np.ndarray) -> None:
    """out = w @ x for row-major (rows, n) f32 -- fastmath so LLVM vectorizes
    the reduction (reassociation only; same deviation class as BLAS order)."""

    for j in range(w.shape[0]):
        s = np.float32(0.0)
        for k in range(w.shape[1]):
            s += w[j, k] * x[k]
        out[j] = s


@njit(cache=True, nogil=True)
def _gru_gates(ig: np.ndarray, hg: np.ndarray, h: np.ndarray, hidden: np.int64, out: np.ndarray) -> None:
    """aten GRUCell body: r=sig(hg_r+ig_r), z=sig(hg_z+ig_z),
    n=tanh(ig_n + r*hg_n), h'=(h-n)*z+n. Gates in float64, cast at the write
    (same +-ulp class as SLEEF float32; measured in tests)."""

    for i in range(hidden):
        r = 1.0 / (1.0 + math.exp(-(np.float64(hg[i]) + np.float64(ig[i]))))
        z = 1.0 / (1.0 + math.exp(-(np.float64(hg[hidden + i]) + np.float64(ig[hidden + i]))))
        n = math.tanh(np.float64(ig[2 * hidden + i]) + r * np.float64(hg[2 * hidden + i]))
        out[i] = np.float32((np.float64(h[i]) - n) * z + n)


# Cephes expf() constants (degree-5 polynomial on the ln2-reduced argument)
_LOG2EF = np.float32(1.44269504088896341)
_EXPF_C1 = np.float32(0.693359375)
_EXPF_C2 = np.float32(-2.12194440e-4)
_EXPF_P0 = np.float32(1.9875691500e-4)
_EXPF_P1 = np.float32(1.3981999507e-3)
_EXPF_P2 = np.float32(8.3334519073e-3)
_EXPF_P3 = np.float32(4.1665795894e-2)
_EXPF_P4 = np.float32(1.6666665459e-1)
_EXPF_P5 = np.float32(5.0000001201e-1)


@njit(cache=True, nogil=True, fastmath=True)
def _expf_arr(x: np.ndarray, e: np.ndarray, ib: np.ndarray) -> None:
    """Elementwise float32 exp(x) -> e, branch-free so LLVM can vectorize.

    Cephes expf polynomial (measured max rel err 2.8e-7, ~2 ulp f32) with
    2**n built from int32 exponent bits (``ib`` is the int32 scratch; a viewed
    multiply replaces the unvectorizable ``ldexp``/``pow`` libm calls).
    Arguments are clamped to +-87, inside which (n + 127) << 23 is a valid
    normal float32.
    """

    n_ = x.shape[0]
    one = np.float32(1.0)
    half = np.float32(0.5)
    for i in range(n_):
        xv = min(max(x[i], np.float32(-87.0)), np.float32(87.0))
        nf = math.floor(_LOG2EF * xv + half)
        px = xv - np.float32(nf) * _EXPF_C1
        px = px - np.float32(nf) * _EXPF_C2
        y = ((((_EXPF_P0 * px + _EXPF_P1) * px + _EXPF_P2) * px + _EXPF_P3) * px + _EXPF_P4) * px + _EXPF_P5
        e[i] = y * px * px + px + one
        ib[i] = (np.int32(nf) + 127) << 23
    s = ib.view(np.float32)
    for i in range(n_):
        e[i] = e[i] * s[i]


@njit(cache=True, nogil=True, fastmath=True)
def _gru_gates_walk(
    ig: np.ndarray, hg: np.ndarray, h: np.ndarray, hidden: np.int64,
    out: np.ndarray, arg: np.ndarray, e: np.ndarray, ib: np.ndarray,
) -> None:
    """``_gru_gates`` restructured around the vectorizable ``_expf_arr``
    (sigmoid via exp(-x); tanh(a) via (1-exp(-2a))/(1+exp(-2a))), all float32.
    Part of the labelled deviation: vs the float64-libm gates the final warm
    ``h`` differs by ~6e-8 (one f32 ulp; measured), and ~2e-6 total vs aten.
    Used by the warm walk ONLY -- the tick kernel keeps
    ``_gru_gates`` so its validated arithmetic is untouched. ``arg``/``e``/
    ``ib`` are (3*hidden,) f32/f32/i32 scratch."""

    one = np.float32(1.0)
    two = np.float32(2.0)
    for i in range(2 * hidden):
        arg[i] = -(hg[i] + ig[i])
    _expf_arr(arg[: 2 * hidden], e[: 2 * hidden], ib[: 2 * hidden])
    for i in range(hidden):
        r = one / (one + e[i])
        arg[2 * hidden + i] = -two * (ig[2 * hidden + i] + r * hg[2 * hidden + i])
    _expf_arr(arg[2 * hidden :], e[2 * hidden :], ib[2 * hidden :])
    for i in range(hidden):
        z = one / (one + e[hidden + i])
        e2 = e[2 * hidden + i]
        n = (one - e2) / (one + e2)
        out[i] = (h[i] - n) * z + n


@njit(cache=True, nogil=True)
def _gru_walk(
    feats: np.ndarray,  # (P, F) f32
    w_iht: np.ndarray,  # (F, 3H) f32
    w_hh: np.ndarray,  # (3H, H) f32 row-major
    b_ih: np.ndarray,  # (3H,) f32
    b_hh: np.ndarray,  # (3H,) f32
) -> np.ndarray:
    hidden = w_hh.shape[1]
    igates = np.dot(feats, w_iht)  # one BLAS gemm for every step's input gates
    for t in range(igates.shape[0]):
        for j in range(igates.shape[1]):
            igates[t, j] += b_ih[j]
    h = np.zeros(hidden, dtype=np.float32)
    out = np.zeros(hidden, dtype=np.float32)
    hg = np.zeros(w_hh.shape[0], dtype=np.float32)
    arg = np.zeros(3 * hidden, dtype=np.float32)
    e = np.zeros(3 * hidden, dtype=np.float32)
    ib = np.zeros(3 * hidden, dtype=np.int32)
    for t in range(igates.shape[0]):
        _matvec(w_hh, h, hg)
        for j in range(hg.shape[0]):
            hg[j] += b_hh[j]
        _gru_gates_walk(igates[t], hg, h, hidden, out, arg, e, ib)
        h, out = out, h
    return h


# --------------------------------------------------------------------------- #
# Warm teacher features matching the canonical Renderer feature construction
# for a prefix-only warm call: reset at the end, state and label both prefix.
# --------------------------------------------------------------------------- #

_EPS = 1e-9  # Match abcurves.renderer.EPS.
_RECENT_WINDOW = 60  # Match abcurves.renderer.RECENT_WINDOW.


@njit(cache=True, nogil=True)
def _warm_features(smooth: np.ndarray, raw: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """Teacher features over the truncated prefix window (float64 math, f32 out).

    ``smooth`` is the (offline-exact) w5-smoothed prefix, ``raw`` the raw
    prefix, both float64 (T, 2); ``regime`` the 5 regime features (f32).
    Feature-for-feature transcription of the offline builder with
    ``reset_idx = n`` (phase pinned at 1.0, accumulator segment = whole window).
    """

    n = smooth.shape[0]
    nf = 16 + regime.shape[0]
    feats = np.zeros((n, nf), dtype=np.float32)

    # _kinematics
    speed = np.zeros(n, dtype=np.float64)
    for t in range(n):
        speed[t] = math.sqrt(smooth[t, 0] * smooth[t, 0] + smooth[t, 1] * smooth[t, 1])
    # _ffill_directions (seed [1, 0])
    lx, ly = 1.0, 0.0
    tang = np.zeros((n, 2), dtype=np.float64)
    for t in range(n):
        if speed[t] > _EPS:
            lx, ly = smooth[t, 0], smooth[t, 1]
        nrm = math.sqrt(lx * lx + ly * ly)
        if nrm <= _EPS:
            nrm = 1.0
        tang[t, 0] = lx / nrm
        tang[t, 1] = ly / nrm

    cs0, cs1, cr0, cr1 = 0.0, 0.0, 0.0, 0.0  # running sums for `desired`
    prev0, prev1 = 0.0, 0.0  # prev_emit[t] = raw[t-1]
    lnz0, lnz1 = 0.0, 0.0  # last nonzero row BEFORE t
    run_a, run_l = False, 0  # _run_state_before(active, False, 0)
    inact_hist = np.zeros(n + 1, dtype=np.float64)  # _recent_zero_rate_excl cumsum

    for t in range(n):
        # kinematics: accel = diff(speed, prepend=speed[0]); curvature via cos
        accel = speed[t] - (speed[t - 1] if t > 0 else speed[0])
        curv = 0.0
        if t > 0:
            na = speed[t - 1]
            nb = speed[t]
            if na > _EPS and nb > _EPS:
                cosv = (smooth[t - 1, 0] * smooth[t, 0] + smooth[t - 1, 1] * smooth[t, 1]) / (na * nb)
                if cosv > 1.0:
                    cosv = 1.0
                elif cosv < -1.0:
                    cosv = -1.0
                curv = 1.0 - cosv

        # desired[t] = sum(smooth[:t+1]) - sum(raw[:t])  (one segment; reset at n)
        cs0 += smooth[t, 0]
        cs1 += smooth[t, 1]
        if t > 0:
            cr0 += raw[t - 1, 0]
            cr1 += raw[t - 1, 1]
        des0 = cs0 - cr0
        des1 = cs1 - cr1

        active_t = raw[t, 0] != 0.0 or raw[t, 1] != 0.0

        # _run_state_before(active, False, 0): state BEFORE t
        ra = 1.0 if run_a else 0.0
        rl = float(run_l)
        if t == 0 and run_l == 0:
            run_a, run_l = active_t, 1
        elif active_t != run_a:
            run_a, run_l = active_t, 1
        else:
            run_l += 1

        # _recent_zero_rate_excl(active, 1.0)
        if t == 0:
            rz = 1.0
        else:
            lo = t - _RECENT_WINDOW
            if lo < 0:
                lo = 0
            rz = (inact_hist[t] - inact_hist[lo]) / float(t - lo)
        inact_hist[t + 1] = inact_hist[t] + (0.0 if active_t else 1.0)

        tx, ty = tang[t, 0], tang[t, 1]
        nx, ny = -ty, tx
        acc_t = des0 * tx + des1 * ty
        acc_n = des0 * nx + des1 * ny
        prev_t = prev0 * tx + prev1 * ty
        prev_n = prev0 * nx + prev1 * ny
        lnz_t = lnz0 * tx + lnz1 * ty
        lnz_n = lnz0 * nx + lnz1 * ny

        feats[t, 0] = np.float32(speed[t] / 4.0)
        feats[t, 1] = np.float32(min(max(accel, -3.0), 3.0))
        feats[t, 2] = np.float32(curv)
        feats[t, 3] = np.float32(1.0)  # phase: whole window is prefix
        feats[t, 4] = np.float32(tx)
        feats[t, 5] = np.float32(ty)
        feats[t, 6] = np.float32(min(max(acc_t, -4.0), 4.0))
        feats[t, 7] = np.float32(min(max(acc_n, -4.0), 4.0))
        feats[t, 8] = np.float32(min(max(prev_t / 4.0, -4.0), 4.0))
        feats[t, 9] = np.float32(min(max(prev_n / 4.0, -4.0), 4.0))
        feats[t, 10] = np.float32(ra)
        feats[t, 11] = np.float32(min(rl, 64.0) / 64.0)
        feats[t, 12] = np.float32(rz)
        feats[t, 13] = np.float32(math.log1p(speed[t]))
        feats[t, 14] = np.float32(min(max(lnz_t / 4.0, -4.0), 4.0))
        feats[t, 15] = np.float32(min(max(lnz_n / 4.0, -4.0), 4.0))
        for j in range(regime.shape[0]):
            feats[t, 16 + j] = regime[j]

        prev0, prev1 = raw[t, 0], raw[t, 1]
        if active_t:
            lnz0, lnz1 = raw[t, 0], raw[t, 1]

    return feats


@njit(cache=True, nogil=True)
def _regime_features_nb(prefix: np.ndarray, window: np.int64) -> np.ndarray:
    """Compute the canonical Renderer prefix-regime summary in one nogil kernel.

    Same float64 arithmetic; the FFT band power is a direct DFT over the same
    rfft bins (30 <= f < 500 Hz), reordered summation only. Measured bit-equal
    to the numpy/fft original after the float32 output cast on every fixture
    event (the f64 reorder noise, ~1e-13 relative, is far below f32
    resolution). nogil matters: this runs on the warm worker thread while the
    main thread is inside ``plan()``'s Python glue -- the numpy original
    serializes both on the GIL.
    """

    n = prefix.shape[0]
    w = window if window >= 1 else 1
    lo = n - w if n > w else 0
    m = n - lo
    out = np.zeros(5, dtype=np.float32)
    if m <= 0:
        return out
    mag = np.zeros(m, dtype=np.float64)
    for i in range(m):
        a0 = abs(prefix[lo + i, 0])
        a1 = abs(prefix[lo + i, 1])
        mag[i] = a0 if a0 > a1 else a1
    n_active = 0
    s_active = 0.0
    for i in range(m):
        if mag[i] > 0.0:
            n_active += 1
            s_active += mag[i]
    active_rate = n_active / m
    mean_mag = s_active / n_active if n_active else 0.0
    # np.percentile(active_mag, 95): default 'linear' interpolation
    p95 = 0.0
    if n_active:
        am = np.zeros(n_active, dtype=np.float64)
        j = 0
        for i in range(m):
            if mag[i] > 0.0:
                am[j] = mag[i]
                j += 1
        am.sort()
        if n_active == 1:
            p95 = am[0]
        else:
            pos = 0.95 * (n_active - 1)
            k = int(pos)
            frac = pos - k
            p95 = am[k] + (am[k + 1] - am[k]) * frac if k + 1 < n_active else am[n_active - 1]
    # Canonical recent-window sign-flip rate (per axis, nonzero values).
    flips = 0
    denom = 0
    for axis in range(2):
        prev = 0.0
        started = False
        for i in range(m):
            v = prefix[lo + i, axis]
            if abs(v) > 0.0:
                s = 1.0 if v > 0 else -1.0
                if started:
                    if s != prev:
                        flips += 1
                    denom += 1
                prev = s
                started = True
    sign_flip = flips / denom if denom else 0.0
    # hf band power: sum |rfft(mag - mean)|^2 over 30 <= freq < 500 Hz, / m
    hf = 0.0
    if m >= 8:
        mean_all = 0.0
        for i in range(m):
            mean_all += mag[i]
        mean_all /= m
        acc = 0.0
        for k in range(1, m // 2 + 1):
            f = k / (m * 1e-3)  # rfftfreq(m, d=1e-3)
            if f < 30.0 or f >= 500.0:
                continue
            cr = 0.0
            ci = 0.0
            ang0 = -2.0 * math.pi * k / m
            for i in range(m):
                v = mag[i] - mean_all
                a = ang0 * i
                cr += v * math.cos(a)
                ci += v * math.sin(a)
            acc += cr * cr + ci * ci
        hf = acc / m
    out[0] = np.float32(active_rate)
    out[1] = np.float32(math.log1p(mean_mag))
    out[2] = np.float32(math.log1p(p95))
    out[3] = np.float32(sign_flip)
    out[4] = np.float32(math.log1p(hf))
    for i in range(5):
        if not np.isfinite(out[i]):
            out[i] = np.float32(0.0)
    return out



@njit(cache=True, nogil=True)
def _smooth_w5(raw: np.ndarray) -> np.ndarray:
    """``smooth_dxdy(prefix.astype(f32), w5).astype(f64)``: delta -> path,
    two edge-padded centered moving averages (window 5), path -> delta,
    with the offline's f64 -> f32 -> f64 dtype round-trips."""

    n = raw.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)
    if n == 0:
        return out
    valid = np.zeros((n, 2), dtype=np.float64)
    for t in range(n):
        valid[t, 0] = np.float64(np.float32(raw[t, 0]))
        valid[t, 1] = np.float64(np.float32(raw[t, 1]))
    if n <= 1:
        for t in range(n):
            out[t, 0] = np.float64(np.float32(valid[t, 0]))
            out[t, 1] = np.float64(np.float32(valid[t, 1]))
        return out
    path = np.zeros((n, 2), dtype=np.float64)
    c0, c1 = 0.0, 0.0
    for t in range(n):
        c0 += valid[t, 0]
        c1 += valid[t, 1]
        path[t, 0] = c0
        path[t, 1] = c1
    once = np.zeros((n, 2), dtype=np.float64)
    twice = np.zeros((n, 2), dtype=np.float64)
    for src, dst in ((path, once), (once, twice)):
        for axis in range(2):
            for i in range(n):
                s = 0.0
                for k in range(5):
                    j = i + k - 2  # edge-padded index
                    if j < 0:
                        j = 0
                    elif j >= n:
                        j = n - 1
                    s += src[j, axis] * 0.2
                dst[i, axis] = s
    for t in range(n):
        p0 = twice[t - 1, 0] if t > 0 else 0.0
        p1 = twice[t - 1, 1] if t > 0 else 0.0
        out[t, 0] = np.float64(np.float32(twice[t, 0] - p0))
        out[t, 1] = np.float64(np.float32(twice[t, 1] - p1))
    return out


@njit(cache=True, nogil=True)
def _begin_precompute(
    smooth64: np.ndarray,  # (d, 2) f64 intent slice
    last_dir: np.ndarray,  # (2,) f64 warm seam direction
    smooth_all: np.ndarray,  # (H, 2) f32 out
    tang: np.ndarray,  # (H, 2) f32 out
    norm: np.ndarray,  # (H, 2) f32 out
    stat: np.ndarray,  # (H, 5) f32 out
) -> None:
    """Compute canonical Renderer kinematics, filled directions, and phase.

    This also builds the static feature table for one intent using float64 math
    and float32 writes, matching the Torch tier's NumPy precompute up to libm
    rounding.
    """

    d = smooth64.shape[0]
    lx, ly = last_dir[0], last_dir[1]
    if math.sqrt(lx * lx + ly * ly) <= _EPS:
        lx, ly = 1.0, 0.0
    prev_speed = 0.0
    denom = float(d - 1) if d > 1 else 1.0
    for t in range(d):
        sx, sy = smooth64[t, 0], smooth64[t, 1]
        smooth_all[t, 0] = np.float32(sx)
        smooth_all[t, 1] = np.float32(sy)
        speed = math.sqrt(sx * sx + sy * sy)
        accel = speed - (prev_speed if t > 0 else speed)
        curv = 0.0
        if t > 0:
            na = prev_speed
            nb = speed
            if na > _EPS and nb > _EPS:
                cosv = (smooth64[t - 1, 0] * sx + smooth64[t - 1, 1] * sy) / (na * nb)
                if cosv > 1.0:
                    cosv = 1.0
                elif cosv < -1.0:
                    cosv = -1.0
                curv = 1.0 - cosv
        if speed > _EPS:
            lx, ly = sx, sy
        nrm = math.sqrt(lx * lx + ly * ly)
        if nrm <= _EPS:
            nrm = 1.0
        tx, ty = lx / nrm, ly / nrm
        tang[t, 0] = np.float32(tx)
        tang[t, 1] = np.float32(ty)
        norm[t, 0] = np.float32(-ty)
        norm[t, 1] = np.float32(tx)
        stat[t, 0] = np.float32(speed / 4.0)
        stat[t, 1] = np.float32(min(max(accel, -3.0), 3.0))
        stat[t, 2] = np.float32(curv)
        stat[t, 3] = np.float32(float(t) / denom)
        stat[t, 4] = np.float32(math.log1p(speed))
        prev_speed = speed


@njit(cache=True, nogil=True)
def _tcn_forward(
    win: np.ndarray,  # (W, 3) f32 -- last-W rows of the normalized prefix tensor
    summary: np.ndarray,  # (S,) f32
    w_in_t: np.ndarray,  # (3, C) f32   input conv k=1
    b_in: np.ndarray,  # (C,)
    conv_w: np.ndarray,  # (n_conv, 5*C, C) f32  im2col-layout weights
    conv_b: np.ndarray,  # (n_conv, C)
    ln_w: np.ndarray,  # (n_conv, C)
    ln_b: np.ndarray,  # (n_conv, C)
    ln_eps: np.float64,
    w_sum_t: np.ndarray,  # (S, C)
    b_sum: np.ndarray,  # (C,)
    w_trunk_t: np.ndarray,  # (2C, C)
    b_trunk: np.ndarray,  # (C,)
    w_out_t: np.ndarray,  # (C, O)
    b_out: np.ndarray,  # (O,)
) -> np.ndarray:
    w = win.shape[0]
    c = b_in.shape[0]
    n_conv = conv_b.shape[0]
    ksz = conv_w.shape[1] // c

    x = np.dot(win, w_in_t)
    for i in range(w):
        for j in range(c):
            x[i, j] += b_in[j]

    # Receptive-cone trim: only x[w-1] is consumed downstream, so conv ``idx``
    # only needs output rows >= (ksz-1)*(idx+1) (and the block-``ci`` residual
    # add rows >= 2*(ksz-1)*(ci+1)). Each computed row is the identical
    # arithmetic on the identical inputs -- rows below the cone are simply
    # never read. ~2.3x fewer conv rows for the 6-conv/k=5 encoder.
    col = np.zeros((w, ksz * c), dtype=np.float32)
    y = np.zeros((w, c), dtype=np.float32)
    for ci in range(n_conv // 2):
        for cj in range(2):
            idx = ci * 2 + cj
            lo = (ksz - 1) * (idx + 1)  # first output row inside the cone
            # left-pad (ksz-1) zeros, im2col, gemm (conv1 reads x, conv2 reads y;
            # y's rows [lo-ksz+1, w) are fresh from conv1 before conv2 overwrites)
            if cj == 0:
                for i in range(lo, w):
                    for kk in range(ksz):
                        src = i + kk - (ksz - 1)
                        for j in range(c):
                            col[i, kk * c + j] = x[src, j] if src >= 0 else np.float32(0.0)
            else:
                for i in range(lo, w):
                    for kk in range(ksz):
                        src = i + kk - (ksz - 1)
                        for j in range(c):
                            col[i, kk * c + j] = y[src, j] if src >= 0 else np.float32(0.0)
            yv = np.dot(col[lo:], conv_w[idx])
            # bias, then LayerNorm + ReLU per row
            for i in range(lo, w):
                for j in range(c):
                    y[i, j] = yv[i - lo, j] + conv_b[idx, j]
                mu = 0.0
                for j in range(c):
                    mu += np.float64(y[i, j])
                mu /= c
                var = 0.0
                for j in range(c):
                    dv = np.float64(y[i, j]) - mu
                    var += dv * dv
                var /= c
                inv = 1.0 / math.sqrt(var + ln_eps)
                for j in range(c):
                    v = np.float32((np.float64(y[i, j]) - mu) * inv) * ln_w[idx, j] + ln_b[idx, j]
                    y[i, j] = v if v > np.float32(0.0) else np.float32(0.0)
        for i in range(2 * (ksz - 1) * (ci + 1), w):
            for j in range(c):
                x[i, j] = x[i, j] + y[i, j]

    last = x[w - 1]
    s = np.dot(summary, w_sum_t)
    for j in range(c):
        sv = s[j] + b_sum[j]
        s[j] = sv if sv > np.float32(0.0) else np.float32(0.0)
    cat = np.zeros(2 * c, dtype=np.float32)
    for j in range(c):
        cat[j] = last[j]
        cat[c + j] = s[j]
    tr = np.dot(cat, w_trunk_t)
    for j in range(c):
        tv = tr[j] + b_trunk[j]
        tr[j] = tv if tv > np.float32(0.0) else np.float32(0.0)
    out = np.dot(tr, w_out_t)
    for j in range(out.shape[0]):
        out[j] += b_out[j]
    return out

class NumbaPlannerForward:
    """Weight pack + entry point for the fast TCN forward (labelled deviation).

    Slices the prefix tensor to the encoder's exact receptive field (25 ticks
    for 3 blocks of two k=5 causal convs): every consumed position has full
    real support inside the window, so the slice is arithmetically identical
    to the full 160-tick forward. The kernel additionally computes only the
    receptive CONE of the last position (rows a conv output at w-1 transitively
    depends on) -- dropped rows are never read, so this is selection, not
    approximation. The only deviation is gemm/libm rounding (measured ~3e-7
    max on y_all vs the torch forward).
    """

    def __init__(self, model):
        enc = model.encoder
        convs = []
        for block in enc.blocks:
            convs.append((block.conv1, block.norm1))
            convs.append((block.conv2, block.norm2))
        ksz = int(convs[0][0].weight.shape[2])
        c = int(convs[0][0].weight.shape[0])
        self.window = 1 + sum(int(cw.weight.shape[2]) - 1 for cw, _ in convs)
        self.hidden = c

        def f32(t):
            return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))

        self.w_in_t = np.ascontiguousarray(f32(enc.input.weight)[:, :, 0].T)  # (3, C)
        self.b_in = f32(enc.input.bias)
        conv_w = np.zeros((len(convs), ksz * c, c), dtype=np.float32)
        conv_b = np.zeros((len(convs), c), dtype=np.float32)
        ln_w = np.zeros((len(convs), c), dtype=np.float32)
        ln_b = np.zeros((len(convs), c), dtype=np.float32)
        eps = None
        for i, (cw, ln) in enumerate(convs):
            wgt = f32(cw.weight)  # (out C, in C, k)
            # im2col layout: col[i, kk*C + ci] pairs with conv_w[kk*C + ci, co]
            conv_w[i] = wgt.transpose(2, 1, 0).reshape(ksz * c, c)
            conv_b[i] = f32(cw.bias)
            ln_w[i] = f32(ln.weight)
            ln_b[i] = f32(ln.bias)
            eps = float(ln.eps)
        self.conv_w = np.ascontiguousarray(conv_w)
        self.conv_b = conv_b
        self.ln_w = ln_w
        self.ln_b = ln_b
        self.ln_eps = np.float64(eps)
        self.w_sum_t = np.ascontiguousarray(f32(enc.summary[0].weight).T)
        self.b_sum = f32(enc.summary[0].bias)
        self.w_trunk_t = np.ascontiguousarray(f32(model.trunk[0].weight).T)
        self.b_trunk = f32(model.trunk[0].bias)
        self.w_out_t = np.ascontiguousarray(f32(model.out.weight).T)
        self.b_out = f32(model.out.bias)
        self.heads = int(model.heads)
        self.out_dim = int(model.out_dim)

    def forward(self, prefix_tensor: np.ndarray, summary: np.ndarray) -> np.ndarray:
        """(1, 160, 3) normalized prefix tensor + (1, S) summary -> (1, heads, out_dim)."""

        win = np.ascontiguousarray(prefix_tensor[0, -self.window :, :].astype(np.float32))
        y = _tcn_forward(
            win,
            np.ascontiguousarray(summary[0].astype(np.float32)),
            self.w_in_t,
            self.b_in,
            self.conv_w,
            self.conv_b,
            self.ln_w,
            self.ln_b,
            self.ln_eps,
            self.w_sum_t,
            self.b_sum,
            self.w_trunk_t,
            self.b_trunk,
            self.w_out_t,
            self.b_out,
        )
        return y.reshape(1, self.heads, self.out_dim)


__all__ = [
    "NumbaPlannerForward",
    "_begin_precompute",
    "_gru_gates",
    "_gru_walk",
    "_matvec",
    "_regime_features_nb",
    "_smooth_w5",
    "_tcn_forward",
    "_warm_features",
]
