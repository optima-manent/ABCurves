"""Numba-compiled selected-head forward pass for the frozen Planner."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


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


__all__ = ["NumbaPlannerForward", "_tcn_forward"]

