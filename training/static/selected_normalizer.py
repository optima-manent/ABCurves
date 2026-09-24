"""Exact selected two-pass source-weighted normalizer, independent of audit DBs."""
from __future__ import annotations

import numpy as np
from training.train_planner import weighted_percentile

CHUNK = 8192


def weighted_stats(array, weights):
    total = float(np.sum(weights, dtype=np.float64))
    shape = tuple(array.shape[1:])
    mean = np.zeros(shape, dtype=np.float64)
    for start in range(0, len(weights), CHUNK):
        x = np.asarray(array[start:start+CHUNK], dtype=np.float64)
        w = weights[start:start+CHUNK].reshape((-1,) + (1,) * len(shape))
        mean += np.sum(x * w, axis=0)
    mean /= total
    variance = np.zeros(shape, dtype=np.float64)
    for start in range(0, len(weights), CHUNK):
        x = np.asarray(array[start:start+CHUNK], dtype=np.float64)
        w = weights[start:start+CHUNK].reshape((-1,) + (1,) * len(shape))
        centered = x - mean
        variance += np.sum(centered * centered * w, axis=0)
    variance /= total
    return mean, np.sqrt(np.maximum(variance, 0.0))


def prefix_stats(prefix, mask, weights):
    total = 0.0
    numerator = np.zeros(2, dtype=np.float64)
    for start in range(0, len(weights), CHUNK):
        x = np.asarray(prefix[start:start+CHUNK], dtype=np.float64)
        m = np.asarray(mask[start:start+CHUNK], dtype=bool)
        w = weights[start:start+CHUNK, None] * m
        total += float(np.sum(w))
        numerator += np.sum(x * w[..., None], axis=(0, 1))
    mean = numerator / total
    variance = np.zeros(2, dtype=np.float64)
    for start in range(0, len(weights), CHUNK):
        x = np.asarray(prefix[start:start+CHUNK], dtype=np.float64)
        m = np.asarray(mask[start:start+CHUNK], dtype=bool)
        w = weights[start:start+CHUNK, None] * m
        variance += np.sum((x - mean) ** 2 * w[..., None], axis=(0, 1))
    return mean.astype(np.float32), np.sqrt(np.maximum(variance / total, 0.0)).astype(np.float32)


def turn_rates(paths, tau):
    rates = np.zeros(len(tau), dtype=np.float64)
    for start in range(0, len(tau), CHUNK):
        stop = min(start+CHUNK, len(tau))
        position = np.asarray(paths[start:stop], dtype=np.float64)
        steps = np.diff(position, axis=1, prepend=np.zeros_like(position[:, :1]))
        first, second = steps[:, :-1], steps[:, 1:]
        n1, n2 = np.linalg.norm(first, axis=-1), np.linalg.norm(second, axis=-1)
        cosine = np.sum(first * second, axis=-1) / np.maximum(n1*n2, 1e-9)
        mean_step = np.linalg.norm(steps, axis=-1).mean(axis=-1, keepdims=True)
        gate = np.minimum(1.0, np.minimum(n1,n2) / np.maximum(0.25*mean_step,1e-9))
        turn = ((1.0-np.clip(cosine,-1.0,1.0))*gate).sum(axis=-1)
        rates[start:stop] = turn / np.maximum(np.asarray(tau[start:stop]),1.0)*100.0
    return rates


def fit(arrays):
    weights = np.asarray(arrays['event_weight'], dtype=np.float64)
    sm, ss = weighted_stats(arrays['summary_features'], weights)
    ym, ys = weighted_stats(arrays['y_raw'], weights)
    pm, ps = prefix_stats(arrays['prefix_raw_dxdy'], arrays['prefix_mask'], weights)
    for std in (ss,ys,ps): std[std < 1e-6] = 1.0
    tau = np.asarray(arrays['tau'], dtype=np.float64)
    rates = turn_rates(arrays['grid_path'],tau)
    thresholds = []
    for lower,upper in [(0.0,150.0),(150.0,250.0),(250.0,float('inf'))]:
        keep = (tau>=lower)&(tau<upper)
        thresholds.append(weighted_percentile(rates[keep],weights[keep],95.0))
    return {'prefix_mean':pm.tolist(),'prefix_std':ps.tolist(),
        'summary_mean':sm.tolist(),'summary_std':ss.tolist(),
        'y_mean':ym.tolist(),'y_std':ys.tolist(),'turn_hinge_thresholds':thresholds}
