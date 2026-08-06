"""Interpretable descriptors of raw 1 kHz mouse-count texture."""

from __future__ import annotations

import math

import numpy as np


TEXTURE_FEATURE_NAMES = (
    "zero_rate",
    "nonzero_mag_mean_log1p",
    "nonzero_mag_max_log1p",
    "sign_flip_rate",
    "zero_run_mean_log1p",
    "active_run_mean_log1p",
    "large_packet_rate",
    "bp_10_30",
    "bp_30_80",
    "bp_80_200",
    "bp_200_500",
    "hf_power_log1p",
    "nonzero_mag_p90_log1p",
    "nonzero_mag_p99_log1p",
    "iri_std_log1p",
    "iri_p95_log1p",
    "mag_ac_lag1",
    "mag_ac_lag5",
    "mag_entropy",
)

_SPECTRAL_BANDS = (
    (10.0, 30.0),
    (30.0, 80.0),
    (80.0, 200.0),
    (200.0, 500.0),
)


def _run_lengths(active: np.ndarray) -> tuple[list[int], list[int]]:
    zero_runs: list[int] = []
    active_runs: list[int] = []
    if len(active) == 0:
        return zero_runs, active_runs
    current = bool(active[0])
    length = 1
    for value in active[1:]:
        if bool(value) == current:
            length += 1
        else:
            (active_runs if current else zero_runs).append(length)
            current = bool(value)
            length = 1
    (active_runs if current else zero_runs).append(length)
    return zero_runs, active_runs


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) < lag + 4:
        return 0.0
    left = values[:-lag] - np.mean(values[:-lag])
    right = values[lag:] - np.mean(values[lag:])
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator >= 1e-9 else 0.0


def texture_features(dxdy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the frozen 19-feature count-texture panel for each event.

    ``dxdy`` has shape ``(N, H, 2)`` and ``mask`` has shape ``(N, H)``.
    Frequency bands assume the project's 1 kHz hardware-count clock.
    """

    streams = np.asarray(dxdy)
    valid = np.asarray(mask)
    if streams.ndim != 3 or streams.shape[2] != 2:
        raise ValueError("dxdy must have shape (N, H, 2)")
    if valid.shape != streams.shape[:2]:
        raise ValueError("mask must have shape (N, H)")
    output = np.zeros((len(valid), len(TEXTURE_FEATURE_NAMES)), dtype=np.float64)
    for row_index in range(len(valid)):
        stream = streams[row_index][valid[row_index] > 0.5].astype(np.float64)
        if len(stream) == 0:
            continue
        packet_magnitude = np.max(np.abs(stream), axis=1)
        active = packet_magnitude > 0.0
        active_magnitude = packet_magnitude[active]
        flip_rates: list[float] = []
        for axis in range(2):
            signs = np.sign(stream[:, axis])
            signs = signs[signs != 0]
            flip_rates.append(
                float(np.mean(signs[1:] * signs[:-1] < 0))
                if len(signs) > 1
                else 0.0
            )
        zero_runs, active_runs = _run_lengths(active)
        row = [
            float(1.0 - np.mean(active)),
            math.log1p(float(np.mean(active_magnitude))) if len(active_magnitude) else 0.0,
            math.log1p(float(np.max(active_magnitude))) if len(active_magnitude) else 0.0,
            float(np.mean(flip_rates)),
            math.log1p(float(np.mean(zero_runs)) if zero_runs else 0.0),
            math.log1p(float(np.mean(active_runs)) if active_runs else 0.0),
            float(np.mean(packet_magnitude >= 3.0)),
        ]
        magnitude = np.linalg.norm(stream, axis=1)
        ticks = len(stream)
        if ticks >= 8:
            centered = magnitude - np.mean(magnitude)
            power = np.abs(np.fft.rfft(centered)) ** 2
            frequencies = np.fft.rfftfreq(ticks, d=1e-3)
            total = float(np.sum(power[1:]))
            bands = [
                float(np.sum(power[(frequencies >= low) & (frequencies < high)]))
                / max(total, 1e-9)
                for low, high in _SPECTRAL_BANDS
            ]
            high_frequency = float(
                np.sum(power[(frequencies >= 30.0) & (frequencies < 500.0)])
            ) / float(ticks)
            active_norm = magnitude[magnitude > 0.0]
            p90 = (
                math.log1p(float(np.percentile(active_norm, 90)))
                if len(active_norm)
                else 0.0
            )
            p99 = (
                math.log1p(float(np.percentile(active_norm, 99)))
                if len(active_norm)
                else 0.0
            )
            inter_report_intervals, _ = _run_lengths(magnitude > 0.0)
            iri_std = (
                math.log1p(float(np.std(inter_report_intervals)))
                if len(inter_report_intervals) >= 2
                else 0.0
            )
            iri_p95 = (
                math.log1p(float(np.percentile(inter_report_intervals, 95)))
                if inter_report_intervals
                else 0.0
            )
            histogram = np.bincount(
                np.clip(np.rint(magnitude).astype(int), 0, 10), minlength=11
            ).astype(np.float64)
            probabilities = histogram / max(float(np.sum(histogram)), 1.0)
            nonzero = probabilities[probabilities > 0]
            entropy = float(
                -np.sum(nonzero * np.log(nonzero)) / np.log(len(probabilities))
            )
            row += [
                *bands,
                math.log1p(high_frequency),
                p90,
                p99,
                iri_std,
                iri_p95,
                _autocorrelation(magnitude, 1),
                _autocorrelation(magnitude, 5),
                entropy,
            ]
        else:
            row += [0.0] * 12
        output[row_index] = row
    output[~np.isfinite(output)] = 0.0
    return output


def wasserstein1_table(
    left: np.ndarray,
    right: np.ndarray,
    *,
    scale: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-feature empirical W1 gaps on a shared 199-quantile grid."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != len(TEXTURE_FEATURE_NAMES) or second.shape[1] != len(TEXTURE_FEATURE_NAMES):
        raise ValueError("feature matrices must both have shape (N, 19)")
    normalizer = (
        np.ones(len(TEXTURE_FEATURE_NAMES), dtype=np.float64)
        if scale is None
        else np.asarray(scale, dtype=np.float64)
    )
    if normalizer.shape != (len(TEXTURE_FEATURE_NAMES),) or np.any(normalizer <= 0):
        raise ValueError("scale must contain 19 positive values")
    grid = np.linspace(0.0, 1.0, 201)[1:-1]
    return {
        name: float(
            np.mean(
                np.abs(
                    np.quantile(first[:, column] / normalizer[column], grid)
                    - np.quantile(second[:, column] / normalizer[column], grid)
                )
            )
        )
        for column, name in enumerate(TEXTURE_FEATURE_NAMES)
    }


__all__ = ["TEXTURE_FEATURE_NAMES", "texture_features", "wasserstein1_table"]
