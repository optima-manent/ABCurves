"""The triangular path smoother used by the final Planner and Renderer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmoothingSpec:
    method: str = "triangular_moving_average_path"
    window: int = 5

    def __post_init__(self) -> None:
        if self.method != "triangular_moving_average_path":
            raise ValueError(
                "the final recipe supports only triangular_moving_average_path"
            )
        if int(self.window) < 1:
            raise ValueError("smoothing window must be positive")

def parse_smoothing_spec(value: str | SmoothingSpec | None) -> SmoothingSpec:
    if isinstance(value, SmoothingSpec):
        return value
    if not value:
        return SmoothingSpec()
    parts = str(value).split(":")
    method = parts[0]
    kwargs: dict[str, int] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid smoothing option: {part!r}")
        key, raw = part.split("=", 1)
        key = key.strip().replace("-", "_")
        raw = raw.strip()
        if key == "window":
            kwargs[key] = int(raw)
        else:
            raise ValueError(f"unknown smoothing option: {key!r}")
    return SmoothingSpec(method=method, **kwargs)


def smooth_dxdy(
    dxdy: np.ndarray,
    spec: str | SmoothingSpec | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    parsed = parse_smoothing_spec(spec)
    arr = np.asarray(dxdy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("dxdy must have shape (T, 2)")
    n = _valid_len(arr, mask)
    out = np.zeros_like(arr, dtype=np.float32)
    if n <= 0:
        return out
    valid = arr[:n].astype(np.float64)
    path = _delta_to_path(valid)
    once = _centered_moving_average(path, parsed.window)
    smoothed = _path_to_delta(_centered_moving_average(once, parsed.window))
    out[:n] = smoothed.astype(np.float32)
    return out


def _valid_len(arr: np.ndarray, mask: np.ndarray | None) -> int:
    if mask is None:
        return int(len(arr))
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 1 or len(mask_arr) != len(arr):
        raise ValueError("mask must have shape (T,)")
    return int(np.sum(mask_arr))


def _delta_to_path(dxdy: np.ndarray) -> np.ndarray:
    return np.cumsum(dxdy, axis=0)


def _path_to_delta(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return path.copy()
    previous = np.vstack([np.zeros((1, path.shape[1]), dtype=path.dtype), path[:-1]])
    return path - previous


def _centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    if window <= 1 or len(values) <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.zeros_like(values, dtype=np.float64)
    for axis in range(values.shape[1]):
        out[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return out


__all__ = [
    "SmoothingSpec",
    "parse_smoothing_spec",
    "smooth_dxdy",
]
