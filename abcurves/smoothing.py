"""Offline smoothing methods for V2 human-continuation decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmoothingSpec:
    method: str = "triangular_moving_average_path"
    window: int = 9
    alpha: float = 0.25
    preserve_endpoint: bool = False
    endpoint_bridge_ms: int = 32

    @property
    def key(self) -> str:
        if self.method in {"centered_moving_average_path", "triangular_moving_average_path"}:
            base = f"{self.method}_w{int(self.window)}"
        elif self.method == "forward_backward_ema_delta":
            base = f"{self.method}_a{float(self.alpha):.3g}".replace(".", "p")
        else:
            base = self.method
        if self.preserve_endpoint:
            # This correction applies to the exact input passed to
            # ``smooth_dxdy``. Callers that later slice the result must not
            # mistake it for preservation of an arbitrary sub-interval.
            base += f"_input_epbridge{int(self.endpoint_bridge_ms)}"
        return base


DEFAULT_SPECS: tuple[SmoothingSpec, ...] = (
    SmoothingSpec("raw_identity"),
    SmoothingSpec("triangular_moving_average_path", window=5),
    SmoothingSpec("centered_moving_average_path", window=9),
    SmoothingSpec("triangular_moving_average_path", window=9),
    SmoothingSpec("triangular_moving_average_path", window=15),
    SmoothingSpec("forward_backward_ema_delta", alpha=0.25),
)


def parse_smoothing_spec(value: str | SmoothingSpec | None) -> SmoothingSpec:
    if isinstance(value, SmoothingSpec):
        return value
    if not value:
        return SmoothingSpec()
    parts = str(value).split(":")
    method = parts[0]
    kwargs: dict[str, float | int | bool] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip().replace("-", "_")
        raw = raw.strip()
        if key == "window":
            kwargs[key] = int(raw)
        elif key == "alpha":
            kwargs[key] = float(raw)
        elif key == "preserve_endpoint":
            normalized = raw.lower()
            if normalized not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(f"invalid preserve_endpoint value: {raw!r}")
            kwargs[key] = normalized in {"true", "1", "yes"}
        elif key == "endpoint_bridge_ms":
            kwargs[key] = int(raw)
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
    if parsed.method == "raw_identity":
        smoothed = valid
    elif parsed.method == "centered_moving_average_path":
        smoothed = _path_to_delta(_centered_moving_average(_delta_to_path(valid), parsed.window))
    elif parsed.method == "triangular_moving_average_path":
        path = _delta_to_path(valid)
        once = _centered_moving_average(path, parsed.window)
        smoothed = _path_to_delta(_centered_moving_average(once, parsed.window))
    elif parsed.method == "forward_backward_ema_delta":
        smoothed = _forward_backward_ema(valid, parsed.alpha)
    else:
        raise ValueError(f"unknown smoothing method: {parsed.method}")
    if parsed.preserve_endpoint:
        smoothed = _preserve_endpoint(
            valid,
            smoothed,
            bridge_ms=parsed.endpoint_bridge_ms,
        )
    out[:n] = smoothed.astype(np.float32)
    return out


def decompose_dxdy(
    dxdy: np.ndarray,
    spec: str | SmoothingSpec | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = smooth_dxdy(dxdy, spec=spec, mask=mask)
    raw = np.asarray(dxdy, dtype=np.float32)
    residual = raw - smooth
    reconstructed = smooth + residual
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        residual[~valid] = 0.0
        reconstructed[~valid] = 0.0
    return smooth, residual, reconstructed


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


def _forward_backward_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    if len(values) <= 1 or alpha >= 1.0:
        return values.copy()
    forward = np.zeros_like(values, dtype=np.float64)
    forward[0] = values[0]
    for idx in range(1, len(values)):
        forward[idx] = alpha * values[idx] + (1.0 - alpha) * forward[idx - 1]
    backward = np.zeros_like(values, dtype=np.float64)
    backward[-1] = forward[-1]
    for idx in range(len(values) - 2, -1, -1):
        backward[idx] = alpha * forward[idx] + (1.0 - alpha) * backward[idx + 1]
    return backward


def _preserve_endpoint(
    raw_dxdy: np.ndarray,
    smooth_dxdy: np.ndarray,
    *,
    bridge_ms: int,
) -> np.ndarray:
    """Backward-compatible input-scoped wrapper for endpoint debt repayment."""

    return preserve_endpoint_bridge(
        raw_dxdy,
        smooth_dxdy,
        bridge_ms=bridge_ms,
    )


def preserve_endpoint_bridge(
    raw_dxdy: np.ndarray,
    smooth_dxdy: np.ndarray,
    *,
    bridge_ms: int | None,
) -> np.ndarray:
    """Repay endpoint debt with a minimum-jerk position bridge.

    ``bridge_ms=None`` distributes the correction across the complete supplied
    interval. A positive integer confines it to that many terminal ticks. The
    operation is deliberately separate from :func:`smooth_dxdy` so a caller
    can preserve a semantically meaningful slice (for example B->C) after a
    context-aware joint A->C smoothing pass.
    """

    raw = np.asarray(raw_dxdy, dtype=np.float64)
    smooth = np.asarray(smooth_dxdy, dtype=np.float64).copy()
    if raw.ndim != 2 or raw.shape[1:] != (2,) or smooth.shape != raw.shape:
        raise ValueError("raw_dxdy and smooth_dxdy must have matching (T, 2) shapes")
    if len(raw) == 0:
        return smooth
    if bridge_ms is None:
        width = len(raw)
    else:
        if int(bridge_ms) < 1:
            raise ValueError("bridge_ms must be positive or None")
        width = min(int(bridge_ms), len(raw))
    debt = raw.sum(axis=0) - smooth.sum(axis=0)
    if not np.any(np.abs(debt) > 0.0):
        return smooth
    t = np.linspace(0.0, 1.0, width + 1, dtype=np.float64)
    position = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    increments = np.diff(position)
    smooth[-width:] += increments[:, None] * debt[None, :]
    return smooth


__all__ = [
    "SmoothingSpec",
    "DEFAULT_SPECS",
    "parse_smoothing_spec",
    "smooth_dxdy",
    "decompose_dxdy",
    "preserve_endpoint_bridge",
]
