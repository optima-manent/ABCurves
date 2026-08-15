"""Python binding for the frozen global H80 Renderer C runtime.

The learned model is a 44,484-byte, zero-copy binary image.  The native core
owns no heap memory: Python allocates one opaque model view and one
target-sized renderer state (5,088 bytes on the validated Windows x64 ABI).
Exactly 256 genuine reports prepare a reusable profile before B; each event
copies that state, begins once, and emits one bounded integer report per
smooth-intent tick.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np


CONTEXT_TICKS = 256
ARTIFACT_BYTES = 44_484
ARTIFACT_SHA256 = "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b"
LATERAL_OFFSET_PENALTY = 1.5
WINDOWS_NATIVE_BYTES = 37_376
WINDOWS_NATIVE_SHA256 = "730bafac2b6a8431628401fc6e2239a12b787cd730791d453b4dec0cb47d3a41"


class RendererRuntimeError(RuntimeError):
    """Raised when the native Renderer or one of its inputs breaks contract."""


class _Report(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_int16), ("dy", ctypes.c_int16)]


_STATUS = {
    -1: "invalid argument",
    -2: "model image is incompatible",
    -3: "model CRC differs",
    -4: "model source identity differs",
    -5: "call is invalid in the current mode",
    -6: "prefix is empty",
    -7: "value is outside the supported range",
    -8: "I/O failure",
}


def _check(status: int, operation: str) -> None:
    if int(status) != 0:
        detail = _STATUS.get(int(status), f"native status {int(status)}")
        raise RendererRuntimeError(f"Renderer {operation} failed: {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_library_path() -> Path:
    """Return the bundled native library path for this operating system."""

    override = os.environ.get("ABCURVES_RENDERER_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    names = {
        "Windows": "abcurves_renderer.dll",
        "Linux": "libabcurves_renderer.so",
        "Darwin": "libabcurves_renderer.dylib",
    }
    name = names.get(platform.system())
    if name is None:
        raise RendererRuntimeError(
            f"unsupported host {platform.system()!r}; build runtime/c for this target"
        )
    bundled = Path(__file__).resolve().parent / "_native" / name
    if bundled.is_file():
        return bundled
    source_build = Path(__file__).resolve().parents[1] / "runtime" / "c" / "build"
    candidates = (
        source_build / "Release" / name,
        source_build / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RendererRuntimeError(
        f"native Renderer library is missing ({bundled}). Build it with "
        "`cmake -S runtime/c -B runtime/c/build` followed by "
        "`cmake --build runtime/c/build --config Release`, or set "
        "ABCURVES_RENDERER_LIBRARY."
    )


class _NativeAPI:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise RendererRuntimeError(f"native Renderer library is missing: {self.path}")
        self.library = ctypes.CDLL(str(self.path))
        lib = self.library
        lib.abc_online_model_size.argtypes = []
        lib.abc_online_model_size.restype = ctypes.c_size_t
        lib.abc_online_renderer_size.argtypes = []
        lib.abc_online_renderer_size.restype = ctypes.c_size_t
        lib.abc_online_report_size.argtypes = []
        lib.abc_online_report_size.restype = ctypes.c_size_t
        lib.abc_online_model_init.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.abc_online_model_init.restype = ctypes.c_int
        lib.abc_online_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.abc_online_reset.restype = ctypes.c_int
        lib.abc_online_observe_raw.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int16,
            ctypes.c_int16,
        ]
        lib.abc_online_observe_raw.restype = ctypes.c_int
        lib.abc_online_begin.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.abc_online_begin.restype = ctypes.c_int
        lib.abc_online_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(_Report),
        ]
        lib.abc_online_step.restype = ctypes.c_int
        if int(lib.abc_online_report_size()) != ctypes.sizeof(_Report):
            raise RendererRuntimeError("native report ABI differs from the Python binding")

    @property
    def model_bytes(self) -> int:
        return int(self.library.abc_online_model_size())

    @property
    def renderer_bytes(self) -> int:
        return int(self.library.abc_online_renderer_size())


def _void_pointer(storage: Any) -> ctypes.c_void_p:
    return ctypes.cast(storage, ctypes.c_void_p)


def validate_physical_context(value: np.ndarray) -> np.ndarray:
    """Return one owned, canonical 256-report physical sample."""

    array = np.asarray(value)
    if array.shape != (CONTEXT_TICKS, 2):
        raise RendererRuntimeError(
            f"Renderer context must be exactly ({CONTEXT_TICKS}, 2) physical reports"
        )
    if not np.issubdtype(array.dtype, np.number) or not bool(np.all(np.isfinite(array))):
        raise RendererRuntimeError("Renderer context must contain finite numeric reports")
    rounded = np.rint(array.astype(np.float64, copy=False))
    if not bool(np.array_equal(array, rounded)):
        raise RendererRuntimeError("Renderer context must contain integer hardware counts")
    if bool(np.any(rounded < -32768.0)) or bool(np.any(rounded > 32767.0)):
        raise RendererRuntimeError("Renderer context exceeds signed int16 count range")
    return np.ascontiguousarray(rounded, dtype=np.int16)


def validate_smooth_future(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the contiguous active prefix of one finite smooth intent."""

    smooth = np.asarray(value, dtype=np.float32)
    valid = np.asarray(mask).reshape(-1)
    if smooth.ndim != 2 or smooth.shape[1] != 2 or len(smooth) != len(valid):
        raise RendererRuntimeError("smooth intent and mask must have shapes (H, 2) and (H,)")
    if not bool(np.all(np.isfinite(smooth))) or not bool(np.all(np.isfinite(valid))):
        raise RendererRuntimeError("smooth intent and mask must be finite")
    active = valid > 0.5
    duration = int(np.sum(active))
    if duration < 1:
        raise RendererRuntimeError("smooth intent must contain at least one active tick")
    if not bool(np.all(active[:duration])) or bool(np.any(active[duration:])):
        raise RendererRuntimeError("future mask must be one contiguous prefix of active ticks")
    return np.ascontiguousarray(smooth[:duration])


def _q16_pair(value: np.ndarray) -> tuple[int, int]:
    pair = np.asarray(value, dtype=np.float32).reshape(-1)
    if pair.shape != (2,) or not bool(np.all(np.isfinite(pair))):
        raise RendererRuntimeError("smooth intent tick must contain two finite values")
    scaled = np.rint(pair.astype(np.float64) * 65536.0)
    if bool(np.any(scaled < -(2**31))) or bool(np.any(scaled > 2**31 - 1)):
        raise RendererRuntimeError("smooth intent tick exceeds signed Q16 range")
    return int(scaled[0]), int(scaled[1])


@dataclass(frozen=True)
class RendererReceipt:
    artifact: str
    artifact_bytes: int
    artifact_sha256: str
    native_library: str
    native_library_sha256: str
    context_ticks: int
    state_bytes: int
    model_view_bytes: int
    lateral_offset_penalty: float


class PortableRendererModel:
    """Authenticated, shared model view used to create independent streams."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        library: str | Path | None = None,
        verify: bool = True,
    ) -> None:
        self.artifact = Path(artifact).expanduser().resolve()
        if not self.artifact.is_file():
            raise RendererRuntimeError(f"Renderer artifact is missing: {self.artifact}")
        if self.artifact.stat().st_size != ARTIFACT_BYTES:
            raise RendererRuntimeError(
                f"Renderer artifact has {self.artifact.stat().st_size} bytes; "
                f"expected {ARTIFACT_BYTES}"
            )
        if verify and _sha256(self.artifact) != ARTIFACT_SHA256:
            raise RendererRuntimeError("Renderer artifact SHA-256 differs")
        library_path = default_library_path() if library is None else Path(library)
        bundled_windows = (
            Path(__file__).resolve().parent / "_native" / "abcurves_renderer.dll"
        )
        using_bundled_windows = (
            library is None
            and not os.environ.get("ABCURVES_RENDERER_LIBRARY")
            and platform.system() == "Windows"
            and library_path.resolve() == bundled_windows.resolve()
        )
        if verify and using_bundled_windows:
            if library_path.stat().st_size != WINDOWS_NATIVE_BYTES:
                raise RendererRuntimeError("bundled native Renderer size differs")
            if _sha256(library_path) != WINDOWS_NATIVE_SHA256:
                raise RendererRuntimeError("bundled native Renderer SHA-256 differs")
        self.api = _NativeAPI(library_path)
        blob = self.artifact.read_bytes()
        self._blob = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        self._model = (ctypes.c_ubyte * self.api.model_bytes)()
        _check(
            self.api.library.abc_online_model_init(
                _void_pointer(self._model), _void_pointer(self._blob), len(blob)
            ),
            "model initialization",
        )
        self.receipt = RendererReceipt(
            artifact=self.artifact.name,
            artifact_bytes=ARTIFACT_BYTES,
            artifact_sha256=ARTIFACT_SHA256,
            native_library=self.api.path.name,
            native_library_sha256=_sha256(self.api.path),
            context_ticks=CONTEXT_TICKS,
            state_bytes=self.api.renderer_bytes,
            model_view_bytes=self.api.model_bytes,
            lateral_offset_penalty=LATERAL_OFFSET_PENALTY,
        )

    def prepare_context(self, raw_dxdy: np.ndarray) -> "PreparedRendererContext":
        context = validate_physical_context(raw_dxdy)
        state = (ctypes.c_ubyte * self.api.renderer_bytes)()
        _check(
            self.api.library.abc_online_reset(
                _void_pointer(state), _void_pointer(self._model)
            ),
            "reset",
        )
        for dx, dy in context:
            _check(
                self.api.library.abc_online_observe_raw(
                    _void_pointer(state), int(dx), int(dy)
                ),
                "observation",
            )
        return PreparedRendererContext(self, state)


class PreparedRendererContext:
    """Reusable snapshot of exactly 256 observed physical reports.

    The stored native state is immutable.  Every event begins from a private
    copy, so one prepared profile can be reused safely without replaying its
    256 reports on the B-critical path.
    """

    def __init__(self, model: PortableRendererModel, storage: Any) -> None:
        self._model = model
        self._storage = storage

    def begin(
        self,
        smooth_dxdy: np.ndarray,
        future_mask: np.ndarray,
        *,
        event_seed: int,
    ) -> "PortableRendererEvent":
        if type(event_seed) is not int or not 0 <= event_seed < 2**64:
            raise RendererRuntimeError("event_seed must be a Python integer in uint64 range")
        smooth = validate_smooth_future(smooth_dxdy, future_mask)
        event_storage = (ctypes.c_ubyte * self._model.api.renderer_bytes)()
        ctypes.memmove(
            _void_pointer(event_storage),
            _void_pointer(self._storage),
            self._model.api.renderer_bytes,
        )
        _check(
            self._model.api.library.abc_online_begin(
                _void_pointer(event_storage), event_seed
            ),
            "begin",
        )
        return PortableRendererEvent(self._model, event_storage, smooth)


class PortableRendererEvent:
    """State-owning one-report-per-call Renderer stream."""

    def __init__(self, model: PortableRendererModel, storage: Any, smooth: np.ndarray) -> None:
        self._model = model
        self._storage = storage
        self._smooth = smooth
        self._tick = 0

    @property
    def duration(self) -> int:
        return len(self._smooth)

    @property
    def complete(self) -> bool:
        return self._tick >= self.duration

    def step(self) -> np.ndarray:
        if self.complete:
            raise RendererRuntimeError("Renderer event is complete")
        x, y = _q16_pair(self._smooth[self._tick])
        report = _Report()
        _check(
            self._model.api.library.abc_online_step(
                _void_pointer(self._storage), x, y, ctypes.byref(report)
            ),
            "step",
        )
        self._tick += 1
        return np.asarray([report.dx, report.dy], dtype=np.int16)

    def render_remaining(self) -> np.ndarray:
        output = np.empty((self.duration - self._tick, 2), dtype=np.int16)
        for index in range(len(output)):
            output[index] = self.step()
        return output


__all__ = [
    "ARTIFACT_BYTES",
    "ARTIFACT_SHA256",
    "CONTEXT_TICKS",
    "LATERAL_OFFSET_PENALTY",
    "PortableRendererEvent",
    "PortableRendererModel",
    "PreparedRendererContext",
    "RendererReceipt",
    "RendererRuntimeError",
    "WINDOWS_NATIVE_BYTES",
    "WINDOWS_NATIVE_SHA256",
    "default_library_path",
    "validate_physical_context",
    "validate_smooth_future",
]
