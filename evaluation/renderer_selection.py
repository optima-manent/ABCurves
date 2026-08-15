"""Carried-full-session selection score for the global Renderer.

This module is deliberately narrower than the detection study.  It answers the
checkpoint-selection question that teacher-forced validation loss cannot:
after one genuine session-start context, does a sampled Renderer preserve raw
count texture and basic mechanics over the rest of the uninterrupted session?

The public input is the hash-bound ``abcurves.full_sessions.v1`` format (or a
validated research export accepted by :mod:`abcurves.global_data`).  Reports
omit source paths and source identity strings.  They contain only ordinal
aliases, hashes, sufficient statistics, and aggregate scores.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import site
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from abcurves.global_data import (
    FullSession,
    FullSessionCollection,
    load_full_sessions,
)
from abcurves.model_store import ModelIntegrityError, resolve_renderer_float
from abcurves.portable_renderer import PortableRendererModel
from abcurves.renderer import CountTextureModel, load_count_model, sample_count_streams
from abcurves.smoothing import smooth_dxdy
from abcurves.texture import (
    TEXTURE_FEATURE_NAMES,
    texture_features,
    wasserstein1_table,
)


REPORT_SCHEMA = "abcurves.renderer_selection.v1"
SELECTOR_CONTRACT = "continuous_v1"
TICK_HZ = 1_000
CONTEXT_TICKS = 256
SEGMENT_TICKS = 512
AF15_LATERAL_PENALTY = 1.5
ALLOWED_SPECS = {
    "w3": "triangular_moving_average_path:window=3",
    "w5": "triangular_moving_average_path:window=5",
}
_FLOAT_SCHEMA = "abcurves.renderer_global_float.v2"


class RendererSelectionError(RuntimeError):
    """Raised when an input or score cannot satisfy the selection contract."""


class SelectionRenderer(Protocol):
    """Small backend surface used by the scorer and its tests."""

    kind: str
    identity: Mapping[str, Any]

    def render(
        self,
        context: np.ndarray,
        smooth_future: np.ndarray,
        *,
        spec: str,
        event_seed: int,
    ) -> np.ndarray: ...

    def gate_eligibility(
        self, smooth_future: np.ndarray, generated: np.ndarray
    ) -> np.ndarray: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_session_seed(seed: int, domain: str, session_id: str) -> int:
    """Return the frozen little-endian SHA-256-derived uint64 stream seed."""

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise RendererSelectionError("seed must be a plain integer in [0, 2^63)")
    if not domain or not session_id:
        raise RendererSelectionError("seed domain and session_id must be non-empty")
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(session_id.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def default_renderer_model(backend: str) -> Path:
    """Locate a shipped Renderer artifact in a clone or installed wheel."""

    if backend == "float":
        try:
            return resolve_renderer_float(verify=True)
        except (ModelIntegrityError, OSError) as error:
            raise RendererSelectionError(
                f"cannot authenticate the shipped float Renderer: {error}"
            ) from error
    filenames = {
        "native": "renderer_global_h80.bin",
    }
    try:
        filename = filenames[backend]
    except KeyError as error:
        raise RendererSelectionError("backend must be 'native' or 'float'") from error
    relative = Path("models") / filename
    candidates = (
        Path(__file__).resolve().parents[1] / relative,
        Path(sys.prefix) / relative,
        Path(site.getuserbase()) / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _native_gate_eligibility(
    smooth_future: np.ndarray, generated: np.ndarray
) -> np.ndarray:
    """Replay the deployed Q16 quiet/debt gate in native runtime order."""

    smooth = np.ascontiguousarray(np.asarray(smooth_future, dtype=np.float32))
    marks = np.asarray(generated)
    if smooth.ndim != 2 or smooth.shape[1:] != (2,) or marks.shape != smooth.shape:
        raise RendererSelectionError("native gate replay requires aligned [ticks,2] streams")
    scaled = np.rint(smooth.astype(np.float64) * 65_536.0)
    if np.any(scaled < -(2**31)) or np.any(scaled > 2**31 - 1):
        raise RendererSelectionError("smooth intent exceeds signed Q16 range")
    q16 = scaled.astype(np.int64)
    emitted = marks.astype(np.int64, copy=False)
    accumulator = np.zeros(2, dtype=np.int64)
    eligible = np.zeros(len(q16), dtype=bool)
    for tick in range(len(q16)):
        accumulator += q16[tick]
        safety = bool(np.any(np.abs(accumulator) >= 32 * 65_536))
        quiet = bool(np.all(q16[tick] == 0))
        low_debt = bool(np.all(np.abs(accumulator) < 32_768))
        eligible[tick] = quiet and low_debt and not safety
        accumulator -= emitted[tick] * 65_536
        if np.any(accumulator < -(2**31)) or np.any(accumulator > 2**31 - 1):
            raise RendererSelectionError("native accumulator replay exceeded int32")
    return eligible


def _float_gate_eligibility(
    smooth_future: np.ndarray,
    generated: np.ndarray,
    *,
    intent_threshold: float,
    accumulator_threshold: float,
    force_release: float,
) -> np.ndarray:
    """Replay the float sampler's quiet/debt gate in inference order."""

    smooth = np.ascontiguousarray(np.asarray(smooth_future, dtype=np.float32))
    marks = np.asarray(generated, dtype=np.int16)
    if smooth.ndim != 2 or smooth.shape[1:] != (2,) or marks.shape != smooth.shape:
        raise RendererSelectionError("float gate replay requires aligned [ticks,2] streams")
    limits = np.asarray(
        [intent_threshold, accumulator_threshold, force_release], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(limits))
        or intent_threshold < 0.0
        or accumulator_threshold <= 0.0
        or force_release <= 0.0
    ):
        raise RendererSelectionError("float gate thresholds are invalid")
    quiet_limit = np.float32(intent_threshold)
    debt_limit = np.float32(accumulator_threshold)
    safety_limit = np.float32(force_release)
    accumulator = np.zeros(2, dtype=np.float32)
    eligible = np.zeros(len(smooth), dtype=bool)
    for tick in range(len(smooth)):
        accumulator[0] = np.float32(accumulator[0] + smooth[tick, 0])
        accumulator[1] = np.float32(accumulator[1] + smooth[tick, 1])
        maximum_intent = max(abs(smooth[tick, 0]), abs(smooth[tick, 1]))
        maximum_debt = max(abs(accumulator[0]), abs(accumulator[1]))
        safety = maximum_debt >= safety_limit
        eligible[tick] = bool(
            maximum_intent <= quiet_limit
            and maximum_debt < debt_limit
            and not safety
        )
        accumulator[0] = np.float32(
            accumulator[0] - np.float32(marks[tick, 0])
        )
        accumulator[1] = np.float32(
            accumulator[1] - np.float32(marks[tick, 1])
        )
    return eligible


class NativeSelectionRenderer:
    """Exact shipped fixed-online H80 Renderer backend."""

    kind = "native_final"

    def __init__(self, artifact: str | Path, *, library: str | Path | None = None) -> None:
        self.model = PortableRendererModel(artifact, library=library, verify=True)
        receipt = asdict(self.model.receipt)
        self.identity = {
            "backend": self.kind,
            "artifact": "authenticated_global_h80",
            "artifact_bytes": receipt["artifact_bytes"],
            "artifact_sha256": receipt["artifact_sha256"],
            "native_library": "native_renderer_library",
            "native_library_sha256": receipt["native_library_sha256"],
            "state_bytes_on_this_abi": receipt["state_bytes"],
            "model_view_bytes_on_this_abi": receipt["model_view_bytes"],
            "online_handoff": "rank16 adapter in the authenticated artifact",
            "generation_core": "int8/fixed-point",
            "prefix_smoothing_spec": ALLOWED_SPECS["w5"],
        }

    def render(
        self,
        context: np.ndarray,
        smooth_future: np.ndarray,
        *,
        spec: str,
        event_seed: int,
    ) -> np.ndarray:
        del spec  # the already-smoothed intent identifies the requested view
        mask = np.ones(len(smooth_future), dtype=np.float32)
        return self.model.prepare_context(context).begin(
            smooth_future, mask, event_seed=event_seed
        ).render_remaining()

    def gate_eligibility(
        self, smooth_future: np.ndarray, generated: np.ndarray
    ) -> np.ndarray:
        return _native_gate_eligibility(smooth_future, generated)


class FloatSelectionRenderer:
    """Sanitized float source checkpoint backend (no online R16 adapter)."""

    kind = "sanitized_float_source"

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise RendererSelectionError(f"float checkpoint is missing: {path}")
        file_bytes = path.stat().st_size
        file_sha = _sha256_file(path)
        model, report = load_count_model(path)
        if path.stat().st_size != file_bytes or _sha256_file(path) != file_sha:
            raise RendererSelectionError("float checkpoint changed while it was loaded")
        if report.get("schema") != _FLOAT_SCHEMA:
            raise RendererSelectionError(
                f"float selection requires sanitized schema {_FLOAT_SCHEMA!r}"
            )
        tensor_sha = _tensor_sha256(model.state_dict())
        has_sanitization = "sanitization" in report
        sanitization = report.get("sanitization")
        if has_sanitization:
            if not isinstance(sanitization, dict):
                raise RendererSelectionError("float sanitization receipt is malformed")
            recorded = sanitization.get("active_tensor_contract")
            if not isinstance(recorded, dict) or not isinstance(
                recorded.get("sha256"), str
            ):
                raise RendererSelectionError("float checkpoint lacks its tensor receipt")
            if tensor_sha != recorded["sha256"]:
                raise RendererSelectionError("float checkpoint tensor receipt differs")
        else:
            # ``training/train_renderer.py`` intentionally writes the direct
            # training report, not a release-export receipt.  Structural model
            # loading and the deployment-law checks below are the authority;
            # file/tensor hashes make this candidate immutable in the result.
            sanitization = {}
        cfg = model.config
        expected = {
            "context_ticks": CONTEXT_TICKS,
            "recurrent_warm_ticks": 128,
            "offset_radius": 5,
            "base_hysteresis": 0.5,
            "force_release": 32.0,
            "max_abs_count": 127,
            "zero_intent_gate": True,
            "zero_intent_threshold": 1e-7,
            "zero_accumulator_threshold": 0.5,
            "emit_logit_bias": 1.5,
            "temperature": 1.3,
            "offset_magnitude_temperature": 0.75,
            "offset_direction_temperature": 0.15,
        }
        for name, value in expected.items():
            if getattr(cfg, name) != value:
                raise RendererSelectionError(
                    f"float checkpoint deployment setting {name} differs"
                )
        if model.n_features != 20:
            raise RendererSelectionError("float checkpoint must use the phase-free 20 features")
        prefix_spec = cfg.prefix_smoothing_spec
        if prefix_spec is not None and prefix_spec not in ALLOWED_SPECS.values():
            raise RendererSelectionError("float checkpoint prefix smoothing is unsupported")
        if has_sanitization and prefix_spec != ALLOWED_SPECS["w5"]:
            raise RendererSelectionError(
                "sanitized float checkpoint must retain the frozen w5 prefix observer"
            )
        self.model: CountTextureModel = model
        self.device = str(torch.device(device))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RendererSelectionError("requested CUDA device is unavailable")
        self.identity = {
            "backend": self.kind,
            "checkpoint_bytes": file_bytes,
            "checkpoint_sha256": file_sha,
            "active_tensor_sha256": tensor_sha,
            "source_container_sha256": sanitization.get("source_container_sha256"),
            "checkpoint_kind": (
                "sanitized_release_float"
                if has_sanitization
                else "fresh_training_checkpoint"
            ),
            "sanitization_receipt_present": has_sanitization,
            "features": model.n_features,
            "hidden": cfg.hidden,
            "joint_offset_classes": (2 * cfg.offset_radius + 1) ** 2,
            "prefix_smoothing_spec": (
                prefix_spec if prefix_spec is not None else "requested_w3_or_w5_view"
            ),
            "online_handoff": "absent; this is the clean source graph, not the native R16 deployment",
            "generation_core": "PyTorch float reference",
            "device": self.device,
            "torch_version": str(torch.__version__),
            "deterministic_algorithms": True,
        }

    def render(
        self,
        context: np.ndarray,
        smooth_future: np.ndarray,
        *,
        spec: str,
        event_seed: int,
    ) -> np.ndarray:
        mask = np.ones((1, len(smooth_future)), dtype=np.float32)
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            sampled = sample_count_streams(
                self.model,
                {"prefix_raw_dxdy": np.asarray(context, dtype=np.float32)[None]},
                np.asarray(smooth_future, dtype=np.float32)[None],
                mask,
                spec_key=ALLOWED_SPECS[spec],
                seed=event_seed,
                device=self.device,
                base_hysteresis=0.5,
                lateral_offset_penalty=AF15_LATERAL_PENALTY,
            )[0]
        finally:
            torch.use_deterministic_algorithms(previous)
        return sampled

    def gate_eligibility(
        self, smooth_future: np.ndarray, generated: np.ndarray
    ) -> np.ndarray:
        cfg = self.model.config
        return _float_gate_eligibility(
            smooth_future,
            generated,
            intent_threshold=float(cfg.zero_intent_threshold),
            accumulator_threshold=float(cfg.zero_accumulator_threshold),
            force_release=float(cfg.force_release),
        )


def load_verified_full_sessions(path: str | Path) -> FullSessionCollection:
    """Load a full-session source and require authenticated source custody."""

    try:
        collection = load_full_sessions(path)
    except (OSError, ValueError) as error:
        raise RendererSelectionError(
            f"full-session custody validation failed: {error}"
        ) from error
    if collection.source_hashes_verified is not True:
        raise RendererSelectionError("full-session source hashes were not verified")
    for session in collection.sessions:
        if not session.source_hashes:
            raise RendererSelectionError("a full session has no bound source hashes")
        if any(
            len(str(value)) != 64
            or any(char not in "0123456789abcdef" for char in str(value))
            for value in session.source_hashes.values()
        ):
            raise RendererSelectionError("a full session has an invalid source hash")
        if len(session.dxdy) < CONTEXT_TICKS + SEGMENT_TICKS:
            raise RendererSelectionError(
                "every session needs 256 context ticks and one complete 512-tick score window"
            )
    return collection


def _source_content_sha256(session: FullSession) -> str:
    values = sorted(str(value) for value in session.source_hashes.values())
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _physical_session_copy(session: FullSession) -> np.ndarray:
    """Copy one verified stream while closing the portable-memmap TOCTOU gap."""

    filename = getattr(session.dxdy, "filename", None)
    expected: str | None = None
    source_path: Path | None = None
    if filename is not None and len(session.source_hashes) == 1:
        source_path = Path(str(filename))
        expected = next(iter(session.source_hashes.values()))
        try:
            observed = _sha256_file(source_path)
        except OSError as error:
            raise RendererSelectionError(
                "a portable session became unreadable after validation"
            ) from error
        if observed != expected:
            raise RendererSelectionError("a portable session changed after validation")
    raw = np.ascontiguousarray(np.asarray(session.dxdy), dtype=np.int16)
    if source_path is not None and expected is not None:
        try:
            observed = _sha256_file(source_path)
        except OSError as error:
            raise RendererSelectionError(
                "a portable session became unreadable while it was copied"
            ) from error
        if observed != expected:
            raise RendererSelectionError("a portable session changed while it was copied")
    return raw


def _validate_generated(value: np.ndarray, ticks: int) -> np.ndarray:
    generated = np.asarray(value)
    if generated.shape != (ticks, 2):
        raise RendererSelectionError(
            f"renderer returned {generated.shape}; expected {(ticks, 2)}"
        )
    if not np.issubdtype(generated.dtype, np.number) or not np.all(
        np.isfinite(generated)
    ):
        raise RendererSelectionError("renderer output must be finite numeric counts")
    rounded = np.rint(generated.astype(np.float64, copy=False))
    if not np.array_equal(generated, rounded):
        raise RendererSelectionError("renderer output is not an integer count stream")
    if np.any(rounded < -127.0) or np.any(rounded > 127.0):
        raise RendererSelectionError("renderer output exceeds the deployed +/-127 bound")
    return np.ascontiguousarray(rounded, dtype=np.int16)


def _segment_matrix(stream: np.ndarray) -> np.ndarray:
    complete = (len(stream) // SEGMENT_TICKS) * SEGMENT_TICKS
    if complete < SEGMENT_TICKS:
        raise RendererSelectionError("session has no complete Texture19 segment")
    return np.ascontiguousarray(stream[:complete]).reshape(-1, SEGMENT_TICKS, 2)


def _stream_mechanics(stream: np.ndarray) -> dict[str, Any]:
    values = np.asarray(stream)
    if values.ndim != 2 or values.shape[1:] != (2,) or len(values) == 0:
        raise RendererSelectionError("mechanics requires one non-empty [ticks,2] stream")
    wide = values.astype(np.int64, copy=False)
    active = np.any(wide != 0, axis=1)
    l1 = np.abs(wide).sum(axis=1, dtype=np.int64)
    as_float = wide.astype(np.float64)
    l2 = np.linalg.norm(as_float, axis=1)
    numerators: list[int] = []
    denominators: list[int] = []
    for axis in range(2):
        signs = np.sign(wide[:, axis])
        signs = signs[signs != 0]
        denominators.append(max(len(signs) - 1, 0))
        numerators.append(
            int(np.sum(signs[1:] * signs[:-1] < 0, dtype=np.int64))
            if len(signs) > 1
            else 0
        )
    net = np.sum(wide, axis=0, dtype=np.int64)
    return {
        "ticks": int(len(wide)),
        "active_ticks": int(np.sum(active, dtype=np.int64)),
        "active_fraction": float(np.mean(active)),
        "l1_total_counts": int(np.sum(l1, dtype=np.int64)),
        "l1_mean_per_tick": float(np.mean(l1)),
        "l2_total_counts": float(np.sum(l2, dtype=np.float64)),
        "l2_mean_per_tick": float(np.mean(l2)),
        "sign_flip_numerator_xy": numerators,
        "sign_flip_denominator_xy": denominators,
        "sign_flip_rate_xy": [
            float(numerators[axis] / denominators[axis])
            if denominators[axis]
            else 0.0
            for axis in range(2)
        ],
        "net_xy_counts": net.tolist(),
    }


def _positive_ratio(name: str, generated: float, human: float) -> dict[str, Any]:
    first = float(generated)
    second = float(human)
    reason: str | None = None
    if not math.isfinite(first) or not math.isfinite(second):
        reason = "non_finite_input"
    elif first <= 0.0 and second <= 0.0:
        reason = "both_values_non_positive"
    elif second <= 0.0:
        reason = "human_value_non_positive"
    elif first <= 0.0:
        reason = "generated_value_non_positive"
    if reason is not None:
        return {
            "name": name,
            "generated": first,
            "human": second,
            "ratio": None,
            "log_ratio": None,
            "abs_log_ratio": None,
            "valid": False,
            "invalid_reason": reason,
        }
    ratio = first / second
    logarithm = math.log(ratio)
    return {
        "name": name,
        "generated": first,
        "human": second,
        "ratio": float(ratio),
        "log_ratio": float(logarithm),
        "abs_log_ratio": abs(float(logarithm)),
        "valid": True,
        "invalid_reason": None,
    }


def _ratios(human: Mapping[str, Any], generated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "r_a": _positive_ratio(
            "r_a", generated["active_fraction"], human["active_fraction"]
        ),
        "r_1": _positive_ratio(
            "r_1", generated["l1_mean_per_tick"], human["l1_mean_per_tick"]
        ),
        "r_2": _positive_ratio(
            "r_2", generated["l2_mean_per_tick"], human["l2_mean_per_tick"]
        ),
        "r_f": _positive_ratio(
            "r_f", generated["sign_flip_rate_xy"][0], human["sign_flip_rate_xy"][0]
        ),
    }


def _session_statistics(
    human: np.ndarray,
    generated: np.ndarray,
    eligibility: np.ndarray,
) -> dict[str, Any]:
    if generated.shape != human.shape or eligibility.shape != (len(human),):
        raise RendererSelectionError("full-future selector streams are misaligned")
    human_mechanics = _stream_mechanics(human)
    generated_mechanics = _stream_mechanics(generated)
    active = np.any(generated != 0, axis=1)
    eligible_ticks = int(np.sum(eligibility, dtype=np.int64))
    false_ticks = int(np.sum(active & eligibility, dtype=np.int64))
    human_net = np.asarray(human_mechanics["net_xy_counts"], dtype=np.int64)
    generated_net = np.asarray(generated_mechanics["net_xy_counts"], dtype=np.int64)
    error = generated_net - human_net
    human_norm = float(np.linalg.norm(human_net.astype(np.float64)))
    relative_error = float(
        np.linalg.norm(error.astype(np.float64)) / max(human_norm, 1.0)
    )
    return {
        "ticks": int(len(human)),
        "human": human_mechanics,
        "generated": generated_mechanics,
        "ratios": _ratios(human_mechanics, generated_mechanics),
        "gate": {
            "eligible_ticks": eligible_ticks,
            "false_activation_ticks": false_ticks,
            "false_activation_rate": (
                float(false_ticks / eligible_ticks) if eligible_ticks else None
            ),
        },
        "net_conservation": {
            "human_raw_net_xy_counts": human_net.tolist(),
            "generated_net_xy_counts": generated_net.tolist(),
            "error_xy_counts": error.tolist(),
            "human_raw_net_l2_counts": human_norm,
            "denominator_counts": max(human_norm, 1.0),
            "denominator_floor_applied": bool(human_norm < 1.0),
            "relative_session_net_error": relative_error,
        },
    }


def _pool_mechanics(rows: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    ticks = int(sum(int(row["ticks"]) for row in rows))
    active = int(sum(int(row[side]["active_ticks"]) for row in rows))
    l1 = int(sum(int(row[side]["l1_total_counts"]) for row in rows))
    l2 = float(sum(float(row[side]["l2_total_counts"]) for row in rows))
    numerator = [
        int(sum(int(row[side]["sign_flip_numerator_xy"][axis]) for row in rows))
        for axis in range(2)
    ]
    denominator = [
        int(sum(int(row[side]["sign_flip_denominator_xy"][axis]) for row in rows))
        for axis in range(2)
    ]
    rates = [
        float(numerator[axis] / denominator[axis]) if denominator[axis] else 0.0
        for axis in range(2)
    ]
    return {
        "ticks": ticks,
        "active_ticks": active,
        "active_fraction": float(active / ticks),
        "l1_total_counts": l1,
        "l1_mean_per_tick": float(l1 / ticks),
        "l2_total_counts": l2,
        "l2_mean_per_tick": float(l2 / ticks),
        "sign_flip_numerator_xy": numerator,
        "sign_flip_denominator_xy": denominator,
        "sign_flip_rate_xy": rates,
    }


def _aggregate_user_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RendererSelectionError("cannot aggregate an empty user")
    human = _pool_mechanics(rows, "human")
    generated = _pool_mechanics(rows, "generated")
    eligible = int(sum(int(row["gate"]["eligible_ticks"]) for row in rows))
    false = int(sum(int(row["gate"]["false_activation_ticks"]) for row in rows))
    session_d = [
        float(row["net_conservation"]["relative_session_net_error"]) for row in rows
    ]
    return {
        "scope": "one user; mechanics counts pooled without cross-session flips; D session-equal",
        "ticks": human["ticks"],
        "sessions": len(rows),
        "human": human,
        "generated": generated,
        "ratios": _ratios(human, generated),
        "gate": {
            "eligible_ticks": eligible,
            "false_activation_ticks": false,
            "false_activation_rate": float(false / eligible) if eligible else None,
        },
        "net_conservation": {
            "aggregation": "arithmetic mean of per-session relative net errors",
            "session_relative_net_errors": session_d,
            "relative_session_net_error": float(np.mean(session_d)),
        },
    }


def continuous_v1_selector(
    texture19_w1: float, statistics: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute the exact continuous_v1 T/R/Z/D/S selector record."""

    ratios = statistics["ratios"]
    invalid = [
        f"{name}:{record['invalid_reason']}"
        for name, record in ratios.items()
        if not bool(record["valid"])
    ]
    z = statistics["gate"]["false_activation_rate"]
    if z is None:
        invalid.append("Z:no_gate_eligible_ticks")
    t = float(texture19_w1)
    d = float(statistics["net_conservation"]["relative_session_net_error"])
    if not math.isfinite(t) or t < 0.0:
        invalid.append("T:invalid")
    if not math.isfinite(d) or d < 0.0:
        invalid.append("D:invalid")
    r = (
        float(np.mean([record["abs_log_ratio"] for record in ratios.values()]))
        if not invalid
        else None
    )
    score = (
        float(t / 0.05 + r / math.log(1.05) + float(z) / 0.001 + d / 0.01)
        if r is not None and z is not None
        else None
    )
    return {
        "contract": SELECTOR_CONTRACT,
        "T": t,
        "R": r,
        "Z": z,
        "D": d,
        "S": score,
        "ratios": dict(ratios),
        "scaled_terms": {
            "T_over_0_05": float(t / 0.05),
            "R_over_ln_1_05": float(r / math.log(1.05)) if r is not None else None,
            "Z_over_0_001": float(z / 0.001) if z is not None else None,
            "D_over_0_01": float(d / 0.01),
        },
        "valid": not invalid,
        "invalid_reasons": invalid,
        "relative_net_safeguard_pass": bool(d <= 0.01),
    }


def _w1_report(
    human: np.ndarray, generated: np.ndarray, scale: np.ndarray
) -> dict[str, Any]:
    table = wasserstein1_table(human, generated, scale=scale)
    values = np.asarray([table[name] for name in TEXTURE_FEATURE_NAMES], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "per_feature": table,
        "n_human": int(len(human)),
        "n_generated": int(len(generated)),
        "scale_source": "pooled selected human Texture19 population standard deviation",
    }


def _aliases(collection: FullSessionCollection) -> tuple[dict[str, str], dict[str, str]]:
    user_ids = sorted({session.user_id for session in collection.sessions})
    session_ids = sorted(session.session_id for session in collection.sessions)
    return (
        {value: f"user_{index + 1:04d}" for index, value in enumerate(user_ids)},
        {value: f"session_{index + 1:04d}" for index, value in enumerate(session_ids)},
    )


def _prefix_smoothing_for(renderer: SelectionRenderer, spec: str) -> str:
    value = str(renderer.identity.get("prefix_smoothing_spec", "backend-defined"))
    return ALLOWED_SPECS[spec] if value == "requested_w3_or_w5_view" else value


def evaluate_loaded_sessions(
    collection: FullSessionCollection,
    renderer: SelectionRenderer,
    *,
    specs: Sequence[str] = ("w5",),
    seed: int = 7001,
) -> dict[str, Any]:
    """Evaluate one renderer after an already completed custody check."""

    if collection.source_hashes_verified is not True:
        raise RendererSelectionError("selection requires hash-verified full sessions")
    requested = tuple(dict.fromkeys(str(value).lower() for value in specs))
    if not requested or any(value not in ALLOWED_SPECS for value in requested):
        raise RendererSelectionError("specs must contain w3, w5, or both")
    # Validate the seed before any expensive smoothing or inference.
    stable_session_seed(seed, "renderer-stream", "contract-check")
    users, sessions = _aliases(collection)

    real_features: dict[str, np.ndarray] = {}
    generated_features: dict[str, dict[str, np.ndarray]] = {
        spec: {} for spec in requested
    }
    session_statistics: dict[str, dict[str, dict[str, Any]]] = {
        spec: {} for spec in requested
    }
    session_receipts: list[dict[str, Any]] = []
    session_to_user: dict[str, str] = {}

    for session in collection.sessions:
        raw = _physical_session_copy(session)
        context = raw[:CONTEXT_TICKS]
        human_future = raw[CONTEXT_TICKS:]
        real_segments = _segment_matrix(human_future)
        mask = np.ones(real_segments.shape[:2], dtype=np.float32)
        real_features[session.session_id] = np.asarray(
            texture_features(real_segments, mask), dtype=np.float64
        )
        event_seed = stable_session_seed(seed, "renderer-stream", session.session_id)
        spec_receipts: dict[str, Any] = {}
        for spec in requested:
            # Smooth once across the complete physical session.  Only after
            # smoothing do we take the post-context intent; this preserves the
            # exact session-start boundary behavior used by selection.
            smooth = smooth_dxdy(raw.astype(np.float32), spec=ALLOWED_SPECS[spec])
            smooth_future = np.ascontiguousarray(smooth[CONTEXT_TICKS:], dtype=np.float32)
            generated = _validate_generated(
                renderer.render(
                    context,
                    smooth_future,
                    spec=spec,
                    event_seed=event_seed,
                ),
                len(human_future),
            )
            eligibility = np.asarray(
                renderer.gate_eligibility(smooth_future, generated), dtype=bool
            )
            if eligibility.shape != (len(human_future),):
                raise RendererSelectionError("backend returned a misaligned gate mask")
            generated_segments = _segment_matrix(generated)
            generated_features[spec][session.session_id] = np.asarray(
                texture_features(generated_segments, mask), dtype=np.float64
            )
            stats = _session_statistics(human_future, generated, eligibility)
            session_statistics[spec][session.session_id] = stats
            spec_receipts[spec] = {
                "future_ticks": len(human_future),
                "texture19_segments": len(real_segments),
                "dropped_texture_tail_ticks": len(human_future)
                - len(real_segments) * SEGMENT_TICKS,
                "gate_eligible_ticks": stats["gate"]["eligible_ticks"],
                "gate_false_activation_ticks": stats["gate"][
                    "false_activation_ticks"
                ],
                "relative_session_net_error": stats["net_conservation"][
                    "relative_session_net_error"
                ],
            }
        session_receipts.append(
            {
                "session": sessions[session.session_id],
                "user": users[session.user_id],
                "source_content_sha256": _source_content_sha256(session),
                "physical_ticks": len(raw),
                "context_range": [0, CONTEXT_TICKS],
                "future_range": [CONTEXT_TICKS, len(raw)],
                "event_seed_u64": event_seed,
                "specs": spec_receipts,
            }
        )
        session_to_user[session.session_id] = session.user_id

    pooled_real = np.concatenate(
        [real_features[name] for name in sorted(real_features)], axis=0
    )
    scale = np.std(pooled_real, axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    by_user: dict[str, list[str]] = {user: [] for user in users}
    for session_id, user_id in session_to_user.items():
        by_user[user_id].append(session_id)

    spec_reports: dict[str, Any] = {}
    for spec in requested:
        user_reports: list[dict[str, Any]] = []
        selectors: list[dict[str, Any]] = []
        for user_id in sorted(by_user):
            member_sessions = sorted(by_user[user_id])
            human = np.concatenate(
                [real_features[value] for value in member_sessions], axis=0
            )
            generated = np.concatenate(
                [generated_features[spec][value] for value in member_sessions], axis=0
            )
            w1 = _w1_report(human, generated, scale)
            statistics = _aggregate_user_statistics(
                [session_statistics[spec][value] for value in member_sessions]
            )
            selector = continuous_v1_selector(float(w1["mean"]), statistics)
            if not selector["valid"]:
                reasons = ", ".join(selector["invalid_reasons"])
                raise RendererSelectionError(
                    f"continuous_v1 is undefined for {users[user_id]}/{spec}: {reasons}"
                )
            selectors.append(selector)
            user_reports.append(
                {
                    "user": users[user_id],
                    "sessions": [sessions[value] for value in member_sessions],
                    "texture19_w1": w1,
                    "selector": selector,
                    "full_future_statistics": statistics,
                }
            )
        macro = {
            name: float(np.mean([float(row[name]) for row in selectors]))
            for name in ("T", "R", "Z", "D", "S")
        }
        reconstructed = float(
            macro["T"] / 0.05
            + macro["R"] / math.log(1.05)
            + macro["Z"] / 0.001
            + macro["D"] / 0.01
        )
        if not math.isclose(macro["S"], reconstructed, rel_tol=0.0, abs_tol=1e-12):
            raise RendererSelectionError("user-macro S does not reconstruct from T/R/Z/D")
        pooled_generated = np.concatenate(
            [generated_features[spec][name] for name in sorted(real_features)], axis=0
        )
        spec_reports[spec] = {
            "intent_smoothing_spec": ALLOWED_SPECS[spec],
            "prefix_observer_smoothing_spec": _prefix_smoothing_for(renderer, spec),
            "rollout": "one uninterrupted carried rollout after the session-start context",
            "af15_lateral_penalty": AF15_LATERAL_PENALTY,
            "pooled_texture19_w1": _w1_report(
                pooled_real, pooled_generated, scale
            ),
            "user_macro": {
                **macro,
                "aggregation": "arithmetic mean of per-user selectors",
                "formula_reconstruction": reconstructed,
            },
            "users": user_reports,
        }

    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete_full_input_panel",
        "protocol": {
            "tick_hz": TICK_HZ,
            "axis_convention": "x-right,y-up",
            "context": "exact physical ticks [0,256) from each session",
            "future": "every remaining physical tick; one begin and no segment reset",
            "prefix_observer_smoothing": {
                value: _prefix_smoothing_for(renderer, value) for value in requested
            },
            "future_boundary_context": (
                "the backend's separately observed/smoothed session-start prefix; "
                "the future never warms or resets the recurrent state"
            ),
            "texture19_segment_ticks": SEGMENT_TICKS,
            "texture19_segmentation": (
                "non-overlapping windows beginning at physical tick 256; "
                "incomplete terminal window excluded from T only"
            ),
            "full_future_mechanics": "all post-context ticks, including Texture19 tail",
            "intent_specs": {value: ALLOWED_SPECS[value] for value in requested},
            "af15_lateral_penalty": AF15_LATERAL_PENALTY,
            "selector": {
                "contract": SELECTOR_CONTRACT,
                "T": "mean standardized W1 across the public 19 Texture features",
                "R": "mean(abs(log(r_a,r_1,r_2,r_f))); r_f is x-axis only",
                "Z": "causal gate-eligible false-activation rate",
                "D": "session-equal relative net error with one-count denominator floor",
                "formula": "T/0.05 + R/ln(1.05) + Z/0.001 + D/0.01",
                "non_positive_ratio_policy": "invalid; no epsilon, clipping, or imputation",
                "primary_aggregation": "compute within user, then arithmetic user mean",
            },
            "session_seed": {
                "method": (
                    "little-endian uint64 from first 8 bytes of "
                    "SHA256(ascii(seed) || NUL || 'renderer-stream' || NUL || session_id)"
                ),
                "seed": seed,
            },
            "population_boundary": (
                "scores describe exactly the supplied panel; this evaluator cannot prove "
                "that its identities were absent from model development"
            ),
        },
        "input": {
            "source_kind": collection.source_kind,
            "source_manifest_sha256": collection.source_manifest_sha256,
            "source_hashes_verified": True,
            "sessions": len(collection.sessions),
            "users": len(users),
            "physical_ticks": int(sum(len(row.dxdy) for row in collection.sessions)),
            "identity_disclosure": "source user/session strings and paths omitted",
        },
        "renderer": dict(renderer.identity),
        "human_reference": {
            "texture19_features": list(TEXTURE_FEATURE_NAMES),
            "texture19_scale": {
                name: float(scale[index])
                for index, name in enumerate(TEXTURE_FEATURE_NAMES)
            },
            "scale_floor": "standard deviations below 1e-6 replaced by 1.0",
            "segments": int(len(pooled_real)),
        },
        "specs": spec_reports,
        "sessions": session_receipts,
    }
    _reject_nonfinite(report)
    return report


def _reject_nonfinite(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RendererSelectionError(f"non-finite value at {path}")


def evaluate_renderer_selection(
    source: str | Path,
    *,
    backend: str = "native",
    model: str | Path | None = None,
    library: str | Path | None = None,
    specs: Sequence[str] = ("w5",),
    seed: int = 7001,
    device: str = "cpu",
) -> dict[str, Any]:
    """Hash-check input, run one backend, and return a sanitized receipt."""

    collection = load_verified_full_sessions(source)
    selected_model = default_renderer_model(backend) if model is None else Path(model)
    if backend == "native":
        if str(device) != "cpu":
            raise RendererSelectionError("--device applies only to the float backend")
        renderer: SelectionRenderer = NativeSelectionRenderer(
            selected_model, library=library
        )
    elif backend == "float":
        if library is not None:
            raise RendererSelectionError("--library applies only to the native backend")
        renderer = FloatSelectionRenderer(selected_model, device=device)
    else:
        raise RendererSelectionError("backend must be 'native' or 'float'")
    return evaluate_loaded_sessions(
        collection, renderer, specs=specs, seed=seed
    )


__all__ = [
    "AF15_LATERAL_PENALTY",
    "ALLOWED_SPECS",
    "CONTEXT_TICKS",
    "FloatSelectionRenderer",
    "NativeSelectionRenderer",
    "REPORT_SCHEMA",
    "RendererSelectionError",
    "SEGMENT_TICKS",
    "SELECTOR_CONTRACT",
    "continuous_v1_selector",
    "default_renderer_model",
    "evaluate_loaded_sessions",
    "evaluate_renderer_selection",
    "load_verified_full_sessions",
    "stable_session_seed",
]
