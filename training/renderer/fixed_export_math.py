"""Selected ABCFIX1 strategy-2 export arithmetic, extracted without reassociation.

Source functions are from the frozen PTQ authority; extraction manifest records
file hashes. Runtime differential fixtures and abandoned strategies are omitted.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any
import zlib
import numpy as np

DATA_LIMIT=46080
HEADER_SIZE=256
MAGIC=b"ABCFIX1\0"
VERSION=1
FEATURE_Q=8
HIDDEN_Q=15
GATE_Q=14
SMOOTH_Q=16
PROB_Q=15
EXP_Q=20
LOG_Q=14
TANGENT_Q=15
LUT_INTERVALS=256
LUT_POINTS=LUT_INTERVALS+1
Q31=1<<31
AF15=1.5
LATERAL_FREE=1.0
OFFSET_RADIUS=5
MODEL_REL=Path("provided_float_checkpoint")
@dataclass
class Params:
    hidden: int
    feature_names: tuple[str, ...]
    feature_indices: np.ndarray
    w_ih: np.ndarray
    w_hh: np.ndarray
    b_ih: np.ndarray
    b_hh: np.ndarray
    w_emit: np.ndarray
    b_emit: np.ndarray
    w_off: np.ndarray
    b_off: np.ndarray
    config: dict[str, Any]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "w_ih": self.w_ih,
            "w_hh": self.w_hh,
            "w_emit": self.w_emit,
            "w_off": self.w_off,
            "b_ih": self.b_ih,
            "b_hh": self.b_hh,
            "b_emit": self.b_emit,
            "b_off": self.b_off,
        }


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = reference.astype(np.float64, copy=False)
    b = candidate.astype(np.float64, copy=False)
    d = b - a
    denom = float(np.linalg.norm(a.ravel()))
    return {
        "rmse": float(np.sqrt(np.mean(d * d))),
        "max_abs": float(np.max(np.abs(d))),
        "relative_l2": float(np.linalg.norm(d.ravel()) / denom) if denom else 0.0,
    }


@dataclass(frozen=True)
class Strategy:
    name: str
    strategy_id: int


@dataclass
class LUTs:
    sigmoid_q15: np.ndarray
    tanh_q15: np.ndarray
    exp_neg_q20: np.ndarray
    log1p_mantissa_q12: np.ndarray


@dataclass
class QuantizedTensor:
    values: np.ndarray
    multipliers_q31: np.ndarray
    row_group: np.ndarray
    scales: np.ndarray


def quantize_scalar(value: float, q: int) -> int:
    return int(np.rint(np.float64(value) * np.float64(1 << q)))


def quantize_array(values: np.ndarray, q: int, dtype: Any) -> np.ndarray:
    info = np.iinfo(dtype)
    out = np.rint(values.astype(np.float64) * float(1 << q))
    return np.ascontiguousarray(np.clip(out, info.min, info.max).astype(dtype))


def align4(data: bytearray) -> None:
    data.extend(b"\0" * ((-len(data)) & 3))


def make_luts() -> LUTs:
    sigmoid_x = np.linspace(-8.0, 8.0, LUT_POINTS, dtype=np.float64)
    tanh_x = sigmoid_x
    exp_x = np.linspace(0.0, 16.0, LUT_POINTS, dtype=np.float64)
    mantissa_x = np.linspace(0.0, 1.0, LUT_POINTS, dtype=np.float64)
    sigmoid = np.rint((1.0 / (1.0 + np.exp(-sigmoid_x))) * (1 << PROB_Q))
    sigmoid = np.clip(sigmoid, 0, (1 << PROB_Q) - 1).astype("<u2")
    tanh = np.rint(np.tanh(tanh_x) * ((1 << HIDDEN_Q) - 1))
    tanh = np.clip(tanh, -(1 << HIDDEN_Q) + 1, (1 << HIDDEN_Q) - 1).astype("<i2")
    exp_neg = np.rint(np.exp(-exp_x) * (1 << EXP_Q))
    exp_neg = np.clip(exp_neg, 0, (1 << EXP_Q)).astype("<u4")
    log1p = np.rint(np.log1p(mantissa_x) * (1 << LOG_Q))
    log1p = np.clip(log1p, 0, np.iinfo(np.int16).max).astype("<i2")
    return LUTs(sigmoid, tanh, exp_neg, log1p)


def grouping_for(name: str, rows: int, hidden: int, strategy: Strategy) -> list[np.ndarray]:
    if strategy.name == "row":
        return [np.asarray([i], dtype=np.int32) for i in range(rows)]
    if name in {"w_ih", "w_hh"}:
        return [np.arange(g * hidden, (g + 1) * hidden, dtype=np.int32) for g in range(3)]
    if strategy.name == "gru_gate_offrow" and name == "w_off":
        return [np.asarray([i], dtype=np.int32) for i in range(rows)]
    return [np.arange(rows, dtype=np.int32)]


def quantize_weight(name: str, values: np.ndarray, hidden: int, strategy: Strategy) -> QuantizedTensor:
    matrix = values.reshape(values.shape[0], -1).astype(np.float32)
    groups = grouping_for(name, matrix.shape[0], hidden, strategy)
    quantized = np.zeros_like(matrix, dtype=np.int8)
    row_group = np.empty(matrix.shape[0], dtype=np.int32)
    scales: list[float] = []
    multipliers: list[int] = []
    input_q = FEATURE_Q if name == "w_ih" else HIDDEN_Q
    factor_power = GATE_Q - input_q
    for group_index, rows in enumerate(groups):
        maximum = float(np.max(np.abs(matrix[rows])))
        scale = maximum / 127.0 if maximum > 0.0 else 0.0
        if scale == 0.0:
            q = np.zeros_like(matrix[rows], dtype=np.int8)
            multiplier = 0
        else:
            q = np.clip(np.rint(matrix[rows].astype(np.float64) / scale), -127, 127).astype(np.int8)
            factor = math.ldexp(scale, factor_power)
            multiplier = int(np.rint(factor * Q31))
            if not (0 < multiplier < Q31):
                raise RuntimeError(f"{strategy.name}/{name}: Q31 factor is not in (0,1): {factor}")
        quantized[rows] = q
        row_group[rows] = group_index
        scales.append(scale)
        multipliers.append(multiplier)
    return QuantizedTensor(
        np.ascontiguousarray(quantized.reshape(values.shape)),
        np.asarray(multipliers, dtype="<i4"),
        row_group,
        np.asarray(scales, dtype=np.float64),
    )


def section_append(body: bytearray, payload: bytes) -> tuple[int, int]:
    align4(body)
    offset = HEADER_SIZE + len(body)
    body.extend(payload)
    length = len(payload)
    return offset, length


def build_blob(params: Params, audit: dict[str, Any], strategy: Strategy, luts: LUTs) -> tuple[bytes, dict[str, Any]]:
    tensors = {
        name: quantize_weight(name, getattr(params, name), params.hidden, strategy)
        for name in ("w_ih", "w_hh", "w_emit", "w_off")
    }
    biases = {
        name: quantize_array(getattr(params, name), GATE_Q, np.int32)
        for name in ("b_ih", "b_hh", "b_emit", "b_off")
    }
    weight_order = ("w_ih", "w_hh", "w_emit", "w_off")
    bias_order = ("b_ih", "b_hh", "b_emit", "b_off")
    body = bytearray()
    sections: dict[str, tuple[int, int]] = {}
    sections["weights"] = section_append(body, b"".join(tensors[name].values.tobytes(order="C") for name in weight_order))
    sections["biases"] = section_append(body, b"".join(biases[name].astype("<i4", copy=False).tobytes() for name in bias_order))
    sections["multipliers"] = section_append(body, b"".join(tensors[name].multipliers_q31.tobytes() for name in weight_order))
    sections["sigmoid"] = section_append(body, luts.sigmoid_q15.astype("<u2", copy=False).tobytes())
    sections["tanh"] = section_append(body, luts.tanh_q15.astype("<i2", copy=False).tobytes())
    sections["exp"] = section_append(body, luts.exp_neg_q20.astype("<u4", copy=False).tobytes())
    sections["log"] = section_append(body, luts.log1p_mantissa_q12.astype("<i2", copy=False).tobytes())
    align4(body)

    config = params.config
    config_q = (
        quantize_scalar(config["emit_logit_bias"], GATE_Q),
        quantize_scalar(1.0 / float(config["temperature"]), 16),
        quantize_scalar(1.0 / float(config["offset_magnitude_temperature"]), 16),
        quantize_scalar(1.0 / float(config["offset_direction_temperature"]), 16),
        quantize_scalar(config["base_hysteresis"], SMOOTH_Q),
        quantize_scalar(config["force_release"], SMOOTH_Q),
        int(config["max_abs_count"]),
        quantize_scalar(config["zero_intent_threshold"], SMOOTH_Q),
        quantize_scalar(config["zero_accumulator_threshold"], SMOOTH_Q),
        quantize_scalar(AF15, 16),
        quantize_scalar(LATERAL_FREE, TANGENT_Q),
    )
    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    struct.pack_into("<HHHHHHI", header, 8, VERSION, HEADER_SIZE, params.hidden, len(params.feature_names), OFFSET_RADIUS, strategy.strategy_id, 0b1111)
    total_bytes = HEADER_SIZE + len(body)
    struct.pack_into("<I", header, 24, total_bytes)
    for index, name in enumerate(("weights", "biases", "multipliers", "sigmoid", "tanh", "exp", "log")):
        struct.pack_into("<II", header, 28 + 8 * index, *sections[name])
    crc = zlib.crc32(body) & 0xFFFFFFFF
    struct.pack_into("<I", header, 84, crc)
    struct.pack_into("<8B", header, 88, FEATURE_Q, HIDDEN_Q, GATE_Q, SMOOTH_Q, PROB_Q, EXP_Q, LOG_Q, TANGENT_Q)
    struct.pack_into("<4I", header, 96, *(int(tensors[n].values.size) for n in weight_order))
    struct.pack_into("<4I", header, 112, *(int(biases[n].size) for n in bias_order))
    struct.pack_into("<4H", header, 128, *(len(tensors[n].multipliers_q31) for n in weight_order))
    struct.pack_into("<11i", header, 136, *config_q)
    header[180:212] = bytes.fromhex(audit["checkpoint_sha256"])
    header[212:228] = hashlib.sha256(b"ABCFIX1-retained-H80-fixed-schema-v1").digest()[:16]
    blob = bytes(header + body)
    if len(blob) != total_bytes:
        raise AssertionError((len(blob), total_bytes))

    error: dict[str, Any] = {}
    for name in weight_order:
        qt = tensors[name]
        dequant = np.empty_like(getattr(params, name), dtype=np.float32).reshape(qt.values.shape[0], -1)
        original = getattr(params, name).reshape(qt.values.shape[0], -1)
        for row in range(len(dequant)):
            dequant[row] = qt.values.reshape(len(dequant), -1)[row].astype(np.float32) * np.float32(qt.scales[qt.row_group[row]])
        error[name] = error_metrics(original, dequant)
    for name in bias_order:
        error[name] = error_metrics(getattr(params, name), biases[name].astype(np.float32) / np.float32(1 << GATE_Q))
    breakdown = {
        "header_bytes": HEADER_SIZE,
        **{f"{name}_bytes_including_alignment": (sections[name][0] - (HEADER_SIZE if i == 0 else sections[("weights", "biases", "multipliers", "sigmoid", "tanh", "exp", "log")[i - 1]][0] + sections[("weights", "biases", "multipliers", "sigmoid", "tanh", "exp", "log")[i - 1]][1])) + sections[name][1] for i, name in enumerate(("weights", "biases", "multipliers", "sigmoid", "tanh", "exp", "log"))},
        "tail_alignment_bytes": total_bytes - (sections["log"][0] + sections["log"][1]),
    }
    if sum(breakdown.values()) != total_bytes:
        raise AssertionError((sum(breakdown.values()), total_bytes, breakdown))
    manifest = {
        "format": "ABCFIX1",
        "strategy": strategy.name,
        "source_checkpoint": str(MODEL_REL).replace("\\", "/"),
        "source_sha256": audit["checkpoint_sha256"],
        "artifact_bytes": total_bytes,
        "margin_to_46080": DATA_LIMIT - total_bytes,
        "compliant": total_bytes <= DATA_LIMIT,
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "body_crc32": f"{crc:08x}",
        "zlib9_bytes": len(zlib.compress(blob, 9)),
        "breakdown": breakdown,
        "sections": {name: {"offset": off, "length": length} for name, (off, length) in sections.items()},
        "tensor_group_counts": {name: len(tensors[name].multipliers_q31) for name in weight_order},
        "parameter_error": error,
        "q_formats": {
            "feature": "signed Q8 in int16",
            "hidden": "signed Q15 in int16",
            "gate_and_logits": f"signed Q{GATE_Q} in int32",
            "smooth_and_accumulator": "signed Q16 in int32/int64 reference state",
            "probability": "unsigned Q15",
            "exp": "unsigned Q20",
            "tangent_normal": "signed Q15",
            "requant_multiplier": "signed Q31 with implicit shift 31",
        },
    }
    return blob, manifest
