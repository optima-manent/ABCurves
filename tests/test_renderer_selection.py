from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from abcurves.global_data import FullSession, write_portable_full_sessions
from abcurves.renderer import (
    RendererConfig,
    save_count_model,
    train_count_texture_model,
)
from evaluation.cli import _emit, build_parser
from evaluation.renderer_selection import (
    FloatSelectionRenderer,
    NativeSelectionRenderer,
    RendererSelectionError,
    _native_gate_eligibility,
    continuous_v1_selector,
    default_renderer_model,
    evaluate_loaded_sessions,
    evaluate_renderer_selection,
    load_verified_full_sessions,
    stable_session_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def _pattern(ticks: int) -> np.ndarray:
    cycle = np.asarray(
        [
            [0, 0],
            [1, 0],
            [-1, 1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 1],
            [-1, -1],
        ],
        dtype=np.int16,
    )
    return np.tile(cycle, (math.ceil(ticks / len(cycle)), 1))[:ticks]


class _IdentityRenderer:
    kind = "test_identity"
    identity = {"backend": kind, "artifact_sha256": "0" * 64}

    def render(self, context, smooth_future, *, spec, event_seed):
        del context, spec, event_seed
        return _pattern(len(smooth_future))

    def gate_eligibility(self, smooth_future, generated):
        del smooth_future
        return np.all(np.asarray(generated) == 0, axis=1)


def _portable_panel(tmp_path):
    sessions = [
        FullSession(
            user_id="secret-alice",
            session_id="secret-session-a",
            dxdy=_pattern(1_280),
            source_ref="source-a",
        ),
        FullSession(
            user_id="secret-bob",
            session_id="secret-session-b",
            dxdy=_pattern(1_280),
            source_ref="source-b",
        ),
    ]
    return write_portable_full_sessions(tmp_path / "panel", sessions)


def test_stable_session_seed_is_frozen_sha256_uint64() -> None:
    assert stable_session_seed(7001, "renderer-stream", "session-alpha") == (
        70_760_051_759_390_107
    )
    assert stable_session_seed(7001, "renderer-stream", "session-alpha") == (
        stable_session_seed(7001, "renderer-stream", "session-alpha")
    )
    assert stable_session_seed(7001, "other-domain", "session-alpha") != (
        stable_session_seed(7001, "renderer-stream", "session-alpha")
    )


def test_continuous_v1_formula_and_no_epsilon_policy() -> None:
    ratios = {
        name: {
            "valid": True,
            "invalid_reason": None,
            "abs_log_ratio": value,
        }
        for name, value in zip(("r_a", "r_1", "r_2", "r_f"), (0.1, 0.2, 0.3, 0.4))
    }
    statistics = {
        "ratios": ratios,
        "gate": {"false_activation_rate": 0.002},
        "net_conservation": {"relative_session_net_error": 0.03},
    }
    selector = continuous_v1_selector(0.1, statistics)
    expected = 0.1 / 0.05 + 0.25 / math.log(1.05) + 0.002 / 0.001 + 0.03 / 0.01
    assert selector["valid"] is True
    assert selector["R"] == pytest.approx(0.25)
    assert selector["S"] == pytest.approx(expected)

    ratios["r_f"] = {
        "valid": False,
        "invalid_reason": "human_value_non_positive",
        "abs_log_ratio": None,
    }
    invalid = continuous_v1_selector(0.1, statistics)
    assert invalid["valid"] is False
    assert invalid["S"] is None
    assert invalid["invalid_reasons"] == ["r_f:human_value_non_positive"]


def test_native_gate_replay_uses_q16_and_safety_priority() -> None:
    smooth = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [32.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    generated = np.asarray([[0, 0], [1, 0], [32, 0], [0, 0]], dtype=np.int16)
    eligible = _native_gate_eligibility(smooth, generated)
    assert eligible.tolist() == [True, True, False, False]


def test_full_session_selection_is_carried_user_macro_and_sanitized(tmp_path) -> None:
    manifest = _portable_panel(tmp_path)
    collection = load_verified_full_sessions(manifest)
    report = evaluate_loaded_sessions(
        collection,
        _IdentityRenderer(),
        specs=("w3", "w5"),
        seed=7001,
    )
    assert report["schema"] == "abcurves.renderer_selection.v1"
    assert report["input"]["source_hashes_verified"] is True
    assert report["input"]["sessions"] == 2
    assert report["human_reference"]["segments"] == 4
    for spec in ("w3", "w5"):
        macro = report["specs"][spec]["user_macro"]
        assert macro["T"] == pytest.approx(0.0, abs=1e-12)
        assert macro["R"] == pytest.approx(0.0, abs=1e-12)
        assert macro["Z"] == pytest.approx(0.0, abs=1e-12)
        assert macro["D"] == pytest.approx(0.0, abs=1e-12)
        assert macro["S"] == pytest.approx(0.0, abs=1e-12)
    for session in report["sessions"]:
        assert session["context_range"] == [0, 256]
        assert session["future_range"] == [256, 1280]
        assert session["specs"]["w5"]["texture19_segments"] == 2
    encoded = json.dumps(report, sort_keys=True)
    assert "secret-alice" not in encoded
    assert "secret-bob" not in encoded
    assert "secret-session-a" not in encoded
    assert str(tmp_path) not in encoded


def test_full_session_hash_mismatch_fails_before_scoring(tmp_path) -> None:
    manifest = _portable_panel(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    array_path = manifest.parent / payload["sessions"][0]["dxdy_npy"]
    values = np.load(array_path, allow_pickle=False)
    values[0, 0] += 1
    np.save(array_path, values, allow_pickle=False)
    with pytest.raises(RendererSelectionError, match="custody validation failed"):
        load_verified_full_sessions(manifest)


def test_renderer_selection_cli_is_publicly_discoverable() -> None:
    parsed = build_parser().parse_args(
        [
            "renderer-selection",
            "sessions.json",
            "--backend",
            "float",
            "--specs",
            "w3",
            "w5",
        ]
    )
    assert parsed.command == "renderer-selection"
    assert parsed.backend == "float"
    assert parsed.specs == ["w3", "w5"]


def test_result_publication_is_atomic_and_never_overwrites(tmp_path) -> None:
    destination = tmp_path / "selection.json"
    _emit({"status": "first"}, str(destination))
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "first"}
    assert not list(tmp_path.glob(".selection.json.*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _emit({"status": "second"}, str(destination))
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "first"}
    assert not list(tmp_path.glob(".selection.json.*.tmp"))


def test_selection_backends_authenticate_both_shipped_models() -> None:
    assert default_renderer_model("float") == (
        ROOT / "models" / "renderer_global_h80_float.pt"
    )
    native = NativeSelectionRenderer(ROOT / "models" / "renderer_global_h80.bin")
    assert native.identity["artifact_sha256"] == (
        "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b"
    )
    floating = FloatSelectionRenderer(
        ROOT / "models" / "renderer_global_h80_float.pt"
    )
    assert floating.identity["active_tensor_sha256"] == (
        "d09a4a269be583eac6e123bf6be9226bf8ee8e1c9fa8f51faed243b965187206"
    )
    assert floating.identity["online_handoff"].startswith("absent")


def test_present_sanitization_receipt_is_strictly_verified(tmp_path) -> None:
    source = ROOT / "models" / "renderer_global_h80_float.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["report"]["sanitization"]["active_tensor_contract"]["sha256"] = "0" * 64
    changed = tmp_path / "changed_receipt.pt"
    torch.save(payload, changed)
    with pytest.raises(RendererSelectionError, match="tensor receipt differs"):
        FloatSelectionRenderer(changed)


def test_fresh_trainer_checkpoint_runs_through_public_selection_backend(tmp_path) -> None:
    raw = np.zeros((1_280, 2), dtype=np.int16)
    for start in range(0, len(raw), 192):
        raw[start : start + 64, 0] = 2
        raw[start + 80 : start + 144, 0] = -2
        raw[start + 16 : start + 48, 1] = 1
        raw[start + 96 : start + 128, 1] = -1
    arrays = {
        "prefix_raw_dxdy": raw[:256].astype(np.float32)[None],
        "future_raw_dxdy": raw[256:1056].astype(np.float32)[None],
    }
    model, training_report = train_count_texture_model(
        arrays,
        RendererConfig(presentation_budget=1, batch_size=1, seed=3),
        device="cpu",
        log_every=0,
    )
    checkpoint = tmp_path / "fresh_renderer.pt"
    save_count_model(model, training_report, checkpoint)
    manifest = write_portable_full_sessions(
        tmp_path / "selection_panel",
        [
            FullSession(
                user_id="candidate-user",
                session_id="candidate-session",
                dxdy=raw,
                source_ref="candidate-source",
            )
        ],
    )
    report = evaluate_renderer_selection(
        manifest,
        backend="float",
        model=checkpoint,
        specs=("w5",),
        seed=7001,
    )
    assert report["status"] == "complete_full_input_panel"
    assert report["renderer"]["checkpoint_kind"] == "fresh_training_checkpoint"
    assert report["renderer"]["sanitization_receipt_present"] is False
    assert report["renderer"]["checkpoint_sha256"]
    assert report["renderer"]["active_tensor_sha256"]
    assert report["specs"]["w5"]["user_macro"]["S"] >= 0.0
