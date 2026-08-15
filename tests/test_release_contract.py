from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from abcurves.model_store import (
    ModelIntegrityError,
    RELEASE_FILE_ANCHORS,
    resolve_model_files,
    resolve_renderer_float,
)
from abcurves.renderer import load_count_model
from abcurves.planner import (
    CONTRACTED_PLANNER_SCHEMA,
    SUPPORTED_PLANNER_SCHEMAS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_every_release_model_matches_manifest() -> None:
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text("utf-8"))
    assert manifest["schema"] == "abcurves.release_models.v2"
    assert manifest["release"] == "1.5.0"
    assert manifest["default_seed"] == 7
    assert manifest["seeds"] == [7, 23]
    assert set(manifest["files"]) == {
        "planner_seed7.pt",
        "planner_seed23.pt",
        "renderer_global_h80.bin",
        "renderer_global_h80_float.pt",
    }
    for name, receipt in manifest["files"].items():
        path = ROOT / "models" / name
        assert path.stat().st_size == receipt["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        assert RELEASE_FILE_ANCHORS[name] == (receipt["bytes"], receipt["sha256"])
    for name, receipt in manifest["native_libraries"].items():
        path = ROOT / name
        assert path.stat().st_size == receipt["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    seed7 = resolve_model_files(7)
    seed23 = resolve_model_files(23)
    assert seed7.planner != seed23.planner
    assert seed7.renderer == seed23.renderer
    assert resolve_renderer_float() == ROOT / "models" / "renderer_global_h80_float.pt"


def test_renderer_manifest_binds_the_final_law() -> None:
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text("utf-8"))
    renderer = manifest["renderer"]
    assert renderer["context_ticks"] == 256
    assert renderer["future_ticks"] == 800
    assert renderer["features"] == 20
    assert renderer["hidden"] == 80
    assert renderer["presentation_budget"] == 118_345
    assert renderer["lateral_offset_penalty"] == 1.5
    assert renderer["windows_x64_caller_state_bytes"] == 5_088
    assert renderer["maximum_axis_emission_counts"] == 127
    assert renderer["source_float_container_sha256"] == (
        "e9951a9bc25b69bf652cebbca8749badd02b8c0675a5f1ad4cde7f1a8624132a"
    )


def test_clean_renderer_float_export_is_active_and_path_free() -> None:
    path = ROOT / "models" / "renderer_global_h80_float.pt"
    model, report = load_count_model(path)
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_362
    assert report["n_features"] == 20
    assert report["sanitization"]["active_tensor_contract"]["sha256"] == (
        "d09a4a269be583eac6e123bf6be9226bf8ee8e1c9fa8f51faed243b965187206"
    )
    text = repr({key: value for key, value in report.items() if key != "history"}).lower()
    assert "c:\\" not in text and "e:\\" not in text and "iabox" not in text


def test_runtime_accepts_only_contracted_planner_schemas() -> None:
    assert SUPPORTED_PLANNER_SCHEMAS == {CONTRACTED_PLANNER_SCHEMA}


def test_torch_model_containers_are_path_free_and_frozen() -> None:
    forbidden = (
        "c:\\",
        "e:\\",
        "iabox",
        "019f",
        "phalm-r-v2",
        "exploratory",
    )
    for path in sorted((ROOT / "models").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        text = repr(
            {
                key: value
                for key, value in payload.items()
                if key not in {"state_dict", "model_state_dict"}
            }
        ).lower()
        assert not any(token in text for token in forbidden)
        status = payload.get(
            "release_status", payload.get("report", {}).get("release_status")
        )
        assert status == "FROZEN"


def test_manifest_detects_a_changed_model(tmp_path: Path) -> None:
    source = ROOT / "models"
    for path in source.iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    target = tmp_path / "planner_seed7.pt"
    with target.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ModelIntegrityError):
        resolve_model_files(7, model_dir=tmp_path)
