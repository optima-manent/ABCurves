from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from abcurves.model_store import ModelIntegrityError, resolve_model_files
from abcurves.personalization import CausalStyleState, zero_causal_state
from abcurves.planner import (
    CONTRACTED_PLANNER_SCHEMA,
    SPLIT_PREFIX_PLANNER_SCHEMA,
    SUPPORTED_PLANNER_SCHEMAS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_every_release_model_matches_manifest() -> None:
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text("utf-8"))
    assert manifest["default_seed"] == 7
    assert manifest["seeds"] == [7, 23]
    for name, receipt in manifest["files"].items():
        path = ROOT / "models" / name
        assert path.stat().st_size == receipt["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    for seed in manifest["seeds"]:
        files = resolve_model_files(seed)
        assert files.planner.is_file()
        assert files.renderer.is_file()
        assert files.renderer_adapter.is_file()


def test_runtime_accepts_only_contracted_planner_schemas() -> None:
    assert SUPPORTED_PLANNER_SCHEMAS == {
        CONTRACTED_PLANNER_SCHEMA,
        SPLIT_PREFIX_PLANNER_SCHEMA,
    }


def test_model_containers_are_path_free_and_frozen() -> None:
    forbidden = (
        "c:\\",
        "e:\\",
        "iabox",
        "019f",
        "phalm-r-v2",
        "exploratory",
    )
    for path in sorted((ROOT / "models").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        text = repr(
            {
                key: value
                for key, value in payload.items()
                if key not in {"state_dict", "model_state_dict"}
            }
        ).lower()
        assert not any(token in text for token in forbidden)
        status = payload.get("release_status", payload.get("report", {}).get("release_status"))
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


def test_causal_style_state_is_prior_only_and_run_local() -> None:
    tracker = CausalStyleState()
    assert np.array_equal(tracker.before_event("run-a"), zero_causal_state())
    for index in range(10):
        # State is requested before the completed human event is observed.
        assert np.array_equal(tracker.before_event("run-a"), zero_causal_state())
        tracker.observe_human("run-a", (index, -index, 1.0))
    state = tracker.before_event("run-a")
    np.testing.assert_allclose(state, np.asarray([2.25, -2.25, 0.5], np.float32))
    assert np.array_equal(tracker.before_event("run-b"), zero_causal_state())
