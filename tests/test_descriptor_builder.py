from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tools.build_descriptor_bundle as builder
from tools.build_descriptor_bundle import _event_seeds


def test_event_seed_domain_selects_a_deterministic_full_pipeline_cell() -> None:
    draw_a = _event_seeds("source-017", 7, "draw-a")
    draw_b = _event_seeds("source-017", 7, "draw-b")

    assert draw_a == (4577226119188579884, 3004856549124330961)
    assert draw_b == (16050897429179582303, 7642141866037911993)
    assert _event_seeds("source-017", 7, "draw-a") == draw_a
    assert draw_a[0] != draw_b[0]
    assert draw_a[1] != draw_b[1]
    assert draw_a[0] != draw_a[1]


def test_builder_uses_both_cell_seeds_and_records_artifact_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "events.npz"
    np.savez(
        dataset,
        prefix_raw_dxdy=np.zeros((2, 160, 2), dtype=np.float32),
        prefix_mask=np.ones((2, 160), dtype=np.float32),
        future_raw_dxdy=np.zeros((2, 8, 2), dtype=np.float32),
        future_mask=np.ones((2, 8), dtype=np.float32),
        target_rel_x_at_B=np.asarray([20.0, 30.0], dtype=np.float32),
        target_rel_y_at_B=np.asarray([5.0, 7.0], dtype=np.float32),
        target_radius=np.asarray([8.0, 9.0], dtype=np.float32),
        progress=np.asarray([0.8, 0.8], dtype=np.float32),
        source_trial_id=np.asarray(["source-a", "source-b"]),
        renderer_context_raw_dxdy=np.zeros((2, 256, 2), dtype=np.int16),
    )
    planner_artifact = tmp_path / "planner_seed7.pt"
    planner_artifact.write_bytes(b"authenticated planner fixture")
    observed: list[tuple[int, int]] = []

    class _Rendered:
        @staticmethod
        def render_remaining() -> np.ndarray:
            return np.zeros((8, 2), dtype=np.float32)

    class _Pending:
        def finish(self, **kwargs):
            observed.append(
                (int(kwargs["planner_seed"]), int(kwargs["renderer_event_seed_u64"]))
            )
            return _Rendered()

    class _Pipeline:
        def __init__(self, *args, **kwargs) -> None:
            self.model_files = SimpleNamespace(planner=planner_artifact)
            self.renderer_receipt = {
                "backend": "test",
                "artifact_sha256": "0" * 64,
            }

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        @staticmethod
        def begin_at_b(*args, **kwargs):
            return _Pending()

    monkeypatch.setattr(builder, "Pipeline", _Pipeline)
    monkeypatch.setattr(
        builder,
        "full_system_features",
        lambda *args, **kwargs: np.zeros(
            (4, len(builder.FULL_SYSTEM_FEATURE_NAMES)), dtype=np.float32
        ),
    )

    _, first_metadata = builder.build_bundle(
        dataset, rows=2, model_seed=7, event_seed_domain="draw-a"
    )
    first = tuple(observed)
    observed.clear()
    builder.build_bundle(dataset, rows=2, model_seed=7, event_seed_domain="draw-a")
    repeated = tuple(observed)
    observed.clear()
    builder.build_bundle(dataset, rows=2, model_seed=7, event_seed_domain="draw-b")
    second = tuple(observed)

    assert repeated == first
    assert all(left[0] != right[0] for left, right in zip(first, second))
    assert all(left[1] != right[1] for left, right in zip(first, second))
    assert first_metadata["event_seed_derivation"]["planner_input"] == (
        "abcurves.public-bundle|planner:draw-a|7|<source_id>"
    )
    assert first_metadata["event_seed_derivation"]["renderer_input"] == (
        "abcurves.public-bundle|renderer:draw-a|7|<source_id>"
    )
    assert first_metadata["planner_artifact"] == "planner_seed7.pt"
    assert first_metadata["planner_artifact_sha256"] == hashlib.sha256(
        planner_artifact.read_bytes()
    ).hexdigest()
