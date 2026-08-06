from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from abcurves.preprocessing import (
    NativeCaptureExporterUnavailable,
    PORTABLE_EVENT_SCHEMA,
    PREPARED_CUT_SCHEMA,
    PlannerPreparationConfig,
    PortableEvent,
    RendererPreparationConfig,
    TinyTargetConfig,
    adaptive_planner_thresholds,
    load_portable_events,
    load_research_export_events,
    planner_shape_metrics,
    planner_shape_reasons,
    prepare_planner,
    prepare_renderer,
    save_prepared_dataset,
    subset_preparation_result,
    write_portable_events,
)
from abcurves.smoothing import smooth_dxdy
from tools.prepare_dataset import main as prepare_dataset_main
from training.train_planner import (
    planner_seam_contract,
    require_prepared_branch,
    source_trial_weight_summary,
)
from training.train_renderer import require_renderer_dataset


def straight_event(
    source_id: str,
    *,
    target_x: int = 100,
    radius: float = 10.0,
    stop_x: int = 95,
    quiet_ms: int = 25,
    split: str = "train",
) -> PortableEvent:
    raw = np.concatenate(
        [
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (stop_x, 1)),
            np.zeros((quiet_ms, 2), dtype=np.float32),
        ]
    )
    return PortableEvent(
        source_trial_id=source_id,
        dxdy=raw,
        target_rel_at_a=np.asarray([target_x, 0.0]),
        target_radius=radius,
        user_id=f"user-{source_id}",
        session_id=f"session-{source_id}",
        split=split,
    )


def test_adaptive_threshold_cap_is_b90_or_b92() -> None:
    config = PlannerPreparationConfig()
    short = adaptive_planner_thresholds(np.asarray([100.0, 0.0]), 10.0, config)
    long = adaptive_planner_thresholds(np.asarray([200.0, 0.0]), 10.0, config)

    assert len(short) == len(long) == 21
    assert short[0] == pytest.approx(0.78)
    assert short[-1] == pytest.approx(0.90)
    assert long[-1] == pytest.approx(0.92)


def test_renderer_does_not_inherit_planner_endpoint_filter() -> None:
    # The final point is inside the target but outside the Planner's inner
    # 75% cohort.  It remains valid Renderer texture supervision.
    event = straight_event("border", stop_x=91)
    planner = prepare_planner([event])
    renderer = prepare_renderer([event])

    assert "endpoint_not_inner" in planner.rejected[event.source_trial_id]
    assert len(renderer.cuts) == 1
    assert renderer.cuts[0].threshold == pytest.approx(0.80)
    assert renderer.cuts[0].source_trial_id == event.source_trial_id


def test_signature_at_one_native_cut_vetoes_physical_planner_event() -> None:
    raw = np.concatenate(
        [
            np.tile([1.0, 0.0], (72, 1)),
            np.tile([1.0, 1.0], (6, 1)),
            np.tile([1.0, -1.0], (12, 1)),
            np.tile([1.0, 1.0], (12, 1)),
            np.tile([0.0, -1.0], (6, 1)),
            np.zeros((24, 2)),
        ]
    ).astype(np.float32)
    event = PortableEvent("signature", raw, np.asarray([100.0, 0.0]), 10.0)
    metrics = planner_shape_metrics(raw, event.target_rel_at_a, event.target_radius, 72)

    assert "post_b_multiple_lateral_swings" in planner_shape_reasons(metrics)
    result = prepare_planner(
        [event], replace(PlannerPreparationConfig(), apply_quality_filter=False)
    )
    assert not result.cuts
    assert "event_veto:post_b_multiple_lateral_swings" in result.rejected["signature"]


def test_dense_and_tiny_rows_keep_one_total_weight_per_source() -> None:
    events = [
        straight_event("a", stop_x=99),
        straight_event("b", target_x=200, stop_x=199, radius=20.0),
    ]
    config = replace(
        PlannerPreparationConfig(),
        tiny_target=TinyTargetConfig(
            enabled=True,
            radius_quotas=((4.0, 10),),
        ),
    )
    result = prepare_planner(events, config)
    assert result.cuts

    for source_id in {cut.source_trial_id for cut in result.cuts}:
        source_rows = [cut for cut in result.cuts if cut.source_trial_id == source_id]
        assert sum(cut.example_weight for cut in source_rows) == pytest.approx(1.0)
        native = sum(not cut.synthetic for cut in source_rows)
        synthetic = sum(cut.synthetic for cut in source_rows)
        assert synthetic <= native
    assert result.source_weight_error() <= 1e-6


def test_non_train_planner_split_is_exact_runtime_b80() -> None:
    result = prepare_planner(
        [
            straight_event("train-row", split="train"),
            straight_event("val-row", split="val"),
        ]
    )
    train = subset_preparation_result(result, "train")
    val = subset_preparation_result(result, "val")

    assert len(train.cuts) > 1
    assert len(val.cuts) == 1
    assert val.cuts[0].threshold == pytest.approx(0.80)


def test_portable_roundtrip_and_self_contained_prepared_output(tmp_path: Path) -> None:
    events_path = write_portable_events(tmp_path / "events.npz", [straight_event("one")])
    loaded = load_portable_events(events_path)
    assert len(loaded) == 1
    assert loaded[0].source_trial_id == "one"
    with np.load(events_path, allow_pickle=False) as data:
        assert str(data["schema"]) == PORTABLE_EVENT_SCHEMA

    prepared, manifest = save_prepared_dataset(
        prepare_renderer(loaded, RendererPreparationConfig()),
        tmp_path / "renderer.npz",
    )
    assert manifest.is_file()
    with np.load(prepared, allow_pickle=False) as data:
        assert str(data["schema"]) == PREPARED_CUT_SCHEMA
        assert data["row_event_index"].tolist() == [0]
        assert data["split_index"].shape == (1,)
        assert data["prefix_raw_dxdy"].shape == (1, 160, 2)
        assert data["future_raw_dxdy"].shape == (1, 1000, 2)
        assert data["future_smooth_dxdy"].shape == (1, 1000, 2)
        split = int(data["split_index"][0])
        prefix = data["prefix_raw_dxdy"][0][data["prefix_mask"][0] > 0.5]
        future_len = int(data["future_mask"][0].sum())
        assert np.array_equal(prefix, loaded[0].dxdy[:split])
        assert np.array_equal(data["future_raw_dxdy"][0, :future_len], loaded[0].dxdy[split:])
        expected_smooth = smooth_dxdy(
            loaded[0].dxdy,
            spec="triangular_moving_average_path:window=5",
        )[split:]
        assert np.array_equal(data["future_smooth_dxdy"][0, :future_len], expected_smooth)
        assert data["event_weight"].tolist() == [1.0]
        assert data["source_trial_id"].astype(str).tolist() == ["one"]
        assert "causal_seam_contract_json" in data.files


def test_cli_writes_directly_trainable_split_filenames(tmp_path: Path) -> None:
    source = tmp_path / "events.npz"
    write_portable_events(
        source,
        [
            straight_event("train", split="train"),
            straight_event("val", split="val"),
        ],
    )
    output = tmp_path / "prepared"
    config = Path(__file__).resolve().parents[1] / "configs" / "final_v2.json"

    assert prepare_dataset_main(
        [str(source), str(output), "--config", str(config), "--branch", "both"]
    ) == 0
    for name in (
        "planner_train.npz",
        "planner_val.npz",
        "renderer_train.npz",
        "renderer_val.npz",
    ):
        with np.load(output / name, allow_pickle=False) as data:
            assert data["prefix_raw_dxdy"].ndim == 3
            assert data["future_raw_dxdy"].shape[1:] == (1000, 2)
    with np.load(output / "planner_train.npz", allow_pickle=False) as train_data:
        train = dict(train_data)
    with np.load(output / "planner_val.npz", allow_pickle=False) as val_data:
        val = dict(val_data)
    assert source_trial_weight_summary(train)["max_abs_weight_sum_error"] <= 1e-6
    contract = planner_seam_contract(train, val)
    assert contract is not None
    assert contract["trigger"]["thresholds"] == [0.8]


def test_trainers_reject_uncontracted_example_fixtures() -> None:
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "examples" / "aim_train.npz", allow_pickle=False) as data:
        fixture = dict(data)
    with pytest.raises(ValueError, match="tools/prepare_dataset.py"):
        require_prepared_branch(fixture, branch="planner", label="train dataset")
    with pytest.raises(ValueError, match="tools/prepare_dataset.py"):
        require_renderer_dataset(fixture)


def test_planner_requires_explicit_seam_metadata() -> None:
    arrays = {
        "schema": np.asarray(PREPARED_CUT_SCHEMA),
        "cohort": np.asarray("planner_train"),
    }
    with pytest.raises(ValueError, match="causal seam contract"):
        planner_seam_contract(arrays, arrays)


def test_raw_capture_zip_requires_real_exporter(tmp_path: Path) -> None:
    with pytest.raises(NativeCaptureExporterUnavailable, match="Capture exporter"):
        load_portable_events(tmp_path / "raw_capture.zip")


def test_research_export_root_uses_validated_profiler_and_user_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    from abcurves import capture_exports

    roots = [tmp_path / "session-a", tmp_path / "session-b"]
    manifests = {}
    for root in roots:
        root.mkdir()
        (root / "export_manifest.json").write_text("{}", encoding="utf-8")
        artifacts = []
        for filename, payload in (
            ("mouse_1ms.csv", b"mouse"),
            ("trainer_events.csv", b"events"),
        ):
            (root / filename).write_bytes(payload)
            artifacts.append(
                {
                    "relative_path": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifests[root.name] = {"artifacts": artifacts}

    raw = straight_event("template").dxdy

    def fake_manifest(path):
        return manifests[Path(path).name]

    def fake_profile(path, **_kwargs):
        name = Path(path).name
        suffix = name[-1]
        return (
            pd.DataFrame(
                [
                    {
                        "source_trial_id": f"session-{suffix}:1",
                        "rejection_reasons": "",
                        "dense_model_start_index": 0,
                        "dense_stop_index": len(raw),
                        "target_rel_at_A_x": 100.0,
                        "target_rel_at_A_y": 0.0,
                        "target_radius_counts": 10.0,
                        "natural_outcome": "hit_click",
                        "technical_outcome": "none",
                        "user_id": f"user-{suffix}",
                        "session_id": f"session-{suffix}",
                    }
                ]
            ),
            {},
        )

    def fake_tables(path):
        mouse = pd.DataFrame({"canonical_dx": raw[:, 0], "canonical_dy": raw[:, 1]})
        return manifests[Path(path).name], pd.DataFrame(), mouse

    monkeypatch.setattr(capture_exports, "load_export_manifest", fake_manifest)
    monkeypatch.setattr(capture_exports, "profile_export_session", fake_profile)
    monkeypatch.setattr(capture_exports, "load_export_tables", fake_tables)
    conversion = load_research_export_events(tmp_path, validation_fraction=0.5)

    assert conversion.profiled_event_count == 2
    assert len(conversion.events) == 2
    assert {event.split for event in conversion.events} == {"train", "val"}
    assert {event.user_id for event in conversion.events} == {"user-a", "user-b"}
