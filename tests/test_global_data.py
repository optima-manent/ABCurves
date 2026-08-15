from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from abcurves.global_data import (
    FULL_SESSION_SCHEMA,
    GLOBAL_RENDERER_DATASET_SCHEMA,
    DatasetPreparationError,
    FullSession,
    FullSessionCollection,
    assign_user_splits,
    load_portable_full_sessions,
    load_preparation_config,
    load_research_full_sessions,
    save_global_renderer_dataset,
    write_portable_full_sessions,
)
from abcurves.preprocessing import PortableEvent, write_portable_events
from tools.prepare_dataset import main as prepare_dataset_main
from training.train_renderer import _load_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final.json"
SPAN = 256 + 800


def full_session(user: str, session: str, *, windows: int = 2, tail: int = 17) -> FullSession:
    ticks = windows * SPAN + tail
    time = np.arange(ticks, dtype=np.int32)
    raw = np.column_stack(((time % 9) - 4, ((time // 7) % 7) - 3)).astype(np.float32)
    return FullSession(
        user_id=user,
        session_id=session,
        dxdy=raw,
        source_ref=f"memory/{session}.npy",
    )


def test_final_renderer_preparation_contract_is_presentation_based() -> None:
    config = load_preparation_config(CONFIG).renderer

    assert config.context_ticks == 256
    assert config.split_seed == "abcurves.continuous_v1.user_split"
    assert config.recurrent_warm_ticks == 128
    assert config.teacher_base_hysteresis == 1.0
    assert config.sampler == "frozen_epoch_view_sampler_v1"
    assert config.deterministic_algorithms is True
    assert config.future_ticks == 800
    assert config.stride_ticks == SPAN
    assert config.presentation_budget == 118_345
    assert config.prefix_loss_weight == 0.0
    assert config.checkpoint_selection == "sampled_full_session_texture"
    assert config.lateral_offset_penalty == 1.5


def test_user_split_reproduces_the_frozen_colon_hash_rule() -> None:
    sessions = [full_session(f"u-{letter}", f"s-{letter}", windows=1) for letter in "abcdef"]
    seed = "abcurves.continuous_v1.user_split"
    expected = min(
        (session.user_id for session in sessions),
        key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).hexdigest(),
    )
    roles = assign_user_splits(sessions, validation_fraction=1 / 6, split_seed=seed)
    assert {user for user, role in roles.items() if role == "val"} == {expected}


def test_portable_full_sessions_are_hash_bound_and_pickle_free(tmp_path: Path) -> None:
    manifest = write_portable_full_sessions(
        tmp_path / "portable",
        [full_session("u-a", "s-a", windows=1)],
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))

    assert record["schema"] == FULL_SESSION_SCHEMA
    assert record["dense_period_ns"] == 1_000_000
    array_path = manifest.parent / record["sessions"][0]["dxdy_npy"]
    assert hashlib.sha256(array_path.read_bytes()).hexdigest() == record["sessions"][0]["sha256"]
    loaded = load_portable_full_sessions(manifest.parent)
    assert loaded.sessions[0].dxdy.shape == (SPAN + 17, 2)
    assert np.array_equal(loaded.sessions[0].dxdy, full_session("u-a", "s-a", windows=1).dxdy)

    record["sessions"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(record), encoding="utf-8")
    try:
        load_portable_full_sessions(manifest)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:  # pragma: no cover - protects the integrity contract itself
        raise AssertionError("tampered portable array was accepted")


def test_preparation_rechecks_portable_array_after_loading(tmp_path: Path) -> None:
    manifest = write_portable_full_sessions(
        tmp_path / "portable",
        [full_session("u-a", "s-a", windows=1)],
    )
    loaded = load_portable_full_sessions(manifest)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    array_path = manifest.parent / record["sessions"][0]["dxdy_npy"]
    changed = np.load(array_path, mmap_mode="r+", allow_pickle=False)
    changed[0, 0] += 1
    changed.flush()
    del changed

    with pytest.raises(DatasetPreparationError, match="changed after validation"):
        save_global_renderer_dataset(
            tmp_path / "prepared",
            loaded,
            load_preparation_config(CONFIG).renderer,
            validation_fraction=0.0,
        )
    assert not (tmp_path / "prepared").exists()


def test_portable_sessions_cannot_alias_one_physical_array(tmp_path: Path) -> None:
    manifest = write_portable_full_sessions(
        tmp_path / "portable",
        [full_session("u-a", "s-a", windows=1)],
    )
    record = json.loads(manifest.read_text(encoding="utf-8"))
    alias = dict(record["sessions"][0])
    alias["user_id"] = "u-b"
    alias["session_id"] = "s-b"
    record["sessions"].append(alias)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="reuse one physical array"):
        load_portable_full_sessions(manifest)


def test_short_source_session_is_retained_without_inflating_split_counts(
    tmp_path: Path,
) -> None:
    seed = "short-session-test"
    probes = [
        full_session("u-a", "s-a", windows=1, tail=0),
        full_session("u-b", "s-b", windows=1, tail=0),
    ]
    roles = assign_user_splits(probes, validation_fraction=0.5, split_seed=seed)
    val_user = next(user for user, role in roles.items() if role == "val")
    train_user = next(user for user, role in roles.items() if role == "train")
    source = write_portable_full_sessions(
        tmp_path / "portable",
        [
            full_session(train_user, "s-train", windows=1, tail=0),
            FullSession(
                val_user,
                "s-short-val",
                np.zeros((512, 2), dtype=np.float32),
                "memory/s-short-val.npy",
            ),
        ],
    )
    output = tmp_path / "prepared"
    assert prepare_dataset_main(
        [
            str(source),
            str(output),
            "--config",
            str(CONFIG),
            "--branch",
            "renderer",
            "--validation-fraction",
            "0.5",
            "--renderer-split-seed",
            seed,
        ]
    ) == 0

    arrays, receipt = _load_split(output / "renderer_val", expected_split="val")
    assert arrays["prefix_raw_dxdy"].shape == (0, 256, 2)
    assert receipt["users"] == receipt["sessions"] == 0
    meta = json.loads((output / "renderer_val" / "meta.json").read_text("utf-8"))
    assert meta["source_users"] == meta["source_sessions"] == 1
    source_index = json.loads((output / "source_index.json").read_text("utf-8"))
    assert source_index["val"][0]["windows"] == 0


def test_cli_materializes_blind_nonoverlap_windows_and_user_split(tmp_path: Path) -> None:
    originals = {
        "s-a1": full_session("u-a", "s-a1"),
        "s-a2": full_session("u-a", "s-a2", windows=1, tail=31),
        "s-b1": full_session("u-b", "s-b1"),
    }
    source = write_portable_full_sessions(tmp_path / "portable", originals.values())
    output = tmp_path / "prepared"

    assert prepare_dataset_main(
        [
            str(source),
            str(output),
            "--config",
            str(CONFIG),
            "--branch",
            "renderer",
            "--validation-fraction",
            "0.5",
            "--renderer-split-seed",
            "unit-test-split",
        ]
    ) == 0

    split_users: dict[str, set[str]] = {}
    seen_sessions: set[str] = set()
    for split in ("train", "val"):
        split_root = output / f"renderer_{split}"
        prefix = np.load(split_root / "prefix_raw_dxdy.npy", mmap_mode="r", allow_pickle=False)
        future = np.load(split_root / "future_raw_dxdy.npy", mmap_mode="r", allow_pickle=False)
        meta = json.loads((split_root / "meta.json").read_text(encoding="utf-8"))
        assert meta["schema"] == GLOBAL_RENDERER_DATASET_SCHEMA
        assert meta["cohort"] == "renderer_global_full_session_v1"
        assert prefix.shape == (meta["windows"], 256, 2)
        assert future.shape == (meta["windows"], 800, 2)
        assert prefix.dtype == future.dtype == np.float32
        assert len(meta["user_id"]) == len(meta["session_id"]) == meta["windows"]
        split_users[split] = set(meta["user_id"])
        seen_sessions.update(meta["session_id"])
        for row, (session_id, start) in enumerate(
            zip(meta["session_id"], meta["window_start_tick"])
        ):
            raw = originals[session_id].dxdy
            assert np.array_equal(prefix[row], raw[start : start + 256])
            assert np.array_equal(future[row], raw[start + 256 : start + SPAN])
            assert start % SPAN == 0

    assert not (split_users["train"] & split_users["val"])
    assert seen_sessions == set(originals)
    source_index = json.loads((output / "source_index.json").read_text(encoding="utf-8"))
    assert source_index["source_hashes_verified"] is True
    assert {row["user_id"] for row in source_index["train"]}.isdisjoint(
        {row["user_id"] for row in source_index["val"]}
    )
    for row in source_index["train"] + source_index["val"]:
        assert row["dropped_incomplete_tail_ticks"] in {17, 31}
        assert not Path(row["source_ref"]).is_absolute()
    root_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    receipts = root_manifest["branches"]["renderer"]["receipts"]
    for name in ("build_report", "source_index"):
        path = output / receipts[name]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipts[name]["sha256"]
    for split in ("train", "val"):
        assert receipts["roles"][split] == json.loads(
            (output / "build_report.json").read_text(encoding="utf-8")
        )["roles"][split]["sha256"]
        meta = json.loads(
            (output / f"renderer_{split}" / "meta.json").read_text(encoding="utf-8")
        )
        assert list(zip(meta["session_id"], meta["user_id"], meta["window_start_tick"])) == sorted(
            zip(meta["session_id"], meta["user_id"], meta["window_start_tick"])
        )


def test_programmatic_collection_does_not_claim_unverified_sources(tmp_path: Path) -> None:
    session = full_session("u-a", "s-a", windows=1)
    collection = FullSessionCollection(
        sessions=(session,),
        source_kind="programmatic",
        source_manifest_ref="source/index",
        source_manifest_sha256="0" * 64,
    )
    config = load_preparation_config(CONFIG).renderer
    report = save_global_renderer_dataset(
        tmp_path / "prepared",
        collection,
        config,
        validation_fraction=0.0,
    )
    source = json.loads(
        (tmp_path / "prepared" / "source_index.json").read_text(encoding="utf-8")
    )
    assert report["all_source_hashes_verified"] is False
    assert source["source_hashes_verified"] is False


def test_prepare_cli_refuses_overwrite_without_touching_destination(tmp_path: Path) -> None:
    source = write_portable_full_sessions(
        tmp_path / "portable",
        [full_session("u-a", "s-a", windows=1)],
    )
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    assert prepare_dataset_main(
        [str(source), str(output), "--config", str(CONFIG), "--branch", "renderer"]
    ) == 2
    assert marker.read_text(encoding="utf-8") == "untouched"
    assert not list(tmp_path.glob(".prepared.stage-*"))


def test_event_only_npz_fails_closed_for_renderer_and_both(tmp_path: Path) -> None:
    raw = np.concatenate(
        [np.ones((90, 2), dtype=np.float32), np.zeros((20, 2), dtype=np.float32)]
    )
    source = write_portable_events(
        tmp_path / "events.npz",
        [
            PortableEvent(
                "trial-1",
                raw,
                np.asarray([100.0, 100.0]),
                10.0,
                user_id="u-1",
                session_id="s-1",
            )
        ],
    )
    for branch in ("renderer", "both"):
        output = tmp_path / f"prepared-{branch}"
        assert prepare_dataset_main(
            [str(source), str(output), "--config", str(CONFIG), "--branch", branch]
        ) == 2
        assert not output.exists()


def test_research_renderer_uses_entire_dense_grid_without_event_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    from abcurves import capture_exports

    export = tmp_path / "one-export"
    export.mkdir()
    (export / "export_manifest.json").write_text("{}", encoding="utf-8")
    payloads = {
        "mouse_1ms.csv": b"mouse",
        "trainer_events.csv": b"events",
    }
    for filename, payload in payloads.items():
        (export / filename).write_bytes(payload)
    manifest = {
        "dense_grid": {"period_ns": 1_000_000},
        "artifacts": [
            {
                "relative_path": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for filename, payload in payloads.items()
        ],
    }
    raw = full_session("u-1", "s-1", windows=2).dxdy
    events = pd.DataFrame(
        [
            {
                "user_id": "u-1",
                "session_id": "s-1",
                "natural_outcome": "miss_click",
                "technical_outcome": "interrupted",
            }
        ]
    )
    mouse = pd.DataFrame({"canonical_dx": raw[:, 0], "canonical_dy": raw[:, 1]})
    monkeypatch.setattr(capture_exports, "load_export_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        capture_exports,
        "load_export_tables",
        lambda _path: (manifest, events, mouse),
    )

    loaded = load_research_full_sessions(tmp_path)

    assert len(loaded.sessions) == 1
    assert len(loaded.sessions[0].dxdy) == len(raw)
    assert np.array_equal(loaded.sessions[0].dxdy, raw)


def test_research_loader_rechecks_sources_after_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    from abcurves import capture_exports

    export = tmp_path / "one-export"
    export.mkdir()
    (export / "export_manifest.json").write_text("{}", encoding="utf-8")
    payloads = {
        "mouse_1ms.csv": b"mouse",
        "trainer_events.csv": b"events",
    }
    for filename, payload in payloads.items():
        (export / filename).write_bytes(payload)
    manifest = {
        "dense_grid": {"period_ns": 1_000_000},
        "artifacts": [
            {
                "relative_path": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for filename, payload in payloads.items()
        ],
    }
    raw = full_session("u-1", "s-1", windows=1).dxdy
    events = pd.DataFrame([{"user_id": "u-1", "session_id": "s-1"}])
    mouse = pd.DataFrame({"canonical_dx": raw[:, 0], "canonical_dy": raw[:, 1]})

    def load_then_change(_path):
        (export / "mouse_1ms.csv").write_bytes(b"changed-after-validation")
        return manifest, events, mouse

    monkeypatch.setattr(capture_exports, "load_export_manifest", lambda _path: manifest)
    monkeypatch.setattr(capture_exports, "load_export_tables", load_then_change)

    with pytest.raises(DatasetPreparationError, match="changed while it was loaded"):
        load_research_full_sessions(tmp_path)
