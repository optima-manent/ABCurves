from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import training.train_renderer as train_renderer_module

from abcurves import Pipeline
from abcurves.global_data import (
    FullSession,
    assign_user_splits,
    write_portable_full_sessions,
)
from abcurves.renderer import (
    BASE_FEATURE_NAMES,
    REGIME_FEATURE_NAMES,
    CountTextureModel,
    RendererConfig,
    _epoch_view_schedule,
    build_teacher_example,
    load_count_model,
    sample_count_streams,
    save_count_model,
    teacher_forced_loss,
    train_count_texture_model,
)
from tools.prepare_dataset import main as prepare_dataset_main
from training.train_renderer import _load_split
from training.train_renderer import main as train_renderer_main


ROOT = Path(__file__).resolve().parents[1]


def _active_tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def test_final_float_architecture_has_one_recurrent_weight_set() -> None:
    config = RendererConfig()
    assert config.context_ticks == 256
    assert config.recurrent_warm_ticks == 128
    assert config.teacher_base_hysteresis == 1.0
    assert config.base_hysteresis == 0.5
    model = CountTextureModel()
    assert len(BASE_FEATURE_NAMES) == 15
    assert len(REGIME_FEATURE_NAMES) == 5
    assert model.n_features == 20
    assert model.config.hidden == 80
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_362
    assert not any(name.startswith("cell.") for name in model.state_dict())


def test_seed7_active_initialization_is_frozen() -> None:
    torch.manual_seed(7)
    model = CountTextureModel()
    assert _active_tensor_sha256(model.state_dict()) == (
        "f68cfa830584d5643533165045705b420a639739f8ffecbf33ef1524dbd7ad26"
    )


def test_manual_step_matches_registered_gru() -> None:
    torch.manual_seed(4)
    model = CountTextureModel()
    features = torch.randn(2, 9, model.n_features)
    expected = model(features)["hidden"][0]
    hidden = torch.zeros(2, model.config.hidden)
    for tick in range(features.shape[1]):
        hidden = model.step_cell(features[:, tick], hidden)
    torch.testing.assert_close(hidden, expected, atol=2e-7, rtol=2e-7)


def test_epoch_teacher_schedule_matches_the_frozen_sampler() -> None:
    order, views = _epoch_view_schedule(17, 2, 7, 1)
    generator = np.random.default_rng([7, 1])
    expected_choices = generator.integers(0, 2, size=17, dtype=np.int64)
    expected_order = generator.permutation(17)
    assert np.array_equal(order, expected_order)
    assert np.array_equal(views, expected_choices[expected_order])


def test_training_does_not_mix_two_epochs_in_one_optimizer_batch() -> None:
    arrays = {
        "prefix_raw_dxdy": np.zeros((3, 256, 2), dtype=np.float32),
        "future_raw_dxdy": np.zeros((3, 800, 2), dtype=np.float32),
    }
    _, report = train_count_texture_model(
        arrays,
        RendererConfig(presentation_budget=4, batch_size=2),
        device="cpu",
        log_every=0,
    )
    assert report["optimizer_steps"] == 3  # 2 + 1, then 1 from the next epoch
    assert report["completed_full_passes"] == 1
    assert report["next_epoch_offset"] == 1


def test_teacher_is_phase_free_and_future_only() -> None:
    raw = np.zeros((1_056, 2), dtype=np.float32)
    raw[300:320, 0] = 1
    row = build_teacher_example(
        raw,
        raw,
        reset_idx=256,
        offset_radius=5,
        base_hysteresis=0.5,
    )
    assert row["features"].shape == (1_056, 20)
    assert not np.any(row["loss_mask"][:256])
    assert np.all(row["loss_mask"][256:] == 1)


def test_af15_is_the_sampler_default() -> None:
    model = CountTextureModel(RendererConfig())
    context = np.zeros((1, 256, 2), dtype=np.float32)
    smooth = np.zeros((1, 8, 2), dtype=np.float32)
    smooth[0, :4, 0] = 0.25
    mask = np.ones((1, 8), dtype=np.float32)
    arrays = {"prefix_raw_dxdy": context}
    implicit = sample_count_streams(
        model,
        arrays,
        smooth,
        mask,
        spec_key="triangular_moving_average_path:window=5",
        seed=18,
    )
    explicit = sample_count_streams(
        model,
        arrays,
        smooth,
        mask,
        spec_key="triangular_moving_average_path:window=5",
        seed=18,
        lateral_offset_penalty=1.5,
    )
    assert np.array_equal(implicit, explicit)


def test_float_sampler_rejects_noncanonical_context_lengths() -> None:
    model = CountTextureModel(RendererConfig())
    smooth = np.zeros((1, 8, 2), dtype=np.float32)
    mask = np.ones((1, 8), dtype=np.float32)
    for ticks in (255, 257):
        with pytest.raises(ValueError, match="prefix_raw_dxdy must have shape"):
            sample_count_streams(
                model,
                {"prefix_raw_dxdy": np.zeros((1, ticks, 2), dtype=np.float32)},
                smooth,
                mask,
                spec_key="triangular_moving_average_path:window=5",
            )


def test_teacher_forced_diagnostic_is_batch_partition_invariant() -> None:
    generator = np.random.default_rng(8)
    arrays = {
        "prefix_raw_dxdy": generator.integers(-2, 3, (5, 256, 2)).astype(np.float32),
        "future_raw_dxdy": generator.integers(-2, 3, (5, 800, 2)).astype(np.float32),
    }
    torch.manual_seed(9)
    model_two = CountTextureModel(RendererConfig(batch_size=2))
    model_three = CountTextureModel(RendererConfig(batch_size=3))
    model_three.load_state_dict(model_two.state_dict())
    first = teacher_forced_loss(model_two, arrays)
    second = teacher_forced_loss(model_three, arrays)
    assert first["loss"] == pytest.approx(second["loss"], abs=1e-6)
    assert first["emit_bce"] == pytest.approx(second["emit_bce"], abs=1e-6)
    assert first["offset_ce"] == pytest.approx(second["offset_ce"], abs=1e-6)


def _write_training_split(root, value: float = 0.0) -> None:
    root.mkdir()
    prefix = np.full((1, 256, 2), value, dtype=np.float32)
    future = np.zeros((1, 800, 2), dtype=np.float32)
    np.save(root / "prefix_raw_dxdy.npy", prefix, allow_pickle=False)
    np.save(root / "future_raw_dxdy.npy", future, allow_pickle=False)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "schema": "abcurves.global_renderer_windows.v1",
                "cohort": "renderer_global_full_session_v1",
                "split": "train",
                "prefix": 256,
                "future": 800,
                "stride": 1056,
                "windows": 1,
                "users": 1,
                "sessions": 1,
                "full_session_id": ["s-1"],
                "session_id": ["s-1"],
                "user_id": ["u-1"],
                "window_start_tick": [0],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("bad", [float("nan"), 0.5, 32768.0])
def test_training_loader_rejects_nonphysical_reports(tmp_path, bad: float) -> None:
    root = tmp_path / "renderer_train"
    _write_training_split(root, bad)
    with pytest.raises(ValueError, match="reports"):
        _load_split(root, expected_split="train")


def test_training_loader_requires_row_identity_contract(tmp_path) -> None:
    root = tmp_path / "renderer_train"
    _write_training_split(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    del meta["user_id"]
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="user_id"):
        _load_split(root, expected_split="train")


def _prepare_two_user_renderer(tmp_path: Path) -> Path:
    raw = np.zeros((1_056, 2), dtype=np.float32)
    source = write_portable_full_sessions(
        tmp_path / "source",
        [
            FullSession("u-a", "s-a", raw, "memory/s-a.npy"),
            FullSession("u-b", "s-b", raw, "memory/s-b.npy"),
        ],
    )
    prepared = tmp_path / "prepared"
    assert prepare_dataset_main(
        [
            str(source),
            str(prepared),
            "--config",
            str(ROOT / "configs" / "final.json"),
            "--branch",
            "renderer",
            "--validation-fraction",
            "0.5",
        ]
    ) == 0
    return prepared


def _prepare_renderer_with_empty_validation(tmp_path: Path) -> Path:
    seed = "empty-validation-test"
    probe = [
        FullSession("u-a", "s-a", np.zeros((1_056, 2), np.float32), "memory/a.npy"),
        FullSession("u-b", "s-b", np.zeros((1_056, 2), np.float32), "memory/b.npy"),
    ]
    roles = assign_user_splits(probe, validation_fraction=0.5, split_seed=seed)
    val_user = next(user for user, role in roles.items() if role == "val")
    train_user = next(user for user, role in roles.items() if role == "train")
    source = write_portable_full_sessions(
        tmp_path / "source",
        [
            FullSession(
                train_user,
                "s-train",
                np.zeros((1_056, 2), np.float32),
                "memory/train.npy",
            ),
            FullSession(
                val_user,
                "s-short-val",
                np.zeros((512, 2), np.float32),
                "memory/short.npy",
            ),
        ],
    )
    prepared = tmp_path / "prepared"
    assert prepare_dataset_main(
        [
            str(source),
            str(prepared),
            "--config",
            str(ROOT / "configs" / "final.json"),
            "--branch",
            "renderer",
            "--validation-fraction",
            "0.5",
            "--renderer-split-seed",
            seed,
        ]
    ) == 0
    return prepared


def _renderer_train_argv(prepared: Path, checkpoint: Path) -> list[str]:
    return [
        "train_renderer.py",
        "--train",
        str(prepared / "renderer_train"),
        "--val",
        str(prepared / "renderer_val"),
        "--out",
        str(checkpoint),
        "--presentations",
        "1",
        "--batch-size",
        "1",
        "--validation-presentations",
        "1",
        "--log-every",
        "0",
        "--device",
        "cpu",
    ]


def test_prepare_train_checkpoint_is_weights_only_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_two_user_renderer(tmp_path)
    checkpoint = tmp_path / "renderer.pt"
    monkeypatch.setattr(sys, "argv", _renderer_train_argv(prepared, checkpoint))
    assert train_renderer_main() == 0

    model, report = load_count_model(checkpoint)
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_362
    assert report["presentations"] == 1
    assert report["data"]["preparation"]["all_source_hashes_verified"] is True

    environment = report["environment"]
    assert all(
        isinstance(environment[name], str)
        for name in ("python", "numpy", "torch", "device")
    )
    assert environment["cuda"] is None or isinstance(environment["cuda"], str)
    assert environment["cudnn"] is None or type(environment["cudnn"]) is int
    sources = {
        "trainer_sha256": ROOT / "training" / "train_renderer.py",
        "renderer_source_sha256": ROOT / "abcurves" / "renderer.py",
        "smoothing_source_sha256": ROOT / "abcurves" / "smoothing.py",
    }
    for field, path in sources.items():
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert environment[field] == expected
    assert report["integrity_recheck"] == {
        "schema": "abcurves.renderer_training_integrity.v1",
        "pretraining_hashes_verified": True,
        "posttraining_input_hashes_unchanged": True,
        "posttraining_implementation_hashes_unchanged": True,
    }

    prefix = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (64, 1))
    context = np.zeros((256, 2), dtype=np.int16)
    context[-64:] = prefix.astype(np.int16)
    with Pipeline(
        float_renderer_checkpoint=checkpoint,
        prewarm=False,
    ) as pipeline:
        profile = pipeline.prepare_renderer_profile(context)
        first = pipeline.generate(
            prefix,
            renderer_profile=profile,
            target_rel_at_B=(36.0, 4.0),
            target_radius=8.0,
            progress_center=0.6,
            seed=19,
        )
        second = pipeline.generate(
            prefix,
            renderer_profile=profile,
            target_rel_at_B=(36.0, 4.0),
            target_radius=8.0,
            progress_center=0.6,
            seed=19,
        )
        exact = pipeline.generate(
            prefix,
            renderer_context_raw_dxdy=context,
            target_rel_at_B=(36.0, 4.0),
            target_radius=8.0,
            progress_center=0.6,
            seed=19,
        )
        assert pipeline.renderer_receipt["backend"] == "pytorch_float_checkpoint"
        assert pipeline.renderer_receipt["native_online_handoff"] is False
    assert first.dtype == np.int16 and first.shape[1] == 2 and len(first) > 0
    assert np.array_equal(first, second)
    assert np.array_equal(first, exact)


def test_train_cli_treats_zero_window_validation_as_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_renderer_with_empty_validation(tmp_path)
    checkpoint = tmp_path / "renderer.pt"
    monkeypatch.setattr(sys, "argv", _renderer_train_argv(prepared, checkpoint))
    assert train_renderer_main() == 0
    _, report = load_count_model(checkpoint)
    assert report["teacher_forced_validation"]["performed"] is False
    assert "zero windows" in report["teacher_forced_validation"]["reason"]


def test_float_checkpoint_exclusive_publication_preserves_existing_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "renderer.pt"
    destination.write_bytes(b"another process won")
    with pytest.raises(FileExistsError, match="refusing to overwrite checkpoint"):
        save_count_model(
            CountTextureModel(),
            {"schema": "test-only"},
            destination,
            overwrite=False,
        )
    assert destination.read_bytes() == b"another process won"
    assert not list(tmp_path.glob(".renderer.pt.*.tmp"))


@pytest.mark.parametrize("changed_kind", ["input", "implementation"])
def test_train_cli_rejects_midrun_hash_changes_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_kind: str,
) -> None:
    prepared = _prepare_two_user_renderer(tmp_path)
    checkpoint = tmp_path / "must_not_exist.pt"
    implementation_files: dict[str, Path] | None = None
    if changed_kind == "implementation":
        implementation_files = {
            name: tmp_path / f"{name}.py"
            for name in (
                "trainer_sha256",
                "renderer_source_sha256",
                "smoothing_source_sha256",
            )
        }
        for name, path in implementation_files.items():
            path.write_text(f"# stable {name}\n", encoding="utf-8")
        monkeypatch.setattr(
            train_renderer_module,
            "_implementation_paths",
            lambda: dict(implementation_files or {}),
        )

    original_train = train_renderer_module.train_count_texture_model

    def train_then_change(*args, **kwargs):
        result = original_train(*args, **kwargs)
        if changed_kind == "input":
            meta = prepared / "renderer_train" / "meta.json"
            meta.write_text(meta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        else:
            assert implementation_files is not None
            implementation_files["renderer_source_sha256"].write_text(
                "# changed during training\n", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(
        train_renderer_module, "train_count_texture_model", train_then_change
    )
    monkeypatch.setattr(sys, "argv", _renderer_train_argv(prepared, checkpoint))
    with pytest.raises(RuntimeError, match=f"Renderer training {changed_kind} changed"):
        train_renderer_main()
    assert not checkpoint.exists()
