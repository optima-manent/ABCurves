"""Train the phase-free global Renderer to an exact presentation budget.

The input is a pair of directory splits written by tools/prepare_dataset.py.
Each row is a blind whole-session window:

    256 observed physical reports | 800 future physical reports

There is no event, A/B/C, success or target filter. The model sees one random
w3/w5 smoothing view whenever a source window is presented. Validation loss
is written as a diagnostic only; this program deliberately performs no
checkpoint selection. Promotion belongs to sampled, carried-state
full-session evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abcurves.renderer import (  # noqa: E402
    RendererConfig,
    save_count_model,
    teacher_forced_loss,
    train_count_texture_model,
)


SPLIT_SCHEMA = "abcurves.global_renderer_windows.v1"
SPLIT_COHORT = "renderer_global_full_session_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_paths() -> dict[str, Path]:
    """Return the source files whose bytes define this training run."""

    source_root = Path(__file__).resolve().parents[1]
    return {
        "trainer_sha256": Path(__file__).resolve(),
        "renderer_source_sha256": source_root / "abcurves" / "renderer.py",
        "smoothing_source_sha256": source_root / "abcurves" / "smoothing.py",
    }


def _hash_paths(paths: dict[str, Path], *, label: str) -> dict[str, str]:
    """Hash a named file set, turning custody failures into one clear error."""

    try:
        return {name: _sha256(path) for name, path in paths.items()}
    except OSError as exc:
        raise RuntimeError(f"cannot hash {label}: {exc}") from exc


def _require_hashes_unchanged(
    paths: dict[str, Path], expected: dict[str, str], *, label: str
) -> None:
    """Fail closed if any file differs from its pre-training digest."""

    observed = _hash_paths(paths, label=label)
    changed = sorted(
        name
        for name in set(paths) | set(expected)
        if observed.get(name) != expected.get(name)
    )
    if changed:
        raise RuntimeError(f"{label} changed during training: {', '.join(changed)}")


def _input_hash_bindings(
    train_root: Path,
    train_receipt: dict[str, Any],
    val_root: Path | None,
    val_receipt: dict[str, Any] | None,
    preparation_receipt: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, str]]:
    """Bind every mapped split file and its two preparation receipts."""

    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    roles: list[tuple[str, Path, dict[str, Any]]] = [
        ("train", train_root.resolve(), train_receipt)
    ]
    if val_root is not None and val_receipt is not None:
        roles.append(("val", val_root.resolve(), val_receipt))
    for role, root, receipt in roles:
        for filename, digest in receipt["files"].items():
            name = f"{role}/{filename}"
            paths[name] = root / filename
            expected[name] = str(digest)

    prepared_root = train_root.resolve().parent
    for filename, receipt_key in (
        ("build_report.json", "build_report_sha256"),
        ("source_index.json", "source_index_sha256"),
    ):
        paths[filename] = prepared_root / filename
        expected[filename] = str(preparation_receipt[receipt_key])
    return paths, expected


def _load_split(
    root: Path, *, expected_split: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = root.expanduser().resolve()
    meta_path = root / "meta.json"
    prefix_path = root / "prefix_raw_dxdy.npy"
    future_path = root / "future_raw_dxdy.npy"
    for path in (meta_path, prefix_path, future_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    split_paths = {
        "meta.json": meta_path,
        "prefix_raw_dxdy.npy": prefix_path,
        "future_raw_dxdy.npy": future_path,
    }
    split_hashes = _hash_paths(split_paths, label=f"Renderer split {root}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != SPLIT_SCHEMA or meta.get("cohort") != SPLIT_COHORT:
        raise ValueError(f"{meta_path}: unsupported global Renderer split")
    split = meta.get("split")
    if split not in {"train", "val"} or (
        expected_split is not None and split != expected_split
    ):
        raise ValueError(f"{meta_path}: split role differs from {expected_split!r}")
    if int(meta.get("prefix", -1)) != 256 or int(meta.get("future", -1)) != 800:
        raise ValueError(f"{meta_path}: expected the frozen 256|800 window")
    if int(meta.get("stride", -1)) != 1_056:
        raise ValueError(f"{meta_path}: expected stride 1056")
    prefix = np.load(prefix_path, mmap_mode="r", allow_pickle=False)
    future = np.load(future_path, mmap_mode="r", allow_pickle=False)
    if prefix.dtype != np.float32 or future.dtype != np.float32:
        raise ValueError("prepared Renderer arrays must be float32")
    if prefix.shape != (len(prefix), 256, 2):
        raise ValueError(f"{prefix_path}: expected [N,256,2], got {prefix.shape}")
    if future.shape != (len(prefix), 800, 2):
        raise ValueError(f"{future_path}: expected [N,800,2], got {future.shape}")
    if int(meta.get("windows", -1)) != len(prefix):
        raise ValueError(f"{meta_path}: window count does not bind the arrays")
    row_fields = ("full_session_id", "session_id", "user_id", "window_start_tick")
    for field in row_fields:
        values = meta.get(field)
        if not isinstance(values, list) or len(values) != len(prefix):
            raise ValueError(f"{meta_path}: {field} must contain one value per row")
    full_ids = meta["full_session_id"]
    session_ids = meta["session_id"]
    user_ids = meta["user_id"]
    starts = meta["window_start_tick"]
    if full_ids != session_ids:
        raise ValueError(f"{meta_path}: full_session_id and session_id differ")
    if not all(isinstance(value, str) and value for value in session_ids + user_ids):
        raise ValueError(f"{meta_path}: user/session identifiers must be nonempty strings")
    if not all(type(value) is int and value >= 0 and value % 1_056 == 0 for value in starts):
        raise ValueError(f"{meta_path}: window starts must be nonnegative multiples of 1056")
    identities = list(zip(session_ids, user_ids, starts))
    if identities != sorted(identities):
        raise ValueError(f"{meta_path}: rows must use frozen (session,user,start) order")
    session_users: dict[str, str] = {}
    for session_id, user_id in zip(session_ids, user_ids):
        previous = session_users.setdefault(session_id, user_id)
        if previous != user_id:
            raise ValueError(f"{meta_path}: one session maps to multiple users")
    if int(meta.get("users", -1)) != len(set(user_ids)) or int(
        meta.get("sessions", -1)
    ) != len(set(session_ids)):
        raise ValueError(f"{meta_path}: user/session counts differ from row metadata")
    for label, array in (("prefix", prefix), ("future", future)):
        for start in range(0, len(array), 1024):
            chunk = np.asarray(array[start : start + 1024])
            if not np.isfinite(chunk).all() or not np.equal(chunk, np.rint(chunk)).all():
                raise ValueError(f"{root}: {label} reports must be finite integers")
            if len(chunk) and (
                float(np.min(chunk)) < -32768.0 or float(np.max(chunk)) > 32767.0
            ):
                raise ValueError(f"{root}: {label} reports exceed signed int16 range")
    _require_hashes_unchanged(
        split_paths,
        split_hashes,
        label=f"Renderer split {root}",
    )
    receipt = {
        "schema": str(meta["schema"]),
        "windows": int(len(prefix)),
        "users": int(meta.get("users", len(set(meta.get("user_id", []))))),
        "sessions": int(meta.get("sessions", len(set(meta.get("session_id", []))))),
        "files": {
            "prefix_raw_dxdy.npy": split_hashes["prefix_raw_dxdy.npy"],
            "future_raw_dxdy.npy": split_hashes["future_raw_dxdy.npy"],
            "meta.json": split_hashes["meta.json"],
        },
    }
    return {
        "prefix_raw_dxdy": prefix,
        "future_raw_dxdy": future,
    }, receipt


def _bind_preparation_receipts(
    train_root: Path,
    train_receipt: dict[str, Any],
    val_root: Path | None,
    val_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify prepared arrays against their source-index/build hash chain."""

    parent = train_root.resolve().parent
    if val_root is not None and val_root.resolve().parent != parent:
        raise ValueError("train and validation Renderer splits must share one prepared root")
    build_path = parent / "build_report.json"
    source_path = parent / "source_index.json"
    if not build_path.is_file() or not source_path.is_file():
        raise ValueError("Renderer splits require sibling build_report.json and source_index.json")
    receipt_paths = {
        "build_report.json": build_path,
        "source_index.json": source_path,
    }
    receipt_hashes = _hash_paths(
        receipt_paths, label="Renderer preparation receipts"
    )
    build = json.loads(build_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if build.get("schema") != "abcurves.global_renderer_build.v1":
        raise ValueError("unsupported Renderer build-report schema")
    if source.get("schema") != "abcurves.global_renderer_sources.v1":
        raise ValueError("unsupported Renderer source-index schema")
    source_sha = receipt_hashes["source_index.json"]
    if build.get("source_index_sha256") != source_sha:
        raise ValueError("Renderer build report does not bind source_index.json")
    if not bool(build.get("all_source_hashes_verified")) or not bool(
        source.get("source_hashes_verified")
    ):
        raise ValueError("Renderer source hashes were not authenticated during preparation")
    config = build.get("config", {})
    expected = {
        "context_ticks": 256,
        "future_ticks": 800,
        "stride_ticks": 1_056,
        "span_ticks": 1_056,
        "drop_incomplete_tail": True,
        "presentation_budget": 118_345,
        "recurrent_warm_ticks": 128,
        "teacher_base_hysteresis": 1.0,
        "sampler": "frozen_epoch_view_sampler_v1",
        "deterministic_algorithms": True,
        "teacher_smoothing_specs": [
            "triangular_moving_average_path:window=3",
            "triangular_moving_average_path:window=5",
        ],
        "sample_weighting": "natural",
        "prefix_loss_weight": 0.0,
        "checkpoint_selection": "sampled_full_session_texture",
        "lateral_offset_penalty": 1.5,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Renderer build report differs from the frozen training contract")
    receipts = {"train": (train_root, train_receipt)}
    if val_root is not None and val_receipt is not None:
        receipts["val"] = (val_root, val_receipt)
    for role, (root, receipt) in receipts.items():
        bound = build.get("roles", {}).get(role, {})
        if bound.get("directory") != root.resolve().name:
            raise ValueError(f"Renderer {role} directory differs from build report")
        if bound.get("sha256") != receipt["files"]:
            raise ValueError(f"Renderer {role} hashes differ from build report")
    _require_hashes_unchanged(
        receipt_paths,
        receipt_hashes,
        label="Renderer preparation receipts",
    )
    return {
        "build_report_sha256": receipt_hashes["build_report.json"],
        "source_index_sha256": source_sha,
        "all_source_hashes_verified": True,
    }


def _identity_set(meta_path: Path, key: str) -> set[str]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    values = meta.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{meta_path}: {key} must be a string list")
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/renderer_p118345.pt"))
    parser.add_argument("--presentations", type=int, default=118_345)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--validation-presentations",
        type=int,
        default=0,
        help="0 evaluates every validation window; ignored without --val",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing checkpoint: {args.out}")
    if args.presentations < 1 or args.batch_size < 1:
        raise SystemExit("--presentations and --batch-size must be positive")
    if args.validation_presentations < 0:
        raise SystemExit("--validation-presentations must be nonnegative")

    # Hash the implementation immediately after argument validation.  The
    # imported functions are already resident in this process, so hashing only
    # after a long fit could bind a concurrently edited file rather than the
    # implementation that actually produced the weights.
    implementation_paths = _implementation_paths()
    implementation_hashes = _hash_paths(
        implementation_paths, label="Renderer training implementation"
    )

    train, train_receipt = _load_split(args.train, expected_split="train")
    validation = None
    validation_receipt = None
    validation_skip_reason = None
    if args.val is not None:
        validation, validation_receipt = _load_split(args.val, expected_split="val")
        train_users = _identity_set(args.train / "meta.json", "user_id")
        val_users = _identity_set(args.val / "meta.json", "user_id")
        overlap = train_users & val_users
        if overlap:
            raise SystemExit(
                f"whole-user isolation failed: {len(overlap)} identities cross splits"
            )
        train_sessions = _identity_set(args.train / "meta.json", "session_id")
        val_sessions = _identity_set(args.val / "meta.json", "session_id")
        if train_sessions & val_sessions:
            raise SystemExit("whole-session isolation failed: sessions cross splits")
        if len(validation["prefix_raw_dxdy"]) == 0:
            validation = None
            validation_skip_reason = (
                "the supplied validation split has zero windows; training proceeds "
                "as the documented train-only case"
            )
            print(f"warning: {validation_skip_reason}", file=sys.stderr)

    preparation_receipt = _bind_preparation_receipts(
        args.train, train_receipt, args.val, validation_receipt
    )
    input_paths, input_hashes = _input_hash_bindings(
        args.train,
        train_receipt,
        args.val,
        validation_receipt,
        preparation_receipt,
    )
    _require_hashes_unchanged(
        input_paths, input_hashes, label="Renderer training input"
    )
    _require_hashes_unchanged(
        implementation_paths,
        implementation_hashes,
        label="Renderer training implementation",
    )

    config = RendererConfig(
        presentation_budget=args.presentations,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    started = time.time()
    model, report = train_count_texture_model(
        train,
        config,
        device=args.device,
        log_every=args.log_every,
    )
    report["data"] = {
        "train": train_receipt,
        "validation": validation_receipt,
        "preparation": preparation_receipt,
    }
    report["environment"] = {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "device": str(args.device),
        **implementation_hashes,
    }
    if validation is not None:
        limit = None if args.validation_presentations == 0 else args.validation_presentations
        report["teacher_forced_validation"] = teacher_forced_loss(
            model,
            validation,
            device=args.device,
            max_presentations=limit,
        )
    elif validation_skip_reason is not None:
        report["teacher_forced_validation"] = {
            "performed": False,
            "reason": validation_skip_reason,
        }
    report["elapsed_seconds"] = float(time.time() - started)
    report["deployment_note"] = (
        "This is a float research checkpoint. The shipped 44,484-byte artifact "
        "is the separately authenticated post-training-quantized promotion."
    )
    # The arrays are memory-mapped and a full fit can run for hours.  Rehash
    # both data and source code immediately before publication so the
    # checkpoint can never claim receipts for bytes that changed mid-run.
    _require_hashes_unchanged(
        input_paths, input_hashes, label="Renderer training input"
    )
    _require_hashes_unchanged(
        implementation_paths,
        implementation_hashes,
        label="Renderer training implementation",
    )
    report["integrity_recheck"] = {
        "schema": "abcurves.renderer_training_integrity.v1",
        "pretraining_hashes_verified": True,
        "posttraining_input_hashes_unchanged": True,
        "posttraining_implementation_hashes_unchanged": True,
    }
    try:
        save_count_model(model, report, args.out, overwrite=False)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"wrote {args.out} at {report['presentations']} presentations "
        f"({report['optimizer_steps']} optimizer steps, {report['n_parameters']} scalars)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
