#!/usr/bin/env python3
"""Generate a portable human/generated Full49 bundle for the public judges.

Example:

    python tools/build_descriptor_bundle.py examples/aim_test.npz \
        results/local/example_descriptors.npz --rows 256

One row nearest B80 is selected per physical source. The shipped pipeline then
generates exactly one continuation for the same context. Human and generated
rows keep the same source/session/key binding so grouped and complete-key
holdouts cannot split siblings across fitting and query populations.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcurves import Pipeline  # noqa: E402
from abcurves.judges import (  # noqa: E402
    FULL_SYSTEM_FEATURE_NAMES,
    TEXTURE_FEATURE_NAMES,
    TRAJ_FEATURE_NAMES,
    full_system_features,
)
from evaluation.bundle import DescriptorBundle, write_descriptor_bundle  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_array(data: Any, name: str, rows: int, default: str) -> np.ndarray:
    if name not in data:
        return np.full(rows, default, dtype=f"<U{max(len(default), 1)}")
    values = np.asarray(data[name]).astype(str).reshape(-1)
    if len(values) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    return values


def _source_ids(data: Any, rows: int) -> np.ndarray:
    for name in ("source_trial_id", "source_id", "event_id"):
        if name in data:
            values = np.asarray(data[name]).astype(str).reshape(-1)
            if len(values) == rows:
                return values
    return np.asarray([f"row-{index:08d}" for index in range(rows)])


def _installation_keys(data: Any, rows: int) -> np.ndarray:
    for name in ("installation_key", "user_id"):
        if name in data:
            values = np.asarray(data[name]).astype(str).reshape(-1)
            if len(values) == rows:
                return values
    return np.full(rows, "public-example", dtype="<U14")


def _select_rows(
    source_ids: np.ndarray,
    keys: np.ndarray,
    progress: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Choose one B80-nearest row per source, balanced over keys."""

    best: dict[str, int] = {}
    for index, source in enumerate(source_ids.astype(str)):
        current = best.get(source)
        candidate = (abs(float(progress[index]) - 0.80), index)
        if current is None or candidate < (abs(float(progress[current]) - 0.80), current):
            best[source] = index
    buckets: dict[str, list[int]] = {}
    for index in best.values():
        buckets.setdefault(str(keys[index]), []).append(index)
    for key, indices in buckets.items():
        indices.sort(
            key=lambda index: hashlib.sha256(
                f"abcurves.descriptors|{key}|{source_ids[index]}".encode("utf-8")
            ).digest()
        )
    selected: list[int] = []
    key_order = sorted(buckets)
    while len(selected) < limit:
        advanced = False
        for key in key_order:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                advanced = True
                if len(selected) == limit:
                    break
        if not advanced:
            break
    return np.asarray(selected, dtype=np.int64)


def _event_seed(source_id: str, model_seed: int) -> int:
    digest = hashlib.sha256(
        f"abcurves.public-bundle|{model_seed}|{source_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def build_bundle(
    dataset: Path,
    *,
    rows: int,
    model_seed: int,
) -> tuple[DescriptorBundle, dict[str, Any]]:
    with np.load(dataset, allow_pickle=False) as data:
        required = {
            "prefix_raw_dxdy",
            "prefix_mask",
            "future_raw_dxdy",
            "future_mask",
            "target_rel_x_at_B",
            "target_rel_y_at_B",
            "target_radius",
            "progress",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"model dataset lacks required arrays: {missing}")
        total = len(data["future_mask"])
        sources_all = _source_ids(data, total)
        keys_all = _installation_keys(data, total)
        progress_all = np.asarray(data["progress"], dtype=np.float64).reshape(-1)
        selected = _select_rows(sources_all, keys_all, progress_all, min(rows, total))
        if len(selected) < 2:
            raise ValueError("at least two distinct physical sources are required")

        prefix = np.asarray(data["prefix_raw_dxdy"][selected], dtype=np.float32)
        prefix_mask = np.asarray(data["prefix_mask"][selected], dtype=np.float32)
        human = np.asarray(data["future_raw_dxdy"][selected], dtype=np.float32)
        human_mask = np.asarray(data["future_mask"][selected], dtype=np.float32)
        targets = np.column_stack(
            [data["target_rel_x_at_B"][selected], data["target_rel_y_at_B"][selected]]
        ).astype(np.float32)
        radii = np.asarray(data["target_radius"][selected], dtype=np.float32)
        progress = progress_all[selected].astype(np.float32)
        source_ids = sources_all[selected].astype(str)
        keys = keys_all[selected].astype(str)
        sessions_all = _text_array(data, "session_id", total, "")
        sessions = sessions_all[selected].astype(str)
        sessions = np.asarray(
            [value if value else f"{key}::public-example" for value, key in zip(sessions, keys)]
        )
        task_all = _text_array(data, "task_type", total, "public-example")
        tasks = task_all[selected].astype(str)

    horizon = human.shape[1]
    generated = np.zeros((len(selected), horizon, 2), dtype=np.float32)
    generated_mask = np.zeros((len(selected), horizon), dtype=np.float32)
    with Pipeline(model_seed=model_seed, prewarm=True) as pipeline:
        for local, source in enumerate(source_ids):
            valid_prefix = prefix[local][prefix_mask[local] > 0.5]
            seed = _event_seed(str(source), model_seed)
            continuation = pipeline.generate(
                valid_prefix,
                target_rel_at_B=targets[local],
                target_radius=float(radii[local]),
                progress_center=float(progress[local]),
                seed=seed,
            )
            take = min(len(continuation), horizon)
            generated[local, :take] = continuation[:take]
            generated_mask[local, :take] = 1.0

    movement = np.concatenate([human, generated], axis=0)
    movement_mask = np.concatenate([human_mask, generated_mask], axis=0)
    doubled_targets = np.concatenate([targets, targets], axis=0)
    doubled_radii = np.concatenate([radii, radii], axis=0)
    doubled_prefix = np.concatenate([prefix, prefix], axis=0)
    doubled_prefix_mask = np.concatenate([prefix_mask, prefix_mask], axis=0)
    features = full_system_features(
        movement,
        movement_mask,
        doubled_targets,
        doubled_radii,
        prefix_dxdy=doubled_prefix,
        prefix_mask=doubled_prefix_mask,
    )
    count = len(selected)
    origins = np.asarray(["human"] * count + ["generated"] * count)
    bundle = DescriptorBundle(
        features=features,
        origin=origins,
        installation_key=np.concatenate([keys, keys]),
        session_id=np.concatenate([sessions, sessions]),
        source_id=np.concatenate([source_ids, source_ids]),
        order=np.concatenate([np.arange(count), np.arange(count)]),
        task=np.concatenate([tasks, tasks]),
        feature_names=tuple(FULL_SYSTEM_FEATURE_NAMES),
        panel_slices={
            "trajectory": (0, len(TRAJ_FEATURE_NAMES)),
            "texture": (
                len(TRAJ_FEATURE_NAMES),
                len(TRAJ_FEATURE_NAMES) + len(TEXTURE_FEATURE_NAMES),
            ),
            "full": (0, len(FULL_SYSTEM_FEATURE_NAMES)),
        },
    )
    metadata = {
        "schema": "abcurves.public_descriptor_build.v1",
        "source_dataset": dataset.name,
        "source_dataset_sha256": _sha256(dataset),
        "model_seed": int(model_seed),
        "physical_sources": int(count),
        "human_rows": int(count),
        "generated_rows": int(count),
        "selection": "one B80-nearest row per source, key-balanced",
    }
    return bundle, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--model-seed", type=int, choices=(7, 23), default=7)
    args = parser.parse_args()
    if args.rows < 2:
        raise SystemExit("--rows must be at least two")
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    bundle, metadata = build_bundle(
        args.dataset,
        rows=int(args.rows),
        model_seed=int(args.model_seed),
    )
    destination = write_descriptor_bundle(args.output, bundle, metadata=metadata)
    print(
        f"wrote {destination} ({bundle.rows} rows, "
        f"{len(bundle.feature_names)} descriptors, sha256={_sha256(destination)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
