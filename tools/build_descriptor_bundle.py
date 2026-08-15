#!/usr/bin/env python3
"""Generate a portable human/generated Full49 bundle for the public judges.

Example:

    python tools/build_descriptor_bundle.py examples/aim_test.npz \
        results/local/example_descriptors.npz --rows 256 --assume-quiet-preroll

One row nearest B80 is selected per physical source. The final pipeline then
generates exactly one continuation using an explicit 256-report Renderer
context. Human and generated
rows keep the same source/session/key binding so grouped and complete-key
holdouts cannot split siblings across fitting and query populations.

``--event-seed-domain`` selects a deterministic full-pipeline draw cell: it
changes both the Planner head draw and the Renderer sampling draw.
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


def _event_seed(source_id: str, model_seed: int, domain: str) -> int:
    digest = hashlib.sha256(
        f"abcurves.public-bundle|{domain}|{model_seed}|{source_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _event_seeds(
    source_id: str, model_seed: int, event_seed_domain: str
) -> tuple[int, int]:
    """Return the independent Planner and Renderer seeds for one audit cell."""

    return (
        _event_seed(
            source_id, model_seed, f"planner:{event_seed_domain}"
        ),
        _event_seed(
            source_id, model_seed, f"renderer:{event_seed_domain}"
        ),
    )


def build_bundle(
    dataset: Path,
    *,
    rows: int,
    model_seed: int,
    assume_quiet_preroll: bool = False,
    event_seed_domain: str = "default",
    float_renderer_checkpoint: Path | None = None,
    float_renderer_device: str = "cpu",
) -> tuple[DescriptorBundle, dict[str, Any]]:
    if rows != 0 and rows < 2:
        raise ValueError("rows must be zero (all sources) or at least two")
    if not event_seed_domain.strip():
        raise ValueError("event_seed_domain must not be empty")
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
        row_limit = total if rows == 0 else min(rows, total)
        selected = _select_rows(sources_all, keys_all, progress_all, row_limit)
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
        target_role_all = _text_array(data, "target_role", total, "default")
        target_roles = target_role_all[selected].astype(str)
        if "block_order" in data:
            block_order_all = np.asarray(data["block_order"], dtype=np.int64).reshape(-1)
        elif "row_event_index" in data:
            block_order_all = np.asarray(data["row_event_index"], dtype=np.int64).reshape(-1)
        else:
            block_order_all = np.arange(total, dtype=np.int64)
        if len(block_order_all) != total:
            raise ValueError("block_order must contain one value per dataset row")
        block_order = block_order_all[selected]
        causal_context = np.column_stack(
            [targets[:, 0], targets[:, 1], radii, progress]
        ).astype(np.float32)
        if "renderer_context_raw_dxdy" in data:
            renderer_context = np.asarray(
                data["renderer_context_raw_dxdy"][selected]
            )
            context_source = "dataset field renderer_context_raw_dxdy"
        elif assume_quiet_preroll:
            renderer_context = np.zeros((len(selected), 256, 2), dtype=np.int16)
            for local in range(len(selected)):
                valid = prefix[local][prefix_mask[local] > 0.5]
                take = min(len(valid), 256)
                renderer_context[local, -take:] = np.rint(valid[-take:]).astype(
                    np.int16
                )
            context_source = "explicit quiet left-padding assumption"
        else:
            raise ValueError(
                "final Renderer evaluation needs renderer_context_raw_dxdy with "
                "shape [N,256,2]; use --assume-quiet-preroll only for a smoke demo"
            )
        if renderer_context.shape != (len(selected), 256, 2):
            raise ValueError("renderer_context_raw_dxdy must have shape [N,256,2]")

    horizon = human.shape[1]
    generated = np.zeros((len(selected), horizon, 2), dtype=np.float32)
    generated_mask = np.zeros((len(selected), horizon), dtype=np.float32)
    with Pipeline(
        model_seed=model_seed,
        float_renderer_checkpoint=float_renderer_checkpoint,
        float_renderer_device=float_renderer_device,
        prewarm=True,
    ) as pipeline:
        for local, source in enumerate(source_ids):
            valid_prefix = prefix[local][prefix_mask[local] > 0.5]
            planner_seed, renderer_seed = _event_seeds(
                str(source), model_seed, event_seed_domain
            )
            pending = pipeline.begin_at_b(
                valid_prefix,
                renderer_context_raw_dxdy=renderer_context[local],
            )
            continuation = pending.finish(
                target_rel_at_B=targets[local],
                target_radius=float(radii[local]),
                progress_center=float(progress[local]),
                planner_seed=planner_seed,
                renderer_event_seed_u64=renderer_seed,
            ).render_remaining()
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
    cell = f"planner-{model_seed}:{event_seed_domain}"
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
        generator_cell=np.asarray(["human"] * count + [cell] * count),
        target_role=np.concatenate([target_roles, target_roles]),
        causal_context=np.concatenate([causal_context, causal_context], axis=0),
        block_order=np.concatenate([block_order, block_order]),
    )
    metadata = {
        "schema": "abcurves.public_descriptor_build.v2",
        "source_dataset": dataset.name,
        "source_dataset_sha256": _sha256(dataset),
        "model_seed": int(model_seed),
        "event_seed_domain": event_seed_domain,
        "event_seed_derivation": {
            "algorithm": "first 8 SHA-256 digest bytes as little-endian uint64",
            "planner_input": (
                "abcurves.public-bundle|planner:"
                f"{event_seed_domain}|{model_seed}|<source_id>"
            ),
            "renderer_input": (
                "abcurves.public-bundle|renderer:"
                f"{event_seed_domain}|{model_seed}|<source_id>"
            ),
        },
        "generator_cell": cell,
        "physical_sources": int(count),
        "human_rows": int(count),
        "generated_rows": int(count),
        "selection": "one B80-nearest row per source, key-balanced",
        "renderer_context": context_source,
        "planner_artifact": pipeline.model_files.planner.name,
        "planner_artifact_sha256": _sha256(pipeline.model_files.planner),
        "renderer_backend": pipeline.renderer_receipt.get("backend", "native_fixed_online"),
        "renderer_artifact_sha256": pipeline.renderer_receipt["artifact_sha256"],
    }
    return bundle, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rows",
        type=int,
        default=256,
        help="maximum physical sources; 0 keeps every source",
    )
    parser.add_argument("--model-seed", type=int, choices=(7, 23), default=7)
    parser.add_argument(
        "--event-seed-domain",
        default="default",
        help="stable label for an independent Planner-and-Renderer draw cell",
    )
    parser.add_argument(
        "--float-renderer-checkpoint",
        type=Path,
        help="use a safely loaded retrained float checkpoint instead of the native artifact",
    )
    parser.add_argument("--float-renderer-device", default="cpu")
    parser.add_argument(
        "--assume-quiet-preroll",
        action="store_true",
        help="smoke-test only: left-pad the event prefix to 256 with quiet reports",
    )
    args = parser.parse_args()
    if args.rows != 0 and args.rows < 2:
        raise SystemExit("--rows must be zero (all sources) or at least two")
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    bundle, metadata = build_bundle(
        args.dataset,
        rows=int(args.rows),
        model_seed=int(args.model_seed),
        assume_quiet_preroll=bool(args.assume_quiet_preroll),
        event_seed_domain=str(args.event_seed_domain),
        float_renderer_checkpoint=args.float_renderer_checkpoint,
        float_renderer_device=str(args.float_renderer_device),
    )
    destination = write_descriptor_bundle(args.output, bundle, metadata=metadata)
    print(
        f"wrote {destination} ({bundle.rows} rows, "
        f"{len(bundle.feature_names)} descriptors, sha256={_sha256(destination)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
