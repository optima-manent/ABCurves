#!/usr/bin/env python3
"""Prepare the Planner event corpus and/or global Renderer session corpus.

The two branches intentionally have different source contracts:

* Planner accepts portable ``events.npz`` or validated Capture exports.
* Renderer accepts validated Capture exports or ``abcurves.full_sessions.v1``.

An event-only NPZ is never widened, padded, or mistaken for a full session.
That file has already lost the pre-A, inter-event, and post-C texture that the
global renderer is designed to learn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PLANNER_SPLIT_SEED = "abcurves.final_v2.user_split"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcurves.capture_preprocess import SeamEligibility  # noqa: E402
from abcurves.global_data import (  # noqa: E402
    FULL_SESSION_SCHEMA,
    load_full_sessions,
    load_preparation_config,
    save_global_renderer_dataset,
)
from abcurves.preprocessing import (  # noqa: E402
    DatasetPreparationError,
    load_portable_events,
    load_research_export_events,
    prepare_planner,
    save_prepared_dataset,
    subset_preparation_result,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_event_input(path: Path) -> bool:
    return path.suffix.lower() == ".npz" or (
        path.is_dir() and (path / "events.npz").is_file()
    )


def _portable_session_manifest(path: Path) -> Path | None:
    candidate = path / "sessions.json" if path.is_dir() else path
    if candidate.suffix.lower() != ".json" or not candidate.is_file():
        return None
    try:
        record = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    return candidate if record.get("schema") == FULL_SESSION_SCHEMA else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the event-aligned Planner corpus and/or dense full-session "
            "global Renderer corpus."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "events.npz, abcurves.full_sessions.v1 sessions.json, or a tree "
            "of validated abcurves.research_export.v1 directories"
        ),
    )
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "final.json",
        help="typed preparation preset (default: configs/final.json)",
    )
    parser.add_argument(
        "--branch",
        choices=("both", "planner", "renderer"),
        default="both",
        help="which model-specific dataset to materialize",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="override the config's stable whole-user validation fraction",
    )
    parser.add_argument(
        "--planner-split-seed",
        default=None,
        help="override the frozen Planner user-split salt",
    )
    parser.add_argument(
        "--renderer-split-seed",
        default=None,
        help="override the frozen Renderer user-split salt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage: Path | None = None
    try:
        config = load_preparation_config(args.config)
        event_input = _is_event_input(args.input)
        portable_session_manifest = _portable_session_manifest(args.input)
        wants_planner = args.branch in {"both", "planner"}
        wants_renderer = args.branch in {"both", "renderer"}

        # Fail before creating any partial output.  A-to-C events cannot
        # reproduce the full-session renderer corpus, and full-session arrays
        # contain no target geometry from which to derive Planner labels.
        if wants_renderer and event_input:
            raise DatasetPreparationError(
                "renderer preparation requires dense 1 ms full sessions; "
                "events.npz is valid only with --branch planner"
            )
        if wants_planner and portable_session_manifest is not None:
            raise DatasetPreparationError(
                "portable full sessions contain no target/event geometry; "
                "sessions.json is valid only with --branch renderer"
            )

        fraction = (
            config.renderer.validation_fraction
            if args.validation_fraction is None
            else float(args.validation_fraction)
        )
        planner_split_seed = (
            PLANNER_SPLIT_SEED
            if args.planner_split_seed is None
            else str(args.planner_split_seed)
        )
        renderer_split_seed = (
            config.renderer.split_seed
            if args.renderer_split_seed is None
            else str(args.renderer_split_seed)
        )
        if not 0.0 <= fraction < 1.0:
            raise DatasetPreparationError("validation_fraction must lie in [0, 1)")
        if not planner_split_seed or not renderer_split_seed:
            raise DatasetPreparationError("split seeds must not be empty")

        destination = args.output.expanduser()
        if destination.exists():
            raise DatasetPreparationError(
                f"refusing to overwrite existing output directory: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.stage-",
                dir=str(destination.parent),
            )
        )

        outputs: dict[str, object] = {}
        conversions: dict[str, object] = {}
        event_count = 0
        session_count = 0

        if wants_planner:
            if event_input:
                events = load_portable_events(args.input)
                resolved = args.input / "events.npz" if args.input.is_dir() else args.input
                conversions["planner"] = {
                    "kind": "portable_event_npz",
                    "path": resolved.name,
                    "sha256": _sha256(resolved),
                }
            else:
                conversion = load_research_export_events(
                    args.input,
                    onset_config=config.planner.materialization.onset,
                    seam_eligibility=SeamEligibility(
                        min_prefix_ms=config.planner.seam.min_prefix_ms,
                        min_future_ms=config.planner.seam.min_future_ms,
                    ),
                    filter_policy=config.planner.quality,
                    validation_fraction=fraction,
                    split_seed=planner_split_seed,
                )
                events = conversion.events
                conversions["planner"] = {
                    "kind": "research_export_directories",
                    "directories": [Path(value).name for value in conversion.export_directories],
                    "profiled_event_count": conversion.profiled_event_count,
                    "converted_event_count": len(events),
                    "excluded_reason_counts": dict(conversion.excluded_reason_counts),
                    "validation_fraction": fraction,
                    "split_seed": planner_split_seed,
                }
                if events and "val" not in {event.split for event in events}:
                    print(
                        "prepare_dataset: fewer than two user identities; "
                        "Planner preparation produced train only",
                        file=sys.stderr,
                    )
            event_count = len(events)
            result = prepare_planner(events, config.planner)
            branch_outputs: dict[str, dict[str, str]] = {}
            for split in sorted({event.split for event in result.events}):
                selected = subset_preparation_result(result, split)
                if not selected.cuts:
                    continue
                safe_split = "".join(
                    char if char.isalnum() or char in {"-", "_"} else "_" for char in split
                ).strip("_")
                if not safe_split:
                    raise DatasetPreparationError(f"split has no safe filename: {split!r}")
                data_path, manifest_path = save_prepared_dataset(
                    selected,
                    stage / f"planner_{safe_split}.npz",
                )
                branch_outputs[split] = {
                    "dataset": data_path.name,
                    "manifest": manifest_path.name,
                    "rejections": data_path.with_suffix(".rejections.csv").name,
                }
            outputs["planner"] = branch_outputs

        if wants_renderer:
            collection = load_full_sessions(args.input)
            session_count = len(collection.sessions)
            build = save_global_renderer_dataset(
                stage,
                collection,
                config.renderer,
                validation_fraction=fraction,
                split_seed=renderer_split_seed,
            )
            conversions["renderer"] = {
                "kind": collection.source_kind,
                "source_manifest_ref": collection.source_manifest_ref,
                "source_manifest_sha256": collection.source_manifest_sha256,
                "session_count": session_count,
                "validation_fraction": fraction,
                "split_seed": renderer_split_seed,
            }
            outputs["renderer"] = {
                split: {
                    "directory": role["directory"],
                    "prefix": f"{role['directory']}/prefix_raw_dxdy.npy",
                    "future": f"{role['directory']}/future_raw_dxdy.npy",
                    "meta": f"{role['directory']}/meta.json",
                    "windows": role["windows"],
                }
                for split, role in build["roles"].items()
            }
            outputs["renderer"]["receipts"] = {
                "build_report": {
                    "path": "build_report.json",
                    "sha256": _sha256(stage / "build_report.json"),
                },
                "source_index": {
                    "path": "source_index.json",
                    "sha256": _sha256(stage / "source_index.json"),
                },
                "roles": {
                    split: dict(role["sha256"])
                    for split, role in build["roles"].items()
                },
                "all_source_hashes_verified": bool(
                    build["all_source_hashes_verified"]
                ),
            }

        root_manifest = {
            "schema": "abcurves.dataset_preparation_run.v2",
            "input_kind": (
                "portable_event_npz"
                if event_input
                else "portable_full_sessions"
                if portable_session_manifest is not None
                else "research_export_directories"
            ),
            "input_conversion": conversions,
            "config": args.config.name,
            "config_sha256": _sha256(args.config),
            "event_count": event_count,
            "full_session_count": session_count,
            "branches": outputs,
        }
        run_manifest = stage / "manifest.json"
        run_manifest.write_text(
            json.dumps(root_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(destination)
        stage = None
    except (DatasetPreparationError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        print(f"prepare_dataset: {exc}", file=sys.stderr)
        return 2
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps(root_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
