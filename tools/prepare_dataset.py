#!/usr/bin/env python3
"""Prepare public Planner and Renderer datasets from portable event NPZ data.

Example:

    python tools/prepare_dataset.py exports/ prepared/ \
        --config configs/final_v2.json --branch both

Input may be a portable ``events.npz`` or a directory tree of validated
``abcurves.research_export.v1`` session directories.  A sealed/raw Capture ZIP
still requires the Capture validator/exporter; native archive decoding is not
silently approximated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcurves.preprocessing import (  # noqa: E402
    DatasetPreparationError,
    load_portable_events,
    load_preparation_config,
    load_research_export_events,
    prepare_planner,
    prepare_renderer,
    save_prepared_dataset,
    subset_preparation_result,
)
from abcurves.capture_preprocess import SeamEligibility  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare source-balanced ABCurves datasets from validated Capture "
            "exports or portable event NPZ data."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="portable events.npz or a tree of validated research-export directories",
    )
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "final_v2.json",
        help="typed preparation preset (default: configs/final_v2.json)",
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
        default=0.15,
        help=(
            "user-level validation fraction for research-export folders "
            "(portable NPZ files keep their existing split column)"
        ),
    )
    parser.add_argument(
        "--split-seed",
        default="abcurves.final_v2.user_split",
        help="stable salt for user-level research-export splitting",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_preparation_config(args.config)
        conversion_record: dict[str, object]
        if args.input.is_dir() and not (args.input / "events.npz").is_file():
            conversion = load_research_export_events(
                args.input,
                onset_config=config.planner.materialization.onset,
                seam_eligibility=SeamEligibility(
                    min_prefix_ms=config.planner.seam.min_prefix_ms,
                    min_future_ms=config.planner.seam.min_future_ms,
                ),
                filter_policy=config.planner.quality,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
            )
            events = conversion.events
            conversion_record = {
                "kind": "research_export_directories",
                "directories": list(conversion.export_directories),
                "profiled_event_count": conversion.profiled_event_count,
                "converted_event_count": len(events),
                "excluded_reason_counts": dict(conversion.excluded_reason_counts),
                "validation_fraction": args.validation_fraction,
                "split_seed": args.split_seed,
            }
            input_hashes = {
                str(Path(directory) / "export_manifest.json"): _sha256(
                    Path(directory) / "export_manifest.json"
                )
                for directory in conversion.export_directories
            }
            if events and "val" not in {event.split for event in events}:
                print(
                    "prepare_dataset: research exports contain fewer than two "
                    "user/session identities; produced train only (add an isolated "
                    "validation user before Planner training)",
                    file=sys.stderr,
                )
        else:
            events = load_portable_events(args.input)
            resolved_input = args.input / "events.npz" if args.input.is_dir() else args.input
            conversion_record = {
                "kind": "portable_event_npz",
                "path": str(resolved_input),
            }
            input_hashes = {str(resolved_input): _sha256(resolved_input)}
        args.output.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, dict[str, dict[str, str]]] = {}

        def write_branch(name: str, result) -> None:
            branch_outputs: dict[str, dict[str, str]] = {}
            splits = sorted({event.split for event in result.events})
            for split in splits:
                selected = subset_preparation_result(result, split)
                if not selected.cuts:
                    continue
                safe_split = "".join(
                    char if char.isalnum() or char in {"-", "_"} else "_"
                    for char in split
                ).strip("_")
                if not safe_split:
                    raise DatasetPreparationError(f"split has no safe filename: {split!r}")
                data_path, manifest_path = save_prepared_dataset(
                    selected,
                    args.output / f"{name}_{safe_split}.npz",
                )
                branch_outputs[split] = {
                    "dataset": data_path.name,
                    "manifest": manifest_path.name,
                    "rejections": data_path.with_suffix(".rejections.csv").name,
                }
            outputs[name] = branch_outputs

        if args.branch in {"both", "planner"}:
            write_branch("planner", prepare_planner(events, config.planner))
        if args.branch in {"both", "renderer"}:
            write_branch("renderer", prepare_renderer(events, config.renderer))

        root_manifest = {
            "schema": "abcurves.dataset_preparation_run.v1",
            "input": str(args.input),
            "input_conversion": conversion_record,
            "input_sha256": input_hashes,
            "config": str(args.config),
            "config_sha256": _sha256(args.config),
            "event_count": len(events),
            "branches": outputs,
        }
        run_manifest = args.output / "manifest.json"
        run_manifest.write_text(
            json.dumps(root_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (DatasetPreparationError, OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"prepare_dataset: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(root_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
