"""Command line entry point for the four evaluation questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bundle import load_descriptor_bundle
from .cold import cold_leave_key_out_report, cold_smoke_report
from .floors import human_distance_floor_report
from .labeled import labeled_c2st_report
from .warm import warm_reference_held_report, warm_smoke_report


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _emit(report: Any, output: str | None) -> None:
    payload = json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")


def _counts(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_result_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    checked = []
    for relative, receipt in manifest.get("artifacts", {}).items():
        artifact = manifest_path.parent / relative
        actual = _sha256(artifact) if artifact.is_file() else None
        expected = str(receipt["sha256"])
        ok = actual == expected
        checked.append({"path": relative, "sha256": actual, "ok": ok})
        if not ok:
            failures.append(relative)
    return {
        "schema": "abcurves.detection_result_verification.v1",
        "manifest": str(manifest_path),
        "ok": not failures,
        "failures": failures,
        "artifacts": checked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="ABCurves leakage-aware evaluation protocols",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    floors = commands.add_parser("floors", help="compute descriptive human-distance floors")
    floors.add_argument("bundle")
    floors.add_argument("--panel", default="full")
    floors.add_argument("--minimum-half-rows", type=int, default=8)
    floors.add_argument("--output")

    judge = commands.add_parser("judge", help="run an offline labeled grouped C2ST")
    judge.add_argument("bundle")
    judge.add_argument("--panel", default="full")
    judge.add_argument("--folds", type=int, default=5)
    judge.add_argument("--repeats", type=int, default=5)
    judge.add_argument("--bootstrap", type=int, default=200)
    judge.add_argument("--permutations", type=int, default=0)
    judge.add_argument("--seed", type=int, default=7)
    judge.add_argument("--output")

    cold = commands.add_parser(
        "cold",
        help="run the frozen enriched-bundle unknown-key release protocol",
    )
    cold.add_argument("bundle")
    cold.add_argument(
        "--panels", nargs="+", default=("trajectory", "texture", "full")
    )
    cold.add_argument("--bag-rows", type=int, default=32)
    cold.add_argument("--counts", type=_counts, default=None)
    cold.add_argument("--ledgers", type=int, default=16)
    cold.add_argument("--output")

    cold_smoke = commands.add_parser(
        "cold-smoke",
        help="run a reduced all-keys bundle check (not the release protocol)",
    )
    cold_smoke.add_argument("bundle")
    cold_smoke.add_argument("--panels", nargs="+")
    cold_smoke.add_argument("--bag-rows", type=int, default=32)
    cold_smoke.add_argument("--counts", type=_counts, default=None)
    cold_smoke.add_argument("--ledgers", type=int, default=4)
    cold_smoke.add_argument("--ridge", type=float, default=1.0)
    cold_smoke.add_argument("--output")

    warm = commands.add_parser(
        "warm",
        help="run the frozen enriched-bundle same-session release protocol",
    )
    warm.add_argument("bundle")
    warm.add_argument(
        "--panels", nargs="+", default=("trajectory", "texture", "full")
    )
    warm.add_argument("--counts", type=_counts, default=None)
    warm.add_argument("--ledgers", type=int, default=32)
    warm.add_argument("--neighbors", type=int, default=48)
    warm.add_argument("--null-fit-draws", type=int, default=512)
    warm.add_argument("--null-calibration-draws", type=int, default=2048)
    warm.add_argument("--output")

    warm_smoke = commands.add_parser(
        "warm-smoke",
        help="run a reduced two-bag matching-history check (not the release protocol)",
    )
    warm_smoke.add_argument("bundle")
    warm_smoke.add_argument("--installation-key", required=True)
    warm_smoke.add_argument("--session-id", required=True)
    warm_smoke.add_argument("--panel", default="full")
    warm_smoke.add_argument(
        "--query-origin", choices=("human", "generated"), default="generated"
    )
    warm_smoke.add_argument("--sample-rows", type=int, default=32)
    warm_smoke.add_argument("--null-draws", type=int, default=2048)
    warm_smoke.add_argument("--seed", type=int, default=7)
    warm_smoke.add_argument("--output")

    verify = commands.add_parser("verify-results", help="verify compact published result hashes")
    verify.add_argument(
        "manifest",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "results" / "detection" / "manifest.json"),
    )
    verify.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "verify-results":
        report = verify_result_manifest(arguments.manifest)
        _emit(report, arguments.output)
        if not report["ok"]:
            raise SystemExit(1)
        return
    bundle = load_descriptor_bundle(arguments.bundle)
    if arguments.command == "floors":
        report = human_distance_floor_report(
            bundle,
            panel=arguments.panel,
            minimum_half_rows=arguments.minimum_half_rows,
        )
    elif arguments.command == "judge":
        report = labeled_c2st_report(
            bundle,
            panel=arguments.panel,
            folds=arguments.folds,
            repeats=arguments.repeats,
            bootstrap=arguments.bootstrap,
            permutations=arguments.permutations,
            seed=arguments.seed,
        )
    elif arguments.command == "cold":
        report = cold_leave_key_out_report(
            bundle,
            panels=arguments.panels,
            bag_rows=arguments.bag_rows,
            contamination_counts=arguments.counts,
            ledgers=arguments.ledgers,
        )
    elif arguments.command == "cold-smoke":
        report = cold_smoke_report(
            bundle,
            panels=arguments.panels,
            bag_rows=arguments.bag_rows,
            contamination_counts=arguments.counts,
            ledgers=arguments.ledgers,
            ridge=arguments.ridge,
        )
    elif arguments.command == "warm":
        report = warm_reference_held_report(
            bundle,
            panels=arguments.panels,
            contamination_counts=(
                arguments.counts
                if arguments.counts is not None
                else (0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32)
            ),
            ledgers=arguments.ledgers,
            neighbors=arguments.neighbors,
            null_fit_draws=arguments.null_fit_draws,
            null_calibration_draws=arguments.null_calibration_draws,
        )
    elif arguments.command == "warm-smoke":
        report = warm_smoke_report(
            bundle,
            installation_key=arguments.installation_key,
            session_id=arguments.session_id,
            panel=arguments.panel,
            query_origin=arguments.query_origin,
            sample_rows=arguments.sample_rows,
            null_draws=arguments.null_draws,
            seed=arguments.seed,
        )
    else:  # pragma: no cover - argparse enforces this branch away
        raise AssertionError(arguments.command)
    _emit(report, arguments.output)


__all__ = ["build_parser", "main", "verify_result_manifest"]
