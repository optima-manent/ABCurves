"""Measure the warmed release pipeline on this machine; no GPU is used."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abcurves import Pipeline


def percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(data, 50)), 3),
        "p95": round(float(np.percentile(data, 95)), 3),
        "p99": round(float(np.percentile(data, 99)), 3),
        "max": round(float(np.max(data)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.trials < 20:
        raise SystemExit("--trials must be at least 20")

    with np.load(ROOT / "examples" / "aim_test.npz", allow_pickle=False) as data:
        prefix = data["prefix_raw_dxdy"][0][data["prefix_mask"][0] > 0.5]
        renderer_profile_window = np.zeros((256, 2), dtype=np.int16)
        renderer_profile_window[-len(prefix) :] = np.rint(prefix).astype(np.int16)
        target = (
            float(data["target_rel_x_at_B"][0]),
            float(data["target_rel_y_at_B"][0]),
        )
        radius = float(data["target_radius"][0])
        progress = float(data["progress"][0])

    started = time.perf_counter_ns()
    pipeline = Pipeline.from_pretrained(prewarm=True)
    startup_ms = (time.perf_counter_ns() - started) / 1e6
    profile_prepare_us: list[float] = []
    profile = None
    for _ in range(min(args.trials, 50)):
        begin = time.perf_counter_ns()
        profile = pipeline.prepare_renderer_profile(renderer_profile_window)
        profile_prepare_us.append((time.perf_counter_ns() - begin) / 1e3)
    assert profile is not None
    ready_us: list[float] = []
    b_to_first_us: list[float] = []
    first_tick_us: list[float] = []
    later_tick_us: list[float] = []
    phase_names = (
        "b_to_context_submit_us",
        "finish_planner_us",
        "wait_for_renderer_context_us",
        "renderer_begin_us",
        "finish_total_us",
    )
    phases: dict[str, list[float]] = {name: [] for name in phase_names}
    try:
        for trial in range(args.trials):
            begin = time.perf_counter_ns()
            pending = pipeline.begin_at_b(
                prefix,
                renderer_profile=profile,
            )
            stream = pending.finish(
                target_rel_at_B=target,
                target_radius=radius,
                progress_center=progress,
                planner_seed=trial,
                renderer_event_seed_u64=trial,
            )
            ready = time.perf_counter_ns()
            stream.step()
            first = time.perf_counter_ns()
            ready_us.append((ready - begin) / 1e3)
            b_to_first_us.append((first - begin) / 1e3)
            first_tick_us.append((first - ready) / 1e3)
            for name in phase_names:
                phases[name].append(float(getattr(stream.timing, name)))
            for _ in range(min(16, stream.duration_ms - 1)):
                tick = time.perf_counter_ns()
                stream.step()
                later_tick_us.append((time.perf_counter_ns() - tick) / 1e3)
    finally:
        pipeline.close()

    result = {
        "schema": "abcurves.runtime_benchmark.v4",
        "model_seed": 7,
        "renderer_artifact_sha256": pipeline.renderer_receipt["artifact_sha256"],
        "renderer_profile": {
            "reports": 256,
            "preparation": "completed before B and reused for every trial",
            "event_handoff": "copy the 5,088-byte prepared native state, then begin",
            "fixture": "example-only quiet left padding",
            "prepare_us": percentiles(profile_prepare_us),
        },
        "trials": args.trials,
        "clock": "time.perf_counter_ns",
        "device": "CPU",
        "startup_ms": round(startup_ms, 3),
        "critical_path": (
            "freeze B, plan the continuation, clone the prepared profile, and begin; "
            "no 256-report replay"
        ),
        "b_to_stream_ready_us": percentiles(ready_us),
        "b_to_first_report_us": percentiles(b_to_first_us),
        "pipeline_phases_us": {
            name: percentiles(values) for name, values in phases.items()
        },
        "first_renderer_tick_us": percentiles(first_tick_us),
        "later_renderer_tick_us": percentiles(later_tick_us),
        "limitations": [
            "Profile preparation is outside the B-critical path.",
            "Host CPU only; excludes USB/HID transport and application scheduling.",
            "Run the benchmark in the intended application before making a hard "
            "deadline claim.",
        ],
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
