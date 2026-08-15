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
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
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
        renderer_context = np.zeros((256, 2), dtype=np.int16)
        renderer_context[-len(prefix) :] = np.rint(prefix).astype(np.int16)
        target = (
            float(data["target_rel_x_at_B"][0]),
            float(data["target_rel_y_at_B"][0]),
        )
        radius = float(data["target_radius"][0])
        progress = float(data["progress"][0])

    started = time.perf_counter_ns()
    pipeline = Pipeline.from_pretrained(prewarm=True)
    startup_ms = (time.perf_counter_ns() - started) / 1e6
    ready_us: list[float] = []
    first_tick_us: list[float] = []
    later_tick_us: list[float] = []
    try:
        for trial in range(args.trials):
            begin = time.perf_counter_ns()
            pending = pipeline.begin_at_b(
                prefix,
                renderer_context_raw_dxdy=renderer_context,
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
            first_tick_us.append((first - ready) / 1e3)
            for _ in range(min(16, stream.duration_ms - 1)):
                tick = time.perf_counter_ns()
                stream.step()
                later_tick_us.append((time.perf_counter_ns() - tick) / 1e3)
    finally:
        pipeline.close()

    result = {
        "schema": "abcurves.runtime_benchmark.v2",
        "model_seed": 7,
        "renderer_artifact_sha256": pipeline.renderer_receipt["artifact_sha256"],
        "renderer_context": "256 reports; example-only quiet left padding",
        "trials": args.trials,
        "clock": "time.perf_counter_ns",
        "device": "CPU",
        "startup_ms": startup_ms,
        "b_to_stream_ready_us": percentiles(ready_us),
        "first_renderer_tick_us": percentiles(first_tick_us),
        "later_renderer_tick_us": percentiles(later_tick_us),
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
        args.out.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
