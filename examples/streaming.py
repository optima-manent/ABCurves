"""Prepare one reusable profile, then emit one raw report per millisecond."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abcurves import Pipeline

with np.load(ROOT / "examples" / "aim_test.npz", allow_pickle=False) as data:
    row = 0
    prefix = data["prefix_raw_dxdy"][row][data["prefix_mask"][row] > 0.5]
    renderer_profile_window = np.zeros((256, 2), dtype=np.int16)
    renderer_profile_window[-len(prefix) :] = np.rint(prefix).astype(np.int16)
    target = (
        float(data["target_rel_x_at_B"][row]),
        float(data["target_rel_y_at_B"][row]),
    )
    radius = float(data["target_radius"][row])
    progress = float(data["progress"][row])

with Pipeline.from_pretrained() as pipeline:
    # Do this before the latency-sensitive B handoff. Reuse the returned
    # immutable profile for later events from the same device/setup.
    profile = pipeline.prepare_renderer_profile(renderer_profile_window)
    pending = pipeline.begin_at_b(prefix, renderer_profile=profile)

    # Bind the exact geometry once the closed B bin is finalized.
    stream = pending.finish(
        target_rel_at_B=target,
        target_radius=radius,
        progress_center=progress,
        planner_seed=2026,
        renderer_event_seed_u64=2026,
    )

    while not stream.complete:
        dx, dy = stream.step()
        # Send (int(dx), int(dy)) to the caller's 1 kHz hardware/output layer.

print(stream.duration_ms, "ticks rendered")
