"""Start Renderer warming at B, then emit one raw count report per millisecond."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abcurves import Pipeline

with np.load(ROOT / "examples" / "aim_test.npz", allow_pickle=False) as data:
    row = 0
    prefix = data["prefix_raw_dxdy"][row][data["prefix_mask"][row] > 0.5]
    target = (
        float(data["target_rel_x_at_B"][row]),
        float(data["target_rel_y_at_B"][row]),
    )
    radius = float(data["target_radius"][row])
    progress = float(data["progress"][row])

with Pipeline.from_pretrained() as pipeline:
    pending = pipeline.begin_at_b(prefix)  # Prefix GRU warming starts now.

    # Bind the exact geometry once the same closed B bin is finalized.
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
