"""Generate one final B->C continuation from a recorded A->B prefix."""

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
    continuation = pipeline.generate(
        prefix,
        target_rel_at_B=target,
        target_radius=radius,
        progress_center=progress,
        seed=2026,
    )

print(continuation.shape, continuation.dtype)
print("endpoint counts:", continuation.sum(axis=0).tolist())
