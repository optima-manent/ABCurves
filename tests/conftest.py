from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abcurves import Pipeline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def example_row() -> dict[str, object]:
    with np.load(ROOT / "examples" / "aim_test.npz", allow_pickle=False) as data:
        row = 0
        prefix = np.asarray(data["prefix_raw_dxdy"][row], dtype=np.float32)
        prefix = prefix[np.asarray(data["prefix_mask"][row]) > 0.5]
        # The public event fixture has no pre-A session history.  Its runtime
        # smoke test therefore makes the cold-start assumption explicit: the
        # device was quiet before the recorded Planner prefix.
        renderer_context = np.zeros((256, 2), dtype=np.int16)
        renderer_context[-len(prefix) :] = np.rint(prefix).astype(np.int16)
        return {
            "prefix": prefix,
            "renderer_context": renderer_context,
            "target": (
                float(data["target_rel_x_at_B"][row]),
                float(data["target_rel_y_at_B"][row]),
            ),
            "radius": float(data["target_radius"][row]),
            "progress": float(data["progress"][row]),
        }


@pytest.fixture(scope="session")
def pipeline7() -> Pipeline:
    runtime = Pipeline.from_pretrained(prewarm=True)
    yield runtime
    runtime.close()
