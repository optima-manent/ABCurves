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
        return {
            "prefix": prefix,
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
