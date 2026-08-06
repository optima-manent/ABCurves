from __future__ import annotations

import numpy as np

from abcurves import BFire, BTrigger, OnsetDetector


def test_onset_is_sustained_and_backtracked() -> None:
    detector = OnsetDetector()
    for _ in range(24):
        assert detector.push(0.0, 0.0, (100.0, 0.0)) is None
    event = None
    for _ in range(12):
        event = detector.push(1.0, 0.0, (100.0, 0.0))
    assert event is not None
    assert event.index == 20  # 24 baseline ticks, then 12-run, backtrack four.
    assert detector.push(1.0, 0.0, (100.0, 0.0)) is None


def test_b_trigger_uses_edge_progress_but_reports_center_progress() -> None:
    trigger = BTrigger()
    trigger.arm((100.0, 0.0), 10.0)
    result = None
    for moved in range(1, 73):
        result = trigger.push_tick(
            1.0,
            0.0,
            target_rel_now=(100.0 - moved, 0.0),
            target_radius_now=10.0,
        )
    assert isinstance(result, BFire)
    assert np.isclose(result.progress_edge, 0.8)
    assert np.isclose(result.progress_center, 0.72)
