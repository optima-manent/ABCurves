"""Offline labeled two-sample judges."""

from __future__ import annotations

from typing import Any

import numpy as np

from abcurves.judges import c2st_report

from .bundle import DescriptorBundle, IDENTITY_NOTE
from .floors import standardized_panel_w1


def labeled_c2st_report(
    bundle: DescriptorBundle,
    *,
    panel: str = "full",
    folds: int = 5,
    repeats: int = 5,
    bootstrap: int = 200,
    permutations: int = 0,
    seed: int = 7,
) -> dict[str, Any]:
    """Run a grouped logistic classifier two-sample test.

    This is a controlled, labeled scientific judge. It answers whether held-out
    generated and human samples can be separated under the declared grouping;
    it is not a rule for accusing an unknown deployment bag.
    """

    features = bundle.panel(panel)
    origins = np.asarray(bundle.origin).astype(str)
    sources = np.asarray(bundle.source_id).astype(str)
    real = origins == "human"
    generated = origins == "generated"
    if np.sum(real) < 2 or np.sum(generated) < 2:
        raise ValueError("a labeled judge needs at least two human and two generated rows")
    report = c2st_report(
        features[real],
        features[generated],
        folds=int(folds),
        repeats=int(repeats),
        seed=int(seed),
        real_groups=sources[real],
        fake_groups=sources[generated],
        bootstrap=int(bootstrap),
        permutations=int(permutations),
    )
    report.update(
        {
            "schema": "abcurves.labeled_c2st.v1",
            "question": "offline labeled human-versus-generated distinguishability",
            "panel": panel,
            "descriptor_width": int(features.shape[1]),
            "identity_semantics": IDENTITY_NOTE,
            "standardized_mean_w1": standardized_panel_w1(
                features[real],
                features[generated],
                scale_reference=features[real],
            ),
            "not_a_cold_detector": True,
            "interpretation": (
                "AUC 0.5 is chance and 1.0 is complete held-out separation. "
                "The judge is given class labels during training, so its AUC "
                "does not establish a false-positive-safe attribution boundary "
                "for a previously unseen person."
            ),
        }
    )
    return report


__all__ = ["labeled_c2st_report"]
