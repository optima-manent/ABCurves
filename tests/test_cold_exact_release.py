from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.bundle import (
    DescriptorBundle,
    load_descriptor_bundle,
    write_descriptor_bundle,
)
from evaluation.cold import cold_leave_key_out_report


def _exact_bundle(*, overlap_roles: bool = False, one_cell: bool = False) -> DescriptorBundle:
    rng = np.random.default_rng(20260805)
    rows: list[dict[str, object]] = []
    reference_keys = (
        ("held-a", "reference-b")
        if overlap_roles
        else ("reference-a", "reference-b")
    )
    held_keys = ("held-a", "held-b", "held-c")
    cells = ("seed7-draw0",) if one_cell else ("seed7-draw0", "seed23-draw1")

    for key_index, key in enumerate(reference_keys):
        for index in range(40):
            rows.append(
                {
                    "origin": "human",
                    "key": key,
                    "session": f"{key}-session",
                    "source": f"{key}-{index:03d}",
                    "role": "reference",
                    "cell": "human",
                    "panel": False,
                    "order": index,
                    "feature": rng.normal(key_index * 0.05, 1.0, 3),
                    "context": rng.normal(0.0, 1.0, 2),
                }
            )

    for key_index, key in enumerate(held_keys):
        human_features: dict[str, np.ndarray] = {}
        human_context: dict[str, np.ndarray] = {}
        for index in range(32):
            source = f"{key}-{index:03d}"
            feature = rng.normal(key_index * 0.08, 1.0, 3)
            context = rng.normal(0.0, 1.0, 2)
            human_features[source] = feature
            human_context[source] = context
            rows.append(
                {
                    "origin": "human",
                    "key": key,
                    "session": f"{key}-session",
                    "source": source,
                    "role": "held",
                    "cell": "human",
                    "panel": True,
                    "order": index,
                    "feature": feature,
                    "context": context,
                }
            )
        for cell_index, cell in enumerate(cells):
            for index, source in enumerate(sorted(human_features)):
                rows.append(
                    {
                        "origin": "generated",
                        "key": key,
                        "session": f"{key}-session",
                        "source": source,
                        "role": "held",
                        "cell": cell,
                        "panel": True,
                        "order": index,
                        "feature": human_features[source]
                        + np.asarray([0.18, 0.10, 0.14])
                        + cell_index * 0.01,
                        "context": human_context[source],
                    }
                )

    return DescriptorBundle(
        features=np.asarray([row["feature"] for row in rows], dtype=np.float64),
        origin=np.asarray([row["origin"] for row in rows]),
        installation_key=np.asarray([row["key"] for row in rows]),
        session_id=np.asarray([row["session"] for row in rows]),
        source_id=np.asarray([row["source"] for row in rows]),
        order=np.asarray([row["order"] for row in rows], dtype=np.int64),
        task=np.full(len(rows), "task-a"),
        feature_names=("trajectory", "texture", "system"),
        panel_slices={
            "trajectory": (0, 1),
            "texture": (1, 2),
            "full": (0, 3),
        },
        population_role=np.asarray([row["role"] for row in rows]),
        generator_cell=np.asarray([row["cell"] for row in rows]),
        target_role=np.full(len(rows), "general"),
        causal_context=np.asarray([row["context"] for row in rows], dtype=np.float64),
        block_order=np.zeros(len(rows), dtype=np.int64),
        audit_panel=np.asarray([row["panel"] for row in rows], dtype=bool),
        audit_order=np.asarray(
            [int(row["order"]) if bool(row["panel"]) else -1 for row in rows],
            dtype=np.int64,
        ),
    )


def test_exact_cold_protocol_uses_only_declared_held_keys_and_excludes_whole_key() -> None:
    report = cold_leave_key_out_report(
        _exact_bundle(),
        contamination_counts=(1, 16, 32),
        ledgers=2,
    )
    assert report["protocol_variant"] == "frozen_reference_held_cell_leaveout"
    assert report["population"]["reference_installation_keys"] == 2
    assert report["population"]["held_installation_keys"] == 3
    assert report["held_human"]["keys_evaluated"] == 3
    assert len(report["folds"]) == 6  # three held keys x two frozen cells
    for fold in report["folds"]:
        held_key = fold["held_installation_key"]
        held_cell = fold["held_generator_cell"]
        assert held_key not in fold["direction_human_keys"]
        assert held_key not in fold["calibration_human_keys"]
        assert held_cell not in fold["direction_generator_cells"]
        assert set(fold["calibration_human_keys"]) == {
            "reference-a",
            "reference-b",
            *({"held-a", "held-b", "held-c"} - {held_key}),
        }
    assert [row["evaluations"] for row in report["candidate_power"]] == [12, 12, 6]


def test_exact_cold_protocol_rejects_role_or_cell_contract_mismatch() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        cold_leave_key_out_report(_exact_bundle(overlap_roles=True))
    with pytest.raises(ValueError, match="at least two predeclared generator cells"):
        cold_leave_key_out_report(_exact_bundle(one_cell=True))


def test_exact_audit_metadata_round_trips_without_pickle(tmp_path: Path) -> None:
    bundle = _exact_bundle()
    path = write_descriptor_bundle(tmp_path / "exact.npz", bundle)
    loaded = load_descriptor_bundle(path)
    np.testing.assert_array_equal(loaded.population_role, bundle.population_role)
    np.testing.assert_array_equal(loaded.generator_cell, bundle.generator_cell)
    np.testing.assert_array_equal(loaded.target_role, bundle.target_role)
    np.testing.assert_allclose(loaded.causal_context, bundle.causal_context, atol=1e-6)
    np.testing.assert_array_equal(loaded.block_order, bundle.block_order)
    np.testing.assert_array_equal(loaded.audit_panel, bundle.audit_panel)
    np.testing.assert_array_equal(loaded.audit_order, bundle.audit_order)
