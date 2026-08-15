from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.bundle import DescriptorBundle, write_descriptor_bundle
from evaluation.cold import cold_reference_held_report
from evaluation.warm import warm_reference_held_report
from tools.build_audit_bundle import build_audit_bundle


def _cell_bundle(offset: float) -> DescriptorBundle:
    human_features: list[np.ndarray] = []
    generated_features: list[np.ndarray] = []
    keys: list[str] = []
    sessions: list[str] = []
    sources: list[str] = []
    order: list[int] = []
    task: list[str] = []
    role: list[str] = []
    context: list[np.ndarray] = []
    for key_index in range(4):
        key = f"key-{key_index}"
        session = f"session-{key_index}"
        for row in range(112):
            source = f"{session}-source-{row:03d}"
            feature = np.asarray(
                [key_index, row / 112.0, row % 7, row % 5, row % 3, row % 2],
                dtype=np.float32,
            )
            human_features.append(feature)
            generated_features.append(feature + offset)
            keys.append(key)
            sessions.append(session)
            sources.append(source)
            order.append(row)
            task.append("speed" if row % 2 else "accuracy")
            role.append("near" if row % 3 else "far")
            context.append(np.asarray([key_index, row / 112.0, row % 4], dtype=np.float32))
    count = len(human_features)
    return DescriptorBundle(
        features=np.asarray(human_features + generated_features),
        origin=np.asarray(["human"] * count + ["generated"] * count),
        installation_key=np.asarray(keys + keys),
        session_id=np.asarray(sessions + sessions),
        source_id=np.asarray(sources + sources),
        order=np.asarray(order + order, dtype=np.int64),
        task=np.asarray(task + task),
        feature_names=tuple(f"f{index}" for index in range(6)),
        panel_slices={"trajectory": (0, 2), "texture": (2, 4), "full": (0, 6)},
        generator_cell=np.asarray(["human"] * count + ["source-cell"] * count),
        target_role=np.asarray(role + role),
        causal_context=np.asarray(context + context, dtype=np.float32),
        block_order=np.zeros(count * 2, dtype=np.int64),
    )


def test_public_builder_emits_a_runnable_exact_audit_bundle(tmp_path: Path) -> None:
    first = tmp_path / "cell-a.npz"
    second = tmp_path / "cell-b.npz"
    write_descriptor_bundle(first, _cell_bundle(0.2))
    write_descriptor_bundle(second, _cell_bundle(0.3))

    bundle, metadata = build_audit_bundle(
        [("draw-a", first), ("draw-b", second)],
        role_seed="test-role-freeze",
    )
    origins = np.asarray(bundle.origin).astype(str)
    population = np.asarray(bundle.population_role).astype(str)
    cells = np.asarray(bundle.generator_cell).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    panel = np.asarray(bundle.audit_panel, dtype=bool)
    audit_order = np.asarray(bundle.audit_order, dtype=np.int64)

    assert np.sum(origins == "human") == 448
    assert not np.any((origins == "generated") & (population != "held"))
    assert set(cells[origins == "generated"]) == {"draw-a", "draw-b"}
    held_sessions = sorted(set(sessions[(origins == "human") & (population == "held")]))
    assert len(held_sessions) == 2
    for session in held_sessions:
        human_panel = (
            (origins == "human") & panel & (sessions == session)
        )
        assert np.sum(human_panel) == 32
        assert set(audit_order[human_panel]) == set(range(32))
        for cell in ("draw-a", "draw-b"):
            assert np.sum(
                (origins == "generated")
                & (cells == cell)
                & (sessions == session)
            ) == 32
    assert metadata["role_assignment"]["outcome_blind"] is True

    cold = cold_reference_held_report(
        bundle,
        contamination_counts=(1, 32),
        ledgers=2,
    )
    warm = warm_reference_held_report(
        bundle,
        contamination_counts=(0, 16, 32),
        ledgers=2,
        neighbors=16,
        null_fit_draws=4,
        null_calibration_draws=100,
    )
    assert cold["protocol_variant"]
    assert warm["protocol_variant"]
