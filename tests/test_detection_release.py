from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.bundle import DescriptorBundle, SCHEMA, load_descriptor_bundle
from evaluation.cli import verify_result_manifest
from evaluation.cold import bag_statistics, cold_smoke_report
from evaluation.floors import human_distance_floor_report, standardized_panel_w1
from evaluation.labeled import labeled_c2st_report
from evaluation.warm import (
    WarmMixtureCell,
    fit_cross_fitted_warm_directions,
    fit_warm_directional_gate,
    warm_directional_bag_statistics,
    warm_directional_mixture_report,
    warm_mixture_masks,
    warm_reference_held_report,
    warm_smoke_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _toy_bundle() -> DescriptorBundle:
    rng = np.random.default_rng(11)
    features = []
    origins = []
    keys = []
    sessions = []
    sources = []
    order = []
    tasks = []
    for key_index, key in enumerate(("key_a", "key_b", "key_c")):
        session = f"{key}_session"
        human = rng.normal(loc=0.15 * key_index, scale=0.45, size=(16, 4))
        generated = rng.normal(loc=2.5 + 0.15 * key_index, scale=0.35, size=(16, 4))
        for origin, values in (("human", human), ("generated", generated)):
            for index, row in enumerate(values):
                features.append(row)
                origins.append(origin)
                keys.append(key)
                sessions.append(session)
                sources.append(f"{key}:{origin}:{index}")
                order.append(index)
                tasks.append("accuracy" if index < 8 else "speed")
    return DescriptorBundle(
        features=np.asarray(features, dtype=np.float64),
        origin=np.asarray(origins),
        installation_key=np.asarray(keys),
        session_id=np.asarray(sessions),
        source_id=np.asarray(sources),
        order=np.asarray(order, dtype=np.int64),
        task=np.asarray(tasks),
        feature_names=("t0", "t1", "x0", "x1"),
        panel_slices={"trajectory": (0, 2), "texture": (2, 4), "full": (0, 4)},
    )


def test_descriptor_bundle_roundtrip_without_pickle(tmp_path: Path) -> None:
    original = _toy_bundle()
    path = tmp_path / "descriptors.npz"
    np.savez_compressed(
        path,
        schema=np.asarray([SCHEMA]),
        features=original.features,
        origin=original.origin,
        installation_key=original.installation_key,
        session_id=original.session_id,
        source_id=original.source_id,
        order=original.order,
        task=original.task,
        feature_names=np.asarray(original.feature_names),
        panel_slices=np.asarray(["trajectory:0:2", "texture:2:4", "full:0:4"]),
    )
    loaded = load_descriptor_bundle(path)
    assert loaded.rows == original.rows
    assert loaded.panel("texture").shape == (original.rows, 2)
    assert loaded.panel_slices["full"] == (0, 4)


def test_distance_floors_are_explicitly_not_cold_detection() -> None:
    bundle = _toy_bundle()
    human = bundle.mask(origin="human")
    generated = bundle.mask(origin="generated")
    distance = standardized_panel_w1(
        bundle.features[human],
        bundle.features[generated],
        scale_reference=bundle.features[human],
    )
    assert distance > 1.0
    report = human_distance_floor_report(bundle, panel="full", minimum_half_rows=4)
    assert report["not_a_detector"] is True
    assert report["target_clean_history_may_be_used"] is True
    assert report["generated_vs_matching_human_session"]["pairs"] == 3


def test_labeled_judge_is_grouped_and_not_a_cold_rule() -> None:
    report = labeled_c2st_report(
        _toy_bundle(),
        panel="full",
        folds=3,
        repeats=1,
        bootstrap=0,
        permutations=0,
    )
    assert report["not_a_cold_detector"] is True
    assert report["auc"] > 0.9
    assert report["n_groups"] == 96


def test_cold_smoke_leaves_complete_key_out_and_calibrates_on_humans() -> None:
    report = cold_smoke_report(
        _toy_bundle(),
        panels=("trajectory", "texture", "full"),
        bag_rows=4,
        contamination_counts=(2, 4),
        ledgers=1,
    )
    threat = report["threat_model"]
    assert report["not_release_protocol"] is True
    assert threat["target_clean_history_used"] is False
    assert threat["generated_outcomes_used_for_threshold_selection"] is False
    assert "complete persistent installation key" in threat["holdout_unit"]
    assert report["held_human"]["keys_evaluated"] == 3
    assert all(row["direction_human_rows"] == 32 for row in report["folds"])
    assert all(row["direction_generated_rows"] == 32 for row in report["folds"])
    assert all(item["evaluations"] > 0 for item in report["candidate_power"])


def test_bag_statistics_contains_dense_sparse_and_subgroup_searches() -> None:
    ranks = np.linspace(0.05, 0.95, 4 * 4, dtype=np.float64).reshape(4, 4)
    statistics = bag_statistics(ranks)
    assert statistics.shape == (1, 5)
    assert np.all(np.isfinite(statistics))


def test_warm_smoke_declares_matching_history() -> None:
    report = warm_smoke_report(
        _toy_bundle(),
        installation_key="key_a",
        session_id="key_a_session",
        panel="full",
        sample_rows=4,
        null_draws=32,
    )
    assert report["not_release_protocol"] is True
    assert report["threat_model"]["target_clean_history_used"] is True
    assert report["not_a_cold_detector"] is True
    assert 0.0 < report["empirical_pvalue"] <= 1.0


def test_compact_result_manifest_and_semantic_boundary() -> None:
    result = verify_result_manifest(ROOT / "results" / "detection" / "manifest.json")
    assert result["ok"], result["failures"]
    assert {artifact["path"] for artifact in result["artifacts"]} == {
        "README.md",
        "pipeline_b80.json",
        "renderer_oracle_b80.json",
    }
    inference = verify_result_manifest(ROOT / "results" / "inference" / "manifest.json")
    assert inference["ok"], inference["failures"]
    current = json.loads(
        (ROOT / "results" / "detection" / "renderer_oracle_b80.json").read_text(
            encoding="utf-8"
        )
    )
    assert current["candidate"]["sha256"] == (
        "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b"
    )
    assert current["scope"]["component"] == "Renderer_only"
    assert "composed Planner-to-Renderer evaluation" in current["scope"]["excludes"]
    ruler = current["known_matching_similarity_ruler"]
    assert ruler["status"] == "descriptive_known_metadata_not_a_detector"
    assert ruler["panels"]["Texture19"]["generated_vs_matching_session"]["pairs"] == 10
    overlap = ruler["texture19_same_session_overlap"]
    assert overlap["human_same_session_pairs"] == 35
    assert overlap["human_same_session_pairs_above_renderer_mean"] == 3
    assert overlap["human_same_session_maximum"] > overlap["renderer_vs_matching_mean"]
    pipeline = json.loads(
        (ROOT / "results" / "detection" / "pipeline_b80.json").read_text(
            encoding="utf-8"
        )
    )
    assert pipeline["subject"] == (
        "current shipped Planner to final native Renderer pipeline"
    )
    assert (
        pipeline["generation_contract"]["human_future_passed_to_planner_or_renderer"]
        is False
    )
    assert len(pipeline["cells"]) == 4
    warm = pipeline["detection"]["warm_known_session"]
    assert warm["tuning"]["alpha"] == 0.0025
    assert warm["human_panel"]["cell_flags"] == 2
    assert warm["mixture_power"][-1]["flags"] == 36
    assert warm["mixture_power"][-1]["evaluations"] == 40
    sweep = warm["cutoff_sweep"]
    assert sweep["selected_alpha"] == 0.0025
    assert [row["alpha"] for row in sweep["rows"]] == [
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
    ]
    assert [row["generated_flags"] for row in sweep["rows"]] == [
        36,
        36,
        36,
        36,
        37,
        39,
        40,
    ]
    assert [row["human_flags"] for row in sweep["rows"]] == [0, 0, 2, 2, 2, 8, 8]
    cold = pipeline["detection"]["cold_unknown_person"]
    assert cold["curve_envelope"]["generated_flags"] == 0
    assert cold["curve_envelope"]["human_flags"] == 0
    assert cold["candidate_power"][-1]["flags"] == 6
    assert cold["candidate_power"][-1]["evaluations"] == 40
    assert cold["held_human"]["keys_flagged"] == 2
    assert cold["held_human"]["keys_evaluated"] == 6


def test_release_detector_code_has_no_machine_specific_paths() -> None:
    paths = list((ROOT / "evaluation").glob("*.py"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "C:\\" not in text
    assert "E:\\" not in text


def _warm_panels(values: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    return {
        "trajectory14": matrix[:, :2],
        "texture19": matrix[:, 2:4],
        "full49": matrix,
    }


def test_warm_directions_exclude_complete_session_and_cell() -> None:
    rng = np.random.default_rng(81)
    human = rng.normal(size=(96, 6))
    sessions = np.repeat(np.asarray(["S0", "S1", "S2"]), 32)
    cells = {
        name: _warm_panels(human + shift + rng.normal(scale=0.08, size=human.shape))
        for name, shift in (("c0", 0.7), ("c1", 0.8), ("c2", 0.9))
    }
    baseline = fit_cross_fitted_warm_directions(
        _warm_panels(human),
        cells,
        session_ids=sessions,
        held_session="S0",
        held_cell="c0",
    )

    changed_human = human.copy()
    changed_human[sessions == "S0"] += 10_000.0
    changed_cells = {
        name: {panel: values.copy() for panel, values in panels.items()}
        for name, panels in cells.items()
    }
    for panel in changed_cells["c0"]:
        changed_cells["c0"][panel] += 20_000.0
    for name in ("c1", "c2"):
        for panel in changed_cells[name]:
            changed_cells[name][panel][sessions == "S0"] -= 30_000.0
    changed = fit_cross_fitted_warm_directions(
        _warm_panels(changed_human),
        changed_cells,
        session_ids=sessions,
        held_session="S0",
        held_cell="c0",
    )

    assert baseline.fit_human_rows == 64
    assert baseline.fit_generated_rows == 128
    assert baseline.fit_generated_cells == ("c1", "c2")
    for left, right in zip(baseline.directions, changed.directions):
        np.testing.assert_allclose(left.location, right.location)
        np.testing.assert_allclose(left.scale, right.scale)
        np.testing.assert_allclose(left.weight, right.weight)


def test_warm_sparse_and_subgroup_statistics_follow_frozen_schedule() -> None:
    rng = np.random.default_rng(82)
    clean = rng.uniform(0.02, 0.98, size=(64, 32, 4))
    mixed = clean.copy()
    mixed[:, :8, :] = 0.995
    clean_stats = warm_directional_bag_statistics(clean)
    mixed_stats = warm_directional_bag_statistics(mixed)
    assert clean_stats.shape == (64, 12)
    # Columns repeat mean, Berk--Jones, subgroup for each of four channels.
    assert np.mean(mixed_stats[:, 1::3]) > np.mean(clean_stats[:, 1::3])
    assert np.mean(mixed_stats[:, 2::3]) > np.mean(clean_stats[:, 2::3])


def test_warm_mixture_ledgers_are_exact_nested_and_outcome_blind() -> None:
    sources = np.asarray([f"source_{index:02d}" for index in range(32)])
    masks, rows = warm_mixture_masks(
        sources,
        counts=(0, 1, 4, 8, 16, 32),
        ledgers=5,
    )
    assert all(
        int(mask.sum()) == int(row["generated_rows"])
        for mask, row in zip(masks, rows)
    )
    for ledger in range(5):
        local = [
            masks[index]
            for index, row in enumerate(rows)
            if int(row["ledger"]) == ledger
            and int(row["generated_rows"]) not in (0, 32)
        ]
        assert all(not np.any(left & ~right) for left, right in zip(local, local[1:]))


def test_exact_warm_gate_uses_disjoint_same_session_reference() -> None:
    rng = np.random.default_rng(83)
    direction_human = rng.normal(size=(96, 6))
    sessions = np.repeat(np.asarray(["S0", "S1", "S2"]), 32)
    direction_cells = {
        name: _warm_panels(
            direction_human + shift + rng.normal(scale=0.12, size=direction_human.shape)
        )
        for name, shift in (("c0", 0.9), ("c1", 1.0), ("c2", 1.1))
    }
    model = fit_cross_fitted_warm_directions(
        _warm_panels(direction_human),
        direction_cells,
        session_ids=sessions,
        held_session="S0",
        held_cell="c0",
    )
    reference = rng.normal(size=(96, 6))
    human_query = rng.normal(size=(32, 6))
    generated_query = human_query + 4.0
    reference_sources = np.asarray([f"reference_{index:03d}" for index in range(96)])
    query_sources = np.asarray([f"query_{index:02d}" for index in range(32)])
    reference_context = rng.normal(size=(96, 3))
    query_context = rng.normal(size=(32, 3))
    reference_task = np.resize(np.asarray(["accuracy", "speed"]), 96)
    query_task = np.resize(np.asarray(["accuracy", "speed"]), 32)
    reference_role = np.resize(np.asarray(["near", "far"]), 96)
    query_role = np.resize(np.asarray(["near", "far"]), 32)
    gate = fit_warm_directional_gate(
        model,
        _warm_panels(reference),
        trusted_session_id="session-zero",
        reference_session_ids=np.full(96, "session-zero"),
        query_session_ids=np.full(32, "session-zero"),
        reference_source_ids=reference_sources,
        query_source_ids=query_sources,
        query_standardized_context=query_context,
        reference_standardized_context=reference_context,
        query_task=query_task,
        reference_task=reference_task,
        query_role=query_role,
        reference_role=reference_role,
        roster_kind="panel",
        null_fit_draws=32,
        null_calibration_draws=128,
    )
    report = warm_directional_mixture_report(
        [
            WarmMixtureCell(
                session="S0",
                cell="c0",
                gate=gate,
                human_panels=_warm_panels(human_query),
                generated_panels=_warm_panels(generated_query),
                source_ids=query_sources,
            )
        ],
        validation_pvalues=(0.2, 0.15, 0.1, 0.08),
        contamination_counts=(0, 8, 16, 32),
        ledgers=4,
    )
    assert report["threat_model"]["target_clean_history_used"] is True
    assert report["threat_model"]["generated_outcomes_used_for_alpha_selection"] is False
    assert report["tuning"]["alpha"] == 0.05
    assert report["mixture_power"][-1]["flag_rate"] == 1.0
    assert report["not_a_cold_detector"] is True


def _exact_warm_bundle() -> DescriptorBundle:
    rng = np.random.default_rng(84)
    rows: list[dict[str, object]] = []
    for index in range(96):
        rows.append(
            {
                "origin": "human",
                "population": "reference",
                "cell": "human",
                "session": f"reference-{index // 48}",
                "key": f"reference-{index // 48}",
                "source": f"reference-{index:03d}",
                "panel": False,
                "audit_order": -1,
                "feature": rng.normal(size=6),
                "context": rng.normal(size=3),
                "task": "accuracy" if index % 2 else "speed",
                "role": "near" if index % 3 else "far",
            }
        )
    for session_index in range(3):
        session = f"held-session-{session_index}"
        key = f"held-key-{session_index}"
        panel_features: dict[str, np.ndarray] = {}
        panel_context: dict[str, np.ndarray] = {}
        for index in range(112):
            source = f"{session}-reference-{index:03d}"
            rows.append(
                {
                    "origin": "human",
                    "population": "held",
                    "cell": "human",
                    "session": session,
                    "key": key,
                    "source": source,
                    "panel": False,
                    "audit_order": -1,
                    "feature": rng.normal(session_index * 0.05, 1.0, 6),
                    "context": rng.normal(size=3),
                    "task": "accuracy" if index % 2 else "speed",
                    "role": "near" if index % 3 else "far",
                }
            )
        for index in range(32):
            source = f"{session}-panel-{index:02d}"
            feature = rng.normal(session_index * 0.05, 1.0, 6)
            context = rng.normal(size=3)
            panel_features[source] = feature
            panel_context[source] = context
            rows.append(
                {
                    "origin": "human",
                    "population": "held",
                    "cell": "human",
                    "session": session,
                    "key": key,
                    "source": source,
                    "panel": True,
                    "audit_order": index,
                    "feature": feature,
                    "context": context,
                    "task": "accuracy" if index % 2 else "speed",
                    "role": "near" if index % 3 else "far",
                }
            )
        for cell_index, cell in enumerate(("cell-a", "cell-b")):
            for index, source in enumerate(panel_features):
                rows.append(
                    {
                        "origin": "generated",
                        "population": "held",
                        "cell": cell,
                        "session": session,
                        "key": key,
                        "source": source,
                        "panel": True,
                        "audit_order": index,
                        "feature": panel_features[source]
                        + 1.8
                        + 0.1 * cell_index,
                        "context": panel_context[source],
                        "task": "accuracy" if index % 2 else "speed",
                        "role": "near" if index % 3 else "far",
                    }
                )
    count = len(rows)
    return DescriptorBundle(
        features=np.asarray([row["feature"] for row in rows], dtype=np.float64),
        origin=np.asarray([row["origin"] for row in rows]),
        installation_key=np.asarray([row["key"] for row in rows]),
        session_id=np.asarray([row["session"] for row in rows]),
        source_id=np.asarray([row["source"] for row in rows]),
        order=np.arange(count, dtype=np.int64),
        task=np.asarray([row["task"] for row in rows]),
        feature_names=tuple(f"f{index}" for index in range(6)),
        panel_slices={
            "trajectory14": (0, 2),
            "texture19": (2, 4),
            "full49": (0, 6),
        },
        population_role=np.asarray([row["population"] for row in rows]),
        generator_cell=np.asarray([row["cell"] for row in rows]),
        target_role=np.asarray([row["role"] for row in rows]),
        causal_context=np.asarray([row["context"] for row in rows], dtype=np.float64),
        block_order=np.zeros(count, dtype=np.int64),
        audit_panel=np.asarray([row["panel"] for row in rows], dtype=bool),
        audit_order=np.asarray([row["audit_order"] for row in rows], dtype=np.int64),
    )


def test_named_warm_route_recreates_disjoint_validation_and_panel_stages() -> None:
    report = warm_reference_held_report(
        _exact_warm_bundle(),
        panels=("trajectory14", "texture19", "full49"),
        contamination_counts=(0, 16, 32),
        ledgers=2,
        neighbors=16,
        null_fit_draws=4,
        null_calibration_draws=100,
    )
    assert report["protocol_variant"] == "frozen_same_session_reference_directional_gate"
    assert report["population"]["held_sessions"] == 3
    assert report["tuning"]["validation_evaluations"] == 6
    assert report["human_panel"]["cell_evaluations"] == 6
    assert report["threat_model"]["generated_outcomes_used_for_alpha_selection"] is False
    assert all(row["reference_rows"] == 80 for row in report["human_split_receipts"])
    assert all(
        receipt["cross_fit"]["held_session_excluded_from_direction_fit"]
        and receipt["cross_fit"]["held_cell_excluded_from_direction_fit"]
        for receipt in report["cross_fit_receipts"]
    )
