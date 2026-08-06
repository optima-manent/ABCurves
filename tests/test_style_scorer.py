from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from abcurves.personalization import CausalStyleState
from abcurves.style_scorer import (
    CONTEXT_FEATURE_NAMES,
    FrozenStyleScorer,
    causal_context_features,
    completed_human_context,
    default_style_scorer_path,
    planned_trajectory_features,
    renderer_context,
)
from abcurves.smoothing import smooth_dxdy
from abcurves.texture import texture_features


def _fixture_event() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = np.asarray(
        [[1, 0], [1, 0], [0, 0], [-1, 1], [2, -1], [0, 0], [1, 1]],
        dtype=np.float64,
    )
    planned = np.asarray(
        [[2, 0], [2, 1], [1, 1], [1, 0], [0, 0], [-1, 0]],
        dtype=np.float64,
    )
    completed = np.asarray(
        [[1, 0], [0, 0], [2, -1], [0, 0], [-1, 1], [0, 0], [0, 0], [1, 1]],
        dtype=np.float64,
    )
    return prefix, planned, completed


def _fixture_context() -> np.ndarray:
    prefix, planned, _ = _fixture_event()
    causal = causal_context_features(
        prefix,
        [20.0, 10.0],
        8.0,
        0.8,
        target_distance_at_a=110.0,
        edge_trigger_progress=0.8,
        edge_realized_progress=0.806,
    )
    shape = planned_trajectory_features(planned, [20.0, 10.0], 8.0)
    return renderer_context(
        causal,
        shape,
        task_type="default_static_flick",
        target_role="general",
    )


def test_artifact_is_sanitized_and_hash_bound() -> None:
    text = default_style_scorer_path().read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["schema"] == "abcurves.style_scorer.v1"
    assert record["provenance"]["identity_fields_included"] is False
    assert record["provenance"]["oof_training_scores_included"] is False
    assert record["provenance"]["source_contract_sha256"] == (
        "7b6ed04871447083649d319f8af3fd3c96d24e1d80c4e93748df549641848c99"
    )
    assert "C:\\" not in text
    assert "E:\\" not in text
    assert "fold_assignment" not in text
    assert '"user_id"' not in text
    FrozenStyleScorer()


def test_tampered_artifact_is_rejected(tmp_path: Path) -> None:
    record = json.loads(default_style_scorer_path().read_text(encoding="utf-8"))
    record["directions"]["C"][0] += 1e-6
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="contract hash differs"):
        FrozenStyleScorer(changed)


def test_context_helpers_match_frozen_numeric_fixture() -> None:
    context = _fixture_context()
    assert context.shape == (40,)
    assert len(CONTEXT_FEATURE_NAMES) == 40
    np.testing.assert_allclose(
        context[:20],
        [
            8.0,
            110.0,
            22.360679774997898,
            0.8,
            0.8,
            0.806,
            7.0,
            7.06449510224598,
            4.123105625617661,
            0.5836376921164382,
            1.00921358603514,
            0.7435835590988135,
            2.23606797749979,
            1.00921358603514,
            0.05281102371107724,
            2.0 / 7.0,
            2.0 / 3.0,
            0.9761870601839527,
            -0.21693045781865616,
            10.0 / 21.0,
        ],
        rtol=0.0,
        atol=2e-9,
    )
    assert context[29] == 1.0  # task::default_static_flick
    assert context[37] == 1.0  # role::general
    assert np.sum(context[27:37]) == 1.0
    assert np.sum(context[37:40]) == 1.0


def test_frozen_score_matches_pinned_deployment_transform() -> None:
    scorer = FrozenStyleScorer()
    _, _, completed = _fixture_event()
    score = scorer.score_completed_event(completed, _fixture_context())
    np.testing.assert_allclose(
        score,
        [46.52064321752472, -21.95763004334849, -26.383926607021264],
        rtol=0.0,
        atol=1e-8,
    )
    texture = texture_features(
        completed[None, :, :], np.ones((1, len(completed)), dtype=np.float64)
    )[0]
    np.testing.assert_array_equal(score, scorer.score_texture(texture, _fixture_context()))


def test_completed_context_convenience_uses_frozen_human_smoothing() -> None:
    prefix, _, completed = _fixture_event()
    direct = completed_human_context(
        prefix,
        completed,
        [20.0, 10.0],
        8.0,
        0.8,
        task_type="default_static_flick",
        target_role="general",
        target_distance_at_a=110.0,
        edge_trigger_progress=0.8,
        edge_realized_progress=0.806,
    )
    causal = causal_context_features(
        prefix,
        [20.0, 10.0],
        8.0,
        0.8,
        target_distance_at_a=110.0,
        edge_trigger_progress=0.8,
        edge_realized_progress=0.806,
    )
    shape = planned_trajectory_features(
        smooth_dxdy(
            np.concatenate([prefix, completed], axis=0),
            "triangular_moving_average_path:window=5",
        )[len(prefix) :],
        [20.0, 10.0],
        8.0,
    )
    manual = renderer_context(
        causal,
        shape,
        task_type="default_static_flick",
        target_role="general",
    )
    np.testing.assert_array_equal(direct, manual)


def test_completed_human_observation_respects_causal_state_contract() -> None:
    scorer = FrozenStyleScorer()
    tracker = CausalStyleState()
    _, _, completed = _fixture_event()
    context = _fixture_context()

    expected_score = scorer.score_completed_event(completed, context)
    for index in range(10):
        np.testing.assert_array_equal(tracker.before_event("same-run"), np.zeros(3, np.float32))
        observed = scorer.observe_completed_human(
            tracker, "same-run", completed, context
        )
        np.testing.assert_array_equal(observed, expected_score)
        assert tracker.support == index + 1

    np.testing.assert_array_equal(
        tracker.before_event("same-run"),
        np.clip(0.5 * expected_score, -2.5, 2.5).astype(np.float32),
    )
    np.testing.assert_array_equal(tracker.before_event("new-run"), np.zeros(3, np.float32))
    assert tracker.support == 0


def test_unknown_task_or_role_fails_closed() -> None:
    prefix, planned, _ = _fixture_event()
    causal = causal_context_features(prefix, [20.0, 10.0], 8.0, 0.8)
    shape = planned_trajectory_features(planned, [20.0, 10.0], 8.0)
    with pytest.raises(ValueError, match="frozen vocabulary"):
        renderer_context(causal, shape, task_type="unknown", target_role="general")
    with pytest.raises(ValueError, match="frozen vocabulary"):
        renderer_context(
            causal,
            shape,
            task_type="default_static_flick",
            target_role="unknown",
        )
