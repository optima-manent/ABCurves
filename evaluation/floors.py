"""Descriptive human-distance floors.

These functions are allowed to use matching session/key metadata because their
question is similarity: "how far apart are two samples from this source?"
That is intentionally different from cold attribution, where the queried
source must remain unknown.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .bundle import DescriptorBundle, IDENTITY_NOTE


def _robust_scale(reference: np.ndarray) -> np.ndarray:
    values = np.asarray(reference, dtype=np.float64)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = values.std(axis=0)
    return np.where(scale >= 1e-8, scale, np.where(fallback >= 1e-8, fallback, 1.0))


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Exact empirical one-dimensional Wasserstein-1 distance."""

    a = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if len(a) == 0 or len(b) == 0:
        raise ValueError("W1 inputs must not be empty")
    support = np.sort(np.concatenate([a, b]))
    if len(support) < 2:
        return 0.0
    widths = np.diff(support)
    cdf_a = np.searchsorted(a, support[:-1], side="right") / float(len(a))
    cdf_b = np.searchsorted(b, support[:-1], side="right") / float(len(b))
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def standardized_panel_w1(
    left: np.ndarray,
    right: np.ndarray,
    *,
    scale_reference: np.ndarray,
) -> float:
    """Mean per-descriptor W1 after robust human-reference scaling."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    reference = np.asarray(scale_reference, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or reference.ndim != 2:
        raise ValueError("left, right and scale_reference must be matrices")
    if a.shape[1] != b.shape[1] or a.shape[1] != reference.shape[1]:
        raise ValueError("all matrices must have the same descriptor width")
    if min(len(a), len(b), len(reference)) == 0:
        raise ValueError("all matrices must contain rows")
    if not all(np.all(np.isfinite(values)) for values in (a, b, reference)):
        raise ValueError("descriptor matrices must be finite")
    scale = _robust_scale(reference)
    distances = [
        _wasserstein_1d(a[:, column] / scale[column], b[:, column] / scale[column])
        for column in range(a.shape[1])
    ]
    return float(np.mean(distances))


def _stable_half(source_ids: Sequence[str], namespace: str) -> np.ndarray:
    bits = []
    for source in source_ids:
        digest = hashlib.sha256(f"{namespace}|{source}".encode("utf-8")).digest()
        bits.append(digest[0] & 1)
    return np.asarray(bits, dtype=np.int8)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"pairs": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "pairs": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def human_distance_floor_report(
    bundle: DescriptorBundle,
    *,
    panel: str = "full",
    minimum_half_rows: int = 8,
) -> dict[str, Any]:
    """Compute same-session and between-human descriptor-distance anchors.

    The scale is fit once from every human row. Sessions are never pooled when
    making pairwise anchors. Generated rows, when present, are compared only to
    the human session whose context key they carry.
    """

    if int(minimum_half_rows) < 2:
        raise ValueError("minimum_half_rows must be at least two")
    features = bundle.panel(panel)
    origins = np.asarray(bundle.origin).astype(str)
    keys = np.asarray(bundle.installation_key).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    sources = np.asarray(bundle.source_id).astype(str)
    human = origins == "human"
    if np.sum(human) < 2 * int(minimum_half_rows):
        raise ValueError("not enough human rows for a distance-floor report")
    reference = features[human]

    session_rows: dict[tuple[str, str], np.ndarray] = {}
    for key, session in sorted(set(zip(keys[human], sessions[human]))):
        rows = np.flatnonzero(human & (keys == key) & (sessions == session))
        if len(rows) >= 2 * int(minimum_half_rows):
            session_rows[(str(key), str(session))] = rows
    if not session_rows:
        raise ValueError("no human session has enough rows to split")

    same_session: list[float] = []
    for (key, session), rows in session_rows.items():
        split = _stable_half(sources[rows], f"{key}|{session}")
        # A degenerate hash split is extremely unlikely; alternating order is a
        # deterministic fallback for tiny test bundles.
        if min(np.sum(split == 0), np.sum(split == 1)) < int(minimum_half_rows):
            split = np.arange(len(rows), dtype=np.int64) & 1
        if min(np.sum(split == 0), np.sum(split == 1)) >= int(minimum_half_rows):
            same_session.append(
                standardized_panel_w1(
                    features[rows[split == 0]],
                    features[rows[split == 1]],
                    scale_reference=reference,
                )
            )

    pair_distances: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
    same_key: list[float] = []
    different_key: list[float] = []
    entries = sorted(session_rows)
    for left, right in combinations(entries, 2):
        distance = standardized_panel_w1(
            features[session_rows[left]],
            features[session_rows[right]],
            scale_reference=reference,
        )
        pair_distances[(left, right)] = distance
        (same_key if left[0] == right[0] else different_key).append(distance)

    nearest_different: list[float] = []
    for entry in entries:
        candidates = [
            distance
            for (left, right), distance in pair_distances.items()
            if entry in (left, right) and left[0] != right[0]
        ]
        if candidates:
            nearest_different.append(min(candidates))

    matched_generated: list[float] = []
    generated = origins == "generated"
    for (key, session), human_rows in session_rows.items():
        generated_rows = np.flatnonzero(generated & (keys == key) & (sessions == session))
        if len(generated_rows) >= int(minimum_half_rows):
            matched_generated.append(
                standardized_panel_w1(
                    features[human_rows],
                    features[generated_rows],
                    scale_reference=reference,
                )
            )

    return {
        "schema": "abcurves.human_distance_floors.v1",
        "question": "descriptive similarity with known matching metadata",
        "not_a_detector": True,
        "target_clean_history_may_be_used": True,
        "identity_semantics": IDENTITY_NOTE,
        "panel": panel,
        "descriptor_width": int(features.shape[1]),
        "human_rows": int(np.sum(human)),
        "human_sessions": int(len(session_rows)),
        "same_session_half": _summary(same_session),
        "same_key_cross_session": _summary(same_key),
        "nearest_different_key": _summary(nearest_different),
        "all_different_key_pairs": _summary(different_key),
        "generated_vs_matching_human_session": _summary(matched_generated),
        "interpretation": (
            "These are scale anchors. Matching a floor does not by itself create "
            "a cold decision rule, because the comparison uses source metadata "
            "that an unknown-identity detector does not have."
        ),
    }


__all__ = ["standardized_panel_w1", "human_distance_floor_report"]
