"""Warm, same-session distribution diagnostics and the frozen bag gate.

The warm study gives every evaluated 32-curve roster a trusted, disjoint pool
of human movement from the same recorded session. This module contains the
exact directional part of that deliberately favorable protocol:

* leave the evaluated session and generated seed/draw cell out of direction
  fitting;
* score trajectory, texture, and full descriptor panels separately, then add
  their equal-weight standardized ensemble;
* convert every query row to a conditional rank inside a 48-neighbour trusted
  same-session pool;
* search dense mean shift, sparse Berk--Jones upper-tail evidence, and a
  maximally selected top-k subgroup contrast; and
* calibrate the maximum over all 12 searches on 512 + 2,048 matched human-null
  bags before selecting alpha on disjoint human validation bags.

This is intentionally separate from :mod:`evaluation.cold`.  Supplying target
history here is part of the threat model, not an implementation convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy.special import ndtri, xlogy

from .bundle import DescriptorBundle, IDENTITY_NOTE
from .floors import standardized_panel_w1


WARM_PANEL_NAMES = ("trajectory14", "texture19", "full49")
WARM_BAG_ROWS = 32
WARM_NEIGHBORS = 48
WARM_NULL_FIT_DRAWS = 512
WARM_NULL_CALIBRATION_DRAWS = 2048
WARM_LDA_SHRINK = 0.25
WARM_SUBGROUP_COUNTS = (1, 2, 4, 8, 12, 16)
WARM_CONTAMINATION_COUNTS = (0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32)
WARM_ALPHA_GRID = (0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _robust_scale(reference: np.ndarray) -> np.ndarray:
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = reference.std(axis=0)
    return np.where(scale >= 1e-8, scale, np.where(fallback >= 1e-8, fallback, 1.0))


def _matrix(value: np.ndarray, name: str, *, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [rows, dimensions]")
    if rows is not None and len(array) != int(rows):
        raise ValueError(f"{name} must have {rows} rows, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _vector(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if len(array) != int(rows):
        raise ValueError(f"{name} must have {rows} rows, got {len(array)}")
    return array


def _validate_panels(
    panels: Mapping[str, np.ndarray],
    panel_names: Sequence[str],
    name: str,
    *,
    rows: int | None = None,
    widths: Mapping[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    output: dict[str, np.ndarray] = {}
    inferred_rows = rows
    for panel in panel_names:
        if panel not in panels:
            raise ValueError(f"{name} is missing panel {panel!r}")
        values = _matrix(panels[panel], f"{name}[{panel!r}]", rows=inferred_rows)
        if inferred_rows is None:
            inferred_rows = len(values)
        if widths is not None and values.shape[1] != int(widths[panel]):
            raise ValueError(
                f"{name}[{panel!r}] has width {values.shape[1]}, "
                f"expected {widths[panel]}"
            )
        output[panel] = values
    assert inferred_rows is not None
    return output, int(inferred_rows)


@dataclass(frozen=True)
class WarmLinearDirection:
    """One frozen shrinkage-linear human-to-generated direction."""

    location: np.ndarray
    scale: np.ndarray
    weight: np.ndarray

    def score(self, values: np.ndarray) -> np.ndarray:
        matrix = _matrix(values, "direction values")
        if matrix.shape[1] != len(self.weight):
            raise ValueError("direction input width differs from its fitted width")
        return ((matrix - self.location) / self.scale) @ self.weight


def _fit_linear_direction(
    human: np.ndarray,
    generated: np.ndarray,
    *,
    shrink: float = WARM_LDA_SHRINK,
) -> WarmLinearDirection:
    """Fit the exact 25%-identity-shrunk direction used by the warm audit."""

    h = _matrix(human, "human direction rows")
    g = _matrix(generated, "generated direction rows")
    if h.shape[1] != g.shape[1] or min(len(h), len(g)) < 2:
        raise ValueError("direction fitting needs matching widths and at least two class rows")
    shrink = float(shrink)
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink must lie in [0, 1]")
    location = h.mean(axis=0)
    scale = h.std(axis=0)
    scale[scale < 1e-6] = 1.0
    hz = (h - location) / scale
    gz = (g - location) / scale
    covariance_h = np.atleast_2d(np.cov(hz, rowvar=False))
    covariance_g = np.atleast_2d(np.cov(gz, rowvar=False))
    pooled = 0.5 * (covariance_h + covariance_g)
    width = h.shape[1]
    regularized = (1.0 - shrink) * pooled + shrink * np.eye(width)
    weight = np.linalg.solve(
        regularized + 1e-6 * np.eye(width),
        gz.mean(axis=0) - hz.mean(axis=0),
    )
    projected_scale = math.sqrt(max(float(weight @ covariance_h @ weight), 1e-8))
    weight = weight / projected_scale
    return WarmLinearDirection(location=location, scale=scale, weight=weight)


def _cell_label(value: Hashable) -> str:
    if isinstance(value, tuple):
        return "/".join(str(part) for part in value)
    return str(value)


@dataclass(frozen=True)
class WarmDirectionalModel:
    """Three cross-fitted panel directions and their leakage receipt."""

    panel_names: tuple[str, ...]
    directions: tuple[WarmLinearDirection, ...]
    held_session: str
    held_cell: str
    fit_human_rows: int
    fit_generated_rows: int
    fit_generated_cells: tuple[str, ...]
    lda_shrink: float

    @property
    def panel_widths(self) -> dict[str, int]:
        return {
            panel: int(len(direction.weight))
            for panel, direction in zip(self.panel_names, self.directions)
        }

    def score_panels(self, panels: Mapping[str, np.ndarray]) -> np.ndarray:
        values, _ = _validate_panels(
            panels,
            self.panel_names,
            "score panels",
            widths=self.panel_widths,
        )
        scores = np.column_stack(
            [
                direction.score(values[panel])
                for panel, direction in zip(self.panel_names, self.directions)
            ]
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("directional scores are non-finite")
        return scores

    @property
    def cross_fit_receipt(self) -> dict[str, Any]:
        return {
            "held_session": self.held_session,
            "held_cell": self.held_cell,
            "held_session_excluded_from_direction_fit": True,
            "held_cell_excluded_from_direction_fit": True,
            "fit_human_rows": int(self.fit_human_rows),
            "fit_generated_rows": int(self.fit_generated_rows),
            "fit_generated_cells": list(self.fit_generated_cells),
            "lda_identity_shrink": float(self.lda_shrink),
        }


def fit_cross_fitted_warm_directions(
    human_panels: Mapping[str, np.ndarray],
    generated_cells: Mapping[Hashable, Mapping[str, np.ndarray]],
    *,
    session_ids: np.ndarray,
    held_session: str,
    held_cell: Hashable,
    panel_names: Sequence[str] = WARM_PANEL_NAMES,
    lda_shrink: float = WARM_LDA_SHRINK,
) -> WarmDirectionalModel:
    """Fit directions after removing the complete session and candidate cell.

    Every generated cell must be aligned row-for-row with ``human_panels`` and
    ``session_ids``.  The generated rows from ``held_cell`` are never inspected
    by the fit.  This mirrors the final study's leave-one-session-and-cell-out
    split; passing a pre-pooled generated matrix would erase that guarantee and
    is deliberately unsupported.
    """

    names = tuple(str(name) for name in panel_names)
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("the frozen warm gate requires exactly three distinct panels")
    human, rows = _validate_panels(human_panels, names, "human panels")
    sessions = _vector(session_ids, rows, "session_ids").astype(str)
    held_session = str(held_session)
    if held_session not in set(sessions):
        raise ValueError("held_session is absent from the aligned direction-fit cohort")
    if held_cell not in generated_cells:
        raise ValueError("held_cell is absent from generated_cells")
    if len(generated_cells) < 2:
        raise ValueError("cell cross-fitting needs at least two generated cells")

    widths = {panel: human[panel].shape[1] for panel in names}
    generated: dict[Hashable, dict[str, np.ndarray]] = {}
    for cell, cell_panels in generated_cells.items():
        generated[cell], _ = _validate_panels(
            cell_panels,
            names,
            f"generated cell {_cell_label(cell)!r}",
            rows=rows,
            widths=widths,
        )
    keep = sessions != held_session
    if int(np.sum(keep)) < 2:
        raise ValueError("fewer than two human rows remain after session holdout")
    other_cells = [cell for cell in generated if cell != held_cell]
    if not other_cells:
        raise ValueError("no generated cell remains after held-cell exclusion")

    directions = tuple(
        _fit_linear_direction(
            human[panel][keep],
            np.concatenate([generated[cell][panel][keep] for cell in other_cells], axis=0),
            shrink=float(lda_shrink),
        )
        for panel in names
    )
    return WarmDirectionalModel(
        panel_names=names,
        directions=directions,
        held_session=held_session,
        held_cell=_cell_label(held_cell),
        fit_human_rows=int(np.sum(keep)),
        fit_generated_rows=int(np.sum(keep) * len(other_cells)),
        fit_generated_cells=tuple(_cell_label(cell) for cell in other_cells),
        lda_shrink=float(lda_shrink),
    )


def matched_same_session_pools(
    query_standardized_context: np.ndarray,
    reference_standardized_context: np.ndarray,
    *,
    query_task: np.ndarray,
    reference_task: np.ndarray,
    query_role: np.ndarray,
    reference_role: np.ndarray,
    reference_order: np.ndarray | None = None,
    neighbors: int = WARM_NEIGHBORS,
) -> np.ndarray:
    """Build the frozen lexicographic task/role/context neighbour pools.

    Context columns must already use the location and scale frozen on the
    separate training-human population.  Matching first prefers exact
    task+role, then task, then role, then any row; mean squared context distance
    breaks those categories, followed by ``reference_order``.
    """

    query = _matrix(query_standardized_context, "query_standardized_context")
    reference = _matrix(reference_standardized_context, "reference_standardized_context")
    if len(query) != WARM_BAG_ROWS:
        raise ValueError(f"the frozen warm roster must contain {WARM_BAG_ROWS} rows")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference context widths differ")
    neighbors = int(neighbors)
    if neighbors < 1 or len(reference) < neighbors:
        raise ValueError("trusted reference has fewer rows than the neighbour budget")
    q_task = _vector(query_task, len(query), "query_task").astype(str)
    r_task = _vector(reference_task, len(reference), "reference_task").astype(str)
    q_role = _vector(query_role, len(query), "query_role").astype(str)
    r_role = _vector(reference_role, len(reference), "reference_role").astype(str)
    tie_order = (
        np.arange(len(reference), dtype=np.int64)
        if reference_order is None
        else _vector(reference_order, len(reference), "reference_order")
    )
    output = np.empty((WARM_BAG_ROWS, neighbors), dtype=np.int64)
    for position in range(WARM_BAG_ROWS):
        distance = np.mean((reference - query[position]) ** 2, axis=1)
        task_match = r_task == q_task[position]
        role_match = r_role == q_role[position]
        category = np.where(
            task_match & role_match,
            0,
            np.where(task_match, 1, np.where(role_match, 2, 3)),
        )
        order = np.lexsort((tie_order, distance, category))
        output[position] = order[:neighbors]
    if any(len(np.unique(row)) != neighbors for row in output):
        raise AssertionError("a conditional neighbour pool contains duplicate rows")
    return output


def draw_matched_human_null_indices(
    pools: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Draw one trusted row per context, avoiding within-bag reuse if possible."""

    values = np.asarray(pools, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != WARM_BAG_ROWS or values.shape[1] < 1:
        raise ValueError("pools must have shape [32, neighbours]")
    if np.any(values < 0):
        raise ValueError("pool indices must be non-negative")
    draws = int(draws)
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(int(seed))
    output = np.empty((draws, WARM_BAG_ROWS), dtype=np.int64)
    for draw in range(draws):
        used: set[int] = set()
        row_order = rng.permutation(WARM_BAG_ROWS)
        selected = np.empty(WARM_BAG_ROWS, dtype=np.int64)
        for position in row_order:
            candidates = values[position, rng.permutation(values.shape[1])]
            choice = next(
                (int(item) for item in candidates if int(item) not in used),
                int(candidates[0]),
            )
            selected[position] = choice
            used.add(choice)
        output[draw] = selected
    return output


def _four_direction_scores(raw_three: np.ndarray, reference_three: np.ndarray) -> np.ndarray:
    raw = _matrix(raw_three, "raw_three")
    reference = _matrix(reference_three, "reference_three")
    if raw.shape[1] != 3 or reference.shape[1] != 3:
        raise ValueError("the warm gate requires three raw direction channels")
    location = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = reference.std(axis=0)
    # The frozen runner used the slightly wider 1e-6 guard here, while its
    # final evidence standardization used 1e-8.  Keep that distinction: it is
    # small, but it is part of numerical parity with the released receipt.
    scale = np.where(
        scale >= 1e-6,
        scale,
        np.where(fallback >= 1e-6, fallback, 1.0),
    )
    standardized = (raw - location) / scale
    return np.column_stack([standardized, np.mean(standardized, axis=1)])


def _conditional_quantiles(values: np.ndarray, pool_values: np.ndarray) -> np.ndarray:
    observed = np.asarray(values, dtype=np.float64)
    pools = np.asarray(pool_values, dtype=np.float64)
    if (
        observed.ndim != 3
        or observed.shape[1:] != (WARM_BAG_ROWS, 4)
        or pools.ndim != 3
        or pools.shape[0] != WARM_BAG_ROWS
        or pools.shape[2] != 4
    ):
        raise ValueError("conditional score-rank shapes differ from [bags,32,4]")
    output = np.empty_like(observed)
    for row in range(WARM_BAG_ROWS):
        comparison = pools[row]
        count = np.sum(comparison[None, :, :] <= observed[:, row, None, :], axis=1)
        output[:, row, :] = (count + 0.5) / float(len(comparison) + 1)
    return output


def _berk_jones_upper(ranks: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks, dtype=np.float64)
    pvalue = np.sort(1.0 - values, axis=1)
    fraction = (
        np.arange(1, WARM_BAG_ROWS + 1, dtype=np.float64)
        / float(WARM_BAG_ROWS)
    )[None, :, None]
    clipped = np.clip(pvalue, 1e-9, 1.0 - 1e-9)
    valid = fraction > clipped
    divergence = xlogy(fraction, fraction / clipped) + xlogy(
        1.0 - fraction,
        (1.0 - fraction) / (1.0 - clipped),
    )
    divergence[~valid] = 0.0
    return float(WARM_BAG_ROWS) * np.max(divergence, axis=1)


def warm_direction_stat_names(panel_names: Sequence[str] = WARM_PANEL_NAMES) -> tuple[str, ...]:
    names = tuple(str(name) for name in panel_names)
    if len(names) != 3:
        raise ValueError("exactly three panel names are required")
    return tuple(
        f"dir_{channel}_{statistic}"
        for channel in (*names, "ensemble")
        for statistic in ("mean", "berk_jones", "subgroup_scan")
    )


def warm_directional_bag_statistics(ranks: np.ndarray) -> np.ndarray:
    """Return the exact 4 channels x 3 predeclared warm bag searches."""

    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1:] != (WARM_BAG_ROWS, 4):
        raise ValueError("ranks must have shape [bags, 32, 4]")
    if not np.all(np.isfinite(values)):
        raise ValueError("ranks must be finite")
    z = ndtri(np.clip(values, 1e-6, 1.0 - 1e-6))
    means = np.mean(z, axis=1)
    berk_jones = _berk_jones_upper(values)
    descending = np.sort(z, axis=1)[:, ::-1, :]
    cumulative = np.cumsum(descending, axis=1)
    total = cumulative[:, -1, :]
    scans: list[np.ndarray] = []
    for count in WARM_SUBGROUP_COUNTS:
        top_mean = cumulative[:, count - 1, :] / float(count)
        rest_mean = (total - cumulative[:, count - 1, :]) / float(
            WARM_BAG_ROWS - count
        )
        scans.append(
            math.sqrt(count * (WARM_BAG_ROWS - count) / WARM_BAG_ROWS)
            * (top_mean - rest_mean)
        )
    subgroup = np.max(np.stack(scans, axis=1), axis=1)
    columns: list[np.ndarray] = []
    for channel in range(4):
        columns.extend(
            [means[:, channel], berk_jones[:, channel], subgroup[:, channel]]
        )
    output = np.column_stack(columns)
    if output.shape != (len(values), 12) or not np.all(np.isfinite(output)):
        raise AssertionError("warm directional statistics are invalid")
    return output


@dataclass(frozen=True)
class WarmDirectionalCalibration:
    """Robust standardization and complete-search empirical null."""

    stat_names: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    null_complete_max: np.ndarray
    null_fit_draws: int
    null_calibration_draws: int

    def evidence(self, statistics: np.ndarray) -> np.ndarray:
        values = np.asarray(statistics, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(self.stat_names):
            raise ValueError("statistics width differs from the warm calibration")
        return np.maximum((values - self.location) / self.scale, 0.0)

    def complete_scores(self, statistics: np.ndarray) -> np.ndarray:
        return np.max(self.evidence(statistics), axis=1)

    def pvalues(self, statistics: np.ndarray) -> np.ndarray:
        scores = self.complete_scores(statistics)
        null = np.sort(np.asarray(self.null_complete_max, dtype=np.float64))
        positions = np.searchsorted(null, scores, side="left")
        return (1.0 + len(null) - positions) / float(len(null) + 1)

    @property
    def minimum_attainable_pvalue(self) -> float:
        return 1.0 / float(self.null_calibration_draws + 1)


def fit_warm_directional_calibration(
    null_rank_bags: np.ndarray,
    *,
    panel_names: Sequence[str] = WARM_PANEL_NAMES,
    null_fit_draws: int = WARM_NULL_FIT_DRAWS,
) -> WarmDirectionalCalibration:
    """Fit on the first null block and calibrate on the disjoint remainder."""

    statistics = warm_directional_bag_statistics(null_rank_bags)
    null_fit_draws = int(null_fit_draws)
    if null_fit_draws < 1 or len(statistics) - null_fit_draws < 100:
        raise ValueError("calibration needs a fit block and at least 100 disjoint null bags")
    fit = statistics[:null_fit_draws]
    calibration = statistics[null_fit_draws:]
    location = np.median(fit, axis=0)
    scale = _robust_scale(fit)
    null_complete = np.max(np.maximum((calibration - location) / scale, 0.0), axis=1)
    return WarmDirectionalCalibration(
        stat_names=warm_direction_stat_names(panel_names),
        location=location,
        scale=scale,
        null_complete_max=null_complete,
        null_fit_draws=null_fit_draws,
        null_calibration_draws=int(len(calibration)),
    )


@dataclass(frozen=True)
class WarmDirectionalGate:
    """One roster-specific, same-session, complete-search bag gate."""

    model: WarmDirectionalModel
    trusted_session_id: str
    roster_kind: str
    reference_source_ids: tuple[str, ...]
    query_source_ids: tuple[str, ...]
    reference_raw_three: np.ndarray
    pool_four: np.ndarray
    calibration: WarmDirectionalCalibration
    neighbors: int
    null_seed: int

    def rank_channels(self, query_panels: Mapping[str, np.ndarray]) -> np.ndarray:
        panels, rows = _validate_panels(
            query_panels,
            self.model.panel_names,
            "query panels",
            rows=WARM_BAG_ROWS,
            widths=self.model.panel_widths,
        )
        assert rows == WARM_BAG_ROWS
        raw = self.model.score_panels(panels)
        four = _four_direction_scores(raw, self.reference_raw_three)
        return _conditional_quantiles(four[None, :, :], self.pool_four)[0]

    def pvalues_from_rank_bags(self, ranks: np.ndarray) -> np.ndarray:
        return self.calibration.pvalues(warm_directional_bag_statistics(ranks))

    def pvalue(self, query_panels: Mapping[str, np.ndarray]) -> float:
        return float(self.pvalues_from_rank_bags(self.rank_channels(query_panels))[0])

    @property
    def threat_model(self) -> dict[str, Any]:
        return {
            "target_clean_history_used": True,
            "reference_scope": "trusted same-session human history",
            "trusted_session_id": self.trusted_session_id,
            "reference_query_source_overlap": 0,
            "bag_rows": WARM_BAG_ROWS,
            "neighbor_rows_per_context": int(self.neighbors),
            "null_fit_draws": int(self.calibration.null_fit_draws),
            "null_calibration_draws": int(
                self.calibration.null_calibration_draws
            ),
            "complete_search_calibrated_as_one_object": True,
            "direction_cross_fit": self.model.cross_fit_receipt,
        }


def fit_warm_directional_gate(
    model: WarmDirectionalModel,
    reference_panels: Mapping[str, np.ndarray],
    *,
    trusted_session_id: str,
    reference_session_ids: np.ndarray,
    query_session_ids: np.ndarray,
    reference_source_ids: np.ndarray,
    query_source_ids: np.ndarray,
    query_standardized_context: np.ndarray,
    reference_standardized_context: np.ndarray,
    query_task: np.ndarray,
    reference_task: np.ndarray,
    query_role: np.ndarray,
    reference_role: np.ndarray,
    reference_order: np.ndarray | None = None,
    roster_kind: str = "panel",
    neighbors: int = WARM_NEIGHBORS,
    null_fit_draws: int = WARM_NULL_FIT_DRAWS,
    null_calibration_draws: int = WARM_NULL_CALIBRATION_DRAWS,
    null_seed: int | None = None,
) -> WarmDirectionalGate:
    """Prepare a roster-specific gate from disjoint trusted same-session rows.

    ``query_standardized_context`` and ``reference_standardized_context`` must
    be standardized with frozen statistics from the separate human training
    population.  Query descriptors are deliberately not accepted here: the
    human-null fit is determined entirely by the trusted reference and query
    *contexts*, before either human-panel or generated-panel outcomes are
    scored.
    """

    reference, reference_rows = _validate_panels(
        reference_panels,
        model.panel_names,
        "trusted reference panels",
        widths=model.panel_widths,
    )
    trusted_session_id = str(trusted_session_id)
    reference_sessions = _vector(
        reference_session_ids, reference_rows, "reference_session_ids"
    ).astype(str)
    query_sessions = _vector(
        query_session_ids, WARM_BAG_ROWS, "query_session_ids"
    ).astype(str)
    if set(reference_sessions) != {trusted_session_id}:
        raise ValueError("every trusted reference row must come from the declared session")
    if set(query_sessions) != {trusted_session_id}:
        raise ValueError("every query context must come from the declared session")
    reference_sources = _vector(
        reference_source_ids, reference_rows, "reference_source_ids"
    ).astype(str)
    query_sources = _vector(
        query_source_ids, WARM_BAG_ROWS, "query_source_ids"
    ).astype(str)
    if len(np.unique(query_sources)) != WARM_BAG_ROWS:
        raise ValueError("the query roster must contain 32 physically unique source IDs")
    overlap = set(reference_sources) & set(query_sources)
    if overlap:
        raise ValueError("trusted reference and query source IDs must be disjoint")

    pools = matched_same_session_pools(
        query_standardized_context,
        reference_standardized_context,
        query_task=query_task,
        reference_task=reference_task,
        query_role=query_role,
        reference_role=reference_role,
        reference_order=reference_order,
        neighbors=int(neighbors),
    )
    total_null_draws = int(null_fit_draws) + int(null_calibration_draws)
    if int(null_fit_draws) < 1 or int(null_calibration_draws) < 100:
        raise ValueError("null fit/calibration budgets are too small")
    seed = (
        _stable_seed(
            "within-bag-null",
            model.held_session,
            str(roster_kind),
            "warm",
            total_null_draws,
        )
        if null_seed is None
        else int(null_seed)
    )
    null_indices = draw_matched_human_null_indices(
        pools,
        draws=total_null_draws,
        seed=seed,
    )
    reference_raw = model.score_panels(reference)
    reference_four = _four_direction_scores(reference_raw, reference_raw)
    pool_four = reference_four[pools]
    null_four = reference_four[null_indices]
    null_ranks = _conditional_quantiles(null_four, pool_four)
    calibration = fit_warm_directional_calibration(
        null_ranks,
        panel_names=model.panel_names,
        null_fit_draws=int(null_fit_draws),
    )
    return WarmDirectionalGate(
        model=model,
        trusted_session_id=trusted_session_id,
        roster_kind=str(roster_kind),
        reference_source_ids=tuple(reference_sources.tolist()),
        query_source_ids=tuple(query_sources.tolist()),
        reference_raw_three=reference_raw,
        pool_four=pool_four,
        calibration=calibration,
        neighbors=int(neighbors),
        null_seed=seed,
    )


def warm_mixture_masks(
    source_ids: np.ndarray,
    *,
    counts: Sequence[int] = WARM_CONTAMINATION_COUNTS,
    ledgers: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create exact, nested, source-ID-led mixture assignments."""

    sources = _vector(source_ids, WARM_BAG_ROWS, "source_ids").astype(str)
    if len(np.unique(sources)) != WARM_BAG_ROWS:
        raise ValueError("mixture ledgers require 32 physically unique source IDs")
    normalized = tuple(int(count) for count in counts)
    if not normalized or any(count < 0 or count > WARM_BAG_ROWS for count in normalized):
        raise ValueError("mixture counts must lie in [0, 32]")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("mixture counts must be unique and increasing")
    ledgers = int(ledgers)
    if ledgers < 1:
        raise ValueError("ledgers must be positive")
    masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for count in normalized:
        repetitions = 1 if count in (0, WARM_BAG_ROWS) else ledgers
        for ledger in range(repetitions):
            order = sorted(
                range(WARM_BAG_ROWS),
                key=lambda row: _stable_seed(
                    "within-bag-ledger-v1", ledger, sources[row]
                ),
            )
            mask = np.zeros(WARM_BAG_ROWS, dtype=bool)
            mask[np.asarray(order[:count], dtype=np.int64)] = True
            masks.append(mask)
            metadata.append(
                {
                    "generated_rows": count,
                    "fraction": float(count / WARM_BAG_ROWS),
                    "ledger": ledger,
                }
            )
    output = np.stack(masks, axis=0)
    for ledger in range(ledgers):
        local = [
            output[index]
            for index, row in enumerate(metadata)
            if int(row["ledger"]) == ledger
            and int(row["generated_rows"]) not in (0, WARM_BAG_ROWS)
        ]
        for left, right in zip(local, local[1:]):
            if np.any(left & ~right):
                raise AssertionError("mixture ledgers are not nested")
    return output, metadata


def tune_warm_alpha(
    validation_pvalues: Sequence[float],
    *,
    alpha_grid: Sequence[float] = WARM_ALPHA_GRID,
    null_calibration_draws: int = WARM_NULL_CALIBRATION_DRAWS,
) -> dict[str, Any]:
    """Select the largest predeclared alpha with zero validation flags."""

    values = np.asarray(validation_pvalues, dtype=np.float64).reshape(-1)
    if len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("validation_pvalues must be a non-empty finite sequence")
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("validation p-values must lie in (0, 1]")
    grid = tuple(float(alpha) for alpha in alpha_grid)
    if not grid or any(alpha <= 0.0 or alpha >= 1.0 for alpha in grid):
        raise ValueError("alpha_grid must contain probabilities in (0, 1)")
    if any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("alpha_grid must be strictly decreasing")
    selected = next(
        (alpha for alpha in grid if int(np.sum(values <= alpha)) == 0),
        grid[-1] / 2.0,
    )
    resolution = 1.0 / float(int(null_calibration_draws) + 1)
    return {
        "alpha": float(selected),
        "alpha_grid": list(grid),
        "validation_evaluations": int(len(values)),
        "validation_false_positives": int(np.sum(values <= selected)),
        "validation_min_p": float(np.min(values)),
        "empirical_p_resolution": resolution,
        "gate_usable": bool(selected >= resolution),
        "generated_outcomes_used_for_alpha_selection": False,
    }


@dataclass(frozen=True)
class WarmMixtureCell:
    """One held session/cell final-panel evaluation binding."""

    session: str
    cell: Hashable
    gate: WarmDirectionalGate
    human_panels: Mapping[str, np.ndarray]
    generated_panels: Mapping[str, np.ndarray]
    source_ids: np.ndarray


def warm_directional_mixture_report(
    evaluations: Sequence[WarmMixtureCell],
    *,
    validation_pvalues: Sequence[float],
    contamination_counts: Sequence[int] = WARM_CONTAMINATION_COUNTS,
    ledgers: int = 32,
    alpha_grid: Sequence[float] = WARM_ALPHA_GRID,
) -> dict[str, Any]:
    """Evaluate the frozen primary gate on outcome-blind nested mixtures.

    Alpha is chosen *only* from ``validation_pvalues``.  Candidate ranks enter
    after that selection and cannot alter the operating point.  Each supplied
    gate must have been cross-fitted against the same held session/cell named
    by its :class:`WarmMixtureCell`.
    """

    cells = list(evaluations)
    if not cells:
        raise ValueError("at least one warm mixture cell is required")
    null_budgets = {
        (
            int(cell.gate.calibration.null_fit_draws),
            int(cell.gate.calibration.null_calibration_draws),
        )
        for cell in cells
    }
    if len(null_budgets) != 1:
        raise ValueError("all warm gates must use the same fit and empirical-null budgets")
    fit_draws, calibration_draws = next(iter(null_budgets))
    tuning = tune_warm_alpha(
        validation_pvalues,
        alpha_grid=alpha_grid,
        null_calibration_draws=calibration_draws,
    )
    alpha = float(tuning["alpha"])
    counts = tuple(int(value) for value in contamination_counts)
    flags = {count: 0 for count in counts}
    totals = {count: 0 for count in counts}
    human_pvalues: list[float] = []
    human_flags = 0
    sessions_with_human_flag: set[str] = set()
    receipts: list[dict[str, Any]] = []

    for cell in cells:
        session = str(cell.session)
        cell_label = _cell_label(cell.cell)
        if cell.gate.model.held_session != session:
            raise ValueError("mixture cell session differs from its cross-fit holdout")
        if cell.gate.model.held_cell != cell_label:
            raise ValueError("mixture cell identity differs from its cross-fit holdout")
        sources = _vector(cell.source_ids, WARM_BAG_ROWS, "cell source_ids").astype(str)
        if tuple(sources.tolist()) != cell.gate.query_source_ids:
            raise ValueError("mixture roster differs from the gate's frozen query roster")
        human_ranks = cell.gate.rank_channels(cell.human_panels)
        generated_ranks = cell.gate.rank_channels(cell.generated_panels)
        human_p = float(cell.gate.pvalues_from_rank_bags(human_ranks)[0])
        human_pvalues.append(human_p)
        if human_p <= alpha:
            human_flags += 1
            sessions_with_human_flag.add(session)

        masks, metadata = warm_mixture_masks(
            sources,
            counts=counts,
            ledgers=int(ledgers),
        )
        mixed = np.where(
            masks[:, :, None],
            generated_ranks[None, :, :],
            human_ranks[None, :, :],
        )
        pvalues = cell.gate.pvalues_from_rank_bags(mixed)
        for pvalue, row in zip(pvalues, metadata):
            count = int(row["generated_rows"])
            totals[count] += 1
            flags[count] += int(float(pvalue) <= alpha)
        receipts.append(
            {
                "session": session,
                "held_cell": cell_label,
                "reference_rows": len(cell.gate.reference_source_ids),
                "neighbor_rows_per_context": int(cell.gate.neighbors),
                "cross_fit": cell.gate.model.cross_fit_receipt,
            }
        )

    mixture_power = [
        {
            "generated_rows": count,
            "fraction": float(count / WARM_BAG_ROWS),
            "evaluations": int(totals[count]),
            "flags": int(flags[count]),
            "flag_rate": float(flags[count] / totals[count]) if totals[count] else None,
        }
        for count in counts
    ]
    return {
        "schema": "abcurves.warm_same_session_directional_bag_gate.v1",
        "question": (
            "mixture evidence when trusted, disjoint movement from the same "
            "recorded session is already known"
        ),
        "threat_model": {
            "target_clean_history_used": True,
            "reference_scope": "trusted same-session human history",
            "bag_rows": WARM_BAG_ROWS,
            "primary_gate": (
                "maximum over cross-fitted trajectory, texture and full-panel "
                "directional dense, sparse-tail and subgroup statistics"
            ),
            "alpha_selection": "disjoint human validation evaluations only",
            "generated_outcomes_used_for_alpha_selection": False,
            "mixture_assignment": "nested source-ID ledgers fixed without outcomes",
            "null_fit_draws": fit_draws,
            "null_calibration_draws": calibration_draws,
        },
        "identity_semantics": IDENTITY_NOTE,
        "tuning": tuning,
        "human_panel": {
            "cell_evaluations": int(len(cells)),
            "cell_flags": int(human_flags),
            "sessions": int(len({str(cell.session) for cell in cells})),
            "sessions_with_any_flag": int(len(sessions_with_human_flag)),
            "minimum_pvalue": float(np.min(human_pvalues)),
        },
        "mixture_power": mixture_power,
        "cross_fit_receipts": receipts,
        "not_a_cold_detector": True,
        "interpretation": (
            "This gate is conditional on trusted same-session history. Its "
            "human-panel flag count is finite-sample evidence, not a universal "
            "zero-false-positive guarantee, and it says nothing about an "
            "unknown-person cold detector."
        ),
    }


def _warm_validation_and_reference_rows(
    bundle: DescriptorBundle,
    *,
    session: str,
    alias: str,
    panel_indices: np.ndarray,
    minimum_reference_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the frozen, outcome-blind 32-row validation reservation."""

    origins = np.asarray(bundle.origin).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    population = np.asarray(bundle.population_role).astype(str)
    panel = np.asarray(bundle.audit_panel, dtype=bool)
    sources = np.asarray(bundle.source_id).astype(str)
    tasks = np.asarray(bundle.task).astype(str)
    roles = np.asarray(bundle.target_role).astype(str)
    candidates = np.flatnonzero(
        (origins == "human")
        & (population == "held")
        & (sessions == str(session))
        & ~panel
    )
    strata: dict[tuple[str, str], list[int]] = {}
    for index in candidates:
        strata.setdefault((tasks[index], roles[index]), []).append(int(index))
    for key in strata:
        strata[key].sort(
            key=lambda index: _stable_seed("validation", alias, key, sources[index])
        )
    desired: dict[tuple[str, str], int] = {}
    for key in zip(tasks[panel_indices].tolist(), roles[panel_indices].tolist()):
        desired[key] = desired.get(key, 0) + 1
    chosen: list[int] = []
    for key in sorted(desired):
        chosen.extend(strata.get(key, [])[: desired[key]])
    chosen_set = set(chosen)
    remaining = sorted(
        (int(index) for index in candidates if int(index) not in chosen_set),
        key=lambda index: _stable_seed("validation-fill", alias, sources[index]),
    )
    chosen.extend(remaining[: WARM_BAG_ROWS - len(chosen)])
    if len(chosen) != WARM_BAG_ROWS or len(set(chosen)) != WARM_BAG_ROWS:
        raise ValueError(
            f"session {session!r} cannot reserve a unique 32-row warm validation bag"
        )
    validation = np.asarray(
        sorted(
            chosen,
            key=lambda index: _stable_seed("validation-order", alias, sources[index]),
        ),
        dtype=np.int64,
    )
    reference = np.asarray(
        [int(index) for index in candidates if int(index) not in set(chosen)],
        dtype=np.int64,
    )
    if len(reference) < int(minimum_reference_rows):
        raise ValueError(
            f"session {session!r} has {len(reference)} trusted reference rows; "
            f"at least {minimum_reference_rows} are required"
        )
    return validation, reference


def warm_reference_held_report(
    bundle: DescriptorBundle,
    *,
    panels: Sequence[str] = ("trajectory", "texture", "full"),
    contamination_counts: Sequence[int] = WARM_CONTAMINATION_COUNTS,
    ledgers: int = 32,
    neighbors: int = WARM_NEIGHBORS,
    null_fit_draws: int = WARM_NULL_FIT_DRAWS,
    null_calibration_draws: int = WARM_NULL_CALIBRATION_DRAWS,
) -> dict[str, Any]:
    """Run the named warm release protocol from an enriched audit bundle.

    The frozen 32-row audit roster is evaluated only after alpha has been
    selected on a separate 32-row human validation roster.  Both rosters are
    disjoint from the larger trusted same-session reference pool.  Direction
    fitting additionally leaves out the complete evaluated session and the
    evaluated generator cell.
    """

    required = {
        "population_role": bundle.population_role,
        "generator_cell": bundle.generator_cell,
        "target_role": bundle.target_role,
        "causal_context": bundle.causal_context,
        "audit_panel": bundle.audit_panel,
        "audit_order": bundle.audit_order,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(
            "exact warm audit requires enriched bundle arrays: " + ", ".join(missing)
        )
    panel_names = tuple(str(name) for name in panels)
    if len(panel_names) != 3 or len(set(panel_names)) != 3:
        raise ValueError("the warm release route requires exactly three distinct panels")
    panel_values = {name: bundle.panel(name) for name in panel_names}
    origins = np.asarray(bundle.origin).astype(str)
    population = np.asarray(bundle.population_role).astype(str)
    cells = np.asarray(bundle.generator_cell).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    sources = np.asarray(bundle.source_id).astype(str)
    tasks = np.asarray(bundle.task).astype(str)
    roles = np.asarray(bundle.target_role).astype(str)
    context = np.asarray(bundle.causal_context, dtype=np.float64)
    frozen = np.asarray(bundle.audit_panel, dtype=bool)
    audit_order = np.asarray(bundle.audit_order, dtype=np.int64)
    human = origins == "human"
    generated = origins == "generated"
    reference_human = human & (population == "reference")
    held_human = human & (population == "held")
    held_generated = generated & (population == "held")
    if np.any(generated & (population == "reference")):
        raise ValueError("warm reference population must contain humans only")
    if np.sum(reference_human) < 2:
        raise ValueError("warm context matching needs a separate reference-human population")
    if np.any(held_generated & ~frozen):
        raise ValueError("every warm generated row must belong to the frozen audit panel")
    cell_ids = sorted(set(cells[held_generated]) - {"", "human", "none"})
    if len(cell_ids) < 2:
        raise ValueError("warm cell cross-fitting needs at least two generator cells")
    held_sessions = sorted(np.unique(sessions[held_human & frozen]).tolist())
    if len(held_sessions) < 2:
        raise ValueError("warm session cross-fitting needs at least two held sessions")
    alias_by_session = {
        session: f"S{position:02d}"
        for position, session in enumerate(held_sessions, start=1)
    }

    ordered_panels: dict[str, np.ndarray] = {}
    validation_rows: dict[str, np.ndarray] = {}
    reference_rows: dict[str, np.ndarray] = {}
    for session in held_sessions:
        indices = np.flatnonzero(
            held_human & frozen & (sessions == session)
        )
        if len(indices) != WARM_BAG_ROWS:
            raise ValueError(
                f"session {session!r} must contain exactly {WARM_BAG_ROWS} frozen human rows"
            )
        local_order = audit_order[indices]
        if len(np.unique(local_order)) != WARM_BAG_ROWS:
            raise ValueError(f"session {session!r} has duplicate frozen audit order values")
        indices = indices[np.argsort(local_order, kind="stable")]
        ordered_panels[session] = indices
        validation_rows[session], reference_rows[session] = (
            _warm_validation_and_reference_rows(
                bundle,
                session=session,
                alias=alias_by_session[session],
                panel_indices=indices,
                minimum_reference_rows=int(neighbors),
            )
        )

    direction_indices = np.concatenate(
        [ordered_panels[session] for session in held_sessions]
    )
    direction_aliases = np.concatenate(
        [
            np.full(WARM_BAG_ROWS, alias_by_session[session])
            for session in held_sessions
        ]
    )
    direction_human_panels = {
        name: values[direction_indices] for name, values in panel_values.items()
    }
    generated_lookup: dict[tuple[str, str, str], int] = {}
    for index in np.flatnonzero(held_generated):
        key = (cells[index], sessions[index], sources[index])
        if key in generated_lookup:
            raise ValueError("warm generated panel has duplicate cell/session/source bindings")
        generated_lookup[key] = int(index)
    direction_generated_indices: dict[str, np.ndarray] = {}
    for cell in cell_ids:
        try:
            direction_generated_indices[cell] = np.asarray(
                [
                    generated_lookup[(cell, sessions[index], sources[index])]
                    for index in direction_indices
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "every warm cell must align to every frozen human source"
            ) from error
    direction_generated_panels = {
        cell: {
            name: values[indices] for name, values in panel_values.items()
        }
        for cell, indices in direction_generated_indices.items()
    }

    context_location = np.mean(context[reference_human], axis=0)
    context_scale = np.std(context[reference_human], axis=0)
    context_scale[context_scale < 1e-8] = 1.0
    standardized_context = (context - context_location) / context_scale
    evaluations: list[WarmMixtureCell] = []
    validation_pvalues: list[float] = []
    split_receipts: list[dict[str, Any]] = []
    for session in held_sessions:
        alias = alias_by_session[session]
        panel_indices = ordered_panels[session]
        validation_indices = validation_rows[session]
        trusted_indices = reference_rows[session]
        split_receipts.append(
            {
                "session": alias,
                "full_session_id": session,
                "panel_rows": WARM_BAG_ROWS,
                "validation_rows": WARM_BAG_ROWS,
                "reference_rows": int(len(trusted_indices)),
            }
        )
        for cell in cell_ids:
            model = fit_cross_fitted_warm_directions(
                direction_human_panels,
                direction_generated_panels,
                session_ids=direction_aliases,
                held_session=alias,
                held_cell=cell,
                panel_names=panel_names,
            )
            common = {
                "trusted_session_id": alias,
                "reference_session_ids": np.full(len(trusted_indices), alias),
                "reference_source_ids": sources[trusted_indices],
                "reference_standardized_context": standardized_context[trusted_indices],
                "reference_task": tasks[trusted_indices],
                "reference_role": roles[trusted_indices],
                "reference_order": trusted_indices,
                "neighbors": int(neighbors),
                "null_fit_draws": int(null_fit_draws),
                "null_calibration_draws": int(null_calibration_draws),
            }
            trusted_panels = {
                name: values[trusted_indices] for name, values in panel_values.items()
            }
            validation_gate = fit_warm_directional_gate(
                model,
                trusted_panels,
                query_session_ids=np.full(WARM_BAG_ROWS, alias),
                query_source_ids=sources[validation_indices],
                query_standardized_context=standardized_context[validation_indices],
                query_task=tasks[validation_indices],
                query_role=roles[validation_indices],
                roster_kind="validation",
                **common,
            )
            validation_pvalues.append(
                validation_gate.pvalue(
                    {
                        name: values[validation_indices]
                        for name, values in panel_values.items()
                    }
                )
            )
            panel_gate = fit_warm_directional_gate(
                model,
                trusted_panels,
                query_session_ids=np.full(WARM_BAG_ROWS, alias),
                query_source_ids=sources[panel_indices],
                query_standardized_context=standardized_context[panel_indices],
                query_task=tasks[panel_indices],
                query_role=roles[panel_indices],
                roster_kind="panel",
                **common,
            )
            generated_indices = direction_generated_indices[cell][
                np.asarray(held_sessions.index(session) * WARM_BAG_ROWS)
                + np.arange(WARM_BAG_ROWS)
            ]
            evaluations.append(
                WarmMixtureCell(
                    session=alias,
                    cell=cell,
                    gate=panel_gate,
                    human_panels={
                        name: values[panel_indices]
                        for name, values in panel_values.items()
                    },
                    generated_panels={
                        name: values[generated_indices]
                        for name, values in panel_values.items()
                    },
                    source_ids=sources[panel_indices],
                )
            )

    report = warm_directional_mixture_report(
        evaluations,
        validation_pvalues=validation_pvalues,
        contamination_counts=contamination_counts,
        ledgers=int(ledgers),
    )
    report["protocol_variant"] = "frozen_same_session_reference_directional_gate"
    report["population"] = {
        "reference_human_rows": int(np.sum(reference_human)),
        "held_human_rows": int(np.sum(held_human)),
        "held_sessions": int(len(held_sessions)),
        "generator_cells": cell_ids,
    }
    report["human_split_receipts"] = split_receipts
    return report


def _two_sided_statistic(
    left: np.ndarray,
    right: np.ndarray,
    scale_reference: np.ndarray,
) -> dict[str, float]:
    scale = _robust_scale(scale_reference)
    mean_shift = float(
        np.max(np.abs(np.mean(left, axis=0) - np.mean(right, axis=0)) / scale)
    )
    left_std = np.maximum(np.std(left, axis=0), 1e-8)
    right_std = np.maximum(np.std(right, axis=0), 1e-8)
    spread_shift = float(np.max(np.abs(np.log(left_std / right_std))))
    distance = standardized_panel_w1(left, right, scale_reference=scale_reference)
    return {
        "mean_shift_max": mean_shift,
        "spread_log_ratio_max": spread_shift,
        "standardized_mean_w1": distance,
        "complete_max": float(max(mean_shift, spread_shift, distance)),
    }


def warm_smoke_report(
    bundle: DescriptorBundle,
    *,
    installation_key: str,
    session_id: str,
    panel: str = "full",
    query_origin: str = "generated",
    sample_rows: int = 32,
    null_draws: int = 2048,
    seed: int = 7,
) -> dict[str, Any]:
    """Small two-bag convenience diagnostic with trusted matching history.

    This API is retained for quick descriptor-bundle checks.  It is not the
    final multi-panel directional gate; use the cross-fit/gate/mixture
    functions above to reproduce that protocol.
    """

    if int(sample_rows) < 4:
        raise ValueError("sample_rows must be at least four")
    if int(null_draws) < 1:
        raise ValueError("null_draws must be positive")
    if query_origin not in {"human", "generated"}:
        raise ValueError("query_origin must be 'human' or 'generated'")
    features = bundle.panel(panel)
    origins = np.asarray(bundle.origin).astype(str)
    keys = np.asarray(bundle.installation_key).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    matching = (keys == str(installation_key)) & (sessions == str(session_id))
    reference = features[matching & (origins == "human")]
    query = features[matching & (origins == query_origin)]
    if query_origin == "human":
        order = np.argsort(
            [
                _stable_seed(seed, installation_key, session_id, index)
                for index in range(len(reference))
            ]
        )
        midpoint = len(order) // 2
        query = reference[order[:midpoint]]
        reference = reference[order[midpoint:]]
    if min(len(reference), len(query)) < int(sample_rows):
        raise ValueError(
            "the matching trusted reference and query must each contain at least "
            f"{sample_rows} rows"
        )
    rng = np.random.default_rng(
        _stable_seed(seed, installation_key, session_id, panel)
    )
    sample_rows = int(sample_rows)
    query_take = rng.choice(len(query), size=sample_rows, replace=False)
    reference_take = rng.choice(len(reference), size=sample_rows, replace=False)
    observed = _two_sided_statistic(
        reference[reference_take],
        query[query_take],
        reference,
    )
    null = np.empty(int(null_draws), dtype=np.float64)
    for draw in range(int(null_draws)):
        left = rng.choice(len(reference), size=sample_rows, replace=True)
        right = rng.choice(len(reference), size=sample_rows, replace=True)
        null[draw] = _two_sided_statistic(reference[left], reference[right], reference)[
            "complete_max"
        ]
    pvalue = float(
        (1 + np.sum(null >= observed["complete_max"])) / (len(null) + 1)
    )
    return {
        "schema": "abcurves.warm_smoke.v1",
        "protocol_variant": "reduced_two_bag_same_session_smoke",
        "not_release_protocol": True,
        "question": "distributional difference from trusted same-session human history",
        "threat_model": {
            "target_clean_history_used": True,
            "reference_scope": "same installation key and same session",
            "query_origin": query_origin,
        },
        "identity_semantics": IDENTITY_NOTE,
        "panel": panel,
        "reference_rows_available": int(len(reference)),
        "query_rows_available": int(len(query)),
        "bag_rows": sample_rows,
        "null_draws": int(null_draws),
        "statistics": observed,
        "empirical_pvalue": pvalue,
        "minimum_attainable_pvalue": float(1 / (int(null_draws) + 1)),
        "not_a_cold_detector": True,
        "final_directional_gate": False,
        "interpretation": (
            "This p-value is conditional on trusted matching history. It says "
            "nothing about the false-positive rate for a person whose clean "
            "movement has never been observed."
        ),
    }


__all__ = [
    "WARM_ALPHA_GRID",
    "WARM_BAG_ROWS",
    "WARM_CONTAMINATION_COUNTS",
    "WARM_LDA_SHRINK",
    "WARM_NEIGHBORS",
    "WARM_NULL_CALIBRATION_DRAWS",
    "WARM_NULL_FIT_DRAWS",
    "WARM_PANEL_NAMES",
    "WARM_SUBGROUP_COUNTS",
    "WarmDirectionalCalibration",
    "WarmDirectionalGate",
    "WarmDirectionalModel",
    "WarmLinearDirection",
    "WarmMixtureCell",
    "draw_matched_human_null_indices",
    "fit_cross_fitted_warm_directions",
    "fit_warm_directional_calibration",
    "fit_warm_directional_gate",
    "matched_same_session_pools",
    "tune_warm_alpha",
    "warm_direction_stat_names",
    "warm_directional_bag_statistics",
    "warm_directional_mixture_report",
    "warm_mixture_masks",
    "warm_reference_held_report",
    "warm_smoke_report",
]
