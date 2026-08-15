"""Specificity-first cold attribution with complete-key holdout.

The queried installation key is absent from direction fitting, route selection,
and human threshold calibration. Generated examples from other keys may teach a
direction, but generated outcomes never select the decision boundary. This is
the relevant protocol when there is no trusted clean history for the collection
identity being queried.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import ndtri, xlogy

from .bundle import DescriptorBundle, IDENTITY_NOTE


BAG_STAT_NAMES = (
    "curve_max",
    "concordant_curve",
    "dense_mean",
    "sparse_tail",
    "subgroup",
)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    location = np.median(array, axis=0)
    q25, q75 = np.quantile(array, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = array.std(axis=0)
    scale = np.where(scale >= 1e-8, scale, np.where(fallback >= 1e-8, fallback, 1.0))
    return location, scale


def _empirical_upper_rank(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    values = np.asarray(query, dtype=np.float64).reshape(-1)
    count = np.searchsorted(ref, values, side="right")
    return np.clip((count + 0.5) / float(len(ref) + 1), 1e-6, 1.0 - 1e-6)


@dataclass(frozen=True)
class _Direction:
    location: np.ndarray
    scale: np.ndarray
    weight: np.ndarray

    def project(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float64) - self.location) / self.scale
        return standardized @ self.weight


def _fit_direction(human: np.ndarray, generated: np.ndarray, *, ridge: float) -> _Direction:
    h = np.asarray(human, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    if h.ndim != 2 or g.ndim != 2 or h.shape[1] != g.shape[1]:
        raise ValueError("human and generated direction matrices must have matching widths")
    if min(len(h), len(g)) < 2:
        raise ValueError("direction fitting needs at least two rows from each class")
    location, scale = _robust_location_scale(h)
    hs = (h - location) / scale
    gs = (g - location) / scale
    covariance = np.cov(hs, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    width = hs.shape[1]
    penalty = max(float(ridge), 0.0)
    covariance = np.asarray(covariance, dtype=np.float64) + np.eye(width) * penalty
    delta = np.mean(gs, axis=0) - np.mean(hs, axis=0)
    weight = np.linalg.solve(covariance, delta)
    norm = float(np.sqrt(weight @ covariance @ weight))
    if norm < 1e-12:
        weight = np.ones(width, dtype=np.float64) / math.sqrt(width)
    else:
        weight /= norm
    if float(np.mean(gs @ weight)) < float(np.mean(hs @ weight)):
        weight *= -1.0
    return _Direction(location=location, scale=scale, weight=weight)


@dataclass(frozen=True)
class _ScoreModel:
    panel_names: tuple[str, ...]
    directions: tuple[_Direction, ...]
    reference_projection: tuple[np.ndarray, ...]
    ensemble_reference: np.ndarray

    def rank_channels(self, panels: Mapping[str, np.ndarray]) -> np.ndarray:
        projections = [
            direction.project(panels[name])
            for name, direction in zip(self.panel_names, self.directions)
        ]
        ranks = [
            _empirical_upper_rank(reference, projection)
            for reference, projection in zip(self.reference_projection, projections)
        ]
        standardized = []
        for reference, projection in zip(self.reference_projection, projections):
            center, spread = _robust_location_scale(reference[:, None])
            standardized.append((projection - center[0]) / spread[0])
        ensemble = np.mean(np.column_stack(standardized), axis=1)
        ranks.append(_empirical_upper_rank(self.ensemble_reference, ensemble))
        return np.column_stack(ranks)


def _fit_score_model(
    panel_names: Sequence[str],
    human_panels: Mapping[str, np.ndarray],
    generated_panels: Mapping[str, np.ndarray],
    *,
    ridge: float,
) -> _ScoreModel:
    directions = tuple(
        _fit_direction(human_panels[name], generated_panels[name], ridge=ridge)
        for name in panel_names
    )
    reference = tuple(
        direction.project(human_panels[name])
        for name, direction in zip(panel_names, directions)
    )
    standardized = []
    for values in reference:
        center, scale = _robust_location_scale(values[:, None])
        standardized.append((values - center[0]) / scale[0])
    ensemble_reference = np.mean(np.column_stack(standardized), axis=1)
    return _ScoreModel(tuple(panel_names), directions, reference, ensemble_reference)


def _berk_jones(ranks: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks, dtype=np.float64)
    pvalue = np.sort(1.0 - values, axis=1)
    rows = values.shape[1]
    fraction = (np.arange(1, rows + 1, dtype=np.float64) / float(rows))[None, :, None]
    clipped = np.clip(pvalue, 1e-9, 1.0 - 1e-9)
    valid = fraction > clipped
    divergence = xlogy(fraction, fraction / clipped) + xlogy(
        1.0 - fraction, (1.0 - fraction) / (1.0 - clipped)
    )
    divergence[~valid] = 0.0
    return float(rows) * np.max(divergence, axis=1)


def bag_statistics(ranks: np.ndarray) -> np.ndarray:
    """Five predeclared dense/sparse/subgroup statistics for rank channels."""

    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1] < 4 or values.shape[2] < 1:
        raise ValueError("ranks must have shape [bags, bag_rows>=4, channels>=1]")
    bag_rows = values.shape[1]
    z = ndtri(np.clip(values, 1e-6, 1.0 - 1e-6))
    curve_max = np.max(z, axis=(1, 2))
    concordant = (
        np.max(np.minimum(z[:, :, 0], z[:, :, 1]), axis=1)
        if z.shape[2] >= 2
        else np.max(z[:, :, 0], axis=1)
    )
    dense = np.max(np.mean(z, axis=1), axis=1)
    sparse = np.max(_berk_jones(values), axis=1)
    descending = np.sort(z, axis=1)[:, ::-1, :]
    cumulative = np.cumsum(descending, axis=1)
    total = cumulative[:, -1, :]
    schedule = sorted({k for k in (1, 2, 4, 8, 12, 16) if k < bag_rows})
    scans = []
    for k in schedule:
        top = cumulative[:, k - 1, :] / float(k)
        rest = (total - cumulative[:, k - 1, :]) / float(bag_rows - k)
        scans.append(math.sqrt(k * (bag_rows - k) / bag_rows) * (top - rest))
    subgroup = np.max(np.stack(scans, axis=1), axis=(1, 2))
    return np.column_stack([curve_max, concordant, dense, sparse, subgroup])


@dataclass(frozen=True)
class _Bag:
    key: str
    session: str
    scheme: str
    indices: np.ndarray

    @property
    def roster(self) -> tuple[int, ...]:
        return tuple(sorted(int(index) for index in self.indices))


def _chunks(order: Sequence[int], bag_rows: int) -> Iterable[np.ndarray]:
    for start in range(0, len(order) - bag_rows + 1, bag_rows):
        yield np.asarray(order[start : start + bag_rows], dtype=np.int64)


def _build_human_bags(bundle: DescriptorBundle, mask: np.ndarray, bag_rows: int) -> list[_Bag]:
    keys = np.asarray(bundle.installation_key).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    sources = np.asarray(bundle.source_id).astype(str)
    order = np.asarray(bundle.order, dtype=np.int64)
    tasks = np.asarray(bundle.task).astype(str)
    bags: list[_Bag] = []
    for key, session in sorted(set(zip(keys[mask], sessions[mask]))):
        local = np.flatnonzero(mask & (keys == key) & (sessions == session))
        if len(local) < bag_rows:
            continue
        orders = {
            "hash": sorted(local.tolist(), key=lambda i: _stable_seed(key, session, sources[i])),
            "chronological": sorted(local.tolist(), key=lambda i: (int(order[i]), sources[i])),
            "task_clustered": sorted(
                local.tolist(), key=lambda i: (tasks[i], _stable_seed(key, session, sources[i]))
            ),
        }
        for scheme, row_order in orders.items():
            bags.extend(
                _Bag(str(key), str(session), scheme, indices)
                for indices in _chunks(row_order, bag_rows)
            )
        chronological = orders["chronological"]
        bags.extend(
            _Bag(
                str(key),
                str(session),
                "chronological_rolling",
                np.asarray(chronological[start : start + bag_rows], dtype=np.int64),
            )
            for start in range(0, len(chronological) - bag_rows + 1)
        )
    return bags


@dataclass(frozen=True)
class _Calibration:
    location: np.ndarray
    scale: np.ndarray
    threshold: float
    component_thresholds: np.ndarray

    def score(self, statistics: np.ndarray) -> np.ndarray:
        evidence = np.maximum((np.asarray(statistics) - self.location) / self.scale, 0.0)
        return np.max(evidence, axis=1)


def _fit_calibration(statistics: np.ndarray) -> _Calibration:
    values = np.asarray(statistics, dtype=np.float64)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("human calibration must contain at least one bag")
    location, scale = _robust_location_scale(values)
    evidence = np.maximum((values - location) / scale, 0.0)
    score = np.max(evidence, axis=1)
    return _Calibration(
        location=location,
        scale=scale,
        threshold=np.nextafter(float(np.max(score)), math.inf),
        component_thresholds=np.nextafter(np.max(values, axis=0), math.inf),
    )


def _stats_for_bags(ranks: np.ndarray, bags: Sequence[_Bag]) -> np.ndarray:
    return np.concatenate([bag_statistics(ranks[bag.indices]) for bag in bags], axis=0)


def _candidate_mixture_stats(
    human_ranks: np.ndarray,
    generated_ranks: np.ndarray,
    bags: Sequence[_Bag],
    generated_keys: np.ndarray,
    generated_sessions: np.ndarray,
    held_key: str,
    counts: Sequence[int],
    *,
    ledgers: int,
) -> dict[int, list[np.ndarray]]:
    output = {int(count): [] for count in counts}
    key_pool = np.flatnonzero(generated_keys == held_key)
    if len(key_pool) == 0:
        return output
    for bag_index, bag in enumerate(bags):
        session_pool = np.flatnonzero(
            (generated_keys == held_key) & (generated_sessions == bag.session)
        )
        pool = session_pool if len(session_pool) else key_pool
        for ledger in range(int(ledgers)):
            rng = np.random.default_rng(_stable_seed("mixture", held_key, bag_index, ledger))
            generated_order = rng.choice(pool, size=max(counts), replace=len(pool) < max(counts))
            positions = rng.permutation(len(bag.indices))
            for count in counts:
                mixed = np.asarray(human_ranks[bag.indices], dtype=np.float64).copy()
                take = int(count)
                mixed[positions[:take]] = generated_ranks[generated_order[:take]]
                output[int(count)].append(bag_statistics(mixed))
    return output


# ---------------------------------------------------------------------------
# Frozen release protocol
# ---------------------------------------------------------------------------


def _fit_release_direction(human: np.ndarray, generated: np.ndarray) -> _Direction:
    """The predeclared pooled-covariance direction used by the final audit."""

    h = np.asarray(human, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    if h.ndim != 2 or g.ndim != 2 or h.shape[1] != g.shape[1]:
        raise ValueError("direction matrices must have matching descriptor widths")
    if min(len(h), len(g)) < 2:
        raise ValueError("release direction fitting needs at least two rows per class")
    location = h.mean(axis=0)
    scale = h.std(axis=0)
    scale[scale < 1e-6] = 1.0
    hs = (h - location) / scale
    gs = (g - location) / scale
    cov_h = np.atleast_2d(np.cov(hs, rowvar=False))
    cov_g = np.atleast_2d(np.cov(gs, rowvar=False))
    pooled = 0.5 * (cov_h + cov_g)
    shrink = 0.25
    regularized = (1.0 - shrink) * pooled + shrink * np.eye(h.shape[1])
    weight = np.linalg.solve(
        regularized + 1e-6 * np.eye(h.shape[1]),
        gs.mean(axis=0) - hs.mean(axis=0),
    )
    projected_scale = math.sqrt(max(float(weight @ cov_h @ weight), 1e-8))
    return _Direction(location=location, scale=scale, weight=weight / projected_scale)


def _ridge_fit(design: np.ndarray, values: np.ndarray, *, ridge: float = 10.0) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ values)


def _context_design(
    context: np.ndarray,
    task: np.ndarray,
    role: np.ndarray,
    *,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    task_levels: Sequence[str],
    role_levels: Sequence[str],
) -> np.ndarray:
    standardized = (np.asarray(context, dtype=np.float64) - context_mean) / context_scale
    tasks = np.asarray(task).astype(str)
    roles = np.asarray(role).astype(str)
    task_hot = np.column_stack([tasks == level for level in task_levels]).astype(np.float64)
    role_hot = np.column_stack([roles == level for level in role_levels]).astype(np.float64)
    return np.column_stack([np.ones(len(standardized)), standardized, task_hot, role_hot])


@dataclass(frozen=True)
class _ConditionalRanks:
    ranks: np.ndarray
    curve_score: np.ndarray


def _conditional_release_ranks(
    bundle: DescriptorBundle,
    raw_scores: np.ndarray,
    reference_human: np.ndarray,
) -> _ConditionalRanks:
    """Nuisance-residualize and Mondrian-rank the three frozen directions.

    The ridge is fit only on reference humans. Their residuals are generated by
    complete-key out-of-fold fits; the final reference-to-query transform is fit
    on every reference human. This is the exact information boundary used by
    the release audit.
    """

    if bundle.causal_context is None or bundle.target_role is None:
        raise ValueError(
            "an exact cold audit requires causal_context and target_role arrays"
        )
    scores = np.asarray(raw_scores, dtype=np.float64)
    context = np.asarray(bundle.causal_context, dtype=np.float64)
    tasks = np.asarray(bundle.task).astype(str)
    roles = np.asarray(bundle.target_role).astype(str)
    keys = np.asarray(bundle.installation_key).astype(str)
    reference_rows = np.flatnonzero(reference_human)
    if len(reference_rows) < 4:
        raise ValueError("exact cold nuisance fitting needs reference-human rows")
    reference_context = context[reference_rows]
    context_mean = reference_context.mean(axis=0)
    context_scale = reference_context.std(axis=0)
    context_scale[context_scale < 1e-8] = 1.0
    task_levels = tuple(sorted(np.unique(tasks[reference_rows]).tolist()))
    role_levels = tuple(sorted(np.unique(roles[reference_rows]).tolist()))
    design = _context_design(
        context,
        tasks,
        roles,
        context_mean=context_mean,
        context_scale=context_scale,
        task_levels=task_levels,
        role_levels=role_levels,
    )

    residual = np.empty_like(scores)
    reference_keys = sorted(np.unique(keys[reference_human]).tolist())
    if len(reference_keys) < 2:
        raise ValueError("exact cold nuisance fitting needs at least two reference keys")
    for key in reference_keys:
        held = reference_human & (keys == key)
        fit = reference_human & (keys != key)
        if np.sum(fit) < 2:
            raise ValueError("reference-key residual fold has too few fitting rows")
        beta = _ridge_fit(design[fit], scores[fit])
        residual[held] = scores[held] - design[held] @ beta
    beta = _ridge_fit(design[reference_human], scores[reference_human])
    non_reference = ~reference_human
    residual[non_reference] = scores[non_reference] - design[non_reference] @ beta

    reference_residual = residual[reference_human]
    location = np.median(reference_residual, axis=0)
    q25, q75 = np.quantile(reference_residual, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = reference_residual.std(axis=0)
    scale = np.where(scale >= 1e-8, scale, np.where(fallback >= 1e-8, fallback, 1.0))
    standardized = (residual - location) / scale
    ensemble = np.mean(standardized, axis=1)
    four = np.column_stack([residual, ensemble])
    curve_score = np.max(np.column_stack([standardized, ensemble]), axis=1)

    ranks = np.full_like(four, np.nan)
    strata = sorted(
        set(zip(tasks[reference_human].tolist(), roles[reference_human].tolist()))
        | set(zip(tasks[~reference_human].tolist(), roles[~reference_human].tolist()))
    )
    for task, role in strata:
        exact = reference_human & (tasks == task) & (roles == role)
        task_only = reference_human & (tasks == task)
        role_only = reference_human & (roles == role)
        if int(np.sum(exact)) >= 64:
            reference_take = exact
        elif int(np.sum(task_only)) >= 128:
            reference_take = task_only
        elif int(np.sum(role_only)) >= 128:
            reference_take = role_only
        else:
            reference_take = reference_human
        query_take = (tasks == task) & (roles == role)
        if np.any(query_take):
            for column in range(four.shape[1]):
                ranks[query_take, column] = _empirical_upper_rank(
                    four[reference_take, column], four[query_take, column]
                )
    if not np.all(np.isfinite(ranks)):
        raise ValueError("cold conditional ranks are non-finite")
    return _ConditionalRanks(ranks=ranks, curve_score=curve_score)


def _release_bags(
    bundle: DescriptorBundle,
    mask: np.ndarray,
    bag_rows: int,
    *,
    namespace: str,
    include_frozen_panel: bool,
) -> list[_Bag]:
    """Build the frozen hash/task/time challenges and every time-window start."""

    keys = np.asarray(bundle.installation_key).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    sources = np.asarray(bundle.source_id).astype(str)
    order = np.asarray(bundle.order, dtype=np.int64)
    blocks = (
        np.asarray(bundle.block_order, dtype=np.int64)
        if bundle.block_order is not None
        else np.zeros(bundle.rows, dtype=np.int64)
    )
    tasks = np.asarray(bundle.task).astype(str)
    panel = (
        np.asarray(bundle.audit_panel, dtype=bool)
        if bundle.audit_panel is not None
        else np.zeros(bundle.rows, dtype=bool)
    )
    bags: list[_Bag] = []
    for key, session in sorted(set(zip(keys[mask], sessions[mask]))):
        local = np.flatnonzero(mask & (keys == key) & (sessions == session))
        if len(local) < bag_rows:
            continue
        orders = {
            "hash": sorted(
                local.tolist(),
                key=lambda i: _stable_seed(namespace, session, sources[i]),
            ),
            "chronological": sorted(
                local.tolist(), key=lambda i: (int(blocks[i]), int(order[i]), sources[i])
            ),
            "task_clustered": sorted(
                local.tolist(),
                key=lambda i: (tasks[i], _stable_seed(namespace, sources[i])),
            ),
        }
        for scheme, row_order in orders.items():
            bags.extend(
                _Bag(str(key), str(session), scheme, indices)
                for indices in _chunks(row_order, bag_rows)
            )
        chronological = orders["chronological"]
        bags.extend(
            _Bag(
                str(key),
                str(session),
                "chronological_rolling",
                np.asarray(chronological[start : start + bag_rows], dtype=np.int64),
            )
            for start in range(0, len(chronological) - bag_rows + 1)
        )
        if include_frozen_panel:
            frozen = np.flatnonzero(mask & panel & (keys == key) & (sessions == session))
            if len(frozen) not in (0, bag_rows):
                raise ValueError(
                    f"audit_panel must select exactly {bag_rows} human rows per held session"
                )
            if len(frozen) == bag_rows:
                bags.append(_Bag(str(key), str(session), "frozen_panel", frozen))
    return bags


def _mixture_masks(
    source_ids: np.ndarray,
    counts: Sequence[int],
    *,
    ledgers: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    sources = np.asarray(source_ids).astype(str)
    bag_rows = len(sources)
    if len(np.unique(sources)) != bag_rows:
        raise ValueError("mixture panel must contain unique physical sources")
    masks: list[np.ndarray] = []
    metadata: list[dict[str, int]] = []
    for count in counts:
        repetitions = 1 if int(count) == bag_rows else int(ledgers)
        for ledger in range(repetitions):
            row_order = sorted(
                range(bag_rows),
                key=lambda row: _stable_seed("within-bag-ledger-v1", ledger, sources[row]),
            )
            mask = np.zeros(bag_rows, dtype=bool)
            mask[np.asarray(row_order[: int(count)], dtype=np.int64)] = True
            masks.append(mask)
            metadata.append({"generated_rows": int(count), "ledger": int(ledger)})
    output = np.stack(masks, axis=0)
    for ledger in range(int(ledgers)):
        local = [
            output[index]
            for index, row in enumerate(metadata)
            if row["ledger"] == ledger and row["generated_rows"] != bag_rows
        ]
        for left, right in zip(local, local[1:]):
            if np.any(left & ~right):
                raise AssertionError("mixture ledgers must be nested")
    return output, metadata


def cold_reference_held_report(
    bundle: DescriptorBundle,
    *,
    panels: Sequence[str] = ("trajectory", "texture", "full"),
    bag_rows: int = 32,
    contamination_counts: Sequence[int] | None = None,
    ledgers: int = 16,
) -> dict[str, Any]:
    """Run the final predeclared reference/held/cell cold protocol.

    This is the reusable implementation behind the release result. Reference
    identities form the human envelope. Only identities explicitly marked
    ``held`` are queried, and their complete key (every session and generator
    row) is absent from direction fitting, route selection, and calibration.
    Each generator cell is evaluated with that cell also excluded from the
    generator-aware direction fit.
    """

    required = {
        "population_role": bundle.population_role,
        "generator_cell": bundle.generator_cell,
        "target_role": bundle.target_role,
        "causal_context": bundle.causal_context,
        "block_order": bundle.block_order,
        "audit_panel": bundle.audit_panel,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(
            "exact cold audit requires enriched bundle arrays: " + ", ".join(missing)
        )
    bag_rows = int(bag_rows)
    if bag_rows < 4 or int(ledgers) < 1:
        raise ValueError("bag_rows must be at least four and ledgers must be positive")
    panel_names = tuple(panels)
    if len(panel_names) != 3:
        raise ValueError("the predeclared cold route requires trajectory, texture and full panels")
    panel_values = {name: bundle.panel(name) for name in panel_names}
    counts = tuple(
        sorted(
            {
                int(value)
                for value in (
                    contamination_counts
                    if contamination_counts is not None
                    else (1, 2, 4, 8, 12, 16, 24, bag_rows)
                )
                if 0 < int(value) <= bag_rows
            }
        )
    )
    if not counts:
        raise ValueError("no contamination count lies inside the bag")

    origins = np.asarray(bundle.origin).astype(str)
    keys = np.asarray(bundle.installation_key).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    sources = np.asarray(bundle.source_id).astype(str)
    population = np.asarray(bundle.population_role).astype(str)
    cells = np.asarray(bundle.generator_cell).astype(str)
    audit_panel = np.asarray(bundle.audit_panel, dtype=bool)
    human = origins == "human"
    generated = origins == "generated"
    reference_human = human & (population == "reference")
    held_human = human & (population == "held")
    held_generated = generated & (population == "held")
    if np.any(generated & (population == "reference")):
        raise ValueError("reference population must contain humans only")
    reference_keys = sorted(np.unique(keys[reference_human]).tolist())
    held_keys = sorted(np.unique(keys[held_human]).tolist())
    if set(reference_keys) & set(held_keys):
        raise ValueError("reference and held installation keys must be disjoint")
    if len(reference_keys) < 2 or len(held_keys) < 2:
        raise ValueError("exact cold audit needs at least two reference and two held keys")
    if set(np.unique(keys[held_generated]).tolist()) - set(held_keys):
        raise ValueError("generated held rows must be bound to declared held keys")
    cell_ids = sorted(set(cells[held_generated]) - {"", "human", "none"})
    if len(cell_ids) < 2:
        raise ValueError("exact cold audit needs at least two predeclared generator cells")
    if np.any(~audit_panel & held_generated):
        raise ValueError("every generated audit row must belong to the frozen audit panel")

    reference_bags = _release_bags(
        bundle,
        reference_human,
        bag_rows,
        namespace="train",
        include_frozen_panel=False,
    )
    development_bags = _release_bags(
        bundle,
        held_human,
        bag_rows,
        namespace="dev",
        include_frozen_panel=True,
    )
    if not reference_bags:
        raise ValueError("reference humans do not form any calibration bags")

    human_rows: list[tuple[_Bag, str, str, bool, np.ndarray]] = []
    mixture_rows: list[tuple[int, bool, np.ndarray]] = []
    fold_rows: list[dict[str, Any]] = []
    human_curve_evaluations = 0
    human_curve_flags = 0
    generated_curve_evaluations = 0
    generated_curve_flags = 0

    for held_key in held_keys:
        target_bags = [bag for bag in development_bags if bag.key == held_key]
        other_bags = [bag for bag in development_bags if bag.key != held_key]
        target_human = held_human & (keys == held_key)
        direction_human = held_human & audit_panel & (keys != held_key)
        if not target_bags or np.sum(direction_human) < 2:
            raise ValueError(f"held key {held_key!r} lacks complete audit panels")
        held_sessions = sorted(np.unique(sessions[target_human]).tolist())

        for held_cell in cell_ids:
            direction_generated = (
                held_generated
                & (keys != held_key)
                & (cells != held_cell)
            )
            # These assertions sit next to the fit call so a future refactor
            # cannot silently become session- or row-level holdout.
            fit_rows = direction_human | direction_generated
            if np.any(keys[fit_rows] == held_key):
                raise AssertionError("held installation key leaked into direction fitting")
            if np.any(cells[direction_generated] == held_cell):
                raise AssertionError("held generator cell leaked into direction fitting")
            models = tuple(
                _fit_release_direction(
                    panel_values[name][direction_human],
                    panel_values[name][direction_generated],
                )
                for name in panel_names
            )
            raw_scores = np.column_stack(
                [model.project(panel_values[name]) for name, model in zip(panel_names, models)]
            )
            conditional = _conditional_release_ranks(bundle, raw_scores, reference_human)

            calibration_bags = reference_bags + other_bags
            calibration_stats = _stats_for_bags(conditional.ranks, calibration_bags)
            calibration = _fit_calibration(calibration_stats)
            component_thresholds = np.nextafter(
                np.max(calibration_stats, axis=0),
                np.full(len(BAG_STAT_NAMES), math.inf),
            )
            calibration_human_rows = reference_human | (held_human & (keys != held_key))
            curve_threshold = np.nextafter(
                float(np.max(conditional.curve_score[calibration_human_rows])), math.inf
            )
            target_stats = _stats_for_bags(conditional.ranks, target_bags)
            target_flags = calibration.score(target_stats) > calibration.threshold
            target_components = target_stats > component_thresholds[None, :]
            for bag, flag, components in zip(
                target_bags, target_flags, target_components
            ):
                human_rows.append((bag, held_key, held_cell, bool(flag), components))
            target_curve_flags = conditional.curve_score[target_human] > curve_threshold
            human_curve_evaluations += int(np.sum(target_human))
            human_curve_flags += int(np.sum(target_curve_flags))

            fold_mixture_evaluations = {count: 0 for count in counts}
            fold_mixture_flags = {count: 0 for count in counts}
            for session in held_sessions:
                human_indices = np.flatnonzero(
                    target_human & audit_panel & (sessions == session)
                )
                generated_indices = np.flatnonzero(
                    held_generated
                    & (keys == held_key)
                    & (sessions == session)
                    & (cells == held_cell)
                )
                if len(human_indices) != bag_rows or len(generated_indices) != bag_rows:
                    raise ValueError(
                        f"held session {session!r}, cell {held_cell!r} must contain "
                        f"exactly {bag_rows} matched human/generated audit rows"
                    )
                generated_by_source = {
                    source: index for source, index in zip(sources[generated_indices], generated_indices)
                }
                if len(generated_by_source) != bag_rows:
                    raise ValueError("generated audit panel contains duplicate physical sources")
                try:
                    aligned_generated = np.asarray(
                        [generated_by_source[source] for source in sources[human_indices]],
                        dtype=np.int64,
                    )
                except KeyError as exc:
                    raise ValueError("human/generated audit panels do not share source bindings") from exc
                masks, metadata = _mixture_masks(
                    sources[human_indices], counts, ledgers=int(ledgers)
                )
                mixed = np.where(
                    masks[:, :, None],
                    conditional.ranks[aligned_generated][None, :, :],
                    conditional.ranks[human_indices][None, :, :],
                )
                statistics = bag_statistics(mixed)
                flags = calibration.score(statistics) > calibration.threshold
                components = statistics > component_thresholds[None, :]
                for row, flag, component in zip(metadata, flags, components):
                    count = int(row["generated_rows"])
                    mixture_rows.append((count, bool(flag), component))
                    fold_mixture_evaluations[count] += 1
                    fold_mixture_flags[count] += int(flag)
                generated_curve = conditional.curve_score[aligned_generated]
                generated_curve_evaluations += len(generated_curve)
                generated_curve_flags += int(np.sum(generated_curve > curve_threshold))

            fold_rows.append(
                {
                    "held_installation_key": held_key,
                    "held_sessions": held_sessions,
                    "held_generator_cell": held_cell,
                    "direction_human_keys": sorted(np.unique(keys[direction_human]).tolist()),
                    "direction_generator_cells": sorted(
                        np.unique(cells[direction_generated]).tolist()
                    ),
                    "calibration_human_keys": sorted(
                        np.unique(keys[calibration_human_rows]).tolist()
                    ),
                    "direction_human_rows": int(np.sum(direction_human)),
                    "direction_generated_rows": int(np.sum(direction_generated)),
                    "calibration_human_bags": int(len(calibration_bags)),
                    "held_human_bags": int(len(target_bags)),
                    "held_human_flags": int(np.sum(target_flags)),
                    "complete_threshold": float(calibration.threshold),
                    "candidate_power_counts": {
                        str(count): {
                            "evaluations": int(fold_mixture_evaluations[count]),
                            "flags": int(fold_mixture_flags[count]),
                        }
                        for count in counts
                    },
                }
            )

    flags = [row for row in human_rows if row[3]]
    flagged_keys = sorted({row[1] for row in flags})
    unique_rosters = {row[0].roster for row in human_rows}
    flagged_rosters = {row[0].roster for row in flags}
    component_frontier = []
    for index, name in enumerate(BAG_STAT_NAMES):
        human_component = [row for row in human_rows if bool(row[4][index])]
        component_frontier.append(
            {
                "rule": name,
                "held_human_cell_flags": int(len(human_component)),
                "held_keys_flagged": int(len({row[1] for row in human_component})),
                "candidate_power": [
                    {
                        "generated_rows": int(count),
                        "evaluations": int(
                            sum(row[0] == count for row in mixture_rows)
                        ),
                        "flag_rate": (
                            float(
                                np.mean(
                                    [
                                        bool(row[2][index])
                                        for row in mixture_rows
                                        if row[0] == count
                                    ]
                                )
                            )
                            if any(row[0] == count for row in mixture_rows)
                            else None
                        ),
                    }
                    for count in counts
                ],
            }
        )
    power = []
    for count in counts:
        local = [row for row in mixture_rows if row[0] == count]
        power.append(
            {
                "generated_rows": int(count),
                "fraction": float(count / bag_rows),
                "evaluations": int(len(local)),
                "flags": int(sum(row[1] for row in local)),
                "flag_rate": float(np.mean([row[1] for row in local])) if local else None,
            }
        )
    return {
        "schema": "abcurves.cold_reference_held_cells.v1",
        "question": "generated attribution for wholly unseen installation keys",
        "protocol_variant": "frozen_reference_held_cell_leaveout",
        "threat_model": {
            "target_clean_history_used": False,
            "holdout_unit": "complete held installation key and every session under it",
            "direction_training": (
                "frozen-panel humans from other held keys and generated rows from "
                "other held keys and other predeclared generator cells"
            ),
            "route_selection": (
                "trajectory, texture, Full panels; pooled directions; nuisance ridge; "
                "Mondrian ranks; and five bag statistics fixed before held outcomes"
            ),
            "threshold_selection": (
                "strictly above the maximum complete-search score from reference humans "
                "and non-target held humans"
            ),
            "generated_outcomes_used_for_threshold_selection": False,
            "panels": list(panel_names),
            "bag_statistics": list(BAG_STAT_NAMES),
            "bag_rows": bag_rows,
            "mixture_ledgers": int(ledgers),
        },
        "identity_semantics": IDENTITY_NOTE,
        "population": {
            "reference_human_rows": int(np.sum(reference_human)),
            "reference_installation_keys": int(len(reference_keys)),
            "held_human_rows": int(np.sum(held_human)),
            "held_installation_keys": int(len(held_keys)),
            "generator_cells": cell_ids,
        },
        "held_human": {
            "keys_evaluated": int(len(held_keys)),
            "keys_flagged": int(len(flagged_keys)),
            "bag_cell_evaluations": int(len(human_rows)),
            "bag_cell_flags": int(len(flags)),
            "unique_rosters": int(len(unique_rosters)),
            "unique_flagged_rosters": int(len(flagged_rosters)),
        },
        "curve_envelope": {
            "human_curve_evaluations": int(human_curve_evaluations),
            "human_flags": int(human_curve_flags),
            "generated_curve_evaluations": int(generated_curve_evaluations),
            "generated_flags": int(generated_curve_flags),
        },
        "candidate_power": power,
        "component_frontier": component_frontier,
        "folds": fold_rows,
        "interpretation": (
            "A useful cold attribution route needs nonzero held-candidate power and an "
            "acceptable false-positive result on complete unseen keys. No target-key "
            "human row participates in fitting, route selection, or calibration."
        ),
    }


def cold_smoke_report(
    bundle: DescriptorBundle,
    *,
    panels: Sequence[str] | None = None,
    bag_rows: int = 32,
    contamination_counts: Sequence[int] | None = None,
    ledgers: int = 4,
    ridge: float = 1.0,
) -> dict[str, Any]:
    """Run a reduced no-target-history implementation check.

    Every session and every generated continuation bound to the held key is
    removed from direction fitting. The complete-search threshold is placed
    strictly above all calibration-human bags from other keys. Candidate power
    is reported afterwards and cannot influence that threshold. This compact
    all-keys route is useful while constructing a descriptor bundle, but it is
    not the frozen reference/held/cell release protocol.
    """

    bag_rows = int(bag_rows)
    if bag_rows < 4:
        raise ValueError("bag_rows must be at least four")
    if int(ledgers) < 1:
        raise ValueError("ledgers must be positive")
    panel_names = tuple(panels or bundle.panel_slices.keys())
    if not panel_names:
        raise ValueError("at least one descriptor panel is required")
    for panel in panel_names:
        bundle.panel(panel)
    counts = tuple(
        sorted(
            set(
                int(value)
                for value in (
                    contamination_counts
                    if contamination_counts is not None
                    else (1, 2, 4, 8, 12, 16, 24, bag_rows)
                )
                if 0 < int(value) <= bag_rows
            )
        )
    )
    if not counts:
        raise ValueError("no contamination count lies inside the bag")

    origins = np.asarray(bundle.origin).astype(str)
    keys = np.asarray(bundle.installation_key).astype(str)
    sessions = np.asarray(bundle.session_id).astype(str)
    human = origins == "human"
    generated = origins == "generated"
    held_keys = sorted(np.unique(keys[human]).tolist())
    if len(held_keys) < 3:
        raise ValueError("a cold leave-key-out audit needs at least three human keys")
    if np.sum(generated) < 2:
        raise ValueError("a cold generated-attribution audit needs labeled development output")

    all_panels = {name: bundle.panel(name) for name in panel_names}
    fold_rows: list[dict[str, Any]] = []
    human_evaluations = 0
    human_flags = 0
    flagged_keys: set[str] = set()
    tested_rosters: set[tuple[int, ...]] = set()
    flagged_rosters: set[tuple[int, ...]] = set()
    component_flags = np.zeros(len(BAG_STAT_NAMES), dtype=np.int64)
    power_flags = {count: 0 for count in counts}
    power_evaluations = {count: 0 for count in counts}

    for held_key in held_keys:
        fit_human = human & (keys != held_key)
        fit_generated = generated & (keys != held_key)
        held_human = human & (keys == held_key)
        # This assertion is intentionally near the fit call: a future refactor
        # must not silently turn complete-key holdout into row-level CV.
        if np.any(keys[fit_human] == held_key) or np.any(keys[fit_generated] == held_key):
            raise AssertionError("held installation key leaked into direction fitting")
        if min(np.sum(fit_human), np.sum(fit_generated)) < 2:
            continue

        model = _fit_score_model(
            panel_names,
            {name: values[fit_human] for name, values in all_panels.items()},
            {name: values[fit_generated] for name, values in all_panels.items()},
            ridge=float(ridge),
        )
        all_ranks = model.rank_channels(all_panels)
        calibration_bags = _build_human_bags(bundle, fit_human, bag_rows)
        held_bags = _build_human_bags(bundle, held_human, bag_rows)
        if not calibration_bags or not held_bags:
            continue
        calibration_stats = _stats_for_bags(all_ranks, calibration_bags)
        calibration = _fit_calibration(calibration_stats)
        held_stats = _stats_for_bags(all_ranks, held_bags)
        scores = calibration.score(held_stats)
        flags = scores > calibration.threshold
        component = held_stats > calibration.component_thresholds[None, :]
        component_flags += np.sum(component, axis=0, dtype=np.int64)

        human_evaluations += len(held_bags)
        human_flags += int(np.sum(flags))
        if np.any(flags):
            flagged_keys.add(held_key)
        for bag, flag in zip(held_bags, flags):
            tested_rosters.add(bag.roster)
            if bool(flag):
                flagged_rosters.add(bag.roster)

        generated_ranks = all_ranks[generated]
        mixtures = _candidate_mixture_stats(
            all_ranks,
            generated_ranks,
            held_bags,
            keys[generated],
            sessions[generated],
            held_key,
            counts,
            ledgers=int(ledgers),
        )
        fold_power: dict[str, Any] = {}
        for count, rows in mixtures.items():
            if not rows:
                continue
            statistics = np.concatenate(rows, axis=0)
            candidate_flags = calibration.score(statistics) > calibration.threshold
            power_flags[count] += int(np.sum(candidate_flags))
            power_evaluations[count] += int(len(candidate_flags))
            fold_power[str(count)] = {
                "evaluations": int(len(candidate_flags)),
                "flags": int(np.sum(candidate_flags)),
            }
        fold_rows.append(
            {
                "held_installation_key": held_key,
                "held_sessions": sorted(np.unique(sessions[held_human]).tolist()),
                "direction_human_rows": int(np.sum(fit_human)),
                "direction_generated_rows": int(np.sum(fit_generated)),
                "calibration_human_bags": int(len(calibration_bags)),
                "held_human_bags": int(len(held_bags)),
                "held_human_flags": int(np.sum(flags)),
                "complete_threshold": float(calibration.threshold),
                "candidate_power_counts": fold_power,
            }
        )

    if not fold_rows:
        raise ValueError("no held key had enough rows to construct calibration and query bags")
    power = [
        {
            "generated_rows": int(count),
            "fraction": float(count / bag_rows),
            "evaluations": int(power_evaluations[count]),
            "flags": int(power_flags[count]),
            "flag_rate": (
                float(power_flags[count] / power_evaluations[count])
                if power_evaluations[count]
                else None
            ),
        }
        for count in counts
    ]
    return {
        "schema": "abcurves.cold_smoke.v1",
        "protocol_variant": "reduced_all_keys_leave_one_out_smoke",
        "not_release_protocol": True,
        "question": "generated attribution for a previously unseen installation key",
        "threat_model": {
            "target_clean_history_used": False,
            "holdout_unit": "complete persistent installation key and every session under it",
            "direction_training": (
                "labeled human and generated development rows from non-target keys only"
            ),
            "route_selection": (
                "descriptor panels and bag statistics must be fixed before querying "
                "the held key; the released default uses every declared panel and "
                "the five predeclared statistics"
            ),
            "threshold_selection": (
                "strictly above the maximum complete-search score from non-target humans"
            ),
            "generated_outcomes_used_for_threshold_selection": False,
            "bag_rows": bag_rows,
            "panels": list(panel_names),
        },
        "identity_semantics": IDENTITY_NOTE,
        "held_human": {
            "keys_evaluated": int(len(fold_rows)),
            "keys_flagged": int(len(flagged_keys)),
            "bag_evaluations": int(human_evaluations),
            "bag_flags": int(human_flags),
            "unique_rosters": int(len(tested_rosters)),
            "unique_flagged_rosters": int(len(flagged_rosters)),
            "component_flags": {
                name: int(value) for name, value in zip(BAG_STAT_NAMES, component_flags)
            },
        },
        "candidate_power": power,
        "folds": fold_rows,
        "interpretation": (
            "A useful cold attribution rule must have both nonzero held-candidate "
            "power and an acceptable false-positive result on wholly excluded human "
            "keys. Zero observed flags is finite-sample evidence, never a universal "
            "zero-false-positive guarantee."
        ),
    }


def cold_leave_key_out_report(
    bundle: DescriptorBundle,
    *,
    panels: Sequence[str] = ("trajectory", "texture", "full"),
    bag_rows: int = 32,
    contamination_counts: Sequence[int] | None = None,
    ledgers: int = 16,
) -> dict[str, Any]:
    """Run the named cold release protocol on an enriched audit bundle.

    Requiring the population, generator-cell, causal-context and frozen-panel
    declarations prevents a small convenience study from being mistaken for
    the evidence protocol.
    """

    return cold_reference_held_report(
        bundle,
        panels=panels,
        bag_rows=bag_rows,
        contamination_counts=contamination_counts,
        ledgers=ledgers,
    )


__all__ = [
    "BAG_STAT_NAMES",
    "bag_statistics",
    "cold_leave_key_out_report",
    "cold_reference_held_report",
    "cold_smoke_report",
]
