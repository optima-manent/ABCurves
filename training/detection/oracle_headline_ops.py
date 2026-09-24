"""Exact historical oracle-headline operations, extracted from the frozen scorer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pathlib import Path

from typing import Any

import math

import numpy as np

PUBLIC_REPO = None
C2ST_FOLDS = 5
C2ST_REPEATS = 3
C2ST_BOOTSTRAP = 200
_W1_GRID = np.linspace(0.0, 1.0, 201, dtype=np.float64)[1:-1]
_SCALE_EPS = 1e-6

class TournamentJudgeError(ValueError):
    """The archived study contract was violated."""

def load_public_judges(_unused=None):
    from abcurves import judges
    return judges

def _as_strings(values: Sequence[Any] | np.ndarray, expected: int, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1).astype(str)
    if len(array) != int(expected):
        raise TournamentJudgeError(f"{name} has {len(array)} rows, expected {expected}")
    if np.any(np.char.strip(array) == ""):
        raise TournamentJudgeError(f"{name} contains an empty identity")
    return array

def _feature_matrix(values: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise TournamentJudgeError(f"{name} must be a two-dimensional feature matrix")
    if width is not None and array.shape[1] != int(width):
        raise TournamentJudgeError(
            f"{name} has feature width {array.shape[1]}, expected {width}"
        )
    if len(array) < 2:
        raise TournamentJudgeError(f"{name} must contain at least two rows")
    if not np.all(np.isfinite(array)):
        raise TournamentJudgeError(f"{name} contains non-finite values")
    return array

def derive_feature_panels_from_arrays(
    raw_dxdy: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    radii: np.ndarray,
    *,
    prefix_dxdy: np.ndarray | None = None,
    prefix_mask: np.ndarray | None = None,
    public_repo: str | Path = PUBLIC_REPO,
) -> dict[str, np.ndarray]:
    """Derive the exact public trajectory14, texture19, and full49 panels."""

    judges = load_public_judges(public_repo)
    raw = np.asarray(raw_dxdy, dtype=np.float32)
    valid = np.asarray(mask, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[2] != 2 or valid.shape != raw.shape[:2]:
        raise TournamentJudgeError("raw/mask shapes must be [N,T,2] and [N,T]")
    n = len(raw)
    target_array = np.asarray(targets, dtype=np.float64)
    radius_array = np.asarray(radii, dtype=np.float64).reshape(-1)
    if target_array.shape != (n, 2) or radius_array.shape != (n,):
        raise TournamentJudgeError("target/radius shapes disagree with raw rows")
    if (
        not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(valid))
        or not np.all(np.isfinite(target_array))
        or not np.all(np.isfinite(radius_array))
        or np.any(radius_array <= 0.0)
    ):
        raise TournamentJudgeError("feature inputs must be finite and radii positive")

    streams = [raw[index][valid[index] > 0.5] for index in range(n)]
    trajectory = np.asarray(
        judges.trajectory_features(
            streams,
            [target_array[index] for index in range(n)],
            radius_array.tolist(),
        ),
        dtype=np.float64,
    )
    texture = np.asarray(judges.texture_features(raw, valid), dtype=np.float64)
    full = np.asarray(
        judges.full_system_features(
            raw,
            valid,
            target_array,
            radius_array,
            prefix_dxdy=prefix_dxdy,
            prefix_mask=prefix_mask,
        ),
        dtype=np.float64,
    )
    if trajectory.shape != (n, 14) or texture.shape != (n, 19) or full.shape != (n, 49):
        raise TournamentJudgeError(
            f"public feature shapes changed: {trajectory.shape}, {texture.shape}, {full.shape}"
        )
    if not np.array_equal(full[:, :14], trajectory) or not np.array_equal(
        full[:, 14:33], texture
    ):
        raise TournamentJudgeError("full49 no longer embeds trajectory14/texture19 exactly")
    return {"trajectory14": trajectory, "texture19": texture, "full49": full}

def wasserstein1(left: np.ndarray, right: np.ndarray) -> float:
    """The public judge's deterministic 199-quantile W1 approximation."""

    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if not len(a) or not len(b) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise TournamentJudgeError("W1 inputs must be non-empty and finite")
    return float(np.mean(np.abs(np.quantile(a, _W1_GRID) - np.quantile(b, _W1_GRID))))

def reference_std(reference_features: np.ndarray) -> np.ndarray:
    reference = _feature_matrix(reference_features, name="reference_features")
    scale = np.std(reference, axis=0, dtype=np.float64)
    scale[scale < _SCALE_EPS] = 1.0
    return scale

def standardized_w1_report(
    real_features: np.ndarray,
    comparison_features: np.ndarray,
    feature_names: Sequence[str],
    *,
    scale_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Public-style standardized W1 mean and complete per-feature readout.

    ``scale_reference`` may be a human-only feature matrix or a precomputed
    scale vector.  It is useful when many human/session/candidate bags must be
    placed on one comparable axis.  With ``None``, public behavior is used:
    standardize by the real bag's population standard deviation.
    """

    names = tuple(str(name) for name in feature_names)
    real = _feature_matrix(real_features, name="real_features", width=len(names))
    comparison = _feature_matrix(
        comparison_features, name="comparison_features", width=len(names)
    )
    if scale_reference is None:
        scale = reference_std(real)
        scale_source = "real_features_population_std"
    else:
        supplied = np.asarray(scale_reference, dtype=np.float64)
        if supplied.ndim == 1:
            if supplied.shape != (len(names),) or not np.all(np.isfinite(supplied)):
                raise TournamentJudgeError("precomputed W1 scale has the wrong shape/values")
            scale = supplied.copy()
            scale[scale < _SCALE_EPS] = 1.0
            scale_source = "supplied_human_scale_vector"
        else:
            scale = reference_std(
                _feature_matrix(supplied, name="scale_reference", width=len(names))
            )
            scale_source = "supplied_human_reference_population_std"
    values = np.asarray(
        [wasserstein1(real[:, j] / scale[j], comparison[:, j] / scale[j]) for j in range(len(names))],
        dtype=np.float64,
    )
    per_feature = {name: float(values[j]) for j, name in enumerate(names)}
    descending = sorted(per_feature.items(), key=lambda item: (-item[1], item[0]))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "per_feature": per_feature,
        "top_gaps": dict(descending[: min(8, len(descending))]),
        "scale": scale.tolist(),
        "scale_source": scale_source,
        "n_real": int(len(real)),
        "n_comparison": int(len(comparison)),
    }

def _orientation_free_c2st_result(report: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_ready(report)
    auc = float(result["auc"])
    result["auc_orientation_free"] = float(0.5 + abs(auc - 0.5))
    result["auc_distance_from_chance"] = float(abs(auc - 0.5))
    result["repeat_auc_orientation_free"] = [
        float(0.5 + abs(float(value) - 0.5)) for value in result["repeat_auc"]
    ]
    return result

def paired_grouped_c2st(
    real_features: np.ndarray,
    comparison_features: np.ndarray,
    source_groups: Sequence[Any] | np.ndarray,
    *,
    seed: int = 7,
    folds: int = C2ST_FOLDS,
    repeats: int = C2ST_REPEATS,
    bootstrap: int = C2ST_BOOTSTRAP,
    public_repo: str | Path = PUBLIC_REPO,
) -> dict[str, Any]:
    """Run the public paired source-grouped descriptor C2ST."""

    real = _feature_matrix(real_features, name="real_features")
    comparison = _feature_matrix(
        comparison_features, name="comparison_features", width=real.shape[1]
    )
    if len(real) != len(comparison):
        raise TournamentJudgeError("paired C2ST requires equal row counts")
    groups = _as_strings(source_groups, len(real), "source_groups")
    judges = load_public_judges(public_repo)
    report = judges.c2st_report(
        real,
        comparison,
        folds=int(folds),
        repeats=int(repeats),
        seed=int(seed),
        groups=groups,
        bootstrap=int(bootstrap),
        confidence=0.95,
        permutations=0,
    )
    # Class labels are arbitrary in a two-sample test. Preserve the exact
    # public AUC for historical comparability and add label-free separation.
    return _orientation_free_c2st_result(report)

def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
