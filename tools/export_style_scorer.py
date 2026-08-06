"""Export the frozen Renderer Stage-0 transform without private metadata.

The source authority contains fold assignments, workstation paths, and study
bookkeeping that are not part of inference.  This utility keeps only the
all-training deployment transform: the nuisance ridge, residual scales,
safe C/M/H basis, and causal-state contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "abcurves.style_scorer.v1"
SOURCE_SCHEMA = "abcurves.renderer_two_logit_adapter_screen.v1.safe_basis_train_only"
SCORE_NAMES = ("C", "M", "H")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def export(source: Path) -> dict[str, Any]:
    authority = json.loads(source.read_text(encoding="utf-8"))
    _require(authority.get("schema") == SOURCE_SCHEMA, "unexpected source schema")

    nuisance = authority["nuisance"]
    ridge = nuisance["full_ridge"]
    texture_names = list(authority["public_texture19"]["feature_names"])
    context_names = list(nuisance["context_feature_names"])
    safe = authority["safe_allowlist"]
    directions = authority["refit_all_train_directions"]
    state = authority["state_contract"]

    _require(len(texture_names) == 19, "texture contract must have 19 features")
    _require(len(context_names) == 40, "context contract must have 40 features")
    _require(len(ridge["x_mean"]) == 40, "ridge mean width differs")
    _require(len(ridge["x_scale"]) == 40, "ridge scale width differs")
    _require(len(ridge["beta"]) == 41, "ridge beta row count differs")
    _require(all(len(row) == 19 for row in ridge["beta"]), "ridge beta width differs")
    _require(len(nuisance["residual_scale"]) == 19, "residual scale width differs")
    _require(list(state["generic"]) == [0.0, 0.0, 0.0], "generic state differs")
    _require(list(authority["selection"]["selected_basis"]) == list(SCORE_NAMES), "selected basis differs")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "score_names": list(SCORE_NAMES),
        "texture_feature_names": texture_names,
        "context_feature_names": context_names,
        "safe_feature_names": list(safe["ordered_feature_names"]),
        "safe_groups": {name: list(safe["groups"][name]) for name in SCORE_NAMES},
        "nuisance_transform": {
            "ridge_alpha": float(nuisance["ridge_alpha"]),
            "beta": ridge["beta"],
            "x_mean": ridge["x_mean"],
            "x_scale": ridge["x_scale"],
            "residual_scale": nuisance["residual_scale"],
        },
        "directions": {name: directions[name] for name in SCORE_NAMES},
        "state_contract": {
            "support_events": int(state["support_rows"]),
            "shrinkage": float(state["shrinkage"]),
            "clip": list(state["clip"]),
            "fewer_than_support": "exact_zero_vector",
        },
        "provenance": {
            "source_schema": SOURCE_SCHEMA,
            "source_file_sha256": _sha256_file(source),
            "source_contract_sha256": str(authority["contract_sha256"]),
            "fit_population": "all frozen training humans",
            "oof_training_scores_included": False,
            "identity_fields_included": False,
        },
    }
    payload["contract_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="frozen safe_basis.train_only.json")
    parser.add_argument("output", type=Path, help="destination style_scorer.json")
    args = parser.parse_args()
    result = export(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(result))
    print(f"wrote {args.output} ({result['contract_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
