#!/usr/bin/env python3
"""Combine independent descriptor cells into an exact warm/cold audit bundle.

First run ``tools/build_descriptor_bundle.py`` at least twice over the same
source rows with distinct full-pipeline ``--event-seed-domain`` values.  This
tool verifies that their human rows and causal context agree, freezes
installation keys into reference/held populations without looking at outcomes,
selects the 32-row held-session panels, and writes every custody array required
by the exact ``evaluation cold`` and ``evaluation warm`` commands.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bundle import (  # noqa: E402
    DescriptorBundle,
    load_descriptor_bundle,
    write_descriptor_bundle,
)


AUDIT_ROWS = 32
WARM_VALIDATION_ROWS = 32
WARM_REFERENCE_ROWS = 48
DEFAULT_ROLE_SEED = "abcurves.public_audit_roles"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(*parts: object) -> bytes:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def _identity(bundle: DescriptorBundle, index: int) -> tuple[str, str, str]:
    return (
        str(np.asarray(bundle.installation_key).astype(str)[index]),
        str(np.asarray(bundle.session_id).astype(str)[index]),
        str(np.asarray(bundle.source_id).astype(str)[index]),
    )


def _index_by_identity(
    bundle: DescriptorBundle,
    *,
    origin: str,
) -> dict[tuple[str, str, str], int]:
    origins = np.asarray(bundle.origin).astype(str)
    indices = np.flatnonzero(origins == origin)
    output: dict[tuple[str, str, str], int] = {}
    for index in indices:
        identity = _identity(bundle, int(index))
        if identity in output:
            raise ValueError(
                f"cell has duplicate {origin} key/session/source identity {identity!r}"
            )
        output[identity] = int(index)
    return output


def _require_enrichment_inputs(bundle: DescriptorBundle, label: str) -> None:
    missing = [
        name
        for name in ("generator_cell", "target_role", "causal_context", "block_order")
        if getattr(bundle, name) is None
    ]
    if missing:
        raise ValueError(
            f"{label} lacks {', '.join(missing)}; rebuild it with "
            "tools/build_descriptor_bundle.py"
        )


def _assert_human_rows_match(
    authority: DescriptorBundle,
    candidate: DescriptorBundle,
    authority_humans: dict[tuple[str, str, str], int],
    candidate_humans: dict[tuple[str, str, str], int],
    *,
    label: str,
) -> None:
    if tuple(authority.feature_names) != tuple(candidate.feature_names):
        raise ValueError(f"{label} feature names differ from the first cell")
    if dict(authority.panel_slices) != dict(candidate.panel_slices):
        raise ValueError(f"{label} panel slices differ from the first cell")
    if set(authority_humans) != set(candidate_humans):
        raise ValueError(f"{label} does not contain the same human source roster")
    for identity, left in authority_humans.items():
        right = candidate_humans[identity]
        comparisons = {
            "features": (
                np.asarray(authority.features)[left],
                np.asarray(candidate.features)[right],
            ),
            "task": (np.asarray(authority.task)[left], np.asarray(candidate.task)[right]),
            "target_role": (
                np.asarray(authority.target_role)[left],
                np.asarray(candidate.target_role)[right],
            ),
            "causal_context": (
                np.asarray(authority.causal_context)[left],
                np.asarray(candidate.causal_context)[right],
            ),
            "block_order": (
                np.asarray(authority.block_order)[left],
                np.asarray(candidate.block_order)[right],
            ),
        }
        for field, (left_value, right_value) in comparisons.items():
            if not np.array_equal(left_value, right_value):
                raise ValueError(
                    f"{label} human {field} differs for source {identity[2]!r}"
                )


def _assert_generated_rows_are_bound(
    bundle: DescriptorBundle,
    humans: dict[tuple[str, str, str], int],
    generated: dict[tuple[str, str, str], int],
    *,
    label: str,
) -> None:
    for identity, human_index in humans.items():
        generated_index = generated[identity]
        comparisons = {
            "task": (
                np.asarray(bundle.task)[human_index],
                np.asarray(bundle.task)[generated_index],
            ),
            "target_role": (
                np.asarray(bundle.target_role)[human_index],
                np.asarray(bundle.target_role)[generated_index],
            ),
            "causal_context": (
                np.asarray(bundle.causal_context)[human_index],
                np.asarray(bundle.causal_context)[generated_index],
            ),
            "block_order": (
                np.asarray(bundle.block_order)[human_index],
                np.asarray(bundle.block_order)[generated_index],
            ),
        }
        for field, (human_value, generated_value) in comparisons.items():
            if not np.array_equal(human_value, generated_value):
                raise ValueError(
                    f"{label} generated {field} is not bound to its human source "
                    f"{identity[2]!r}"
                )


def _population_split(
    keys: Sequence[str], *, held_fraction: float, role_seed: str
) -> tuple[set[str], set[str]]:
    unique = sorted(set(str(key) for key in keys))
    if len(unique) < 4:
        raise ValueError(
            "an exact audit needs at least four installation keys so two can be "
            "reference and two wholly held"
        )
    if not 0.0 < held_fraction < 1.0:
        raise ValueError("held_fraction must lie strictly between zero and one")
    ordered = sorted(unique, key=lambda key: _stable_key(role_seed, key))
    held_count = max(2, min(len(ordered) - 2, round(held_fraction * len(ordered))))
    held = set(ordered[:held_count])
    return set(ordered[held_count:]), held


def build_audit_bundle(
    cells: Sequence[tuple[str, Path]],
    *,
    held_fraction: float = 0.5,
    role_seed: str = DEFAULT_ROLE_SEED,
) -> tuple[DescriptorBundle, dict[str, object]]:
    if len(cells) < 2:
        raise ValueError("exact warm/cold cell cross-fitting needs at least two cells")
    labels = [str(label).strip() for label, _ in cells]
    if any(not label or label in {"human", "none"} for label in labels):
        raise ValueError("cell labels must be non-empty and cannot be 'human' or 'none'")
    if len(set(labels)) != len(labels):
        raise ValueError("cell labels must be unique")
    if not role_seed:
        raise ValueError("role_seed must not be empty")

    loaded: list[tuple[str, Path, DescriptorBundle]] = []
    for label, source in cells:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"descriptor cell is missing: {path}")
        bundle = load_descriptor_bundle(path)
        _require_enrichment_inputs(bundle, label)
        loaded.append((label, path, bundle))

    authority = loaded[0][2]
    authority_humans = _index_by_identity(authority, origin="human")
    if not authority_humans:
        raise ValueError("descriptor cells contain no human rows")
    generated_maps: dict[str, dict[tuple[str, str, str], int]] = {}
    for label, _, bundle in loaded:
        humans = _index_by_identity(bundle, origin="human")
        _assert_human_rows_match(
            authority,
            bundle,
            authority_humans,
            humans,
            label=label,
        )
        generated = _index_by_identity(bundle, origin="generated")
        if set(generated) != set(authority_humans):
            raise ValueError(
                f"{label} must contain exactly one generated sibling for every human row"
            )
        _assert_generated_rows_are_bound(
            bundle,
            humans,
            generated,
            label=label,
        )
        generated_maps[label] = generated

    human_indices = np.asarray(
        sorted(authority_humans.values()),
        dtype=np.int64,
    )
    keys = np.asarray(authority.installation_key).astype(str)
    sessions = np.asarray(authority.session_id).astype(str)
    sources = np.asarray(authority.source_id).astype(str)
    reference_keys, held_keys = _population_split(
        keys[human_indices],
        held_fraction=held_fraction,
        role_seed=role_seed,
    )

    session_to_key: dict[str, str] = {}
    held_by_session: dict[str, list[int]] = {}
    for index in human_indices:
        key = str(keys[index])
        if key not in held_keys:
            continue
        session = str(sessions[index])
        previous = session_to_key.setdefault(session, key)
        if previous != key:
            raise ValueError(
                f"session label {session!r} is shared by different installation keys"
            )
        held_by_session.setdefault(session, []).append(int(index))
    if len(held_by_session) < 2:
        raise ValueError("an exact warm audit needs at least two held sessions")
    minimum_rows = AUDIT_ROWS + WARM_VALIDATION_ROWS + WARM_REFERENCE_ROWS
    panel_indices: list[int] = []
    audit_order_by_identity: dict[tuple[str, str, str], int] = {}
    for session, indices in sorted(held_by_session.items()):
        if len(indices) < minimum_rows:
            raise ValueError(
                f"held session {session!r} has {len(indices)} human rows; "
                f"warm evaluation needs at least {minimum_rows}"
            )
        ordered = sorted(
            indices,
            key=lambda index: _stable_key(
                role_seed,
                "audit-panel",
                session,
                sources[index],
            ),
        )[:AUDIT_ROWS]
        panel_indices.extend(ordered)
        for order, index in enumerate(ordered):
            audit_order_by_identity[_identity(authority, index)] = order
    panel_identities = [_identity(authority, index) for index in panel_indices]

    feature_blocks = [np.asarray(authority.features)[human_indices]]
    origin_blocks = [np.full(len(human_indices), "human")]
    key_blocks = [keys[human_indices]]
    session_blocks = [sessions[human_indices]]
    source_blocks = [sources[human_indices]]
    order_blocks = [np.asarray(authority.order, dtype=np.int64)[human_indices]]
    task_blocks = [np.asarray(authority.task).astype(str)[human_indices]]
    population_blocks = [
        np.asarray(
            ["held" if key in held_keys else "reference" for key in keys[human_indices]]
        )
    ]
    cell_blocks = [np.full(len(human_indices), "human")]
    role_blocks = [np.asarray(authority.target_role).astype(str)[human_indices]]
    context_blocks = [np.asarray(authority.causal_context)[human_indices]]
    block_order_blocks = [np.asarray(authority.block_order, dtype=np.int64)[human_indices]]
    human_panel = np.asarray(
        [_identity(authority, int(index)) in audit_order_by_identity for index in human_indices]
    )
    audit_panel_blocks = [human_panel]
    audit_order_blocks = [
        np.asarray(
            [
                audit_order_by_identity.get(_identity(authority, int(index)), -1)
                for index in human_indices
            ],
            dtype=np.int64,
        )
    ]

    for label, _, bundle in loaded:
        indices = np.asarray(
            [generated_maps[label][identity] for identity in panel_identities],
            dtype=np.int64,
        )
        feature_blocks.append(np.asarray(bundle.features)[indices])
        origin_blocks.append(np.full(len(indices), "generated"))
        key_blocks.append(np.asarray(bundle.installation_key).astype(str)[indices])
        session_blocks.append(np.asarray(bundle.session_id).astype(str)[indices])
        source_blocks.append(np.asarray(bundle.source_id).astype(str)[indices])
        order_blocks.append(np.asarray(bundle.order, dtype=np.int64)[indices])
        task_blocks.append(np.asarray(bundle.task).astype(str)[indices])
        population_blocks.append(np.full(len(indices), "held"))
        cell_blocks.append(np.full(len(indices), label))
        role_blocks.append(np.asarray(bundle.target_role).astype(str)[indices])
        context_blocks.append(np.asarray(bundle.causal_context)[indices])
        block_order_blocks.append(np.asarray(bundle.block_order, dtype=np.int64)[indices])
        audit_panel_blocks.append(np.ones(len(indices), dtype=bool))
        audit_order_blocks.append(
            np.asarray(
                [audit_order_by_identity[identity] for identity in panel_identities],
                dtype=np.int64,
            )
        )

    output = DescriptorBundle(
        features=np.concatenate(feature_blocks, axis=0),
        origin=np.concatenate(origin_blocks),
        installation_key=np.concatenate(key_blocks),
        session_id=np.concatenate(session_blocks),
        source_id=np.concatenate(source_blocks),
        order=np.concatenate(order_blocks),
        task=np.concatenate(task_blocks),
        feature_names=tuple(authority.feature_names),
        panel_slices=dict(authority.panel_slices),
        population_role=np.concatenate(population_blocks),
        generator_cell=np.concatenate(cell_blocks),
        target_role=np.concatenate(role_blocks),
        causal_context=np.concatenate(context_blocks, axis=0),
        block_order=np.concatenate(block_order_blocks),
        audit_panel=np.concatenate(audit_panel_blocks),
        audit_order=np.concatenate(audit_order_blocks),
    )
    metadata: dict[str, object] = {
        "schema": "abcurves.public_audit_build.v1",
        "role_assignment": {
            "unit": "installation_key",
            "seed": role_seed,
            "held_fraction": float(held_fraction),
            "reference_keys": len(reference_keys),
            "held_keys": len(held_keys),
            "outcome_blind": True,
        },
        "audit_panel": {
            "rows_per_held_session": AUDIT_ROWS,
            "held_sessions": len(held_by_session),
            "selection": "sha256(role_seed, audit-panel, session, source_id)",
        },
        "cells": [
            {
                "label": label,
                "file": path.name,
                "sha256": _sha256(path),
            }
            for label, path, _ in loaded
        ],
        "human_rows": int(len(human_indices)),
        "generated_rows": int(len(panel_indices) * len(loaded)),
    }
    return output, metadata


def _cell_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--cell must use LABEL=DESCRIPTORS.npz")
    return label.strip(), Path(path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        type=_cell_argument,
        action="append",
        required=True,
        help="independent descriptor cell as LABEL=DESCRIPTORS.npz; repeat twice or more",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-fraction", type=float, default=0.5)
    parser.add_argument("--role-seed", default=DEFAULT_ROLE_SEED)
    args = parser.parse_args(argv)
    bundle, metadata = build_audit_bundle(
        args.cell,
        held_fraction=float(args.held_fraction),
        role_seed=str(args.role_seed),
    )
    destination = write_descriptor_bundle(args.output, bundle, metadata=metadata)
    print(
        f"wrote {destination} ({bundle.rows} rows, "
        f"{len(args.cell)} generator cells, sha256={_sha256(destination)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
