"""Portable descriptor-bundle schema used by the evaluation CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "abcurves.descriptor_bundle.v1"
IDENTITY_NOTE = (
    "installation_key is a persistent collection key used for leakage control; "
    "it is not a verified biological identity"
)


def _vector(value: Any, rows: int, name: str, *, dtype: Any | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=dtype).reshape(-1)
    if len(out) != rows:
        raise ValueError(f"{name} must have {rows} rows, got {len(out)}")
    return out


@dataclass(frozen=True)
class DescriptorBundle:
    """Descriptors plus the provenance fields required for honest splitting.

    ``origin`` must be ``"human"`` or ``"generated"``. Generated rows carry
    the installation/session key of the human context they continue. That
    binding lets a cold audit remove *all* rows associated with the queried
    key while fitting its direction.
    """

    features: np.ndarray
    origin: np.ndarray
    installation_key: np.ndarray
    session_id: np.ndarray
    source_id: np.ndarray
    order: np.ndarray
    task: np.ndarray
    feature_names: tuple[str, ...]
    panel_slices: Mapping[str, tuple[int, int]]
    # Optional fields used by the frozen release detector audit.  ``reference``
    # rows form the training-human population envelope; ``held`` rows are the
    # wholly unseen development identities that may be queried.  Generated rows
    # in an exact audit are held rows and carry a predeclared generator cell.
    population_role: np.ndarray | None = None
    generator_cell: np.ndarray | None = None
    target_role: np.ndarray | None = None
    causal_context: np.ndarray | None = None
    block_order: np.ndarray | None = None
    audit_panel: np.ndarray | None = None
    audit_order: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("features must have shape [rows, dimensions]")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        rows, width = values.shape
        origins = _vector(self.origin, rows, "origin").astype(str)
        invalid = sorted(set(origins) - {"human", "generated"})
        if invalid:
            raise ValueError(f"origin contains unsupported labels: {invalid}")
        for name, value in (
            ("installation_key", self.installation_key),
            ("session_id", self.session_id),
            ("source_id", self.source_id),
            ("order", self.order),
            ("task", self.task),
        ):
            _vector(value, rows, name)
        if self.population_role is not None:
            roles = _vector(self.population_role, rows, "population_role").astype(str)
            invalid_roles = sorted(set(roles) - {"reference", "held"})
            if invalid_roles:
                raise ValueError(
                    f"population_role contains unsupported labels: {invalid_roles}"
                )
        if self.generator_cell is not None:
            _vector(self.generator_cell, rows, "generator_cell")
        if self.target_role is not None:
            _vector(self.target_role, rows, "target_role")
        if self.block_order is not None:
            _vector(self.block_order, rows, "block_order")
        if self.audit_panel is not None:
            _vector(self.audit_panel, rows, "audit_panel")
        if self.audit_order is not None:
            _vector(self.audit_order, rows, "audit_order")
        if self.causal_context is not None:
            context = np.asarray(self.causal_context, dtype=np.float64)
            if context.ndim != 2 or context.shape[0] != rows or context.shape[1] < 1:
                raise ValueError("causal_context must have shape [rows, context_dimensions]")
            if not np.all(np.isfinite(context)):
                raise ValueError("causal_context must be finite")
        if len(self.feature_names) != width:
            raise ValueError("feature_names must match the descriptor width")
        if not self.panel_slices:
            raise ValueError("at least one panel slice is required")
        for name, bounds in self.panel_slices.items():
            if len(bounds) != 2:
                raise ValueError(f"panel {name!r} must have (start, stop) bounds")
            start, stop = (int(bounds[0]), int(bounds[1]))
            if not 0 <= start < stop <= width:
                raise ValueError(f"panel {name!r} has invalid bounds {bounds}")

    @property
    def rows(self) -> int:
        return int(np.asarray(self.features).shape[0])

    def mask(self, *, origin: str | None = None, key: str | None = None) -> np.ndarray:
        keep = np.ones(self.rows, dtype=bool)
        if origin is not None:
            keep &= np.asarray(self.origin).astype(str) == str(origin)
        if key is not None:
            keep &= np.asarray(self.installation_key).astype(str) == str(key)
        return keep

    def panel(self, name: str) -> np.ndarray:
        if name not in self.panel_slices:
            raise KeyError(f"unknown panel {name!r}; choose from {sorted(self.panel_slices)}")
        start, stop = self.panel_slices[name]
        return np.asarray(self.features, dtype=np.float64)[:, int(start) : int(stop)]


def _parse_panel_slices(raw: np.ndarray | None, width: int) -> dict[str, tuple[int, int]]:
    if raw is None:
        return {"full": (0, width)}
    values = np.asarray(raw).astype(str).reshape(-1)
    panels: dict[str, tuple[int, int]] = {}
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(
                "panel_slices entries must use 'name:start:stop', "
                f"got {value!r}"
            )
        name, start, stop = parts
        if not name or name in panels:
            raise ValueError(f"invalid or duplicate panel name in {value!r}")
        panels[name] = (int(start), int(stop))
    return panels


def load_descriptor_bundle(path: str | Path) -> DescriptorBundle:
    """Load a non-pickled ``.npz`` descriptor bundle.

    Required arrays are ``features``, ``origin``, ``installation_key``,
    ``session_id`` and ``source_id``. Optional ``order`` and ``task`` default
    to row order and ``"unknown"``. Optional ``panel_slices`` entries use
    ``name:start:stop``. Exact release-audit bundles additionally carry
    ``population_role``, ``generator_cell``, ``target_role``,
    ``causal_context``, ``block_order``, ``audit_panel`` and ``audit_order``.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        missing = [
            name
            for name in (
                "features",
                "origin",
                "installation_key",
                "session_id",
                "source_id",
            )
            if name not in archive
        ]
        if missing:
            raise ValueError(f"descriptor bundle is missing arrays: {missing}")
        features = np.asarray(archive["features"], dtype=np.float64)
        rows, width = features.shape if features.ndim == 2 else (len(features), 0)
        schema = str(np.asarray(archive["schema"]).reshape(-1)[0]) if "schema" in archive else SCHEMA
        if schema != SCHEMA:
            raise ValueError(f"unsupported descriptor-bundle schema {schema!r}")
        feature_names = (
            tuple(np.asarray(archive["feature_names"]).astype(str).reshape(-1))
            if "feature_names" in archive
            else tuple(f"feature_{index}" for index in range(width))
        )
        return DescriptorBundle(
            features=features,
            origin=np.asarray(archive["origin"]).astype(str),
            installation_key=np.asarray(archive["installation_key"]).astype(str),
            session_id=np.asarray(archive["session_id"]).astype(str),
            source_id=np.asarray(archive["source_id"]).astype(str),
            order=(
                np.asarray(archive["order"], dtype=np.int64)
                if "order" in archive
                else np.arange(rows, dtype=np.int64)
            ),
            task=(
                np.asarray(archive["task"]).astype(str)
                if "task" in archive
                else np.full(rows, "unknown", dtype="U7")
            ),
            feature_names=feature_names,
            panel_slices=_parse_panel_slices(
                np.asarray(archive["panel_slices"]) if "panel_slices" in archive else None,
                width,
            ),
            population_role=(
                np.asarray(archive["population_role"]).astype(str)
                if "population_role" in archive
                else None
            ),
            generator_cell=(
                np.asarray(archive["generator_cell"]).astype(str)
                if "generator_cell" in archive
                else None
            ),
            target_role=(
                np.asarray(archive["target_role"]).astype(str)
                if "target_role" in archive
                else None
            ),
            causal_context=(
                np.asarray(archive["causal_context"], dtype=np.float64)
                if "causal_context" in archive
                else None
            ),
            block_order=(
                np.asarray(archive["block_order"], dtype=np.int64)
                if "block_order" in archive
                else None
            ),
            audit_panel=(
                np.asarray(archive["audit_panel"], dtype=bool)
                if "audit_panel" in archive
                else None
            ),
            audit_order=(
                np.asarray(archive["audit_order"], dtype=np.int64)
                if "audit_order" in archive
                else None
            ),
        )


def _text(values: np.ndarray) -> np.ndarray:
    strings = np.asarray(values).astype(str).reshape(-1)
    width = max((len(value) for value in strings), default=1)
    return strings.astype(f"<U{width}")


def write_descriptor_bundle(
    path: str | Path,
    bundle: DescriptorBundle,
    *,
    overwrite: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write one validated, pickle-free descriptor bundle.

    Extra ``metadata`` values must be JSON-compatible. They are stored as one
    canonical JSON string so the numerical schema stays small and explicit.
    """

    # Re-run validation even when a caller constructed a mutable array after
    # the frozen dataclass was initialized.
    DescriptorBundle(
        features=np.asarray(bundle.features),
        origin=np.asarray(bundle.origin),
        installation_key=np.asarray(bundle.installation_key),
        session_id=np.asarray(bundle.session_id),
        source_id=np.asarray(bundle.source_id),
        order=np.asarray(bundle.order),
        task=np.asarray(bundle.task),
        feature_names=tuple(bundle.feature_names),
        panel_slices=dict(bundle.panel_slices),
        population_role=(
            None if bundle.population_role is None else np.asarray(bundle.population_role)
        ),
        generator_cell=(
            None if bundle.generator_cell is None else np.asarray(bundle.generator_cell)
        ),
        target_role=None if bundle.target_role is None else np.asarray(bundle.target_role),
        causal_context=(
            None if bundle.causal_context is None else np.asarray(bundle.causal_context)
        ),
        block_order=None if bundle.block_order is None else np.asarray(bundle.block_order),
        audit_panel=None if bundle.audit_panel is None else np.asarray(bundle.audit_panel),
        audit_order=None if bundle.audit_order is None else np.asarray(bundle.audit_order),
    )
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite descriptor bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel_records = [
        f"{name}:{int(bounds[0])}:{int(bounds[1])}"
        for name, bounds in bundle.panel_slices.items()
    ]
    payload: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "features": np.asarray(bundle.features, dtype=np.float32),
        "origin": _text(np.asarray(bundle.origin)),
        "installation_key": _text(np.asarray(bundle.installation_key)),
        "session_id": _text(np.asarray(bundle.session_id)),
        "source_id": _text(np.asarray(bundle.source_id)),
        "order": np.asarray(bundle.order, dtype=np.int64),
        "task": _text(np.asarray(bundle.task)),
        "feature_names": _text(np.asarray(bundle.feature_names)),
        "panel_slices": _text(np.asarray(panel_records)),
    }
    if bundle.population_role is not None:
        payload["population_role"] = _text(np.asarray(bundle.population_role))
    if bundle.generator_cell is not None:
        payload["generator_cell"] = _text(np.asarray(bundle.generator_cell))
    if bundle.target_role is not None:
        payload["target_role"] = _text(np.asarray(bundle.target_role))
    if bundle.causal_context is not None:
        payload["causal_context"] = np.asarray(bundle.causal_context, dtype=np.float32)
    if bundle.block_order is not None:
        payload["block_order"] = np.asarray(bundle.block_order, dtype=np.int64)
    if bundle.audit_panel is not None:
        payload["audit_panel"] = np.asarray(bundle.audit_panel, dtype=bool)
    if bundle.audit_order is not None:
        payload["audit_order"] = np.asarray(bundle.audit_order, dtype=np.int64)
    if metadata is not None:
        import json

        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    np.savez_compressed(destination, **payload)
    return destination


__all__ = [
    "SCHEMA",
    "IDENTITY_NOTE",
    "DescriptorBundle",
    "load_descriptor_bundle",
    "write_descriptor_bundle",
]
