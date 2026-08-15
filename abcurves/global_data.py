"""Whole-session dataset preparation for the global texture renderer.

The global renderer learns the complete 1 kHz mouse stream: movement before A,
the aimed movement itself, gaps between events, and post-C motion.  Event
outcomes and target geometry are therefore deliberately absent from this data
path.  The only sample boundary is a blind, non-overlapping time window::

    [256 observed ticks | 800 future ticks]

An incomplete final window is dropped rather than padded.  Users, not windows
or sessions, are assigned to validation so one person's texture can never
appear on both sides of the split.

Two input forms are supported:

* one or more validated ``abcurves.research_export.v1`` directories; or
* a portable ``abcurves.full_sessions.v1`` ``sessions.json`` whose entries
  bind relative, pickle-free ``.npy`` arrays by SHA-256.

The prepared result is intentionally a pair of directories instead of one
large NPZ.  NumPy can memory-map ``prefix_raw_dxdy.npy`` and
``future_raw_dxdy.npy`` while training on the full corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .capture_preprocess import CausalOnsetConfig, ShotFilterPolicy
from .preprocessing import (
    DatasetPreparationError,
    MaterializationConfig,
    NativeCaptureExporterUnavailable,
    PlannerPreparationConfig,
    PlannerShapeConfig,
    SeamPreparationConfig,
    TinyTargetConfig,
)


FULL_SESSION_SCHEMA = "abcurves.full_sessions.v1"
GLOBAL_RENDERER_DATASET_SCHEMA = "abcurves.global_renderer_windows.v1"
GLOBAL_RENDERER_SOURCE_INDEX_SCHEMA = "abcurves.global_renderer_sources.v1"
GLOBAL_RENDERER_BUILD_SCHEMA = "abcurves.global_renderer_build.v1"
PREPARATION_CONFIG_SCHEMA = "abcurves.dataset_preparation.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_sha256(path: Path, expected: str) -> bool:
    """Return false, rather than leaking an OS race, when a bound file moves."""

    try:
        return path.is_file() and _sha256(path) == expected
    except OSError:
        return False


def _stable_key(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()


def _session_order(session: "FullSession") -> tuple[str, str]:
    """Order rows exactly as the frozen P0 preparation did."""

    return session.session_id, session.user_id


def _split_key(seed: str, user_id: str) -> str:
    """Reproduce the frozen whole-user split used by the promoted corpus."""

    return hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).hexdigest()


def _safe_relative_reference(value: str, *, label: str) -> str:
    """Return a normalized relative POSIX path with no traversal."""

    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetPreparationError(f"{label} must be a traversal-free relative path")
    return path.as_posix()


def _validate_counts(array: np.ndarray, *, label: str) -> None:
    if array.ndim != 2 or array.shape[1] != 2:
        raise DatasetPreparationError(f"{label} must have shape (ticks, 2)")
    if array.dtype.kind not in "iuf":
        raise DatasetPreparationError(f"{label} must contain numeric hardware counts")
    # Work in chunks so validating a multi-hour session does not allocate a
    # second full-session boolean array.
    for start in range(0, len(array), 1_000_000):
        chunk = np.asarray(array[start : start + 1_000_000])
        if not np.isfinite(chunk).all():
            raise DatasetPreparationError(f"{label} contains non-finite values")
        if not np.equal(chunk, np.rint(chunk)).all():
            raise DatasetPreparationError(f"{label} must contain integer physical counts")
        if len(chunk) and (
            float(np.min(chunk)) < -32768.0 or float(np.max(chunk)) > 32767.0
        ):
            raise DatasetPreparationError(
                f"{label} exceeds the signed int16 physical-report range"
            )


@dataclass(frozen=True)
class FullSession:
    """One complete dense 1 ms physical-count stream."""

    user_id: str
    session_id: str
    dxdy: np.ndarray
    source_ref: str
    source_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_id or self.user_id.lower() in {"unknown", "nan", "none"}:
            raise DatasetPreparationError("full sessions require a real user_id")
        if not self.session_id or self.session_id.lower() in {"unknown", "nan", "none"}:
            raise DatasetPreparationError("full sessions require a real session_id")
        _safe_relative_reference(self.source_ref, label="source_ref")
        _validate_counts(np.asarray(self.dxdy), label=f"session {self.session_id!r}")
        for name, digest in self.source_hashes.items():
            _safe_relative_reference(str(name), label="source hash name")
            if len(str(digest)) != 64 or any(char not in "0123456789abcdef" for char in str(digest)):
                raise DatasetPreparationError("source hashes must be lowercase SHA-256 values")


@dataclass(frozen=True)
class FullSessionCollection:
    sessions: tuple[FullSession, ...]
    source_kind: str
    source_manifest_ref: str
    source_manifest_sha256: str
    source_hashes_verified: bool = False

    def __post_init__(self) -> None:
        if not self.sessions:
            raise DatasetPreparationError("no full sessions were found")
        identities = [(session.user_id, session.session_id) for session in self.sessions]
        if len(set(identities)) != len(identities):
            raise DatasetPreparationError("duplicate (user_id, session_id) full sessions")
        session_ids = [session.session_id for session in self.sessions]
        if len(set(session_ids)) != len(session_ids):
            raise DatasetPreparationError("session_id must uniquely identify a full session")
        _safe_relative_reference(self.source_manifest_ref, label="source_manifest_ref")
        if len(self.source_manifest_sha256) != 64:
            raise DatasetPreparationError("source manifest must have a SHA-256 digest")


@dataclass(frozen=True)
class GlobalRendererPreparationConfig:
    """Final whole-session windowing and training-budget contract."""

    context_ticks: int = 256
    future_ticks: int = 800
    stride_ticks: int = 1_056
    validation_fraction: float = 0.15
    # Frozen for exact train/validation role reproduction: the seed label is
    # part of the split, not a cosmetic version string.
    split_seed: str = "abcurves.continuous_v1.user_split"
    presentation_budget: int = 118_345
    recurrent_warm_ticks: int = 128
    teacher_base_hysteresis: float = 1.0
    sampler: str = "frozen_epoch_view_sampler_v1"
    deterministic_algorithms: bool = True
    teacher_smoothing_specs: tuple[str, ...] = (
        "triangular_moving_average_path:window=3",
        "triangular_moving_average_path:window=5",
    )
    sample_weighting: str = "natural"
    prefix_loss_weight: float = 0.0
    checkpoint_selection: str = "sampled_full_session_texture"
    lateral_offset_penalty: float = 1.5

    def __post_init__(self) -> None:
        if self.context_ticks < 1 or self.future_ticks < 1:
            raise DatasetPreparationError("renderer context/future lengths must be positive")
        if self.stride_ticks != self.context_ticks + self.future_ticks:
            raise DatasetPreparationError(
                "renderer windows must be back-to-back and non-overlapping"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise DatasetPreparationError("validation_fraction must lie in [0, 1)")
        if not self.split_seed:
            raise DatasetPreparationError("split_seed must not be empty")
        if self.presentation_budget < 1:
            raise DatasetPreparationError("presentation_budget must be positive")
        if not 1 <= self.recurrent_warm_ticks <= self.context_ticks:
            raise DatasetPreparationError(
                "recurrent_warm_ticks must lie within the observed context"
            )
        if self.teacher_base_hysteresis != 1.0:
            raise DatasetPreparationError("the frozen teacher labels use hysteresis 1.0")
        if self.sampler != "frozen_epoch_view_sampler_v1":
            raise DatasetPreparationError("the final renderer uses the frozen epoch-view sampler")
        if self.deterministic_algorithms is not True:
            raise DatasetPreparationError("the final renderer enables deterministic algorithms")
        if self.teacher_smoothing_specs != (
            "triangular_moving_average_path:window=3",
            "triangular_moving_average_path:window=5",
        ):
            raise DatasetPreparationError("the final renderer uses the w3/w5 teacher mixture")
        if self.sample_weighting != "natural":
            raise DatasetPreparationError("the final renderer uses natural window weighting")
        if self.prefix_loss_weight != 0.0:
            raise DatasetPreparationError("the global renderer has no prefix loss")
        if self.checkpoint_selection != "sampled_full_session_texture":
            raise DatasetPreparationError(
                "renderer checkpoints must be selected on sampled full-session texture"
            )
        if self.lateral_offset_penalty != 1.5:
            raise DatasetPreparationError("the deployed renderer contract requires AF1.5")

    @property
    def span_ticks(self) -> int:
        return self.context_ticks + self.future_ticks


@dataclass(frozen=True)
class PreparationConfig:
    schema: str
    planner: PlannerPreparationConfig
    renderer: GlobalRendererPreparationConfig


def _parse_planner_config(raw: Mapping[str, Any]) -> PlannerPreparationConfig:
    """Parse the independent event-aligned Planner section."""

    planner_raw = dict(raw)
    planner_seam = SeamPreparationConfig(**dict(planner_raw.pop("seam", {})))
    materialization_raw = dict(planner_raw.pop("materialization", {}))
    onset = CausalOnsetConfig(**dict(materialization_raw.pop("onset", {})))
    materialization = MaterializationConfig(onset=onset, **materialization_raw)
    quality = ShotFilterPolicy(**dict(planner_raw.pop("quality", {})))
    shape = PlannerShapeConfig(**dict(planner_raw.pop("shape", {})))
    tiny_raw = dict(planner_raw.pop("tiny_target", {}))
    quota_raw = tiny_raw.pop("radius_quotas", None)
    if quota_raw is not None:
        if not isinstance(quota_raw, dict):
            raise DatasetPreparationError("tiny_target.radius_quotas must be an object")
        tiny_raw["radius_quotas"] = tuple(
            (float(radius), int(quota))
            for radius, quota in sorted(quota_raw.items(), key=lambda item: float(item[0]))
        )
    tiny = TinyTargetConfig(**tiny_raw)
    return PlannerPreparationConfig(
        seam=planner_seam,
        materialization=materialization,
        quality=quality,
        shape=shape,
        tiny_target=tiny,
        **planner_raw,
    )


def load_preparation_config(path: str | Path) -> PreparationConfig:
    """Load the final Planner + global-renderer preparation preset."""

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise DatasetPreparationError("configuration root must be an object")
    unknown = sorted(set(record) - {"schema", "planner", "renderer"})
    if unknown:
        raise DatasetPreparationError(f"unknown top-level configuration fields: {', '.join(unknown)}")
    schema = str(record.get("schema", ""))
    if schema != PREPARATION_CONFIG_SCHEMA:
        raise DatasetPreparationError(f"unsupported preparation configuration: {schema!r}")

    renderer_raw = dict(record.get("renderer", {}))
    unknown_renderer = sorted(set(renderer_raw) - {"source", "window", "split", "training", "deployment"})
    if unknown_renderer:
        raise DatasetPreparationError(
            f"unknown renderer configuration fields: {', '.join(unknown_renderer)}"
        )
    if renderer_raw.pop("source", None) != "dense_full_sessions_1ms":
        raise DatasetPreparationError("renderer.source must be 'dense_full_sessions_1ms'")
    window = dict(renderer_raw.pop("window", {}))
    if window.pop("drop_incomplete_tail", None) is not True:
        raise DatasetPreparationError("renderer.window.drop_incomplete_tail must be true")
    split = dict(renderer_raw.pop("split", {}))
    if split.pop("unit", None) != "user":
        raise DatasetPreparationError("renderer.split.unit must be 'user'")
    training = dict(renderer_raw.pop("training", {}))
    teacher_specs = training.pop("teacher_smoothing_specs", None)
    if teacher_specs is not None:
        training["teacher_smoothing_specs"] = tuple(str(value) for value in teacher_specs)
    deployment = dict(renderer_raw.pop("deployment", {}))
    renderer = GlobalRendererPreparationConfig(
        **window,
        validation_fraction=float(split.pop("validation_fraction", 0.15)),
        split_seed=str(split.pop("seed", "abcurves.continuous_v1.user_split")),
        **training,
        **deployment,
    )
    if split:
        raise DatasetPreparationError(f"unknown renderer split fields: {', '.join(sorted(split))}")
    return PreparationConfig(
        schema=schema,
        planner=_parse_planner_config(dict(record.get("planner", {}))),
        renderer=renderer,
    )


def _portable_manifest_path(path: Path) -> Path:
    if path.is_dir():
        return path / "sessions.json"
    return path


def _resolve_bound_array(root: Path, relative: str) -> Path:
    safe = _safe_relative_reference(relative, label="dxdy_npy")
    candidate = (root / Path(*PurePosixPath(safe).parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetPreparationError("dxdy_npy resolves outside the portable dataset") from exc
    return candidate


def load_portable_full_sessions(path: str | Path) -> FullSessionCollection:
    """Load and hash-check ``abcurves.full_sessions.v1``.

    ``sessions.json`` has this intentionally small public shape::

        {
          "schema": "abcurves.full_sessions.v1",
          "dense_period_ns": 1000000,
          "sessions": [{
            "user_id": "...", "session_id": "...", "ticks": 1234,
            "dxdy_npy": "arrays/session.npy", "sha256": "..."
          }]
        }

    Arrays must have shape ``(ticks, 2)`` and contain integer physical counts.
    No event, outcome, target, A, B, or C fields participate in this schema.
    """

    manifest_path = _portable_manifest_path(Path(path))
    if not manifest_path.is_file():
        raise DatasetPreparationError(f"portable full-session manifest does not exist: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        record = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetPreparationError(
            f"cannot read portable full-session manifest: {manifest_path}"
        ) from exc
    schema = record.get("schema") if isinstance(record, dict) else None
    if not isinstance(record, dict) or schema != FULL_SESSION_SCHEMA:
        raise DatasetPreparationError(f"unsupported portable full-session schema: {schema!r}")
    if int(record.get("dense_period_ns", 0)) != 1_000_000:
        raise DatasetPreparationError("portable renderer sessions must use a dense 1 ms grid")
    rows = record.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise DatasetPreparationError("portable full-session manifest has no sessions")

    sessions: list[FullSession] = []
    array_bindings: list[tuple[Path, str, str]] = []
    seen_array_paths: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetPreparationError(f"portable session {index} must be an object")
        required = {"user_id", "session_id", "ticks", "dxdy_npy", "sha256"}
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - required)
        if missing or unknown:
            detail = f"missing {missing}" if missing else f"unknown {unknown}"
            raise DatasetPreparationError(f"portable session {index}: {detail}")
        relative = _safe_relative_reference(str(row["dxdy_npy"]), label="dxdy_npy")
        if PurePosixPath(relative).suffix.lower() != ".npy":
            raise DatasetPreparationError("dxdy_npy must name one pickle-free .npy array")
        array_path = _resolve_bound_array(manifest_path.parent, relative)
        if array_path in seen_array_paths:
            raise DatasetPreparationError(
                f"portable sessions must not reuse one physical array: {relative}"
            )
        seen_array_paths.add(array_path)
        expected = str(row["sha256"])
        if len(expected) != 64 or not _matches_sha256(array_path, expected):
            raise DatasetPreparationError(f"portable full-session hash mismatch: {relative}")
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DatasetPreparationError(f"cannot load portable full session: {relative}") from exc
        if not isinstance(array, np.ndarray):
            raise DatasetPreparationError("dxdy_npy must be a single .npy array, not an archive")
        if len(array) != int(row["ticks"]):
            raise DatasetPreparationError(f"portable tick count mismatch: {relative}")
        session = FullSession(
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            dxdy=array,
            source_ref=relative,
            source_hashes={relative: expected},
        )
        if not _matches_sha256(array_path, expected):
            raise DatasetPreparationError(
                f"portable full session changed while it was loaded: {relative}"
            )
        sessions.append(session)
        array_bindings.append((array_path, expected, relative))

    # The arrays are memory maps, so this is an instant-of-load custody check;
    # materialization performs the same check again around every copy.  The
    # manifest is also rechecked so the digest below always names the bytes
    # whose records were actually parsed.
    for array_path, expected, relative in array_bindings:
        if not _matches_sha256(array_path, expected):
            raise DatasetPreparationError(
                f"portable full session changed while the manifest was loaded: {relative}"
            )
    if not _matches_sha256(manifest_path, manifest_hash):
        raise DatasetPreparationError("portable full-session manifest changed while it was loaded")
    return FullSessionCollection(
        sessions=tuple(sorted(sessions, key=_session_order)),
        source_kind="portable_full_sessions",
        source_manifest_ref=manifest_path.name,
        source_manifest_sha256=manifest_hash,
        source_hashes_verified=True,
    )


def write_portable_full_sessions(
    path: str | Path,
    sessions: Iterable[FullSession],
) -> Path:
    """Write a self-contained public full-session manifest and bound arrays."""

    rows = tuple(sorted(sessions, key=_session_order))
    if not rows:
        raise DatasetPreparationError("cannot write an empty full-session dataset")
    session_ids = [session.session_id for session in rows]
    identities = [(session.user_id, session.session_id) for session in rows]
    if len(set(session_ids)) != len(session_ids) or len(set(identities)) != len(identities):
        raise DatasetPreparationError("portable full-session identities must be unique")
    manifest_path = Path(path)
    if manifest_path.suffix.lower() != ".json":
        manifest_path = manifest_path / "sessions.json"
    array_root = manifest_path.parent / "arrays"
    if manifest_path.exists():
        raise DatasetPreparationError(f"refusing to overwrite: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    array_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, session in enumerate(rows):
        token = _stable_key("abcurves.full_sessions.v1", f"{session.user_id}\0{session.session_id}")[:20]
        filename = f"session_{index:04d}_{token}.npy"
        array_path = array_root / filename
        if array_path.exists():
            raise DatasetPreparationError(f"refusing to overwrite: {array_path}")
        np.save(array_path, np.asarray(session.dxdy, dtype=np.float32), allow_pickle=False)
        relative = f"arrays/{filename}"
        records.append(
            {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "ticks": int(len(session.dxdy)),
                "dxdy_npy": relative,
                "sha256": _sha256(array_path),
            }
        )
    manifest = {
        "schema": FULL_SESSION_SCHEMA,
        "dense_period_ns": 1_000_000,
        "sessions": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _research_export_directories(root: Path) -> tuple[Path, ...]:
    if (root / "export_manifest.json").is_file():
        return (root,)
    directories = tuple(sorted(path.parent for path in root.rglob("export_manifest.json")))
    if directories:
        return directories
    if any(root.rglob("*.zip")):
        raise NativeCaptureExporterUnavailable(
            "folder contains raw Capture ZIPs; run the Capture validator/exporter first"
        )
    raise DatasetPreparationError("no validated research exports or sessions.json were found")


def load_research_full_sessions(path: str | Path) -> FullSessionCollection:
    """Load complete dense grids from validated Capture research exports.

    ``trainer_events.csv`` is consulted only to establish the user/session
    identity.  Its outcomes, targets and event intervals never select or crop
    renderer data.
    """

    root = Path(path)
    if not root.is_dir():
        raise DatasetPreparationError(f"research export root is not a directory: {root}")
    try:
        from . import capture_exports
    except ModuleNotFoundError as exc:
        raise DatasetPreparationError(
            "research-export folders require the data extra: pip install -e '.[data]'"
        ) from exc

    export_dirs = _research_export_directories(root)
    sessions: list[FullSession] = []
    aggregate = hashlib.sha256()
    for index, export_dir in enumerate(export_dirs):
        manifest_path = export_dir / "export_manifest.json"
        try:
            manifest_hash = _sha256(manifest_path)
        except OSError as exc:
            raise DatasetPreparationError(f"invalid research export: {export_dir}") from exc
        try:
            manifest = capture_exports.load_export_manifest(export_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetPreparationError(f"invalid research export: {export_dir}") from exc
        if int(manifest.get("dense_grid", {}).get("period_ns", 0)) != 1_000_000:
            raise DatasetPreparationError("renderer research exports must use a dense 1 ms grid")
        declared = {
            str(item.get("relative_path")): str(item.get("sha256", ""))
            for item in manifest.get("artifacts", [])
        }
        source_hashes: dict[str, str] = {}
        for relative in ("mouse_1ms.csv", "trainer_events.csv"):
            artifact = export_dir / relative
            expected = declared.get(relative, "")
            if len(expected) != 64 or not _matches_sha256(artifact, expected):
                raise DatasetPreparationError(f"research export hash mismatch: {artifact}")
            source_hashes[relative] = expected
        if not _matches_sha256(manifest_path, manifest_hash):
            raise DatasetPreparationError(
                f"research export manifest changed while it was loaded: {manifest_path}"
            )
        source_hashes["export_manifest.json"] = manifest_hash
        try:
            _, events, mouse = capture_exports.load_export_tables(export_dir)
        except (OSError, ValueError) as exc:
            raise DatasetPreparationError(f"invalid dense research export: {export_dir}") from exc
        users = sorted({str(value) for value in events["user_id"].dropna().tolist()})
        session_ids = sorted({str(value) for value in events["session_id"].dropna().tolist()})
        if len(users) != 1 or len(session_ids) != 1:
            raise DatasetPreparationError(
                "each research export must contain exactly one user_id and session_id"
            )
        # Retain only the two renderer channels, in the same float32 form the
        # prepared memory maps use.  Keeping pandas' wider table blocks for
        # every long session would otherwise dominate preparation memory.
        raw = mouse[["canonical_dx", "canonical_dy"]].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        for relative, expected in source_hashes.items():
            artifact = export_dir / relative
            if not _matches_sha256(artifact, expected):
                raise DatasetPreparationError(
                    f"research export changed while it was loaded: {artifact}"
                )
        aggregate.update(bytes.fromhex(manifest_hash))
        sessions.append(
            FullSession(
                user_id=users[0],
                session_id=session_ids[0],
                dxdy=raw,
                source_ref=f"export_{index:04d}",
                source_hashes=source_hashes,
            )
        )
    return FullSessionCollection(
        sessions=tuple(sorted(sessions, key=_session_order)),
        source_kind="research_export_directories",
        source_manifest_ref="research_exports/index",
        source_manifest_sha256=aggregate.hexdigest(),
        source_hashes_verified=True,
    )


def load_full_sessions(path: str | Path) -> FullSessionCollection:
    """Load either supported full-session input form, rejecting event NPZs."""

    source = Path(path)
    event_npz = source.suffix.lower() == ".npz" or (
        source.is_dir() and (source / "events.npz").is_file()
    )
    if event_npz:
        raise DatasetPreparationError(
            "renderer preparation requires dense 1 ms full sessions; an A-to-C "
            "events.npz cannot recover pre-A, inter-event, or post-C texture"
        )
    if source.suffix.lower() == ".json" or (
        source.is_dir() and (source / "sessions.json").is_file()
    ):
        return load_portable_full_sessions(source)
    return load_research_full_sessions(source)


def assign_user_splits(
    sessions: Sequence[FullSession],
    *,
    validation_fraction: float,
    split_seed: str,
) -> dict[str, str]:
    """Return ``user_id -> train|val`` with deterministic whole-user isolation."""

    if not 0.0 <= float(validation_fraction) < 1.0:
        raise DatasetPreparationError("validation_fraction must lie in [0, 1)")
    users = sorted(
        {session.user_id for session in sessions},
        key=lambda value: _split_key(split_seed, value),
    )
    validation_count = 0
    if validation_fraction > 0.0 and len(users) >= 2:
        validation_count = max(1, min(len(users) - 1, int(round(len(users) * validation_fraction))))
    validation = set(users[:validation_count])
    return {user: ("val" if user in validation else "train") for user in users}


def _window_count(ticks: int, config: GlobalRendererPreparationConfig) -> int:
    if ticks < config.span_ticks:
        return 0
    return 1 + (ticks - config.span_ticks) // config.stride_ticks


def _window_metadata(
    sessions: Sequence[FullSession],
    split_by_user: Mapping[str, str],
    config: GlobalRendererPreparationConfig,
) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for session in sorted(sessions, key=_session_order):
        count = _window_count(len(session.dxdy), config)
        consumed = 0 if count == 0 else (count - 1) * config.stride_ticks + config.span_ticks
        by_split[split_by_user[session.user_id]].append(
            {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "source_ref": session.source_ref,
                "source_hashes": dict(sorted(session.source_hashes.items())),
                "ticks": int(len(session.dxdy)),
                "windows": int(count),
                "dropped_incomplete_tail_ticks": int(len(session.dxdy) - consumed),
            }
        )
    return by_split


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise DatasetPreparationError(f"refusing to overwrite prepared output: {existing[0]}")


def _portable_memmap_binding(
    session: FullSession,
    *,
    source_hashes_verified: bool,
) -> tuple[Path, str] | None:
    """Return the verified backing file for a portable memmap, when present."""

    filename = getattr(session.dxdy, "filename", None)
    if not source_hashes_verified or filename is None:
        return None
    if len(session.source_hashes) != 1:
        raise DatasetPreparationError(
            "a verified portable session must bind exactly one array hash"
        )
    expected = str(next(iter(session.source_hashes.values()))).lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise DatasetPreparationError("a verified portable session has an invalid hash")
    return Path(str(filename)).resolve(), expected


def save_global_renderer_dataset(
    output: str | Path,
    collection: FullSessionCollection,
    config: GlobalRendererPreparationConfig,
    *,
    validation_fraction: float | None = None,
    split_seed: str | None = None,
) -> dict[str, Any]:
    """Materialize memory-mappable whole-session renderer train/val arrays."""

    fraction = config.validation_fraction if validation_fraction is None else float(validation_fraction)
    seed = config.split_seed if split_seed is None else str(split_seed)
    split_by_user = assign_user_splits(collection.sessions, validation_fraction=fraction, split_seed=seed)
    entries = _window_metadata(collection.sessions, split_by_user, config)
    root = Path(output)
    target_files = [root / "source_index.json", root / "build_report.json"]
    for split in ("train", "val"):
        split_root = root / f"renderer_{split}"
        target_files.extend(
            [
                split_root / "prefix_raw_dxdy.npy",
                split_root / "future_raw_dxdy.npy",
                split_root / "meta.json",
            ]
        )
    _ensure_absent(target_files)
    # Portable arrays stay memory-mapped after loading. Recheck them before any
    # output is created, then again around each materialization pass below so a
    # changed source can never inherit the earlier verified-hash claim.
    for session in collection.sessions:
        binding = _portable_memmap_binding(
            session,
            source_hashes_verified=collection.source_hashes_verified,
        )
        if binding is not None and not _matches_sha256(binding[0], binding[1]):
            raise DatasetPreparationError(
                f"portable source changed after validation: {session.session_id}"
            )
    root.mkdir(parents=True, exist_ok=True)

    roles: dict[str, Any] = {}
    sessions_by_id = {session.session_id: session for session in collection.sessions}
    for split in ("train", "val"):
        split_root = root / f"renderer_{split}"
        split_root.mkdir(parents=True, exist_ok=True)
        rows = entries[split]
        contributing_rows = [row for row in rows if int(row["windows"]) > 0]
        window_total = sum(int(row["windows"]) for row in rows)
        prefix_path = split_root / "prefix_raw_dxdy.npy"
        future_path = split_root / "future_raw_dxdy.npy"
        prefix = np.lib.format.open_memmap(
            prefix_path,
            mode="w+",
            dtype=np.float32,
            shape=(window_total, config.context_ticks, 2),
        )
        future = np.lib.format.open_memmap(
            future_path,
            mode="w+",
            dtype=np.float32,
            shape=(window_total, config.future_ticks, 2),
        )
        user_ids: list[str] = []
        session_ids: list[str] = []
        starts: list[int] = []
        cursor = 0
        for row in rows:
            session = sessions_by_id[str(row["session_id"])]
            binding = _portable_memmap_binding(
                session,
                source_hashes_verified=collection.source_hashes_verified,
            )
            if binding is not None and not _matches_sha256(binding[0], binding[1]):
                raise DatasetPreparationError(
                    f"portable source changed after validation: {session.session_id}"
                )
            raw = session.dxdy
            for window_index in range(int(row["windows"])):
                start = window_index * config.stride_ticks
                seam = start + config.context_ticks
                stop = seam + config.future_ticks
                prefix[cursor] = np.asarray(raw[start:seam], dtype=np.float32)
                future[cursor] = np.asarray(raw[seam:stop], dtype=np.float32)
                user_ids.append(session.user_id)
                session_ids.append(session.session_id)
                starts.append(start)
                cursor += 1
            if binding is not None and not _matches_sha256(binding[0], binding[1]):
                raise DatasetPreparationError(
                    f"portable source changed while it was materialized: {session.session_id}"
                )
        if cursor != window_total:
            raise DatasetPreparationError("internal renderer window-count mismatch")
        prefix.flush()
        future.flush()
        del prefix, future
        meta = {
            "schema": GLOBAL_RENDERER_DATASET_SCHEMA,
            "cohort": "renderer_global_full_session_v1",
            "split": split,
            "prefix": config.context_ticks,
            "future": config.future_ticks,
            "stride": config.stride_ticks,
            "windows": window_total,
            "users": len({row["user_id"] for row in contributing_rows}),
            "sessions": len(contributing_rows),
            "source_users": len({row["user_id"] for row in rows}),
            "source_sessions": len(rows),
            "source_ticks": sum(int(row["ticks"]) for row in rows),
            "full_session_id": session_ids,
            "session_id": session_ids,
            "user_id": user_ids,
            "window_start_tick": starts,
            "boundary": "blind fixed time windows; no event, target, outcome, A, B, or C filtering",
        }
        meta_path = split_root / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        roles[split] = {
            "directory": split_root.name,
            "windows": window_total,
            "users": len({row["user_id"] for row in contributing_rows}),
            "sessions": len(contributing_rows),
            "source_users": len({row["user_id"] for row in rows}),
            "source_sessions": len(rows),
            "ticks": sum(int(row["ticks"]) for row in rows),
            "dropped_incomplete_tail_ticks": sum(
                int(row["dropped_incomplete_tail_ticks"]) for row in rows
            ),
            "sha256": {
                "prefix_raw_dxdy.npy": _sha256(prefix_path),
                "future_raw_dxdy.npy": _sha256(future_path),
                "meta.json": _sha256(meta_path),
            },
        }

    train_users = {row["user_id"] for row in entries["train"]}
    val_users = {row["user_id"] for row in entries["val"]}
    if train_users & val_users:
        raise DatasetPreparationError("user leakage across renderer train/val splits")
    source_index = {
        "schema": GLOBAL_RENDERER_SOURCE_INDEX_SCHEMA,
        "source_kind": collection.source_kind,
        "source_manifest_ref": collection.source_manifest_ref,
        "source_manifest_sha256": collection.source_manifest_sha256,
        "source_hashes_verified": bool(collection.source_hashes_verified),
        "split_unit": "user",
        "split_seed": seed,
        "validation_fraction": fraction,
        "train": entries["train"],
        "val": entries["val"],
    }
    source_path = root / "source_index.json"
    source_path.write_text(json.dumps(source_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_report = {
        "schema": GLOBAL_RENDERER_BUILD_SCHEMA,
        "config": {
            **asdict(config),
            "teacher_smoothing_specs": list(config.teacher_smoothing_specs),
            "span_ticks": config.span_ticks,
            "drop_incomplete_tail": True,
        },
        "effective_split": {
            "unit": "user",
            "validation_fraction": fraction,
            "seed": seed,
            "user_leakage": False,
        },
        "source_kind": collection.source_kind,
        "all_source_hashes_verified": bool(collection.source_hashes_verified),
        "roles": roles,
        "source_index_sha256": _sha256(source_path),
    }
    report_path = root / "build_report.json"
    report_path.write_text(json.dumps(build_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return build_report


__all__ = [
    "FULL_SESSION_SCHEMA",
    "GLOBAL_RENDERER_BUILD_SCHEMA",
    "GLOBAL_RENDERER_DATASET_SCHEMA",
    "GLOBAL_RENDERER_SOURCE_INDEX_SCHEMA",
    "PREPARATION_CONFIG_SCHEMA",
    "FullSession",
    "FullSessionCollection",
    "GlobalRendererPreparationConfig",
    "PreparationConfig",
    "assign_user_splits",
    "load_full_sessions",
    "load_portable_full_sessions",
    "load_preparation_config",
    "load_research_full_sessions",
    "save_global_renderer_dataset",
    "write_portable_full_sessions",
]
