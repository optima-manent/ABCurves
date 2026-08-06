"""Locate and authenticate the model files shipped with ABCurves."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


class ModelIntegrityError(RuntimeError):
    """Raised when a requested release model is absent or has changed."""


@dataclass(frozen=True)
class ModelFiles:
    seed: int
    planner: Path
    renderer: Path
    renderer_adapter: Path
    manifest: dict[str, Any]


def default_model_dir() -> Path:
    """Return the bundled ``models`` directory from a clone or wheel install."""

    repository_models = Path(__file__).resolve().parents[1] / "models"
    if (repository_models / "manifest.json").is_file():
        return repository_models
    installed_models = Path(sys.prefix) / "models"
    if (installed_models / "manifest.json").is_file():
        return installed_models
    # Preserve the most useful error path for a source checkout.
    return repository_models


@lru_cache(maxsize=32)
def _sha256(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "manifest.json"
    if not path.is_file():
        raise ModelIntegrityError(f"model manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(f"cannot read model manifest: {path}") from exc
    if manifest.get("schema") != "abcurves.release_models.v1":
        raise ModelIntegrityError(f"unsupported model manifest schema in {path}")
    return manifest


def _verified_file(model_dir: Path, name: str, manifest: dict[str, Any]) -> Path:
    record = manifest.get("files", {}).get(name)
    if not isinstance(record, dict):
        raise ModelIntegrityError(f"{name!r} is not declared by the model manifest")
    path = model_dir / name
    if not path.is_file():
        raise ModelIntegrityError(f"release model is missing: {path}")
    stat = path.stat()
    expected_bytes = int(record.get("bytes", -1))
    if stat.st_size != expected_bytes:
        raise ModelIntegrityError(
            f"release model size differs for {path.name}: "
            f"expected {expected_bytes}, observed {stat.st_size}"
        )
    observed = _sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    expected = str(record.get("sha256", "")).lower()
    if observed != expected:
        raise ModelIntegrityError(
            f"release model hash differs for {path.name}: "
            f"expected {expected}, observed {observed}"
        )
    return path


def resolve_model_files(
    seed: int = 7,
    *,
    model_dir: str | Path | None = None,
    verify: bool = True,
) -> ModelFiles:
    """Resolve one matched Planner/Renderer/adapter seed cell.

    Seed 7 is the default release cell.  Seed 23 is shipped as an independent
    replication.  The two cells are complete units: do not mix component
    seeds.
    """

    root = default_model_dir() if model_dir is None else Path(model_dir).expanduser()
    root = root.resolve()
    manifest = _load_manifest(root)
    chosen = int(seed)
    if chosen not in tuple(int(value) for value in manifest.get("seeds", ())):
        raise ModelIntegrityError(
            f"unsupported release seed {chosen}; choose one of {manifest.get('seeds', [])}"
        )
    names = {
        "planner": f"planner_seed{chosen}.pt",
        "renderer": f"renderer_seed{chosen}.pt",
        "renderer_adapter": f"renderer_adapter_seed{chosen}.pt",
    }
    if verify:
        paths = {key: _verified_file(root, name, manifest) for key, name in names.items()}
    else:
        paths = {key: root / name for key, name in names.items()}
    return ModelFiles(seed=chosen, manifest=manifest, **paths)


__all__ = [
    "ModelFiles",
    "ModelIntegrityError",
    "default_model_dir",
    "resolve_model_files",
]
