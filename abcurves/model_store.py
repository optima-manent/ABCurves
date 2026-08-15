"""Locate and authenticate the model files shipped with ABCurves."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import site
import sys
from typing import Any


class ModelIntegrityError(RuntimeError):
    """Raised when a requested release model is absent or has changed."""


# These are code-level release anchors, deliberately independent of the
# adjacent JSON manifest.  A modified model plus a modified manifest must not
# turn verification into self-attestation.
RELEASE_FILE_ANCHORS: dict[str, tuple[int, str]] = {
    "planner_seed7.pt": (
        1_500_345,
        "d82c93071224f7eb225d1f2bcf46d52669a7270db414431d7622e032439b280d",
    ),
    "planner_seed23.pt": (
        1_500_389,
        "d691ba155c4fa9b403c5a3e2ed9c44123fe00d3d1bee15c55ee9226f4531a23e",
    ),
    "renderer_global_h80.bin": (
        44_484,
        "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b",
    ),
    "renderer_global_h80_float.pt": (
        144_457,
        "696efe3bbcbc7e8991e26058bc9b8195285e5f5cb5e5f8cc5f64fcd30d1ac840",
    ),
}


@dataclass(frozen=True)
class ModelFiles:
    seed: int
    planner: Path
    renderer: Path
    manifest: dict[str, Any]


def default_model_dir() -> Path:
    """Return the bundled ``models`` directory from a clone or wheel install."""

    repository_models = Path(__file__).resolve().parents[1] / "models"
    if (repository_models / "manifest.json").is_file():
        return repository_models
    installed_models = Path(sys.prefix) / "models"
    if (installed_models / "manifest.json").is_file():
        return installed_models
    # ``pip install --user`` places data-files under the user base, not under
    # the user site-packages directory containing this module.
    user_models = Path(site.getuserbase()) / "models"
    if (user_models / "manifest.json").is_file():
        return user_models
    # Preserve the most useful error path for a source checkout.
    return repository_models


def _sha256(path_text: str) -> str:
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
    if manifest.get("schema") != "abcurves.release_models.v2":
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
    anchor = RELEASE_FILE_ANCHORS.get(name)
    if anchor is None:
        raise ModelIntegrityError(f"{name!r} has no immutable release anchor")
    expected_bytes, expected = anchor
    if int(record.get("bytes", -1)) != expected_bytes or str(
        record.get("sha256", "")
    ).lower() != expected:
        raise ModelIntegrityError(f"manifest declaration differs from the release anchor for {name}")
    if stat.st_size != expected_bytes:
        raise ModelIntegrityError(
            f"release model size differs for {path.name}: "
            f"expected {expected_bytes}, observed {stat.st_size}"
        )
    observed = _sha256(str(path.resolve()))
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
    """Resolve one Planner seed and the shared global Renderer image.

    Seed 7 is the default Planner cell and seed 23 is its independent
    replication.  Both use the same selected full-corpus global Renderer,
    whose identity is independent of Planner training seed.
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
        "renderer": "renderer_global_h80.bin",
    }
    if verify:
        paths = {key: _verified_file(root, name, manifest) for key, name in names.items()}
    else:
        paths = {key: root / name for key, name in names.items()}
    return ModelFiles(seed=chosen, manifest=manifest, **paths)


def resolve_renderer_float(
    *,
    model_dir: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Resolve the sanitized float Renderer checkpoint used for research.

    This checkpoint documents the learned graph behind the packed deployment
    image. The default native path does not need it; callers can authenticate it
    here and select it explicitly with ``Pipeline(float_renderer_checkpoint=...)``.
    """

    root = default_model_dir() if model_dir is None else Path(model_dir).expanduser()
    root = root.resolve()
    manifest = _load_manifest(root)
    name = "renderer_global_h80_float.pt"
    return _verified_file(root, name, manifest) if verify else root / name


__all__ = [
    "ModelFiles",
    "ModelIntegrityError",
    "RELEASE_FILE_ANCHORS",
    "default_model_dir",
    "resolve_model_files",
    "resolve_renderer_float",
]
