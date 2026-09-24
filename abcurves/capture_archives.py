"""Bounded raw-Capture archive handling; scientific validation belongs to Capture."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def session_manifest(archive):
    """Read transport metadata only; this does not assert session validity."""
    entries = [p for p in archive.infolist() if
               p.filename.rstrip("/").endswith("/manifest.json")
               or p.filename == "manifest.json"]
    candidates = []
    for entry in entries:
        if entry.file_size > 4 * 1024 * 1024:
            continue
        value = json.loads(archive.read(entry))
        if value.get("schema") == "abcurves.capture.session.v2":
            candidates.append((entry.filename, value))
    if len(candidates) != 1:
        raise ValueError("Archive must contain exactly one Capture session manifest")
    return candidates[0]


@contextmanager
def archive_sources(source, *, scratch=None):
    """Yield (path, lineage) pairs from a collection ZIP, ZIP folder or session.

    Nested session ZIPs are streamed to one temporary file at a time; never
    materialize an entire large collection in RAM. Original archives are read
    only. Explicit directory entries remain valid transport metadata.
    """
    source = Path(source).resolve()

    def walk(path):
        if path.is_dir():
            if (path / "COMPLETE").is_file() and (path / "manifest.json").is_file():
                yield path, {"source": path.name, "kind": "extracted_session"}
                return
            sessions = sorted(p.parent for p in path.rglob("COMPLETE")
                              if (p.parent / "manifest.json").is_file())
            if sessions:
                for session in sessions:
                    yield session, {"source": session.relative_to(path).as_posix(),
                                    "kind": "extracted_session"}
            archives = sorted(p for p in path.rglob("*.zip")
                              if not any(p.is_relative_to(s) for s in sessions))
            if not archives and not sessions:
                raise ValueError("Input folder contains no session archives or sealed sessions")
            for item in archives:
                for archive, lineage in walk(item):
                    yield archive, dict(lineage, folder_member=item.relative_to(path).as_posix())
            return
        if path.suffix.lower() != ".zip":
            raise ValueError("Expected a ZIP archive or directory of Capture sessions")
        with zipfile.ZipFile(path) as archive:
            nested = [i for i in archive.infolist() if i.filename.lower().endswith(".zip")]
            if not nested:
                session_manifest(archive)
                yield path, {"source": path.name, "kind": "session_archive",
                             "archive_sha256": sha256(path)}
                return
            outer_hash = sha256(path)
            for entry in sorted(nested, key=lambda i: i.filename):
                with tempfile.TemporaryDirectory(prefix="abcurves-session-", dir=scratch) as tmp:
                    target = Path(tmp) / "session.zip"
                    with archive.open(entry) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    with zipfile.ZipFile(target) as child:
                        session_manifest(child)
                    yield target, {"source": path.name, "member": entry.filename,
                                   "kind": "collection_member", "collection_sha256": outer_hash,
                                   "archive_sha256": sha256(target)}

    yield walk(source)


def extract_session(source, destination):
    """Extract regular ZIP files within one new scratch directory, retaining bytes."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(source) as archive:
        manifest_name, manifest = session_manifest(archive)
        sealed_prefix = PurePosixPath(manifest_name).parent
        seen = set()
        total = 0
        for entry in archive.infolist():
            relative = PurePosixPath(entry.filename)
            if not entry.is_dir() and not relative.is_relative_to(sealed_prefix):
                raise ValueError("Archive contains a file outside its sealed session: " + entry.filename)
            if (relative.is_absolute() or ".." in relative.parts or
                    "\\" in entry.filename or ":" in entry.filename):
                raise ValueError("Unsafe archive member: " + entry.filename)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Archive links are not source evidence")
            folded = entry.filename.rstrip("/").casefold()
            if folded in seen:
                raise ValueError("Duplicate archive member: " + entry.filename)
            seen.add(folded)
            total += entry.file_size
            if entry.file_size > 16 * 1024**3 or total > 64 * 1024**3:
                raise ValueError("Session exceeds Capture's documented expanded-size limits")
            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive escapes the extraction directory")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    return destination.joinpath(*PurePosixPath(manifest_name).parts).parent, manifest


def protocol(manifest):
    """Use recorded protocol identity; empty static journals are valid tracking."""
    identity = manifest.get("protocol_id")
    if identity == "abcurves.tracking.protocol-v1":
        return "tracking"
    if identity == "abcurves.capture-trainer.protocol-v3":
        return "static"
    raise ValueError("Unsupported recorded Capture gameplay protocol")
