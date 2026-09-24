"""Validate raw Capture inputs and export their native evidence for preparation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abcurves.capture_archives import archive_sources, extract_session, protocol, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture-bin", type=Path, required=True,
                        help="Directory containing public abct_session_tool and abct_research_export")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Explicit smoke-test limit; never a complete-corpus claim")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed exports in an interrupted output")
    parser.add_argument("--keep-going", action="store_true", help="Record rejected sessions and process the remaining collection")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    suffix = ".exe" if sys.platform == "win32" else ""
    validator = (args.capture_bin / ("abct_session_tool" + suffix)).resolve()
    exporter = (args.capture_bin / ("abct_research_export" + suffix)).resolve()
    for executable in [validator] + ([] if args.validate_only else [exporter]):
        if not executable.is_file():
            parser.error(f"Missing public Capture executable: {executable}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    records, identities = [], set()
    receipt = {"schema": "abcurves.capture_ingestion.v1", "complete": False,
               "scope": "limited_smoke" if args.limit else "all_input_sessions",
               "tools": {p.name: sha256(p) for p in [validator] + ([] if args.validate_only else [exporter])},
               "sessions": records}
    if args.resume:
        previous = json.loads((output / "ingestion.json").read_text())
        if previous["tools"] != receipt["tools"] or previous["scope"] != receipt["scope"]:
            raise ValueError("Resume requires the original tool versions and input scope")
        records.extend(previous["sessions"])
    previous_by_id = {r["session_id"]: r for r in records}
    def save():
        temporary = output / "ingestion.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output / "ingestion.json")
    save()
    try:
        with archive_sources(args.input, scratch=output) as sources:
            for index, (source, lineage) in enumerate(sources):
                if args.limit and index >= args.limit:
                    break
                with tempfile.TemporaryDirectory(prefix=".extract-", dir=output) as tmp:
                    if source.is_dir():
                        session = source
                        manifest = json.loads((source / "manifest.json").read_text())
                    else:
                        session, manifest = extract_session(source, tmp)
                    identity = manifest["session_id"]
                    if identity in identities:
                        raise ValueError(f"Duplicate session identity: {identity}")
                    identities.add(identity)
                    kind = protocol(manifest)
                    sealed_hashes = {name: sha256(session / name) for name in ("checksums.sha256", "COMPLETE")}
                    prior = previous_by_id.get(identity)
                    if prior and prior.get("accepted") is not False:
                        checked = subprocess.run([str(validator), "validate", str(session)],
                                                 check=True, text=True, capture_output=True)
                        for key, value in lineage.items():
                            if prior.get(key) != value:
                                raise ValueError("Resume source identity changed: " + identity)
                        if prior["manifest_sha256"] != sha256(session / "manifest.json"):
                            raise ValueError("Resume source manifest changed")
                        if prior.get("sealed_hashes") != sealed_hashes and (
                                "sealed_hashes" in prior or lineage["kind"] == "extracted_session"):
                            raise ValueError("Resume requires the original sealed inventory and completion marker")
                        if not args.validate_only:
                            export = output / prior["export"]
                            if sha256(export / "export_manifest.json") != prior["export_manifest_sha256"]:
                                raise ValueError("Resumed export manifest changed")
                            exported = json.loads((export / "export_manifest.json").read_text())
                            for item in exported["source_artifacts"]:
                                artifact = (session / item["relative_path"]).resolve()
                                if not artifact.is_relative_to(session.resolve()) or sha256(artifact) != item["sha256"]:
                                    raise ValueError("Resumed source artifact changed")
                            for item in exported["artifacts"]:
                                artifact = (export / item["relative_path"]).resolve()
                                if not artifact.is_relative_to(export.resolve()) or sha256(artifact) != item["sha256"]:
                                    raise ValueError("Resumed export artifact changed")
                        record = dict(prior, accepted=True, sealed_hashes=sealed_hashes)
                    else:
                        record = {**lineage, "session_id": identity, "protocol": kind,
                                  "manifest_sha256": sha256(session / "manifest.json"),
                                  "sealed_hashes": sealed_hashes,
                                  "validated": False, "accepted": False}
                        try:
                            checked = subprocess.run([str(validator), "validate", str(session)],
                                                     check=True, text=True, capture_output=True)
                            record.update(validation=checked.stdout.strip(), validated=True)
                            if not args.validate_only:
                                export = output / "exports" / kind / session.name
                                export.parent.mkdir(parents=True, exist_ok=True)
                                if args.resume and (export / "export_manifest.json").is_file():
                                    # Recover an export completed before its ingestion
                                    # receipt was saved, only after rebinding source
                                    # evidence and every derived artifact.
                                    exported = json.loads((export / "export_manifest.json").read_text())
                                    if exported["source_session"]["session_id"] != identity:
                                        raise ValueError("Existing export belongs to another session")
                                    for base, items in [(session, exported["source_artifacts"]),
                                                        (export, exported["artifacts"])]:
                                        for item in items:
                                            path = (base / item["relative_path"]).resolve()
                                            if not path.is_relative_to(base.resolve()) or sha256(path) != item["sha256"]:
                                                raise ValueError("Existing export source or artifact changed")
                                else:
                                    subprocess.run([str(exporter), str(session), str(export)],
                                                   check=True, text=True, capture_output=True)
                                record["export"] = export.relative_to(output).as_posix()
                                record["export_manifest_sha256"] = sha256(export / "export_manifest.json")
                            record["accepted"] = True
                        except subprocess.CalledProcessError as error:
                            record["rejection"] = (error.stderr or error.stdout or str(error)).strip()
                            if not args.keep_going:
                                records[:] = [r for r in records if r["session_id"] != identity]
                                records.append(record)
                                save()
                                raise
                    records[:] = [r for r in records if r["session_id"] != identity]
                    records.append(record)
                    save()
                    print(f"{len(records)}: {kind} {identity}: " + ("accepted" if record["accepted"] else record["rejection"]), flush=True)
        if set(previous_by_id) - identities:
            raise ValueError("Resumed input no longer contains previously processed sessions")
        receipt["complete"] = True  # every requested input was processed
        receipt["accepted_sessions"] = sum(r["accepted"] for r in records)
        receipt["rejected_sessions"] = [r["session_id"] for r in records if not r["accepted"]]
        save()
    except Exception as exc:
        receipt["failure"] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            receipt["tool_error"] = exc.stderr or exc.stdout
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
