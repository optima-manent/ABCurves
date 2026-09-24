"""Build local release attachments with byte-identical raw session ZIPs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abcurves.capture_archives import archive_sources, extract_session, protocol, session_manifest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--collection", choices=["static", "tracking"], required=True)
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--part-limit-mib", type=int, default=1800)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    validator = args.validator.resolve()
    if not validator.is_file():
        parser.error("Capture validator is missing")
    if not 1 <= args.part_limit_mib <= 1900:
        parser.error("part limit must be 1..1900 MiB")
    stem = f"abcurves-{args.collection}-captures-{args.version}"
    inventory = {"schema": "abcurves.raw_release.v1", "license": "CC-BY-4.0",
                 "attribution": "ABCurves datasets — Optima Manent and contributors",
                 "version": args.version, "collection": args.collection,
                 "capture_validator_sha256": sha256(validator),
                 "complete": False, "sessions": [], "parts": []}
    active, part_records, part_path = None, [], None
    def write_inventory():
        (output / f"{stem}-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    def close_part():
        nonlocal active
        if active is None:
            return
        active.writestr("inventory.json", json.dumps({"license": "CC-BY-4.0", "sessions": part_records}, indent=2))
        active.writestr("DATASET_LICENSE.md", (ROOT / "DATASET_LICENSE.md").read_text(encoding="utf-8"))
        active.writestr("README.txt", "ABCurves raw Capture sessions. Extract this ZIP, then validate and prepare with the ABCurves tools.\n"
                        "Session archives and their sealed source evidence are unchanged.\n"
                        "https://github.com/optima-manent/ABCurves/blob/main/docs/DATASET.md\n")
        active.close()
        inventory["parts"].append({"file": part_path.name, "bytes": part_path.stat().st_size,
            "sha256": sha256(part_path), "sessions": len(part_records),
            "intended_url": f"https://github.com/optima-manent/ABCurves/releases/download/v{args.version}/{part_path.name}"})
        active = None
    seen = set()
    try:
        with archive_sources(args.input, scratch=output) as sources:
            for source, lineage in sources:
                if not source.is_file():
                    raise ValueError("Packaging preserves original ZIP bytes; supply session archives")
                with zipfile.ZipFile(source) as z:
                    name, manifest = session_manifest(z)
                    checksum_path = (Path(name).parent / "checksums.sha256").as_posix()
                    checksums_sha = hashlib.sha256(z.read(checksum_path)).hexdigest()
                kind = protocol(manifest)
                if kind != args.collection:
                    raise ValueError(f"{manifest['session_id']} belongs to {kind}, not {args.collection}")
                identity = manifest["session_id"]
                if identity in seen:
                    raise ValueError("Duplicate session identity: " + identity)
                seen.add(identity)
                # Some original transports contain explicit ZIP directories that
                # Capture's ZIP reader rejects. Validate unchanged sealed files;
                # the release attachment still carries the exact original ZIP.
                with tempfile.TemporaryDirectory(prefix="validate-", dir=output) as scratch:
                    session, _ = extract_session(source, scratch)
                    validation = subprocess.run([str(validator), "validate", str(session)],
                                                 capture_output=True, text=True, check=True)
                size = source.stat().st_size
                limit = args.part_limit_mib * 1024**2
                if size > limit:
                    raise ValueError("One unchanged session archive exceeds the part limit")
                if active is not None and part_path.stat().st_size + size + 1024**2 > limit:
                    close_part()
                if active is None:
                    part_path = output / f"{stem}-part{len(inventory['parts'])+1:02d}.zip"
                    active = zipfile.ZipFile(part_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True)
                    part_records = []
                original_name = Path(lineage.get("member", lineage["source"])).name
                member = f"sessions/{identity}/{original_name}"
                active.write(source, member)
                record = {**lineage, "source_member": lineage.get("member"), "session_id": identity, "user_id": manifest["user_id"],
                          "protocol_id": manifest["protocol_id"],
                          "application_version": manifest["application_version"],
                          "capture_source_revision": manifest.get("source_revision"),
                          "status": manifest["status"], "bytes": size,
                          "archive_sha256": sha256(source), "checksums_sha256": checksums_sha,
                          "validation": validation.stdout.strip(), "part": part_path.name, "member": member}
                inventory["sessions"].append(record)
                part_records.append(record)
                write_inventory()
                print(f"{len(seen)} validated and packaged: {identity}", flush=True)
        close_part()
        inventory["complete"] = True
        inventory["session_count"] = len(seen)
        inventory["recorded_user_ids"] = len({r["user_id"] for r in inventory["sessions"]})
        write_inventory()
        (output / f"{stem}-SHA256SUMS.txt").write_text("".join(f"{p['sha256']}  {p['file']}\n" for p in inventory["parts"]) + f"{sha256(output / (stem + '-inventory.json'))}  {stem}-inventory.json\n", encoding="utf-8")
    finally:
        if active is not None:
            active.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
