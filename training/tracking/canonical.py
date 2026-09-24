"""Canonical Capture readers and physical adapters, extracted with source hashes.

Numerical bodies are copied from selected preparation sources. See provenance.json.
Only import/path ownership was changed; rejected model families are absent.
"""
from array import array
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile
import zlib
import numpy as np

SCHEMA = "abcurves.human_tracking_witness_1ms.v1"
SPLITS = ("train", "validation", "calibration", "test")
WARM, LABEL, LABEL_EXTRA, STRIDE = 160, 256, 2, 64
CRC = re.compile(rb'^\{"_crc32":"([0-9a-f]{8})",')
COMMON = .0003509487083280618

class _LegacyVerifiedZip:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.z = zipfile.ZipFile(self.path)
        names = self.z.namelist()
        roots = [n[:-len("manifest.json")] for n in names if n.endswith("/manifest.json")]
        if len(roots) != 1 or len(names) != len(set(names)) or len(names) > 256:
            raise ValueError("invalid archive inventory")
        self.root = roots[0]
        for n in names:
            p = Path(n)
            if not n.startswith(self.root) or "\\" in n or ":" in n or ".." in p.parts:
                raise ValueError("unsafe archive member")
        checks = self.z.read(self.root + "checksums.sha256")
        complete = json.loads(self.z.read(self.root + "COMPLETE"))
        if hashlib.sha256(checks).hexdigest() != complete["checksums_sha256"]:
            raise ValueError("checksum inventory mismatch")
        self.expected = {}
        for line in checks.decode().splitlines():
            digest, name = line.split("  ", 1)
            if name in self.expected or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("invalid artifact digest")
            self.expected[name] = digest
        if {self.root + n for n in self.expected} | {self.root + "checksums.sha256", self.root + "COMPLETE"} != set(names):
            raise ValueError("unlisted or missing ZIP member")
        self.verified = set()
        self.manifest = json.loads(self.read("manifest.json"))
        if self.manifest["session_id"] != complete["session_id"] or self.manifest["status"] != complete["status"]:
            raise ValueError("completion identity mismatch")
        if self.manifest["status"] != "complete" or self.manifest["protocol_id"] != "abcurves.tracking.protocol-v1":
            raise ValueError("expected completed tracking session")
        if self.manifest["qpc_frequency"] % 1_000_000:
            raise ValueError("this exact integer-microsecond adapter needs integral QPC ticks/us")

    def chunks(self, name):
        h = hashlib.sha256()
        with self.z.open(self.root + name) as f:
            while b := f.read(1 << 20):
                h.update(b)
                yield b
        if h.hexdigest() != self.expected[name]:
            raise ValueError(f"artifact checksum mismatch: {name}")
        self.verified.add(name)

    def read(self, name):
        return b"".join(self.chunks(name))

    def records(self, name, schema):
        h = hashlib.sha256()
        with self.z.open(self.root + name) as f:
            for seq, line in enumerate(f):
                h.update(line)
                m = CRC.match(line)
                if not m or not line.endswith(b"\n") or len(line) > 4 << 20:
                    raise ValueError(f"invalid journal record in {name}")
                if zlib.crc32(b"{" + line[m.end():-1]) != int(m[1], 16):
                    raise ValueError(f"journal CRC mismatch: {name} row {seq}")
                r = json.loads(line)
                if r["sequence"] != seq or r["schema"] != schema:
                    raise ValueError(f"journal schema/sequence mismatch: {name}")
                for key in ("session_id", "user_id", "qpc_frequency", "trainer_sensitivity"):
                    if r[key] != self.manifest[key]:
                        raise ValueError(f"journal identity mismatch: {name}")
                yield r
        if h.hexdigest() != self.expected[name]:
            raise ValueError(f"artifact checksum mismatch: {name}")
        self.verified.add(name)

    def finish_verification(self):
        for n in sorted(self.expected.keys() - self.verified):
            for _ in self.chunks(n):
                pass

class VerifiedZip(_LegacyVerifiedZip):
    """Same strict inventory/checksums, plus verified empty directory entries."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.z = zipfile.ZipFile(self.path)
        infos = self.z.infolist()
        names = [i.filename for i in infos]
        roots = [n[:-len("manifest.json")] for n in names if n.endswith("/manifest.json")]
        if len(roots) != 1 or len(names) != len(set(names)) or len(names) > 256:
            raise ValueError("invalid archive inventory")
        self.root = roots[0]
        self.directory_members = []
        for i in infos:
            n, parts = i.filename, PurePosixPath(i.filename).parts
            if not n.startswith(self.root) or "\\" in n or ":" in n or ".." in parts or PurePosixPath(n).is_absolute():
                raise ValueError("unsafe archive member")
            if i.flag_bits & 1 or stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError("encrypted or symlink archive member")
            if i.is_dir():
                if i.file_size or self.z.read(n):
                    raise ValueError("nonempty directory entry")
                self.directory_members.append(n)
        checks = self.z.read(self.root + "checksums.sha256")
        complete = json.loads(self.z.read(self.root + "COMPLETE"))
        if hashlib.sha256(checks).hexdigest() != complete["checksums_sha256"]:
            raise ValueError("checksum inventory mismatch")
        self.expected = {}
        for line in checks.decode("ascii").splitlines():
            checksum, name = line.split("  ", 1)
            if name in self.expected or not re.fullmatch("[0-9a-f]{64}", checksum):
                raise ValueError("invalid artifact digest")
            if ".." in PurePosixPath(name).parts or ":" in name or "\\" in name or PurePosixPath(name).is_absolute():
                raise ValueError("unsafe inventory name")
            self.expected[name] = checksum
        expected_names = {self.root + n for n in self.expected} | {self.root + "checksums.sha256", self.root + "COMPLETE"}
        if expected_names != set(names) - set(self.directory_members):
            raise ValueError("unlisted or missing ZIP file")
        self.verified = set()
        self.manifest = json.loads(self.read("manifest.json"))
        if self.manifest["session_id"] != complete["session_id"] or self.manifest["status"] != complete["status"]:
            raise ValueError("completion identity mismatch")
        if self.manifest["status"] != "complete" or self.manifest["protocol_id"] != "abcurves.tracking.protocol-v1":
            raise ValueError("expected completed tracking session")
        if int(self.manifest["qpc_frequency"]) % 1_000_000:
            raise ValueError("exact integer-microsecond adapter needs integral QPC ticks/us")

def read_tracking(source):
    specs = {"sample_qpc": "q", "submit_qpc": "q", "present_return_qpc": "q", "scenario_id": "q", "segment_id": "q",
             "elapsed_ms": "d", "target_x_counts": "d", "target_y_counts": "d", "radius_x_counts": "d", "radius_y_counts": "d",
             "cursor_x_counts": "q", "cursor_y_counts": "q", "successful": "b", "active_frame": "b", "guide_enabled": "b"}
    cols = {k: array(t) for k, t in specs.items()}
    scenarios, boundaries, types = {}, [], Counter()
    phases, last_qpc, active_contacts = Counter(), -1, set()
    for r in source.records("gameplay/tracking.jsonl", "abcurves.gameplay.tracking.v1"):
        kind = r["record_type"]
        types[kind] += 1
        if kind == "tracking_scenario":
            sid = int(r["scenario_id"])
            if sid in scenarios:
                raise ValueError("duplicate scenario")
            scenarios[sid] = {k: v for k, v in r.items() if k not in ("session_id", "user_id", "_crc32")}
        elif kind == "tracking_boundary":
            boundaries.append({k: v for k, v in r.items() if k not in ("session_id", "user_id", "_crc32")})
            if r["kind"] == "contact_acquired":
                active_contacts.add((r["scenario_id"], r["segment_id"]))
            if r["kind"] == "segment_start" and (r["scenario_id"], r["segment_id"]) not in active_contacts:
                raise ValueError("active segment lacks contact gate")
        elif kind == "tracking_frame":
            if not 0 <= r["sample_qpc"] <= r["submit_qpc"] <= r["present_return_qpc"] or r["present_return_qpc"] < last_qpc:
                raise ValueError("invalid frame clock order")
            last_qpc = r["present_return_qpc"]
            if r["active_frame"] and r["target_phase"] != "tracking":
                raise ValueError("active frame outside tracking")
            for k, col in cols.items():
                value = r[k]
                if specs[k] == "d" and not math.isfinite(value):
                    raise ValueError("nonfinite geometry")
                col.append(value)
            if min(r["radius_x_counts"], r["radius_y_counts"]) <= 0:
                raise ValueError("invalid target ellipse")
            phases[r["target_phase"]] += 1
        else:
            raise ValueError("unknown tracking record type")
    ended = {r["scenario_id"] for r in boundaries if r["kind"] == "scenario_end"}
    if ended != set(scenarios) or len(scenarios) != source.manifest["event_count"]:
        raise ValueError("incomplete scenario inventory")
    return {k: np.asarray(v) for k, v in cols.items()}, scenarios, boundaries, {"record_types": dict(types), "frame_phases": dict(phases)}

def read_witness(source):
    keys = ("receipt_qpc", "dx_counts", "dy_counts", "cursor_before_x_counts", "cursor_before_y_counts", "cursor_after_x_counts", "cursor_after_y_counts")
    cols = {k: array("q") for k in keys}
    cols.update({k: array("b") for k in ("gameplay_applied", "cursor_clamped", "discontinuity")})
    previous, stats = None, Counter()
    for r in source.records("gameplay/raw_input_witness.jsonl", "abcurves.gameplay.raw_input_witness.v1"):
        if r["authoritative"] is not False or r["authority"] != "non_authoritative_windows_raw_input_witness":
            raise ValueError("witness authority mislabel")
        before = (r["cursor_before_x_counts"], r["cursor_before_y_counts"])
        after = (r["cursor_after_x_counts"], r["cursor_after_y_counts"])
        discontinuity = previous is not None and before != previous[1]
        if previous is not None and r["receipt_qpc"] < previous[0]:
            raise ValueError("witness clock regressed")
        applied = bool(r["gameplay_applied"])
        clamped = bool(r["cursor_clamped"])
        if applied and not clamped and (after[0]-before[0], after[1]-before[1]) != (r["dx_counts"], r["dy_counts"]):
            raise ValueError("unexplained applied-count mismatch")
        if not applied and before != after:
            raise ValueError("nonapplied packet changes cursor")
        for k in keys:
            cols[k].append(r[k])
        for k, v in (("gameplay_applied", applied), ("cursor_clamped", clamped), ("discontinuity", discontinuity)):
            cols[k].append(v)
        stats["packets"] += 1
        stats["applied"] += applied
        stats["clamped"] += clamped
        stats["position_discontinuities"] += discontinuity
        stats["zero_delta_packets"] += not (r["dx_counts"] or r["dy_counts"])
        previous = r["receipt_qpc"], after
    return {k: np.asarray(v) for k, v in cols.items()}, dict(stats)

def leakage_groups(scenarios):
    """Union repeated seeds, explicit repeat IDs, exact paths, and shared templates."""
    parent = {sid: sid for sid in scenarios}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for sid, s in sorted(scenarios.items()):
        tokens = [("seed", str(s["path_seed"]))]
        if s.get("repeat_group", -1) >= 0:
            tokens.append(("repeat", s["family"], s["repeat_group"]))
        if s.get("recoil_style", "none") not in ("none", ""):
            tokens.append(("template", s["family"], s["recoil_style"]))
        if "knots_ms_x_y_vx_vy" in s:
            tokens.append(("path", hashlib.sha256(np.asarray(s["knots_ms_x_y_vx_vy"], dtype="<f8").tobytes()).hexdigest()))
        for token in tokens:
            if token in seen:
                a, b = find(sid), find(seen[token])
                parent[max(a, b)] = min(a, b)
            seen[token] = sid
    return {sid: f"g{find(sid):03d}" for sid in scenarios}

def split_groups(scenarios, groups, seed=20260913):
    """Deterministic source-only allocation; no cursor quality enters the split."""
    grouped = defaultdict(list)
    for sid, group in groups.items():
        grouped[group].append(sid)
    strata = sorted({(s["family"], bool(s["guide_enabled"])) for s in scenarios.values()})
    dims = {v: i for i, v in enumerate(strata)}
    weights = {}
    for group, sids in grouped.items():
        v = np.zeros(len(strata), dtype=float)
        for sid in sids:
            s = scenarios[sid]
            v[dims[s["family"], bool(s["guide_enabled"])]] += s["duration_ms"]
        weights[group] = v
    total = sum(weights.values())
    proportions = np.array([.65, .15, .10, .10])
    target = proportions[:, None] * total
    best = None
    rng = np.random.default_rng(seed)
    # Source metadata-only multi-start greedy balancing; independent of movement.
    for attempt in range(96):
        order = sorted(grouped, key=lambda g: (-float(np.sum(weights[g])) * float(rng.uniform(.5, 1.5)), g))
        actual = np.zeros_like(target)
        assigned = {}
        for group in order:
            scores = []
            for j in range(4):
                trial = actual.copy()
                trial[j] += weights[group]
                scores.append(float(np.sum((trial-target)**2 / np.maximum(total[None, :], 1)**2)) + .1*float(np.sum((trial.sum(axis=1)-target.sum(axis=1))**2)/total.sum()**2))
            j = int(np.argmin(scores))
            assigned[group] = SPLITS[j]
            actual[j] += weights[group]
        score = float(np.sum((actual-target)**2/np.maximum(total[None, :], 1)**2))
        score += 5 * sum(not any(v == split for v in assigned.values()) for split in SPLITS)
        if best is None or score < best[0]:
            best = score, assigned
    return {sid: best[1][groups[sid]] for sid in scenarios}

def active_intervals(boundaries):
    current, result = None, []
    for r in boundaries:
        if r["kind"] == "segment_start":
            if current is not None:
                raise ValueError("overlapping active segments")
            current = r
        elif current is not None and r["kind"] in ("scenario_end", "manual_pause", "focus_lost", "capture_error"):
            if r["scenario_id"] != current["scenario_id"]:
                raise ValueError("mismatched active boundary")
            result.append((current, r))
            current = None
    if current is not None:
        raise ValueError("unterminated active segment")
    return result

def contiguous_runs(valid):
    edges = np.diff(np.r_[False, np.asarray(valid, bool), False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def dense_segment(w, f, start, end, ticks_per_ms):
    """Right-closed receipt bins (origin,origin+1ms]; only causally available frames."""
    origin = ((int(start["qpc"]) + ticks_per_ms - 1) // ticks_per_ms) * ticks_per_ms
    n = (int(end["qpc"]) - origin) // ticks_per_ms
    if n < 1:
        return None
    endpoints = origin + np.arange(1, n+1, dtype=np.int64) * ticks_per_ms
    tq = w["receipt_qpc"]
    begin_index = np.searchsorted(tq, origin, side="right") - 1
    if begin_index < 0:
        raise ValueError("no witnessed initial cursor")
    wi = np.searchsorted(tq, endpoints, side="right") - 1
    initial = np.array([w["cursor_after_x_counts"][begin_index], w["cursor_after_y_counts"][begin_index]], dtype=np.int64)
    cursor = np.column_stack((w["cursor_after_x_counts"][wi], w["cursor_after_y_counts"][wi]))
    lo, hi = begin_index+1, int(np.searchsorted(tq, endpoints[-1], side="right"))
    bins = (tq[lo:hi] - origin - 1) // ticks_per_ms
    raw = np.zeros((n, 2), np.int64)
    np.add.at(raw[:, 0], bins, w["dx_counts"][lo:hi])
    np.add.at(raw[:, 1], bins, w["dy_counts"][lo:hi])
    bad = (~w["gameplay_applied"][lo:hi].astype(bool)) | w["cursor_clamped"][lo:hi].astype(bool) | w["discontinuity"][lo:hi].astype(bool)
    valid = np.ones(n, bool)
    valid[bins[bad]] = False
    same_scene = (f["scenario_id"] == start["scenario_id"]) & (f["segment_id"] == start["segment_id"]) & f["successful"].astype(bool)
    fi = np.flatnonzero(same_scene)
    if len(fi) == 0:
        raise ValueError("active segment lacks successful frames")
    chosen = np.searchsorted(f["present_return_qpc"][fi], endpoints, side="right")-1
    valid &= chosen >= 0
    selected = fi[np.maximum(chosen, 0)]
    actual_delta = np.diff(np.vstack((initial, cursor)), axis=0)
    valid &= np.all(actual_delta == raw, axis=1)
    return dict(origin=origin, raw=raw, cursor=cursor, initial=initial, valid=valid, selected=selected)

def quality_windows(data):
    episodes, starts, occupancies, errors = [], [], [], []
    offsets = data["episode_offsets"]
    length = WARM+LABEL+LABEL_EXTRA
    for e in range(len(offsets)-1):
        if data["split"][e] != "train":
            continue
        lo, hi = offsets[e:e+2]
        norm = np.sqrt(np.sum(((data["cursor_xy"][lo:hi]-data["target_xy"][lo:hi])/data["target_radius_xy"][lo:hi])**2, axis=1))
        ci = np.r_[0., np.cumsum(norm <= 1)]
        ce = np.r_[0., np.cumsum(np.minimum(norm, 5))]
        for at in range(0, int(hi-lo)-length+1, STRIDE):
            stop = at+WARM+LABEL
            episodes.append(e); starts.append(at)
            occupancies.append((ci[stop]-ci[at])/(WARM+LABEL))
            errors.append((ce[stop]-ce[at])/(WARM+LABEL))
    episodes, starts = np.asarray(episodes, np.int64), np.asarray(starts, np.int64)
    occupancies, errors = np.asarray(occupancies), np.asarray(errors)
    # Worst occupancy first; clipped mean ellipse error only breaks occupancy ties.
    order = np.lexsort((starts, episodes, -errors, occupancies))
    dropped = int(math.floor(len(order)*.05))
    retained = np.ones(len(order), bool)
    retained[order[:dropped]] = False
    return dict(training_window_episode=episodes[retained], training_window_start=starts[retained],
                all_training_window_episode=episodes, all_training_window_start=starts,
                all_training_window_retained=retained, all_training_window_occupancy=occupancies,
                all_training_window_error=errors)

def make_dataset(w, f, scenarios, boundaries, frequency, seed):
    groups = leakage_groups(scenarios)
    splits = split_groups(scenarios, groups, seed)
    tick_fields = defaultdict(list)
    epi = defaultdict(list)
    offsets = [0]
    excluded_bins = short_bins = 0
    ticks_per_ms, ticks_per_us = frequency//1000, frequency//1_000_000
    for start, end in active_intervals(boundaries):
        d = dense_segment(w, f, start, end, ticks_per_ms)
        if d is None:
            continue
        excluded_bins += int(np.sum(~d["valid"]))
        for frag, (a, b) in enumerate(contiguous_runs(d["valid"])):
            if b-a < WARM+LABEL+LABEL_EXTRA:
                short_bins += int(b-a)
                continue
            origin = d["origin"] + int(a)*ticks_per_ms
            chosen = d["selected"][a:b]
            cursor = d["cursor"][a:b].astype(np.float64)
            raw = d["raw"][a:b].astype(np.float32)
            initial = d["initial"] if a == 0 else d["cursor"][a-1]
            if not np.array_equal(raw.astype(np.float64).cumsum(axis=0)+initial, cursor):
                raise ValueError("episode failed exact cursor integration")
            receipt = (f["present_return_qpc"][chosen]-origin+ticks_per_us-1)//ticks_per_us
            sampled = (f["sample_qpc"][chosen]-origin+ticks_per_us-1)//ticks_per_us
            endpoint_us = np.arange(1, b-a+1, dtype=np.int64)*1000
            if np.any(receipt > endpoint_us) or np.any(sampled > receipt):
                raise ValueError("future target leaked into dense history")
            guided = f["guide_enabled"][chosen].astype(bool)
            if not np.all(guided == guided[0]):
                raise ValueError("guide change within episode requires explicit boundary")
            fields = dict(raw_dxdy=raw, cursor_xy=cursor,
                          target_xy=np.column_stack((f["target_x_counts"][chosen], f["target_y_counts"][chosen])),
                          target_radius_xy=np.column_stack((f["radius_x_counts"][chosen], f["radius_y_counts"][chosen])),
                          target_available_us=np.maximum(receipt, 0), target_sampled_us=np.maximum(sampled, 0),
                          source_target_available_us=receipt, source_target_sampled_us=sampled,
                          target_receipt_age_us=endpoint_us-receipt, target_valid=np.ones(b-a, bool))
            for k, v in fields.items():
                tick_fields[k].append(v)
            sid = int(start["scenario_id"])
            values = dict(episode_id=f"s{sid:03d}_segment{int(start['segment_id']):03d}_part{frag:02d}", scenario_id=sid,
                          segment_id=int(start["segment_id"]), group_id=groups[sid], split=splits[sid],
                          start_cursor_xy=initial.astype(np.float64), start_qpc=origin,
                          guide_enabled=bool(guided[0]), guide_horizon_us=1_000_000 if guided[0] else 0,
                          family=scenarios[sid]["family"], speed_band=scenarios[sid]["speed_band"])
            for k, v in values.items():
                epi[k].append(v)
            offsets.append(offsets[-1]+int(b-a))
    data = {k: np.concatenate(v) for k, v in tick_fields.items()}
    data.update({k: np.asarray(v) for k, v in epi.items()})
    data.update(episode_offsets=np.asarray(offsets, np.int64), schema=np.array(SCHEMA), sample_period_us=np.array(1000, np.int64))
    data.update(quality_windows(data))
    if any(v.dtype.hasobject for v in data.values()):
        raise ValueError("pickle-requiring array prohibited")
    for group in np.unique(data["group_id"]):
        if len(np.unique(data["split"][data["group_id"] == group])) != 1:
            raise ValueError("group split leakage")
    return data, dict(excluded_fault_bins=excluded_bins, excluded_short_fragment_bins=short_bins,
                      scenario_group={str(k): v for k, v in groups.items()}, scenario_split={str(k): v for k, v in splits.items()})

def varuint(data, offset):
    value = 0
    for shift in range(0, 64, 7):
        if offset >= len(data):
            raise ValueError("truncated report varuint")
        byte = data[offset]; offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if shift == 63 and byte > 1:
                raise ValueError("overflowing report varuint")
            return value, offset
    raise ValueError("overflowing report varuint")

def varint(data, offset):
    value, offset = varuint(data, offset)
    return (value >> 1) ^ -(value & 1), offset

def parse_usb(source):
    descriptor = source.read("capture/hid_report_descriptor.bin")
    stream = io.BytesIO(source.read("capture/mouse_reports.abcr2"))
    if stream.read(8) != b"ABCRPT2\0":
        raise ValueError("invalid ABCR2 magic")
    version, bus, device, endpoint, reserved, evidence_length, spec_length, frequency = struct.unpack("<HHHBBIIQ", stream.read(24))
    d = source.manifest["device"]
    if (version, reserved, bus, device, endpoint, frequency) != (2, 0, d["usb_bus"], d["usb_device"], d["interrupt_in_endpoint"], source.manifest["qpc_frequency"]):
        raise ValueError("report identity mismatch")
    descriptor_sha = stream.read(64).decode("ascii")
    evidence, spec = stream.read(evidence_length), stream.read(spec_length)
    if descriptor != evidence or hashlib.sha256(evidence).hexdigest() != descriptor_sha or not spec.startswith(b"abdc.hid.decoder.v1\n"):
        raise ValueError("report descriptor evidence mismatch")
    cols = {k: array("q") for k in ("capture_sequence", "pcap_sequence", "report_index_in_transfer", "reports_in_transfer", "capture_unix_ns", "observed_qpc", "hid_dx", "hid_dy", "buttons", "quality_flags")}
    previous = [None]*4
    blocks = 0
    while magic := stream.read(4):
        if magic != b"RBLK":
            raise ValueError("invalid report block magic")
        size, count, crc = struct.unpack("<III", stream.read(12))
        if not 0 < size <= 16 << 20 or not 0 < count <= 65536:
            raise ValueError("invalid report block size")
        block = stream.read(size)
        if len(block) != size or zlib.crc32(block) != crc:
            raise ValueError("report block CRC failure")
        offset = 0
        for _ in range(count):
            seq, offset = varuint(block, offset)
            pcap, offset = varuint(block, offset)
            ri, offset = varuint(block, offset)
            nr, offset = varuint(block, offset)
            capture, offset = varint(block, offset)
            qpc, offset = varint(block, offset)
            vals = [seq, pcap, capture, qpc]
            vals = [v if prev is None else v+prev for v, prev in zip(vals, previous)]
            seq, pcap, capture, qpc = vals
            if (previous[0] is not None and seq != previous[0]+1) or (previous[1] is not None and pcap < previous[1]) or (previous[3] is not None and qpc < previous[3]) or not 0 <= ri < nr:
                raise ValueError("report ordering failure")
            _, status, _, rb, rd = struct.unpack_from("<QIHHH", block, offset); offset += 18
            ep, transfer, info, _ = struct.unpack_from("<BBBB", block, offset); offset += 4
            dx, offset = varint(block, offset)
            dy, offset = varint(block, offset)
            _, offset = varint(block, offset)
            _, offset = varint(block, offset)
            buttons, offset = varuint(block, offset)
            quality, offset = varuint(block, offset)
            payload, offset = varuint(block, offset)
            if (status, rb, rd, ep, transfer, info) != (0, bus, device, endpoint, 1, 1) or quality & ~3 or not 0 < payload <= 4096:
                raise ValueError("report violates authoritative interrupt-IN contract")
            offset += payload
            if offset > len(block):
                raise ValueError("truncated report payload")
            for k, v in zip(cols, (seq, pcap, ri, nr, capture, qpc, dx, dy, buttons, quality)):
                cols[k].append(v)
            previous = vals
        if offset != size:
            raise ValueError("report block trailing data")
        blocks += 1
    if len(cols["capture_sequence"]) != int(source.manifest["decoded_reports"]):
        raise ValueError("report count mismatch")
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["raw_hid_dxdy"] = np.column_stack((out.pop("hid_dx"), out.pop("hid_dy"))).astype(np.int32)
    out["canonical_dxdy"] = out["raw_hid_dxdy"]*np.array([1, -1], dtype=np.int32)
    out["buttons"] = out["buttons"].astype(np.uint32)
    out["quality_flags"] = out["quality_flags"].astype(np.uint32)
    return out, blocks

def small_journals(source):
    out = {}
    for name in ("clocks/anchors.jsonl", "gameplay/lifecycle.jsonl", "gameplay/pauses.jsonl"):
        rows = []
        for seq, line in enumerate(source.read(name).splitlines(keepends=True)):
            match = CRC.match(line)
            if not match or not line.endswith(b"\n") or zlib.crc32(b"{" + line[match.end():-1]) != int(match[1], 16):
                raise ValueError("small journal CRC mismatch")
            r = json.loads(line)
            if r["sequence"] != seq or int(r["qpc_frequency"]) != source.manifest["qpc_frequency"]:
                raise ValueError("small journal sequence/clock mismatch")
            rows.append(r)
        out[name] = rows
    return out

def dense_from_boundary(w,f,beginning,end):
    """Original dense adapter, with genuine boundary state before first witness.

    A session's first countdown can precede its first movement packet. The
    boundary's recorded cursor supplies initial state in that case; no zero USB
    or WM report is invented. The first witness before-position must agree.
    """
    origin=((int(beginning["qpc"])+9999)//10000)*10000
    if np.searchsorted(w["receipt_qpc"],origin,side="right")>0:
        return dense_segment(w,f,beginning,end,10000),"preceding_witness"
    n=(int(end["qpc"])-origin)//10000
    initial=np.array([beginning["cursor_x_counts"],beginning["cursor_y_counts"]],np.int64)
    first_before=np.array([w["cursor_before_x_counts"][0],w["cursor_before_y_counts"][0]],np.int64)
    if not np.array_equal(initial,first_before):
        raise ValueError("Initial authentic boundary and first witness disagree")
    endpoints=origin+np.arange(1,n+1,dtype=np.int64)*10000
    wi=np.searchsorted(w["receipt_qpc"],endpoints,side="right")-1
    cursor=np.column_stack((w["cursor_after_x_counts"][wi.clip(0)],w["cursor_after_y_counts"][wi.clip(0)]))
    cursor[wi<0]=initial
    hi=int(np.searchsorted(w["receipt_qpc"],endpoints[-1],side="right"))
    bins=(w["receipt_qpc"][:hi]-origin-1)//10000
    raw=np.zeros((n,2),np.int64)
    np.add.at(raw[:,0],bins,w["dx_counts"][:hi]);np.add.at(raw[:,1],bins,w["dy_counts"][:hi])
    bad=(~w["gameplay_applied"][:hi].astype(bool))|w["cursor_clamped"][:hi].astype(bool)|w["discontinuity"][:hi].astype(bool)
    valid=np.ones(n,bool);valid[bins[bad]]=False
    fi=np.flatnonzero((f["scenario_id"]==beginning["scenario_id"])&(f["segment_id"]==beginning["segment_id"])&f["successful"].astype(bool))
    chosen=np.searchsorted(f["present_return_qpc"][fi],endpoints,side="right")-1
    valid &= chosen>=0
    valid &= np.all(np.diff(np.vstack((initial,cursor)),axis=0)==raw,axis=1)
    return dict(origin=origin,raw=raw,cursor=cursor,initial=initial,valid=valid,selected=fi[chosen.clip(0)]),"authentic_boundary_before_first_witness"

def smooth(xy, initial):
    return np.column_stack([np.convolve(np.pad(xy[:, j], (4, 0), constant_values=initial[j]),
                                       np.array([1, 2, 3, 2, 1]) / 9., 'valid') for j in range(2)])

def bin_development_episode(source,row,allowed_roles):
    """One fixed-clock slice; reject held/falsified roles before report access."""
    person,scenario=str(row["participant_id"]),int(row["scenario_id"])
    role=str(row["split"])
    if role not in ("train","validation") or allowed_roles.get((person,scenario))!=role:
        raise ValueError("USB development derivation rejects held or mismatched roles")
    if source.source_id!=person:raise ValueError("USB source identity differs from episode")
    origin,length=int(row["start_qpc"]),int(row["ticks"])
    if length<=0 or source.frequency!=10_000_000:raise ValueError("invalid development time grid")
    ends=origin+np.arange(length+1,dtype=np.int64)*source.step
    if ends[0]<source.times[0] or ends[-1]>source.times[-1]:raise ValueError("episode outside captured coverage")
    index=np.searchsorted(source.times,ends,side="right")
    selected=source.order[int(index[0]):int(index[-1])]
    fields={name:value[selected] for name,value in source.fields.items()}
    if np.any(fields["quality_flags"]):raise ValueError("flagged report in development interval")
    if not np.array_equal(fields["canonical_dxdy"],fields["raw_hid_dxdy"]*np.array([1,-1])):
        raise ValueError("USB canonical axis relation changed")
    counts=fields["canonical_dxdy"].astype(np.int64)
    cumulative=np.vstack((np.zeros(2,np.int64),np.cumsum(counts,axis=0)))
    local=index-index[0]
    native=np.diff(cumulative[local],axis=0)
    report_count=np.diff(local).astype(np.int64)
    # Capture endpoints are right-closed. A report exactly on an endpoint
    # belongs to the preceding interval, including availability aggregation.
    bins=(fields["estimated_capture_qpc"]-origin-1)//source.step
    observed_latest=np.full(length,-1,np.int64)
    observed_us=(fields["observed_qpc"]-origin+9)//10
    np.maximum.at(observed_latest,bins,observed_us)
    if not np.array_equal(observed_latest==-1,report_count==0):raise ValueError("empty-bin availability sentinel mismatch")
    if np.any(native[report_count==0]):raise ValueError("empty USB bin has nonzero counts")
    if len(selected) and np.any(np.diff(source.fields["estimated_capture_qpc"][np.sort(selected)])<0):
        raise ValueError("capture regression is not eligible for silent repair")
    return dict(native=native,report_count=report_count,observed_latest_us=observed_latest,
        capture_qpc=fields["estimated_capture_qpc"],observed_qpc=fields["observed_qpc"],
        source_report_indices=selected,ends=ends,local_indices=local)

def derive_position(native,scale,origin):
    delta=np.asarray(native,np.int64)*np.asarray(scale,np.float64)
    cursor=np.asarray(origin,np.float64)+np.cumsum(delta,axis=0)
    if not np.isfinite(delta).all() or not np.isfinite(cursor).all():raise ValueError("nonfinite USB geometry")
    return delta,cursor
