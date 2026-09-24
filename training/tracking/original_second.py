"""Original person2 segmentation, retained independently of new3 acquisition."""
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from . import canonical as base
SOURCE_ID = "person2"
SOURCE_SCHEMA = "abcurves.human_tracking_unassigned_source.v1"
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def token(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def make_unassigned_dataset(w, f, scenarios, boundaries, frequency, manifest):
    groups = base.leakage_groups(scenarios)
    ticks, episodes, offsets = defaultdict(list), defaultdict(list), [0]
    excluded = short = 0
    ticks_per_ms, ticks_per_us = frequency // 1000, frequency // 1_000_000
    for start, end in base.active_intervals(boundaries):
        d = base.dense_segment(w, f, start, end, ticks_per_ms)
        if d is None:
            continue
        excluded += int(np.sum(~d["valid"]))
        # A guide visibility change is a source contract boundary, not a quality filter.
        guide = f["guide_enabled"][d["selected"]].astype(bool)
        edges = np.flatnonzero(guide[1:] != guide[:-1]) + 1
        for fragment, (a, b) in enumerate(base.contiguous_runs(d["valid"])):
            spans = np.r_[a, edges[(edges > a) & (edges < b)], b]
            for part, (a, b) in enumerate(zip(spans[:-1], spans[1:])):
                if b - a < base.WARM + base.LABEL + base.LABEL_EXTRA:
                    short += int(b - a)
                    continue
                origin = d["origin"] + int(a) * ticks_per_ms
                chosen = d["selected"][a:b]
                raw = d["raw"][a:b].astype(np.float32)
                cursor = d["cursor"][a:b].astype(np.float64)
                initial = d["initial"] if a == 0 else d["cursor"][a-1]
                if not np.array_equal(raw.astype(np.float64).cumsum(axis=0) + initial, cursor):
                    raise ValueError("episode failed exact applied cursor integration")
                receipt = (f["present_return_qpc"][chosen] - origin + ticks_per_us - 1) // ticks_per_us
                sampled = (f["sample_qpc"][chosen] - origin + ticks_per_us - 1) // ticks_per_us
                endpoint_us = np.arange(1, b-a+1, dtype=np.int64) * 1000
                if np.any(receipt > endpoint_us) or np.any(sampled > receipt):
                    raise ValueError("future target leaked into dense source")
                fields = dict(raw_dxdy=raw, cursor_xy=cursor,
                              target_xy=np.column_stack((f["target_x_counts"][chosen], f["target_y_counts"][chosen])),
                              target_radius_xy=np.column_stack((f["radius_x_counts"][chosen], f["radius_y_counts"][chosen])),
                              target_available_us=np.maximum(receipt, 0), target_sampled_us=np.maximum(sampled, 0),
                              source_target_available_us=receipt, source_target_sampled_us=sampled,
                              target_receipt_age_us=endpoint_us-receipt, target_valid=np.ones(b-a, bool))
                for key, value in fields.items():
                    ticks[key].append(value)
                sid, segment = int(start["scenario_id"]), int(start["segment_id"])
                scenario, guided = scenarios[sid], bool(guide[a])
                style = scenario.get("recoil_style", "none")
                named = f"{scenario['family']}:{style}" if style not in ("", "none") else ""
                values = dict(episode_id=f"{SOURCE_ID}_s{sid:03d}_segment{segment:03d}_part{fragment:02d}_{part:02d}",
                              scenario_id=sid, segment_id=segment, physical_parent_id=f"{SOURCE_ID}:scenario:{sid:03d}",
                              group_id=f"{SOURCE_ID}:{groups[sid]}", source_group_id=f"{SOURCE_ID}:{groups[sid]}",
                              global_named_template_id=named, source_id=SOURCE_ID, participant_id="participant_2",
                              split="unassigned", start_cursor_xy=initial.astype(np.float64), start_qpc=origin,
                              guide_enabled=guided, guide_horizon_us=int(scenario.get("guide_horizon_ms", 1000))*1000 if guided else 0,
                              family=scenario["family"], speed_band=scenario["speed_band"],
                              effective_radians_per_count=float(manifest["protocol_plan"]["effective_radians_per_count"]))
                for key, value in values.items():
                    episodes[key].append(value)
                offsets.append(offsets[-1] + int(b-a))
    data = {k: np.concatenate(v) for k, v in ticks.items()}
    data.update({k: np.asarray(v) for k, v in episodes.items()})
    data.update(episode_offsets=np.asarray(offsets, np.int64), schema=np.array(base.SCHEMA),
                custody_schema=np.array(SOURCE_SCHEMA), sample_period_us=np.array(1000, np.int64),
                source_zip_sha256=np.array(digest(Path(manifest["__source_path"]))),
                participant_identity_sha256=np.array(token(manifest["user_id"])),
                session_identity_sha256=np.array(token(manifest["session_id"])),
                training_window_episode=np.empty(0, np.int64), training_window_start=np.empty(0, np.int64),
                all_training_window_episode=np.empty(0, np.int64), all_training_window_start=np.empty(0, np.int64),
                all_training_window_retained=np.empty(0, bool))
    if any(v.dtype.hasobject for v in data.values()):
        raise ValueError("pickle-requiring array prohibited")
    return data, dict(excluded_fault_bins=excluded, excluded_short_fragment_bins=short,
                      scenario_group={str(k): f"{SOURCE_ID}:{v}" for k, v in groups.items()},
                      no_quality_filter=True, all_roles_unassigned=True)
