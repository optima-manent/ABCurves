"""Person3 active data and its TRAIN-only acquisition replacements.

Bodies extracted from the selected sources. Numerical slices, clocks, phase
labels, and source-specific IDs are retained; paths and custody inputs are explicit.
"""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from .canonical import active_intervals,contiguous_runs,dense_segment,dense_from_boundary
from .canonical import smooth as smooth_position,COMMON as REFERENCE_RAD
from .support import read,write,sha,load as load_npz

def save_episode(out, metadata, *, xy, initial, target, radius, available, valid, raw=None):
    values = dict(xy=np.asarray(xy, np.float64), initial=np.asarray(initial, np.float64),
                  target=np.asarray(target, np.float64), radius=np.asarray(radius, np.float64),
                  available=np.asarray(available, np.int64), valid=np.asarray(valid, bool))
    values["delta"] = np.diff(np.vstack((values["initial"], values["xy"])), axis=0)
    if raw is not None:
        values["raw"] = np.asarray(raw, np.float64)
    length = len(xy)
    if not all(np.isfinite(values[k]).all() for k in ("xy", "initial", "target", "radius", "delta")):
        raise ValueError("Nonfinite episode data")
    if not (values["radius"] > 0).all() or not values["valid"].all():
        raise ValueError("Invalid target in retained episode")
    if not np.all(values["available"] <= np.arange(1, length+1) * 1000):
        raise ValueError("Future target receipt in episode")
    name = metadata["id"].replace(":", "_") + ".npz"
    path = out / "episodes" / name
    np.savez_compressed(path, **values)
    return {**metadata, "ticks": length, "file": "episodes/" + name, "sha256": sha(path)}

def prepare_active(projection, out, contract, included=("train", "validation")):
    roles=contract["roles"]
    meta=read(projection/"tracking.json")
    (out/"episodes").mkdir(parents=True,exist_ok=False)
    selected = {int(r["scenario_id"]): r for r in roles["person3"] if r["split"] in included}
    verified = dict(geometry=contract["geometry"])
    print(json.dumps({"event": "loading_projected_person3", "included_roles": list(included)}), flush=True)
    f, w, usb = [load_npz(projection / name) for name in ("frames.npz", "witness.npz", "usb.npz")]
    calibration = dict(coordinate_calibration=read(projection/"metadata.json")["manifest"]["protocol_plan"])
    factor = calibration["coordinate_calibration"]["effective_radians_per_count"] / REFERENCE_RAD
    times = usb["estimated_capture_qpc"]
    if np.any(np.diff(times) < 0) or np.any(usb["quality_flags"]):
        raise ValueError("Unusable native capture clock/quality")
    cumulative = np.vstack((np.zeros(2, np.int64), usb["canonical_dxdy"].astype(np.int64).cumsum(0)))
    # Subset frames by segment once; avoids scanning ten million rows for every episode.
    scene = np.column_stack((f["scenario_id"], f["segment_id"]))
    edges = np.r_[0, np.flatnonzero(np.any(np.diff(scene,axis=0),axis=1))+1, len(scene)]
    spans = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        spans.setdefault(tuple(map(int,scene[lo])), []).append((int(lo),int(hi)))
    rows, excluded = [], []
    for start, end in active_intervals(meta["boundaries"]):
        sid, segment = int(start["scenario_id"]), int(start["segment_id"])
        if sid not in selected:
            continue
        pieces = spans.get((sid, segment), [])
        indices = np.concatenate([np.arange(a,b) for a,b in pieces])
        local = {k: v[indices] for k,v in f.items()}
        d = dense_segment(w, local, start, end, 10000)
        if d is None:
            continue
        guide = local["guide_enabled"][d["selected"]].astype(bool)
        guides = np.flatnonzero(guide[1:] != guide[:-1]) + 1
        for fragment, (first, last) in enumerate(contiguous_runs(d["valid"])):
            cuts = np.r_[first, guides[(guides > first)&(guides < last)], last]
            for part, (a,b) in enumerate(zip(cuts[:-1], cuts[1:])):
                length = int(b-a)
                if length < 192:
                    excluded.append({"scenario": sid,"reason":"short_fragment","ticks":length}); continue
                origin = int(d["origin"] + a*10000)
                endpoints = origin + np.arange(length+1,dtype=np.int64)*10000
                if endpoints[0] < times[0] or endpoints[-1] > times[-1]:
                    excluded.append({"scenario":sid,"reason":"outside_capture_coverage","ticks":length}); continue
                ix = np.searchsorted(times,endpoints,side="right")
                raw = np.diff(cumulative[ix],axis=0) * factor
                initial = (d["initial"] if a == 0 else d["cursor"][a-1]) * factor
                xy = smooth_position(initial + raw.cumsum(0), initial)
                chosen = d["selected"][a:b]
                target = np.column_stack((local["target_x_counts"][chosen],local["target_y_counts"][chosen])) * factor
                radius = np.column_stack((local["radius_x_counts"][chosen],local["radius_y_counts"][chosen])) * factor
                available = (local["present_return_qpc"][chosen]-origin+9)//10
                role = selected[sid]
                metadata = {"id":f"person3_s{sid:03d}_seg{segment:03d}_{fragment:02d}_{part:02d}",
                            "person":"person3","parent":f"person3:scenario:{sid:03d}", "group":"person3:"+role["group"],
                            "split":role["split"],"family":role["family"],"guide":bool(guide[a]),"scenario_id":sid,
                            "source_to_common":factor,"source_counts_per_pixel":[verified["geometry"]["counts_per_pixel_x"], verified["geometry"]["counts_per_pixel_y"]],
                            "source":"person3_authenticated_capture", "start_qpc":origin}
                rows.append(save_episode(out, metadata, xy=xy,initial=initial,target=target,radius=radius,
                                         available=available,valid=np.ones(length,bool),raw=raw))
        print(json.dumps({"event":"person3_segment","scenario":sid,"retained_episodes":len(rows)}), flush=True)
    return rows, excluded

def prepare_acquisition(projection,out,contract,original_rows):
    roles=contract["roles"]
    source_index=dict(episodes=original_rows)
    role_by_sid={int(r["scenario_id"]):r for r in roles["person3"]}
    train_sids={sid for sid,r in role_by_sid.items() if r["split"]=="train"}
    old_rows={r["scenario_id"]:r for r in source_index["episodes"] if r["person"]=="person3" and r["split"]=="train"}
    if train_sids != set(old_rows):
        raise ValueError("Expected matching TRAIN parents")

    meta=read(projection/"tracking.json")
    factor=read(projection/"metadata.json")["manifest"]["protocol_plan"]["effective_radians_per_count"]/REFERENCE_RAD
    f,w,usb=[load_npz(projection/(name+".npz")) for name in ("frames","witness","usb")]
    times=usb["estimated_capture_qpc"]
    if np.any(np.diff(times)<0) or np.any(usb["quality_flags"]):
        raise ValueError("Bad authoritative capture projection")
    cumulative=np.vstack((np.zeros(2,np.int64),usb["canonical_dxdy"].astype(np.int64).cumsum(axis=0)))
    starts={}
    for boundary in meta["boundaries"]:
        key=(int(boundary["scenario_id"]),int(boundary["segment_id"]))
        if key[0] not in train_sids:
            continue
        if boundary["kind"] in ("scenario_countdown","resume_countdown"):
            starts[key]=boundary
        elif boundary["kind"] in ("manual_pause","focus_lost","capture_error"):
            starts.pop(key,None)
    # Build projection spans without altering the original frame row identities.
    scene=np.column_stack((f["scenario_id"],f["segment_id"]))
    edges=np.r_[0,np.flatnonzero(np.any(np.diff(scene,axis=0),axis=1))+1,len(scene)]
    spans={}
    for a,b in zip(edges[:-1],edges[1:]):
        spans.setdefault(tuple(map(int,scene[a])),[]).append((int(a),int(b)))
    out.mkdir()
    (out/"episodes").mkdir()
    # Byte-identical existing role authority. P3 TRAIN membership is immutable.
    write(out/"roles.json",roles)
    result,excluded,checks=[],[],[]
    phase_kinds={"scenario_countdown":0,"resume_countdown":0,"awaiting_presentation":1,
                 "awaiting_contact":2,"contact_acquired":3,"segment_start":3}
    for active_start,end in active_intervals(meta["boundaries"]):
        sid,segment=int(active_start["scenario_id"]),int(active_start["segment_id"])
        if sid not in train_sids:
            continue
        beginning=starts.get((sid,segment))
        if beginning is None or beginning["qpc"]>=active_start["qpc"]:
            raise ValueError("Missing uninterrupted authentic acquisition beginning")
        if any(b["kind"] in ("manual_pause","focus_lost","capture_error") and beginning["qpc"]<b["qpc"]<end["qpc"] for b in meta["boundaries"]):
            raise ValueError("Refuse to cross interruption")
        ix=np.concatenate([np.arange(a,b) for a,b in spans[(sid,segment)]])
        local={k:v[ix] for k,v in f.items()}
        dense,initial_authority=dense_from_boundary(w,local,beginning,end)
        if dense is None:
            excluded.append({"scenario_id":sid,"reason":"empty"});continue
        chosen0=dense["selected"]
        guide=local["guide_enabled"][chosen0].astype(bool)
        guide_breaks=np.flatnonzero(guide[1:]!=guide[:-1])+1
        for frag,(first,last) in enumerate(contiguous_runs(dense["valid"])):
            cuts=np.r_[first,guide_breaks[(guide_breaks>first)&(guide_breaks<last)],last]
            for part,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
                length=int(b-a)
                if length<192:
                    excluded.append({"scenario_id":sid,"reason":"short_fragment","ticks":length});continue
                origin=int(dense["origin"]+a*10000)
                endpoints=origin+np.arange(length+1,dtype=np.int64)*10000
                if endpoints[0]<times[0] or endpoints[-1]>times[-1]:
                    raise ValueError("Outside capture coverage")
                usbix=np.searchsorted(times,endpoints,side="right")
                raw_native=np.diff(cumulative[usbix],axis=0)
                raw=raw_native*factor
                initial_native=dense["initial"] if a==0 else dense["cursor"][a-1]
                initial=initial_native*factor
                xy=smooth_position(initial+raw.cumsum(axis=0),initial)
                delta=np.diff(np.vstack((initial,xy)),axis=0)
                selected=chosen0[a:b]
                target=np.column_stack((local["target_x_counts"][selected],local["target_y_counts"][selected]))*factor
                radius=np.column_stack((local["radius_x_counts"][selected],local["radius_y_counts"][selected]))*factor
                frame_return=local["present_return_qpc"][selected]
                available=(frame_return-origin+9)//10
                t=endpoints[1:]
                phase=np.full(length,-1,np.int8)
                for bound in meta["boundaries"]:
                    if (bound["scenario_id"],bound["segment_id"])==(sid,segment) and bound["kind"] in phase_kinds:
                        phase[t>=bound["qpc"]]=phase_kinds[bound["kind"]]
                if np.any(phase<0) or np.any(frame_return>t) or np.any(available>np.arange(1,length+1)*1000):
                    raise ValueError("Phase/receipt causality failure")
                acquisition=t<=active_start["qpc"]
                witness_xy=dense["cursor"][a:b]*factor
                usb_xy=initial+raw.cumsum(axis=0)
                # Target motion during countdown must be nearly exactly frozen.
                ac_target=target[acquisition]
                target_extent=float(np.ptp(ac_target,axis=0).max()) if len(ac_target) else 0.
                if target_extent>1e-8:
                    raise ValueError("Countdown target is not actually frozen")
                ident=f"person3_acq_s{sid:03d}_seg{segment:03d}_{frag:02d}_{part:02d}"
                filename="episodes/"+ident+".npz"
                np.savez_compressed(out/filename,xy=xy,initial=initial,delta=delta,target=target,radius=radius,
                    available=available.astype(np.int64),valid=np.ones(length,bool),raw=raw,
                    raw_native=raw_native,witness_xy=witness_xy,witness_raw=dense["raw"][a:b]*factor,
                    phase=phase,time_qpc=t,frame_sample_qpc=local["sample_qpc"][selected],
                    frame_submit_qpc=local["submit_qpc"][selected],frame_return_qpc=frame_return,
                    frame_index=ix[selected],frame_active=local["active_frame"][selected].astype(bool),
                    usb_report_begin=usbix[:-1],usb_report_end=usbix[1:])
                role=role_by_sid[sid]
                base=old_rows[sid]
                result.append({"id":ident,"person":"person3","parent":base["parent"],"group":base["group"],
                    "split":"train","family":base["family"],"guide":bool(guide[a]),"scenario_id":sid,
                    "source_to_common":factor,"source_counts_per_pixel":base["source_counts_per_pixel"],
                    "source":"person3_authenticated_capture_with_acquisition","ticks":length,"file":filename,
                    "sha256":sha(out/filename),"start_qpc":origin,"active_start_qpc":int(active_start["qpc"]),
                    "countdown_start_qpc":int(beginning["qpc"]),"end_qpc":int(end["qpc"]),
                    "acquisition_ticks":int(acquisition.sum()),"phase_counts":dict(Counter(map(int,phase))),
                    "replacement_for":base["id"]})
                result[-1]["initial_state_authority"]=initial_authority
                checks.append({"id":ident,"acquisition_target_extent":target_extent,
                    "acquisition_initial_error_counts":float(np.linalg.norm(initial-target[0])),
                    "acquisition_terminal_error_counts":float(np.linalg.norm(xy[np.flatnonzero(acquisition)[-1]]-target[np.flatnonzero(acquisition)[-1]])) if acquisition.any() else None,
                    "acquisition_path_length_counts":float(np.linalg.norm(delta[acquisition],axis=1).sum()),
                    "usb_virtual_minus_witness_rms_counts":float(np.sqrt(np.square(usb_xy-witness_xy).sum(axis=1).mean())),
                    "usb_virtual_minus_witness_max_counts":float(np.linalg.norm(usb_xy-witness_xy,axis=1).max()),
                    "active_frame_during_acquisition_ticks":int(local["active_frame"][selected][acquisition].sum()),
                    "nonactive_frames_after_active_start_ticks":int(np.sum(~local["active_frame"][selected][~acquisition].astype(bool))),
                    "start_receipt_age_us":int(1000-available[0])})
        print(json.dumps({"event":"parent","scenario":sid,"episodes":len(result)}),flush=True) if len(result)%15==0 else None
    if {r["scenario_id"] for r in result} != train_sids or any(r["split"]!="train" for r in result):
        raise ValueError("TRAIN parent coverage/membership failed")
    return result,excluded,checks
