"""Build the rank-16 handoff fitting data directly from observed raw windows.

Requires the public ABCurves package. No prepared teacher cache, future reports,
research imports or private source directory is needed. Optional source row IDs
preserve the selected adapter's historical deterministic development mask.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from abcurves.renderer import load_count_model, build_teacher_example, _regime_features_from_prefix
from abcurves.smoothing import smooth_dxdy
from .export_renderer import sha256, tensor_digest
from .handoff_features import CONTRACT, begin_input, continuous_observer_features


def canonical_warm_features(raw: np.ndarray) -> np.ndarray:
    output=np.empty((len(raw),128,20),np.float32)
    for i,full in enumerate(raw):
        observed=np.asarray(full[-128:],np.float32)
        smooth=smooth_dxdy(observed,"triangular_moving_average_path:window=5")
        regime=_regime_features_from_prefix(full,window=256)
        output[i]=build_teacher_example(smooth,observed,reset_idx=128,offset_radius=5,regime_features=regime,base_hysteresis=1.0)["features"]
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix",type=Path,required=True,help="float32 [N,256,2] canonical integer-valued counts")
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--select-rows",type=Path,help="optional int64 .npy indices into prefix; preserves their order")
    parser.add_argument("--source-row-ids",type=Path,help="optional int64 IDs used by the frozen 90/10 row mask; defaults to input row indices")
    parser.add_argument("--chunk",type=int,default=512)
    parser.add_argument("--limit",type=int,help="smoke-test rows, not exact full-corpus reproduction")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.chunk<1:raise ValueError("chunk must be positive")
    raw=np.load(args.prefix,mmap_mode="r",allow_pickle=False)
    if raw.shape[1:]!=(256,2) or raw.dtype!=np.float32:raise ValueError("expected float32 [N,256,2]")
    rows=np.arange(len(raw),dtype=np.int64) if args.select_rows is None else np.load(args.select_rows,allow_pickle=False)
    if rows.ndim!=1 or rows.dtype!=np.int64 or not len(rows) or np.min(rows)<0 or np.max(rows)>=len(raw) or len(np.unique(rows))!=len(rows):raise ValueError("invalid row selection")
    ids=rows.copy() if args.source_row_ids is None else np.load(args.source_row_ids,allow_pickle=False)
    if ids.shape!=rows.shape or ids.dtype!=np.int64 or np.min(ids)<0 or len(np.unique(ids))!=len(ids):raise ValueError("source IDs must be unique nonnegative int64 values")
    if args.limit is not None:
        if args.limit<1:raise ValueError("limit must be positive")
        rows,ids=rows[:args.limit],ids[:args.limit]
    model,report=load_count_model(args.model)
    if model.config.hidden!=80 or model.n_features!=20:raise ValueError("expected H80, 20-feature model")
    model=model.to(args.device).eval()
    payload=torch.load(args.model,map_location="cpu",weights_only=True)
    stage=args.output.with_name(args.output.name+f".building-{os.getpid()}")
    stage.mkdir(parents=True,exist_ok=False)
    x=np.lib.format.open_memmap(stage/"adapter_input.npy",mode="w+",dtype=np.float32,shape=(len(rows),145))
    y=np.lib.format.open_memmap(stage/"target_hidden.npy",mode="w+",dtype=np.float32,shape=(len(rows),80))
    started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(rows),args.chunk):
            stop=min(start+args.chunk,len(rows))
            batch=np.asarray(raw[rows[start:stop]],np.float32)
            if not np.isfinite(batch).all() or not np.equal(batch,np.rint(batch)).all() or np.min(batch)<-32768 or np.max(batch)>32767:raise ValueError("prefix is not signed int16 physical counts")
            canonical=canonical_warm_features(batch)
            online=continuous_observer_features(batch)
            hidden=model.gru(torch.from_numpy(online).to(args.device))[1][0].cpu().numpy()
            target=model.gru(torch.from_numpy(canonical).to(args.device))[1][0].cpu().numpy()
            x[start:stop]=begin_input(hidden,canonical[:,-4:,:15],canonical[:,0,15:])
            y[start:stop]=target
            if start == 0 or stop == len(rows) or stop % (16*args.chunk) == 0:
                print(f"states {stop}/{len(rows)} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    x.flush();y.flush();del x,y
    dev=(((ids.astype(np.uint64)*np.uint64(2654435761))&np.uint64(0xFFFFFFFF))%10)==0
    if not dev.any() or dev.all():raise ValueError("need both training and development rows")
    np.save(stage/"dev_mask.npy",dev,allow_pickle=False)
    np.save(stage/"source_rows.npy",ids,allow_pickle=False)
    manifest={"schema":"abcurves.online_handoff_state_cache.v1","contract":CONTRACT,"rows":len(rows),"train_rows":int(np.sum(~dev)),"dev_rows":int(np.sum(dev)),"model_active_tensor_sha256":tensor_digest(payload["state_dict"]),"model_container_sha256":sha256(args.model),"raw_prefix_sha256":sha256(args.prefix),"selection_sha256":None if args.select_rows is None else sha256(args.select_rows),"source_row_ids_sha256":None if args.source_row_ids is None else sha256(args.source_row_ids),"files":{name:{"sha256":sha256(stage/name)} for name in ("adapter_input.npy","target_hidden.npy","dev_mask.npy","source_rows.npy")},"smoke_limit":args.limit,"device":args.device,"torch":torch.__version__,"seconds":time.perf_counter()-started,"known_history_only":True}
    (stage/"manifest.json").write_text(json.dumps(manifest,indent=2)+'\n')
    os.replace(stage,args.output)
    print(json.dumps({"output":str(args.output),"rows":len(rows),"seconds":time.perf_counter()-started},indent=2))


if __name__=="__main__":main()
