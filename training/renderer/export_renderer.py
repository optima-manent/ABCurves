"""Export the H80 global renderer plus its compatible rank-16 handoff.

Frozen mode reconstructs the released bytes from authenticated public tensors.
Candidate mode binds a fitted adapter to its float model for a custom native build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from .fixed_export_math import Params, Strategy, build_blob, make_luts

FROZEN_FLOAT = "696efe3bbcbc7e8991e26058bc9b8195285e5f5cb5e5f8cc5f64fcd30d1ac840"
FROZEN_TENSORS = "d09a4a269be583eac6e123bf6be9226bf8ee8e1c9fa8f51faed243b965187206"
FROZEN_SOURCE = "e9951a9bc25b69bf652cebbca8749badd02b8c0675a5f1ad4cde7f1a8624132a"
FROZEN_ADAPTER = "2a90e36c1b4c4b34b505ab73fcbfc6dfe7129e5cec209bd14e4447074b49ea13"
FROZEN_COMBINED = "8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b"
FEATURES = (
    "speed_scaled", "accel", "curvature", "tangent_x", "tangent_y",
    "acc_tangent", "acc_normal", "prev_emit_tangent", "prev_emit_normal",
    "run_active", "run_length_norm", "recent_zero_rate", "log1p_speed",
    "last_nonzero_emit_tangent", "last_nonzero_emit_normal",
    "regime_active_rate", "regime_log1p_active_mag_mean",
    "regime_log1p_active_mag_p95", "regime_sign_flip_rate", "regime_hf_power_log1p",
)
TENSOR_MAP = {
    "w_ih": ("gru.weight_ih_l0", (240,20)),
    "w_hh": ("gru.weight_hh_l0", (240,80)),
    "b_ih": ("gru.bias_ih_l0", (240,)),
    "b_hh": ("gru.bias_hh_l0", (240,)),
    "w_emit": ("head_emit.weight", (1,80)),
    "b_emit": ("head_emit.bias", (1,)),
    "w_off": ("head_off.weight", (121,80)),
    "b_off": ("head_off.bias", (121,)),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor=state[name].detach().cpu().contiguous()
        h.update(name.encode()+b"\0")
        h.update(str(tensor.dtype).encode("ascii")+b"\0")
        h.update(np.asarray(tensor.shape,dtype="<i8").tobytes())
        h.update(tensor.numpy().tobytes(order="C"))
    return h.hexdigest()


def load_params(path: Path):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    report=payload["report"]
    config=dict(report["config"])
    if tuple(report["feature_names"])!=FEATURES:
        raise ValueError("expected the canonical phase-free 20-feature order")
    if config["hidden"]!=80 or config["offset_radius"]!=5:
        raise ValueError("this exporter implements only the selected H80/R5 family")
    state=payload["state_dict"]
    if set(state)!={v[0] for v in TENSOR_MAP.values()}:
        raise ValueError("expected eight active tensors; compatibility graphs are not accepted")
    arrays={}
    for name,(key,shape) in TENSOR_MAP.items():
        value=state[key]
        if value.dtype!=torch.float32 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid active tensor: {key}")
        arrays[name]=np.ascontiguousarray(value.numpy())
    if not config.get("zero_intent_gate"):
        raise ValueError("the native law requires the zero-intent gate")
    for key in ("temperature","offset_magnitude_temperature","offset_direction_temperature"):
        if not np.isfinite(config[key]) or config[key]<=0:
            raise ValueError(f"invalid positive sampling control: {key}")
    params=Params(hidden=80,feature_names=FEATURES,feature_indices=np.arange(20),config=config,**arrays)
    return params,payload,tensor_digest(state)


def validate_adapter(path: Path) -> bytes:
    data=path.read_bytes()
    if len(data)!=4972 or data[:8]!=b"OHV1R16\0" or struct.unpack_from("<4I",data,8)!=(1,145,16,80):
        raise ValueError("invalid OHV1R16 handoff shape or header")
    for offset,count,dtype in ((24,145,"<f2"),(314,145,"<f2"),(604,16,"<f4"),(668,16,"<f4"),(3052,80,"<f4"),(3372,80,"<f4")):
        if not np.isfinite(np.frombuffer(data,dtype=dtype,count=count,offset=offset)).all():
            raise ValueError("adapter contains non-finite constants")
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,required=True)
    parser.add_argument("--adapter-receipt",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mode",choices=("frozen","candidate"),default="candidate")
    args=parser.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix+".json").exists():
        raise FileExistsError("export refuses to overwrite output or receipt")
    params,payload,active_sha=load_params(args.model)
    model_sha=sha256(args.model)
    adapter=validate_adapter(args.adapter)
    adapter_sha=sha256(args.adapter)
    if args.mode=="frozen":
        if (model_sha,active_sha,adapter_sha)!=(FROZEN_FLOAT,FROZEN_TENSORS,FROZEN_ADAPTER):
            raise ValueError("frozen reconstruction requires exact public float and selected adapter")
        source_identity=FROZEN_SOURCE
        embedded=payload["report"]["sanitization"]
        if embedded["source_container_sha256"]!=source_identity or embedded["active_tensor_contract"]["sha256"]!=active_sha:
            raise ValueError("public sanitization identity does not bind the original source")
    else:
        if args.adapter_receipt is None:
            raise ValueError("candidate export requires a model-bound adapter training receipt")
        receipt=json.loads(args.adapter_receipt.read_text())
        if receipt.get("model_active_tensor_sha256")!=active_sha:
            raise ValueError("adapter was not fitted for these active float tensors")
        if receipt["artifacts"]["packed"]["sha256"]!=adapter_sha:
            raise ValueError("adapter binary differs from its training receipt")
        source_identity=model_sha
    base,manifest=build_blob(params,{"checkpoint_sha256":source_identity},Strategy("gru_gate_offrow",2),make_luts())
    combined=base+adapter
    combined_sha=hashlib.sha256(combined).hexdigest()
    if len(combined)!=44484:
        raise RuntimeError("combined H80/R16 byte accounting differs")
    if args.mode=="frozen" and combined_sha!=FROZEN_COMBINED:
        raise RuntimeError("frozen export is not byte-identical to the released renderer")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:stream.write(combined)
    result={"schema":"abcurves.renderer_export.v1","mode":args.mode,"float_checkpoint_sha256":model_sha,"model_active_tensor_sha256":active_sha,"embedded_source_container_sha256":source_identity,"adapter_sha256":adapter_sha,"base_sha256":hashlib.sha256(base).hexdigest(),"combined_sha256":combined_sha,"combined_bytes":len(combined),"base_bytes":len(base),"adapter_bytes":len(adapter),"bit_exact_selected_reconstruction":args.mode=="frozen","base_manifest":manifest,"candidate_note":"A new artifact needs a compatible candidate loader or newly reviewed trust anchors plus numerical and carried-session qualification; this script does not promote it."}
    args.out.with_suffix(args.out.suffix+".json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ["mode","combined_bytes","combined_sha256","bit_exact_selected_reconstruction"]},indent=2))


if __name__=="__main__":main()
