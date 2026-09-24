#!/usr/bin/env python3
"""Fit and pack the selected rank-16 observer handoff for an H80 float model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MAGIC = b"OHV1R16\0"
SCHEMA = "abcurves.online_handoff_adapter.v1"
INPUT = 145
HIDDEN = 80
RANK = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Adapter(nn.Module):
    def __init__(self, mean: torch.Tensor, invstd: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("invstd", invstd)
        self.v = nn.Linear(INPUT, RANK)
        self.u = nn.Linear(RANK, HIDDEN)
        nn.init.zeros_(self.u.weight)
        nn.init.zeros_(self.u.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :HIDDEN] + self.u(torch.tanh(self.v((x - self.mean) * self.invstd)))


def quantize_rows(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(np.abs(weight), axis=1)
    scale = np.maximum(maximum / 127.0, np.float32(1.0e-12)).astype(np.float32)
    q = np.clip(np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def packed_predict(model: Adapter, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    # Normalization is stored as f16 in the artifact; validate that exact law.
    state["mean"] = state["mean"].astype(np.float16).astype(np.float32)
    state["invstd"] = state["invstd"].astype(np.float16).astype(np.float32)
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("Adapter parameters cannot be represented finitely in the packed format")
    qv, sv = quantize_rows(state["v.weight"])
    qu, su = quantize_rows(state["u.weight"])
    normalized = (x - state["mean"]) * state["invstd"]
    rank = np.tanh(normalized @ (qv.astype(np.float32) * sv[:, None]).T + state["v.bias"])
    pred = x[:, :HIDDEN] + rank @ (qu.astype(np.float32) * su[:, None]).T + state["u.bias"]
    return pred, {"qv": qv, "sv": sv, "qu": qu, "su": su, **state}


def write_packed(path: Path, arrays: dict[str, np.ndarray]) -> None:
    # Refuse a failed conversion before creating a success-looking artifact.
    for name, dtype in (("mean", "<f2"), ("invstd", "<f2"), ("sv", "<f4"),
                        ("v.bias", "<f4"), ("su", "<f4"), ("u.bias", "<f4")):
        with np.errstate(over='ignore'):
            if not np.isfinite(np.asarray(arrays[name], dtype=dtype)).all():
                raise ValueError("Adapter value overflows packed format: " + name)
    # Explicit little-endian layout; all normalization/bias/scale bytes count.
    with path.open("xb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<4I", 1, INPUT, RANK, HIDDEN))
        for name, dtype in (
            ("mean", "<f2"), ("invstd", "<f2"), ("sv", "<f4"),
            ("v.bias", "<f4"), ("qv", "i1"), ("su", "<f4"),
            ("u.bias", "<f4"), ("qu", "i1"),
        ):
            handle.write(np.asarray(arrays[name], dtype=dtype).tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs, batch and learning rate must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    stage = args.output.with_name(f".{args.output.name}.building-{os.getpid()}")
    stage.mkdir(parents=True)
    random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    torch.use_deterministic_algorithms(True)
    x = np.asarray(np.load(args.cache / "adapter_input.npy", mmap_mode="r"), np.float32)
    y = np.asarray(np.load(args.cache / "target_hidden.npy", mmap_mode="r"), np.float32)
    dev = np.load(args.cache / "dev_mask.npy").astype(bool)
    cache_manifest = json.loads((args.cache / "manifest.json").read_text())
    if not cache_manifest.get("model_active_tensor_sha256"):
        raise ValueError("cache must bind the active float model tensors")
    for name in ("adapter_input.npy", "target_hidden.npy", "dev_mask.npy", "source_rows.npy"):
        if sha256(args.cache / name) != cache_manifest["files"][name]["sha256"]:
            raise ValueError(f"cache file differs from manifest: {name}")
    if x.ndim != 2 or x.shape[1] != INPUT or y.shape != (len(x), HIDDEN) or dev.shape != (len(x),):
        raise ValueError("adapter cache dimensions differ")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not dev.any() or dev.all():
        raise ValueError("adapter cache needs finite values and both fitting splits")
    train = ~dev
    mean = x[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train].std(axis=0, dtype=np.float64).astype(np.float32)
    invstd = np.ones_like(std)
    np.divide(1.0, std, out=invstd, where=std > 1e-5)
    model = Adapter(torch.from_numpy(mean), torch.from_numpy(invstd)).to(args.device)
    dataset = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]))
    loader = DataLoader(dataset, args.batch, shuffle=True, generator=torch.Generator().manual_seed(7), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    started = time.time(); best = float("inf"); best_state = None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            pred = model(torch.from_numpy(x[dev]).to(args.device)).cpu().numpy()
        rmse = float(np.sqrt(np.mean((pred - y[dev]) ** 2)))
        if rmse < best:
            best = rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:03d} dev_rmse={rmse:.9f} best={best:.9f}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        float_pred = model(torch.from_numpy(x[dev]).to(args.device)).cpu().numpy()
    packed_pred, arrays = packed_predict(model, x[dev])
    float_rmse = float(np.sqrt(np.mean((float_pred - y[dev]) ** 2)))
    packed_rmse = float(np.sqrt(np.mean((packed_pred - y[dev]) ** 2)))
    raw_rmse = float(np.sqrt(np.mean((x[dev, :HIDDEN] - y[dev]) ** 2)))
    checkpoint = stage / "adapter_float.pt"
    torch.save({"schema": SCHEMA, "rank": RANK, "state_dict": best_state}, checkpoint)
    packed = stage / "adapter_int8.bin"; write_packed(packed, arrays)
    expected_bytes = 8 + 16 + 2*INPUT*2 + RANK*4 + RANK*4 + RANK*INPUT + HIDDEN*4 + HIDDEN*4 + HIDDEN*RANK
    if packed.stat().st_size != expected_bytes:
        raise RuntimeError((packed.stat().st_size, expected_bytes))
    receipt = {
        "schema": SCHEMA + ".training_receipt", "status": "complete", "seed": 7,
        "model_active_tensor_sha256": cache_manifest["model_active_tensor_sha256"],
        "architecture": {"law": "h_online + U*tanh(V*normalize(x)+bv)+bu", "input": INPUT, "rank": RANK, "output": HIDDEN, "begin_mac": INPUT*RANK + RANK*HIDDEN},
        "metrics": {"raw_dev_hidden_rmse": raw_rmse, "float_dev_hidden_rmse": float_rmse, "packed_dev_hidden_rmse": packed_rmse},
        "artifacts": {"float": {"path": str((args.output / checkpoint.name).resolve()), "bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)}, "packed": {"path": str((args.output / packed.name).resolve()), "bytes": packed.stat().st_size, "sha256": sha256(packed)}},
        "source_cache": {"path": str(args.cache.resolve()), "manifest_sha256": sha256(args.cache / "manifest.json"), "adapter_input_sha256": sha256(args.cache / "adapter_input.npy"), "target_hidden_sha256": sha256(args.cache / "target_hidden.npy")},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seconds": time.time()-started, "torch": torch.__version__, "numpy": np.__version__},
    }
    receipt_path = stage / "receipt.json"; receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    os.replace(stage, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
