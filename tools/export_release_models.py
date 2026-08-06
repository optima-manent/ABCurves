"""Build path-free, versioned ABCurves release checkpoints.

Source training containers can include local custody paths, optimizer state,
and experiment bookkeeping. This tool preserves the exact tensors and the
runtime/training contracts while emitting compact public containers.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


RELEASE_DATE = "2026-08-05"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_contract(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensors: list[dict[str, Any]] = []
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        record = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes": int(array.nbytes),
        }
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
        tensors.append(record)
    return {"sha256": digest.hexdigest(), "tensors": tensors}


def cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(state.items())
    }


def export_planner(source: Path, destination: Path, seed: int) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = cpu_state(payload["model_state_dict"])
    source_hash = sha256_file(source)
    train_summary = dict(payload.get("train_source_trial_summary", {}))
    release = {
        "schema": "abcurves.planner.v2",
        "release_schema": "abcurves.release_planner.v1",
        "release_date": RELEASE_DATE,
        "release_status": "FROZEN",
        "seed": int(seed),
        "model_state_dict": state,
        "planner_config": dict(payload["planner_config"]),
        "heads": int(payload["heads"]),
        "target_dim": int(payload["target_dim"]),
        "summary_dim": int(payload["summary_dim"]),
        "summary_feature_names": list(payload["summary_feature_names"]),
        "summary_mean": payload["summary_mean"].detach().cpu(),
        "summary_std": payload["summary_std"].detach().cpu(),
        "prefix_mean": payload["prefix_mean"].detach().cpu(),
        "prefix_std": payload["prefix_std"].detach().cpu(),
        "y_mean": payload["y_mean"].detach().cpu(),
        "y_std": payload["y_std"].detach().cpu(),
        "horizon": int(payload["horizon"]),
        "hinge_thresholds": list(payload.get("hinge_thresholds", ())),
        "prodmp": dict(payload["prodmp"]),
        "prefix_representation": dict(payload["prefix_representation"]),
        "seam_contract": dict(payload["seam_contract"]),
        "training_contract": {
            "epochs": 260,
            "rwta_heads": 16,
            "rwta_epsilon_start": 0.50,
            "rwta_epsilon_end": 0.05,
            "rwta_anneal_epochs": 45,
            "cut_sampling": "one_per_source_per_epoch",
            "cut_schedule": "source_specific_shuffled_cycle.v1",
            "optimizer_examples_per_epoch": int(
                payload.get("optimizer_examples_per_epoch", 0)
            ),
            "train_event_weight_sum": float(payload.get("train_event_weight_sum", 0.0)),
            "validation_event_weight_sum": float(
                payload.get("val_event_weight_sum", 0.0)
            ),
            "train_source_trials": int(train_summary.get("source_trials", 0)),
            "model_selection": "terminal_epoch_260",
        },
        "tensor_contract": tensor_contract(state),
        "source_container_sha256": source_hash,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, destination)
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "tensor_sha256": release["tensor_contract"]["sha256"],
    }


def export_renderer(source: Path, destination: Path, seed: int) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    original = dict(payload["report"])
    state = cpu_state(payload["state_dict"])
    report = {
        "schema": "abcurves.renderer.v1",
        "release_schema": "abcurves.release_renderer.v1",
        "release_date": RELEASE_DATE,
        "release_status": "FROZEN",
        "arm": "F0",
        "seed": int(seed),
        "config": dict(original["config"]),
        "feature_names": list(original["feature_names"]),
        "n_features": len(original["feature_names"]),
        "architecture": "GRU + hysteretic delta-sigma + joint offset head",
        "terminal_policy": "TP1 threshold-only; no forced final report",
        "training_contract": {
            "epochs": 12,
            "events": int(original.get("n_events", 0)),
            "parameters": int(original.get("n_parameters", 0)),
            "teacher_smoothing": [
                "triangular_moving_average_path:window=5",
                "triangular_moving_average_path:window=9",
            ],
        },
        "tensor_contract": tensor_contract(state),
        "source_container_sha256": sha256_file(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "report": report}, destination)
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "tensor_sha256": report["tensor_contract"]["sha256"],
    }


def export_adapter(source: Path, destination: Path, seed: int) -> dict[str, Any]:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = cpu_state(payload["adapter_state_dict"])
    release = {
        "schema": "abcurves.renderer_adapter.v1",
        "release_date": RELEASE_DATE,
        "release_status": "FROZEN",
        "seed": int(seed),
        "state_dict": state,
        "architecture": {
            "rank": 4,
            "hidden_width": 96,
            "state_width": 3,
            "state_feature_names": ["block_state_C", "block_state_M", "block_state_H"],
            "equation": "h_state = h + U @ (tanh(Wh @ h) * tanh(Ws @ s))",
            "recurrent_feedback": False,
            "exact_zero_bypass": True,
        },
        "state_contract": {
            "history": "ten preceding human events in one contiguous semantic run",
            "current_event_excluded": True,
            "shrinkage": 0.5,
            "clip": 2.5,
            "unsupported_fallback": [0.0, 0.0, 0.0],
        },
        "training_contract": {
            "epochs": 12,
            "parameters": 780,
            "base_renderer_frozen": True,
            "checkpoint_selection": False,
        },
        "tensor_contract": tensor_contract(state),
        "source_container_sha256": sha256_file(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, destination)
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "tensor_sha256": release["tensor_contract"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for seed in (7, 23):
        records = [
            export_planner(
                args.source_dir / f"planner_k1_b80_seed{seed}_epoch260.pt",
                output / f"planner_seed{seed}.pt",
                seed,
            ),
            export_renderer(
                args.source_dir / f"renderer_f0_seed{seed}_epoch12.pt",
                output / f"renderer_seed{seed}.pt",
                seed,
            ),
            export_adapter(
                args.source_dir / f"renderer_causal_c_seed{seed}.pt",
                output / f"renderer_adapter_seed{seed}.pt",
                seed,
            ),
        ]
        for record in records:
            print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
