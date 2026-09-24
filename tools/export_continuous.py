"""Export all Continuous Planner components; no research checkout is required."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abcurves.continuous_model import Config, Motor as MotorModel, Selector, Components
from continuous_export_graphs import Motor, Choice, ChoiceSplit, Brake


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


FROZEN_CHECKPOINTS = {'motor': '52db2d89495d8ebe76c3d09162a99e32132d6d2e73ed76ce6605cd956c6dc86f', 'selector': '8675c549da677f096b3e40fcf9125a9aedf15ab57e6da94e0bcb055101bdeb1d', 'events': '371d19e08d80d720ad5f83e1052096076ae03e79fbd1b98c3389c5b16bb04d4a'}


def component(path, role):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != "abcurves.continuous_component.v1" or value.get("role") != role:
        raise ValueError(f"Expected a contracted {role} checkpoint")
    state = value.get("state_dict", {})
    if not state or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Component tensors must be finite")
    stage = value.get("metadata", {}).get("recipe", {}).get("stage")
    expected_stage = {"motor": "motor", "selector": "selector", "events": "brake"}[role]
    if sha(path) != FROZEN_CHECKPOINTS[role] and stage != expected_stage:
        raise ValueError(f"Expected the final {expected_stage} stage, not a precursor component")
    if role == "motor" and Config(**value["config"]).to_dict() != Config(commit=32).to_dict():
        raise ValueError("Motor configuration is incompatible with deployment")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motor", type=Path, default=ROOT / "models/continuous/motor.pt")
    p.add_argument("--selector", type=Path, default=ROOT / "models/continuous/selector.pt")
    p.add_argument("--events", type=Path, default=ROOT / "models/continuous/events.pt")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    m, s, e = component(args.motor, "motor"), component(args.selector, "selector"), component(args.events, "events")
    motor = MotorModel(m["config"])
    motor.load_state_dict(m["state_dict"], strict=True)
    selector = Selector(coherent=True)
    selector.load_state_dict(s["state_dict"], strict=True)
    motor.selection = selector
    events = Components()
    events.load_state_dict(e["state_dict"], strict=True)
    if any(not torch.isfinite(v).all() for net in (motor, events) for v in net.state_dict().values()):
        raise ValueError("Component tensor overflow in deployment dtype")
    args.output.mkdir(parents=True, exist_ok=False)
    arrays, learned = {}, {}
    for prefix, model in (("motor", motor), ("events", events)):
        model.eval()
        for name, tensor in model.state_dict().items():
            arrays[prefix + "." + name] = tensor.detach().cpu().numpy().copy()
        for name, tensor in model.named_parameters():
            a = tensor.detach().numpy()
            learned[prefix + "." + name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())
    np.savez(args.output / "weights.npz", **arrays)
    specs = [
        ("motor", Motor(motor), (torch.zeros(1,70,9), torch.zeros(1,54), torch.zeros(1,8)),
         ["coarse", "fine", "dynamics"], ["encoded", "coefficients"]),
        ("choice", Choice(selector), (torch.zeros(1,96),torch.zeros(1,16,16),torch.zeros(1,16,20),torch.ones(1,1)),
         ["encoded", "unary_geometry", "pairs", "previous_valid"], ["logits"]),
        ("choice_split", ChoiceSplit(selector), (torch.zeros(1,96),torch.zeros(1,16,16),torch.zeros(1,16,20),torch.ones(1,1)),
         ["encoded", "unary_geometry", "pairs", "previous_valid"], ["logits"]),
        ("events", events, (torch.zeros(1,32),), ["context"], ["duration", "coefficients", "frequency", "hazard"]),
        ("hazard", events.hazard, (torch.zeros(1,32),), ["context"], ["hazard"]),
        ("brake", Brake(events), (torch.zeros(1,32),), ["context"], ["duration", "coefficients", "frequency"]),
    ]
    import onnx
    with torch.inference_mode():
        for name, net, inputs, input_names, output_names in specs:
            path = args.output / (name + ".onnx")
            torch.onnx.export(net.eval(), inputs, str(path), opset_version=17, dynamo=False,
                              input_names=input_names, output_names=output_names)
            onnx.checker.check_model(str(path))
    reference = {"model_id": "continuous", "family": "C2", "branch": "T",
                 "motor_config": asdict(motor.config), "active_dependencies": {}}
    for key, path, payload in [("motor_and_preprocessing", args.motor, m),
                               ("movement_choice", args.selector, s),
                               ("brake_and_hazards", args.events, e)]:
        reference["active_dependencies"][key] = {"path": path.name,
            "sha256": payload.get("original_sha256", sha(path)), "public_checkpoint_sha256": sha(path)}
    manifest = {"schema": "phalm.b223.assets.v1", "reference": reference,
                "learned_tensors": learned,
                "parameter_count": sum(int(np.prod(v["shape"])) for v in learned.values()),
                "export": {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
                           "script_sha256": sha(__file__), "arithmetic": "float32 neural; float64 physical geometry"},
                "files": {p.name: sha(p) for p in sorted(args.output.iterdir())}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"parameters": manifest["parameter_count"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
