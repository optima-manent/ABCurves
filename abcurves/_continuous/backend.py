"""Verified portable assets and fixed batch-one CPU neural execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


FROZEN_FILES = {'brake.onnx': 'ec511415341b72284498ae0333cb12da6bf26e1f9c06562bc468766bfa636c24', 'choice.onnx': '805f517a27219bc20fd0418be02cbf5316a601f040533056fe61304ef5e2ac74', 'choice_split.onnx': '7a8d36ab01e719423854051a6b1a4f35536e5688e2eaf3e9e154f32e5e174bf1', 'events.onnx': 'cd839db370bf0804550cc42e63667736ddd2c2f8fa05cd46453c7254c28dabe3', 'hazard.onnx': 'ebd385e53ca15bb5ec5609018bdd93f2213875a41b3c78912986b00dfda12e53', 'motor.onnx': '116126ef8a6dc7aae027de596158d51e44a06fa30dcfd7058dfcde463106542c', 'weights.npz': '018a096c66213c62b429caf4a8ae6f0470f6e283248a24f476f0fed3e36ea881'}


class Assets:
    def __init__(self, directory, *, allow_custom=False):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf8"))
        if self.manifest.get("schema") != "phalm.b223.assets.v1":
            raise ValueError("Unsupported B2-23 asset manifest")
        required = {"weights.npz", "motor.onnx", "choice.onnx", "events.onnx"}
        if not required.issubset(self.manifest.get("files", {})):
            raise ValueError("Asset manifest omits a required file digest")
        if not allow_custom and self.manifest["reference"]["active_dependencies"]["brake_and_hazards"]["sha256"] != (
            "4c71cc4d29b4a13e613afa88438f772c478fd542c1c9095d23447ecb3c4a3afa"
        ):
            raise ValueError("Bundle is not the frozen B2-23")
        if not allow_custom and self.manifest["files"] != FROZEN_FILES:
            raise ValueError("Asset bundle differs from the selected release; custom training requires explicit opt-in")
        for name, digest in self.manifest["files"].items():
            path = (self.directory / name).resolve()
            if path.parent != self.directory or sha256(path) != digest:
                raise ValueError("Asset digest mismatch: " + name)
        with np.load(self.directory / "weights.npz", allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}
        self.config = self.manifest["reference"]["motor_config"]
        if (self.config["enc"], self.config["decoder"], self.config["history"],
            self.config["forecast"], self.config["commit"], self.config["memory"],
            self.config["dynamics"]) != ("gru", "prodmp", 640, 128, 32, False, True):
            raise ValueError("Unsupported B2-23 configuration")


class OnnxBackend:
    def __init__(self, assets, *, split_choice=False):
        import onnxruntime as ort
        self.sessions = {}
        names = ["motor", "choice", "events"]
        if "hazard.onnx" in assets.manifest["files"] and "brake.onnx" in assets.manifest["files"]:
            names += ["hazard", "brake"]
        if split_choice and "choice_split.onnx" not in assets.manifest["files"]:
            raise ValueError("This bundle has no split-choice graph; re-export assets")
        for name in names:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # No idle worker pools burning cores between a 32 ms decision cadence.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            filename = "choice_split" if name == "choice" and split_choice else name
            self.sessions[name] = ort.InferenceSession(
                str(assets.directory / (filename + ".onnx")), options,
                providers=["CPUExecutionProvider"])

    def motor(self, coarse, fine, dynamics):
        e, q = self.sessions["motor"].run(None, dict(coarse=coarse, fine=fine, dynamics=dynamics))
        return e, q[0]

    def choice(self, encoded, unary_geometry, pairs, previous_valid):
        return self.sessions["choice"].run(None, {
            "encoded": encoded, "unary_geometry": unary_geometry,
            "pairs": pairs, "previous_valid": np.asarray([[previous_valid]], np.float32)})[0][0]

    def events(self, context):
        duration, coefficients, frequency, hazard = self.sessions["events"].run(None, {"context": context})
        return duration[0], coefficients[0], frequency[0], hazard[0]

    def hazard(self, context):
        if "hazard" not in self.sessions:
            return self.events(context)[3]
        return self.sessions["hazard"].run(None, {"context": context})[0][0]

    def brake(self, context):
        if "brake" not in self.sessions:
            return self.events(context)[:3]
        duration, coefficients, frequency = self.sessions["brake"].run(None, {"context": context})
        return duration[0], coefficients[0], frequency[0]


class NativeBackend:
    """All learned networks with unchanged float32 weights and fixed workspaces.

    Motor uses an accurate Padé activation; smaller networks retain
    float32 exponential SiLU. This backend does not construct ONNX sessions.
    """
    def __init__(self, assets, *, motor_variant="pade9_vector"):
        from .neural_native import NativeMotor, NativeHeads
        self._motor = NativeMotor(assets, motor_variant)
        self._heads = NativeHeads(assets, activation="exact")

    def motor(self, coarse, fine, dynamics):
        return self._motor.motor(coarse, fine, dynamics)

    def choice(self, encoded, unary_geometry, pairs, previous_valid):
        return self._heads.choice(encoded, unary_geometry, pairs, previous_valid)

    def events(self, context):
        return self._heads.events(context)

    def hazard(self, context):
        return self._heads.hazard(context)

    def brake(self, context):
        return self._heads.brake(context)
