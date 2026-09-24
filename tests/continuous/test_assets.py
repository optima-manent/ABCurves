"""Bundle integrity checks and an optional standalone inference import smoke."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from abcurves._continuous.backend import Assets


_REQUIRED = ("weights.npz", "motor.onnx", "choice.onnx", "events.onnx")
_REFERENCE_SHA = "4c71cc4d29b4a13e613afa88438f772c478fd542c1c9095d23447ecb3c4a3afa"


def _manifest(files):
    return {"schema": "phalm.b223.assets.v1", "files": files, "reference": {
        "active_dependencies": {"brake_and_hazards": {"sha256": _REFERENCE_SHA}},
        "motor_config": {"enc": "gru", "decoder": "prodmp", "history": 640,
            "forecast": 128, "commit": 32, "memory": False, "dynamics": True}}}


@pytest.mark.parametrize("missing", _REQUIRED)
def test_manifest_requires_digest_for_every_core_asset(tmp_path, missing):
    files = {name: "0" * 64 for name in _REQUIRED if name != missing}
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(files)), encoding="utf8")
    with pytest.raises(ValueError, match="omits a required file digest"):
        Assets(tmp_path)


@pytest.mark.parametrize("corrupt", ("motor.onnx", "choice.onnx", "events.onnx"))
def test_corrupted_graph_is_rejected_before_backend_loading(tmp_path, corrupt):
    np.savez(tmp_path / "weights.npz", example=np.array([1.0], np.float32))
    for name in _REQUIRED[1:]:
        (tmp_path / name).write_bytes(b"integrity-test-placeholder")
    files = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in _REQUIRED}
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(files)), encoding="utf8")
    # The integrity loader need not parse a graph. Prove the intact digest set is
    # accepted, then that changing a single graph is detected before ORT runs.
    with pytest.raises(ValueError, match="custom training requires explicit opt-in"):
        Assets(tmp_path)
    assert Assets(tmp_path, allow_custom=True).arrays["example"][0] == 1.0
    (tmp_path / corrupt).write_bytes(b"changed-integrity-test-placeholder")
    with pytest.raises(ValueError, match="Asset digest mismatch"):
        Assets(tmp_path, allow_custom=True)


@pytest.mark.parametrize("backend,kernel_backend", [("onnx", "numpy"), ("onnx", "numba"), ("native", "numba")])
def test_portable_inference_does_not_import_torch_or_research(backend, kernel_backend):
    source = """
import sys
import numpy as np
import abcurves._continuous as b223
assert 'torch' not in sys.modules
runtime = b223.load(sys.argv[1], backend=sys.argv[2], kernel_backend=sys.argv[3])
runtime.update_target((160.0, 40.0), timestamp_us=0)
out = runtime.advance(1000)
assert out['xy'].shape == (1, 2)
assert np.isfinite(out['xy']).all()
assert out['time_us'].tolist() == [1000]
assert 'torch' not in sys.modules
assert not any(name.startswith(('research', '_md_delivery', 'coherence_bound', 'distribution_bound')) for name in sys.modules)
print('portable inference without Torch/research: PASS')
"""
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    result = subprocess.run([sys.executable, "-c", source, str(Path(env.get("B223_ASSETS", Path(__file__).resolve().parents[2] / "models/continuous")).resolve()), backend, kernel_backend],
        cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 0, result.stdout + result.stderr
