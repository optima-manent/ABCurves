"""Generate one continuation from a reusable Renderer profile."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abcurves import StaticPipeline as Pipeline

from abcurves.seam import BTrigger, BFire, BReject

with np.load(ROOT / "examples/data/static_event.npz", allow_pickle=False) as data:
    raw = data["raw_dxdy"]
    target_a = data["target_rel_a"]
    radius = float(data["target_radius"])
    renderer_profile_window = data["profile_before_a"]

# A was audited in this recorded example. In live use the causal OnsetDetector
# establishes A, then this trigger observes only the completed bins since A.
trigger = BTrigger.recommended()
trigger.arm(target_a, radius)
for index, delta in enumerate(raw):
    target_now = target_a - raw[:index+1].sum(axis=0, dtype=np.float64)
    event = trigger.push_tick(*delta, target_rel_now=target_now, target_radius_now=radius)
    if isinstance(event, BReject):
        raise RuntimeError(event.reason)
    if isinstance(event, BFire):
        prefix = raw[:index+1]
        target = event.target_rel_at_B
        progress = event.progress_center
        break
else:
    raise RuntimeError("This movement has no eligible handoff")

with Pipeline.from_pretrained() as pipeline:
    profile = pipeline.prepare_renderer_profile(renderer_profile_window)
    continuation = pipeline.generate(
        prefix,
        renderer_profile=profile,
        target_rel_at_B=target,
        target_radius=radius,
        progress_center=progress,
        seed=2026,
    )

print(continuation.shape, continuation.dtype)
print("endpoint counts:", continuation.sum(axis=0).tolist())
