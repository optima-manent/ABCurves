"""Compose the Continuous Planner with a persistent hardware-count Renderer."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abcurves import ContinuousPipeline, CountTransform

# An authentic sample is bundled with its source lineage. In an application,
# provide exactly 256 genuine 1 ms reports representative of its device/setup.
with np.load(ROOT/'examples/data/human_start.npz') as data:
    profile = data['profile_hardware']
    transform = CountTransform(float(data['radians_per_count']), y_down=True)

stream = ContinuousPipeline(profile, transform=transform, seed=2026,
                            renderer_seed=101, initial_xy=(0., 0.))
stream.update_target((100., 30.), timestamp_us=0)
for tick in range(1, 1001):
    if tick == 501:
        stream.update_target((160., -40.), timestamp_us=500_000)
    output = stream.advance(tick*1000)
    dx, dy = output['reports'][0]
    # Hand (int(dx), int(dy)) to your paced output layer here.
print('1,000 reports; rendered position:', stream.rendered_xy)
