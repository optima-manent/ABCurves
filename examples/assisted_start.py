"""Initialize from real human history, retaining the observed physical anchor."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abcurves import ContinuousPipeline, CountTransform, prepare_history

with np.load(ROOT/'examples/data/human_start.npz') as data:
    start = prepare_history(data['raw_common'], current_xy=data['observed_xy'])
    profile, target = data['profile_hardware'], data['target_xy']
    transform = CountTransform(float(data['radians_per_count']), y_down=True)

# The same causal displacement history can start at the actual observed cursor.
# start.initial_xy also exposes the lagged filtered anchor for exact training-
# representation studies; it is not the physical cursor's current position.
stream = ContinuousPipeline(profile, transform=transform,
                            initial_xy=start.observed_xy, history=start.history,
                            seed=2026, renderer_seed=101)
stream.update_target(target, timestamp_us=0)
output = stream.advance(128_000)
print(output['reports'].shape, 'reports after a genuine human prefix')
