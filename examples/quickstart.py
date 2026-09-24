"""Generate independently, then change the target at its actual receipt time."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abcurves

movement = abcurves.load(seed=2026)
movement.update_target((100.0, 30.0), timestamp_us=0)
first = movement.advance(32_000)
movement.update_target((125.0, 45.0), timestamp_us=40_000)
next_part = movement.advance(64_000)
print(next_part['time_us'])       # Newly completed 1 ms endpoints
print(movement.advance(1_000_000)['xy'][-1])        # Absolute position in common angular counts

# Keep this instance alive. Advance sample time as observations arrive; the
# caller owns real-time pacing. Loading once avoids repeated preparation.
