# Integrating ABCurves

The Continuous Planner is the default persistent stream. The Static Planner is a
separate fixed-target continuation: it observes A→B, then generates a complete B→C
finish. Choose the one whose state and timing match the application. Both can feed
the same Renderer.

## Installation and first runs

Python 3.11 or later is required. From this repository:

```bash
python -m pip install -e .
python examples/quickstart.py
```

The default Continuous inference uses NumPy, SciPy and Numba. The `static` extra
adds PyTorch. Optional `onnx` inference requires ONNX Runtime. First use may compile
Numba kernels; load and prewarm before starting the live clock. The tested versions
are recorded in `constraints-tested.txt`.

Windows includes the accepted native Renderer. On Linux/macOS, build it locally:

```bash
cmake -S runtime/c -B runtime/c/build
cmake --build runtime/c/build --config Release
ctest --test-dir runtime/c/build -C Release --output-on-failure
```

Then try the rendered examples and Static continuation:

```bash
python examples/streaming.py
python examples/assisted_start.py
python -m pip install -e ".[static]"
python examples/static_quickstart.py
```

`ABCURVES_RENDERER_LIBRARY` can identify an explicit build. Distribution artifacts
include source for native builds; an ESP32 port still needs device-specific build,
USB integration and timing qualification.

## Count space and application coordinates

Native count space is the mouse's two-axis displacement before the application
maps it to a cursor, camera or other control. The Static Planner and Renderer use
this space directly. The Continuous Planner uses common angular counts: the
normalized two-axis frame used in its training recordings.

| Boundary | Quantity | Axes and scale |
| --- | --- | --- |
| Continuous Planner input/output | Absolute target/current positions; displacement history | X right, Y up; common angular counts |
| One common angular count | Angular displacement | `0.0003509487083280618` radians |
| Static Planner | Raw prefix deltas, relative target and radius | Canonical native counts, X right, Y up |
| Renderer input | Smooth displacement per closed 1 ms interval | Canonical native counts, X right, Y up |
| Static Pipeline / direct Renderer output | Signed integer displacement reports | Canonical native counts, X right, Y up |
| Continuous Pipeline output | Signed integer displacement reports | Native counts; `CountTransform.y_down` selects the device Y convention |

For a **constant angular gain** `s` radians per native count, a native delta becomes
`delta*s/common_scale`. `CountTransform` implements this fixed conversion with one
or two positive axis scales and an optional Y inversion. The supplied gain already
includes the application's sensitivity. Static planning and direct rendering need
no angular conversion.

### Mapping a 3D scene or nonlinear control

The application owns the mapping from its screen or world coordinates to the
planner's two-axis frame. A 3D target may require the camera pose and projection;
acceleration, zoom-dependent gain, axis coupling or a warped control surface may
require a state-dependent mapping. A sensitivity number and Y flip describe only
the fixed, axis-aligned case handled by `CountTransform` and `ContinuousPipeline`.

For a custom mapping, use the planner and Renderer separately. Static needs the
observed prefix, target vector and radius expressed consistently in native counts.
Continuous needs its initial position, displacement history and received targets
in a coherent common-angular frame. Map smooth motion into native-count deltas
before rendering, and retain one Renderer stream across consecutive samples:

```python
from abcurves import PortableRendererModel

model = PortableRendererModel("models/renderer_global_h80.bin")
profile = model.prepare_context(canonical_256_raw_reports)
renderer = profile.begin_stream(event_seed=2026)
report = renderer.step(smooth_native_delta)
```

That component interface leaves the environment mapping with your application.
The Continuous planner still carries its own generated position and history;
feeding back a different executed position requires a controller that reconciles
that state. The current API supports initialization from observed motion, as
described below.

The Continuous Planner returns **absolute positions**. The composed pipeline
differences consecutive positions before converting and rendering. The Static
Planner already returns **smooth deltas**. Do not difference those again.

## Continuous stream

```python
import abcurves

movement = abcurves.load(seed=2026, initial_xy=(0.0, 0.0))
movement.update_target((100.0, 30.0), timestamp_us=0)
first = movement.advance(500_000)
movement.update_target((125.0, 45.0), timestamp_us=540_000)
second = movement.advance(1_000_000)
```

`load()` returns `ContinuousPlanner`. A successful `advance(t)` returns a dictionary
with `time_us: int64[N]` and `xy: float64[N,2]`, containing every newly closed 1 ms
endpoint through `t`. Time is an integer microsecond offset from initialization.
The initial position is at time zero; the first output is at 1,000 μs. Fractional
milliseconds carry forward. Omitting a timestamp uses elapsed monotonic time.

Four clocks/concepts remain distinct:

1. **Target receipt:** the earliest time an observation is available. Receipts must
   be nondecreasing and cannot precede already committed output time.
2. **Output sample time:** the closed 1 ms endpoint associated with each position.
3. **Planning cadence:** one committed 32 ms block. A new target does not rewrite
   that block. The next uncommitted plan uses then-available observations.
4. **Computation latency:** wall time spent producing a plan or serving a buffered
   sample. It does not change the sample timestamp or pace an output device.

A receipt between endpoints becomes visible at the next endpoint. Queuing a future
receipt does not expose it early. Give only observations already received in a live
application; stored future trajectory knots are not target input. Before any target,
the stream returns stationary motion.

The selected system combines a sixteen-head ProDMP motor, a learned coherent
selector, learned stop/restart hazards, and finite C2 braking trajectories. Movement,
braking and holding are persistent policy states. A hold can restart when new
evidence calls for movement. Keeping the instance alive preserves selector memory,
hazard state, seed streams and the 640 ms internal history. This is the selected
B2-23 system; the final brake checkpoint alone is not a complete planner.

The public constructor supports `backend="native"` or `"onnx"`,
`kernel_backend="numba"` or `"numpy"`, `prewarm`, `diagnostics` and `skip_unused`.
The default executes the accepted specialized neural arithmetic. ONNX is a
numerical/portability reference. Disabling unused-work skipping is a validation
control, not a different learned policy. See the numerical evidence in
[training and export](TRAINING_AND_INFERENCE.md).

## Optional human-history start

The runtime accepts exactly 160 chronological 1 ms **displacement** samples.
Those samples are known, including genuine zero motion; the older part of the
640 ms model history is unknown. No historical target observations are fabricated.
Targets must still be supplied through timestamped receipts.

The selected tracking representation applies a trailing position filter with
weights `[1,2,3,2,1]/9`, then differences it. `prepare_history` performs the equivalent
stable local calculation from **164 real input deltas**, yielding 160 filtered
deltas and the filtered position anchor:

```python
from abcurves import ContinuousPlanner, prepare_history

start = prepare_history(raw_common_deltas_164, current_xy=observed_common_xy)
movement = ContinuousPlanner(initial_xy=start.observed_xy, history=start.history)
movement.update_target(received_target_xy, timestamp_us=0)
```

All input units are common angular counts with Y up. Convert raw hardware deltas
first. `current_xy` is the observed position **after** the final input interval.
The filter lag equals `(8*d[-1]+6*d[-2]+3*d[-3]+d[-4])/9`; consequently
`start.initial_xy` is usually behind the actual cursor. It is exposed for exact
training-representation experiments. For a physical handoff, use the actual
`start.observed_xy` as above, with the same filtered displacement history.

Alternatively `ContinuousPipeline(initial_xy=start.initial_xy,
history=start.history, observed_xy=start.observed_xy, ...)` retains a constant
physical output offset. This is a separate experimental anchoring choice. It does
not feed the physical position back into the model and is not the recommended
physical initialization.

Qualification covered 576 real-source and 132 synthetic composed streams, all
finite. In the static-target physical-anchor comparison, the median
rendered-minus-intent endpoint gap fell from 16.30 to 1.55 counts, with 44/48 cases
improving relative to retaining the filtered anchor. In tracking comparisons,
correct history improved the initial error in 113/144 paired cases versus zero
history; final error did not improve uniformly. History chiefly helped the early
continuation in this panel.

The runtime assumes generated movement is executed. Human history initializes the
stream; ongoing position correction or human/model blending would also have to
reconcile pending motion, history, selector/brake state and Renderer state.

## Composing integer reports

```python
from abcurves import ContinuousPipeline, CountTransform

stream = ContinuousPipeline(
    genuine_256_hardware_reports,
    transform=CountTransform(recorded_radians_per_count, y_down=True),
    initial_xy=observed_common_xy,
    seed=2026, renderer_seed=101,
)
stream.update_target(target_common_xy, timestamp_us=0)
block = stream.advance(32_000)
```

The result contains `time_us`, ideal absolute `xy`, integer `reports`, and
`rendered_xy` accumulated from those reports under the fixed `CountTransform`.
This calculated position includes integer sampling and residual displacement debt;
it is not an observation of the cursor or 3D application. All samples refer to the
same closed endpoints.

The Renderer profile is exactly 256 chronological genuine integer reports from a
representative device/setup. It need not end at the human handoff. Preparation
creates one immutable native state; each stream clones it and begins once. Do not
restart the Renderer at each plan, brake, hold, or target update: that would reset
its recurrence, packet history and accumulator. Replace a profile between streams
when the physical setup changes materially, not to manufacture sampling variation.

## The static handoff

Install the `static` extra. `StaticPlanner.from_pretrained(model_seed=7)` loads a
planner alone; `StaticPipeline.from_pretrained(model_seed=7)` includes rendering.
Model seeds 7 and 23 select independently trained checkpoints of the same family.
Runtime `seed`/`planner_seed` and `renderer_event_seed_u64` select draws, not models.

```python
from abcurves import StaticPipeline

with StaticPipeline.from_pretrained(model_seed=7) as pipeline:
    profile = pipeline.prepare_renderer_profile(canonical_256_raw_reports)
    reports = pipeline.generate(
        raw_prefix_from_A,
        renderer_profile=profile,
        target_rel_at_B=relative_target_counts,
        target_radius=radius_counts,
        progress_center=center_progress,
        seed=2026,
    )
```

Prefix and geometry use the same canonical native-count frame. The last 160 raw
prefix samples feed the encoder; a known-history mask distinguishes padding from
observed stillness. The last raw delta supplies initial velocity. `begin_at_b()`
can prepare prefix work, then `.finish(...)` binds finalized B geometry. Its
`PreparedStream.step()` yields one report until `complete`; `render_remaining()`
returns the rest. See `examples/static_streaming.py`.

`BTrigger.recommended()` requests **90% edge progress**. With initial target vector
`g`, radius `r`, observed cumulative motion `m` and unit direction `u=g/|g|`:

```text
edge progress   = dot(m,u) / (|g|-r)
center progress = dot(m,u) / |g|
remaining edge distance = |g-m|-r
```

The trigger uses edge progress; the model receives center progress. Ninety percent
is not elapsed time, path length travelled, or 90% center progress. At the first
crossing, the recommended trigger requires at least 24 observed milliseconds and
8 counts outside the target edge. It preserves the established center-progress,
regression, timeout and outside-target conditions. A rejected first crossing is
final; it does not wait for a later one. The training requirement of at least 12 ms
future motion cannot be checked prospectively online.

The paired study used 192 real-source cases across 38 sessions and 33 installation
keys. Among 128 retained cases, 80% admitted 125 and 90% admitted 96. At 90%, 31 cases were rejected for insufficient remaining edge margin,
compared with 2 at 80%; there were 29 additional rejections overall. Only 4/32 short
cases below 80 initial edge counts were eligible at 90%, versus 30/32 at 80%.

Across the 96 paired valid cases, all sixteen heads, both models and two Renderer
draws were checked. Landing rose from 97.27% to 99.22% for seed 7 and from 96.42% to
99.48% for seed 23. The join discontinuity p95 improved, while its median slightly
worsened. Use the 90% recommendation within its eligibility conditions. If no eligible handoff exists,
keep the observed movement or choose an earlier policy before crossing.

`BTrigger()` and `configs/final.json` use the 80% policy from training and the
detection study. `from_seam_contract()` loads that measured contract, while
`BTrigger.recommended()` selects the 90% inference setting.

## State, reset and failure

Each Continuous instance is one mutable stream and is not thread-safe. Independent
instances isolate movement, policy, seeds and Renderer state. Prepared Renderer
profiles can create independent streams. Static model/profile work can be reused,
but an active event stream must have one owner; do not race closure against it.

Continuous `reset()` repeats the most recent seed and initialization. Explicit
`initial_xy`, `history`, or `seed` replace those defaults; explicit zero history
discards a human start. Reset clears target receipts, pending motion, clocks,
policy and failure. The composed pipeline resets both state machines atomically
after validating new initialization. It preserves the immutable texture profile.

A planning failure exposes valid preceding samples through `RuntimeFailure.partial`.
A composition failure uses `ContinuousPipelineFailure.partial`, containing only
successfully rendered output. Further advances are rejected until reset. Already
executed physical output cannot be undone by retrying the same numerical interval.

The model is not an exact point-landing or deadline controller. For example, one
800-count synthetic acquisition took about 7.87 seconds to settle within 5.10
counts. Target speed, sensitivity, history, sampling and starting state affect
behavior. Validate the intended operating range rather than treating a 32 ms
planning interval as a movement-completion promise.

## New trained models and native ports

The [training guide](TRAINING_AND_INFERENCE.md) exports every Continuous component.
Loading a new bundle requires explicit `allow_custom_assets=True`; default loading
checks the accepted release hashes. Newly trained Static checkpoints can be passed
to `StaticPlanner(path)` or `StaticPipeline(planner_checkpoint=path)`.

For a new Renderer, the export command produces a model-bound receipt. Generate a
native binding header and build into a separate directory, then load with
`PortableRendererModel.from_custom(artifact, export_receipt, library=...)`.
`ContinuousPipeline(renderer_model=model, ...)` composes that explicit model.
Default builds retain the accepted model, body and adapter identities. Custom
builds check their own bound artifact and run the same state/integrity tests.

The C99 API is documented in [runtime/c/README.md](../runtime/c/README.md).
Its smooth input is signed Q16 displacement, not position. Retain the model image
and caller-owned state for their full lifetime. The native core allocates no heap;
profile preparation still uses floating-point math. Host benchmark time, firmware
execution time, output scheduling and physical USB delivery are separate quantities.

See [PERFORMANCE.md](PERFORMANCE.md) for measured startup, planning and buffered-output costs.

The [qualification receipts](../results/qualification/README.md) preserve the source
cases, seeds, eligibility and measured outcomes behind these recommendations.
