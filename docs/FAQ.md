# Questions people ask about ABCurves

## What is ABCurves actually doing?

ABCurves generates target-directed human movement. The Continuous Planner can start
independently, follow a changing target, or begin with a correctly prepared human
history. The Static Planner watches a real A→B beginning and generates its B→C
finish toward a fixed target. Both use the same Renderer when integer hardware
reports are wanted.

The two planners solve different problems. A fixed finish can be planned as one
coherent gesture; a continuous stream must keep deciding when to pursue, correct,
pause, brake and restart.

## What exactly comes out?

`ContinuousPlanner` returns absolute positions in common angular coordinates at
1 ms sample endpoints. `ContinuousPipeline` converts their differences to native
count-space intent and returns signed integer `(dx, dy)` reports. `StaticPlanner`
returns a finite smooth continuation; `StaticPipeline` renders that continuation.

The coordinates and clocks are explicit in [INTEGRATION.md](INTEGRATION.md).
ABCurves returns motion to the caller. It does not open a USB device, schedule HID
polls, or own firmware and operating-system queues.

## Why are there two models?

There are two planner families and one shared **Renderer**. A planner chooses
movement intent; the Renderer supplies millisecond packet texture. Geometry and
packet cadence are different-scale problems, and separating them makes both easier
to train and measure.

The default Continuous Planner is a composed learned system: its motor generates
candidate motion, while learned choices, activity transitions and braking balance
purposeful pursuit with varied human movement. The Static Planner instead represents
one finite continuation with ProDMP. The training guide describes every component.

## Why does the Static Planner keep sixteen answers?

The same A→B beginning can have several legitimate human finishes. A single
regressor tends to average them into one safe middle curve. Relaxed winner-takes-all
training lets sixteen ProDMP heads divide those possibilities. Runtime samples one
head uniformly; it does not generate sixteen attempts and choose the nicest one.

## Does it copy a movement from the training data?

No. It does not retrieve and replay a nearby recording. Each planner generates motion from its current inputs and learned state. The Renderer samples each integer report
from the smooth intent and its causal state.

## What makes the Renderer “global”?

It was trained on complete dense sessions rather than only one event phase. Its blind
windows include movement before A, inside events, between events, after C, and idle
periods. The Renderer sees no target, outcome, success, A, B, or C label.

That makes it a general smooth-intent-to-count-texture model within the learned
1 kHz count-space regimes. ABCurves uses it for both continuous streams and B→C plans.

## Does the Renderer just add random jitter?

No. Independent noise does not reproduce real zero runs, packet magnitudes,
spectrum, or the way cadence changes with speed. A width-80 GRU chooses when to emit
and which nearby two-axis integer offset to use around a hysteretic delta-sigma
accumulator.

The accumulator remembers fractional movement instead of discarding it at every
rounding step. That keeps residual displacement debt small and bounded; individual
streams can end with a small residual difference from the smooth path.

## How large is the Renderer?

The float training graph has 20 inputs, hidden width 80, a radius-5 joint offset
head, and 34,362 learned scalars. The selected deployment image is 44,484 bytes:
39,512 bytes for the quantized base plus 4,972 bytes for the rank-16 prefix handoff.

On the validated Windows x64 ABI, the C runtime uses a 208-byte zero-copy model view
and 5,088 bytes for each caller-owned Renderer state. The normal reusable-profile
path retains one prepared state and copies it into one state per active event. Those
structure sizes depend on the target ABI, so ports query them from the library. Its
hot generated-tick path performs 33,760 int8 multiply-accumulates and allocates no
heap.

## Is `RendererProfile` a personalization model?

It is a reusable texture state. The rank-16 handoff compresses one representative
256-report sample into the Renderer's initial recurrent state. Its information
comes from those reports, without a user identity or a person's earlier events.

## Why does the Renderer need exactly 256 reports?

That is the observation length used by the training windows and the validated
artifact. All 256 reports define the packet-regime summary and prepared handoff
state. The float recurrent warm-up itself uses the most recent 128; this does not
make 128 a valid public profile length.

The runtime does not silently slice a longer buffer or pad a shorter one. The caller
prepares a `RendererProfile` from shape `(256, 2)`, containing finite chronological
integer physical counts. The sample should be representative of the intended device
or setup, but it does not need to end at B. The same immutable profile can be reused
across events.

A one-draw Renderer probe found little dependence on millisecond-perfect profile
alignment, with overlapping uncertainty intervals. The
[profile sensitivity receipt](../results/inference/renderer_profile_sensitivity.json)
records that comparison; [DETECTION.md](../DETECTION.md) records the contexts used
in the detection study.

## Should I keep an observer running or refresh the profile on a timer?

Prepare one representative 256-report profile. Static events clone it at each
handoff; a Continuous pipeline initializes from it once and then carries Renderer
state across replanning, braking, holding and restarting. Reinitializing at each
32 ms plan would break that continuity.

If the physical device or setup changes materially, prepare a replacement before
starting a new stream.

## Is there a hard jump where ABCurves takes over?

The Static Planner inherits the movement's position and velocity at B. The Renderer
profile supplies packet-state conditioning as it textures that plan. Together they
anchor the handoff, while individual sampled reports can still vary. The
[handoff comparison](INTEGRATION.md#the-static-handoff) measures the resulting join.

## How is B chosen?

For practical Static continuation, use `BTrigger.recommended()`: its first eligible
crossing is at 90% progress toward the near target edge. The cursor must still be
outside the target with at least 8 counts of edge distance remaining and at least
24 observed milliseconds, along with the established eligibility conditions.
Short movements can have no valid 90% handoff. Do not force a finish in that case.

The real-source comparison improved landing and the upper tail of join error,
while median join error slightly worsened.

Edge progress selects B; center progress is the model input. The original 80%
configuration remains frozen for training validation and the detection study.
[INTEGRATION.md](INTEGRATION.md) gives the formulas and qualification counts.

## Why not train the Renderer only on successful aiming finishes?

That teaches the model the selection rule as well as the texture. It excludes idle
time, ordinary movement, failed attempts, and everything outside the annotated
finish. The global corpus instead preserves the natural proportions of zeros, bursts,
speeds, and regimes across the whole physical session.

## Why use window-3 and window-5 teachers?

Renderer training is self-supervised: a smoothed view is the plan and the original
raw reports are the answer. The two teachers expose the same physical window at
slightly different smoothness without changing its packet target. One is sampled per
presentation.

## What is a Renderer “presentation”?

One source window shown once under one randomly selected teacher. It is the useful
unit of training work because an epoch changes meaning when corpus size changes.

The selected model saw 118,345 presentations: approximately 1.447875 passes over
81,737 training windows, completed in 463 optimizer steps. Copying an epoch count
onto a larger corpus would train for a very different amount of work.

## Why not select the lowest validation loss?

Teacher-forced loss lets the model see the real earlier packet at every step. During
sampling, it must live with its own outputs. In the experiments, held-out
teacher-forced loss could keep improving after sampled carried-session texture became
worse.

Validation loss is therefore diagnostic. Checkpoint promotion is based on sampled
texture with state carried across held-out full sessions.

## Did the larger full corpus beat the pruned corpus?

They tied at the useful precision. The pruned alternative scored `S=1.2723731`;
the full corpus scored `1.2730297`, or `0.052%` higher/worse on this
lower-is-better development score. The full corpus won 2 of 8
model-seed/smoothing/draw cells and 31 of 64 per-user cell comparisons, so the
simpler full-corpus rule
was kept despite the tiny aggregate disadvantage. The score and non-protected
eight-session panel are defined in the public promotion receipt.

This pruning step added complexity without a meaningful gain.

## What does the AF1.5 safeguard do?

It is an always-on soft penalty against rare implausible sideways offset spikes.
Offsets inside a one-count lateral band are unchanged; beyond it each candidate
offset logit loses `1.5 * max(abs(offset dot normal) - 1, 0)`. The smooth plan itself
is untouched.

The release runtime includes AF1.5. Evaluating with that penalty disabled measures a
different sampler.

## What does “like the same human” mean?

It is a measured comparison, not a statement that generated and human movement are
literally identical. The study asks how far generated movement is from its matching
human and compares that distance with variation between real sessions and different
people or setups.

Those matching relationships are useful for a similarity ruler, but an unknown-user
detector is not allowed to know them. The distinction, final results, and
reproducible commands are in the top-level
**[DETECTION.md](../DETECTION.md)**.

## Is ABCurves impossible to detect?

No universal impossibility theorem is claimed. Detection depends on the threat model,
features, population, sample size, and false-positive cost. The repository reports
the scoped tests it actually ran and keeps held-out people isolated.

Read [DETECTION.md](../DETECTION.md) for the results, the warm and cold protocols,
and the exact boundary of each claim.

## Can I train it on my own data?

Yes. Public raw Capture downloads and their validator are the starting point in
[DATASET.md](DATASET.md). Exact selected-model recipes for both planner families and
the Renderer are in [TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md).

For new Static/Renderer experiments, the general preparation tools accept several
input forms:

A validated Capture export tree can prepare both:

```bash
python tools/prepare_dataset.py validated_exports/ prepared/ \
  --config configs/final.json --branch both
```

A portable event NPZ contains Planner labels only:

```bash
python tools/prepare_dataset.py events.npz prepared_planner/ \
  --config configs/final.json --branch planner
```

A portable full-session manifest contains Renderer history only:

```bash
python tools/prepare_dataset.py full_sessions/sessions.json prepared_renderer/ \
  --config configs/final.json --branch renderer
```

An event-only file cannot build the Renderer, because the reports outside A→C are
already gone. See [DATASET.md](DATASET.md) for both schemas.

## Does retraining create the 44,484-byte file?

The float Renderer trainer produces a checkpoint. The complete public export path
then quantizes that base, fits the observed-history handoff, and binds the resulting
image to a separately built native runtime. The [training guide](TRAINING_AND_INFERENCE.md)
includes both exact reconstruction of the selected artifact and a new-model workflow.

For direct float experimentation, use
`StaticPipeline(float_renderer_checkpoint="runs/renderer_p118345.pt")`. That backend
replays its raw 256-report context and samples the continuation at event start.
Its timing is separate from native profile preparation and per-tick rendering.

## Can it run on an ESP32?

The native core is C99, no-heap, fixed-point/int8 in the hot recurrent path, and small
enough to be an ESP32-class starting point. Profile preparation and the rank-16
handoff use float/double and math-library operations. A board port needs its own
firmware integration and timing checks.

## Will it work at another polling rate?

The models use 1 ms bins. Other polling rates need the documented causal binning
contract and checks on the intended device.

## What about screen coordinates, 3D scenes or warped controls?

Your application maps its targets and motion into the planner's two-axis frame.
Static planning and direct rendering use native mouse counts. Continuous uses the
common angular counts from its training representation. Camera projection,
acceleration or a nonlinear control surface belongs in that application mapping.

`CountTransform` and `ContinuousPipeline` cover a fixed angular gain per axis and
an optional Y flip. The planner and Renderer remain separately usable for custom
integrations. See [count space and application coordinates](INTEGRATION.md#count-space-and-application-coordinates).

## What still gives the system trouble?

The Static Planner can struggle with very small targets, and short movements often
have no eligible late handoff. The Continuous Planner can hesitate or settle slowly;
it is not a guaranteed point-landing or deadline controller. A human-history start
can improve the early continuation, but it does not consistently improve every
later tracking metric.

Device timing, USB integration and external position correction need separate
implementation and validation. The current continuous stream assumes its generated
movement is executed.

## Can I use only the Planner or only the Renderer?

Yes. They are separate by design. Preserve the chosen planner's units, 1 ms sample
clock, causal history and target receipt rules, and the Renderer's displacement and
state contracts. [INTEGRATION.md](INTEGRATION.md) shows both boundaries.

## Where should I begin?

Start with the [README](../README.md), open the
[detection study](../DETECTION.md), then run
[`examples/quickstart.py`](../examples/quickstart.py) and read
[`examples/streaming.py`](../examples/streaming.py).
