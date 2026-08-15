# Questions people ask about ABCurves

## What is ABCurves actually doing?

A person begins moving toward a target. ABCurves watches A→B and generates only the
finish, B→C. The observed beginning reveals current speed, approach, hand, device,
and packet rhythm. A model starting from nothing would have to invent all of that.

## What exactly comes out?

A variable-length sequence of signed integer `(dx, dy)` mouse counts, one report per
millisecond. It is not a list of screen pixels or a smooth Bézier curve. The output
contains the zeros, bursts, cadence, and quantization of a raw hardware stream.

ABCurves returns those integers to the caller. It does not open a USB device, schedule
HID polls, or own firmware and operating-system queues.

## Why are there two models?

The **Planner** chooses long-range shape, speed, duration, and landing. The **global
Renderer** turns that smooth intent into causal integer reports. Geometry and
millisecond texture are different-scale problems, and separating them makes both
easier to train and measure.

## Why does the Planner keep sixteen answers?

The same A→B beginning can have several legitimate human finishes. A single
regressor tends to average them into one safe middle curve. Relaxed winner-takes-all
training lets sixteen ProDMP heads divide those possibilities. Runtime samples one
head uniformly; it does not generate sixteen attempts and choose the nicest one.

## Does it copy a movement from the training data?

No. It does not retrieve and replay a nearby recording. The Planner creates a new
finish from the observed prefix and target. The Renderer samples each integer report
from the smooth intent and its causal state.

## What makes the Renderer “global”?

It was trained on complete dense sessions rather than only one event phase. Its blind
windows include movement before A, inside events, between events, after C, and idle
periods. The Renderer sees no target, outcome, success, A, B, or C label.

That makes it a general smooth-intent-to-count-texture model within the learned 1 kHz
count-space regimes. ABCurves happens to use it for B→C plans, but the texture law is
not tied to B→C training crops; inputs far outside those regimes are not guaranteed.

## Does the Renderer just add random jitter?

No. Independent noise does not reproduce real zero runs, packet magnitudes,
spectrum, or the way cadence changes with speed. A width-80 GRU chooses when to emit
and which nearby two-axis integer offset to use around a hysteretic delta-sigma
accumulator.

The accumulator remembers fractional movement instead of discarding it at every
rounding step. That keeps residual displacement debt small and bounded. It is not a
promise of exact endpoint equality on every sampled stream.

## How large is the Renderer?

The float training graph has 20 inputs, hidden width 80, a radius-5 joint offset
head, and 34,362 learned scalars. The selected deployment image is 44,484 bytes:
39,512 bytes for the quantized base plus 4,972 bytes for the rank-16 prefix handoff.

On the validated Windows x64 ABI, the C runtime uses a 208-byte zero-copy model view
and 5,088 bytes of caller-owned state per stream. Those structure sizes depend on the
target ABI, so ports query them from the library. Its hot generated-tick path
performs 33,760 int8 multiply-accumulates and allocates no heap.

## Is the rank-16 handoff a personalization model?

No. It compresses the current 256-report observation into the initial recurrent
state. It receives no user identity and does not summarize a named person's earlier
events. It is part of making the exact prefix handoff agree with the selected model.

## Why does the Renderer need exactly 256 reports?

That is the context length used by both training windows and the validated deployment
artifact. All 256 reports define the packet-regime summary and online handoff. The
float recurrent warm-up itself uses the most recent 128; this does not make 128 a
valid public context length.

The runtime does not silently slice a longer buffer or pad a shorter one. The caller
must pass shape `(256, 2)`, containing finite integer physical counts ending at the
same B as the Planner prefix. If the Planner prefix itself is exactly 256 reports, it
can serve both roles.

## Can I keep one observer running forever?

Not under the published contract yet. The validated path is reset, observe exactly
256 reports, then begin one event. A continuously rolling observer, an
arbitrary-length history, or reuse of one prepared context for several events has not
been established as equivalent.

## Is there a hard jump where ABCurves takes over?

The system is built to avoid one: the Planner inherits the measured boundary motion,
and the Renderer observes the exact physical context before generation. Seam
continuity is measured rather than assumed. Individual samples can still vary, so
“no jump is possible” would be a stronger claim than the system makes.

## How is B chosen?

The live state machine first detects sustained movement toward the target. B fires at
80% of progress toward the near target edge while the cursor remains outside and
enough movement remains. Planner training uses nearby handoffs for robustness, while
validation and live inference use the fixed 0.80 rule.

Edge progress explains the trigger; centre progress is the value passed to the
Planner. The exact causal rules are in
[`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md).

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

This does not prove that more data always improves every model. It means this
particular pruning step added complexity without a meaningful gain.

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

Yes, but the branches need different inputs.

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

No. `training/train_renderer.py` writes a float research checkpoint. The released
binary is a separately quantized and authenticated promotion. Replacing it properly
requires quantization validation, C/Python differential tests, new hashes, and a new
manifest—not just renaming the float checkpoint.

You do not need a new binary to use the retrained model. Pass the checkpoint to
`Pipeline(float_renderer_checkpoint="runs/renderer_p118345.pt")`. The same Planner
and 256-report context API then runs the PyTorch float sampler directly. It pre-renders
the continuation at event start, so it is a research/Python backend rather than an
embedded timing claim.

## Can it run on an ESP32?

The native core is C99, no-heap, fixed-point/int8 in the hot recurrent path, and small
enough to be a serious ESP32-class starting point. The observer and rank-16 handoff
still use float/double and math-library operations, however. No ESP32 timing,
firmware integration, or platform certification is claimed in this release.

## Will it work at another polling rate or in screen pixels?

Not unchanged. ABCurves runs on one closed 1 ms raw-count bin per step. Other
polling rates must first be accumulated causally into that grid, and a faithful model
should be tested or retrained on the target hardware. Screen pixels introduce scaling
and operating-system transforms, so they cannot be mixed with raw count geometry.

## What still gives the system trouble?

Very small targets remain a clear Planner weakness. The Renderer is also published
with a deliberately narrow handoff contract: exactly 256 reports and one event per
prepared context. Device-specific performance outside the tested platforms, USB
integration, and arbitrary rolling observation remain work for an implementer to
validate.

## Can I use only the Planner or only the Renderer?

Yes. They are separate by design. If one side is replaced, preserve raw count space,
the 1 ms clock, causal masks, the smooth-intent convention, and the handoff contract
described in the technical guide.

## Where should I begin?

Start with the [README](../README.md), open the
[detection study](../DETECTION.md), then run
[`examples/quickstart.py`](../examples/quickstart.py) and read
[`examples/streaming.py`](../examples/streaming.py).
