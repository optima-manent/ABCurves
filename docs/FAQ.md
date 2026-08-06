# Questions people ask about ABCurves

## What is ABCurves actually doing?

A person begins moving toward a target. ABCurves watches A→B and generates only the
finish, B→C. That prefix reveals the current speed, approach, hand, mouse, and whether
the movement is a fast flick or a careful adjustment. A model starting from nothing
would have to guess all of that.

## What exactly comes out?

A variable-length sequence of signed integer `(dx, dy)` mouse counts, one report per
millisecond. It is not a list of screen pixels or a smooth Bézier curve. The output
includes the bursts, gaps, cadence, and quantization of raw hardware movement.

## Why are there two models?

The **Planner** chooses the long-range path, speed, duration, and landing. The
**Renderer** turns that smooth intent into causal integer hardware reports. Geometry
and millisecond packet texture are different-scale problems, and separating them
makes both easier to train and judge.

## Why does the Planner keep sixteen answers?

The same A→B prefix can have several legitimate human finishes. A single regressor
tends to average them into one safe middle curve. Relaxed winner-takes-all training
lets sixteen ProDMP heads divide those possibilities, and runtime samples one head
uniformly. This is not sixteen attempts followed by a best-looking choice.

## Does it copy a movement from the training data?

No. It does not retrieve and replay a nearby recording. The Planner creates a new
finish from the live prefix and target, and the Renderer generates its reports one
millisecond at a time.

## Is there a visible join where the model takes over?

No. The Planner inherits the measured position and velocity at B,
while the Renderer warms its state on the last 128 raw prefix reports. Seam
continuity is also measured inside the Full49 judge.

## How is B chosen?

The live state machine first finds sustained movement toward the target. B fires at
80% of progress toward the near target edge while the cursor remains outside.
Planner training uses a dense neighborhood around that point, but validation and
live inference use the fixed B80 seam. The exact causal rules are in
[`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md).

## Does the Renderer just add random jitter?

No. Random roughness does not reproduce real zero runs, packet magnitudes, spectrum,
or the way texture changes with speed. A recurrent model chooses when to emit and
which small integer offset to use around a hysteretic delta-sigma accumulator. The
accumulator preserves displacement instead of losing fractional motion to rounding.

## What does “like the same human” mean here?

Across 49 shape, texture, target, and seam measurements, ABCurves scores `0.290465`
against the matching human session. Two groups of real movement from that same
session score `0.272369`, the same person or setup in another session scores
`0.295468`, and a different setup averages `0.496151`.

The generated-to-matching-human distance is only **6.64% above the practical
same-session baseline**. It sits at the scale of variation already seen around the
human it is continuing.

## Is ABCurves statistically indistinguishable from humans?

Yes, but that sentence is too broad. A classifier explicitly trained with human/generated
labels reaches Full49 held-out AUC `0.845903`, so small population differences
exist.

The more practical result is different. When all previous movement from the person
or setup being judged was hidden, the tested judges could not turn those differences
into useful detection without human false positives.

## Why is the human-distance result not already a detector?

Because the distance ruler knows which recordings match. It can compare two groups
from one session, two sessions under one collection key, or ABCurves with the human
session it continued. An unknown-person detector receives none of that matching
history. One asks **how close?** and the other asks **can this unfamiliar source be
accused safely?**

## So is detection impossible?

No universal theorem is claimed, but without prior examples from the same person, the practical answer is **yes**.

ABCurves already sits inside natural human variation and near the same-human floor. At that point, there is no clean boundary left for a detector: make it strict enough to catch ABCurves and it begins accusing real humans too.

That is the scoped empirical result. The exact protocol and receipts are in
[`DETECTION.md`](DETECTION.md).

## What is the optional style adapter?

It is a 780-parameter causal adjustment driven by cadence, magnitude, and
high-frequency texture from the previous ten completed **human** events in the same
uninterrupted run. The current event is excluded and generated events are never fed
back. Without enough supported history, its input is exactly zero and the base
Renderer is unchanged.

## Can I train it on my own data?

Yes. The public builder creates separate Planner and Renderer datasets from validated
Capture exports or portable events:

```bash
python tools/prepare_dataset.py \
  <validated-export-root-or-events.npz> prepared \
  --config configs/final_v2.json --branch both
```

See [`DATASET.md`](DATASET.md) for filtering and formats, then
[`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md) for the exact recipes. The
base networks can be retrained publicly. Rebuilding the optional adapter correctly
also needs the full person-by-person chronological corpus, which is not in the small
examples.

## Why is the full dataset not included yet?

Roughly 100 people contributed sessions from their own hands, mice, computers, and
settings. The corpus is valuable, still growing, and worth handling carefully.
Contributors who share a Capture session can request research access through the
[Discord](https://discord.gg/Nyf272vUjz). The intention is to publish it fully later.

## Will it work at another polling rate or in screen pixels?

Not unchanged. ABCurves always runs on one closed 1 ms count bin per step. Other
hardware polling rates must first be accumulated causally into those bins, and a
faithful version should be retrained on data from that hardware. Screen pixels also
introduce scaling and operating-system transforms, so they cannot be mixed with raw
count geometry.

## What still gives it trouble?

Very small targets are the clearest retained weakness. Across both final seed cells,
the scheduled tiny-target slice remained poor at the Planner and rendered stages.

## Can I use only the Planner or only the Renderer?

Yes. They are separate modules on purpose. If you replace one side, preserve the raw
count coordinate system, 1 ms clock, masks, smooth-teacher convention, and B-boundary
contract described in the technical guide.

## Where should I begin?

Start with the [live demo](https://optima-manent.github.io/ABCurves/), then run
[`examples/quickstart.py`](../examples/quickstart.py) and read
[`examples/streaming.py`](../examples/streaming.py).
