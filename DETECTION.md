# Trying to catch ABCurves

The part of this project I was most excited to reach was the moment I could turn the
whole problem around. Instead of asking whether an ABCurves movement looked good to
me, I could build judges whose only job was to find the difference.

That matters because a curve can look perfect while repeating a habit no real hand
repeats, or while its millisecond reports are much too clean. A useful detection
study has to inspect both the visible curve and the raw hardware texture, then ask
whether any difference is large and reliable enough to separate generation from the
natural variation of real people.

## What the judges see

No single number describes a movement, so I built three readable views. Each turns a
movement into a set of measurements that can be compared across many human and
generated examples.

| Judge | What it measures | What it is meant to catch |
|---|---|---|
| **Trajectory14** | 14 measurements of duration, speed, deceleration, reversals, jerk, straightness, overshoot, and landing | A curve whose large-scale shape or timing is too clean |
| **Texture19** | 19 measurements of zeros, active runs, packet sizes, sign flips, report spacing, autocorrelation, entropy, and frequency energy | Smooth rounding or decorative jitter that does not behave like mouse hardware |
| **Full49** | Trajectory14 + Texture19 + 16 measurements of the join at B and the target landing | Any combination of shape, texture, continuation, and settling behavior |

Their implementations live in [`abcurves/judges.py`](abcurves/judges.py) and
[`abcurves/texture.py`](abcurves/texture.py).

These judges answer two different questions:

1. **How close is the Renderer to the human movement it is trying to reproduce?**
   Here we know which recording belongs with which output and use real-human
   variation as a ruler.
2. **Can ABCurves be detected when the person is not known beforehand?** Here a
   detector must catch generated movement without falsely accusing a new human.

The first question measures similarity. The second is the practical detection test.

## 1. The Renderer on its own

The Renderer has one job: take a smooth plan and turn it into the uneven integer
reports of a physical mouse. To test that job without mixing in Planner error, I gave
it a smooth version of the real human future. A five-report triangular average,
called **W5**, removes the original packet texture while preserving the intended
path. The Renderer then has to rebuild the zeros, bursts, cadence, and quantization.

For this frozen experiment, before each continuation it also reads the 256 genuine
reports ending at B. That is the only real history it receives; no future raw
reports are passed to it. This describes the measurement protocol, not the newer
recommended schedule of reusing a representative profile prepared before B.

### A human ruler for the result

The table below uses the same distance in every column. Smaller means the two groups
behave more alike.

- **Real vs real, same session** splits genuine movements from one recording into
  two independent groups. Even real data does not match itself perfectly, so this is
  the practical floor.
- **Renderer vs matching human** compares generated output with genuine movements
  from the recording it is recreating.
- **Average different human/setup** shows the much larger scale of ordinary
  differences across unrelated recording sources.

| What the judge sees | Real vs real, same session | Renderer vs matching human | Average different human/setup |
|---|---:|---:|---:|
| Trajectory14 | 0.219 | **0.323** | 0.561 |
| Texture19 | 0.150 | **0.263** | 0.639 |
| Full49 | 0.157 | **0.243** | 0.509 |

Texture19 is the cleanest answer because this experiment isolates the Renderer. Its
distance to the matching human is **0.263**. The same person/setup recorded in
another session is 0.240, the closest different source is 0.280, and the average
different source is 0.639.

At **0.263**, the Renderer's texture sits inside the local human range. It is close to
the same source recorded in another session, slightly closer than the nearest
different source, and its measured distance is **about 59% smaller than the average
different person or setup**. The combined Full49 view tells the same story. This is
why I describe the global Renderer as reproducing texture on the local human scale
rather than merely adding generic noise.

Human movement itself sometimes varies more: in **3 of 35** same-session comparisons,
two real-human samples were farther apart than the Renderer was from the human it
followed. The human and Renderer ranges overlap.

Here, a recording source is the persistent installation identifier used to keep
related sessions together during evaluation. It is a practical person-and-setup
grouping, not proof that one biological identity, mouse, or sensitivity remained
unchanged forever.

## 2. Can it be detected without seeing the person first?

This is the harder question and the result I care about most.

Imagine movement arriving from somebody completely new. The detector does not know
their normal speed, mouse texture, sensitivity, or habits. It has to catch ABCurves
while avoiding the much worse mistake of accusing genuine human movement.

Unlike the Renderer ruler above, the cold and warm study tests what users actually
run: the complete shipped Planner→Renderer pipeline. Each frozen movement is passed
through both Planner seeds, and each plan is rendered twice. That gives four complete
finishes per movement while keeping the human panel and B80 handoff fixed.

The key distinction is what the detector already knows:

| Test | What the detector receives beforehand | The question it answers |
|---|---|---|
| **Cold** | No movement from the person/setup being judged | Can ABCurves be caught for a new source without human false alarms? |
| **Warm** | Trusted clean movement from the exact same recorded session | In an unusually favorable laboratory setting, can the remaining difference be found? |

These are different problems, not two thresholds for the same claim.

### Cold: a completely new person or setup

For each source being judged, the cold test:

1. removes that source and all of its sessions from detector fitting;
2. learns its directions from other sources;
3. sets the accusation boundary using genuine human movement only; and
4. reveals the held human and generated groups after the boundary is frozen.

A group contains 32 movements. The test replaces different amounts of that group
with complete Planner→Renderer movement, from one generated row through a fully
generated group.

| Cold route | Generated movement caught | Genuine human movement flagged |
|---|---:|---:|
| Judges that made no human accusations | **0 / 1,280** trials | **0 / 15,336** trials |
| Broader group search | **6 / 40** fully generated groups | **25 / 15,516** human group checks, across **2 / 6** unseen humans |

**Under this practical cold condition, ABCurves had no reliable detector.** The
judges that made no human accusations caught none of 1,280 generated trials. The
broader search found some fully generated groups, but it also accused genuine human
movement from two of the six unseen humans. It therefore offered no safe way to
attribute new movement to ABCurves. Across these tests, detection did not appear
possible without a trusted matching baseline.

### Warm: the same recorded session is already known

The warm test gives the detector trusted clean movement from the exact same session.
That removes much of the variation between hands, mice, sensitivities, and setups.
The detector first uses a separate human validation group to choose how strong the
evidence must be before raising a flag. The code calls this cutoff **alpha**; a
smaller alpha makes an accusation harder. It selects `alpha = 0.0025` before seeing
any generated outcome. That cutoff flags none of the 40 validation groups, then flags
2 of 40 held real-human groups.

With that unusually strong reference, some generated groups become detectable:

| Generated movement in each 32-row group | Warm groups flagged |
|---:|---:|
| 1 / 32 (3.1%) | 4.61% |
| 8 / 32 (25%) | 13.59% |
| 16 / 32 (50%) | 53.05% |
| 24 / 32 (75%) | 86.17% |
| 32 / 32 (100%) | **90.00% (36 / 40)** |

The important result is simpler: once it receives a clean matching session, the warm
detector can find a strong difference in fully generated groups. That power depends
on already owning a perfectly matched human reference; it is not available to a cold
detector meeting a new person or setup.

The line is thin: detecting 90% of the ABCurves groups came with a 5% human
false-positive rate. Pushing the same detector to catch every ABCurves group raised
the human false-positive rate to 20%. It is a useful laboratory separation, not a
boundary that can be assumed to stay fixed.

Warm detection is close to a best-case laboratory test, not a normal real-world
advantage. Its reference must stay clean and perfectly matched. A real person can
change sensitivity, mousepad, grip, posture, fatigue, or habits; those changes can
move their measurements farther than the small residual difference ABCurves leaves.
Once the baseline moves, a warm detector can lose power or begin flagging genuine
human change. More users and sessions also give rare human outliers more chances to
cross the same line; even if the percentage stayed flat, the number of false
accusations would grow as more people were screened. This study did not pretend to
foresee every such change, which is why the warm result is a theoretical upper bound
on what matching history can reveal.

## What the study answers

| Question | Answer on this panel |
|---|---|
| Does the Renderer reproduce the matching human's packet texture? | Yes. Its Texture19 distance sits inside the local human range, is about 59% below the average distance between unrelated sources, and overlaps real same-session variation. |
| Can the tested detector catch ABCurves for a new person/setup? | Not reliably. The false-positive-free judges caught none of 1,280 generated trials; the broader search caught some output only by also accusing genuine humans from two of six unseen humans. |
| What if a clean copy of the same session is already available? | Under that idealized assumption, the warm detector caught 90% of fully generated groups and flagged 2 of 40 held-human groups. |

My conclusion is therefore simple. **For a new person or setup, ABCurves remained
undetectable by a reliable cold judge.** The strict judges that never accused human
movement caught none of 1,280 generated trials. The only broader search to catch
some complete output also falsely accused humans from two of six unseen humans.
Only the artificial warm condition, which uses clean movement from the exact same
session, exposed a strong signal. Across the people, setups, and thorough tests in
this study, detection did not appear possible without a matching personal baseline.

## Exact boundary of the numbers

The Renderer similarity ruler contains 320 continuations from ten recorded sessions
and two random draws. B is cut at the `0.80` target-edge progress point, called B80.
The observer is reset, shown exactly the 256 genuine reports immediately before B,
and then started once. The Renderer runs with W5 smooth human intent and its AF1.5
lateral safeguard. This deliberately isolates the Renderer and does not include
Planner error.

The cold and warm numbers use the complete shipped pipeline on the same frozen B80
panel. Each movement runs through both Planner seeds, and each plan receives two
Renderer draws. The Renderer still receives the genuine 256-report context ending
at B and uses the same AF1.5 safeguard.

Neither result establishes that an arbitrary rolling history or an observer carried
from session start would produce the same state at every possible B. They also do not
formally evaluate the reusable representative-profile schedule in the 1.5.1 runtime;
the exact per-event windows are retained here so the frozen protocol is not rewritten
after seeing a later one-draw engineering sensitivity probe. That probe had
overlapping uncertainty intervals and is not a promotion or equivalence test; its
separate scope is recorded in the
[`Renderer profile sensitivity receipt`](results/inference/renderer_profile_sensitivity.json).
The values here are descriptive point estimates on this ten-session panel; no
confidence interval is claimed.

The exact artifact identities, full-precision values, aggregation counts, and sealed
receipt digests are in
[`results/detection/renderer_oracle_b80.json`](results/detection/renderer_oracle_b80.json)
and [`results/detection/pipeline_b80.json`](results/detection/pipeline_b80.json).

## Reproducing the public machinery

Install the evaluation dependencies:

```bash
python -m pip install -e ".[evaluation]"
```

A prepared event dataset can build a descriptor bundle when it contains genuine
pre-B Renderer context as `renderer_context_raw_dxdy` with shape `[N,256,2]`:

```bash
python tools/build_descriptor_bundle.py prepared_events.npz \
  results/local/descriptors.npz --rows 256

python -m evaluation floors results/local/descriptors.npz --panel full
python -m evaluation judge results/local/descriptors.npz --panel full
python -m evaluation cold-smoke results/local/descriptors.npz \
  --panels trajectory texture full
```

The bundled `examples/aim_test.npz` does not contain that complete pre-B context. It
can exercise the code with `--assume-quiet-preroll`, but that is a smoke demonstration
rather than a measurement of this panel.

The exact cold and warm algorithms require an enriched audit bundle with the source,
session, generator-cell, chronology, and frozen-panel fields described in
[`evaluation/README.md`](evaluation/README.md):

```bash
python -m evaluation cold results/local/enriched_audit.npz \
  --panels trajectory texture full
python -m evaluation warm results/local/enriched_audit.npz \
  --panels trajectory texture full

python -m evaluation verify-results
```

The private mouse recordings and frozen panel rows are not redistributed. A clone can
verify the compact result, run the same machinery on another correctly structured
corpus, and exercise the bundled smoke path, but it cannot reconstruct these exact
tables from public rows alone.
