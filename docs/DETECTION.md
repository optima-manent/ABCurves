# Trying to catch ABCurves

The part of this project I was most excited to reach was the moment I could turn the
whole problem around. Instead of asking whether an ABCurves movement looked good to
me, I could build judges whose only job was to find the difference.

That matters because a curve can look perfect while repeating a habit no real hand
repeats, or while its millisecond packets are much too clean. The judges let me find
those weaknesses, improve the pipeline, and then test it again.

## The two questions

ABCurves first had to produce movement that looks human. It needed believable shape,
speed, correction, landing, and raw hardware texture.

The more interesting goal was to continue the **same human who began the movement**.
A person has already travelled from A to B. ABCurves reads that prefix and generates
only B to C, so it can use the speed, approach, hand, mouse, and packet rhythm that
are already visible.

That leaves two separate measurements:

1. **How close is ABCurves to the human it continued?** Here we are allowed to know
   which human session matches the generated output.
2. **Can it be detected without seeing that person first?** Here the detector must
   not receive any trusted movement from the person or setup it is judging.

The first is a similarity ruler. The second is the false-positive test.

## 1. How close is it to the same human?

The broadest readable comparison uses 49 measurements of movement shape, hardware
texture, target landing, and the join at B. It reports standardized Wasserstein-1
distance, or W1. Smaller means the two groups behave more alike.

| What was compared | Full49 W1 distance |
|---|---:|
| Two groups of real movement from the same session | **0.272369** |
| **ABCurves and the matching human session** | **0.290465** |
| Closest different person or setup | **0.290924** |
| Same person or setup in another session | **0.295468** |
| A different person or setup on average | **0.496151** |

The `0.272369` first row is the practical real-against-real baseline. People do not
repeat a movement perfectly, so even two groups from one genuine session have some
distance between them.

ABCurves measures `0.290465` against the session it continued. That is only **6.64%
above the same-session baseline**. It is slightly closer to its matching session than
another session from that person or setup, a hair closer than the nearest different
setup in this study, and far closer than the average different setup at `0.496151`.

That is the same-human result. It uses known matching information on purpose, so it
does not by itself answer whether an unknown person can be detected.

The full receipt is
[`human_distance_floors.json`](../results/detection/human_distance_floors.json).

## What the judges actually measure

No single statistic can describe a movement, so the release contains several views.
Their implementations live in [`abcurves/judges.py`](../abcurves/judges.py) and
[`abcurves/texture.py`](../abcurves/texture.py).

### Trajectory14

These fourteen descriptors look at the large motion a person would notice first:
duration, landing and lateral error, peak speed and its timing, deceleration,
reversals, jerk, straightness, overshoot, and related path behavior. This panel can
catch a finish that is geometrically too clean even when its packets look plausible.

### Texture19

These nineteen descriptors ignore the pretty curve and inspect the raw 1 kHz integer
stream. They measure zero and active runs, packet magnitudes, sign flips,
inter-report intervals, autocorrelation, entropy, and energy in four frequency bands.
This catches smooth floats, naive rounding, or decorative jitter pretending to be
hardware texture.

### Full49

Full49 combines Trajectory14 and Texture19 with sixteen target-and-seam descriptors.
Those extra measurements check whether B to C joins A to B cleanly, enters and stays
inside the target, lands centrally, and settles instead of crossing the target with
residual motion. It is the widest interpretable judge in the release.

### Raw trajectory CNN

The raw judge reads the sequence itself rather than measurements chosen by hand. A
small temporal network gets a chance to discover patterns the named panels missed.
It is complementary rather than automatically stronger; its final held-out AUC was
lower than Full49's.

### The three reported scores

- **AUC** comes from a classifier trained with human/generated labels. `0.5` is
  guessing and `1.0` is perfect held-out separation.
- **Standardized W1** measures how far each feature distribution must move to overlap
  after scaling on human data. Lower is closer.
- **Sliced W1** repeats that distance along many multivariate projections, allowing
  relationships between features to matter.

These scores answer different questions. A labeled classifier finding a population
difference does not automatically give us a safe accusation threshold for a person
it has never seen.

## 2. Can it be detected without seeing the person first?

This is the harder question and the result I care about most.

Imagine movements arrive from somebody completely new. The detector does not know
their normal speed, mouse texture, or personal habits. It has to catch ABCurves while
avoiding the much worse mistake of accusing genuine human movement.

### How the unknown-person test was kept honest

The exact holdout unit is a persistent ABCurves Capture installation key. I describe
it as a person or setup because that is the intuitive reading, but a key is not proof
that one biological person, mouse, and computer stayed together forever.

For every key being tested, the evaluation did the following:

1. Removed that key and **all of its sessions** from direction fitting.
2. Fitted judge directions using human and final-pipeline rows from other keys.
3. Kept the queried key out of rule and detector selection too.
4. Set the accusation boundary using non-target human groups only.
5. Placed that boundary strictly beyond the largest eligible human score.
6. Froze everything before examining the excluded human and ABCurves groups.

Generated outcomes never chose the threshold. The detector could learn what the
final generator looked like on other keys, but it could not learn what normal looked
like for the key it was about to judge.

The study used 9,858 reference curves from 25 installation keys. The completely
unseen side contained 3,834 genuine curves from ten sessions under six held keys.
Each evaluated group contained 32 movements. The search included dense shifts,
sparse tails, extreme individual curves, concordant anomalies, and maximally
selected subgroups.

### What happened

| Unknown-person result | Outcome |
|---|---:|
| Completely unseen keys | 6 |
| Unseen keys with any false alarm | **1 / 6** |
| Unique human group rosters flagged | 33 / 3,879 |
| Correlated human cell evaluations flagged | 126 / 15,516 |
| Human flags on the smaller frozen panel | 0 / 40 |
| Human curves beyond the non-target-human envelope | 0 / 15,336 |
| ABCurves curves beyond the human-only envelope | 0 / 1,280 |
| Detection power with 1, 2, 4, 8, 12, 16, 24, or 32 generated rows | **0% at every level** |

The 15,516 cell evaluations are overlapping windows crossed with seed and draw
cells. They are not 15,516 independent people or trials, which is why the unique
group and complete-key counts are shown beside them.

The useful frontier was simple:

- Rules strict enough to produce no false alarm on any unseen key had zero power
  against the final pipeline.
- The complete rule still accused one of six unseen keys and also had zero power
  against the final pipeline.

**Across the tested people, setups, and judges, I did not find a useful way to detect
the final ABCurves pipeline without human false positives.** Natural differences
between unfamiliar humans consumed the small remaining generation signal first.

This is strong evidence from the movement data tested here, not a mathematical law
about every future sensor or detector. The machine-readable result is
[`cold_unknown_person.json`](../results/detection/cold_unknown_person.json).

## The labeled diagnostic

There is also an easier experiment. Give a classifier examples already labeled
`human` and `generated`, keep sibling rows from the same physical source together
across folds, and test it on held-back examples from those known populations.

| Judge | Held-out AUC | Standardized W1 | Sliced W1 |
|---|---:|---:|---:|
| Trajectory14 | 0.703060 | 0.174353 | 0.234357 |
| Texture19 | 0.755941 | 0.159562 | 0.133050 |
| Full49 | **0.845903** | 0.181863 | 0.207150 |
| Raw trajectory CNN | 0.640538 | not applicable | not applicable |

Full49 reaching `0.845903` means small, learnable population differences remain when
the judge is shown the answer labels during training. That is useful because it tells
us where future versions can improve, but it is not the same as safely judging a new
person without trusted history.

The exact values are in
[`labeled_judges.json`](../results/detection/labeled_judges.json).

## What changes when the detector already knows the person?

If the detector owns a large, trusted set of clean movements from the same session,
it can learn that session's baseline and look for the small differences left by
generation. This is a separate and easier operating condition, so it is not used for
the unknown-person claim.

| Generated rows in a 32-row group | Generated fraction | Known-history flag rate |
|---:|---:|---:|
| 1 | 3.125% | 1.25% |
| 2 | 6.25% | 3.98% |
| 4 | 12.5% | 10.94% |
| 6 | 18.75% | 17.50% |
| 8 | 25% | 27.89% |
| 12 | 37.5% | 50.55% |
| 16 | 50% | 74.14% |
| 20 | 62.5% | 91.64% |
| 24 | 75% | 97.34% |
| 28 | 87.5% | 99.84% |
| 32 | 100% | 100% |

The frozen human panel produced `0/40` flags. The useful lesson is informational:
once matching clean history exists, evidence accumulates as more generated movement
enters the group. Without that history, variation between unseen humans dominates.

More liberal gates were rejected rather than folded into the headline. Full49 alone
and the directional ensemble each flagged `4/40` frozen human evaluations, while a
dense-mean rule flagged `11/40`.

The complete result is in
[`warm_known_reference.json`](../results/detection/warm_known_reference.json) and
[`warm_power.csv`](../results/detection/warm_power.csv).

## Reproducing the measurements

Install the evaluation dependencies:

```bash
python -m pip install -e ".[evaluation]"
```

The bundled example can build descriptors and exercise the public APIs:

```bash
python tools/build_descriptor_bundle.py examples/aim_test.npz \
  results/local/descriptors.npz --rows 256

python -m evaluation floors results/local/descriptors.npz --panel full
python -m evaluation judge results/local/descriptors.npz --panel full

# Small implementation checks, not reproductions of the release study.
python -m evaluation cold-smoke results/local/descriptors.npz \
  --panels trajectory texture full
python -m evaluation warm-smoke results/local/descriptors.npz \
  --installation-key KEY --session-id SESSION
```

The exact unknown-person and known-history routes need an enriched audit bundle:

```bash
python -m evaluation cold results/local/enriched_audit.npz \
  --panels trajectory texture full
python -m evaluation warm results/local/enriched_audit.npz \
  --panels trajectory texture full

python -m evaluation verify-results
```

The contributed hardware corpus and frozen E260 panel are not redistributed yet.
The release routes can run on another correctly enriched corpus, while the compact
published tables remain tied to their source hashes. The required bundle fields and
exact defaults are documented in
[`evaluation/README.md`](../evaluation/README.md). The public verification boundary
is recorded in
[`protocol_parity.json`](../results/detection/protocol_parity.json).
