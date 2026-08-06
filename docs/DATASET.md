# Building datasets for ABCurves

The dataset builder starts with complete human movements from **A to C** and makes
the two training sets ABCurves needs.

The input should be true hardware movement data with one raw `dx, dy` count pair
for every millisecond. Each movement also needs its target position and radius, its
outcome, and a stable ID for the source recording. The builder finds usable A→B→C
examples, keeps an audit trail for everything it accepts or rejects, and writes
files the training scripts can open directly.

It deliberately builds **two different datasets**. The Planner learns where and
how a successful movement should finish, so it needs especially clean examples of
human intent. The Renderer learns the packet rhythm of the hardware, so it needs a
broader set of genuine successful movements without duplicating their texture.

## The one command

Install the data tools, then point the builder at either a validated Capture export
tree or a portable event file:

```bash
python -m pip install -e ".[data]"

python tools/prepare_dataset.py \
  <validated-export-root-or-events.npz> prepared \
  --config configs/final_v2.json --branch both
```

The result is ready for training. No second conversion step is needed.

```text
prepared/
  planner_train.npz
  planner_train.manifest.json
  planner_train.rejections.csv
  planner_val.npz
  planner_val.manifest.json
  planner_val.rejections.csv
  renderer_train.npz
  renderer_train.manifest.json
  renderer_train.rejections.csv
  renderer_val.npz
  renderer_val.manifest.json
  renderer_val.rejections.csv
  manifest.json
```

Validation files are written when the input contains people or setups assigned to
validation. Pass the Planner and Renderer NPZ files straight to
[`train_planner.py`](../training/train_planner.py) and
[`train_renderer.py`](../training/train_renderer.py).

## What the builder expects

There are two supported ways to supply movements. Use the validated Capture export
when you still need the project's clock, click, and onset auditing. Use the portable
NPZ when another trusted collector has already done that work.

### Validated ABCurves Capture exports

Point the command at one `abcurves.research_export.v1` directory, or at a directory
tree containing several exports. For every session, the loader finds
`export_manifest.json` and verifies the SHA-256 bindings of `mouse_1ms.csv` and
`trainer_events.csv`.

The existing Capture profiler then handles the details that are easy to get subtly
wrong: fitting the clocks together, slicing dense 1 ms reports, deciding which click
really ended the event, and finding A without looking into the future. Only the
audited A→C interval moves on. Events with capture, clock, onset, coverage,
or ambiguous ending failures are rejected before either model sees them, and the
reason is written to the run manifest.

All movements tied to one collection key stay together. The same person or setup can
therefore never appear in both training and validation by accident. By default, 15%
of the keys are assigned to validation in a repeatable way:

```bash
python tools/prepare_dataset.py exports/ prepared/ \
  --validation-fraction 0.15 \
  --split-seed abcurves.final_v2.user_split
```

Changing the seed changes which keys go to validation, not the filters. When an
export contains only one usable key, the safe output is train-only. The tool warns
that a separate validation key is still needed before training the Planner.

Direct export import uses pandas. That is why this route needs the `data` extra.
Installing `.[all,dev]` also includes it.

### Portable event NPZ

If the events have already been audited elsewhere, use the pickle-free
`abcurves.portable_events.v1` format. It stores variable-length movements in one
concatenated count array and uses offsets to mark their boundaries.

| Array | Shape | What it contains |
| --- | ---: | --- |
| `schema` | scalar | The value `abcurves.portable_events.v1` |
| `dxdy` | `[sum(T), 2]` | Dense raw 1 ms A→C counts |
| `event_offsets` | `[N+1]` | Start and end offsets into `dxdy` |
| `source_trial_id` | `[N]` | Unique ID of each physical movement |
| `target_rel_at_a` | `[N, 2]` | Target centre relative to the cursor at A |
| `target_radius` | `[N]` | Radius in the same raw-count space |
| `outcome` | `[N]` | Terminal result such as `hit_click` |
| `technical_outcome` | `[N]` | Capture/application failure or `none` |
| `user_id` | `[N]` | Optional person/setup key for safe splitting |
| `session_id` | `[N]` | Optional session boundary |
| `split` | `[N]` | Optional declared `train`, `val`, or other split |

The builder preserves a portable file's declared split column. It trusts that A, C,
and the 1 ms clock were already audited, because information missing from the file
cannot be reconstructed later. Code that exports this format can create
`PortableEvent` objects and call `write_portable_events(...)` from
[`abcurves/preprocessing.py`](../abcurves/preprocessing.py).

### Why a raw Capture ZIP is not accepted

A sealed Capture archive carries important clock and file-integrity information that
belongs in the Capture project's native validator. Guessing those details inside a
training script would make a convenient command but an unreliable dataset.

For that reason, a folder containing only raw ZIP files stops with a clear message.
Validate and export the sessions first, then point `prepare_dataset.py` at the
result. From the validated export onward, the command above is the complete path.

## Why the Planner and Renderer get different movements

Both models start from the same audited human recordings, but they need different
views of them:

| Branch | What it should learn | What it receives |
|---|---|---|
| Planner | Clean human intent, geometry, timing, and landing | Several nearby B hand-offs from each accepted movement, with total source weight kept at one |
| Renderer | The real cadence and quantization of hardware packets | One successful movement at the normal 80% hand-off, with no dense duplication or Planner-only shape filtering |

An awkward but successful hand movement may be a poor Planner label and still be
excellent Renderer supervision. Keeping the branches separate protects both signals.

## How A and B are chosen

Both datasets must agree on where the person starts and where ABCurves takes over.
Most importantly, that decision may use only mouse reports that have already arrived.
The code calls this a causal seam.

B progress is measured toward the near edge of the target, while the Planner also
stores progress toward the target centre. Those values sound similar, but they are
not interchangeable.

| Rule | Released value |
| --- | ---: |
| Shortest A -> B prefix | 24 ms |
| Shortest B -> C future | 12 ms |
| Longest future | 1,000 ms |
| Longest A -> B prefix | 1,500 ms |
| Minimum distance remaining | 8 counts |
| Minimum margin outside the target edge | 8 counts |
| Maximum centre progress | 0.92 |
| Progress-regression rejection | 0.18 |

Every prepared file includes the serialized seam contract. The training loader can
therefore reject a train/validation mismatch instead of quietly mixing two
definitions of B. Planner training sees several nearby seams, but validation and
every non-training split always use the normal live 80% hand-off.

## How the Planner branch is cleaned

The filters are easiest to understand as one question: **would this movement teach
a clean, successful way to reach the target?**

The filters have two layers:

- **Quality gates** require a valid terminal outcome, no technical failure, a
  reasonable duration, a settled inner-target endpoint, and enough quiet tail. They
  also bound path inefficiency, radial return, reversal count, and total turning.
- **Shape and behavior gates** remove extreme arc/chord ratios, true backtracking,
  wide detours, repeated lateral or forward-back swings, large sustained turns,
  long stationary pauses, and repeated or distant target exit and re-entry.

One clean correction is not rejected merely because its angle is large. The
count-space path measurements look for repeated bad structure, not every sharp turn.

Some failures reject the whole physical movement rather than only one possible B.
If one hand-off exposes repeated swings, reversals, a long pause, or a large turn
after B, a nearby hand-off cannot sneak the same movement back into training.

The rejection CSV gives every reason a stable name. All numeric limits live in
[`configs/final_v2.json`](../configs/final_v2.json), so this page does not need to
hide the actual rules inside prose.

### Several B cuts, but one movement's worth of weight

For each accepted training movement, the builder requests 21 evenly spaced edge
progress thresholds:

- B78 through B90 when the target-edge distance is below 150 counts;
- B78 through B92 for longer movements;
- exact B80 for validation and other non-training splits.

Several thresholds can land on the same millisecond. When that happens, duplicates
are collapsed to the requested threshold closest to the realised progress.

Every row keeps its original `source_trial_id`. The weights of all versions of that
movement add up to exactly one. During training, the sampler chooses one
available cut from each source per epoch. Dense cuts teach the model to tolerate a
slightly different hand-off without pretending that one recorded movement was
twenty-one independent examples.

### Small-target examples

Small targets are useful but easy to fake badly. The builder only creates a smaller
target when the real endpoint already stays safely inside that proposed radius for
the required final window. The target centre and the recorded movement never move.

This augmentation is Planner-only and train-only. Quotas are filled round-robin
across identities instead of being taken mostly from the largest contributor.
Synthetic rows are tied to trusted real hand-offs, cannot outnumber the original
rows, and share the same one-source total weight.

This gives the Planner more honest small-target labels. It does not erase the
measured result that the smallest targets remain difficult for the released models.

## How the Renderer branch is kept honest

The Renderer receives:

- one causal B80 continuation per physical source;
- only successful endings without technical interruption;
- no dense-B duplication;
- no Planner-only shape filter; and
- no target-shrink augmentation.

This keeps the observed zero runs, packet sizes, cadence, and spectrum in their real
proportions.

The prepared file also contains a smooth window-5 teacher. It is made by smoothing
the complete A→C path first and slicing at B afterwards, so the smoothing state
continues naturally through the seam. The Renderer trainer rebuilds and samples the
frozen window-5 and window-9 views from the raw stream.

## What is saved in each prepared file

Each NPZ has a fast layer for the model and a compact audit layer that leads back to
the original physical movement.

The model-facing arrays include:

- right-aligned `prefix_raw_dxdy [N,160,2]` and `prefix_mask`;
- left-aligned `future_raw_dxdy [N,1000,2]`, `future_smooth_dxdy`, and
  `future_mask`;
- target geometry at B, radius, centre and edge progress;
- boundary speed and acceleration, duration, and outcome;
- source, identity, session, and split fields;
- `event_weight`, the versioned seam JSON, and the exact smoothing record.

The audit layer stores every accepted A→C event once as `dxdy` with
`event_offsets`. Every padded model row maps back through `row_event_index`,
`split_index`, `requested_index`, `variant_id`, `b_threshold`, `synthetic`, and
`borderline`. You can inspect a surprising row without reverse-engineering its
padded tensors.

Each split also has two human-readable companions:

- `*.manifest.json` records counts, settings, hashes, source-weight error, and
  rejection totals;
- `*.rejections.csv` maps excluded physical movements to stable reasons.

The top-level `manifest.json` binds the input, configuration, conversion mode,
outputs, and hashes for the complete preparation run.

## The scale used for the released models

These rules were exercised on the full contributed corpus, not chosen on a toy
example.

The final Planner preparation started with 108,945 profiled events. Of those,
39,311 were eligible and 31,285 physical movements were retained, including 1,919
narrowly accepted borderline movements. Before the final split files were written,
they produced 482,933 real hand-off rows.

The frozen Planner training file contained 361,080 rows from 21,302 physical
sources, 58 collection keys, and 59 sessions. This included 27,754 controlled
small-target rows. The Renderer branch kept 9,858 successful B80 movements, exactly
one per source.

Those numbers are a receipt for the released corpus, not acceptance quotas in the
code. A new dataset should be understood through its own manifests.

## Train the result

The complete recipes and explanations are in
[`Training and running ABCurves`](TRAINING_AND_INFERENCE.md). These are the main
commands for seed 7:

```bash
python training/train_planner.py \
  --train prepared/planner_train.npz \
  --val prepared/planner_val.npz \
  --out runs/planner_seed7.pt \
  --epochs 260 --wta-anneal-epochs 45 \
  --heads 16 --cut-sampling one_per_source_per_epoch \
  --model-selection-interval 0 --prefix-representation raw \
  --seed 7 --device cuda

python training/train_renderer.py \
  --train prepared/renderer_train.npz \
  --out runs/renderer_seed7.pt \
  --epochs 12 --offset-radius 5 \
  --seed 7 --device cuda
```

Seed 23 should be trained as its own complete replication, not mixed with seed 7.
