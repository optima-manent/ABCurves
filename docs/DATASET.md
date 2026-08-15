# Building datasets for ABCurves

ABCurves has two models, but they should not receive two copies of the same
event-shaped dataset.

The **Planner** learns a clean human finish from A through B to C. It needs target
geometry, outcomes, and carefully audited movement boundaries. The **global
Renderer** learns the packet law of a physical mouse stream. It needs the complete
dense session, including movement before A, pauses between events, and motion after
C.

That distinction is one of the main lessons in this release:

| Branch | Question | Required source |
| --- | --- | --- |
| Planner | How should this person finish this target-directed movement? | Audited A→C events with target and outcome metadata |
| Renderer | How does smooth movement become raw 1 kHz hardware texture? | Uninterrupted dense 1 ms sessions |

The preparation tool enforces this boundary. It will not pad an event crop and call
it a session, nor infer target labels from a raw session that does not contain them.

## Choose the command that matches your source

Install the optional data dependencies first:

```bash
python -m pip install -e ".[data]"
```

### Validated Capture exports: build both branches

A tree of `abcurves.research_export.v1` directories contains audited events and the
complete dense stream, so it can build both datasets in one run:

```bash
python tools/prepare_dataset.py validated_exports/ prepared/ \
  --config configs/final.json --branch both
```

### Portable events: build the Planner only

If another trusted collector has already audited A, C, target geometry, and the 1 ms
clock, a portable event file can build the Planner branch:

```bash
python tools/prepare_dataset.py events.npz prepared_planner/ \
  --config configs/final.json --branch planner
```

`--branch renderer` or `--branch both` deliberately fails for `events.npz`. Once a
session has been cropped into events, pre-A, inter-event, and post-C reports cannot
be reconstructed.

### Portable full sessions: build the Renderer only

A portable `abcurves.full_sessions.v1` manifest can build the Renderer branch:

```bash
python tools/prepare_dataset.py full_sessions/sessions.json prepared_renderer/ \
  --config configs/final.json --branch renderer
```

This form intentionally has no target or event fields, so it is not valid for the
Planner or `--branch both`.

## What a complete output looks like

For a validated export tree with both train and validation identities, the layout is:

```text
prepared/
  planner_train.npz
  planner_train.manifest.json
  planner_train.rejections.csv
  planner_val.npz
  planner_val.manifest.json
  planner_val.rejections.csv

  renderer_train/
    prefix_raw_dxdy.npy
    future_raw_dxdy.npy
    meta.json
  renderer_val/
    prefix_raw_dxdy.npy
    future_raw_dxdy.npy
    meta.json

  source_index.json
  build_report.json
  manifest.json
```

Planner arrays are compact event-aligned NPZ files. Renderer arrays are separate
`.npy` files so NumPy can memory-map the full corpus during training. The root
`manifest.json` binds the input conversion, configuration hash, branch outputs, and
source counts for the whole preparation run.

The tool refuses to overwrite an existing output directory. It writes a fresh
sibling staging directory and renames it into place only after both requested
branches and the root manifest succeed. A changed configuration, failed run, or
source tree therefore cannot silently mix files from two preparations.

## Source form 1: validated Capture exports

Point the command at one `abcurves.research_export.v1` directory or a tree containing
several of them. Each export must contain:

```text
export_manifest.json
mouse_1ms.csv
trainer_events.csv
```

The loader verifies the manifest-bound SHA-256 for both CSV files. It also requires a
dense period of exactly 1,000,000 ns. A directory of raw Capture ZIP files is not
accepted: clock fitting and archive validation belong in Capture's native validator,
where the original evidence is still available.

From the same validated export, the two branches read different views:

- The Planner profiler uses `trainer_events.csv` and the dense stream to fit clocks,
  resolve clicks, find A causally, slice A→C, and reject ambiguous capture failures.
- The Renderer uses `trainer_events.csv` only to establish the one `user_id` and one
  `session_id` belonging to the export. It then takes every canonical `dx, dy` row
  from `mouse_1ms.csv`. Event intervals, target fields, and outcomes never select or
  crop Renderer data.

This is why a failed click can be irrelevant to the Renderer yet disqualify a
Planner label. Hardware texture is still real even when the person's target outcome
is not useful supervision for intent.

## Source form 2: portable Planner events

The pickle-free `abcurves.portable_events.v1` schema stores variable-length A→C
events in one count array, with offsets marking event boundaries.

| Array | Shape | Meaning |
| --- | ---: | --- |
| `schema` | scalar | `abcurves.portable_events.v1` |
| `dxdy` | `[sum(T), 2]` | Dense raw 1 ms A→C counts |
| `event_offsets` | `[N+1]` | Start and stop offsets into `dxdy` |
| `source_trial_id` | `[N]` | Stable ID of each physical movement |
| `target_rel_at_a` | `[N, 2]` | Target centre relative to the cursor at A |
| `target_radius` | `[N]` | Radius in the same raw-count space |
| `outcome` | `[N]` | Terminal outcome such as `hit_click` |
| `technical_outcome` | `[N]` | Capture/application failure or `none` |
| `user_id` | `[N]` | Optional stable person/setup key |
| `session_id` | `[N]` | Optional session boundary |
| `split` | `[N]` | Optional declared split such as `train` or `val` |

The builder preserves a declared split. It trusts that the event boundaries and 1 ms
clock were audited before export, because information omitted from this portable
schema cannot be recovered later.

Code that produces this form can create `PortableEvent` objects and call
`write_portable_events(...)` from
[`abcurves/preprocessing.py`](../abcurves/preprocessing.py).

## Source form 3: portable full sessions

The Renderer-only portable schema keeps the public contract deliberately small. A
`sessions.json` file looks like this:

```json
{
  "schema": "abcurves.full_sessions.v1",
  "dense_period_ns": 1000000,
  "sessions": [
    {
      "user_id": "person-001",
      "session_id": "session-001",
      "ticks": 123456,
      "dxdy_npy": "arrays/session_001.npy",
      "sha256": "64-lowercase-hex-characters"
    }
  ]
}
```

Each referenced array must be a pickle-free `.npy` file with shape `(ticks, 2)`.
Values must be finite integer physical counts on a dense 1 ms grid. Paths must be
relative and traversal-free, and every array is checked against its SHA-256 before
windowing.

`user_id` is required because Renderer validation is whole-user. `session_id` must
uniquely identify one uninterrupted dense stream. No event, outcome, target, A, B,
or C field participates in this schema.

Use `FullSession` and `write_portable_full_sessions(...)` from
[`abcurves/global_data.py`](../abcurves/global_data.py) when exporting this form.

## How the Planner data is built

The Planner branch asks one question: **would this event teach a clean, successful
way to reach the target?** Its filters are strict because a bad geometric label can
teach the model the wrong finish.

### A and B remain causal

A is estimated from sustained, target-aligned movement using only reports that have
already arrived. The default onset detector uses:

| Rule | Value |
| --- | ---: |
| Quiet/noise window | 24 ms |
| Moving bins needed | 12 |
| Backtrack before confirmed run | 4 ms |
| Speed floor | 0.35 counts/ms |
| Noise threshold | median + 6 × MAD |
| Minimum target alignment cosine | 0.15 |

B is measured toward the near target edge, while `progress_center` is measured
toward the target centre. Those quantities are related but not interchangeable.

| Seam eligibility rule | Value |
| --- | ---: |
| Shortest A→B prefix | 24 ms |
| Shortest B→C future | 12 ms |
| Longest planned future | 1,000 ms |
| Longest A→B prefix | 1,500 ms |
| Minimum distance remaining | 8 counts |
| Minimum outside-edge margin | 8 counts |
| Maximum centre progress | 0.92 |
| Progress-regression rejection | 0.18 |

Every prepared Planner file carries the serialized seam contract. Training rejects a
train/validation mismatch instead of quietly joining two definitions of B.

### Quality and shape filters

The first layer requires a valid success outcome, no technical interruption, a
reasonable duration, a settled inner-target endpoint, and enough quiet tail. It also
bounds path inefficiency, radial return, reversal count, and total turning.

The second layer removes repeated bad structure: extreme detours, true backtracking,
wide lateral swings, repeated axial reversals, large sustained turns, long stationary
pauses, and repeated or distant target exit and re-entry. One clean human correction
is not rejected merely because its angle is large.

Some problems reject the whole physical movement rather than one candidate hand-off.
A nearby cut cannot sneak the same repeated swing or long pause back into training.
All thresholds live in [`configs/final.json`](../configs/final.json), and every
rejection gets a stable reason in the CSV receipt.

### Several hand-offs, one movement's worth of weight

The training builder requests 21 evenly spaced edge-progress thresholds:

- 0.78 through 0.90 for movements whose edge distance is below 150 counts;
- 0.78 through 0.92 for longer movements; and
- the fixed live 0.80 hand-off for validation and other non-training splits.

Several requested thresholds can land on the same millisecond. Those duplicates are
collapsed. Every row keeps its original `source_trial_id`, and all rows from one
physical event share total weight one. During each epoch, the training sampler uses
one available cut per source. Dense hand-offs teach seam tolerance without pretending
one recording is many independent people.

### Controlled small-target rows

Small targets are useful but easy to fabricate dishonestly. A smaller radius is
created only when the unchanged real endpoint already remains safely inside that
radius for the required final window. The target centre and recorded path never
move.

This augmentation is Planner-only and train-only. Quotas are filled across identities
instead of mostly from the largest contributor. Synthetic rows cannot outnumber their
trusted parents and retain the same one-source total weight.

### Planner files and receipts

The model-facing arrays include:

- right-aligned `prefix_raw_dxdy [N,160,2]` and `prefix_mask`;
- left-aligned `future_raw_dxdy [N,1000,2]`, `future_smooth_dxdy`, and
  `future_mask`;
- target geometry at B, radius, centre and edge progress;
- boundary speed and acceleration, duration, and outcome;
- source, user, session, and split fields; and
- `event_weight`, the seam contract, and the smoothing record.

The audit layer stores every accepted A→C event once as `dxdy` plus
`event_offsets`. Each padded row maps back through its event index, split index,
requested index, variant, threshold, and augmentation flags. The companion manifest
records settings, hashes, source-weight error, and rejection totals.

## How the global Renderer data is built

The Renderer branch performs no event selection. It does not ask whether a movement
was successful, clean, target-directed, or even inside an annotated event.

For each full session, it starts at tick zero and makes blind, back-to-back windows:

```text
start +   0 ... start + 255   observed physical context
start + 256 ... start + 1055  raw future target
next start = start + 1056
```

There is no overlap and no special alignment to A, B, C, clicks, targets, or
outcomes. If fewer than 1,056 ticks remain, that final tail is recorded as dropped
and is not padded. A session shorter than 1,056 ticks contributes zero windows but
remains visible in the source index.

### Whole-user splitting

Users are assigned deterministically to train or validation from a stable split
seed. Every session and every window belonging to one user stays on the same side.
This prevents a device/person texture from leaking through another session.

The default validation fraction is 0.15. With fewer than two users, the safe result
is train-only. The release seed is frozen so rebuilding the selected corpus preserves
the original train/validation roles. Changing it changes which users are held out and
therefore creates a different study, not merely another file order.

The rule is explicit: hash the UTF-8 string `split_seed:user_id` with SHA-256, sort
users by that digest, and assign the first bounded `round(0.15 * user_count)` users to
validation. Sessions are then materialized in `(session_id, user_id)` order, with
window starts `0, 1056, 2112, ...`. Those details matter because changing row order
also changes the frozen training shuffle.

```bash
python tools/prepare_dataset.py validated_exports/ prepared/ \
  --branch renderer \
  --validation-fraction 0.15 \
  --renderer-split-seed abcurves.continuous_v1.user_split
```

When `--branch both` reads validated exports, the two branches deliberately keep
their frozen salts: Planner uses `abcurves.final_v2.user_split`, while Renderer
uses `abcurves.continuous_v1.user_split`. Override them independently with
`--planner-split-seed` and `--renderer-split-seed`. The default combined build does
not therefore create one shared whole-system holdout; a composed generalization
study must declare a common joint split explicitly.

### What is saved

Each split directory contains:

| File | Shape or role |
| --- | --- |
| `prefix_raw_dxdy.npy` | float32 `[N, 256, 2]`, exact integer-valued observed reports |
| `future_raw_dxdy.npy` | float32 `[N, 800, 2]`, exact integer-valued future reports |
| `meta.json` | schema, split, counts, user/session IDs, and window start ticks |

In `meta.json`, `users` and `sessions` count sources that actually contributed at
least one window. `source_users` and `source_sessions` count every assigned source,
including a short session with zero windows. The complete zero-window record remains
in `source_index.json`, so nothing disappears merely because it was too short to
train on.

The raw reports remain the self-supervised answer. Smooth window-3 and window-5
teacher views are created by the trainer, with one deterministic random choice per
source in each shuffled pass. For pass `epoch`,
`numpy.random.default_rng([seed, epoch])` draws every w3/w5 choice first and only then
draws the row permutation. Within each training window, all 256 context reports
define the regime and learned handoff state; the float recurrent warm-up uses the
most recent 128. At deployment those same 256-report semantics are prepared once as
a reusable representative profile before B. Reuse changes the input schedule, not
the training window. Teacher offset labels use base hysteresis `1.0`; deployment
later calibrates the sampler to `0.5` without relabeling the corpus. No smooth array
needs to be frozen into the prepared dataset.

At the output root:

- `source_index.json` binds every source session, verified hash, split, tick count,
  window count, and dropped tail;
- `build_report.json` records the exact training window, split, training-budget,
  smoothing, and AF1.5 deployment contract plus hashes of the generated arrays; and
- `manifest.json` binds both preparation branches and the configuration hash.

### Scale of the selected Renderer corpus

The selected full-corpus Renderer preparation produced:

| Split | Windows | Sessions | Users |
| --- | ---: | ---: | ---: |
| Train | **81,737** | **54** | **45** |
| Validation | **10,807** | **8** | **8** |

These are receipts for the selected corpus, not hard-coded quotas. Another dataset
should be understood through its own `source_index.json` and `build_report.json`.

A pruned alternative scored `S=1.2723731`; the full corpus scored `1.2730297`, which
is `0.052%` higher/worse because `S` is lower-is-better. The full corpus won 2 of 8
model-seed/smoothing/draw cells and 31 of 64 per-user cell comparisons, so this was
treated as a practical tie and the simpler no-pruning rule was retained. That is a
scoped comparison on the non-protected eight-session development panel, not a
general promise that every extra sample always helps. The
score definition and panel receipt are in
[`renderer_promotion.json`](../results/inference/renderer_promotion.json).

## Train the prepared data

Planner seed 7:

```bash
python training/train_planner.py \
  --train prepared/planner_train.npz \
  --val prepared/planner_val.npz \
  --out runs/planner_seed7.pt \
  --epochs 260 --wta-anneal-epochs 45 \
  --heads 16 --seed 7 --device cuda
```

Global Renderer:

```bash
python training/train_renderer.py \
  --train prepared/renderer_train \
  --val prepared/renderer_val \
  --out runs/renderer_p118345.pt \
  --presentations 118345 --batch-size 256 \
  --seed 7 --device cuda
```

The Renderer budget is presentations, not epochs. The exact reason and checkpoint
selection rule are explained in
[`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md).

Both training programs write new research checkpoints and refuse to overwrite an
existing file. A float Renderer checkpoint is not the authenticated 44,484-byte
deployment image; quantization and release promotion are separate, verified steps.
