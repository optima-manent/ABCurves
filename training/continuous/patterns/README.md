# Human-pattern preparation

This NumPy stage reproduces the selected transition labels, hazard contexts, initial brake targets, and the source metadata consumed by coherent-choice training. Exact extracted source identities are in `provenance.json`.

```powershell
python -m training.continuous.patterns.prepare --static <motor-prepared> --old-index <original225/development.json> --new-index <new282/episodes.json> --output <human-pattern-data>
```

The static root contains the selected `static_parents.json`, `static_sessions.json`, `enrollment.npz`, `tracking_rows.json`, and native caches referenced by relative `cache_path`. The original and later tracking indices specify the selected episode roles, source SHA256 values, and relative files. The later index folder also contains `windows.npz`.

API:

- `source.SourceRoots(static, old_index, new_index)` gives explicit source roots.
- `source.sources(roots)` yields `(row, episode)`; tracking episode is `None` until opened from the row's verified file. Static episodes are generated from the complete native movement and its explicitly synthetic tail.
- `source.runs`, `source.transition_rows`, `features.physical_features`, `features.context`, and `geometry.local_basis` preserve the selected algorithms.
- `prepare.pack` is the exact causal 640 ms history/future-label pack used by selector preparation; its future fields are training labels, not model inputs.

The selected footprint is 1,395 sources: 167 original tracking, 204 new tracking, and 1,024 static movements. The original motor index had 58 Person3 active episodes replaced by acquisition episodes. HP excludes IDs containing `acq_`, so those 58 sources are absent here. Static source selection sorts the strict training enrollment by source-ID hash and takes 1,024. Static 512 ms zero-report tails pass through the same causal W5 operation as authentic motion, and their synthetic label mass is multiplied by 0.15.

Transition labels operate every 32 ms from 160 ms onward, require active/known target support across the seam, and recognize quiet intervals of at least 32 ms with motion norm at most 0.01 common units/ms. Retrospective speed peaks and future duration create labels only. Right-censored pauses create survival exposure, never false restart examples. Role 0 is fit, role 1 grouped training holdout, and role 2 inherited external validation. Test/calibration sources do not enter.

Outputs:

- `transitions/frames.npz`: 239,271 transition rows, 22 physical features, labels, synthetic flags, exact group roles and natural weights.
- `components/decisions.npz`: 239,271 rows with 32-dimensional contexts, first-at-risk stop exposure, restart labels, masks and unchanged natural weights.
- `components/brakes.npz`: 9,757 initial component brake labels with local incoming velocity, fitted finite C1 coefficients and duration, inherited roles/weights and synthetic flags. Final C2 training additionally rebuilds actual 17-position paths and causal acceleration from these source rows; this later stage is separate.
- `components/teacher_metadata.npz`: 63,608 hash-selected cuts, at most 96/source, with exact inclusion-mass restoration.
- `components/teacher_metadata_filtered_v1.npz`: 50,173 rows selected by the motor window filter, with exact per-source mass restoration. Pause support in decisions/brakes remains intact. Coherent-choice preparation consumes this metadata and the source episodes.

Full preparation from fresh public Capture exports and all six authenticated
tracking archives reproduced every output array exactly, including teacher metadata
weights. The complete public workflow is in
[training and export](../../../docs/TRAINING_AND_INFERENCE.md).

The selected HP-T7 policy uses branch T, the MC-A motor and the seed 7 component checkpoint, with no separately learned policy scalars. Its state machine accumulates `-log(1-p)` hazards from Components against sampled exponential budgets. Stop/brake and resume use independent seed-derived random streams. Stable rest requires target/position innovation or error over 10 common units to resume, and brake completion resets the hold anchor and restart budget. B2-23 combines this policy with the C2 brake and coherent selector.
