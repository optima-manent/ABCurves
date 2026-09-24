# Reproducing the detection study

This guide reproduces the **Static Planner at 80% target-edge progress plus
Renderer**, together with the Renderer-only diagnostic reported in
[DETECTION.md](../DETECTION.md). The recipe fixes the source recordings, handoff,
model variants, Renderer initialization and detector protocols used in the study.

## From public raw sessions to the frozen panel

Follow [DATASET.md](DATASET.md) to download the static collection, validate it and
produce `prepared/capture-static/`. All 35 study source sessions are present in that
collection. The three documented export rejections do not remove a study source.
The compact [recipe](../recipes/detection/recipe.json) binds source identities,
ordered cuts, role assignments, target geometry, head schedules, renderer seeds
and expected array/stream hashes.

```powershell
python -m pip install -e ".[training,evaluation]"
python -m training.detection.prepare_frozen --exports prepared/capture-static --output prepared/detection
```

Preparation verifies the original raw-artifact identities, export files, native
clock and entire count streams before constructing model rows. It creates:

| Population | Rows | Sessions | Recorded installation keys |
| --- | ---: | ---: | ---: |
| Human reference | 9,858 | 25 | 25 |
| Held development | 3,834 | 10 | 6 |
| Frozen generated panel | 320 | 10 | 6 |

Human reference and held development use disjoint installation keys; the generated
panel is selected from held development. These keys preserve grouping;
they are not verified distinct-person counts. The 3,834 development rows are a
different scientific input from the Static training validation set's 3,787 rows.

The seam is the first eligible edge-80 decision, with the measured minimum prefix
24 ms, minimum future 12 ms, minimum remaining edge distance 8 counts, outside-target
condition, maximum A→B length 1,500 ms, maximum center progress 0.92 and recorded
regression policy. B is causal; future-length eligibility is a separate offline
selection condition. Each row stores a 160-bin prefix and at most 1,000 future bins.

Renderer contexts are exactly the 256 genuine pre-B reports after observer reset
at B−256. Matching context has twenty causal target/prefix columns, and the
human descriptor vector has forty-nine columns.

## Regenerate with the measured runtime

The measured implementation is the public commit below. Use a separate checkout so
its original source, model containers and bundled Windows DLL remain identifiable.
Exact stream reproduction also depends on runtime arithmetic, because small
numerical differences can change a sampled integer report.

```powershell
git worktree add --detach ../ABCurves-study 42be8957f4045cba0b34b043795103ea27b81cb2
python -m training.detection.generate_frozen --runtime-root ../ABCurves-study --prepared prepared/detection --output prepared/detection-generated
python -m training.detection.assemble_bundle --kind pipeline --prepared prepared/detection --generated prepared/detection-generated/pipeline.npz --output prepared/detection-bundles/pipeline.npz
python -m training.detection.assemble_bundle --kind oracle --prepared prepared/detection --generated prepared/detection-generated/oracle.npz --output prepared/detection-bundles/oracle.npz
```

Generation authenticates the historical source and artifacts, runs CPU inference
with one Torch thread, and checks every stream against its archived hash. The
measured bundled Windows x64 DLL is 37,376 bytes with SHA256
`730bafac2b6a8431628401fc6e2239a12b787cd730791d453b4dec0cb47d3a41`.
For a port or rebuilt library, compare its generated streams with the same hashes
to check numerical equivalence.

`--rows 0,159` is available for a clearly marked generation smoke check. A subset
cannot be assembled as the complete study. Full pipeline and oracle bundles contain
14,972 and 14,332 rows respectively, including the human populations. The provenance
JSON describes the fresh construction; scientific arrays retain their exact identities.

The oracle uses W5-smoothed **human future** as idealized motion intent. This is a
Renderer diagnostic, not a deployable predictor. Actual raw future reports are
not renderer input, and generated outcomes do not select detector thresholds.

## Run the original statistics

The following commands reproduce the original protocols on those frozen bundles.
General-purpose `build_descriptor_bundle.py` and `build_audit_bundle.py` remain useful
for new experiments, but their fresh row/role selection does not reconstruct this panel.

```powershell
python -m evaluation cold prepared/detection-bundles/pipeline.npz --panels trajectory texture full --bag-rows 32 --ledgers 16 --output study-results/cold.json
python -m evaluation warm prepared/detection-bundles/pipeline.npz --panels trajectory texture full --ledgers 32 --neighbors 48 --null-fit-draws 512 --null-calibration-draws 2048 --output study-results/warm.json
python -m evaluation floors prepared/detection-bundles/oracle.npz --panel texture --minimum-half-rows 8 --output study-results/oracle-texture-ruler.json
python -m training.detection.oracle_headlines --prepared prepared/detection --generated prepared/detection-generated/oracle.npz --output study-results/oracle-headlines.json
```

Repeat `floors` with `trajectory` and `full` for the other ruler views. The headline
helper preserves the original paired 320-row, two-draw protocol: float64 descriptors,
human standard-deviation scaling, 199-quantile W1 approximation, five-fold C2ST,
three repeats, 200 bootstrap draws and seed 7. Estimates are averaged arithmetically
across the two Renderer draws; session-macro values use the ten recorded sessions.
The separate known-matching ruler uses the existing robust-scale implementation
and all 13,692 human rows. These are different aggregations, not interchangeable
invocations of the generic judge defaults.

Use `--verify-inputs-only` on `oracle_headlines` to check all float64 feature matrices
without running a statistical judge. `python -m evaluation verify-results` checks
the published numerical receipts.

## Reconstruction checks

Fresh public exports reproduced all 55 original arrays in each human population,
the complete causal contexts and human descriptors, and all 320 pre-B Renderer
windows exactly. All 1,280 pipeline streams and 640 oracle streams were regenerated through the
authenticated historical checkout and matched their archived hashes exactly. Bundle assembly and the separate
float64 oracle inputs were checked against their sealed array hashes. The judges
and thresholds are recorded in the original study receipts. The commands above
rerun their statistical evaluation on the reconstructed inputs.
