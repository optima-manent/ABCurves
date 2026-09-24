# Tracking preparation

This data adapter reproduces the selected tracking preparations from complete raw ABCurves Capture session archives. It requires Python 3.11 or newer, NumPy, and the public Capture `abct_session_tool` validator.

The source argument selects an explicit historical preparation contract. The supplied frozen metadata binds raw archive SHA256 values, scenario roles, stimulus groups, and sensitivity. It does not assign fresh splits or turn a new capture into reproduction of an existing cohort.

```powershell
python -m pip install numpy
python -m training.tracking.tracking_prepare prepare --archive <session6099.zip> --source 6099d979e7a873da4da7e4c8a7c6d6b7 --validator <abct_session_tool.exe> --output <new6099-output>
python -m training.tracking.tracking_prepare prepare --archive <person1-session.zip> --source person1 --validator <abct_session_tool.exe> --output <person1-output>
python -m training.tracking.tracking_prepare prepare --archive <person2-tracking.zip> --source person2 --validator <abct_session_tool.exe> --output <person2-output>
python -m training.tracking.tracking_prepare prepare --archive <person3-session.zip> --source person3 --validator <abct_session_tool.exe> --output <person3-output>
```

New sources `9324cb2ed9e6368f0725db627d8cb498` and `aa81ab6d6b91af2762dc24e6f3fe81ff` use the same command as 6099 with their exact source IDs and archives. All output directories must be new. A previous verified projection may be supplied with `--reuse-projection <output/projection/session-id-without-s-prefix>`; it is checked against the raw archive identity and stored projection digests before reuse.

The adapter keeps the following contracts separate:

- Original person1/person2: their original active-segment cuts, guide rules, 418ms minimum support, frozen development roles, and once-anchored native virtual cursor. Native capture bins are right-closed; Windows witness validity and causally received target geometry are retained. Target availability at the original episode start uses the recorded historical clamp to zero.
- Person3: original active data plus exactly 58 TRAIN-only acquisition replacements. The replacements include authentic countdown/contact history and its phase evidence. They replace matching active trajectories, retaining all 18 original validation episodes. This source uses the original once-anchored cursor.
- Later new3 sources: authentic acquisition and active segments, 192ms minimum fragments, recorded interruption and guide segmentation, native-quality/coverage masks, and exact 32-nonzero-report USB/WM subsequence anchoring. At least two matching anchors must agree on the coordinate origin. This changes position translation only; deltas and times remain unchanged.

All use canonical X-right/Y-up native counts, one recorded sensitivity conversion into `.0003509487083280618` radians per common unit, and trailing W5 position smoothing. Targets come from the last successful frame whose presentation return was available at that millisecond endpoint. Stored path knots support metadata grouping only; they never supply future target input. USB capture time, worker-observed QPC, witness receipt, and frame sample/submit/return remain separate evidence.

Raw archives are never rewritten. Capture may reject person2's otherwise valid ZIP wrapper (`ZIP central member is not canonical`). In that case the adapter extracts byte-identical inventoried files into a checked temporary directory, verifies every artifact digest, validates the sealed directory with the public tool, and removes only that temporary directory. The original ZIP hash remains the source identity.

Each completed source output contains:

- `projection/<session>/`: checksummed native, witness, and target projections; complete original manifest; verification receipt.
- `prepared/episodes.json` and `prepared/episodes/*.npz`: portable relative episode paths and source/group/role lineage.
- `receipt.json`: source archive and contract hashes, counts, exclusions, elapsed time, and exact dtype/shape/array-byte inventories.
- Person3 additionally retains `original-active` and `acquisition` evidence before replacement. New3 additionally retains unanchored episode and origin-anchor evidence.

Prepare all three new-source outputs before computing the full-cohort training weights:

```powershell
python -m training.tracking.tracking_prepare assemble-new --inputs <new6099-output> <new9324-output> <newaa81-output> --output <new3-cohort>
```

It requires all three exact source identities. It uses the canonical ordinary-window 10% exclusion with protected acquisition/quiet/braking transitions and the original person/family/guide/parent weighting. A missing source is an error.

For independent comparison against an existing selected preparation:

```powershell
python -m training.tracking.tracking_prepare compare --prepared <output/prepared> --reference <selected/episodes.json-or-development.json> --output <comparison.json>
```

Comparison checks exact data type, shape, every C-order array byte, source episode coverage, and role/group/family/guide/tick metadata. `provenance.json` names every copied canonical function, its original relative source, original file hash, and extracted-body hash. The private extraction scripts are outside this portable directory.

The original motor cohort assembles separately:

```powershell
python -m training.tracking.tracking_prepare assemble-original --inputs <person1-output> <person2-output> <person3-output> --output <original-cohort>
```

It preserves the 225-episode roster, including exactly 58 Person3 acquisition replacements. The selected phase-preserving 95% law retains 48,695 windows, including 2,645 acquisition windows. The command checks exact enrollment and weight digests against the frozen selected preparation. Its episode filenames and complete NPZ file hashes match the original unified motor source roster. Unified static/tracking enrollment is the next stage. The HP component preparation omits IDs containing `acq_`, so the 58 Person3 replacements enter the motor's original cohort but not HP transition/component data.

Real-source qualification completed for all six training archives. The three original sources reproduce 225 episodes; all 169 motor-enrolled training NPZ files match their exact source SHA256. The three later sources reproduce 282 episodes; all array dtypes, shapes, and bytes match. Their assembled 49,286-window NPZ file is byte-identical to the selected preparation. The seventh raw tracking source, Person4, is intentionally reserved confirmation and is absent from training.
