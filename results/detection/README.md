# Detection results

This directory keeps the compact, publishable record behind
[`DETECTION.md`](../../DETECTION.md). The private mouse recordings and large
per-group ledgers are not copied into the repository.

[`renderer_oracle_b80.json`](renderer_oracle_b80.json) is the isolated Renderer
result. It records:

- the exact 44,484-byte native Renderer and its SHA-256;
- the 320-row, ten-session panel using window-5-smoothed human intent at the 0.80 handoff;
- the 256 genuine reports supplied before generation and the AF1.5 lateral safeguard;
- the three descriptor views and labeled AUC diagnostic;
- the known-matching human-distance ruler; and
- the supporting Renderer-only diagnostics.

[`pipeline_b80.json`](pipeline_b80.json) is the complete-pipeline detection result.
It records:

- the same frozen B80 human panel;
- two Planner seeds and two Renderer draws per movement;
- the false-positive-free and broader cold searches; and
- the warm mixture sweep and its predeclared cutoff sensitivity.

The files keep the aggregate counts, exact artifact identities, test boundaries, and
digests of the sealed research record. They do not contain the private source
movements needed to recalculate those exact numbers from a clone.

The matching-human ruler isolates the Renderer, while the published cold and warm
results run the complete Planner→Renderer pipeline. The ruler and population tables
also use different aggregations, so values should be compared within each section
rather than across them.

Warm and cold results answer different questions: warm detection owns trusted
movement from the same session, while cold detection must judge a completely
held-out installation key. The warm case is a best-case laboratory assumption. A
changed sensitivity, mousepad, grip, posture, fatigue level, or habit can make a real
person's old reference stop matching; the compact result does not treat such a
reference as permanently stable.

`manifest.json` authenticates the compact result artifacts by byte count and
SHA-256. Verify them with:

```bash
python -m pip install -e ".[evaluation]"
python -m evaluation verify-results
```

See [`evaluation/README.md`](../../evaluation/README.md) for descriptor-bundle
fields, runnable commands, and the distinction between smoke runs and the full
warm and cold studies.
