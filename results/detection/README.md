# What is in this folder?

These are the small, publishable result files behind the detection write-up. The
private mouse recordings and large group-by-group audit records are not copied into the
repository, so each file keeps the final numbers in a compact form and
`manifest.json` hashes them.

The easiest place to understand the story is
[`docs/DETECTION.md`](../../docs/DETECTION.md). The files here are the receipts:

- `human_distance_floors.json` compares ABCurves with its matching human and with
  ordinary human variation. It is allowed to know which recordings match because it
  is measuring closeness, not trying to detect an unknown person.
- `labeled_judges.json` records how well classifiers separate already-labeled human
  and generated populations.
- `cold_unknown_person.json` contains the result that matters for the real
  false-positive question. The identity being judged and all of its sessions were
  absent from fitting and human-only threshold selection.
- `protocol_parity.json` says exactly which release-code calculations were checked
  against the sealed runners and which private-data rerun was not repeated.
- `warm_known_reference.json` and `warm_power.csv` contain the separate technical
  experiment where trusted same-session human history is available. They are not
  evidence for the unknown-person claim.

The headline result is straightforward. **Among the people held completely out, we
did not find a useful detector that caught the final ABCurves pipeline without also
accusing real human movement.**

Verify every file with:

```bash
python -m pip install -e ".[evaluation]"
python -m evaluation verify-results
```
