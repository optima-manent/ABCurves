# Running the measurements

If you want the result and the intuition behind it, start with
[`docs/DETECTION.md`](../docs/DETECTION.md). This page is for people who want to run
the measurement code themselves.

The most important rule is simple. A comparison may use the known matching human
when it is asking **how close** two populations are. A detector may not use that
history when it is asking **whether an unknown person's movement can be accused of
being generated**.

The code keeps those two jobs separate.

## Try the public example first

This builds one human/generated descriptor bundle from the included test movements:

```bash
python -m pip install -e ".[evaluation]"

python tools/build_descriptor_bundle.py examples/aim_test.npz \
  results/local/descriptors.npz --rows 256
```

You can then measure human variation and run the labeled judge:

```bash
python -m evaluation floors results/local/descriptors.npz --panel full
python -m evaluation judge results/local/descriptors.npz --panel full
```

`floors` knows which sessions and identities match. That is intentional because it
is measuring distance, not trying to identify an unknown source.

`judge` is given human/generated labels and learns the easiest population-level
difference it can find. Its held-out AUC tells us whether some difference remains,
but it does not tell us whether a real-world detector can use that difference
without false accusations.

The small public bundle can also exercise a reduced version of the unknown-person
code:

```bash
python -m evaluation cold-smoke results/local/descriptors.npz \
  --panels trajectory texture full
```

Its output says `not_release_protocol: true`. It is a useful code check, not a
reproduction of the final study.

## Running the real unknown-person test

The release command is `cold`:

```bash
python -m evaluation cold results/local/enriched_audit.npz \
  --panels trajectory texture full
```

For every identity being judged, the exact runner removes:

- the complete installation key;
- every human session stored under that key; and
- the matching generated rows and generator cell being evaluated.

It learns its directions from other identities. It sets its boundary using human
movement only. Generated outcomes are inspected only after that boundary is fixed.

This is what makes the result a genuine unknown-source false-positive test instead
of a same-person comparison in disguise.

## What is inside a descriptor bundle?

Every command reads a pickle-free `.npz` with schema
`abcurves.descriptor_bundle.v1`.

| Array | Shape | What it contains |
|---|---:|---|
| `features` | `[N,D]` | Finite trajectory, texture and combined descriptors |
| `origin` | `[N]` | `human` or `generated` |
| `installation_key` | `[N]` | The complete identity unit held out by the cold test |
| `session_id` | `[N]` | Recorded session boundary |
| `source_id` | `[N]` | Physical source-trial binding |
| `order` | `[N]` | Event order for chronological groups |
| `task` | `[N]` | Task or challenge stratum |
| `feature_names` | `[D]` | Names of the descriptor columns |
| `panel_slices` | `[P]` | `name:start:stop` declarations for each panel |

The exact release studies need a little more custody information:

| Array | Shape | Why it is needed |
|---|---:|---|
| `population_role` | `[N]` | Separates reference humans from wholly held evaluation rows |
| `generator_cell` | `[N]` | Keeps model seed and draw cells separate during fitting |
| `target_role` | `[N]` | Preserves the task/role comparison stratum |
| `causal_context` | `[N,C]` | Stores information available at generation time, never future outcomes |
| `block_order` | `[N]` | Preserves stable ordering inside chronological groups |
| `audit_panel` | `[N]` | Marks the frozen 32-row evaluation panels |
| `audit_order` | `[N]` | Keeps frozen within-session order; non-panel rows may use `-1` |

Generated rows carry the installation key, session, and source binding of the human
context they continue. Exact cold mode therefore removes the target key's human and
generated rows together. Exact warm mode aligns each candidate cell to the same
frozen physical-source roster.

The exact `cold` and `warm` commands require these fields. They never silently fall
back to the smaller smoke calculation.

## The known-history experiment

There is also a `warm` command. It asks a different and easier technical question:
what can a detector notice when it already owns a large, trusted set of clean human
movement from the same recorded session?

Its primary gate is the maximum over cross-fitted Trajectory14, Texture19 and Full49
directional dense, sparse-tail/Berk-Jones, and subgroup statistics. A disjoint human
validation group selects `alpha=0.01` before the frozen panel or any generated
outcome is evaluated.

```bash
python -m evaluation warm-smoke results/local/descriptors.npz \
  --installation-key KEY --session-id SESSION

python -m evaluation warm results/local/enriched_audit.npz \
  --panels trajectory texture full
```

The smoke command is only a small implementation check and marks its result
`not_release_protocol: true`.

That experiment is useful when studying the judges, but it is **not** used for the
unknown-person claim. It lives here for completeness and for people deliberately
working on detection.

## Exact release defaults

The unknown-person route uses 32-row groups and 16 nested mixture ledgers. The
known-history route uses 48 context neighbours, 512 null-fit draws, 2,048 disjoint
null-calibration draws, 32 nested ledgers, and the predeclared contamination
schedule. These are the defaults behind the compact release results, not settings
inferred from the bundled smoke example.

## What can be reproduced from this repository?

The public repository can:

- verify every compact result file by SHA-256;
- build the same trajectory, texture and Full49 descriptors;
- run and test every split and statistic;
- run the exact algorithms on another correctly enriched corpus; and
- check the inference path and the descriptor/evaluation smoke paths on the bundled
  examples.

```bash
python -m evaluation verify-results
```

The exact published population numbers also need the contributed hardware corpus
and frozen E260 panel. Those rows include 25 reference installation keys, six wholly
held keys across ten sessions and four generator cells. They are not redistributed
yet.

The compact files in [`results/detection/`](../results/detection/) keep the final
numbers and their source hashes auditable without pretending the private movement
rows are already public. The deterministic known-history calculation was checked
against its sealed runner at the level of direction fits, four direction channels,
all twelve dense/sparse/subgroup statistics, nested mixture masks, and empirical
p-values. The precise verification boundary is recorded in
[`protocol_parity.json`](../results/detection/protocol_parity.json).
