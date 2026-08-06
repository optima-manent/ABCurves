# The released models

This folder contains two complete ABCurves model sets.

- **Seed 7** is the default. Use it unless you are deliberately checking whether a
  result repeats under another training initialization.
- **Seed 23** is that independent replication.

Each set contains a Planner, a Renderer, and the small optional style adapter that
belongs to that Renderer.

| Part | Seed 7 | Seed 23 |
| --- | --- | --- |
| Planner | [`planner_seed7.pt`](planner_seed7.pt) | [`planner_seed23.pt`](planner_seed23.pt) |
| Renderer | [`renderer_seed7.pt`](renderer_seed7.pt) | [`renderer_seed23.pt`](renderer_seed23.pt) |
| Style adapter | [`renderer_adapter_seed7.pt`](renderer_adapter_seed7.pt) | [`renderer_adapter_seed23.pt`](renderer_adapter_seed23.pt) |

[`style_scorer.json`](style_scorer.json) is shared. It turns a completed earlier
human event and its known context into the three texture scores used by the
style adapter. It is a frozen transform, not a third model seed.

Do not mix files across the two columns. The adapter is trained for the internal
features of its matching Renderer, and every released measurement treats one column as a
complete unit.

## Loading a model set

```python
from abcurves import Pipeline

with Pipeline.from_pretrained() as pipeline:
    # Complete seed-7 set.
    pass

with Pipeline(model_seed=23) as replication:
    # Complete seed-23 set.
    pass
```

`model_seed` chooses trained weights. It is different from the event `seed` passed
to `Pipeline.generate()`, which chooses one of the sixteen valid Planner heads and
drives Renderer sampling for that movement.

The two model sets are not an ensemble. Normal inference loads seed 7, samples one
Planner head uniformly, and renders that one answer. It does not average both model
sets, generate several candidates, or keep whichever result looks best.

The adapter is loaded with its model set. When there is not enough supported prior
human history, the runtime passes `[0, 0, 0]`. That takes an exact bypass through the
adapter, so the same files also provide the complete generic model.

## What is inside

Both seeds use the same architecture and runtime contract:

| File | Frozen contract |
|---|---|
| Planner | 369,904 parameters; raw 160 ms prefix plus validity; 62 causal summaries; width-96, 3-block TCN; RWTA-16; 20 ProDMP forcing bases plus learned goal; 1,000 ms horizon; epoch 260 |
| Renderer | 80,378 parameters; 21 inputs; width-96 GRU; Bernoulli emit head plus 121 joint offsets; epoch 12; TP1 threshold-only ending |
| Style adapter | 780 parameters; rank 4; C/M/H texture inputs; previous ten supported human events from the same run; shrinkage 0.5; clip 2.5; exact-zero bypass |

The exact TCN, GRU, ProDMP, offset, and training settings are kept in
[`Training and running ABCurves`](../docs/TRAINING_AND_INFERENCE.md) and in the model
containers themselves.

Each `.pt` file contains the tensors and a compact copy of its training and runtime
contract. The public files contain no local dataset or machine paths.

## File integrity

[`manifest.json`](manifest.json) is the machine-readable inventory. It records the
schema `abcurves.release_models.v1`, release version `1.0.0`, default seed, valid
seed sets, the frozen pipeline composition, exact file sizes, and SHA-256 hashes for
every `.pt` file and the shared scorer.

It also stores a `tensor_sha256` for every ordered tensor set. The ordinary hash
checks the entire saved file, including metadata. The tensor hash checks names,
dtypes, shapes, and tensor bytes in a stable order. The scorer has an equivalent
`contract_sha256` over its canonical JSON contents, which `FrozenStyleScorer()`
verifies before it exposes the transform.

The pipeline verifies sizes and file hashes before loading. A missing, changed, or
cross-seed file raises `ModelIntegrityError`. Keep that verification enabled when
you distribute the bundle.

```python
from abcurves.model_store import resolve_model_files

seed7 = resolve_model_files(7, verify=True)
seed23 = resolve_model_files(23, verify=True)

print(seed7.planner, seed7.renderer, seed7.renderer_adapter)
```

Run the release checks with:

```bash
python -m pytest \
  tests/test_release_contract.py tests/test_runtime.py tests/test_style_scorer.py
```

The actual digests live only in the manifest so they cannot drift between a prose
table and the loader that enforces them.

Each model container also records the SHA-256 of the internal source container from
which its public tensors were exported. This keeps the public bundle tied to its
training provenance without exposing workstation paths.
