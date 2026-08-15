# The released models

This folder contains two independently trained Planners and one shared selected
global Renderer.

| Role | File | Meaning |
| --- | --- | --- |
| Default Planner | [`planner_seed7.pt`](planner_seed7.pt) | Selected epoch-260 Planner |
| Planner replication | [`planner_seed23.pt`](planner_seed23.pt) | Independent epoch-260 training cell |
| Global Renderer | [`renderer_global_h80.bin`](renderer_global_h80.bin) | Shared full-corpus quantized Renderer and prefix handoff |
| Selected float law | [`renderer_global_h80_float.pt`](renderer_global_h80_float.pt) | Sanitized 20-feature active float graph behind the promotion |
| Integrity inventory | [`manifest.json`](manifest.json) | Sizes, hashes, architecture, and release contract |

Seed 7 and seed 23 are not an ensemble. Normal inference loads one Planner, samples
one of its sixteen heads, and sends that one smooth intent to the same global
Renderer. It does not average the Planners or keep the best-looking candidate.

## Loading the release

```python
from abcurves import Pipeline

with Pipeline.from_pretrained() as default:
    # Planner seed 7 + shared global Renderer.
    pass

with Pipeline(model_seed=23) as replication:
    # Planner seed 23 + the same shared global Renderer.
    pass
```

`model_seed` chooses Planner weights. The event `seed` passed to
`Pipeline.generate()` is different: it makes the one Planner-head draw and Renderer
sample repeatable for that movement.

The Renderer prepares a reusable profile from exactly 256 chronological integer
reports. Prepare it before a latency-sensitive B handoff, then clone that immutable
state for each event. The profile contract is independent of which Planner seed is
loaded, and the representative sample does not need to end at B.

## Planner contract

Both Planner files share the same architecture and training rule:

| Part | Frozen value |
| --- | --- |
| Prefix | Last 160 raw 1 ms bins plus validity |
| Causal summaries | 62 movement and target features |
| Network | Width-96, 3-block causal TCN, dropout 0.15 |
| Output | RWTA-16, 43 values per head |
| Movement primitive | 20 ProDMP forcing bases plus learned goal |
| Maximum future | 1,000 ms |
| Learned parameters | 369,904 |
| Selection | Terminal epoch 260 |

Each `.pt` container stores only the public tensor and runtime contract: model state,
normalizers, seam rules, prefix representation, ProDMP settings, tensor hash, source
container identity, and training metadata. It contains no workstation dataset path.

## Global Renderer contract

The shared Renderer was trained on blind windows from complete dense sessions:

```text
256 observed physical reports | 800 future physical reports
```

The selected full training split contains 81,737 windows, 54 sessions, and 45
users. Validation contains 10,807 windows, 8 sessions, and 8 users held out from
Renderer training. Planner and Renderer preserve different frozen split salts,
so this is not by itself a joint whole-system holdout. No A, B, C, target, outcome,
or success filter participates in Renderer windowing.

The float training graph is phase-free and has:

- 20 causal inputs;
- one width-80 GRU;
- an emit head and 121 joint offset classes for radius 5; and
- 34,362 unique learned scalars.

It was trained with natural window weighting, randomly selected window-3/window-5
teachers, future-only loss, Adam at learning rate `2e-3` and weight decay `1e-5`, and
an exact budget of 118,345 presentations. Teacher-forced loss is diagnostic;
checkpoint promotion used sampled carried full-session texture.

Teacher offset labels use base hysteresis `1.0`. The later deployment calibration
uses `0.5`; that sampled-law setting did not change the learned tensors.

The frozen sampler chooses one w3/w5 view for every source in each shuffled pass.
Pass boundaries are optimizer boundaries: the first full pass ends with a 73-window batch, followed
by 143 full batches from the next pass, for 463 optimizer steps total.

Within every training window, all 256 context reports define the regime and learned
handoff state. The float recurrent warm-up uses the most recent 128 reports. The
packed rank-16 handoff bridges the full observation to that canonical recurrent
boundary; callers must still supply exactly 256 reports when preparing a deployment
profile. Reusing the prepared profile does not change this training contract.

## What the binary contains

`renderer_global_h80.bin` is a combined, zero-copy model image:

| Component | Bytes |
| --- | ---: |
| Post-training-quantized base | 39,512 |
| Rank-16 prefix handoff | 4,972 |
| **Total** | **44,484** |

The handoff compresses one representative 256-report sample into recurrent state.
`RendererProfile` is the name of that reusable prepared state; it is not a user
identity or prior-event personalization model. The Planner still supplies the
event-specific smooth-intent boundary at B; the profile supplies Renderer packet
state, including prior emission, last smooth motion, run state, and recent activity.

The deployment calibration is frozen with the artifact:

| Control | Value |
| --- | ---: |
| Emit bias / temperature | 1.5 / 1.3 |
| Offset magnitude / direction temperature | 0.75 / 0.15 |
| Axis hysteresis | 0.5 |
| Safety release | 32 counts; evaluated before quiet gating |
| Quiet gate | intent `<=1e-7` in float / exact zero in Q16, both debt axes `<0.5` |
| Maximum emitted axis | `+/-127` counts |
| Lateral-offset safeguard | `1.5 * max(abs(offset dot normal) - 1, 0)` logit penalty |

The hot recurrence is fixed-point/int8 and costs 33,760 int8
multiply-accumulates per generated tick. On the validated Windows x64 ABI, the
no-heap C99 runtime uses a 208-byte model view and 5,088 bytes for each caller-owned
Renderer state. The reusable path retains one profile state and copies it into one
state per active event. Ports must query the size helpers on their target ABI.
Profile preparation and the rank-16 handoff still use float/double math; the artifact
is functionally portable but is not presented as an ESP32 timing result.

## Exact artifact identity

The selected Renderer SHA-256 is:

```text
8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b
```

The fixed-online eight-session promotion scored `S=1.4328467` versus `1.4311797` for its
float source (`+0.11648%`, lower is better). `S` combines Texture19 W1, packet-rate
ratio error, false activation on true zero intent, and relative session net error.
This is a frozen eight-session/eight-user non-protected development panel, not an
untouched-user claim. Separately, the rank-16 online handoff
and the same fixed model with warm replay differed in only 8 of 22,829,006 scalar
components, each by one count. That second comparison validates the handoff; it is
not a float-versus-quantized result.

The exact score formula, component values, panel size, scope, and sealed source
digest are in
[`renderer_promotion.json`](../results/inference/renderer_promotion.json).

The artifact digest appears in the model manifest and is enforced by the Python
loader. The sealed research-receipt digest is separately bound by the inference
result manifest. The C loader then verifies the packed format, CRC, and source
identity before exposing the model view.

## Manifest and verification

[`manifest.json`](manifest.json) records:

- supported Planner seeds and the default seed;
- exact filenames, byte sizes, and SHA-256 digests;
- stable tensor digests for both Planner containers;
- the shared Renderer architecture, corpus, presentation budget, and selection rule;
- binary base/handoff sizes; and
- deployment state, compute, and AF1.5 settings.

`Pipeline` verifies model files before loading by default. A missing, changed, or
undeclared file raises `ModelIntegrityError`.

```python
from abcurves.model_store import resolve_model_files, resolve_renderer_float

seed7 = resolve_model_files(7, verify=True)
seed23 = resolve_model_files(23, verify=True)

print(seed7.planner, seed7.renderer)
print(seed23.planner, seed23.renderer)
assert seed7.renderer == seed23.renderer
print(resolve_renderer_float())  # Independently authenticated research checkpoint.
```

Keep verification enabled for distributed bundles. `verify=False` exists for narrow
development work; it removes a release safeguard and should not be used to describe
an artifact as the published model.

## Float checkpoints are not deployment images

`renderer_global_h80_float.pt` is the sanitized selected float graph: 34,362 active
learned scalars across eight tensors, a phase-free 20-feature contract, and no
duplicate compatibility cell, workstation paths, or private roster history. Its
active tensors match the selected
P118345 source, whose container digest is retained in the manifest. Load this
specific file with `load_count_model(resolve_renderer_float())` so the immutable
release anchor is checked first.

[`training/train_renderer.py`](../training/train_renderer.py) writes a new float
research checkpoint. Use it directly with
`Pipeline(float_renderer_checkpoint="runs/renderer_p118345.pt")`; this safely loads
the float graph through the ordinary `RendererProfile` interface. A float profile
object may be reused, but that backend replays its raw 256-report window and
pre-renders the sampled continuation at every event start. It does not recreate or
overwrite `renderer_global_h80.bin`, and its timing is not the native profile-clone
claim. A new binary promotion would additionally need post-training quantization,
context-handoff fitting, C/Python differential tests, a new digest, and an updated
manifest.

Similarly, replacing a Planner `.pt` without updating its tensor contract and
manifest does not make a valid model directory.

## Native library

The model image is platform-neutral data. The code that executes it lives in
[`runtime/c`](../runtime/c). Windows installs include the built DLL; other targets can
compile the same C99 source:

```bash
cmake -S runtime/c -B runtime/c/build
cmake --build runtime/c/build --config Release
ctest --test-dir runtime/c/build -C Release --output-on-failure
```

The validated native lifecycle is prepare a template from exactly 256 representative
reports → copy the template for one event → begin once → step once per
smooth-intent tick. The template is immutable and reusable. It is prepared before B;
rolling or arbitrary-length observation is not silently treated as equivalent.

Run the Python release checks with:

```bash
python -m pytest \
  tests/test_release_contract.py \
  tests/test_runtime.py \
  tests/test_renderer_reference.py
```

The architecture, training budget, sampled-selection rule, and integration contract
are explained in
[`TRAINING_AND_INFERENCE.md`](../docs/TRAINING_AND_INFERENCE.md).
