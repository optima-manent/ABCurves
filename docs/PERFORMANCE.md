# Integrated runtime measurements

The combined runtime was measured on a Ryzen 7 9800X3D (8 cores / 16 threads), Windows 11 build 26200, using CPU inference throughout. These are local synchronous call timings with normal desktop background activity; they do not measure physical mouse delivery, operating-system scheduling guarantees, or end-to-end application latency. The campaign ran on 24 September 2026.

Python 3.13.14; NumPy 2.4.6; Numba 0.66.0 / llvmlite 0.48.0; PyTorch 2.12.0+cu130 loaded on CPU for Static. Continuous used the default native `pade9_vector` neural backend and Numba physical kernels, `skip_unused=True`, `diagnostics=False`. Renderer used the selected 44,484-byte model and bundled Windows C runtime. OpenBLAS, Numba, OMP and MKL limits were one thread; Static PyTorch intra/inter-op limits were one. CPU affinity was unrestricted, normal process priority, garbage collection enabled. No outliers were removed.

Measurements use the release checkout identified by the source hashes in [`combined-runtime.json`](../results/performance/combined-runtime.json). Every runtime Python file, selected model, continuous asset and native library was hashed before and after each worker. All hashes remained unchanged and agreed across workers. The receipt also records the benchmark tool, fixtures, library versions and thread-pool metadata.

## Warm continuous calls

Seven independently reset 8-second numerical traces used planner sampling seeds 7–13, with the same causal target-receipt schedule for planner alone and composition. A complete unmeasured trace preceded each campaign. Each measured call emits exactly one closed 1 ms sample; fresh decisions occur every 32 samples. The trace includes acquisition, moving receipts at 16 ms intervals, settling, holding and a later target jump. All three movement modes occurred. Initial position was (0,0), with no supplied movement history. The composed path used the authentic 256-report hardware profile and recorded sensitivity in `examples/data/human_start.npz`; hardware Y was down, network coordinates remained canonical Y-up.

All values below are **microseconds**. A fresh call includes its first output sample, and the composed fresh call also includes its first rendered hardware report. Buffered calls include the public allocation/copy boundary. Fresh decisions without a motor evaluation are listed separately.

| Public call | Samples | Median µs | p95 µs | Max µs |
|---|---:|---:|---:|---:|
| Continuous fresh call, motor evaluated | 1210 | 209.050 | 306.875 | 375.800 |
| Continuous fresh call, motor skipped | 540 | 43.900 | 118.735 | 246.900 |
| Continuous buffered 1 ms output | 54250 | 3.600 | 5.600 | 30.500 |
| Continuous target receipt | 1596 | 2.700 | 5.300 | 23.500 |
| Continuous reset | 7 | 253.100 | 329.630 | 354.200 |
| Continuous + Renderer fresh call, motor evaluated | 1210 | 233.650 | 319.810 | 572.700 |
| Continuous + Renderer fresh call, motor skipped | 540 | 74.600 | 151.815 | 398.700 |
| Continuous + Renderer buffered 1 ms output | 54250 | 25.500 | 30.600 | 1074.500 |
| Continuous + Renderer target receipt | 1596 | 2.900 | 3.800 | 30.100 |
| Continuous + Renderer reset | 7 | 271.900 | 302.910 | 312.000 |

There were 1,750 fresh decisions and 54,250 buffered calls per campaign; 1,210 fresh decisions evaluated the motor and 540 skipped it. The full combined fresh-call distribution (including all modes) was median 224.400 µs, p95 304.810 µs, max 572.700 µs. A buffered composed call reached 1,074.500 µs, exceeding one sample interval despite the smaller p95. Prepare and buffer work ahead of the actual output deadline; a 32 ms planning cadence is a numerical commitment, not 32 ms of computation, and 1 ms output spacing is not an operating-system timing guarantee.

System-wide CPU usage sampled immediately before the warm workers was 3.1% for planner alone, 16.9% for composition, 9.0% for controlled Static and 8.4% for sampled-head Static. Per-repetition values are retained in the receipt.

## Warm static continuation and renderer

Three actual eligible first-crossing B90 cases from three recorded training sessions cover initial edge distances 80–150, 150–400 and ≥400 native counts. Their B indices are 65, 168 and 189 ms; edge margins are 22.62, 25.20 and 52.33 counts. All raw example spans and genuine 256-report profiles were compared exactly to fresh public Capture exports. The cases illustrate runtime costs; they are not held-out quality evidence.

The main table uses ordinary **runtime sampled heads**, five rounds × three cases × sixteen draws = 240 complete generated continuations per independently trained model variant. All sixteen heads were sampled. Model-selection seeds 7/23 remain separate from runtime sampling seeds. Profiles were prepared once before the timed handoff. “Complete” measures one actual handoff through the last generated report; it is not a sum of independently measured component medians.

| Public operation | Samples | Median µs | p95 µs | Max µs |
|---|---:|---:|---:|---:|
| Static seed 7: fresh sampled-head plan | 240 | 133.200 | 237.960 | 455.800 |
| Static seed 7: plan + begin from prepared profile | 240 | 146.850 | 245.895 | 318.400 |
| Static seed 7: render complete planned movement | 240 | 1986.250 | 5581.085 | 6542.400 |
| Static seed 7: complete plan + render | 240 | 2121.350 | 5746.705 | 6735.600 |
| Static seed 7: prepare genuine 256-report profile | 120 | 2576.600 | 2731.055 | 3124.400 |
| Static seed 7: output one rendered tick | 36390 | 15.500 | 20.200 | 55.500 |
| Static seed 23: fresh sampled-head plan | 240 | 201.150 | 237.350 | 285.800 |
| Static seed 23: plan + begin from prepared profile | 240 | 171.350 | 264.940 | 331.300 |
| Static seed 23: render complete planned movement | 240 | 2101.700 | 4948.855 | 7336.800 |
| Static seed 23: complete plan + render | 240 | 2270.400 | 5123.610 | 7552.700 |
| Static seed 23: prepare genuine 256-report profile | 120 | 2575.450 | 2732.130 | 3538.900 |
| Static seed 23: output one rendered tick | 36115 | 15.500 | 19.700 | 50.100 |

The explicit all-sixteen-head campaign additionally produced 240 events per model, five repetitions per case/head. Its single-tick output exactly equaled complete-event output for every event. Output durations were 47–393 ms for seed 7 and 48–383 ms for seed 23. The one-tick and profile rows come from that controlled campaign; all-head measurements and encode/forward/decode timing are retained in the compact receipt. Whole-event rendering cost varies with the generated duration, so it is not interchangeable with per-tick cost.

## Preparation and process startup

Each kind/seed ran once with an empty dedicated Numba disk cache, then five times in a fresh process with that disk cache populated. File-system caches were warm: provenance hashing deliberately read the artifacts before timed startup. This is first compilation versus cached-process startup, not an OS cold-boot benchmark. Cold compilation has only one observation per kind/seed; do not report a percentile for it.

Continuous loading was staged as constructor `prewarm=False`, followed immediately by the same policy prewarm called by the default constructor. “Load + prewarm” times that full region directly; no stream samples are consumed. Composed loading additionally verifies/loads the renderer and prepares its genuine profile. Static uses its ordinary `prewarm=True` constructor, including full duration-cache preparation and one disposable handoff. Imports are separate from the constructor.

Warm-cache fresh-process values below are **milliseconds**, n=5 each.

| Stage | Samples | Median ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| Continuous imports | 5 | 63.056 | 68.227 | 68.319 |
| Continuous load/verify before prewarm | 5 | 211.048 | 230.871 | 231.244 |
| Continuous prewarm | 5 | 320.439 | 332.594 | 335.244 |
| Continuous load + prewarm | 5 | 529.045 | 563.154 | 566.488 |
| Continuous + Renderer imports | 5 | 58.720 | 63.955 | 64.955 |
| Continuous + Renderer load/verify before prewarm | 5 | 213.335 | 222.510 | 223.565 |
| Continuous + Renderer prewarm | 5 | 316.289 | 325.639 | 325.863 |
| Continuous + Renderer load + prewarm | 5 | 531.890 | 542.163 | 543.032 |
| Static seed 7 imports | 5 | 1341.737 | 1366.459 | 1371.993 |
| Static seed 7 constructor + prewarm | 5 | 528.533 | 542.554 | 543.684 |
| Static seed 23 imports | 5 | 1353.362 | 1379.700 | 1380.346 |
| Static seed 23 constructor + prewarm | 5 | 531.095 | 535.100 | 535.398 |

With the empty Numba cache, Continuous load + prewarm took 5.984 s, and composed loading + prewarm took 5.327 s. Static constructors with prewarm took 3.861 s (seed 7) and 3.928 s (seed 23), after imports. First post-prewarm real calls and every startup observation are retained in the compact receipt. Parent-process wall measurements include provenance hashing, a 0.5-second CPU-load probe and report writing; they are instrumentation totals and must not be presented as application startup latency.

## Reproduce these measurements

From a source checkout with the selected models and native libraries available:

```powershell
python -m pip install ".[static]" psutil threadpoolctl
python tools/benchmark_runtime.py --output benchmark-input-check --check
python tools/benchmark_runtime.py --output benchmark-output
```

The tool derives the checkout from its own location. An explicit `--release-root` selects another checkout; `--fixtures` selects its fixture directory when needed. Every worker imports that checkout. The output path is explicit, and all generated JSON, per-call arrays, `SUMMARY.md` tables and Numba caches remain below it. Use a fresh output directory for each complete run. The optional `--check` writes its receipt to the separate `benchmark-input-check` directory above; it must not populate the full campaign output directory. The full campaign includes startup, continuous generation, composed rendering, the controlled sixteen-head static sweep and ordinary static head sampling. Source and model hashes are recorded before and after each worker. Compare those bindings before comparing timings from different builds.

The four recorded fixtures and their lineage are in `examples/data`: `human_start.npz`, `static_event.npz`, `static_event_short.npz` and `static_event_long.npz`. `benchmark_static.json` fixes the three static cases and their order. Fixture checksums are validated before timing. The static input spans and hardware profiles were compared exactly with fresh public Capture exports. Session identities, source-export hashes, A/B/C bins, eligibility margins and source-roster hashes remain in the fixture metadata. These are recorded input examples, not held-out quality tests.

The preserved [`combined-runtime.json`](../results/performance/combined-runtime.json) receipt contains every median, p95, maximum, mean and sample-count summary, per-case and per-repetition details, cold observations, startup stages, phase counters, measured source/model hashes, benchmark-tool SHA-256 and original evidence hashes. Its accompanying `samples/` directory preserves the measured per-call arrays byte for byte. Statistics were independently checked against those arrays before packaging.
