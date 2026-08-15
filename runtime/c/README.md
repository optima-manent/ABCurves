# Compact Renderer C runtime

This directory runs the exact 44,484-byte Renderer in
`models/renderer_global_h80.bin`. The core is C99, allocates no heap memory,
and keeps all mutable state in a caller-owned `abc_online_renderer_t`.

## Build and test

```sh
cmake -S runtime/c -B runtime/c/build
cmake --build runtime/c/build --config Release
ctest --test-dir runtime/c/build -C Release --output-on-failure
```

The loader rejects the wrong byte length, CRC, source identity, API order, or
an extra observation. The public Python wrapper also authenticates the file's
SHA-256 before calling C.

Run the native-core benchmark on the target machine with:

```sh
runtime/c/build/Release/abc_renderer_benchmark.exe \
  models/renderer_global_h80.bin native_renderer_local.json 128
```

The checked-in Windows x64 receipt is
[`results/inference/native_renderer_windows_x64.json`](../../results/inference/native_renderer_windows_x64.json).
It separates off-path profile preparation from the profile-copy/begin and generated
tick paths, and records the host, compiler, sample counts, and excluded integration
layers.

## Call order

```c
int status;
abc_online_renderer_t profile;
abc_online_renderer_t event;

status = abc_online_model_init(&model, blob, blob_bytes);
if (status != ABC_FIXED_OK) return 1;
status = abc_online_reset(&profile, &model);
if (status != ABC_FIXED_OK) return 1;

for (tick = 0; tick < 256; ++tick) {
    status = abc_online_observe_raw(&profile, sample[tick].dx, sample[tick].dy);
    if (status != ABC_FIXED_OK) return 1;
}

/* Later, at B: clone the immutable template into independent event state. */
event = profile;
status = abc_online_begin(&event, event_seed);
if (status != ABC_FIXED_OK) return 1;

while (intent_is_active) {
    status = abc_online_step(&event, smooth_dx_q16, smooth_dy_q16, &report);
    if (status != ABC_FIXED_OK) return 1;
    /* The caller owns USB/HID transport. */
}
```

Profile input is exactly 256 chronological, signed-int16 hardware reports after
reset. It should be a representative physical sample, but it does not need to end at
B. Do not silently truncate a longer history or pad a shorter one.

Keep the fully observed `profile` unchanged. C structure assignment copies it into
an independent `event`; `begin` is one-shot on that event copy. The smooth input to
`step` is signed Q16 counts. The profile may be reused for later events by copying it
into separate caller-owned states, as long as the model and its backing blob remain
alive. The caller is responsible for synchronization and object lifetimes. Prepare a
replacement off-path and select it only between events if the physical setup changes
materially. There is no timer-driven refresh or continuously rolling observer in
this contract.

Frozen training and detection experiments may deliberately prepare an exact
per-event window ending at B. That remains valid, but it is an experimental input
schedule rather than a requirement of the recommended real-time lifecycle.

## Embedded scope

The packed model is always 44,484 bytes and a generated tick uses 33,760 int8
MACs. In the validated Windows x64 ABI, each profile or active-event state is 5,088
bytes and the zero-copy model view is 208 bytes; call `abc_online_renderer_size()` and
`abc_online_model_size()` on every other compiler/target instead of assuming
those two struct sizes. One retained profile plus one active event therefore uses two
Renderer-state objects. The hot generation path is fixed-point/int8, while profile
preparation uses double-precision summary math and the rank-16 begin adapter uses
float32 and `tanhf`.

That makes this a practical experimental starting point for ESP-IDF and other
small targets, not a measured ESP32 1 kHz certification. Porters should profile
cycles, stack high-water, code size and scheduling jitter on their exact MCU.
