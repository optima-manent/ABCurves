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
It records warmed observe/begin/step distributions, host and compiler identity,
sample counts, and the fact that USB/HID and application scheduling are excluded.

## Call order

```c
int status;

status = abc_online_model_init(&model, blob, blob_bytes);
if (status != ABC_FIXED_OK) return 1;
status = abc_online_reset(&renderer, &model);
if (status != ABC_FIXED_OK) return 1;

for (tick = 0; tick < 256; ++tick) {
    status = abc_online_observe_raw(&renderer, raw[tick].dx, raw[tick].dy);
    if (status != ABC_FIXED_OK) return 1;
}

status = abc_online_begin(&renderer, event_seed);
if (status != ABC_FIXED_OK) return 1;

while (intent_is_active) {
    status = abc_online_step(&renderer, smooth_dx_q16, smooth_dy_q16, &report);
    if (status != ABC_FIXED_OK) return 1;
    /* The caller owns USB/HID transport. */
}
```

The context is exactly 256 chronological, signed-int16 hardware reports after
reset. `begin` is one-shot. The smooth input to `step` is signed Q16 counts.
Do not silently truncate a longer history or keep observing after report 256.
The frozen validation covers this exact reset/observe/begin contract; an
arbitrary-length or continuously rolling observer needs a separate validation.

## Embedded scope

The packed model is always 44,484 bytes and a generated tick uses 33,760 int8
MACs. In the validated Windows x64 ABI, state is 5,088 bytes and the zero-copy
model view is 208 bytes; call `abc_online_renderer_size()` and
`abc_online_model_size()` on every other compiler/target instead of assuming
those two struct sizes. The hot generation
path is fixed-point/int8, while the observer uses double-precision summary math
and the rank-16 begin adapter uses float32 and `tanhf`.

That makes this a practical experimental starting point for ESP-IDF and other
small targets, not a measured ESP32 1 kHz certification. Porters should profile
cycles, stack high-water, code size and scheduling jitter on their exact MCU.
