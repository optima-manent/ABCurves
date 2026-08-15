#ifndef ABC_ONLINE_H
#define ABC_ONLINE_H

#include "abc_fixed.h"
#include "abc_w5_stream.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(ABC_RENDERER_EXPORTS)
#define ABC_RENDERER_API __declspec(dllexport)
#elif defined(__GNUC__) && defined(ABC_RENDERER_EXPORTS)
#define ABC_RENDERER_API __attribute__((visibility("default")))
#else
#define ABC_RENDERER_API
#endif

#define ABC_ONLINE_BLOB_BYTES 44484U
#define ABC_ONLINE_FIXED_BYTES 39512U
#define ABC_ONLINE_ADAPTER_BYTES 4972U
#define ABC_ONLINE_ADAPTER_INPUT 145
#define ABC_ONLINE_ADAPTER_RANK 16

/* ABI helpers for language bindings that keep the structs opaque. */
ABC_RENDERER_API size_t abc_online_model_size(void);
ABC_RENDERER_API size_t abc_online_renderer_size(void);
ABC_RENDERER_API size_t abc_online_report_size(void);

typedef struct abc_online_adapter {
    const uint8_t *blob;
    const uint8_t *mean_f16;
    const uint8_t *invstd_f16;
    const uint8_t *v_scale_f32;
    const uint8_t *v_bias_f32;
    const int8_t *v_q;
    const uint8_t *u_scale_f32;
    const uint8_t *u_bias_f32;
    const int8_t *u_q;
} abc_online_adapter_t;

typedef struct abc_online_model {
    const uint8_t *blob;
    size_t blob_bytes;
    abc_fixed_model_t fixed;
    abc_online_adapter_t adapter;
} abc_online_model_t;

/* Incremental state for the exact 256-report observation window. A second
 * core starts at canonical observation 128 and removes replay from begin(). */
typedef struct abc_online_core_state {
    double cumulative_smooth[2];
    double cumulative_previous_raw[2];
    double previous_raw[2];
    double last_nonzero[2];
    int16_t last_axis_nonzero[2];
    double last_direction[2];
    double previous_smooth[2];
    double previous_speed;
    uint16_t run_length;
    uint8_t run_active;
    uint8_t active_ring[60];
    uint8_t ring_pos;
    uint8_t ring_count;
} abc_online_core_state_t;

typedef struct abc_online_renderer {
    const abc_online_model_t *model;
    abc_fixed_renderer_t fixed;
    abc_w5_stream_t smoother;
    abc_w5_stream_t target_smoother;
    uint16_t observed;
    abc_online_core_state_t observer_core;
    abc_online_core_state_t target_core;
    uint8_t canonical_active_tail[60];

    uint32_t active_count;
    double active_sum;
    uint16_t running_max;
    uint16_t previous_magnitude;
    uint32_t sign_flip_count;
    uint32_t sign_comparison_count;
    int8_t previous_sign[2];
    double delta_energy;

    uint16_t sorted_active_magnitudes[256];
    double magnitude_sum_squares;
    double low_dft_real[7];
    double low_dft_imag[7];
    double low_dft_twiddle_real[7];
    double low_dft_twiddle_imag[7];
    double low_dft_oscillator_real[7];
    double low_dft_oscillator_imag[7];
    double nyquist;

    int16_t last_observer_feature_q8[20];
    int16_t final_regime_q8[5];
    int16_t target_tail4_core_q8[4][15];
    float adapter_input[145];
    float adapter_output[80];
} abc_online_renderer_t;

ABC_RENDERER_API int abc_online_model_init(
    abc_online_model_t *model,
    const void *combined_blob,
    size_t combined_bytes
);

ABC_RENDERER_API int abc_online_reset(
    abc_online_renderer_t *renderer,
    const abc_online_model_t *model
);
ABC_RENDERER_API int abc_online_observe_raw(
    abc_online_renderer_t *renderer,
    int16_t dx,
    int16_t dy
);
ABC_RENDERER_API int abc_online_begin(abc_online_renderer_t *renderer, uint64_t event_seed);
ABC_RENDERER_API int abc_online_step(
    abc_online_renderer_t *renderer,
    int32_t smooth_dx_q16,
    int32_t smooth_dy_q16,
    abc_fixed_report_t *report
);

#ifdef __cplusplus
}
#endif

#endif
