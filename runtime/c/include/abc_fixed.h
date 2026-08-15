#ifndef ABC_FIXED_H
#define ABC_FIXED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ABC_FIXED_HIDDEN 80
#define ABC_FIXED_FEATURES 20
#define ABC_FIXED_RADIUS 5
#define ABC_FIXED_SIDE 11
#define ABC_FIXED_GRID 121
#define ABC_FIXED_GATES 240
#define ABC_FIXED_PREFIX_WINDOW 256
#define ABC_FIXED_WARM_WINDOW 128
#define ABC_FIXED_RECENT_WINDOW 60
#define ABC_FIXED_ARTIFACT_BYTES 39512U

typedef enum abc_fixed_status {
    ABC_FIXED_OK = 0,
    ABC_FIXED_ERR_ARGUMENT = -1,
    ABC_FIXED_ERR_MODEL = -2,
    ABC_FIXED_ERR_CRC = -3,
    ABC_FIXED_ERR_SOURCE = -4,
    ABC_FIXED_ERR_MODE = -5,
    ABC_FIXED_ERR_EMPTY_PREFIX = -6,
    ABC_FIXED_ERR_RANGE = -7,
    ABC_FIXED_ERR_IO = -8
} abc_fixed_status_t;

typedef enum abc_fixed_mode {
    ABC_FIXED_MODE_UNINITIALIZED = 0,
    ABC_FIXED_MODE_OBSERVE = 1,
    ABC_FIXED_MODE_GENERATE = 2
} abc_fixed_mode_t;

/*
 * Zero-copy view into one ABCFIX1 byte image. The caller retains ownership of
 * `blob` for the lifetime of this object and every renderer that refers to it.
 */
typedef struct abc_fixed_model {
    const uint8_t *blob;
    size_t blob_bytes;
    const int8_t *weights;
    const uint8_t *biases_le;
    const uint8_t *multipliers_le;
    const uint8_t *sigmoid_le;
    const uint8_t *tanh_le;
    const uint8_t *exp_le;
    const uint8_t *log_le;
    int32_t config[11];
    uint32_t body_crc32;
} abc_fixed_model_t;

typedef struct abc_fixed_report {
    int16_t dx;
    int16_t dy;
} abc_fixed_report_t;

typedef struct abc_fixed_hot_scratch {
    /* Reused chronologically as raw logits, transformed logits, then Q20 exp. */
    int32_t logits[ABC_FIXED_GRID];
    int16_t next_hidden[ABC_FIXED_HIDDEN];
    int16_t feature[ABC_FIXED_FEATURES];
} abc_fixed_hot_scratch_t;

typedef struct abc_fixed_prefix_scratch {
    uint16_t magnitudes[ABC_FIXED_PREFIX_WINDOW];
} abc_fixed_prefix_scratch_t;

typedef union abc_fixed_scratch {
    abc_fixed_hot_scratch_t hot;
    abc_fixed_prefix_scratch_t prefix;
} abc_fixed_scratch_t;

typedef struct abc_fixed_renderer {
    const abc_fixed_model_t *model;
    abc_fixed_mode_t mode;

    int16_t hidden[ABC_FIXED_HIDDEN];
    int32_t accumulator_q16[2];
    int16_t previous_emit[2];
    int16_t last_nonzero[2];
    int16_t last_axis_nonzero[2];
    uint8_t run_active;
    uint16_t run_length;
    uint8_t active_ring[ABC_FIXED_RECENT_WINDOW];
    uint8_t active_ring_pos;
    uint8_t active_ring_count;

    int16_t prefix_raw[ABC_FIXED_PREFIX_WINDOW][2];
    uint16_t prefix_pos;
    uint16_t prefix_count;
    uint64_t observed_ticks;

    int16_t regime_q8[5];
    int16_t tangent_q15[2];
    int32_t last_smooth_q16[2];
    uint32_t previous_speed_q16;

    uint64_t event_seed;
    uint64_t generation_tick;

    /* Last-step diagnostics; not inference-required and removable in firmware. */
    uint32_t last_emit_probability_q15;
    uint16_t last_joint_index;
    uint32_t last_emit_top23;
    uint32_t last_offset_top23;

    /* Caller-owned, fixed-size storage. No heap is used by the core. */
    abc_fixed_scratch_t scratch;
} abc_fixed_renderer_t;

/* Fully prepared non-neural handoff state supplied by the canonical online
 * observer. This contains no learned data and is copied once at begin. */
typedef struct abc_fixed_online_boundary {
    int16_t regime_q8[5];
    int16_t previous_emit[2];
    int16_t last_nonzero[2];
    int16_t last_axis_nonzero[2];
    int32_t last_smooth_q16[2];
    uint16_t run_length;
    uint8_t run_active;
    uint8_t active_ring[ABC_FIXED_RECENT_WINDOW];
    uint8_t active_ring_pos;
    uint8_t active_ring_count;
} abc_fixed_online_boundary_t;

/* This header exposes storage layout needed by the stack-allocated online
 * wrapper. The fixed core itself is private; use the abc_online_* API. */

#ifdef __cplusplus
}
#endif

#endif
