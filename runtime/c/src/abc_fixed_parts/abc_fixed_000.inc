#include "abc_fixed.h"

#include <float.h>
#include <math.h>
#include <string.h>

#define ABC_HEADER_BYTES 256U
#define ABC_FEATURE_Q 8
#define ABC_HIDDEN_Q 15
#define ABC_GATE_Q 14
#define ABC_SMOOTH_Q 16
#define ABC_PROB_Q 15
#define ABC_EXP_Q 20
#define ABC_LOG_Q 14
#define ABC_TANGENT_Q 15
#define ABC_LUT_INTERVALS 256
#define ABC_LUT_POINTS 257
#define ABC_LUT_DOMAIN_Q14 (8 << ABC_GATE_Q)
#define ABC_EXP_DOMAIN_Q14 (16 << ABC_GATE_Q)
#define ABC_Q31 UINT64_C(2147483648)
#define ABC_I32_MIN_VALUE (-INT32_MAX - 1)
#define ABC_GAMMA_U64 UINT64_C(0x9E3779B97F4A7C15)
#define ABC_MIX1_U64 UINT64_C(0xBF58476D1CE4E5B9)
#define ABC_MIX2_U64 UINT64_C(0x94D049BB133111EB)
#define ABC_EXPECTED_BODY_CRC32 UINT32_C(0x6be2896c)

enum {
    CFG_EMIT_BIAS_Q14 = 0,
    CFG_INV_EMIT_Q16 = 1,
    CFG_INV_MAG_Q16 = 2,
    CFG_INV_DIR_Q16 = 3,
    CFG_HYSTERESIS_Q16 = 4,
    CFG_FORCE_RELEASE_Q16 = 5,
    CFG_MAX_ABS_COUNT = 6,
    CFG_ZERO_INTENT_Q16 = 7,
    CFG_ZERO_DEBT_Q16 = 8,
    CFG_AF_Q16 = 9,
    CFG_LATERAL_FREE_Q15 = 10
};

static const uint8_t ABC_MAGIC[8] = {
    'A', 'B', 'C', 'F', 'I', 'X', '1', 0
};

static const uint8_t ABC_EXPECTED_SOURCE_SHA256[32] = {
    0xe9, 0x95, 0x1a, 0x9b, 0xc2, 0x5b, 0x69, 0xbf,
    0x65, 0x2c, 0xeb, 0xbc, 0xa8, 0x74, 0x9b, 0xad,
    0xd0, 0x2b, 0x8c, 0x06, 0x75, 0xa5, 0xf1, 0xad,
    0x4c, 0xde, 0x7f, 0x1a, 0x86, 0x24, 0x13, 0x2a
};

static const uint8_t ABC_EXPECTED_SCHEMA_TAG[16] = {
    0x34, 0x85, 0xfc, 0x08, 0xbf, 0xb9, 0xfb, 0xc7,
    0xee, 0x7f, 0x30, 0x73, 0x4e, 0x9f, 0xd0, 0xa4
};

static uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8U));
}

static int16_t read_i16_le(const uint8_t *p) {
    return (int16_t)read_u16_le(p);
}

static uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] |
        ((uint32_t)p[1] << 8U) |
        ((uint32_t)p[2] << 16U) |
        ((uint32_t)p[3] << 24U);
}

static int32_t read_i32_le(const uint8_t *p) {
    return (int32_t)read_u32_le(p);
}

static int32_t clamp_i32(int64_t value, int32_t low, int32_t high) {
    if (value < (int64_t)low) {
        return low;
    }
    if (value > (int64_t)high) {
        return high;
    }
    return (int32_t)value;
}

static int16_t clamp_i16(int64_t value, int16_t low, int16_t high) {
    if (value < (int64_t)low) {
        return low;
    }
    if (value > (int64_t)high) {
        return high;
    }
    return (int16_t)value;
}

static uint64_t abs_i64_u64(int64_t value) {
    if (value >= 0) {
        return (uint64_t)value;
    }
    return (uint64_t)(-(value + 1)) + 1U;
}

static int64_t round_div_even_i64(int64_t value, uint64_t divisor) {
    uint64_t absolute;
    uint64_t quotient;
    uint64_t remainder;
    uint64_t twice;
    int increment;
    int64_t result;

    absolute = abs_i64_u64(value);
    quotient = absolute / divisor;
    remainder = absolute % divisor;
    twice = remainder * 2U;
    increment = twice > divisor || (twice == divisor && (quotient & 1U));
    quotient += (uint64_t)increment;
    result = (int64_t)quotient;
    return value < 0 ? -result : result;
}

static int32_t round_shift_even_i64(int64_t value, unsigned shift) {
    return (int32_t)round_div_even_i64(value, UINT64_C(1) << shift);
}

static int16_t quantize_q8_f32(float value) {
    double scaled = (double)value * 256.0;
    double rounded = nearbyint(scaled);
    if (rounded < -32768.0) {
        return INT16_MIN;
    }
    if (rounded > 32767.0) {
        return INT16_MAX;
    }
    return (int16_t)rounded;
}

uint32_t abc_fixed_crc32(const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = UINT32_C(0xffffffff);
    size_t index;
    unsigned bit;
    for (index = 0U; index < bytes; ++index) {
        crc ^= p[index];
        for (bit = 0U; bit < 8U; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1U);
            crc = (crc >> 1U) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return crc ^ UINT32_C(0xffffffff);
}

static int validate_section(
    uint32_t offset,
    uint32_t length,
    uint32_t expected_offset,
    uint32_t expected_length
) {
    return offset == expected_offset && length == expected_length;
}

int abc_fixed_model_init(
    abc_fixed_model_t *model,
    const void *blob_void,
    size_t blob_bytes
) {
    const uint8_t *blob = (const uint8_t *)blob_void;
    static const uint8_t expected_q[8] = {8, 15, 14, 16, 15, 20, 14, 15};
    static const uint32_t expected_weight_counts[4] = {4800, 19200, 80, 9680};
    static const uint32_t expected_bias_counts[4] = {240, 240, 1, 121};
    static const uint16_t expected_group_counts[4] = {3, 3, 1, 121};
    static const int32_t expected_config[11] = {
        24576, 50412, 87381, 436907, 32768, 2097152,
        127, 0, 32768, 98304, 32768
    };
    static const uint32_t section_offsets[7] = {
        256, 34016, 36424, 36936, 37452, 37968, 38996
    };
    static const uint32_t section_lengths[7] = {
        33760, 2408, 512, 514, 514, 1028, 514
    };
    unsigned index;
    uint32_t crc;

    if (model == NULL || blob == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    memset(model, 0, sizeof(*model));
    if (blob_bytes != ABC_FIXED_ARTIFACT_BYTES || blob_bytes < ABC_HEADER_BYTES) {
        return ABC_FIXED_ERR_MODEL;
    }
    if (memcmp(blob, ABC_MAGIC, sizeof(ABC_MAGIC)) != 0 ||
        read_u16_le(blob + 8) != 1U ||
        read_u16_le(blob + 10) != ABC_HEADER_BYTES ||
        read_u16_le(blob + 12) != ABC_FIXED_HIDDEN ||
        read_u16_le(blob + 14) != ABC_FIXED_FEATURES ||
        read_u16_le(blob + 16) != ABC_FIXED_RADIUS ||
        read_u16_le(blob + 18) != 2U ||
        read_u32_le(blob + 20) != 15U ||
        read_u32_le(blob + 24) != ABC_FIXED_ARTIFACT_BYTES) {
        return ABC_FIXED_ERR_MODEL;
    }
    for (index = 0U; index < 7U; ++index) {
        if (!validate_section(
                read_u32_le(blob + 28U + 8U * index),
                read_u32_le(blob + 32U + 8U * index),
                section_offsets[index],
                section_lengths[index])) {
            return ABC_FIXED_ERR_MODEL;
        }
    }
    if (memcmp(blob + 88, expected_q, sizeof(expected_q)) != 0) {
        return ABC_FIXED_ERR_MODEL;
    }
    for (index = 0U; index < 4U; ++index) {
        if (read_u32_le(blob + 96U + 4U * index) != expected_weight_counts[index] ||
            read_u32_le(blob + 112U + 4U * index) != expected_bias_counts[index] ||
            read_u16_le(blob + 128U + 2U * index) != expected_group_counts[index]) {
            return ABC_FIXED_ERR_MODEL;
        }
    }
    for (index = 0U; index < 11U; ++index) {
        if (read_i32_le(blob + 136U + 4U * index) != expected_config[index]) {
            return ABC_FIXED_ERR_MODEL;
        }
    }
    if (memcmp(blob + 180, ABC_EXPECTED_SOURCE_SHA256, 32U) != 0 ||
        memcmp(blob + 212, ABC_EXPECTED_SCHEMA_TAG, 16U) != 0) {
        return ABC_FIXED_ERR_SOURCE;
    }
    for (index = 228U; index < ABC_HEADER_BYTES; ++index) {
        if (blob[index] != 0U) {
            return ABC_FIXED_ERR_MODEL;
        }
    }
    if (read_u32_le(blob + 84) != ABC_EXPECTED_BODY_CRC32) {
        return ABC_FIXED_ERR_CRC;
    }
    crc = abc_fixed_crc32(blob + ABC_HEADER_BYTES, blob_bytes - ABC_HEADER_BYTES);
    if (crc != ABC_EXPECTED_BODY_CRC32) {
        return ABC_FIXED_ERR_CRC;
    }

    model->blob = blob;
    model->blob_bytes = blob_bytes;
    model->weights = (const int8_t *)(blob + section_offsets[0]);
    model->biases_le = blob + section_offsets[1];
    model->multipliers_le = blob + section_offsets[2];
    model->sigmoid_le = blob + section_offsets[3];
    model->tanh_le = blob + section_offsets[4];
    model->exp_le = blob + section_offsets[5];
