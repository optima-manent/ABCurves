#include "abc_online.h"
#include "abc_fixed_internal.h"

#include <math.h>
#include <string.h>

#define OHV_HEADER_BYTES 24U
#define OHV_CRC32 UINT32_C(0x6aa9101c)

static const uint8_t OHV_MAGIC[8] = {'O','H','V','1','R','1','6',0};

size_t abc_online_model_size(void) {
    return sizeof(abc_online_model_t);
}

size_t abc_online_renderer_size(void) {
    return sizeof(abc_online_renderer_t);
}

size_t abc_online_report_size(void) {
    return sizeof(abc_fixed_report_t);
}

static uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) |
        ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
}

static float read_f32_le(const uint8_t *p) {
    uint32_t bits = read_u32_le(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float read_f16_le(const uint8_t *p) {
    uint32_t h = (uint32_t)p[0] | ((uint32_t)p[1] << 8U);
    uint32_t sign = (h & 0x8000U) << 16U;
    uint32_t exponent = (h >> 10U) & 0x1fU;
    uint32_t fraction = h & 0x3ffU;
    uint32_t bits;
    float value;
    if (exponent == 0U) {
        if (fraction == 0U) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((fraction & 0x400U) == 0U) {
                fraction <<= 1U;
                --unbiased;
            }
            fraction &= 0x3ffU;
            bits = sign | ((uint32_t)(unbiased + 127) << 23U) | (fraction << 13U);
        }
    } else if (exponent == 31U) {
        bits = sign | 0x7f800000U | (fraction << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
    }
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Frozen float32 dot topology: two four-lane FMA accumulators, pairwise lane
 * reduction, then scalar remainder. This is deterministic across compiler
 * vector widths and matches the adapter's sealing reference. */
static float adapter_dot_f32(
    const int8_t *quantized,
    float scale,
    const float *values,
    unsigned count
) {
    float lane0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float lane1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float merged[4];
    float sum;
    unsigned column = 0U;
    unsigned lane;
    while (column + 8U <= count) {
        for (lane = 0U; lane < 4U; ++lane) {
            float weight0 = (float)quantized[column + lane] * scale;
            float weight1 = (float)quantized[column + 4U + lane] * scale;
            lane0[lane] = fmaf(weight0, values[column + lane], lane0[lane]);
            lane1[lane] = fmaf(weight1, values[column + 4U + lane], lane1[lane]);
        }
        column += 8U;
    }
    for (lane = 0U; lane < 4U; ++lane) merged[lane] = lane0[lane] + lane1[lane];
    sum = (merged[0] + merged[1]) + (merged[2] + merged[3]);
    while (column < count) {
        float weight = (float)quantized[column] * scale;
        sum = fmaf(weight, values[column], sum);
        ++column;
    }
    return sum;
}

int abc_online_model_init(
    abc_online_model_t *model,
    const void *combined_blob,
    size_t combined_bytes
) {
    const uint8_t *blob = (const uint8_t *)combined_blob;
    const uint8_t *adapter;
    size_t offset;
    int status;
    if (model == NULL || blob == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    memset(model, 0, sizeof(*model));
    if (combined_bytes != ABC_ONLINE_BLOB_BYTES) {
        return ABC_FIXED_ERR_MODEL;
    }
    status = abc_fixed_model_init(&model->fixed, blob, ABC_ONLINE_FIXED_BYTES);
    if (status != ABC_FIXED_OK) {
        return status;
    }
    adapter = blob + ABC_ONLINE_FIXED_BYTES;
    if (abc_fixed_crc32(adapter, ABC_ONLINE_ADAPTER_BYTES) != OHV_CRC32 ||
        memcmp(adapter, OHV_MAGIC, sizeof(OHV_MAGIC)) != 0 ||
        read_u32_le(adapter + 8U) != 1U ||
        read_u32_le(adapter + 12U) != ABC_ONLINE_ADAPTER_INPUT ||
        read_u32_le(adapter + 16U) != ABC_ONLINE_ADAPTER_RANK ||
        read_u32_le(adapter + 20U) != ABC_FIXED_HIDDEN) {
        return ABC_FIXED_ERR_MODEL;
    }
    offset = OHV_HEADER_BYTES;
    model->adapter.blob = adapter;
    model->adapter.mean_f16 = adapter + offset; offset += 145U * 2U;
    model->adapter.invstd_f16 = adapter + offset; offset += 145U * 2U;
    model->adapter.v_scale_f32 = adapter + offset; offset += 16U * 4U;
    model->adapter.v_bias_f32 = adapter + offset; offset += 16U * 4U;
    model->adapter.v_q = (const int8_t *)(adapter + offset); offset += 16U * 145U;
    model->adapter.u_scale_f32 = adapter + offset; offset += 80U * 4U;
    model->adapter.u_bias_f32 = adapter + offset; offset += 80U * 4U;
    model->adapter.u_q = (const int8_t *)(adapter + offset); offset += 80U * 16U;
    if (offset != ABC_ONLINE_ADAPTER_BYTES) {
        memset(model, 0, sizeof(*model));
        return ABC_FIXED_ERR_MODEL;
    }
    model->blob = blob;
    model->blob_bytes = combined_bytes;
    return ABC_FIXED_OK;
}

static int adapter_predict(
    const abc_online_adapter_t *adapter,
    const float input[ABC_ONLINE_ADAPTER_INPUT],
    float output[ABC_FIXED_HIDDEN]
) {
    float normalized[ABC_ONLINE_ADAPTER_INPUT];
    float rank[ABC_ONLINE_ADAPTER_RANK];
    unsigned row;
    unsigned column;
    if (adapter == NULL || adapter->blob == NULL || input == NULL || output == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    for (column = 0U; column < ABC_ONLINE_ADAPTER_INPUT; ++column) {
        normalized[column] = (input[column] - read_f16_le(adapter->mean_f16 + 2U * column)) *
            read_f16_le(adapter->invstd_f16 + 2U * column);
    }
    for (row = 0U; row < ABC_ONLINE_ADAPTER_RANK; ++row) {
        float scale = read_f32_le(adapter->v_scale_f32 + 4U * row);
        float sum = adapter_dot_f32(
            adapter->v_q + row * ABC_ONLINE_ADAPTER_INPUT,
            scale, normalized, ABC_ONLINE_ADAPTER_INPUT
        );
        sum += read_f32_le(adapter->v_bias_f32 + 4U * row);
        rank[row] = tanhf(sum);
    }
    for (row = 0U; row < ABC_FIXED_HIDDEN; ++row) {
        float scale = read_f32_le(adapter->u_scale_f32 + 4U * row);
        float sum = adapter_dot_f32(
            adapter->u_q + row * ABC_ONLINE_ADAPTER_RANK,
            scale, rank, ABC_ONLINE_ADAPTER_RANK
        );
        sum = input[row] + sum;
        sum += read_f32_le(adapter->u_bias_f32 + 4U * row);
        output[row] = sum;
    }
    return ABC_FIXED_OK;
}

static int16_t q8_from_double(double value) {
    float rounded_source = (float)value;
    double rounded = nearbyint((double)rounded_source * 256.0);
    if (rounded < -32768.0) return INT16_MIN;
    if (rounded > 32767.0) return INT16_MAX;
    return (int16_t)rounded;
}

static int16_t q15_from_float(float value) {
    double clipped = value;
    double rounded;
    if (clipped < -1.0) clipped = -1.0;
    if (clipped > 1.0) clipped = 1.0;
    rounded = nearbyint(clipped * 32768.0);
    if (rounded < -32767.0) rounded = -32767.0;
    if (rounded > 32767.0) rounded = 32767.0;
    return (int16_t)rounded;
}

static int32_t q16_from_double(double value) {
    double rounded = nearbyint((double)(float)value * 65536.0);
    if (rounded < -2147483648.0) return INT32_MIN;
    if (rounded > 2147483647.0) return INT32_MAX;
    return (int32_t)rounded;
}

static uint16_t magnitude_of(int16_t dx, int16_t dy) {
    int32_t ax = dx < 0 ? -(int32_t)dx : dx;
    int32_t ay = dy < 0 ? -(int32_t)dy : dy;
    return (uint16_t)(ax > ay ? ax : ay);
}

static void insert_active_magnitude(abc_online_renderer_t *r, uint16_t value) {
    uint32_t index = r->active_count;
    while (index > 0U && r->sorted_active_magnitudes[index - 1U] > value) {
        r->sorted_active_magnitudes[index] = r->sorted_active_magnitudes[index - 1U];
        --index;
    }
    r->sorted_active_magnitudes[index] = value;
}

static void update_summaries(
    abc_online_renderer_t *r,
    int16_t dx,
    int16_t dy,
    int16_t summary_q8[5]
) {
    uint16_t magnitude = magnitude_of(dx, dy);
    uint32_t tick = r->observed;
    unsigned axis;
    if (magnitude > 0U) {
        insert_active_magnitude(r, magnitude);
        ++r->active_count;
        r->active_sum += magnitude;
        r->magnitude_sum_squares += (double)magnitude * magnitude;
        if (magnitude > r->running_max) r->running_max = magnitude;
    }
    for (axis = 0U; axis < 2U; ++axis) {
        int16_t value = axis == 0U ? dx : dy;
        int8_t sign = (int8_t)((value > 0) - (value < 0));
        if (sign != 0) {
            if (r->previous_sign[axis] != 0) {
                ++r->sign_comparison_count;
                if (sign != r->previous_sign[axis]) ++r->sign_flip_count;
            }
            r->previous_sign[axis] = sign;
        }
    }
    if (tick > 0U) {
        double delta = (double)magnitude - r->previous_magnitude;
        r->delta_energy += delta * delta;
    }
    r->previous_magnitude = magnitude;
    for (axis = 0U; axis < 7U; ++axis) {
        double real = r->low_dft_oscillator_real[axis];
        double imag = r->low_dft_oscillator_imag[axis];
        r->low_dft_real[axis] += magnitude * real;
        r->low_dft_imag[axis] += magnitude * imag;
        r->low_dft_oscillator_real[axis] =
            real * r->low_dft_twiddle_real[axis] -
            imag * r->low_dft_twiddle_imag[axis];
        r->low_dft_oscillator_imag[axis] =
            real * r->low_dft_twiddle_imag[axis] +
            imag * r->low_dft_twiddle_real[axis];
    }
    r->nyquist += (tick & 1U) ? -(double)magnitude : (double)magnitude;

    summary_q8[0] = q8_from_double((double)r->active_count / (tick + 1U));
    summary_q8[1] = q8_from_double(log1p(
        r->active_count ? r->active_sum / r->active_count : 0.0
    ));
    summary_q8[2] = q8_from_double(log1p((double)r->running_max));
    summary_q8[3] = q8_from_double(
        r->sign_comparison_count
            ? (double)r->sign_flip_count / r->sign_comparison_count
            : 0.0
    );
    summary_q8[4] = q8_from_double(
        (double)(float)0.69508773f * log1p(r->delta_energy)
    );
}

static void build_core15(
    abc_online_core_state_t *state,
    uint16_t logical_tick,
    int16_t raw_x,
    int16_t raw_y,
    const double smooth[2],
    int16_t feature[15]
) {
    double speed = sqrt(smooth[0] * smooth[0] + smooth[1] * smooth[1]);
    double accel = logical_tick == 0U ? 0.0 : speed - state->previous_speed;
    double curvature = 0.0;
    double norm, tx, ty, nx, ny, desired_x, desired_y;
    double acc_t, acc_n, prev_t, prev_n, last_t, last_n, recent_zero;
    uint32_t inactive = 0U;
    uint32_t index;
    uint8_t active = (uint8_t)(raw_x != 0 || raw_y != 0);
    if (logical_tick > 0U && state->previous_speed > 1.0e-9 && speed > 1.0e-9) {
        double cosine = (state->previous_smooth[0] * smooth[0] +
            state->previous_smooth[1] * smooth[1]) / (state->previous_speed * speed);
        if (cosine < -1.0) cosine = -1.0;
        if (cosine > 1.0) cosine = 1.0;
        curvature = 1.0 - cosine;
    }
    if (speed > 1.0e-9) {
        state->last_direction[0] = smooth[0];
        state->last_direction[1] = smooth[1];
    }
    norm = sqrt(state->last_direction[0] * state->last_direction[0] +
        state->last_direction[1] * state->last_direction[1]);
    if (norm <= 1.0e-9) norm = 1.0;
    tx = state->last_direction[0] / norm; ty = state->last_direction[1] / norm;
    nx = -ty; ny = tx;
    state->cumulative_smooth[0] += smooth[0]; state->cumulative_smooth[1] += smooth[1];
    if (logical_tick > 0U) {
        state->cumulative_previous_raw[0] += state->previous_raw[0];
        state->cumulative_previous_raw[1] += state->previous_raw[1];
    }
    desired_x = state->cumulative_smooth[0] - state->cumulative_previous_raw[0];
    desired_y = state->cumulative_smooth[1] - state->cumulative_previous_raw[1];
    acc_t = desired_x * tx + desired_y * ty; acc_n = desired_x * nx + desired_y * ny;
    prev_t = state->previous_raw[0] * tx + state->previous_raw[1] * ty;
    prev_n = state->previous_raw[0] * nx + state->previous_raw[1] * ny;
    last_t = state->last_nonzero[0] * tx + state->last_nonzero[1] * ty;
    last_n = state->last_nonzero[0] * nx + state->last_nonzero[1] * ny;
    for (index = 0U; index < state->ring_count; ++index) inactive += !state->active_ring[index];
    recent_zero = state->ring_count ? (double)inactive / state->ring_count : 1.0;
    feature[0] = q8_from_double(speed / 4.0);
    feature[1] = q8_from_double(fmin(fmax(accel, -3.0), 3.0));
    feature[2] = q8_from_double(curvature); feature[3] = q8_from_double(tx); feature[4] = q8_from_double(ty);
    feature[5] = q8_from_double(fmin(fmax(acc_t, -4.0), 4.0));
    feature[6] = q8_from_double(fmin(fmax(acc_n, -4.0), 4.0));
    feature[7] = q8_from_double(fmin(fmax(prev_t / 4.0, -4.0), 4.0));
    feature[8] = q8_from_double(fmin(fmax(prev_n / 4.0, -4.0), 4.0));
    feature[9] = (int16_t)(state->run_active << 8U);
    feature[10] = q8_from_double((double)(state->run_length > 64U ? 64U : state->run_length) / 64.0);
    feature[11] = q8_from_double(recent_zero); feature[12] = q8_from_double(log1p(speed));
    feature[13] = q8_from_double(fmin(fmax(last_t / 4.0, -4.0), 4.0));
    feature[14] = q8_from_double(fmin(fmax(last_n / 4.0, -4.0), 4.0));
    state->run_length = (state->run_length == 0U || active != state->run_active)
        ? 1U : (uint16_t)(state->run_length + 1U);
    state->run_active = active;
    state->active_ring[state->ring_pos] = active;
    state->ring_pos = (uint8_t)((state->ring_pos + 1U) % 60U);
    if (state->ring_count < 60U) ++state->ring_count;
    state->previous_raw[0] = raw_x; state->previous_raw[1] = raw_y;
    if (active) { state->last_nonzero[0] = raw_x; state->last_nonzero[1] = raw_y; }
    if (raw_x != 0) state->last_axis_nonzero[0] = raw_x;
    if (raw_y != 0) state->last_axis_nonzero[1] = raw_y;
    state->previous_smooth[0] = smooth[0]; state->previous_smooth[1] = smooth[1];
    state->previous_speed = speed;
}

int abc_online_reset(abc_online_renderer_t *r, const abc_online_model_t *model) {
    unsigned index;
    double base_real;
    double base_imag;
    if (r == NULL || model == NULL || model->blob == NULL) return ABC_FIXED_ERR_ARGUMENT;
    memset(r, 0, sizeof(*r)); r->model = model;
    r->observer_core.last_direction[0] = 1.0;
    r->target_core.last_direction[0] = 1.0;
    /* Six half-angle steps derive exp(-i*pi/128) from cos(pi/2)=0. This
     * consumes no packed or hidden lookup data and introduces no trig call. */
    base_real = 0.0;
    base_imag = 0.0;
    for (index = 0U; index < 6U; ++index) {
        base_imag = -sqrt((1.0 - base_real) * 0.5);
        base_real = sqrt((1.0 + base_real) * 0.5);
    }
    for (index = 0U; index < 7U; ++index) {
        if (index == 0U) {
            r->low_dft_twiddle_real[index] = base_real;
            r->low_dft_twiddle_imag[index] = base_imag;
        } else {
            double previous_real = r->low_dft_twiddle_real[index - 1U];
            double previous_imag = r->low_dft_twiddle_imag[index - 1U];
            r->low_dft_twiddle_real[index] =
                previous_real * base_real - previous_imag * base_imag;
            r->low_dft_twiddle_imag[index] =
                previous_real * base_imag + previous_imag * base_real;
        }
        r->low_dft_oscillator_real[index] = 1.0;
    }
    abc_w5_reset(&r->smoother);
    abc_w5_reset(&r->target_smoother);
    return abc_fixed_reset(&r->fixed, &model->fixed);
}

int abc_online_observe_raw(abc_online_renderer_t *r, int16_t dx, int16_t dy) {
    double smooth[2]; int emitted; int status; int16_t summary[5]; unsigned i;
    uint16_t tick;
    if (r == NULL || r->model == NULL) return ABC_FIXED_ERR_ARGUMENT;
    if (r->observed >= 256U) return ABC_FIXED_ERR_MODE;
    tick = r->observed;
    status = abc_fixed_observe_prefix(&r->fixed, dx, dy); if (status) return status;
    update_summaries(r, dx, dy, summary);
    emitted = abc_w5_push_delta(&r->smoother, dx, dy, smooth);
    memset(r->last_observer_feature_q8, 0, 15U * sizeof(int16_t));
    if (emitted) {
        uint16_t logical = (uint16_t)(tick - 4U);
        build_core15(
            &r->observer_core, logical,
            r->fixed.prefix_raw[logical][0], r->fixed.prefix_raw[logical][1],
            smooth, r->last_observer_feature_q8
        );
    }
    if (tick >= 128U) {
        int16_t discarded_feature[15];
        emitted = abc_w5_push_delta(&r->target_smoother, dx, dy, smooth);
        if (emitted) {
            uint16_t logical = (uint16_t)(tick - 128U - 4U);
            uint16_t raw_index = (uint16_t)(128U + logical);
            build_core15(
                &r->target_core, logical,
                r->fixed.prefix_raw[raw_index][0], r->fixed.prefix_raw[raw_index][1],
                smooth, discarded_feature
            );
        }
        if (tick >= 196U) {
            r->canonical_active_tail[tick - 196U] = (uint8_t)(dx != 0 || dy != 0);
        }
    }
    for (i = 0U; i < 5U; ++i) r->last_observer_feature_q8[15U + i] = summary[i];
    status = abc_fixed_online_gru_step_q8(&r->fixed, r->last_observer_feature_q8);
    if (!status) ++r->observed;
    return status;
}

static void final_regime(abc_online_renderer_t *r) {
    double mean = r->active_count ? r->active_sum / r->active_count : 0.0;
    double p95 = 0.0, total_centered, low = 0.0, target, hf;
    unsigned i;
    if (r->active_count == 1U) p95 = r->sorted_active_magnitudes[0];
    else if (r->active_count > 1U) {
        double pos = 0.95 * (r->active_count - 1U); uint32_t k = (uint32_t)pos;
        p95 = r->sorted_active_magnitudes[k] +
            (r->sorted_active_magnitudes[k + 1U] - r->sorted_active_magnitudes[k]) * (pos - k);
    }
    total_centered = 256.0 * (r->magnitude_sum_squares - r->active_sum * r->active_sum / 256.0);
    for (i = 0U; i < 7U; ++i) low += r->low_dft_real[i] * r->low_dft_real[i] +
        r->low_dft_imag[i] * r->low_dft_imag[i];
    target = (total_centered - r->nyquist * r->nyquist) * 0.5 - low;
    hf = target > 0.0 ? target / 256.0 : 0.0;
    r->final_regime_q8[0] = q8_from_double((double)r->active_count / 256.0);
    r->final_regime_q8[1] = q8_from_double(log1p(mean));
    r->final_regime_q8[2] = q8_from_double(log1p(p95));
    r->final_regime_q8[3] = q8_from_double(r->sign_comparison_count ?
        (double)r->sign_flip_count / r->sign_comparison_count : 0.0);
    r->final_regime_q8[4] = q8_from_double(log1p(hf));
}

int abc_online_begin(abc_online_renderer_t *r, uint64_t seed) {
    int status; int16_t corrected[80]; unsigned i;
    double tail[4][2];
    size_t tail_count;
    abc_fixed_online_boundary_t boundary;
    if (r == NULL || r->model == NULL) return ABC_FIXED_ERR_ARGUMENT;
    if (r->observed != 256U) return ABC_FIXED_ERR_MODE;
    final_regime(r);
    tail_count = abc_w5_flush(&r->target_smoother, tail);
    if (tail_count != 4U) return ABC_FIXED_ERR_MODE;
    for (i = 0U; i < 4U; ++i) {
        uint16_t logical = (uint16_t)(124U + i);
        uint16_t raw_index = (uint16_t)(128U + logical);
        build_core15(
            &r->target_core, logical,
            r->fixed.prefix_raw[raw_index][0], r->fixed.prefix_raw[raw_index][1],
            tail[i], r->target_tail4_core_q8[i]
        );
    }
    memset(&boundary, 0, sizeof(boundary));
    memcpy(boundary.regime_q8, r->final_regime_q8, sizeof(boundary.regime_q8));
    for (i = 0U; i < 2U; ++i) {
        boundary.previous_emit[i] = (int16_t)r->target_core.previous_raw[i];
        boundary.last_nonzero[i] = (int16_t)r->target_core.last_nonzero[i];
        boundary.last_axis_nonzero[i] = r->target_core.last_axis_nonzero[i];
        boundary.last_smooth_q16[i] = q16_from_double(r->target_core.previous_smooth[i]);
    }
    boundary.run_length = r->target_core.run_length;
    boundary.run_active = r->target_core.run_active;
    memcpy(boundary.active_ring, r->canonical_active_tail, sizeof(boundary.active_ring));
    boundary.active_ring_count = 60U;
    boundary.active_ring_pos = 0U;
    status = abc_fixed_online_install_boundary(&r->fixed, &boundary);
    if (status) return status;
    for (i = 0U; i < 80U; ++i) r->adapter_input[i] = r->fixed.hidden[i] / 32768.0f;
    for (i = 0U; i < 60U; ++i) r->adapter_input[80U + i] =
        r->target_tail4_core_q8[i / 15U][i % 15U] / 256.0f;
    for (i = 0U; i < 5U; ++i) r->adapter_input[140U + i] = r->final_regime_q8[i] / 256.0f;
    status = adapter_predict(&r->model->adapter, r->adapter_input, r->adapter_output);
    if (status) return status;
    for (i = 0U; i < 80U; ++i) corrected[i] = q15_from_float(r->adapter_output[i]);
    return abc_fixed_online_begin_from_hidden(&r->fixed, corrected, seed);
}

int abc_online_step(abc_online_renderer_t *r, int32_t x, int32_t y, abc_fixed_report_t *out) {
    if (r == NULL) return ABC_FIXED_ERR_ARGUMENT;
    return abc_fixed_step(&r->fixed, x, y, out);
}
