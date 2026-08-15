    );
    if (renderer->active_ring_count < ABC_FIXED_RECENT_WINDOW) {
        ++renderer->active_ring_count;
    }
}

static int32_t quantize_q16_from_f32(float value) {
    double rounded = nearbyint((double)value * 65536.0);
    if (rounded < (double)ABC_I32_MIN_VALUE) {
        return ABC_I32_MIN_VALUE;
    }
    if (rounded > (double)INT32_MAX) {
        return INT32_MAX;
    }
    return (int32_t)rounded;
}

static uint64_t isqrt_u64(uint64_t value) {
    uint64_t result = 0U;
    uint64_t bit = UINT64_C(1) << 62U;
    while (bit > value) {
        bit >>= 2U;
    }
    while (bit != 0U) {
        if (value >= result + bit) {
            value -= result + bit;
            result = (result >> 1U) + bit;
        } else {
            result >>= 1U;
        }
        bit >>= 2U;
    }
    return result;
}

static uint32_t speed_q16(int32_t x, int32_t y) {
    uint64_t ax = abs_i64_u64(x);
    uint64_t ay = abs_i64_u64(y);
    uint64_t square = ax * ax + ay * ay;
    uint64_t speed = isqrt_u64(square);
    return speed > UINT32_MAX ? UINT32_MAX : (uint32_t)speed;
}

static void normalize_q15(
    int32_t x_q16,
    int32_t y_q16,
    int16_t tangent[2],
    uint32_t *speed_out
) {
    uint32_t speed = speed_q16(x_q16, y_q16);
    int64_t tx;
    int64_t ty;
    *speed_out = speed;
    if (speed == 0U) {
        tangent[0] = 0;
        tangent[1] = 0;
        return;
    }
    tx = round_div_even_i64((int64_t)x_q16 * 32767, speed);
    ty = round_div_even_i64((int64_t)y_q16 * 32767, speed);
    tangent[0] = clamp_i16(tx, -32767, 32767);
    tangent[1] = clamp_i16(ty, -32767, 32767);
}

static int prepare_prefix_boundary(
    abc_fixed_renderer_t *renderer,
    int16_t tail4_core_q8[4][15]
) {
    uint16_t count = renderer->prefix_count < ABC_FIXED_WARM_WINDOW
        ? renderer->prefix_count
        : ABC_FIXED_WARM_WINDOW;
    uint16_t warm_start = (uint16_t)(renderer->prefix_count - count);
    double cumulative_smooth[2] = {0.0, 0.0};
    double cumulative_previous_raw[2] = {0.0, 0.0};
    double previous_raw[2] = {0.0, 0.0};
    double last_nonzero[2] = {0.0, 0.0};
    double last_direction[2] = {1.0, 0.0};
    double previous_smooth[2] = {0.0, 0.0};
    double previous_speed = 0.0;
    uint16_t tick;
    abc_fixed_hot_scratch_t *hot = &renderer->scratch.hot;

    int16_t saved_hidden[ABC_FIXED_HIDDEN];
    memcpy(saved_hidden, renderer->hidden, sizeof(saved_hidden));
    clear_history(renderer);
    memcpy(renderer->hidden, saved_hidden, sizeof(saved_hidden));
    for (tick = 0U; tick < count; ++tick) {
        double smooth[2];
        double raw[2];
        double speed;
        double accel;
        double curvature = 0.0;
        double norm;
        double tx;
        double ty;
        double nx;
        double ny;
        double desired[2];
        double acc_t;
        double acc_n;
        double prev_t;
        double prev_n;
        double last_t;
        double last_n;
        double recent_zero;
        uint16_t history_start;
        uint16_t history_count;
        uint16_t history_tick;
        uint16_t inactive = 0U;
        uint16_t feature;

        smooth[0] = warm_smooth_value(renderer, warm_start, count, tick, 0U);
        smooth[1] = warm_smooth_value(renderer, warm_start, count, tick, 1U);
        raw[0] = warm_raw_value(renderer, warm_start, tick, 0U);
        raw[1] = warm_raw_value(renderer, warm_start, tick, 1U);
        speed = sqrt(smooth[0] * smooth[0] + smooth[1] * smooth[1]);
        accel = tick == 0U ? 0.0 : speed - previous_speed;
        if (tick > 0U && previous_speed > 1.0e-9 && speed > 1.0e-9) {
            double cosine = (
                previous_smooth[0] * smooth[0] +
                previous_smooth[1] * smooth[1]
            ) / (previous_speed * speed);
            if (cosine > 1.0) {
                cosine = 1.0;
            } else if (cosine < -1.0) {
                cosine = -1.0;
            }
            curvature = 1.0 - cosine;
        }
        if (speed > 1.0e-9) {
            last_direction[0] = smooth[0];
            last_direction[1] = smooth[1];
        }
        norm = sqrt(
            last_direction[0] * last_direction[0] +
            last_direction[1] * last_direction[1]
        );
        if (norm <= 1.0e-9) {
            norm = 1.0;
        }
        tx = last_direction[0] / norm;
        ty = last_direction[1] / norm;
        nx = -ty;
        ny = tx;

        cumulative_smooth[0] += smooth[0];
        cumulative_smooth[1] += smooth[1];
        if (tick > 0U) {
            cumulative_previous_raw[0] += previous_raw[0];
            cumulative_previous_raw[1] += previous_raw[1];
        }
        desired[0] = cumulative_smooth[0] - cumulative_previous_raw[0];
        desired[1] = cumulative_smooth[1] - cumulative_previous_raw[1];
        acc_t = desired[0] * tx + desired[1] * ty;
        acc_n = desired[0] * nx + desired[1] * ny;
        prev_t = previous_raw[0] * tx + previous_raw[1] * ty;
        prev_n = previous_raw[0] * nx + previous_raw[1] * ny;
        last_t = last_nonzero[0] * tx + last_nonzero[1] * ty;
        last_n = last_nonzero[0] * nx + last_nonzero[1] * ny;

        if (tick == 0U) {
            recent_zero = 1.0;
        } else {
            history_start = tick > ABC_FIXED_RECENT_WINDOW
                ? (uint16_t)(tick - ABC_FIXED_RECENT_WINDOW)
                : 0U;
            history_count = (uint16_t)(tick - history_start);
            for (history_tick = history_start; history_tick < tick; ++history_tick) {
                int16_t hx = prefix_value(
                    renderer,
                    (uint16_t)(warm_start + history_tick),
                    0U
                );
                int16_t hy = prefix_value(
                    renderer,
                    (uint16_t)(warm_start + history_tick),
                    1U
                );
                if (hx == 0 && hy == 0) {
                    ++inactive;
                }
            }
            recent_zero = (double)inactive / (double)history_count;
        }

        hot->feature[0] = quantize_q8_f32((float)(speed / 4.0));
        hot->feature[1] = quantize_q8_f32((float)fmin(fmax(accel, -3.0), 3.0));
        hot->feature[2] = quantize_q8_f32((float)curvature);
        hot->feature[3] = quantize_q8_f32((float)tx);
        hot->feature[4] = quantize_q8_f32((float)ty);
        hot->feature[5] = quantize_q8_f32((float)fmin(fmax(acc_t, -4.0), 4.0));
        hot->feature[6] = quantize_q8_f32((float)fmin(fmax(acc_n, -4.0), 4.0));
        hot->feature[7] = quantize_q8_f32((float)fmin(fmax(prev_t / 4.0, -4.0), 4.0));
        hot->feature[8] = quantize_q8_f32((float)fmin(fmax(prev_n / 4.0, -4.0), 4.0));
        hot->feature[9] = (int16_t)(renderer->run_active << ABC_FEATURE_Q);
        hot->feature[10] = quantize_q8_f32((float)(
            (double)(renderer->run_length > 64U ? 64U : renderer->run_length) / 64.0
        ));
        hot->feature[11] = quantize_q8_f32((float)recent_zero);
        hot->feature[12] = quantize_q8_f32((float)log1p(speed));
        hot->feature[13] = quantize_q8_f32((float)fmin(fmax(last_t / 4.0, -4.0), 4.0));
        hot->feature[14] = quantize_q8_f32((float)fmin(fmax(last_n / 4.0, -4.0), 4.0));
        for (feature = 0U; feature < 5U; ++feature) {
            hot->feature[15U + feature] = renderer->regime_q8[feature];
        }
        if (tail4_core_q8 != NULL && tick + 4U >= count) {
            memcpy(tail4_core_q8[tick + 4U - count], hot->feature, 15U * sizeof(int16_t));
        }
        update_history(renderer, (int16_t)raw[0], (int16_t)raw[1]);
        previous_raw[0] = raw[0];
        previous_raw[1] = raw[1];
        if (raw[0] != 0.0 || raw[1] != 0.0) {
            last_nonzero[0] = raw[0];
            last_nonzero[1] = raw[1];
        }
        previous_smooth[0] = smooth[0];
        previous_smooth[1] = smooth[1];
        previous_speed = speed;
    }

    /* Match FixedEvent's canonical direct preload of the active tail. */
    {
        uint16_t tail_count = count < ABC_FIXED_RECENT_WINDOW
            ? count
            : ABC_FIXED_RECENT_WINDOW;
        uint16_t tail_start = (uint16_t)(count - tail_count);
        memset(renderer->active_ring, 0, sizeof(renderer->active_ring));
        for (tick = 0U; tick < tail_count; ++tick) {
            int16_t x = prefix_value(
                renderer,
                (uint16_t)(warm_start + tail_start + tick),
                0U
            );
            int16_t y = prefix_value(
                renderer,
                (uint16_t)(warm_start + tail_start + tick),
                1U
            );
            renderer->active_ring[tick] = (uint8_t)(x != 0 || y != 0);
        }
        renderer->active_ring_count = (uint8_t)tail_count;
        renderer->active_ring_pos = (uint8_t)(tail_count % ABC_FIXED_RECENT_WINDOW);
