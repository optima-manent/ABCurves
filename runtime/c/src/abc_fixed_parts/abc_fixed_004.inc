    }

    renderer->last_smooth_q16[0] = quantize_q16_from_f32((float)previous_smooth[0]);
    renderer->last_smooth_q16[1] = quantize_q16_from_f32((float)previous_smooth[1]);
    normalize_q15(
        renderer->last_smooth_q16[0],
        renderer->last_smooth_q16[1],
        renderer->tangent_q15,
        &renderer->previous_speed_q16
    );
    if (renderer->tangent_q15[0] == 0 && renderer->tangent_q15[1] == 0) {
        renderer->tangent_q15[0] = 32767;
    }
    memset(renderer->accumulator_q16, 0, sizeof(renderer->accumulator_q16));
    return count;
}

int abc_fixed_reset(
    abc_fixed_renderer_t *renderer,
    const abc_fixed_model_t *model
) {
    if (renderer == NULL || model == NULL || model->blob == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    memset(renderer, 0, sizeof(*renderer));
    renderer->model = model;
    renderer->mode = ABC_FIXED_MODE_OBSERVE;
    renderer->tangent_q15[0] = 32767;
    return ABC_FIXED_OK;
}

int abc_fixed_observe_prefix(
    abc_fixed_renderer_t *renderer,
    int16_t dx,
    int16_t dy
) {
    if (renderer == NULL || renderer->model == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    if (renderer->mode != ABC_FIXED_MODE_OBSERVE) {
        return ABC_FIXED_ERR_MODE;
    }
    renderer->prefix_raw[renderer->prefix_pos][0] = dx;
    renderer->prefix_raw[renderer->prefix_pos][1] = dy;
    renderer->prefix_pos = (uint16_t)(
        (renderer->prefix_pos + 1U) % ABC_FIXED_PREFIX_WINDOW
    );
    if (renderer->prefix_count < ABC_FIXED_PREFIX_WINDOW) {
        ++renderer->prefix_count;
    }
    ++renderer->observed_ticks;
    return ABC_FIXED_OK;
}

int abc_fixed_online_gru_step_q8(
    abc_fixed_renderer_t *renderer,
    const int16_t feature_q8[ABC_FIXED_FEATURES]
) {
    if (renderer == NULL || renderer->model == NULL || feature_q8 == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    if (renderer->mode != ABC_FIXED_MODE_OBSERVE) {
        return ABC_FIXED_ERR_MODE;
    }
    memcpy(renderer->scratch.hot.feature, feature_q8, sizeof(renderer->scratch.hot.feature));
    gru_step(renderer, renderer->scratch.hot.feature);
    return ABC_FIXED_OK;
}

int abc_fixed_online_prepare_boundary(
    abc_fixed_renderer_t *renderer,
    const int16_t regime_q8[5],
    int16_t tail4_core_q8[4][15]
) {
    if (renderer == NULL || renderer->model == NULL || regime_q8 == NULL ||
        tail4_core_q8 == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    if (renderer->mode != ABC_FIXED_MODE_OBSERVE ||
        renderer->prefix_count != ABC_FIXED_PREFIX_WINDOW) {
        return ABC_FIXED_ERR_MODE;
    }
    memcpy(renderer->regime_q8, regime_q8, sizeof(renderer->regime_q8));
    return prepare_prefix_boundary(renderer, tail4_core_q8);
}

int abc_fixed_online_begin_from_hidden(
    abc_fixed_renderer_t *renderer,
    const int16_t hidden_q15[ABC_FIXED_HIDDEN],
    uint64_t event_seed
) {
    if (renderer == NULL || renderer->model == NULL || hidden_q15 == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    if (renderer->mode != ABC_FIXED_MODE_OBSERVE ||
        renderer->prefix_count != ABC_FIXED_PREFIX_WINDOW) {
        return ABC_FIXED_ERR_MODE;
    }
    memcpy(renderer->hidden, hidden_q15, sizeof(renderer->hidden));
    memset(renderer->accumulator_q16, 0, sizeof(renderer->accumulator_q16));
    renderer->event_seed = event_seed;
    renderer->generation_tick = 0U;
    renderer->mode = ABC_FIXED_MODE_GENERATE;
    return ABC_FIXED_OK;
}

int abc_fixed_online_install_boundary(
    abc_fixed_renderer_t *renderer,
    const abc_fixed_online_boundary_t *boundary
) {
    if (renderer == NULL || renderer->model == NULL || boundary == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    if (renderer->mode != ABC_FIXED_MODE_OBSERVE ||
        renderer->prefix_count != ABC_FIXED_PREFIX_WINDOW ||
        boundary->active_ring_count > ABC_FIXED_RECENT_WINDOW ||
        boundary->active_ring_pos >= ABC_FIXED_RECENT_WINDOW) {
        return ABC_FIXED_ERR_MODE;
    }
    memcpy(renderer->regime_q8, boundary->regime_q8, sizeof(renderer->regime_q8));
    memcpy(renderer->previous_emit, boundary->previous_emit, sizeof(renderer->previous_emit));
    memcpy(renderer->last_nonzero, boundary->last_nonzero, sizeof(renderer->last_nonzero));
    memcpy(renderer->last_axis_nonzero, boundary->last_axis_nonzero, sizeof(renderer->last_axis_nonzero));
    memcpy(renderer->last_smooth_q16, boundary->last_smooth_q16, sizeof(renderer->last_smooth_q16));
    renderer->run_length = boundary->run_length;
    renderer->run_active = boundary->run_active;
    memcpy(renderer->active_ring, boundary->active_ring, sizeof(renderer->active_ring));
    renderer->active_ring_pos = boundary->active_ring_pos;
    renderer->active_ring_count = boundary->active_ring_count;
    normalize_q15(
        renderer->last_smooth_q16[0], renderer->last_smooth_q16[1],
        renderer->tangent_q15, &renderer->previous_speed_q16
    );
    if (renderer->tangent_q15[0] == 0 && renderer->tangent_q15[1] == 0) {
        renderer->tangent_q15[0] = 32767;
    }
    memset(renderer->accumulator_q16, 0, sizeof(renderer->accumulator_q16));
    return ABC_FIXED_OK;
}

static int32_t mul_q16(int32_t value_q14, int32_t factor_q16) {
    return round_shift_even_i64((int64_t)value_q14 * factor_q16, 16U);
}

static int32_t ratio_q15(int64_t numerator, uint64_t denominator) {
    uint64_t remainder;
    uint64_t quotient;
    unsigned bit;
    int round_up;
    int64_t result;
    int negative = numerator < 0;
    if (denominator == 0U) {
        return 0;
    }
    remainder = abs_i64_u64(numerator);
    quotient = remainder / denominator;
    remainder %= denominator;
    for (bit = 0U; bit < ABC_TANGENT_Q; ++bit) {
        quotient <<= 1U;
        if (remainder >= denominator - remainder) {
            remainder -= denominator - remainder;
            quotient |= 1U;
        } else {
            remainder <<= 1U;
        }
    }
    round_up = remainder > denominator - remainder ||
        (remainder == denominator - remainder && (quotient & 1U));
    quotient += (uint64_t)round_up;
    result = (int64_t)quotient;
    return (int32_t)(negative ? -result : result);
}

static int build_future_feature(
    abc_fixed_renderer_t *renderer,
    int32_t smooth_x,
    int32_t smooth_y,
    int16_t normal_q15[2]
) {
    abc_fixed_hot_scratch_t *hot = &renderer->scratch.hot;
    const abc_fixed_model_t *model = renderer->model;
    int64_t next_acc_x = (int64_t)renderer->accumulator_q16[0] + smooth_x;
    int64_t next_acc_y = (int64_t)renderer->accumulator_q16[1] + smooth_y;
    uint32_t speed;
    int16_t new_tangent[2];
    int32_t accel;
    int32_t curvature_q8 = 0;
    int64_t tangent_dot;
    int64_t normal_dot;
    int64_t previous_tangent_dot;
    int64_t previous_normal_dot;
    int64_t last_tangent_dot;
    int64_t last_normal_dot;
    uint32_t inactive = 0U;
    uint32_t index;

    if (next_acc_x < ABC_I32_MIN_VALUE || next_acc_x > INT32_MAX ||
        next_acc_y < ABC_I32_MIN_VALUE || next_acc_y > INT32_MAX) {
        return ABC_FIXED_ERR_RANGE;
    }
    renderer->accumulator_q16[0] = (int32_t)next_acc_x;
    renderer->accumulator_q16[1] = (int32_t)next_acc_y;
    normalize_q15(smooth_x, smooth_y, new_tangent, &speed);
    if (speed != 0U) {
        renderer->tangent_q15[0] = new_tangent[0];
        renderer->tangent_q15[1] = new_tangent[1];
    }
    normal_q15[0] = (int16_t)-renderer->tangent_q15[1];
    normal_q15[1] = renderer->tangent_q15[0];

    accel = (int32_t)((int64_t)speed - renderer->previous_speed_q16);
    if (accel < -(3 << ABC_SMOOTH_Q)) {
        accel = -(3 << ABC_SMOOTH_Q);
    } else if (accel > (3 << ABC_SMOOTH_Q)) {
        accel = 3 << ABC_SMOOTH_Q;
    }
    if (renderer->previous_speed_q16 > 0U && speed > 0U) {
        int64_t dot =
            (int64_t)renderer->last_smooth_q16[0] * smooth_x +
