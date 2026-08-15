            (int64_t)renderer->last_smooth_q16[1] * smooth_y;
        uint64_t denominator =
            (uint64_t)renderer->previous_speed_q16 * speed;
        int32_t cosine_q15 = ratio_q15(dot, denominator);
        if (cosine_q15 < -(1 << ABC_TANGENT_Q)) {
            cosine_q15 = -(1 << ABC_TANGENT_Q);
        } else if (cosine_q15 > (1 << ABC_TANGENT_Q)) {
            cosine_q15 = 1 << ABC_TANGENT_Q;
        }
        curvature_q8 = (int32_t)round_div_even_i64(
            (int64_t)((1 << ABC_TANGENT_Q) - cosine_q15) << ABC_FEATURE_Q,
            UINT64_C(1) << ABC_TANGENT_Q
        );
    }

    tangent_dot =
        (int64_t)renderer->accumulator_q16[0] * renderer->tangent_q15[0] +
        (int64_t)renderer->accumulator_q16[1] * renderer->tangent_q15[1];
    normal_dot =
        (int64_t)renderer->accumulator_q16[0] * normal_q15[0] +
        (int64_t)renderer->accumulator_q16[1] * normal_q15[1];
    previous_tangent_dot =
        (int64_t)renderer->previous_emit[0] * renderer->tangent_q15[0] +
        (int64_t)renderer->previous_emit[1] * renderer->tangent_q15[1];
    previous_normal_dot =
        (int64_t)renderer->previous_emit[0] * normal_q15[0] +
        (int64_t)renderer->previous_emit[1] * normal_q15[1];
    last_tangent_dot =
        (int64_t)renderer->last_nonzero[0] * renderer->tangent_q15[0] +
        (int64_t)renderer->last_nonzero[1] * renderer->tangent_q15[1];
    last_normal_dot =
        (int64_t)renderer->last_nonzero[0] * normal_q15[0] +
        (int64_t)renderer->last_nonzero[1] * normal_q15[1];
    for (index = 0U; index < renderer->active_ring_count; ++index) {
        inactive += renderer->active_ring[index] == 0U;
    }

    hot->feature[0] = clamp_i16(
        round_div_even_i64(speed, UINT64_C(1) << 10U),
        INT16_MIN,
        INT16_MAX
    );
    hot->feature[1] = clamp_i16(
        round_shift_even_i64(accel, ABC_SMOOTH_Q - ABC_FEATURE_Q),
        -768,
        768
    );
    hot->feature[2] = clamp_i16(curvature_q8, INT16_MIN, INT16_MAX);
    hot->feature[3] = (int16_t)round_shift_even_i64(
        renderer->tangent_q15[0],
        ABC_TANGENT_Q - ABC_FEATURE_Q
    );
    hot->feature[4] = (int16_t)round_shift_even_i64(
        renderer->tangent_q15[1],
        ABC_TANGENT_Q - ABC_FEATURE_Q
    );
    hot->feature[5] = clamp_i16(
        round_shift_even_i64(
            tangent_dot,
            ABC_SMOOTH_Q + ABC_TANGENT_Q - ABC_FEATURE_Q
        ),
        -(4 << ABC_FEATURE_Q),
        4 << ABC_FEATURE_Q
    );
    hot->feature[6] = clamp_i16(
        round_shift_even_i64(
            normal_dot,
            ABC_SMOOTH_Q + ABC_TANGENT_Q - ABC_FEATURE_Q
        ),
        -(4 << ABC_FEATURE_Q),
        4 << ABC_FEATURE_Q
    );
    hot->feature[7] = clamp_i16(
        round_shift_even_i64(
            previous_tangent_dot,
            ABC_TANGENT_Q - ABC_FEATURE_Q + 2U
        ),
        -(4 << ABC_FEATURE_Q),
        4 << ABC_FEATURE_Q
    );
    hot->feature[8] = clamp_i16(
        round_shift_even_i64(
            previous_normal_dot,
            ABC_TANGENT_Q - ABC_FEATURE_Q + 2U
        ),
        -(4 << ABC_FEATURE_Q),
        4 << ABC_FEATURE_Q
    );
    hot->feature[9] = (int16_t)(renderer->run_active << ABC_FEATURE_Q);
    hot->feature[10] = (int16_t)(
        (renderer->run_length > 64U ? 64U : renderer->run_length) * 4U
    );
    hot->feature[11] = renderer->active_ring_count == 0U
        ? 256
        : (int16_t)round_div_even_i64(
            (int64_t)inactive << ABC_FEATURE_Q,
            renderer->active_ring_count
        );
    hot->feature[12] = clamp_i16(
        round_shift_even_i64(
            ln1p_q14_from_q16(model, speed),
            ABC_LOG_Q - ABC_FEATURE_Q
        ),
        INT16_MIN,
        INT16_MAX
    );
    hot->feature[13] = clamp_i16(
        round_shift_even_i64(
            last_tangent_dot,
            ABC_TANGENT_Q - ABC_FEATURE_Q + 2U
        ),
        -(4 << ABC_FEATURE_Q),
        4 << ABC_FEATURE_Q
    );
    hot->feature[14] = clamp_i16(
        round_shift_even_i64(
            last_normal_dot,
            ABC_TANGENT_Q - ABC_FEATURE_Q + 2U
        ),
        -(4 << ABC_FEATURE_Q),
        4 << ABC_FEATURE_Q
    );
    for (index = 0U; index < 5U; ++index) {
        hot->feature[15U + index] = renderer->regime_q8[index];
    }
    renderer->last_smooth_q16[0] = smooth_x;
    renderer->last_smooth_q16[1] = smooth_y;
    renderer->previous_speed_q16 = speed;
    return ABC_FIXED_OK;
}

static uint64_t splitmix64_value(uint64_t seed, uint64_t ordinal) {
    uint64_t value = seed + ABC_GAMMA_U64 * ordinal;
    value = (value ^ (value >> 30U)) * ABC_MIX1_U64;
    value = (value ^ (value >> 27U)) * ABC_MIX2_U64;
    return value ^ (value >> 31U);
}

static int sign_i32(int32_t value) {
    return (value > 0) - (value < 0);
}

int abc_fixed_step_q16(
    abc_fixed_renderer_t *renderer,
    int32_t smooth_dx_q16,
    int32_t smooth_dy_q16,
    abc_fixed_report_t *report
) {
    const abc_fixed_model_t *model;
    abc_fixed_hot_scratch_t *hot;
    const int8_t *w_emit;
    const int8_t *w_off;
    int16_t normal_q15[2];
    int status;
    int32_t emit_logit;
    int32_t emit_scaled;
    int32_t emit_probability;
    uint32_t row;
    uint32_t ring;
    uint64_t total = 0U;
    uint64_t target;
    uint64_t cumulative = 0U;
    uint32_t joint = ABC_FIXED_GRID - 1U;
    int emit;
    int safety;
    int32_t mark[2];
    int64_t next_acc[2];
    int offset_x;
    int offset_y;

    if (renderer == NULL || report == NULL || renderer->model == NULL) {
        return ABC_FIXED_ERR_ARGUMENT;
    }
    if (renderer->mode != ABC_FIXED_MODE_GENERATE) {
        return ABC_FIXED_ERR_MODE;
    }
    model = renderer->model;
    hot = &renderer->scratch.hot;
    status = build_future_feature(
        renderer,
        smooth_dx_q16,
        smooth_dy_q16,
        normal_q15
    );
    if (status != ABC_FIXED_OK) {
        return status;
    }
    gru_step(renderer, hot->feature);

    w_emit = model->weights + 24000;
    w_off = model->weights + 24080;
    emit_logit = dot_requant(
        w_emit,
        renderer->hidden,
        ABC_FIXED_HIDDEN,
        read_multiplier(model, 6U)
    ) + read_bias(model, 480U) + model->config[CFG_EMIT_BIAS_Q14];
    emit_scaled = mul_q16(emit_logit, model->config[CFG_INV_EMIT_Q16]);
    emit_probability = sigmoid_q15(model, emit_scaled);
    for (row = 0U; row < ABC_FIXED_GRID; ++row) {
        hot->logits[row] = dot_requant(
            w_off + row * ABC_FIXED_HIDDEN,
            renderer->hidden,
            ABC_FIXED_HIDDEN,
            read_multiplier(model, 7U + row)
        ) + read_bias(model, 481U + row);
    }
    for (ring = 0U; ring <= ABC_FIXED_RADIUS; ++ring) {
        int32_t maximum = ABC_I32_MIN_VALUE;
        uint64_t ring_sum = 0U;
        int32_t lse;
        for (row = 0U; row < ABC_FIXED_GRID; ++row) {
            int x = (int)(row / ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
            int y = (int)(row % ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
            uint32_t row_ring = (uint32_t)(
                (x < 0 ? -x : x) > (y < 0 ? -y : y)
                    ? (x < 0 ? -x : x)
                    : (y < 0 ? -y : y)
            );
            if (row_ring == ring && hot->logits[row] > maximum) {
                maximum = hot->logits[row];
            }
        }
        for (row = 0U; row < ABC_FIXED_GRID; ++row) {
            int x = (int)(row / ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
            int y = (int)(row % ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
            uint32_t row_ring = (uint32_t)(
                (x < 0 ? -x : x) > (y < 0 ? -y : y)
                    ? (x < 0 ? -x : x)
                    : (y < 0 ? -y : y)
            );
            if (row_ring == ring) {
                ring_sum += exp_neg_q20(
                    model,
                    hot->logits[row] - maximum
                );
            }
        }
        lse = maximum + ln_positive_q14(model, ring_sum);
        for (row = 0U; row < ABC_FIXED_GRID; ++row) {
            int x = (int)(row / ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
            int y = (int)(row % ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
            uint32_t row_ring = (uint32_t)(
                (x < 0 ? -x : x) > (y < 0 ? -y : y)
                    ? (x < 0 ? -x : x)
                    : (y < 0 ? -y : y)
            );
            if (row_ring == ring) {
                hot->logits[row] =
                    mul_q16(lse, model->config[CFG_INV_MAG_Q16]) +
