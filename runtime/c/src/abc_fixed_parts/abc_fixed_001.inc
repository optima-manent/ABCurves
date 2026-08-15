    model->log_le = blob + section_offsets[6];
    for (index = 0U; index < 11U; ++index) {
        model->config[index] = read_i32_le(blob + 136U + 4U * index);
    }
    model->body_crc32 = crc;
    return ABC_FIXED_OK;
}

static int32_t lerp_i32(
    int32_t a,
    int32_t b,
    uint32_t remainder,
    unsigned interval_q
) {
    int64_t delta = (int64_t)b - (int64_t)a;
    return a + (int32_t)round_div_even_i64(
        delta * (int64_t)remainder,
        UINT64_C(1) << interval_q
    );
}

static int32_t sigmoid_q15(const abc_fixed_model_t *model, int32_t value_q14) {
    uint32_t position;
    uint32_t index;
    uint32_t remainder;
    int32_t a;
    int32_t b;
    if (value_q14 < -ABC_LUT_DOMAIN_Q14) {
        value_q14 = -ABC_LUT_DOMAIN_Q14;
    } else if (value_q14 > ABC_LUT_DOMAIN_Q14) {
        value_q14 = ABC_LUT_DOMAIN_Q14;
    }
    position = (uint32_t)(value_q14 + ABC_LUT_DOMAIN_Q14);
    index = position >> (ABC_GATE_Q - 4);
    if (index >= ABC_LUT_INTERVALS) {
        return (int32_t)read_u16_le(model->sigmoid_le + 2U * ABC_LUT_INTERVALS);
    }
    remainder = position & ((1U << (ABC_GATE_Q - 4)) - 1U);
    a = (int32_t)read_u16_le(model->sigmoid_le + 2U * index);
    b = (int32_t)read_u16_le(model->sigmoid_le + 2U * (index + 1U));
    return lerp_i32(a, b, remainder, ABC_GATE_Q - 4);
}

static int32_t tanh_q15(const abc_fixed_model_t *model, int32_t value_q14) {
    uint32_t position;
    uint32_t index;
    uint32_t remainder;
    int32_t a;
    int32_t b;
    if (value_q14 < -ABC_LUT_DOMAIN_Q14) {
        value_q14 = -ABC_LUT_DOMAIN_Q14;
    } else if (value_q14 > ABC_LUT_DOMAIN_Q14) {
        value_q14 = ABC_LUT_DOMAIN_Q14;
    }
    position = (uint32_t)(value_q14 + ABC_LUT_DOMAIN_Q14);
    index = position >> (ABC_GATE_Q - 4);
    if (index >= ABC_LUT_INTERVALS) {
        return (int32_t)read_i16_le(model->tanh_le + 2U * ABC_LUT_INTERVALS);
    }
    remainder = position & ((1U << (ABC_GATE_Q - 4)) - 1U);
    a = (int32_t)read_i16_le(model->tanh_le + 2U * index);
    b = (int32_t)read_i16_le(model->tanh_le + 2U * (index + 1U));
    return lerp_i32(a, b, remainder, ABC_GATE_Q - 4);
}

static uint32_t exp_neg_q20(const abc_fixed_model_t *model, int32_t value_q14) {
    uint32_t magnitude;
    uint32_t index;
    uint32_t remainder;
    int32_t a;
    int32_t b;
    int64_t mag64 = -(int64_t)value_q14;
    if (mag64 < 0) {
        mag64 = 0;
    } else if (mag64 > ABC_EXP_DOMAIN_Q14) {
        mag64 = ABC_EXP_DOMAIN_Q14;
    }
    magnitude = (uint32_t)mag64;
    index = magnitude >> (ABC_GATE_Q - 4);
    if (index >= ABC_LUT_INTERVALS) {
        return read_u32_le(model->exp_le + 4U * ABC_LUT_INTERVALS);
    }
    remainder = magnitude & ((1U << (ABC_GATE_Q - 4)) - 1U);
    a = (int32_t)read_u32_le(model->exp_le + 4U * index);
    b = (int32_t)read_u32_le(model->exp_le + 4U * (index + 1U));
    return (uint32_t)lerp_i32(a, b, remainder, ABC_GATE_Q - 4);
}

static int32_t ln_positive_q14(const abc_fixed_model_t *model, uint64_t value_q20) {
    uint64_t value = value_q20;
    unsigned exponent = 0U;
    uint64_t mantissa_q24;
    uint32_t fraction_q24;
    uint32_t index;
    uint32_t remainder;
    int32_t a;
    int32_t b;
    int32_t log_mantissa;
    const int32_t ln2_q14 = 11357;
    if (value == 0U) {
        return -ABC_EXP_DOMAIN_Q14;
    }
    while (exponent < 63U && (value >> (exponent + 1U)) != 0U) {
        ++exponent;
    }
    if (exponent >= 24U) {
        mantissa_q24 = value >> (exponent - 24U);
    } else {
        mantissa_q24 = value << (24U - exponent);
    }
    fraction_q24 = (uint32_t)(mantissa_q24 - (UINT64_C(1) << 24U));
    index = fraction_q24 >> 16U;
    if (index >= ABC_LUT_INTERVALS) {
        log_mantissa = read_i16_le(model->log_le + 2U * ABC_LUT_INTERVALS);
    } else {
        remainder = fraction_q24 & 0xffffU;
        a = read_i16_le(model->log_le + 2U * index);
        b = read_i16_le(model->log_le + 2U * (index + 1U));
        log_mantissa = lerp_i32(a, b, remainder, 16U);
    }
    return log_mantissa + ((int32_t)exponent - ABC_EXP_Q) * ln2_q14;
}

static int32_t ln1p_q14_from_q16(
    const abc_fixed_model_t *model,
    uint32_t value_q16
) {
    uint64_t shifted = ((uint64_t)value_q16 + (UINT64_C(1) << 16U)) << 4U;
    return ln_positive_q14(model, shifted);
}

static int32_t read_bias(const abc_fixed_model_t *model, uint32_t index) {
    return read_i32_le(model->biases_le + 4U * index);
}

static int32_t read_multiplier(const abc_fixed_model_t *model, uint32_t index) {
    return read_i32_le(model->multipliers_le + 4U * index);
}

static int32_t requant_q31(int32_t accumulator, int32_t multiplier) {
    int64_t product = (int64_t)accumulator * (int64_t)multiplier;
    return (int32_t)round_div_even_i64(product, ABC_Q31);
}

static int32_t dot_requant(
    const int8_t *weights,
    const int16_t *vector,
    uint32_t columns,
    int32_t multiplier
) {
    int64_t accumulator = 0;
    uint32_t column;
    for (column = 0U; column < columns; ++column) {
        accumulator += (int32_t)weights[column] * (int32_t)vector[column];
    }
    return requant_q31((int32_t)accumulator, multiplier);
}

static void gru_step(
    abc_fixed_renderer_t *renderer,
    const int16_t feature[ABC_FIXED_FEATURES]
) {
    const abc_fixed_model_t *model = renderer->model;
    abc_fixed_hot_scratch_t *hot = &renderer->scratch.hot;
    const int8_t *w_ih = model->weights;
    const int8_t *w_hh = w_ih + 4800;
    uint32_t index;
    for (index = 0U; index < ABC_FIXED_HIDDEN; ++index) {
        uint32_t reset_row = index;
        uint32_t update_row = ABC_FIXED_HIDDEN + index;
        uint32_t new_row = 2U * ABC_FIXED_HIDDEN + index;
        int32_t reset_input = dot_requant(
            w_ih + reset_row * ABC_FIXED_FEATURES,
            feature,
            ABC_FIXED_FEATURES,
            read_multiplier(model, 0U)
        ) + read_bias(model, reset_row);
        int32_t reset_hidden = dot_requant(
            w_hh + reset_row * ABC_FIXED_HIDDEN,
            renderer->hidden,
            ABC_FIXED_HIDDEN,
            read_multiplier(model, 3U)
        ) + read_bias(model, ABC_FIXED_GATES + reset_row);
        int32_t update_input = dot_requant(
            w_ih + update_row * ABC_FIXED_FEATURES,
            feature,
            ABC_FIXED_FEATURES,
            read_multiplier(model, 1U)
        ) + read_bias(model, update_row);
        int32_t update_hidden = dot_requant(
            w_hh + update_row * ABC_FIXED_HIDDEN,
            renderer->hidden,
            ABC_FIXED_HIDDEN,
            read_multiplier(model, 4U)
        ) + read_bias(model, ABC_FIXED_GATES + update_row);
        int32_t new_input = dot_requant(
            w_ih + new_row * ABC_FIXED_FEATURES,
            feature,
            ABC_FIXED_FEATURES,
            read_multiplier(model, 2U)
        ) + read_bias(model, new_row);
        int32_t new_hidden = dot_requant(
            w_hh + new_row * ABC_FIXED_HIDDEN,
            renderer->hidden,
            ABC_FIXED_HIDDEN,
            read_multiplier(model, 5U)
        ) + read_bias(model, ABC_FIXED_GATES + new_row);
        int32_t r = sigmoid_q15(model, reset_input + reset_hidden);
        int32_t z = sigmoid_q15(model, update_input + update_hidden);
        int32_t reset_hg = (int32_t)round_div_even_i64(
            (int64_t)r * new_hidden,
            UINT64_C(1) << ABC_PROB_Q
        );
        int32_t n = tanh_q15(model, new_input + reset_hg);
        int64_t mixed =
            (int64_t)((1U << ABC_PROB_Q) - z) * n +
            (int64_t)z * renderer->hidden[index];
        int32_t value = (int32_t)round_div_even_i64(
            mixed,
            UINT64_C(1) << ABC_PROB_Q
        );
        hot->next_hidden[index] = clamp_i16(value, -32767, 32767);
    }
    memcpy(renderer->hidden, hot->next_hidden, sizeof(renderer->hidden));
}

static int16_t prefix_value(
    const abc_fixed_renderer_t *renderer,
    uint16_t chronological_index,
    unsigned axis
) {
    uint16_t oldest = renderer->prefix_count == ABC_FIXED_PREFIX_WINDOW
        ? renderer->prefix_pos
        : 0U;
    uint16_t physical = (uint16_t)(
        (oldest + chronological_index) % ABC_FIXED_PREFIX_WINDOW
    );
    return renderer->prefix_raw[physical][axis];
}
