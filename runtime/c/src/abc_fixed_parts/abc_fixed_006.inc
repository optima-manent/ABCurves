                    mul_q16(
                        hot->logits[row] - lse,
                        model->config[CFG_INV_DIR_Q16]
                    );
            }
        }
    }
    for (row = 0U; row < ABC_FIXED_GRID; ++row) {
        int64_t lateral;
        int32_t excess;
        offset_x = (int)(row / ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
        offset_y = (int)(row % ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
        lateral = (int64_t)offset_x * normal_q15[0] +
            (int64_t)offset_y * normal_q15[1];
        if (lateral < 0) {
            lateral = -lateral;
        }
        excess = (int32_t)lateral - model->config[CFG_LATERAL_FREE_Q15];
        if (excess > 0) {
            hot->logits[row] -= round_shift_even_i64(
                (int64_t)excess * model->config[CFG_AF_Q16],
                ABC_TANGENT_Q + 16U - ABC_GATE_Q
            );
        }
    }
    {
        int32_t maximum = hot->logits[0];
        for (row = 1U; row < ABC_FIXED_GRID; ++row) {
            if (hot->logits[row] > maximum) {
                maximum = hot->logits[row];
            }
        }
        for (row = 0U; row < ABC_FIXED_GRID; ++row) {
            uint32_t exponential = exp_neg_q20(
                model,
                hot->logits[row] - maximum
            );
            hot->logits[row] = (int32_t)exponential;
            total += exponential;
        }
    }

    renderer->last_emit_top23 = (uint32_t)(splitmix64_value(
        renderer->event_seed,
        2U * renderer->generation_tick + 1U
    ) >> 41U);
    renderer->last_offset_top23 = (uint32_t)(splitmix64_value(
        renderer->event_seed,
        2U * renderer->generation_tick + 2U
    ) >> 41U);
    emit = ((uint64_t)(2U * renderer->last_emit_top23 + 1U) << ABC_PROB_Q) <
        (uint64_t)emit_probability * 2U * (UINT64_C(1) << 23U);
    safety =
        abs_i64_u64(renderer->accumulator_q16[0]) >=
            (uint32_t)model->config[CFG_FORCE_RELEASE_Q16] ||
        abs_i64_u64(renderer->accumulator_q16[1]) >=
            (uint32_t)model->config[CFG_FORCE_RELEASE_Q16];
    if (!safety) {
        int quiet =
            abs_i64_u64(smooth_dx_q16) <=
                (uint32_t)model->config[CFG_ZERO_INTENT_Q16] &&
            abs_i64_u64(smooth_dy_q16) <=
                (uint32_t)model->config[CFG_ZERO_INTENT_Q16];
        int low_debt =
            abs_i64_u64(renderer->accumulator_q16[0]) <
                (uint32_t)model->config[CFG_ZERO_DEBT_Q16] &&
            abs_i64_u64(renderer->accumulator_q16[1]) <
                (uint32_t)model->config[CFG_ZERO_DEBT_Q16];
        if (quiet && low_debt) {
            emit = 0;
        }
    } else {
        emit = 1;
    }
    target = (uint64_t)(2U * renderer->last_offset_top23 + 1U) * total;
    for (row = 0U; row < ABC_FIXED_GRID; ++row) {
        cumulative += (uint32_t)hot->logits[row];
        if (target <= cumulative * 2U * (UINT64_C(1) << 23U)) {
            joint = row;
            break;
        }
    }
    mark[0] = (int32_t)round_div_even_i64(
        renderer->accumulator_q16[0],
        UINT64_C(1) << ABC_SMOOTH_Q
    );
    mark[1] = (int32_t)round_div_even_i64(
        renderer->accumulator_q16[1],
        UINT64_C(1) << ABC_SMOOTH_Q
    );
    for (row = 0U; row < 2U; ++row) {
        if (renderer->last_axis_nonzero[row] != 0 && mark[row] != 0 &&
            sign_i32(renderer->last_axis_nonzero[row]) != sign_i32(mark[row]) &&
            abs_i64_u64(renderer->accumulator_q16[row]) <
                (uint32_t)model->config[CFG_HYSTERESIS_Q16]) {
            mark[row] = 0;
        }
    }
    offset_x = (int)(joint / ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
    offset_y = (int)(joint % ABC_FIXED_SIDE) - ABC_FIXED_RADIUS;
    mark[0] += offset_x;
    mark[1] += offset_y;
    mark[0] = clamp_i32(
        mark[0],
        -model->config[CFG_MAX_ABS_COUNT],
        model->config[CFG_MAX_ABS_COUNT]
    );
    mark[1] = clamp_i32(
        mark[1],
        -model->config[CFG_MAX_ABS_COUNT],
        model->config[CFG_MAX_ABS_COUNT]
    );
    if (!emit) {
        mark[0] = 0;
        mark[1] = 0;
    }
    next_acc[0] = (int64_t)renderer->accumulator_q16[0] -
        (int64_t)mark[0] * (INT64_C(1) << ABC_SMOOTH_Q);
    next_acc[1] = (int64_t)renderer->accumulator_q16[1] -
        (int64_t)mark[1] * (INT64_C(1) << ABC_SMOOTH_Q);
    if (next_acc[0] < ABC_I32_MIN_VALUE || next_acc[0] > INT32_MAX ||
        next_acc[1] < ABC_I32_MIN_VALUE || next_acc[1] > INT32_MAX) {
        return ABC_FIXED_ERR_RANGE;
    }
    renderer->accumulator_q16[0] = (int32_t)next_acc[0];
    renderer->accumulator_q16[1] = (int32_t)next_acc[1];
    update_history(renderer, (int16_t)mark[0], (int16_t)mark[1]);
    report->dx = (int16_t)mark[0];
    report->dy = (int16_t)mark[1];
    renderer->last_emit_probability_q15 = (uint32_t)emit_probability;
    renderer->last_joint_index = (uint16_t)joint;
    ++renderer->generation_tick;
    return ABC_FIXED_OK;
}

int abc_fixed_step(
    abc_fixed_renderer_t *renderer,
    int32_t smooth_dx_q16,
    int32_t smooth_dy_q16,
    abc_fixed_report_t *report
) {
    return abc_fixed_step_q16(
        renderer,
        smooth_dx_q16,
        smooth_dy_q16,
        report
    );
}
