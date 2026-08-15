static double warm_raw_value(
    const abc_fixed_renderer_t *renderer,
    uint16_t warm_start,
    uint16_t index,
    unsigned axis
) {
    return (double)prefix_value(renderer, (uint16_t)(warm_start + index), axis);
}

static double warm_path_value(
    const abc_fixed_renderer_t *renderer,
    uint16_t warm_start,
    uint16_t index,
    unsigned axis
) {
    uint16_t tick;
    double total = 0.0;
    for (tick = 0U; tick <= index; ++tick) {
        total += warm_raw_value(renderer, warm_start, tick, axis);
    }
    return total;
}

static uint16_t edge_index(int index, uint16_t count) {
    if (index < 0) {
        return 0U;
    }
    if (index >= (int)count) {
        return (uint16_t)(count - 1U);
    }
    return (uint16_t)index;
}

static double warm_once_value(
    const abc_fixed_renderer_t *renderer,
    uint16_t warm_start,
    uint16_t count,
    int index,
    unsigned axis
) {
    double sum = 0.0;
    int tap;
    uint16_t centered = edge_index(index, count);
    for (tap = 0; tap < 5; ++tap) {
        uint16_t source = edge_index((int)centered + tap - 2, count);
        sum += warm_path_value(renderer, warm_start, source, axis) * 0.2;
    }
    return sum;
}

static double warm_twice_value(
    const abc_fixed_renderer_t *renderer,
    uint16_t warm_start,
    uint16_t count,
    int index,
    unsigned axis
) {
    double sum = 0.0;
    int tap;
    uint16_t centered = edge_index(index, count);
    for (tap = 0; tap < 5; ++tap) {
        uint16_t source = edge_index((int)centered + tap - 2, count);
        sum += warm_once_value(
            renderer,
            warm_start,
            count,
            (int)source,
            axis
        ) * 0.2;
    }
    return sum;
}

static double warm_smooth_value(
    const abc_fixed_renderer_t *renderer,
    uint16_t warm_start,
    uint16_t count,
    uint16_t tick,
    unsigned axis
) {
    double current = warm_twice_value(
        renderer,
        warm_start,
        count,
        (int)tick,
        axis
    );
    double previous = tick == 0U ? 0.0 : warm_twice_value(
        renderer,
        warm_start,
        count,
        (int)(tick - 1U),
        axis
    );
    return (double)(float)(current - previous);
}

static void clear_history(abc_fixed_renderer_t *renderer) {
    memset(renderer->hidden, 0, sizeof(renderer->hidden));
    memset(renderer->accumulator_q16, 0, sizeof(renderer->accumulator_q16));
    memset(renderer->previous_emit, 0, sizeof(renderer->previous_emit));
    memset(renderer->last_nonzero, 0, sizeof(renderer->last_nonzero));
    memset(renderer->last_axis_nonzero, 0, sizeof(renderer->last_axis_nonzero));
    memset(renderer->active_ring, 0, sizeof(renderer->active_ring));
    renderer->run_active = 0U;
    renderer->run_length = 0U;
    renderer->active_ring_pos = 0U;
    renderer->active_ring_count = 0U;
}

static void update_history(
    abc_fixed_renderer_t *renderer,
    int16_t dx,
    int16_t dy
) {
    uint8_t active = (uint8_t)(dx != 0 || dy != 0);
    if (renderer->run_length == 0U || active != renderer->run_active) {
        renderer->run_length = 1U;
    } else if (renderer->run_length != UINT16_MAX) {
        ++renderer->run_length;
    }
    renderer->run_active = active;
    renderer->previous_emit[0] = dx;
    renderer->previous_emit[1] = dy;
    if (active) {
        renderer->last_nonzero[0] = dx;
        renderer->last_nonzero[1] = dy;
    }
    if (dx != 0) {
        renderer->last_axis_nonzero[0] = dx;
    }
    if (dy != 0) {
        renderer->last_axis_nonzero[1] = dy;
    }
    renderer->active_ring[renderer->active_ring_pos] = active;
    renderer->active_ring_pos = (uint8_t)(
        (renderer->active_ring_pos + 1U) % ABC_FIXED_RECENT_WINDOW
