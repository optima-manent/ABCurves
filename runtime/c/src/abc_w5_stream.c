#include "abc_w5_stream.h"

#include <string.h>

static void append_history(
    double history[5][2],
    uint32_t *count,
    const double value[2]
) {
    uint32_t index;
    if (*count < 5U) {
        history[*count][0] = value[0];
        history[*count][1] = value[1];
        *count += 1U;
        return;
    }
    for (index = 0U; index < 4U; ++index) {
        history[index][0] = history[index + 1U][0];
        history[index][1] = history[index + 1U][1];
    }
    history[4][0] = value[0];
    history[4][1] = value[1];
}

static void centered_left_box5(
    const double history[5][2],
    uint64_t pushes,
    double output[2]
) {
    uint32_t axis;
    for (axis = 0U; axis < 2U; ++axis) {
        double sum = 0.0;
        if (pushes == 3U) {
            /* Keep the same left-to-right multiply/add order as
             * abcurves.smoothing.smooth_dxdy.  This is intentional:
             * grouping the repeated edge value as 3*x/5 is not bit exact. */
            sum += history[0][axis] * 0.2;
            sum += history[0][axis] * 0.2;
            sum += history[0][axis] * 0.2;
            sum += history[1][axis] * 0.2;
            sum += history[2][axis] * 0.2;
        } else if (pushes == 4U) {
            sum += history[0][axis] * 0.2;
            sum += history[0][axis] * 0.2;
            sum += history[1][axis] * 0.2;
            sum += history[2][axis] * 0.2;
            sum += history[3][axis] * 0.2;
        } else {
            sum += history[0][axis] * 0.2;
            sum += history[1][axis] * 0.2;
            sum += history[2][axis] * 0.2;
            sum += history[3][axis] * 0.2;
            sum += history[4][axis] * 0.2;
        }
        output[axis] = sum;
    }
}

static int push_first_stage(
    abc_w5_stream_t *stream,
    const double value[2],
    double finalized_dxdy[2]
) {
    double smoothed_path[2];
    uint32_t axis;

    if (stream->first_stage_pushes == 0U) {
        stream->first_stage_first[0] = value[0];
        stream->first_stage_first[1] = value[1];
    }
    stream->first_stage_last[0] = value[0];
    stream->first_stage_last[1] = value[1];
    append_history(
        stream->first_stage_history,
        &stream->first_stage_history_count,
        value
    );
    stream->first_stage_pushes += 1U;
    if (stream->first_stage_pushes < 3U) {
        return 0;
    }

    centered_left_box5(
        stream->first_stage_history,
        stream->first_stage_pushes,
        smoothed_path
    );
    for (axis = 0U; axis < 2U; ++axis) {
        finalized_dxdy[axis] = (double)(float)(
            smoothed_path[axis] - (
                stream->have_previous_smoothed_path
                    ? stream->previous_smoothed_path[axis]
                    : 0.0
            )
        );
        stream->previous_smoothed_path[axis] = smoothed_path[axis];
    }
    stream->have_previous_smoothed_path = 1U;
    stream->emitted_ticks += 1U;
    return 1;
}

static int push_path(
    abc_w5_stream_t *stream,
    const double value[2],
    double finalized_dxdy[2]
) {
    double first_stage_value[2];

    if (stream->path_pushes == 0U) {
        stream->first_path[0] = value[0];
        stream->first_path[1] = value[1];
    }
    stream->last_path[0] = value[0];
    stream->last_path[1] = value[1];
    append_history(stream->path_history, &stream->path_history_count, value);
    stream->path_pushes += 1U;
    if (stream->path_pushes < 3U) {
        return 0;
    }
    centered_left_box5(
        stream->path_history,
        stream->path_pushes,
        first_stage_value
    );
    return push_first_stage(stream, first_stage_value, finalized_dxdy);
}

void abc_w5_reset(abc_w5_stream_t *stream) {
    if (stream != NULL) {
        memset(stream, 0, sizeof(*stream));
    }
}

int abc_w5_push_delta(
    abc_w5_stream_t *stream,
    double dx,
    double dy,
    double finalized_dxdy[2]
) {
    double path[2];
    if (stream == NULL || finalized_dxdy == NULL || stream->flushed) {
        return 0;
    }
    stream->cumulative[0] += dx;
    stream->cumulative[1] += dy;
    path[0] = stream->cumulative[0];
    path[1] = stream->cumulative[1];
    stream->real_ticks += 1U;
    return push_path(stream, path, finalized_dxdy);
}

size_t abc_w5_flush(
    abc_w5_stream_t *stream,
    double finalized_tail[4][2]
) {
    double value[2];
    size_t count = 0U;
    uint32_t index;
    int emitted;

    if (stream == NULL || finalized_tail == NULL || stream->flushed) {
        return SIZE_MAX;
    }
    stream->flushed = 1U;
    if (stream->real_ticks == 0U) {
        return 0U;
    }

    /* Finish the two unavailable right-edge path positions. */
    for (index = 0U; index < 2U; ++index) {
        emitted = push_path(stream, stream->last_path, value);
        if (emitted) {
            if (count >= 4U) {
                return SIZE_MAX;
            }
            finalized_tail[count][0] = value[0];
            finalized_tail[count][1] = value[1];
            count += 1U;
        }
    }

    /* Then edge-pad the already-complete first stage, exactly as offline. */
    for (index = 0U; index < 2U; ++index) {
        emitted = push_first_stage(stream, stream->first_stage_last, value);
        if (emitted) {
            if (count >= 4U) {
                return SIZE_MAX;
            }
            finalized_tail[count][0] = value[0];
            finalized_tail[count][1] = value[1];
            count += 1U;
        }
    }

    if (stream->emitted_ticks != stream->real_ticks) {
        return SIZE_MAX;
    }
    return count;
}
