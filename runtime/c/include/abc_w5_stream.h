#ifndef ABC_W5_STREAM_H
#define ABC_W5_STREAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Streaming transcription of:
 *   delta -> cumulative path -> centered boxcar-5 -> centered boxcar-5
 *         -> path difference
 * with edge padding at both ends.  A real push can finalize at most one
 * sample, four ticks late.  The final four (or fewer) samples are returned by
 * one bounded flush.
 */
typedef struct abc_w5_stream {
    double cumulative[2];
    double first_path[2];
    double last_path[2];
    double path_history[5][2];
    uint32_t path_history_count;
    uint64_t path_pushes;

    double first_stage_first[2];
    double first_stage_last[2];
    double first_stage_history[5][2];
    uint32_t first_stage_history_count;
    uint64_t first_stage_pushes;

    double previous_smoothed_path[2];
    uint8_t have_previous_smoothed_path;
    uint8_t flushed;
    uint64_t real_ticks;
    uint64_t emitted_ticks;
} abc_w5_stream_t;

void abc_w5_reset(abc_w5_stream_t *stream);

/* Returns 1 and writes one finalized delta, or returns 0 during the 4-tick fill. */
int abc_w5_push_delta(
    abc_w5_stream_t *stream,
    double dx,
    double dy,
    double finalized_dxdy[2]
);

/* Writes chronological tail deltas. Returns 0..4, or SIZE_MAX on misuse. */
size_t abc_w5_flush(
    abc_w5_stream_t *stream,
    double finalized_tail[4][2]
);

#ifdef __cplusplus
}
#endif

#endif
