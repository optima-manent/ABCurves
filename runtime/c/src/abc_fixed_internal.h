#ifndef ABC_FIXED_INTERNAL_H
#define ABC_FIXED_INTERNAL_H

#include "abc_fixed.h"

/* Private seams between the packed fixed-point core and abc_online.c.  They
 * are deliberately absent from the public header and hidden in shared builds;
 * the supported integration surface is abc_online_* only. */
int abc_fixed_model_init(abc_fixed_model_t *, const void *, size_t);
int abc_fixed_reset(abc_fixed_renderer_t *, const abc_fixed_model_t *);
int abc_fixed_observe_prefix(abc_fixed_renderer_t *, int16_t, int16_t);
int abc_fixed_step(
    abc_fixed_renderer_t *, int32_t, int32_t, abc_fixed_report_t *
);
uint32_t abc_fixed_crc32(const void *, size_t);
int abc_fixed_online_gru_step_q8(
    abc_fixed_renderer_t *, const int16_t[ABC_FIXED_FEATURES]
);
int abc_fixed_online_prepare_boundary(
    abc_fixed_renderer_t *, const int16_t[5], int16_t[4][15]
);
int abc_fixed_online_begin_from_hidden(
    abc_fixed_renderer_t *, const int16_t[ABC_FIXED_HIDDEN], uint64_t
);
int abc_fixed_online_install_boundary(
    abc_fixed_renderer_t *, const abc_fixed_online_boundary_t *
);

#endif
