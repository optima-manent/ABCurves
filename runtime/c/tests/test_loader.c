#include "abc_online.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned char *load(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb"); long n; unsigned char *p;
    if (!f || fseek(f, 0, SEEK_END)) return NULL;
    n = ftell(f); if (n < 0 || fseek(f, 0, SEEK_SET)) return NULL;
    p = (unsigned char *)malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) return NULL;
    fclose(f); *bytes = (size_t)n; return p;
}

int main(int argc, char **argv) {
    unsigned char *blob; size_t bytes; abc_online_model_t model;
    abc_online_renderer_t profile;
    abc_online_renderer_t renderer;
    abc_online_renderer_t replica;
    static const int16_t expected[16][2] = {
        {1, 0}, {1, 1}, {1, 0}, {0, 0}, {2, 2}, {1, 0}, {1, 0}, {1, 1},
        {1, 0}, {1, 1}, {1, 0}, {1, 1}, {1, 0}, {1, 1}, {1, 0}, {1, 1}
    };
    abc_fixed_report_t report;
    abc_fixed_report_t replica_report;
    unsigned i;
    if (argc != 2 || !(blob = load(argv[1], &bytes))) return 2;
    if (abc_online_model_size() != sizeof(model) ||
        abc_online_renderer_size() != sizeof(renderer) ||
        abc_online_report_size() != sizeof(abc_fixed_report_t)) return 16;
    if (abc_online_model_init(&model, blob, bytes) != 0) return 3;
    if (model.fixed.blob != blob || model.adapter.blob != blob + ABC_ONLINE_FIXED_BYTES) return 4;
    if (abc_online_model_init(&model, blob, bytes - 1U) != ABC_FIXED_ERR_MODEL) return 5;
    blob[300U] ^= 1U;
    if (abc_online_model_init(&model, blob, bytes) != ABC_FIXED_ERR_CRC) return 19;
    blob[300U] ^= 1U;
    blob[ABC_ONLINE_FIXED_BYTES + 100U] ^= 1U;
    if (abc_online_model_init(&model, blob, bytes) != ABC_FIXED_ERR_MODEL) return 6;
    blob[ABC_ONLINE_FIXED_BYTES + 100U] ^= 1U;
    if (abc_online_model_init(&model, blob, bytes) != ABC_FIXED_OK) return 7;
    if (abc_online_reset(&profile, &model) != ABC_FIXED_OK) return 10;
    if (abc_online_begin(&profile, 0U) != ABC_FIXED_ERR_MODE) return 11;
    for (i = 0U; i < 256U; ++i) {
        if (abc_online_observe_raw(&profile, (int16_t)(i & 1U), 0) != ABC_FIXED_OK) return 12;
    }
    if (abc_online_observe_raw(&profile, 0, 0) != ABC_FIXED_ERR_MODE) return 13;
    renderer = profile;
    replica = profile;
    if (abc_online_begin(&renderer, 123U) != ABC_FIXED_OK) return 14;
    if (abc_online_begin(&renderer, 123U) != ABC_FIXED_ERR_MODE) return 15;
    if (abc_online_begin(&replica, 123U) != ABC_FIXED_OK) return 20;
    if (profile.observed != 256U || profile.fixed.mode != ABC_FIXED_MODE_OBSERVE) return 21;
    for (i = 0U; i < 16U; ++i) {
        if (abc_online_step(&renderer, 65536, 32768, &report) != ABC_FIXED_OK) return 17;
        if (abc_online_step(&replica, 65536, 32768, &replica_report) != ABC_FIXED_OK) return 22;
        if (report.dx != expected[i][0] || report.dy != expected[i][1]) return 18;
        if (replica_report.dx != report.dx || replica_report.dy != report.dy) return 23;
    }
    replica = profile;
    if (abc_online_begin(&replica, 123U) != ABC_FIXED_OK) return 24;
    if (abc_online_step(&replica, 65536, 32768, &replica_report) != ABC_FIXED_OK) return 25;
    if (replica_report.dx != expected[0][0] || replica_report.dy != expected[0][1]) return 26;
    printf("online loader/adapter/API passed blob=%zu model=%zu state=%zu\n",
        bytes, sizeof(model), sizeof(renderer));
    free(blob); return 0;
}
