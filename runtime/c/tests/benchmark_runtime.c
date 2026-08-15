#include "abc_online.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
static double now_us(void) {
    LARGE_INTEGER counter;
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1.0e6 / (double)frequency.QuadPart;
}
#else
#include <time.h>
static double now_us(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1.0e6 + (double)value.tv_nsec / 1.0e3;
}
#endif

static unsigned char *load_blob(const char *path, size_t *bytes) {
    FILE *stream = fopen(path, "rb");
    long length;
    unsigned char *data;
    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) return NULL;
    length = ftell(stream);
    if (length < 0 || fseek(stream, 0, SEEK_SET) != 0) return NULL;
    data = (unsigned char *)malloc((size_t)length);
    if (data == NULL || fread(data, 1, (size_t)length, stream) != (size_t)length) {
        free(data);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    *bytes = (size_t)length;
    return data;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double quantile(double *values, size_t count, double probability) {
    size_t index;
    if (count == 0U) return NAN;
    qsort(values, count, sizeof(*values), compare_double);
    index = (size_t)ceil(probability * (double)count);
    if (index == 0U) index = 1U;
    if (index > count) index = count;
    return values[index - 1U];
}

static void statistics(
    FILE *out, const char *name, double *values, size_t count
) {
    size_t index;
    double sum = 0.0;
    double p50;
    double p95;
    double p99;
    for (index = 0U; index < count; ++index) sum += values[index];
    p50 = quantile(values, count, 0.50);
    p95 = quantile(values, count, 0.95);
    p99 = quantile(values, count, 0.99);
    fprintf(
        out,
        "    \"%s\": {\"samples\": %zu, \"mean_us\": %.9g, "
        "\"p50_us\": %.9g, \"p95_us\": %.9g, \"p99_us\": %.9g}",
        name, count, sum / (double)count, p50, p95, p99
    );
}

int main(int argc, char **argv) {
    enum { OBSERVATIONS = 256, STEPS = 800 };
    size_t bytes;
    unsigned char *blob;
    abc_online_model_t model;
    abc_online_renderer_t profile;
    abc_online_renderer_t renderer;
    abc_fixed_report_t report;
    unsigned cycles = 128U;
    unsigned cycle;
    unsigned tick;
    size_t observe_cursor = 0U;
    size_t step_cursor = 0U;
    double *observe;
    double *prepare;
    double *clone_begin;
    double *step;
    FILE *out;
    const char *processor;
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL.bin OUTPUT.json [cycles]\n", argv[0]);
        return 2;
    }
    if (argc == 4) {
        long parsed = strtol(argv[3], NULL, 10);
        if (parsed < 2 || parsed > 100000) return 3;
        cycles = (unsigned)parsed;
    }
    blob = load_blob(argv[1], &bytes);
    if (blob == NULL || abc_online_model_init(&model, blob, bytes) != ABC_FIXED_OK) return 4;
    observe = (double *)malloc((size_t)cycles * OBSERVATIONS * sizeof(double));
    prepare = (double *)malloc((size_t)cycles * sizeof(double));
    clone_begin = (double *)malloc((size_t)cycles * sizeof(double));
    step = (double *)malloc((size_t)cycles * STEPS * sizeof(double));
    if (observe == NULL || prepare == NULL || clone_begin == NULL || step == NULL) return 5;

    /* Warm code and data caches once. This cycle is deliberately not timed. */
    if (abc_online_reset(&profile, &model) != ABC_FIXED_OK) return 6;
    for (tick = 0U; tick < OBSERVATIONS; ++tick) {
        int16_t dx = (int16_t)(tick % 7U == 0U);
        int16_t dy = (int16_t)-(int)(tick % 11U == 0U);
        if (abc_online_observe_raw(&profile, dx, dy) != ABC_FIXED_OK) return 7;
    }
    renderer = profile;
    if (abc_online_begin(&renderer, 7000U) != ABC_FIXED_OK) return 8;
    for (tick = 0U; tick < STEPS; ++tick) {
        if (abc_online_step(&renderer, 65536, 32768, &report) != ABC_FIXED_OK) return 9;
    }

    for (cycle = 0U; cycle < cycles; ++cycle) {
        double started;
        started = now_us();
        if (abc_online_reset(&profile, &model) != ABC_FIXED_OK) return 6;
        for (tick = 0U; tick < OBSERVATIONS; ++tick) {
            int16_t dx = (int16_t)(tick % 7U == 0U);
            int16_t dy = (int16_t)-(int)(tick % 11U == 0U);
            if (abc_online_observe_raw(&profile, dx, dy) != ABC_FIXED_OK) return 7;
        }
        prepare[cycle] = now_us() - started;

        if (abc_online_reset(&renderer, &model) != ABC_FIXED_OK) return 6;
        for (tick = 0U; tick < OBSERVATIONS; ++tick) {
            int16_t dx = (int16_t)(tick % 7U == 0U);
            int16_t dy = (int16_t)-(int)(tick % 11U == 0U);
            started = now_us();
            if (abc_online_observe_raw(&renderer, dx, dy) != ABC_FIXED_OK) return 7;
            observe[observe_cursor++] = now_us() - started;
        }
        started = now_us();
        renderer = profile;
        if (abc_online_begin(&renderer, (uint64_t)(7001U + cycle)) != ABC_FIXED_OK) return 8;
        clone_begin[cycle] = now_us() - started;
        for (tick = 0U; tick < STEPS; ++tick) {
            started = now_us();
            if (abc_online_step(&renderer, 65536, 32768, &report) != ABC_FIXED_OK) return 9;
            step[step_cursor++] = now_us() - started;
        }
    }
    out = fopen(argv[2], "wb");
    if (out == NULL) return 10;
    fprintf(out, "{\n");
    fprintf(out, "  \"schema\": \"abcurves.native_renderer_benchmark.v2\",\n");
    fprintf(out, "  \"status\": \"host_microbenchmark_not_usb_latency\",\n");
    fprintf(out, "  \"artifact\": {\"bytes\": 44484, \"sha256\": \"8fea217f76c3f501dab9576cbac5cd26970d30d01eedb95da3ca3946a0f52f8b\"},\n");
    fprintf(out, "  \"protocol\": {\"cycles\": %u, \"context_reports_per_cycle\": 256, \"generated_reports_per_cycle\": 800, \"timer\": \"monotonic high-resolution wall clock\", \"includes_timer_overhead\": true},\n", cycles);
#ifdef _MSC_FULL_VER
    fprintf(out, "  \"compiler\": {\"family\": \"MSVC\", \"full_version\": %ld, \"reproducible_link\": true},\n", (long)_MSC_FULL_VER);
#else
    fprintf(out, "  \"compiler\": {\"family\": \"non-MSVC\", \"reproducible_link\": false},\n");
#endif
    processor = getenv("PROCESSOR_IDENTIFIER");
    fprintf(
        out,
        "  \"host\": {\"processor_identifier\": \"%s\"},\n",
        processor == NULL ? "unknown" : processor
    );
    fprintf(out, "  \"measurements\": {\n");
    statistics(out, "prepare_profile_256_off_b_path", prepare, cycles);
    fprintf(out, ",\n");
    statistics(out, "observe_one_report", observe, observe_cursor);
    fprintf(out, ",\n");
    statistics(out, "clone_profile_and_begin", clone_begin, cycles);
    fprintf(out, ",\n");
    statistics(out, "generate_one_report", step, step_cursor);
    fprintf(out, "\n  },\n");
    fprintf(out, "  \"limitations\": [\"Profile preparation is outside the B-critical path.\", \"Host CPU only; excludes Planner, USB/HID transport, application scheduling, and operating-system jitter.\", \"Run this executable on the intended target before making a real-time claim.\"]\n");
    fprintf(out, "}\n");
    fclose(out);
    free(observe);
    free(prepare);
    free(clone_begin);
    free(step);
    free(blob);
    return 0;
}
