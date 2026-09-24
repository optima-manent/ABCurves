/* Exercise private integer kernels without extending the public C API. */
#include "../src/abc_fixed.c"
#include <stdio.h>

static int32_t reference_requant(int64_t dot, int32_t multiplier) {
    int64_t product = dot * multiplier;
    int64_t divisor = INT64_C(2147483648);
    int64_t quotient = product / divisor;
    int64_t remainder = product % divisor;
    int64_t magnitude = remainder < 0 ? -remainder : remainder;
    if (magnitude > divisor / 2 ||
        (magnitude == divisor / 2 && quotient % 2 != 0)) {
        quotient += product < 0 ? -1 : 1;
    }
    return (int32_t)quotient;
}

int main(void) {
    const uint32_t dimensions[] = {ABC_FIXED_FEATURES, ABC_FIXED_HIDDEN};
    const int8_t weights[] = {INT8_MIN, INT8_MAX, -1, 0, 1};
    const int16_t values[] = {INT16_MIN, INT16_MAX, -1, 0, 1};
    const int32_t multipliers[] = {0, 1, 1073741824, INT32_MAX};
    int8_t w[ABC_FIXED_HIDDEN];
    int16_t v[ABC_FIXED_HIDDEN];
    unsigned d, wi, vi, mi, pattern;
    for (d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); ++d) {
        for (wi = 0; wi < sizeof(weights) / sizeof(weights[0]); ++wi) {
            for (vi = 0; vi < sizeof(values) / sizeof(values[0]); ++vi) {
                for (pattern = 0; pattern < 2; ++pattern) {
                    uint32_t i;
                    int64_t dot = 0;
                    for (i = 0; i < dimensions[d]; ++i) {
                        w[i] = weights[wi];
                        v[i] = pattern && (i & 1U) ? values[(vi + 1U) % 5U] : values[vi];
                        dot += (int64_t)w[i] * v[i];
                    }
                    for (mi = 0; mi < sizeof(multipliers) / sizeof(multipliers[0]); ++mi) {
                        if (dot_requant(w, v, dimensions[d], multipliers[mi]) !=
                            reference_requant(dot, multipliers[mi])) return 1;
                    }
                }
            }
        }
    }
    /* Halfway values, both signs and both even/odd quotients. */
    for (wi = 0; wi < 64; ++wi) {
        int32_t value = (int32_t)wi - 32;
        if (requant_q31(value, 1073741824) !=
            reference_requant(value, 1073741824)) return 2;
    }
    puts("fixed dot bounds and ties-to-even requantization passed");
    return 0;
}
