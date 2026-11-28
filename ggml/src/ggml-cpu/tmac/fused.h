#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <cstring>

extern "C" {
int32_t partial_max_reset(void * scales);
int32_t partial_max_g4_int8_k8(void * scales, void * input);
}
int32_t lut_ctor_g4_int8_k0_b4(int32_t k, int8_t * lut, void * input, void * scales, void * biases);

#if defined(__clang__)
#define GGML_TMAC_UNROLL_ROWS _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define GGML_TMAC_UNROLL_ROWS _Pragma("GCC unroll 2")
#else
#define GGML_TMAC_UNROLL_ROWS
#endif

namespace ggml_tmac {

struct row {
    const int8_t * lut;
    const float16_t * scales;
    const float16_t * biases;
    float * output;
};

inline int16x8_t add_low(int16x8_t sum, int8x16_t values) {
#if defined(__GNUC__) && !defined(__clang__)
    // GCC 10 copies the low half before the equivalent intrinsic.
    int16x8_t result;
    __asm__("saddw %0.8h, %1.8h, %2.8b" : "=w"(result) : "w"(sum), "w"(values));
    return result;
#else
    return vaddw_s8(sum, vget_low_s8(values));
#endif
}

inline void pack(int k, const uint8_t * weights, const float16_t * scales, uint8_t * packed, float16_t * packed_scales) {
    const int groups = k / 32;
    for (int block = 0; block < 8; ++block) {
        for (int group = 0; group < groups; ++group) {
            const size_t from = size_t(group) * 8 + block;
            const size_t to = size_t(block) * groups + group;
            std::memcpy(packed + to * 128, weights + from * 128, 128);
            std::memcpy(packed_scales + to * 8, scales + from * 8, 8 * sizeof(float16_t));
        }
    }
}

inline void prepare(int groups, float16_t * input, int8_t * lut, float16_t * scales, float16_t * biases) {
    for (int group = 0; group < groups; ++group) {
        partial_max_reset(scales + group);
        partial_max_g4_int8_k8(scales + group, input + group * 32);
    }
    for (int group = 0; group < groups; ++group) {
        lut_ctor_g4_int8_k0_b4(32, lut + group * 128, input + group * 32, scales + group, biases + group);
    }
}

// The frozen layout has 256 bit-plane entries for 64 output channels.
template<int Rows>
inline void fused(int k, const uint8_t * weights, const float16_t * scales, const row (&input)[Rows]) {
    static_assert(Rows == 1 || Rows == 2, "unsupported T-MAC row count");
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int block = 0; block < 8; ++block) {
        float16x8_t partial[Rows][4];
        GGML_TMAC_UNROLL_ROWS
        for (int r = 0; r < Rows; ++r) {
            for (int plane = 0; plane < 4; ++plane) {
                partial[r][plane] = vdupq_n_f16(0);
            }
        }
        for (int group = 0; group < k / 32; ++group) {
            int16x8_t sums[Rows][4];
            GGML_TMAC_UNROLL_ROWS
            for (int r = 0; r < Rows; ++r) {
                for (int plane = 0; plane < 4; ++plane) {
                    sums[r][plane] = vdupq_n_s16(0);
                }
            }
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
// Limit live lookup temporaries on GCC 10.
#pragma GCC unroll 2
#endif
            for (int j = 0; j < 8; ++j) {
                const uint8x16_t packed = vld1q_u8(weights + (size_t(block) * (k / 32) + group) * 128 + j * 16);
                const uint8x16_t low = vandq_u8(packed, mask);
                const uint8x16_t high = vshrq_n_u8(packed, 4);
                GGML_TMAC_UNROLL_ROWS
                for (int r = 0; r < Rows; ++r) {
                    const int8x16_t table = vld1q_s8(input[r].lut + group * 128 + j * 16);
                    const int8x16_t bottom = vqtbl1q_s8(table, low);
                    const int8x16_t top = vqtbl1q_s8(table, high);
                    sums[r][0] = add_low(sums[r][0], bottom);
                    sums[r][1] = vaddw_high_s8(sums[r][1], bottom);
                    sums[r][2] = add_low(sums[r][2], top);
                    sums[r][3] = vaddw_high_s8(sums[r][3], top);
                }
            }
            const float16x8_t scale = vld1q_f16(scales + (size_t(block) * (k / 32) + group) * 8);
            GGML_TMAC_UNROLL_ROWS
            for (int r = 0; r < Rows; ++r) {
                const float16x8_t lut_scale = vdupq_n_f16(input[r].scales[group]);
                const float16x8_t bias = vdupq_n_f16(input[r].biases[group]);
                const float16x8_t v0 = vcvtq_f16_s16(sums[r][0]) * lut_scale + bias;
                const float16x8_t v1 = vcvtq_f16_s16(sums[r][1]) * lut_scale;
                const float16x8_t v2 = vcvtq_f16_s16(sums[r][2]) * lut_scale;
                const float16x8_t v3 = vcvtq_f16_s16(sums[r][3]) * lut_scale;
                partial[r][0] = partial[r][0] + v0 * scale;
                partial[r][1] = partial[r][1] + v1 * scale;
                partial[r][2] = partial[r][2] + v2 * scale;
                partial[r][3] = partial[r][3] + v3 * scale;
            }
        }
        GGML_TMAC_UNROLL_ROWS
        for (int r = 0; r < Rows; ++r) {
#if defined(TMAC_HALF_STORAGE_FALLBACK)
            const float32x4_t low = ((vcvt_f32_f16(vget_low_f16(partial[r][0])) * 0.5f + vcvt_f32_f16(vget_low_f16(partial[r][1]))) + vcvt_f32_f16(vget_low_f16(partial[r][2])) * 2.0f) + vcvt_f32_f16(vget_low_f16(partial[r][3])) * 4.0f;
            const float32x4_t high = ((vcvt_high_f32_f16(partial[r][0]) * 0.5f + vcvt_high_f32_f16(partial[r][1])) + vcvt_high_f32_f16(partial[r][2]) * 2.0f) + vcvt_high_f32_f16(partial[r][3]) * 4.0f;
            // Keep the original final FP16 rounding before writing FP32 output.
            vst1q_f32(input[r].output + block * 8, vcvt_f32_f16(vcvt_f16_f32(low)));
            vst1q_f32(input[r].output + block * 8 + 4, vcvt_f32_f16(vcvt_f16_f32(high)));
#else
            const float16x8_t merged = ((partial[r][0] * vdupq_n_f16(0.5f) + partial[r][1]) + partial[r][2] * vdupq_n_f16(2.0f)) + partial[r][3] * vdupq_n_f16(4.0f);
            vst1q_f32(input[r].output + block * 8, vcvt_f32_f16(vget_low_f16(merged)));
            vst1q_f32(input[r].output + block * 8 + 4, vcvt_high_f32_f16(merged));
#endif
        }
    }
}

} // namespace ggml_tmac

#undef GGML_TMAC_UNROLL_ROWS
