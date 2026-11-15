#ifndef TMAC_BITPLANE_FUSE_H
#define TMAC_BITPLANE_FUSE_H
#include <arm_neon.h>
#include <cstdint>

// The partial bounds are 3072, 7168 and 15360, so the weighted merge fits INT16.
template<int BM, int K>
inline int32_t tmac_bitplane_fuse(const uint8_t * weights, const int8_t * lut, const float16_t * scales,
                                const float16_t * lut_scales, const float16_t * biases, float16_t * output) {
#pragma clang fp contract(off)
    static_assert(BM % 32 == 0 && K % 32 == 0, "unsupported B shape");
    constexpr int outputs = BM / 4;
    float partial[outputs] = {};
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int group = 0; group < K / 32; ++group) {
        int8x16_t tables[8];
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; ++k) {
            tables[k] = vld1q_s8(lut + group * 128 + k * 16);
        }
        for (int block = 0; block < outputs / 8; ++block) {
            int16x8_t u0 = vdupq_n_s16(0), u1 = u0, u2 = u0, u3 = u0;
#pragma clang loop unroll(full)
            for (int k = 0; k < 8; ++k) {
                uint8x16_t packed = vld1q_u8(weights + group * BM * 4 + block * 128 + k * 16);
                int8x16_t low = vqtbl1q_s8(tables[k], vandq_u8(packed, mask));
                int8x16_t high = vqtbl1q_s8(tables[k], vshrq_n_u8(packed, 4));
                u0 = vaddw_s8(u0, vget_low_s8(low));
                u1 = vaddw_s8(u1, vget_high_s8(low));
                u2 = vaddw_s8(u2, vget_low_s8(high));
                u3 = vaddw_s8(u3, vget_high_s8(high));
            }
            int16x8_t merged = vmlaq_n_s16(u0, u1, 2);
            merged = vmlaq_n_s16(merged, u2, 4);
            merged = vmlaq_n_s16(merged, u3, 8);
            int32x4_t lower = vmovl_s16(vget_low_s16(merged));
            int32x4_t upper = vmovl_high_s16(merged);
            float32x4_t low = vfmaq_n_f32(vdupq_n_f32(biases[group]), vcvtq_f32_s32(lower), lut_scales[group]);
            float32x4_t high = vfmaq_n_f32(vdupq_n_f32(biases[group]), vcvtq_f32_s32(upper), lut_scales[group]);
            float16x8_t d = vld1q_f16(scales + group * outputs + block * 8);
            low = vmulq_f32(low, vmulq_n_f32(vcvt_f32_f16(vget_low_f16(d)), 0.5f));
            high = vmulq_f32(high, vmulq_n_f32(vcvt_high_f32_f16(d), 0.5f));
            vst1q_f32(partial + block * 8, vaddq_f32(vld1q_f32(partial + block * 8), low));
            vst1q_f32(partial + block * 8 + 4, vaddq_f32(vld1q_f32(partial + block * 8 + 4), high));
        }
    }
    for (int i = 0; i < outputs; i += 8) {
        vst1q_f16(output + i, vcombine_f16(vcvt_f16_f32(vld1q_f32(partial + i)), vcvt_f16_f32(vld1q_f32(partial + i + 4))));
    }
    return 0;
}
#endif
