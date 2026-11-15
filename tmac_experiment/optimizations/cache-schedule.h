#ifndef TMAC_CACHE_SCHEDULE_H
#define TMAC_CACHE_SCHEDULE_H
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace kunpeng_cache {
template<typename T>
inline T * aligned_data(std::vector<T> & storage) {
    return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(storage.data()) + 63) & ~uintptr_t(63));
}
inline uint64_t layout_id(int tiles, int k, int r, int bk) {
    if (tiles < 1 || tiles > 65535 || k < 32 || k > 65536 || k % 32 || bk < 32 || bk > 65536 || bk % 32 || (r != 1 && r != 2 && r != 4)) {
        throw std::runtime_error("unsupported C layout parameters");
    }
    return (uint64_t(1) << 56) | uint64_t(k) | (uint64_t(bk) << 17) | (uint64_t(r) << 34) | (uint64_t(tiles) << 37);
}
struct packed_weights {
    packed_weights() = default;
    packed_weights(const packed_weights &) = delete;
    packed_weights & operator=(const packed_weights &) = delete;
    packed_weights(packed_weights &&) = default;
    packed_weights & operator=(packed_weights &&) = default;
    uint64_t id = 0;
    std::vector<uint8_t> weights;
    std::vector<float16_t> scales;
    uint8_t * weight_data() { return aligned_data(weights); }
    float16_t * scale_data() { return aligned_data(scales); }
};
inline packed_weights pack(const uint8_t * w, const float16_t * s, int tiles, int k, int r, int bk) {
    packed_weights result;
    result.id = layout_id(tiles, k, r, bk);
    const int groups = k / 32;
    const int wg = 1024, sg = 64;
    result.weights.resize(size_t(tiles) * groups * wg + 63);
    result.scales.resize(size_t(tiles) * groups * sg + 31);
    size_t cursor = 0;
    for (int first = 0; first < tiles; first += r) {
        for (int g = 0; g < groups; g += bk / 32) {
            const int count = std::min(bk / 32, groups - g);
            for (int tile = first; tile < std::min(first + r, tiles); ++tile) {
                const size_t original = size_t(tile) * groups + g;
                std::memcpy(result.weight_data() + cursor * wg, w + original * wg, size_t(count) * wg);
                std::memcpy(result.scale_data() + cursor * sg, s + original * sg, size_t(count) * sg * sizeof(float16_t));
                if (std::memcmp(result.weight_data() + cursor * wg, w + original * wg, size_t(count) * wg) ||
                    std::memcmp(result.scale_data() + cursor * sg, s + original * sg, size_t(count) * sg * sizeof(float16_t))) {
                    throw std::runtime_error("C packing byte verification failed");
                }
                cursor += count;
            }
        }
    }
    return result;
}
}

extern "C" __attribute__((visibility("default"))) int32_t kunpeng_cache_compute_v1(
    int tiles, int k, int r, int bk, uint64_t id, const uint8_t * w, const float16_t * s,
    int8_t * lut, float16_t * ls, float16_t * bias, float16_t * output, float16_t * partial);

#ifdef TMAC_CACHE_KERNEL
extern "C" int32_t kunpeng_cache_compute_v1(
    int tiles, int k, int r, int bk, uint64_t id, const uint8_t * w, const float16_t * s,
    int8_t * lut, float16_t * ls, float16_t * bias, float16_t * output, float16_t * partial) {
    if (id != kunpeng_cache::layout_id(tiles, k, r, bk)) {
        throw std::runtime_error("C layout identifier mismatch");
    }
    size_t cursor = 0;
    for (int first = 0; first < tiles; first += r) {
        const int active = std::min(r, tiles - first);
        std::memset(partial, 0, size_t(active) * 256 * sizeof(float16_t));
        for (int g = 0; g < k / 32; g += bk / 32) {
            const int count = std::min(bk / 32, k / 32 - g);
            for (int tile = 0; tile < active; ++tile) {
                for (int inner = 0; inner < count; ++inner) {
                    const int group = g + inner;
                    tbl_g4_int8_float_update_impl<true, 8, 4, 8, false, false, false>(256,
                        partial + tile * 256, lut + group * 128,
                        const_cast<uint8_t *>(w + (cursor + inner) * 1024),
                        const_cast<float16_t *>(s + (cursor + inner) * 64), ls + group, bias + group);
                }
                cursor += count;
            }
        }
        for (int tile = 0; tile < active; ++tile) {
            const float16_t * c = partial + tile * 256;
            for (int o = 0; o < 64; ++o) {
                int base = (o / 8) * 32 + o % 8;
                output[(first + tile) * 64 + o] = ((float(c[base]) * 0.5f + float(c[base + 8])) + float(c[base + 16]) * 2.0f) + float(c[base + 24]) * 4.0f;
            }
        }
    }
    return 0;
}
#endif
#endif
