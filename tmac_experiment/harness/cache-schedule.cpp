#include "../optimizations/cache-schedule.h"
#include <cmath>
#include <cstdio>
#include <random>

int main() {
    try {
        std::mt19937 rng(20241025);
        int cases = 0;
        for (int tiles : {1, 3, 5}) {
            for (int k : {32, 96, 128}) {
                for (int mode = 0; mode < 3; ++mode) {
                    int groups = k / 32;
                    std::vector<uint8_t> weights(tiles * groups * 1024);
                    std::vector<int8_t> lut(groups * 128);
                    std::vector<float16_t> scales(tiles * groups * 64), ls(groups), bias(groups);
                    for (auto & x : weights) { x = rng() % 256; }
                    for (auto & x : lut) { x = mode ? int(rng() % 255) - 127 : 0; }
                    for (auto & x : scales) { x = (int(rng() % 21) - 10) * 0.01f; }
                    for (auto & x : ls) { x = mode == 2 ? 0.0001f : 0.01f; }
                    for (auto & x : bias) { x = mode ? (int(rng() % 21) - 10) * 0.01f : 0.0f; }
                    std::vector<float16_t> expected(tiles * 64), c(256);
                    for (int tile = 0; tile < tiles; ++tile) {
                        std::fill(c.begin(), c.end(), 0);
                        for (int g = 0; g < groups; ++g) {
                            tbl_g4_int8_float_update_impl<true, 8, 4, 8, false, false, false>(256,
                                c.data(), lut.data() + g * 128, weights.data() + (tile * groups + g) * 1024,
                                scales.data() + (tile * groups + g) * 64, ls.data() + g, bias.data() + g);
                        }
                        for (int o = 0; o < 64; ++o) {
                            int base = (o / 8) * 32 + o % 8;
                            expected[tile * 64 + o] = ((float(c[base]) * 0.5f + float(c[base + 8])) + float(c[base + 16]) * 2.0f) + float(c[base + 24]) * 4.0f;
                        }
                    }
                    for (int r : {1, 2, 4}) {
                        for (int bk : {32, 64, 256}) {
                            auto packed = kunpeng_cache::pack(weights.data(), scales.data(), tiles, k, r, bk);
                            std::vector<float16_t> scratch(r * 256 + 31), result(tiles * 64 + 16, 91);
                            kunpeng_cache_compute_v1(tiles, k, r, bk, packed.id, packed.weight_data(), packed.scale_data(), lut.data(), ls.data(), bias.data(), result.data() + 8, kunpeng_cache::aligned_data(scratch));
                            if (std::memcmp(expected.data(), result.data() + 8, expected.size() * sizeof(float16_t))) {
                                throw std::runtime_error("C schedule differs from original tile-first reference");
                            }
                            for (int i = 0; i < 8; ++i) {
                                if (result[i] != 91 || result[tiles * 64 + 8 + i] != 91) { throw std::runtime_error("C buffer overwrite"); }
                            }
                            if (reinterpret_cast<uintptr_t>(packed.weight_data()) % 64 || reinterpret_cast<uintptr_t>(packed.scale_data()) % 64 || reinterpret_cast<uintptr_t>(kunpeng_cache::aligned_data(scratch)) % 64) {
                                throw std::runtime_error("C buffer alignment mismatch");
                            }
                            if (r == 1 && (std::memcmp(weights.data(), packed.weight_data(), weights.size()) || std::memcmp(scales.data(), packed.scale_data(), scales.size() * sizeof(float16_t)))) {
                                throw std::runtime_error("R=1 layout is not original layout");
                            }
                            bool rejected = false;
                            try {
                                kunpeng_cache_compute_v1(tiles, k, r, bk, packed.id ^ 1, packed.weight_data(), packed.scale_data(), lut.data(), ls.data(), bias.data(), result.data() + 8, kunpeng_cache::aligned_data(scratch));
                            } catch (const std::runtime_error &) { rejected = true; }
                            if (!rejected) { throw std::runtime_error("C accepted an invalid layout identifier"); }
                            ++cases;
                        }
                    }
                }
            }
        }
        std::printf("C layout/schedule bitwise checks: %d cases passed\n", cases);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
