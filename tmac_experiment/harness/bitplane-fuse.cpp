#include "../optimizations/bitplane-fuse.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static double maximum_abs = 0.0, maximum_rel = 0.0;
static int cases = 0;

template<int BM, int K>
static void check() {
#pragma clang fp contract(off)
    constexpr int O = BM / 4;
    std::mt19937 rng(20241025 + K + BM);
    for (int mode = 0; mode < 6; ++mode) {
        for (int round = 0; round < 10; ++round) {
            std::vector<uint8_t> raw(O * K), packed(O * K / 2, 0);
            std::vector<int8_t> lut(K * 4);
            std::vector<float16_t> scales(O * K / 32), ls(K / 32), bias(K / 32);
            std::vector<float16_t> result(O + 16, 91);
            for (auto & q : raw) { q = rng() % 16; }
            for (auto & q : lut) { q = mode == 0 ? 0 : mode == 1 ? -128 : mode == 2 ? 127 : int(rng() % 256) - 128; }
            for (auto & d : scales) { d = mode == 3 ? 0.0001f : (int(rng() % 3) - 1) * 0.25f; }
            for (int group = 0; group < K / 32; ++group) {
                ls[group] = mode == 4 ? 0.0f : mode == 3 ? 0.0001f : 0.25f;
                bias[group] = mode == 0 ? 0.0f : (int(rng() % 33) - 16) * 0.125f;
                for (int o = 0; o < O; ++o) {
                    for (int bit = 0; bit < 4; ++bit) {
                        for (int sub = 0; sub < 8; ++sub) {
                            unsigned index = 0;
                            for (int j = 0; j < 4; ++j) {
                                index |= ((raw[o * K + group * 32 + sub * 4 + j] >> bit) & 1) << j;
                            }
                            int lane = bit * 8 + o % 8;
                            packed[group * BM * 4 + (o / 8) * 128 + sub * 16 + lane % 16] |= index << (4 * (lane / 16));
                        }
                    }
                }
            }
            tmac_bitplane_fuse<BM, K>(packed.data(), lut.data(), scales.data(), ls.data(), bias.data(), result.data() + 8);
            for (int o = 0; o < O; ++o) {
                float expected = 0.0f;
                double precise = 0.0;
                for (int group = 0; group < K / 32; ++group) {
                    double plane_sum = 0.0;
                    for (int bit = 0; bit < 4; ++bit) {
                        int u = 0;
                        for (int sub = 0; sub < 8; ++sub) {
                            int index = 0;
                            for (int j = 0; j < 4; ++j) {
                                index += ((raw[o * K + group * 32 + sub * 4 + j] >> bit) & 1) * (1 << j);
                            }
                            u += lut[group * 128 + sub * 16 + index];
                        }
                        plane_sum += u * (0.5 * (1 << bit));
                    }
                    float d = scales[group * O + o];
                    expected += std::fma(float(plane_sum * 2), float(ls[group]), float(bias[group])) * (d * 0.5f);
                    precise += d * (double(ls[group]) * plane_sum + double(bias[group]) * 0.5);
                }
                float16_t rounded = expected;
                if (!std::isfinite(float(result[o + 8])) || std::memcmp(&rounded, &result[o + 8], sizeof(rounded))) {
                    std::fprintf(stderr, "B mismatch BM=%d K=%d mode=%d round=%d o=%d\n", BM, K, mode, round, o);
                    throw std::runtime_error("B independent reference mismatch");
                }
                double error = std::abs(double(result[o + 8]) - precise);
                if (error > 0.0005 * std::abs(precise) + 0.0001) { throw std::runtime_error("B double-reference error bound exceeded"); }
                maximum_abs = std::max(maximum_abs, error);
                maximum_rel = std::max(maximum_rel, error / std::max(1e-4, std::abs(precise)));
            }
            for (int i = 0; i < 8; ++i) {
                if (result[i] != 91 || result[O + 8 + i] != 91) { throw std::runtime_error("B buffer overwrite"); }
            }
            ++cases;
        }
    }
}

int main() {
    try {
        check<128, 32>(); check<256, 64>(); check<128, 96>(); check<256, 128>(); check<256, 4096>(); check<256, 14336>(); check<1024, 128>();
        std::printf("{\"status\":\"pass\",\"cases\":%d,\"nonfinite\":0,\"max_abs_vs_double\":%.9g,\"max_relative_vs_double_floor_1e_4\":%.9g}\n", cases, maximum_abs, maximum_rel);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
