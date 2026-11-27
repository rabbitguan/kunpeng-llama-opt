#include TMAC_REFERENCE_SOURCE
#include "../../ggml/src/ggml-cpu/tmac/fused.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

template<int Rows>
static void check(int k, int mode, std::mt19937 & random) {
    const int groups = k / 32;
    std::vector<uint8_t> weights(size_t(groups) * 1024);
    std::vector<float16_t> scales(size_t(groups) * 64);
    std::vector<int8_t> tables[Rows];
    std::vector<float16_t> lut_scales[Rows], biases[Rows];
    std::vector<float> outputs[Rows];
    ggml_tmac::row input[Rows];
    for (auto & value : weights) value = uint8_t(random());
    for (auto & value : scales) value = float(int(random() % 21) - 10) * 0.001f;
    for (int r = 0; r < Rows; ++r) {
        tables[r].resize(size_t(groups) * 128);
        lut_scales[r].resize(groups);
        biases[r].resize(groups);
        outputs[r].assign(80, 91);
        for (auto & value : tables[r]) value = mode == 0 ? 0 : mode == 4 ? (random() % 2 ? -128 : 127) : int(random() % 255) - 127;
        for (auto & value : lut_scales[r]) value = mode == 2 ? 0.00001f : 0.003f;
        for (auto & value : biases[r]) value = mode == 0 ? 0 : float(int(random() % 21) - 10) * 0.001f;
        input[r] = {tables[r].data(), lut_scales[r].data(), biases[r].data(), outputs[r].data() + 8};
    }
    std::vector<float16_t> activation(k), expected_scales(groups), expected_biases(groups), actual_scales(groups), actual_biases(groups);
    std::vector<int8_t> expected_lut(size_t(groups) * 128), actual_lut(expected_lut.size());
    for (auto & value : activation) value = mode == 0 ? 0.0f : float(int(random() % 257) - 128) / 128;
    if (preprocessor_int8(16384, k, 1, 4, activation.data(), expected_scales.data(), expected_biases.data(), expected_lut.data()) == 0) {
        for (int group = 0; group < groups; group += 32) {
            const int count = groups - group < 32 ? groups - group : 32;
            ggml_tmac::prepare(count, activation.data() + group * 32, actual_lut.data() + group * 128, actual_scales.data() + group, actual_biases.data() + group);
        }
        if (expected_lut != actual_lut || std::memcmp(expected_scales.data(), actual_scales.data(), groups * sizeof(float16_t)) || std::memcmp(expected_biases.data(), actual_biases.data(), groups * sizeof(float16_t))) {
            throw std::runtime_error("chunked LUT preparation differs from original");
        }
    }
    std::vector<uint8_t> packed(weights.size() + 128, 193);
    std::vector<float16_t> packed_scales(scales.size() + 16, 91);
    ggml_tmac::pack(k, weights.data(), scales.data(), packed.data() + 64, packed_scales.data() + 8);
    for (int block = 0; block < 8; ++block) {
        for (int group = 0; group < groups; ++group) {
            const size_t from = size_t(group) * 8 + block, to = size_t(block) * groups + group;
            if (std::memcmp(packed.data() + 64 + to * 128, weights.data() + from * 128, 128) ||
                std::memcmp(packed_scales.data() + 8 + to * 8, scales.data() + from * 8, 8 * sizeof(float16_t))) {
                throw std::runtime_error("fused packing changed weight or scale bytes");
            }
        }
    }
    for (int i = 0; i < 64; ++i) {
        if (packed[i] != 193 || packed[weights.size() + 64 + i] != 193) {
            throw std::runtime_error("fused weight packing overwrote canaries");
        }
    }
    for (int i = 0; i < 8; ++i) {
        if (packed_scales[i] != 91 || packed_scales[scales.size() + 8 + i] != 91) {
            throw std::runtime_error("fused scale packing overwrote canaries");
        }
    }
    ggml_tmac::fused(k, packed.data() + 64, packed_scales.data() + 8, input);
    for (int r = 0; r < Rows; ++r) {
        alignas(64) float16_t partial[256] = {};
        for (int group = 0; group < groups; ++group) {
            tbl_g4_int8_float_update_impl<true, 8, 4, 8, false>(256, partial, tables[r].data() + group * 128,
                weights.data() + group * 1024, scales.data() + group * 64, lut_scales[r].data() + group, biases[r].data() + group);
        }
        half generated[64];
        const bool generated_available = qgemm_lut_int8(256, k, 1, 4, weights.data(), tables[r].data(), scales.data(), lut_scales[r].data(), biases[r].data(), generated) == 0;
        for (int out = 0; out < 64; ++out) {
            const int base = (out / 8) * 32 + out % 8;
            const half value = ((half(partial[base]) * half(0.5f) + half(partial[base + 8])) + half(partial[base + 16]) * half(2.0f)) + half(partial[base + 24]) * half(4.0f);
            const float expected = float(value);
            const float actual = input[r].output[out];
            if (generated_available && std::memcmp(&value, generated + out, sizeof(half))) {
                throw std::runtime_error("fused reference differs from generated qgemm");
            }
            if (!std::isfinite(actual) || std::memcmp(&expected, &actual, sizeof(float))) {
                std::fprintf(stderr, "mismatch rows=%d K=%d mode=%d row=%d output=%d expected=%.9g actual=%.9g\n", Rows, k, mode, r, out, expected, actual);
                throw std::runtime_error("fused output differs from original");
            }
        }
        for (int i = 0; i < 8; ++i) {
            if (outputs[r][i] != 91 || outputs[r][72 + i] != 91) {
                throw std::runtime_error("fused output overwrote canaries");
            }
        }
    }
}

int main() {
    try {
        std::mt19937 random(20261127);
        int cases = 0;
        for (int k : {32, 96, 4096, 11008, 14336}) {
            for (int mode = 0; mode < 5; ++mode) {
                check<1>(k, mode, random);
                check<2>(k, mode, random);
                cases += 2;
            }
        }
        std::printf("{\"status\":\"pass\",\"cases\":%d,\"bitwise_equal\":true}\n", cases);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
