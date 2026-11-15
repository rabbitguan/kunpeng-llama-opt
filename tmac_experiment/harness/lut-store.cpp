#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" void lut_ref(int, int8_t *, float16_t *, float16_t *, float16_t *);
extern "C" void lut_opt(int, int8_t *, float16_t *, float16_t *, float16_t *);

int main() {
    std::mt19937 rng(20241025);
    int cases = 0;
    for (int size : {32, 64, 96, 128}) {
        for (float amplitude : {0.0f, 0.0001f, 1.0f, 64.0f}) {
            for (int round = 0; round < 100; ++round) {
                std::vector<float16_t> b(size);
                for (auto & value : b) {
                    value = amplitude * (int(rng() % 2001) - 1000) / 1000;
                }
                std::vector<int8_t> ref(size * 4 + 32, 91), opt(ref);
                float16_t sr = amplitude / 16, so = sr, br = 0, bo = 0;
                lut_ref(size, ref.data() + 16, b.data(), &sr, &br);
                lut_opt(size, opt.data() + 16, b.data(), &so, &bo);
                if (ref != opt || std::memcmp(&sr, &so, sizeof(sr)) || std::memcmp(&br, &bo, sizeof(br))) {
                    std::fprintf(stderr, "LUT mismatch size=%d amplitude=%g round=%d\n", size, amplitude, round);
                    return 1;
                }
                if (!std::all_of(opt.begin(), opt.begin() + 16, [](int8_t v) { return v == 91; }) ||
                    !std::all_of(opt.end() - 16, opt.end(), [](int8_t v) { return v == 91; })) {
                    return 2;
                }
                ++cases;
            }
        }
    }
    std::printf("LUT byte/scale/bias equality: %d cases passed\n", cases);
}
