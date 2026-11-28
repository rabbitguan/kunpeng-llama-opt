#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <random>
#include <memory>
#include <fstream>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    try {
        const bool benchmark = argc >= 2 && (!strcmp(argv[1], "--bench-tmac") || !strcmp(argv[1], "--bench-native"));
        require(argc == 1 || (benchmark && argc <= 5), "usage: tmac-latest-check [--bench-tmac|--bench-native [1|16|32 [OUTPUT [1024|4096]]]]");
        if (argc == 5) require(!strcmp(argv[4], "1024") || !strcmp(argv[4], "4096"), "invalid output dimension");
        int selected_threads = 0;
        if (argc >= 3) {
            require(!strcmp(argv[2], "1") || !strcmp(argv[2], "16") || !strcmp(argv[2], "32"), "invalid thread count");
            selected_threads = std::atoi(argv[2]);
        }
        std::ofstream output_file;
        if (argc >= 4) {
            output_file.open(argv[3], std::ios::binary);
            require(output_file.good(), "cannot open operator output file");
        }
        const bool native = benchmark && !strcmp(argv[1], "--bench-native");
        const char * buffer_name = native ? "CPU_REPACK" : "CPU_TMAC";
        ggml_backend_t cpu = ggml_backend_cpu_init();
        using extra_fn = ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t);
        auto extra = reinterpret_cast<extra_fn>(ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_dev_get_extra_bufts"));
        require(extra != nullptr, "CPU extra buffers unavailable");
        auto dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        ggml_backend_buffer_type_t buft = nullptr;
        for (auto * current = extra(dev); *current; ++current) {
            if (!strcmp(ggml_backend_buft_name(*current), buffer_name)) {
                buft = *current;
            }
        }
        require(buft != nullptr, "requested CPU buffer is unavailable");
        constexpr int k = 4096;
        const int m = argc == 5 ? std::atoi(argv[4]) : 4096;
        auto * wc = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
        require(wc != nullptr, "weight context failed");
        auto * w = ggml_new_tensor_2d(wc, GGML_TYPE_Q4_0, k, m);
        ggml_set_name(w, "blk.0.attn_q.weight");
        auto * wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, buft);
        require(wb != nullptr, "weight allocation failed");
        std::vector<uint8_t> raw(ggml_nbytes(w));
        std::mt19937 random(20241025);
        for (size_t block = 0; block < raw.size() / 18; ++block) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(block % 2 ? -0.03125f : 0.015625f);
            memcpy(raw.data() + block * 18, &d, 2);
            for (int j = 0; j < 16; ++j) {
                raw[block * 18 + 2 + j] = uint8_t(random());
            }
        }
        ggml_backend_tensor_set(w, raw.data(), 0, raw.size());
        if (!native) {
            std::vector<uint8_t> readback(raw.size());
            ggml_backend_tensor_get(w, readback.data(), 0, readback.size());
            require(readback == raw, "raw weight readback changed");
        }
        for (int rows : {1, 7, 32}) {
            auto * ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
            require(ctx != nullptr, "graph context failed");
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
            auto * y = ggml_mul_mat(ctx, w, x);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, y);
            auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
            require(buffer != nullptr, "graph allocation failed");
            std::vector<float> input(size_t(k) * rows), expected(size_t(m) * rows);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = float(int((i * 13 + 7) % 257) - 128) / 128;
            }
            for (int row = 0; row < rows; ++row) {
                for (int out = 0; out < m; ++out) {
                    double sum = 0;
                    for (int col = 0; col < k; ++col) {
                        const size_t block = (size_t(out) * k + col) / 32;
                        ggml_fp16_t d;
                        memcpy(&d, raw.data() + block * 18, 2);
                        const uint8_t packed = raw[block * 18 + 2 + col % 16];
                        const int q = ((packed >> (col % 32 / 16 * 4)) & 15) - 8;
                        sum += q * ggml_fp16_to_fp32(d) * input[size_t(row) * k + col];
                    }
                    expected[size_t(row) * m + out] = float(sum);
                }
            }
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            std::vector<float> single, result(expected.size());
            for (int threads : {1, 16, 32}) {
                if (selected_threads && threads != selected_threads) continue;
                auto pool_params = ggml_threadpool_params_default(threads);
                std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&pool_params), ggml_threadpool_free);
                require(bool(pool), "threadpool creation failed");
                ggml_backend_cpu_set_threadpool(cpu, pool.get());
                ggml_backend_cpu_set_n_threads(cpu, threads);
                require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
                ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
                if (threads > 1 && single.empty()) {
                    ggml_backend_cpu_set_threadpool(cpu, nullptr);
                    ggml_backend_cpu_set_n_threads(cpu, 1);
                    require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "single-thread reference compute failed");
                    single.resize(expected.size());
                    ggml_backend_tensor_get(y, single.data(), 0, single.size() * sizeof(float));
                    ggml_backend_cpu_set_n_threads(cpu, threads);
                    ggml_backend_cpu_set_threadpool(cpu, pool.get());
                }
                double error = 0, norm = 0, max_abs = 0;
                for (size_t i = 0; i < result.size(); ++i) {
                    require(std::isfinite(result[i]), "nonfinite output");
                    const double delta = result[i] - expected[i];
                    error += delta * delta;
                    norm += double(expected[i]) * expected[i];
                    max_abs = std::fmax(max_abs, std::fabs(delta));
                }
                require(norm > 0 && error / norm <= 0.001 && max_abs <= 1.0, "independent Q4_0 reference mismatch");
                if (threads == 1) {
                    single = result;
                } else {
                    require(single.size() == result.size(), "single-thread reference is unavailable");
                    require(memcmp(single.data(), result.data(), result.size() * sizeof(float)) == 0, "thread counts changed outputs");
                }
                if (output_file.is_open()) {
                    output_file.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
                    require(output_file.good(), "cannot write operator output file");
                }
                if (benchmark) {
                    std::vector<double> samples;
                    for (int round = -3; round < 10; ++round) {
                        const auto start = std::chrono::steady_clock::now();
                        require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "timed compute failed");
                        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                        if (round >= 0) samples.push_back(us);
                    }
                    auto ordered = samples;
                    std::sort(ordered.begin(), ordered.end());
                    std::printf("{\"buffer\":\"%s\",\"K\":%d,\"M\":%d,\"rows\":%d,\"threads\":%d,\"nmse\":%.9g,\"max_abs\":%.9g,\"median_graph_us\":%.6f,\"samples_us\":[",
                                buffer_name, k, m, rows, threads, error / norm, max_abs, (ordered[4] + ordered[5]) / 2);
                    for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", samples[i]);
                    std::printf("],\"status\":\"pass\"}\n");
                } else {
                    std::printf("{\"rows\":%d,\"threads\":%d,\"nmse\":%.9g,\"max_abs\":%.9g,\"status\":\"pass\"}\n", rows, threads, error / norm, max_abs);
                }
                std::fflush(stdout);
                ggml_backend_cpu_set_threadpool(cpu, nullptr);
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        ggml_backend_buffer_free(wb);
        ggml_free(wc);
        ggml_backend_free(cpu);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tmac-latest-check: %s\n", error.what());
        return 1;
    }
}
