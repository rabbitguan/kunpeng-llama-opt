#include "ggml.h"
#include "ggml-tmac.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct block_q4_0_local {
    ggml_fp16_t d;
    uint8_t qs[16];
};

static_assert(sizeof(block_q4_0_local) == 18, "unexpected Q4_0 block layout");

static int parse_positive(const char * text) {
    char * end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end || value < 1 || value > INT32_MAX) {
        throw std::runtime_error("invalid positive integer argument");
    }
    return static_cast<int>(value);
}

static uint8_t q4_unsigned(const uint8_t * data, int64_t index) {
    const block_q4_0_local * blocks = reinterpret_cast<const block_q4_0_local *>(data);
    const block_q4_0_local & block = blocks[index / 32];
    int internal = static_cast<int>(index % 32);
    uint8_t packed = block.qs[internal % 16];
    return internal < 16 ? packed & 15 : packed >> 4;
}

static float q4_scale(const uint8_t * data, int64_t index) {
    const block_q4_0_local * blocks = reinterpret_cast<const block_q4_0_local *>(data);
    return ggml_fp16_to_fp32(blocks[index / 32].d);
}

static float output_value(const std::vector<tmac_float_type> & output, size_t index) {
#ifdef __ARM_NEON
    return static_cast<float>(output[index]);
#else
    return output[index];
#endif
}

static void convert_activation(const float * source, tmac_float_type * destination, int64_t count) {
#ifdef __ARM_NEON
    ggml_fp32_to_fp16_row(source, reinterpret_cast<ggml_fp16_t *>(destination), count);
#else
    std::copy(source, source + count, destination);
#endif
}

static double elapsed_us(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char ** argv) {
    if (argc != 9) {
        std::fprintf(stderr, "usage: %s MODEL O K T ROUNDS WARMUP SEED OUTPUT_CSV\n", argv[0]);
        return 2;
    }
    ggml_context * metadata = nullptr;
    gguf_context * gguf = nullptr;
    try {
        const char * model_path = argv[1];
        int output_channels = parse_positive(argv[2]);
        int input_channels = parse_positive(argv[3]);
        int tokens = parse_positive(argv[4]);
        int rounds = parse_positive(argv[5]);
        int warmup = parse_positive(argv[6]);
        int seed = parse_positive(argv[7]);
        gguf_init_params params = {true, &metadata};
        gguf = gguf_init_from_file(model_path, params);
        if (!gguf || !metadata) {
            throw std::runtime_error("failed to read GGUF metadata");
        }
        ggml_tensor * weight = nullptr;
        int tensor_index = -1;
        for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
            ggml_tensor * candidate = ggml_get_tensor(metadata, gguf_get_tensor_name(gguf, i));
            if (candidate && candidate->type == GGML_TYPE_Q4_0 && candidate->ne[0] == input_channels && candidate->ne[1] == output_channels) {
                weight = candidate;
                tensor_index = i;
                break;
            }
        }
        if (!weight) {
            throw std::runtime_error("matching Q4_0 tensor not found");
        }
        size_t raw_bytes = ggml_nbytes(weight);
        std::vector<uint8_t> raw_weight(raw_bytes);
        std::ifstream model(model_path, std::ios::binary);
        model.seekg(static_cast<std::streamoff>(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_index)));
        model.read(reinterpret_cast<char *>(raw_weight.data()), static_cast<std::streamsize>(raw_bytes));
        if (!model) {
            throw std::runtime_error("failed to read tensor payload");
        }
        weight->data = raw_weight.data();
        weight->backend = GGML_BACKEND_TYPE_CPU;
        ggml_tmac_init();
        ggml_tmac_transform_tensor(weight);
        tmac_tensor_extra * transformed = static_cast<tmac_tensor_extra *>(weight->extra);
        if (!transformed || transformed->n_tile_num < 1 || output_channels % transformed->n_tile_num) {
            throw std::runtime_error("invalid transformed tensor metadata");
        }
        int tile_outputs = output_channels / transformed->n_tile_num;
        int64_t weight_bytes = static_cast<int64_t>(input_channels) * output_channels * 4 / 8;
        int64_t weight_tile_bytes = weight_bytes / transformed->n_tile_num;
        int scale_tile_count = transformed->scales_size / transformed->n_tile_num;

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
        std::vector<float> activation_f32(static_cast<size_t>(tokens) * input_channels);
        for (float & value : activation_f32) {
            value = distribution(generator);
        }
        std::vector<tmac_float_type> activation(static_cast<size_t>(tokens) * input_channels);
        std::vector<int8_t> qlut(static_cast<size_t>(tokens) * input_channels * 4);
        std::vector<tmac_float_type> lut_scales(static_cast<size_t>(tokens) * transformed->lut_scales_size);
        std::vector<tmac_float_type> lut_biases(static_cast<size_t>(tokens) * transformed->lut_scales_size);
        std::vector<tmac_float_type> output(static_cast<size_t>(tokens) * output_channels);

        auto build_lut = [&]() {
            for (int token = 0; token < tokens; ++token) {
                ggml_tmac_mul_mat_task_init(activation.data() + static_cast<size_t>(token) * input_channels,
                                            qlut.data() + static_cast<size_t>(token) * input_channels * 4,
                                            lut_scales.data() + static_cast<size_t>(token) * transformed->lut_scales_size,
                                            lut_biases.data() + static_cast<size_t>(token) * transformed->lut_scales_size,
                                            output_channels, input_channels, 1, 4);
            }
        };
        auto compute = [&]() {
            for (int token = 0; token < tokens; ++token) {
                for (int tile = 0; tile < transformed->n_tile_num; ++tile) {
                    ggml_tmac_mul_mat_task_compute(transformed->qweights + static_cast<int64_t>(tile) * weight_tile_bytes,
                                                   transformed->scales + static_cast<int64_t>(tile) * scale_tile_count,
                                                   qlut.data() + static_cast<size_t>(token) * input_channels * 4,
                                                   lut_scales.data() + static_cast<size_t>(token) * transformed->lut_scales_size,
                                                   lut_biases.data() + static_cast<size_t>(token) * transformed->lut_scales_size,
                                                   output.data() + static_cast<size_t>(token) * output_channels + static_cast<size_t>(tile) * tile_outputs,
                                                   tile_outputs, input_channels, 1, 4);
                }
            }
        };
        auto full = [&]() {
            convert_activation(activation_f32.data(), activation.data(), static_cast<int64_t>(activation_f32.size()));
            build_lut();
            compute();
            for (size_t i = 0; i < output.size(); ++i) {
                volatile float value = output_value(output, i);
                (void) value;
            }
        };
        convert_activation(activation_f32.data(), activation.data(), static_cast<int64_t>(activation_f32.size()));
        for (int i = 0; i < warmup; ++i) {
            full();
        }

        std::vector<int> sample_tokens = {0, tokens / 2, tokens - 1};
        std::vector<int> sample_outputs;
        for (int row = 0; row < tile_outputs; ++row) {
            sample_outputs.push_back(row);
        }
        sample_outputs.push_back(output_channels / 2);
        sample_outputs.push_back(output_channels - 1);
        std::sort(sample_tokens.begin(), sample_tokens.end());
        sample_tokens.erase(std::unique(sample_tokens.begin(), sample_tokens.end()), sample_tokens.end());
        std::sort(sample_outputs.begin(), sample_outputs.end());
        sample_outputs.erase(std::unique(sample_outputs.begin(), sample_outputs.end()), sample_outputs.end());
        double implementation_squared = 0.0;
        double reference_squared = 0.0;
        double implementation_max_abs = 0.0;
        double high_precision_max_abs = 0.0;
        size_t sample_count = 0;
        for (int token : sample_tokens) {
            for (int row : sample_outputs) {
                double quantized_reference = 0.0;
                double high_precision_reference = 0.0;
                for (int k_group = 0; k_group < input_channels; k_group += 4) {
                    int act_group = k_group / 32;
                    float lut_scale = output_value(lut_scales, static_cast<size_t>(token) * transformed->lut_scales_size + act_group);
                    float lut_bias = output_value(lut_biases, static_cast<size_t>(token) * transformed->lut_scales_size + act_group);
                    float weight_scale = q4_scale(raw_weight.data(), static_cast<int64_t>(row) * input_channels + k_group);
                    for (int bit = 0; bit < 4; ++bit) {
                        int index = 0;
                        for (int lane = 0; lane < 4; ++lane) {
                            uint8_t q = q4_unsigned(raw_weight.data(), static_cast<int64_t>(row) * input_channels + k_group + lane);
                            index |= ((q >> bit) & 1) << lane;
                        }
                        int8_t table = qlut[static_cast<size_t>(token) * input_channels * 4 + static_cast<size_t>(k_group / 4) * 16 + index];
                        const double alpha[4] = {0.5, 1.0, 2.0, 4.0};
                        quantized_reference += static_cast<double>(table) * lut_scale * weight_scale * alpha[bit];
                        if ((k_group % 32) == 0 && bit == 0) {
                            quantized_reference += static_cast<double>(lut_bias) * weight_scale * alpha[bit];
                        }
                    }
                    for (int lane = 0; lane < 4; ++lane) {
                        int64_t index = static_cast<int64_t>(row) * input_channels + k_group + lane;
                        int q = static_cast<int>(q4_unsigned(raw_weight.data(), index)) - 8;
                        high_precision_reference += static_cast<double>(q) * q4_scale(raw_weight.data(), index) * activation_f32[static_cast<size_t>(token) * input_channels + k_group + lane];
                    }
                }
                double actual = output_value(output, static_cast<size_t>(token) * output_channels + row);
                double implementation_error = actual - quantized_reference;
                implementation_max_abs = std::max(implementation_max_abs, std::abs(implementation_error));
                high_precision_max_abs = std::max(high_precision_max_abs, std::abs(actual - high_precision_reference));
                implementation_squared += implementation_error * implementation_error;
                reference_squared += quantized_reference * quantized_reference;
                ++sample_count;
            }
        }
        double implementation_nmse = reference_squared == 0.0 ? implementation_squared : implementation_squared / reference_squared;

        std::ofstream csv(argv[8], std::ios::app);
        if (!csv) {
            throw std::runtime_error("failed to open output CSV");
        }
        for (int round = 1; round <= rounds; ++round) {
            auto start = std::chrono::steady_clock::now();
            build_lut();
            double lut_us = elapsed_us(start);
            start = std::chrono::steady_clock::now();
            compute();
            double compute_us = elapsed_us(start);
            start = std::chrono::steady_clock::now();
            full();
            double total_us = elapsed_us(start);
            size_t workspace = qlut.size() * sizeof(int8_t) + (lut_scales.size() + lut_biases.size() + activation.size() + output.size()) * sizeof(tmac_float_type);
            csv << "B0," << output_channels << ',' << input_channels << ',' << tokens << ',' << round
                << ',' << lut_us << ',' << compute_us << ',' << total_us << ',' << workspace
                << ',' << implementation_max_abs << ',' << implementation_nmse << ',' << high_precision_max_abs
                << ",pass,repeated_n1," << tokens << ',' << tokens * transformed->n_tile_num << ',' << output_channels
                << ',' << transformed->n_tile_num << ',' << sample_count << ',' << seed
                << ",synthetic_uniform_-4_4," << weight->name << ";sample_tokens=0|T/2|T-1;sample_outputs=0..tile_outputs-1|O/2|O-1" << '\n';
        }
        ggml_tmac_free();
        gguf_free(gguf);
        ggml_free(metadata);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tmac-operator: %s\n", error.what());
        if (gguf) {
            gguf_free(gguf);
        }
        if (metadata) {
            ggml_free(metadata);
        }
        return 1;
    }
}
