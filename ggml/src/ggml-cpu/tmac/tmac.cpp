#include "tmac.h"
#include "traits.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "t-mac/INIReader.h"
#include "t-mac/kernels.h"
#if defined(GGML_TMAC_FUSED)
#include "fused.h"
#endif

#include <atomic>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {
struct config {
    int bm, simd_n_in, simd_n_out, kfactor, group_size, lut_scales_size, scales_size, n_tile_num;
};
using half = __fp16;

static size_t aligned(size_t n) {
    GGML_ASSERT(n <= SIZE_MAX - 63);
    return (n + 63) & ~size_t(63);
}

static bool use_fused(const config & c) {
#if defined(GGML_TMAC_FUSED)
    return c.bm == 256 && c.simd_n_in == 16 && c.simd_n_out == 8 && c.kfactor == 8;
#else
    (void) c;
    return false;
#endif
}

static config kernel_config(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_Q4_0 || t->view_src || !ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1 ||
        t->ne[0] < 32 || t->ne[0] > 65536 || t->ne[0] % 32 || t->ne[1] < 1 || t->ne[1] > 65536 ||
        !strcmp(t->name, "token_embd.weight") || !strcmp(t->name, "output.weight")) {
        return {};
    }
    static const INIReader reader([]() -> std::string {
        if (const char * path = std::getenv("TMAC_KCFG_FILE")) {
            return path;
        }
#define TMAC_STRING_IMPL(x) #x
#define TMAC_STRING(x) TMAC_STRING_IMPL(x)
        return TMAC_STRING(TMAC_KCFG_FILE);
#undef TMAC_STRING
#undef TMAC_STRING_IMPL
    }());
    GGML_ASSERT(reader.ParseError() == 0);
    const std::string section = "qgemm_lut_t1_int8_m" + std::to_string(t->ne[1] * 4) + "_k" + std::to_string(t->ne[0]) + "_n1_b4";
    auto field = [&](const char * key) {
        const long value = reader.GetInteger(section, key, 0);
        GGML_ASSERT(value >= 0 && value <= INT_MAX);
        return int(value);
    };
    const config c = {field("bm"), field("simd_n_in"), field("simd_n_out"), field("kfactor"),
                      field("group_size"), field("lut_scales_size"), field("scales_size"), field("n_tile_num")};
    if (!c.bm) {
        return {};
    }
    GGML_ASSERT(c.bm > 0 && c.bm <= 4096 && c.bm % 4 == 0);
    GGML_ASSERT(c.simd_n_in > 0 && c.simd_n_in <= 64 && c.simd_n_out > 0 && c.simd_n_out <= 64);
    GGML_ASSERT(c.kfactor > 0 && c.kfactor <= t->ne[0] / 4);
    GGML_ASSERT(c.group_size == 32 && c.lut_scales_size == t->ne[0] / 32);
    GGML_ASSERT(c.scales_size == t->ne[0] * t->ne[1] / 32);
    GGML_ASSERT(c.bm % (2 * c.simd_n_in) == 0 && c.bm % (4 * c.simd_n_out) == 0);
    GGML_ASSERT(t->ne[0] % (4 * c.kfactor) == 0 && (4 * t->ne[1]) % c.bm == 0);
    GGML_ASSERT(c.n_tile_num == 4 * t->ne[1] / c.bm);
    return c;
}

// The base layout follows the pinned ggml-tmac.cpp Q4_0 transformer.
static void pack(const ggml_tensor * t, const config & c, uint8_t * weights, half * scales) {
    const size_t k = t->ne[0], m = t->ne[1], kg = k / 4;
    const size_t bm = c.bm, si = c.simd_n_in, so = c.simd_n_out, kf = c.kfactor;
    const auto * raw = static_cast<const block_q4_0 *>(t->data);
    memset(weights, 0, m * k / 2);
    for (size_t row = 0; row < m; ++row) {
        for (size_t group = 0; group < kg; ++group) {
            for (size_t bit = 0; bit < 4; ++bit) {
                uint8_t nibble = 0;
                for (size_t lane = 0; lane < 4; ++lane) {
                    const size_t column = group * 4 + lane;
                    const auto & block = raw[(row * k + column) / 32];
                    const uint8_t q = block.qs[column % 16] >> (column % 32 / 16 * 4);
                    nibble |= ((q >> bit) & 1) << lane;
                }
                size_t index = ((row / so * 4 + bit) * so + row % so) * kg + group;
                const size_t pair = index / (si * kg) % 2;
                index = (index / (2 * si * kg) * si + index / kg % si) * 2 * kg + pair * kg + group;
                const size_t tile = index / (bm * kg);
                const size_t block = index / (2 * si * kg) % (bm / (2 * si));
                const size_t input = index / (2 * kg) % si;
                const size_t ki = index % kg;
                const size_t out = (((tile * (kg / kf) + ki / kf) * (bm / (2 * si)) + block) * kf + ki % kf) * si + input;
                weights[out] |= nibble << (pair * 4);
            }
        }
        for (size_t group = 0; group < k / 32; ++group) {
            const size_t tile_rows = bm / 4;
            const size_t index = (row / tile_rows * (k / 32) + group) * tile_rows + row % tile_rows;
            const ggml_fp16_t scale = raw[row * (k / 32) + group].d;
            memcpy(scales + index, &scale, sizeof(scale));
        }
    }
#if defined(GGML_TMAC_FUSED)
    if (use_fused(c)) {
        const size_t weight_stride = size_t(k) * 32, scale_stride = size_t(k) * 2;
        std::vector<uint8_t> original_weights(weight_stride);
        std::vector<float16_t> original_scales(scale_stride);
        for (int tile = 0; tile < c.n_tile_num; ++tile) {
            auto * w = weights + size_t(tile) * weight_stride;
            auto * s = reinterpret_cast<float16_t *>(scales) + size_t(tile) * scale_stride;
            memcpy(original_weights.data(), w, weight_stride);
            memcpy(original_scales.data(), s, scale_stride * sizeof(float16_t));
            ggml_tmac::pack(int(k), original_weights.data(), original_scales.data(), w, s);
        }
    }
#endif
}

class tensor_traits final : public ggml::cpu::tensor_traits {
public:
    config c;
    std::atomic<unsigned> logged{0};
    explicit tensor_traits(config value) : c(value) {}

    static bool eligible(const ggml_tensor * op) {
        return op->op == GGML_OP_MUL_MAT && op->src[0]->type == GGML_TYPE_Q4_0 &&
               op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
               op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 &&
               op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
               op->src[1]->ne[0] == op->src[0]->ne[0] &&
               op->src[1]->ne[1] > 0 && op->src[1]->ne[1] <= 65536 &&
               op->src[1]->nb[0] == sizeof(float) && ggml_is_contiguous(op);
    }

    size_t row_workspace(const ggml_tensor * op) const {
        return aligned(size_t(op->src[0]->ne[0]) * 4) + 2 * aligned(size_t(c.lut_scales_size) * sizeof(half));
    }

    size_t thread_workspace(const ggml_tensor * op) const {
        return aligned(size_t(op->src[0]->ne[0]) * sizeof(half)) + aligned(size_t(c.bm / 4) * sizeof(half));
    }

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        if (!eligible(op)) {
            return false;
        }
        GGML_ASSERT(n_threads > 0);
        const size_t rows = op->src[1]->ne[1], rw = row_workspace(op), tw = thread_workspace(op);
        GGML_ASSERT(rows <= (SIZE_MAX - 63) / rw);
        GGML_ASSERT(size_t(n_threads) <= (SIZE_MAX - 63 - rows * rw) / tw);
        size = 63 + rows * rw + size_t(n_threads) * tw;
        return true;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (!eligible(op)) {
            return false;
        }
        const auto * weight = op->src[0];
        const auto * activation = op->src[1];
        const int k = int(weight->ne[0]), m = int(weight->ne[1]), rows = int(activation->ne[1]);
        const size_t rw = row_workspace(op), tw = thread_workspace(op);
        size_t required;
        work_size(params->nth, op, required);
        GGML_ASSERT(params->wsize >= required);
        auto * workspace = reinterpret_cast<uint8_t *>((uintptr_t(params->wdata) + 63) & ~uintptr_t(63));
        auto * local = workspace + size_t(rows) * rw + size_t(params->ith) * tw;
        auto * input = reinterpret_cast<half *>(local);
        auto * output = reinterpret_cast<half *>(local + aligned(size_t(k) * sizeof(half)));
        const bool fused = use_fused(c);
#if defined(GGML_TMAC_FUSED)
        if (fused) {
            const int groups = k / 32, chunks = (groups + 31) / 32;
            for (int64_t task = params->ith; task < int64_t(rows) * chunks; task += params->nth) {
                const size_t row = task / chunks;
                const int group = int(task % chunks) * 32;
                const int count = groups - group < 32 ? groups - group : 32;
                const auto * src = reinterpret_cast<const float *>(static_cast<const uint8_t *>(activation->data) + row * activation->nb[1]) + group * 32;
                ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t *>(input), count * 32);
                auto * lut = workspace + row * rw;
                auto * scale = reinterpret_cast<float16_t *>(lut + aligned(size_t(k) * 4));
                auto * bias = reinterpret_cast<float16_t *>(lut + aligned(size_t(k) * 4) + aligned(size_t(c.lut_scales_size) * sizeof(half)));
                ggml_tmac::prepare(count, reinterpret_cast<float16_t *>(input), reinterpret_cast<int8_t *>(lut) + group * 128, scale + group, bias + group);
            }
        }
#endif
        if (!fused) {
            for (int row = params->ith; row < rows; row += params->nth) {
                const auto * src = reinterpret_cast<const float *>(static_cast<const uint8_t *>(activation->data) + size_t(row) * activation->nb[1]);
                ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t *>(input), k);
                auto * lut = workspace + size_t(row) * rw;
                auto * scale = lut + aligned(size_t(k) * 4);
                auto * bias = scale + aligned(size_t(c.lut_scales_size) * sizeof(half));
                GGML_ASSERT(preprocessor_int8(m * 4, k, 1, 4, input, scale, bias, lut) == 0);
            }
        }
        ggml_barrier(params->threadpool);
        auto * weights = static_cast<uint8_t *>(weight->data) + aligned(ggml_nbytes(weight));
        auto * scales = weights + aligned(size_t(m) * k / 2);
        const size_t weight_stride = size_t(k) * (c.bm / 4) / 2;
        const size_t scale_stride = size_t(c.scales_size / c.n_tile_num) * sizeof(half);
#if defined(GGML_TMAC_FUSED)
        if (fused) {
            const int row_blocks = (rows + 1) / 2;
            for (int64_t task = params->ith; task < int64_t(row_blocks) * c.n_tile_num; task += params->nth) {
                const size_t first = size_t(task / c.n_tile_num) * 2, tile = task % c.n_tile_num;
                auto * packed = weights + tile * weight_stride;
                const auto * packed_scales = reinterpret_cast<const float16_t *>(scales + tile * scale_stride);
                auto make_row = [&](size_t row) {
                    const auto * lut = workspace + row * rw;
                    const auto * scale = lut + aligned(size_t(k) * 4);
                    const auto * bias = scale + aligned(size_t(c.lut_scales_size) * sizeof(half));
                    auto * dst = reinterpret_cast<float *>(static_cast<uint8_t *>(op->data) + row * op->nb[1]) + tile * 64;
                    return ggml_tmac::row{reinterpret_cast<const int8_t *>(lut), reinterpret_cast<const float16_t *>(scale), reinterpret_cast<const float16_t *>(bias), dst};
                };
                if (first + 1 < size_t(rows)) {
                    const ggml_tmac::row input_rows[2] = {make_row(first), make_row(first + 1)};
                    ggml_tmac::fused(k, packed, packed_scales, input_rows);
                } else {
                    const ggml_tmac::row input_rows[1] = {make_row(first)};
                    ggml_tmac::fused(k, packed, packed_scales, input_rows);
                }
            }
        }
#endif
        if (!fused) {
            for (int64_t task = params->ith; task < int64_t(rows) * c.n_tile_num; task += params->nth) {
                const size_t row = task / c.n_tile_num, tile = task % c.n_tile_num;
                auto * lut = workspace + row * rw;
                auto * scale = lut + aligned(size_t(k) * 4);
                auto * bias = scale + aligned(size_t(c.lut_scales_size) * sizeof(half));
                GGML_ASSERT(qgemm_lut_int8(c.bm, k, 1, 4, weights + tile * weight_stride, lut,
                                         scales + tile * scale_stride, scale, bias, output) == 0);
                auto * dst = reinterpret_cast<float *>(static_cast<uint8_t *>(op->data) + row * op->nb[1]) + tile * (c.bm / 4);
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(output), dst, c.bm / 4);
            }
        }
        if (params->ith == 0) {
            const unsigned phase = rows == 1 ? 1 : 2;
            if (!(logged.fetch_or(phase) & phase)) {
                GGML_LOG_INFO("tmac: compute tensor=%s K=%d M=%d rows=%d threads=%d kernel_N=%d tiles=%d consumer=%s\n",
                              weight->name, k, m, rows, params->nth, fused && rows > 1 ? 2 : 1, c.n_tile_num, fused ? "fused" : "generated");
            }
        }
        return true;
    }
};

struct buffer_context {
    ggml_backend_buffer_t raw;
    std::vector<std::unique_ptr<tensor_traits>> tensors;
};

static buffer_context * context(ggml_backend_buffer_t buffer) {
    return static_cast<buffer_context *>(buffer->context);
}

static void free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_buffer_free(context(buffer)->raw);
    delete context(buffer);
}

static void * get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_get_base(context(buffer)->raw);
}

static ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t) {
    auto c = kernel_config(t);
    if (!c.bm) {
        return GGML_STATUS_FAILED;
    }
    auto traits = std::make_unique<tensor_traits>(c);
    t->extra = traits.get();
    context(buffer)->tensors.push_back(std::move(traits));
    return GGML_STATUS_SUCCESS;
}

static void set_tensor(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(t));
    memcpy(t->data, data, size);
    auto * traits = static_cast<tensor_traits *>(t->extra);
    auto * weights = static_cast<uint8_t *>(t->data) + aligned(size);
    auto * scales = reinterpret_cast<half *>(weights + aligned(size_t(t->ne[0]) * t->ne[1] / 2));
    pack(t, traits->c, weights, scales);
    GGML_LOG_INFO("tmac: packed tensor=%s K=%lld M=%lld tiles=%d\n", t->name,
                  (long long) t->ne[0], (long long) t->ne[1], traits->c.n_tile_num);
}

static void get_tensor(ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    memcpy(data, static_cast<const uint8_t *>(t->data) + offset, size);
}

static void clear_buffer(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_buffer_clear(context(buffer)->raw, value);
}

static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * raw = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (!raw) {
        return nullptr;
    }
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = free_buffer;
    iface.get_base = get_base;
    iface.init_tensor = init_tensor;
    iface.set_tensor = set_tensor;
    iface.get_tensor = get_tensor;
    iface.clear = clear_buffer;
    return ggml_backend_buffer_init(buft, iface, new buffer_context{raw, {}}, size);
}

static const char * get_name(ggml_backend_buffer_type_t) { return "CPU_TMAC"; }
static size_t get_alignment(ggml_backend_buffer_type_t) { return 64; }
static size_t get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    const auto c = kernel_config(t);
    return c.bm ? aligned(ggml_nbytes(t)) + aligned(size_t(t->ne[0]) * t->ne[1] / 2) + aligned(size_t(c.scales_size) * sizeof(half)) : ggml_nbytes(t);
}

class extra_buffer_type final : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        return tensor_traits::eligible(op) && kernel_config(op->src[0]).bm != 0;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        if (tensor_traits::eligible(op) && op->src[0]->buffer && op->src[0]->buffer->buft->context == this) {
            return static_cast<tensor_traits *>(op->src[0]->extra);
        }
        return nullptr;
    }
};
} // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_tmac_buffer_type() {
    static extra_buffer_type extra;
    static ggml_backend_buffer_type buft = [] {
        ggml_backend_buffer_type result = {};
        result.iface.get_name = get_name;
        result.iface.alloc_buffer = alloc_buffer;
        result.iface.get_alignment = get_alignment;
        result.iface.get_alloc_size = get_alloc_size;
        result.device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        result.context = &extra;
        return result;
    }();
    return &buft;
}
