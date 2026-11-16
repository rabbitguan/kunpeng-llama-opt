#include "llama.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    try {
        llama_backend_init();
        if (argc == 2 && std::string(argv[1]) == "--probe") {
            ggml_context * init = ggml_init({1024 * 1024, nullptr, true});
            require(init != nullptr, "ggml initialization failed");
            std::printf("{\"neon\":%d,\"dotprod\":%d,\"i8mm\":%d,\"sve\":%d,\"sve_bytes\":%d,\"fp16\":%d,\"llamafile\":%d}\n",
                        ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_sve(),
                        ggml_cpu_get_sve_cnt(), ggml_cpu_has_fp16_va(), ggml_cpu_has_llamafile());
            ggml_free(init);
        } else {
            require(argc == 4, "usage: llama-cpu-smoke --probe, or llama-cpu-smoke MODEL OUTPUT 4x4|8x8");
            std::string kernel(argv[3]);
            require(kernel == "4x4" || kernel == "8x8", "unsupported kernel");
            using set_kernel_t = void (*)(ggml_backend_cpu_repack_q4_0_kernel);
            auto set_kernel = reinterpret_cast<set_kernel_t>(ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(), "ggml_backend_cpu_repack_set_q4_0_kernel"));
            require(set_kernel != nullptr, "Q4_0 kernel selector is unavailable");
            set_kernel(kernel == "4x4" ? GGML_BACKEND_CPU_REPACK_Q4_0_KERNEL_4X4 : GGML_BACKEND_CPU_REPACK_Q4_0_KERNEL_8X8);
            llama_model_params mp = llama_model_default_params();
            mp.n_gpu_layers = 0;
            mp.use_extra_bufts = true;
            std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
            require(bool(model), "model load failed");
            std::vector<llama_token> tokens = {128000, 791, 4062, 14198, 39935, 35308, 13};
            int32_t count = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            for (llama_token token : tokens) {
                require(token >= 0 && token < count, "fixed token is outside vocabulary");
            }
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 512;
            cp.n_batch = cp.n_ubatch = 32;
            cp.n_threads = cp.n_threads_batch = 1;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
            require(bool(context), "context creation failed");
            require(llama_decode(context.get(), llama_batch_get_one(tokens.data(), tokens.size())) == 0, "fixed-token decode failed");
            const float * logits = llama_get_logits_ith(context.get(), -1);
            require(logits != nullptr && count > 0, "logits are unavailable");
            for (int32_t i = 0; i < count; ++i) {
                require(std::isfinite(logits[i]), "nonfinite logits");
            }
            std::ofstream output(argv[2], std::ios::binary);
            uint32_t magic = 0x544d4c47, size = count;
            output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
            output.write(reinterpret_cast<const char *>(&size), sizeof(size));
            output.write(reinterpret_cast<const char *>(logits), count * sizeof(float));
            require(output.good(), "cannot write logits");
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama-cpu-smoke: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
}
