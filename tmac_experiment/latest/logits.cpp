#include "llama.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: tmac-latest-logits MODEL OUTPUT");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        std::vector<llama_token> tokens = {128000, 791, 4062, 14198, 39935, 35308, 13};
        for (auto token : tokens) {
            require(token >= 0 && token < vocab, "fixed token outside vocabulary");
        }
        std::ofstream output(argv[2], std::ios::binary);
        const uint32_t header[] = {0x544d4c47, uint32_t(vocab) * 9};
        output.write(reinterpret_cast<const char *>(header), sizeof(header));
        for (int threads : {1, 16, 32}) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 2048;
            cp.n_batch = cp.n_ubatch = 512;
            cp.n_threads = cp.n_threads_batch = threads;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
            require(bool(ctx), "context creation failed");
            auto save = [&] {
                const float * logits = llama_get_logits_ith(ctx.get(), -1);
                require(logits != nullptr, "logits unavailable");
                for (int i = 0; i < vocab; ++i) {
                    require(std::isfinite(logits[i]), "nonfinite logits");
                }
                output.write(reinterpret_cast<const char *>(logits), size_t(vocab) * sizeof(float));
                require(output.good(), "logits write failed");
            };
            require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), int(tokens.size()))) == 0, "prefill failed");
            save();
            require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data() + 6, 1)) == 0, "decode failed");
            save();
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            std::vector<llama_token> history(1024);
            for (size_t i = 0; i < history.size(); ++i) {
                history[i] = tokens[i % tokens.size()];
            }
            for (int offset = 0; offset < 1024; offset += 512) {
                require(llama_decode(ctx.get(), llama_batch_get_one(history.data() + offset, 512)) == 0, "history prefill failed");
            }
            require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data() + 6, 1)) == 0, "depth decode failed");
            save();
        }
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tmac-latest-logits: %s\n", error.what());
        return 1;
    }
}
