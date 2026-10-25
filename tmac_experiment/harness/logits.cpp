#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static int32_t parse_i32(const char * text) {
    char * end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT32_MAX) {
        throw std::runtime_error("invalid positive integer argument");
    }
    return static_cast<int32_t>(value);
}

static std::vector<llama_token> parse_tokens(const std::string & text) {
    std::vector<llama_token> tokens;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        std::string item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        tokens.push_back(parse_i32(item.c_str()));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return tokens;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL OUTPUT THREADS TOKEN_IDS\n", argv[0]);
        return 2;
    }
    try {
        int32_t threads = parse_i32(argv[3]);
        std::vector<llama_token> tokens = parse_tokens(argv[4]);
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_load_model_from_file(argv[1], model_params);
        if (!model) {
            throw std::runtime_error("failed to load model");
        }
        int32_t n_vocab = llama_n_vocab(model);
        for (llama_token token : tokens) {
            if (token < 0 || token >= n_vocab) {
                throw std::runtime_error("token id is outside vocabulary");
            }
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 512;
        context_params.n_batch = 512;
        context_params.n_ubatch = 512;
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.logits_all = false;
        llama_context * context = llama_new_context_with_model(model, context_params);
        if (!context) {
            llama_free_model(model);
            throw std::runtime_error("failed to create context");
        }
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()), 0, 0);
        if (llama_decode(context, batch) != 0) {
            llama_free(context);
            llama_free_model(model);
            throw std::runtime_error("llama_decode failed");
        }
        float * logits = llama_get_logits_ith(context, -1);
        if (!logits) {
            llama_free(context);
            llama_free_model(model);
            throw std::runtime_error("logits are unavailable");
        }
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        uint32_t magic = 0x544d4c47;
        uint32_t count = static_cast<uint32_t>(n_vocab);
        output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        output.write(reinterpret_cast<const char *>(&count), sizeof(count));
        output.write(reinterpret_cast<const char *>(logits), sizeof(float) * count);
        if (!output) {
            throw std::runtime_error("failed to write logits");
        }
        llama_free(context);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tmac-logits: %s\n", error.what());
        return 1;
    }
}
