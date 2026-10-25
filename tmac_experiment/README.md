# Kunpeng T-MAC baseline experiment

This directory reproduces the unmodified T-MAC algorithm at upstream commit `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`. The only source patch adds a named Kunpeng AArch64 compilation target. It does not change LUT construction, table lookup, bit-plane accumulation, quantization, or grouping.

## Scope and verified source behavior

The baseline uses the existing Q4_0 GGUF directly. The pinned llama.cpp fork transforms Q4_0 weights during model loading, so this workflow does not download another model and does not requantize weights.

Experiment dimensions map to the pinned source as follows:

- `T` is T-MAC `N`.
- `O` is the logical output channel count. `deploy/compile.py` multiplies it by `bits` before passing it as `QGeMMLUTBitsCodegen.M` because that tensor dimension contains bit planes.
- `K` is T-MAC `K`.

The pinned llama.cpp integration generates kernels with `N=1`. The evidence is in pinned file `3rdparty/llama.cpp/ggml/src/ggml.c`: the initialization loop starting at line 12586 calls `ggml_tmac_mul_mat_task_init(..., 1, bits)` once per `ine11`, and the compute loops starting at lines 12613 and 12680 call `ggml_tmac_mul_mat_task_compute(..., 1, bits)` once per token and output tile. Results from `T=32` or `T=128` are therefore recorded as `execution_mode=repeated_n1`, with the token call count and covered output channels, not as native multi-token GEMM.

With `bits=4`, `weight_group_size=32`, `act_group_size=32`, and `g=4`, the official constraint accepts `kfactor=8` and rejects 16 because `(kfactor*g)` must divide both activation and weight groups. ARM output is `float16`, the LUT is `int8`, and accumulation is `int32`. Fast aggregation and zero point are disabled.

## Server use

Upload this directory to `/data/home/2300013234/kunpeng-llama-opt/tmac_experiment`, then run:

```sh
cd /data/home/2300013234/kunpeng-llama-opt
bash tmac_experiment/run.sh setup
bash tmac_experiment/run.sh baseline
```

Paths can be changed without editing scripts:

```sh
bash tmac_experiment/run.sh setup --set paths.model=/path/model.gguf --set paths.tmac_root=/path/work
```

`setup` requires Git, CMake, Make, the TVM system development packages, and Python 3.8 through 3.11 before toolchain preparation. Clang is not an installation prerequisite. Setup first reuses a validated project-local LLVM 17.0.6 toolchain. If none exists, the pinned T-MAC build hook downloads the fixed Linux AArch64 LLVM 17.0.6 release under `tmac/source/build/` and builds the pinned TVM submodule. If LLVM already exists but TVM does not, setup reuses that LLVM instead of downloading another copy. It creates a local virtual environment under `tmac/`, recursively checks every pinned commit, and applies the adaptation patch idempotently. It does not use sudo, Conda, perf, system library replacement, compiler aliases, or global shell changes.

After preparation, setup injects the project LLVM `bin`, TVM Python directory, LLVM/TVM library directories, and `TVM_ROOT` into every subprocess. It verifies `clang --version`, `clang++ --version`, `llvm-config --version`, the TVM shared libraries and CMake configuration, `import t_mac, tvm`, and a compiled and executed C++ smoke program. Only then does it write `.setup-fingerprint`. Output from downloads and long builds is streamed to both the terminal and `tmac/logs/setup/console.log`. A failed retry preserves downloaded source, extracted LLVM, the TVM build directory, and the Kunpeng patch.

Source preparation is local-first. Setup checks the pinned main and submodule objects with `git cat-file -e <commit>^{commit}`. It skips fetch and submodule update when those objects already exist, then checks out and verifies every exact SHA. Missing objects use temporary HTTP/1.1 with three attempts and a 180-second timeout per attempt. Complete stdout and stderr from every failed attempt are printed and retained in the setup log.

`baseline` generates and tunes official kernels, builds B0 and C0 from the same pinned llama.cpp, builds the two harness programs, runs the operator matrix, compares B0/C0 logits, and runs C1 from the configured existing build. It selects one allowed logical CPU per physical core on NUMA node 0 and uses `taskset`; if `numactl` exists it also binds memory. It does not assume consecutive CPU IDs. B0, C0, and C1 are interleaved in each round. C1 remains a cross-version reference.

Results are written to `results/tmac_baseline_TIMESTAMP/`, including the required manifest, environment, raw CSV files, dispatch evidence, console log, Chinese summary, and a small `send_to_chatgpt.tar.gz`. A failure or timeout still reaches result packaging. The archive excludes the model, repositories, generated kernels, and build trees.

## Operator and correctness harnesses

`harness/operator.cpp` uses ggml's pinned GGUF reader to find a real Q4_0 tensor with the requested `(O,K)` shape and reads only that tensor payload. It calls the pinned `ggml_tmac_transform_tensor`, builds LUTs through `ggml_tmac_mul_mat_task_init`, and covers every output tile through `ggml_tmac_mul_mat_task_compute` using the exact `n_tile_num`, qweight offset, and scale offset rules from the fixed fork. Allocation, file input, transformation, and warmup are outside steady-state timing. The full interval independently measures required activation conversion, LUT construction, all tile calls, and output conversion/readback.

The harness checks sampled output coordinates against an independent computation over the real Q4_0 nibbles, scales, generated quantized LUT, LUT scales, and LUT biases. It separately reports error against a direct high-precision accumulation using the original synthetic float activation. Activations use the recorded fixed seed and uniform `[-4,4]` distribution.

`harness/logits.cpp` loads the same GGUF through the public pinned llama API, evaluates a fixed token sequence, and writes the final-token logits. The runner executes it once with B0 and once with C0, checks B0 dispatch evidence, and records NMSE and maximum absolute difference. It does not use generated text as a correctness test.

## Local and server status

The local reference test independently checks Q4_0 nibble order, signed logical values, LUT index ordering, bit-plane coefficients, scale mapping, group boundaries, zero input, sign changes, extremes, NaN/Inf rejection, and a fixed random case. Run it with:

```sh
bash tmac_experiment/run.sh local-test
```

The C++ harnesses are compiled and executed by `baseline`; no extra manual command is required. On non-AArch64 development machines only the scalar tests run. ARM kernel correctness, operator timings, logits comparison, dynamic loading, NUMA binding, and end-to-end measurements remain pending until the server run. A failed configuration is written with `status=failed`; it is not accepted as a performance result. Model dispatch is accepted only when B0 output contains `ggml_tmac_init`, has no missing-kcfg fatal message, and exits successfully.

Directories for future `baseline`, `A`, `B`, and `AB` variants are reserved in configuration only. No A/B algorithm change is implemented here.
