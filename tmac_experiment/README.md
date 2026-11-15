# Kunpeng T-MAC baseline experiment

## Current server layout (2026-11-15)

Use `/data/home/2300013234/kunpeng-llama-opt/repo` as the only Git working tree. The parent directory is an experiment workspace, not a Git repository. Keep the existing `tmac`, `setup`, `builds`, and `tmac_baseline_clean_source` directories in place because dependency and venv files can contain absolute paths. Models stay under `/data/home/2300013234/kunpeng-llm/models`. These server dependencies and compiled outputs are intentionally separate from tracked application sources.

The unified entry point now defaults to `paths.results_root/kunpeng`, outside the Git working tree. Override `--output` only when a separate run directory is needed. The config's `project_root` names the Git working tree; `tmac_root`, model and C1 build paths remain unchanged. Rerun setup after updating the config because its fingerprint includes configuration inputs.

After stopping old experiment jobs, run `bash tmac_experiment/cleanup_server.sh --list` to inspect the explicit deletion list and then `bash tmac_experiment/cleanup_server.sh --apply` to remove old experiment scripts, backups and results. The script preserves the Git tree and known environment directories. It rejects symlink deletion targets and scans selected directories for common Python, LLVM and TVM environment markers before deleting anything. This scan is a guard, not a complete dependency detector; retained legacy sources and builds should be audited separately before removal. No environment is moved. Old instructions below that upload scripts to the workspace root or clone `kunpeng-round1-git` are historical; use the existing `repo` tree instead.

Run from the Git tree:

```sh
cd /data/home/2300013234/kunpeng-llama-opt/repo
bash tmac_experiment/cleanup_server.sh --list
bash tmac_experiment/cleanup_server.sh --apply
python3 tools/kunpeng_exp.py probe
bash tmac_experiment/run.sh setup
python3 tools/kunpeng_exp.py check --source /data/home/2300013234/kunpeng-llama-opt/tmac/source
python3 tools/kunpeng_exp.py build --variants tmac_ref,A
```

Future updates use `git pull --ff-only origin experiment/tmac-baseline` in this directory. Keep local code edits in the development machine, and commit/push them before updating the server. Never use `git clean -fdx` for environment maintenance. Builds still require a fresh output directory when an identical snapshot already exists.

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

`setup` requires Git, CMake, Make, Python 3.8 through 3.11, and the existing project-local LLVM 17.0.6 toolchain. Clang is not required on the initial system `PATH`; setup validates and uses the fixed binaries under `tmac/source/build/`. It never downloads LLVM. It creates a local virtual environment under `tmac/`, verifies every used pinned commit, and applies the adaptation patches idempotently. It does not use sudo, Conda, perf, system library replacement, compiler aliases, or global shell changes.

If the project-local LLVM directory is incomplete, setup reports each missing or non-executable tool and does not start another download. It first looks for the fixed archive in `tmac/source/build/`, `tmac/source/`, `tmac/`, and `tmac/downloads/`; `TMAC_LLVM_ARCHIVE` can name another local archive. A valid local archive is extracted over the incomplete directory without deleting existing files, then all three LLVM tools are executed and checked again before the incremental TVM build starts.

Setup reads the selected LLVM's `--cxxflags`, `--ldflags`, `--libfiles`, `--system-libs`, and `--targets-built` outputs separately from stderr and checks them against the fixed TVM `FindLLVM.cmake`. Every zlib, zstd, libxml2, or tinfo dependency named by that actual link chain is compiled, linked, and executed with the selected Clang. Missing development files are built under `tmac/deps/<fingerprint>/` from the SHA256-verified source archives in `tmac_experiment/vendor/`; no platform binary is vendored. The archive origin, version, license identifier, embedded license path, and checksum are recorded in `vendor/manifest.json`.

TVM uses a build directory keyed by the source, patch, setup script, vendor archive, LLVM metadata, compiler identity, and dependency fingerprints. Its `config.cmake` is regenerated deterministically with libbacktrace, segfault backtraces, CUDA, OpenCL, Vulkan, CUTLASS, FlashInfer, and VTA disabled. Setup uses only `MANUAL_BUILD=1`; it never calls the upstream automatic LLVM download/build hook. Before TVM compilation it runs an LLVM AArch64 API link smoke. After compilation it executes a TVM LLVM operator and a small T-MAC Q4 NEON/FP16 kernel correctness smoke. `setup-environment.json` and `.setup-fingerprint` are written only after all checks pass and are consumed automatically by `baseline`.

After preparation, setup injects the project LLVM `bin`, TVM Python directory, LLVM/TVM library directories, and `TVM_ROOT` into every subprocess. It verifies `clang --version`, `clang++ --version`, `llvm-config --version`, the TVM shared libraries and CMake configuration, `import t_mac, tvm`, and a compiled and executed C++ smoke program. Only then does it write `.setup-fingerprint`. Output from downloads and long builds is streamed to both the terminal and `tmac/logs/setup/console.log`. A failed retry preserves downloaded source, extracted LLVM, the TVM build directory, and the Kunpeng patch.

Source preparation is local-first. Setup checks the pinned main and submodule objects with `git cat-file -e <commit>^{commit}`. It skips fetch and submodule update when those objects already exist, then checks out and verifies every exact SHA. Only `3rdparty/llama.cpp`, `3rdparty/tvm`, and TVM's `dlpack`, `dmlc-core`, and `rang` submodules are initialized. ExecuTorch and TVM's optional VTA, libbacktrace, CUTLASS, OpenCL, cnpy, FlashAttention, and FlashInfer submodules remain uninitialized; setup verifies their gitlink SHAs and records them as skipped. Missing required objects use temporary HTTP/1.1 with three attempts and a 180-second timeout per attempt. Complete stdout and stderr from every failed attempt are printed and retained in the setup log.

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

## Experimental A delivery (2026-11-13)

This is a partial implementation of the requested nine-variant experiment. Do not use it to claim a completed three-baseline comparison. The old `baseline` command and its B0/C0/C1 labels retain their original meaning.

New files:

- `../tools/kunpeng_exp.py`: probe, isolated T-MAC/A builds, local checks, source bundle and conservative raw-result collection. `bench` deliberately fails until per-variant operator/model correctness gates are connected. `--variants A,B,AB` selects three experiments; `--opts A,B` selects AB. Unsupported implementations fail explicitly.
- `optimizations/A.patch`: opt-in change to the pinned `lut_ctor.cc` only. Three transpose stages on each 8x8 half produce eight contiguous 16-byte stores. The quantization, scale/bias, loop bounds and output layout are unchanged. This patch is outside `patches/`, so legacy setup never applies it to the reference.
- `harness/lut-store.cpp`: actual original/optimized NEON comparison, including buffer canaries, zero, small, signed random and larger input, 32/64/96/128 activation spans. These spans exercise the original constructor's supported multiple-of-32 contract; this does not add arbitrary tails.

The unified build creates separate source snapshots, generated kernels, tuning directories, libraries and harnesses for `tmac_ref`, `A`, and `--opts ''` (unoptimized development control). LLVM, TVM, venv and read-only llama.cpp dependencies are reused. Rebuilding an existing snapshot fails; choose a new output directory. Both B0 and C0 helper binaries are produced internally by the reused builder; only B0 corresponds to the requested T-MAC variant. These C0 binaries are not `llama_ref`. All builds remain `built_pending_correctness`. Snapshot sources do not contain Git metadata; source commit provenance comes from verified original repositories. The builder rejects unexpected tracked source changes, but the two allowed adaptation files still require review against the saved patches before freezing a formal reference.

Confirmed versions: workspace `9f15bbedd2c67bfe0233e379dd27efa35450796a`, T-MAC `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`, integrated llama.cpp fork `eb07ecf0172230d58fff5d23a3fd6feebda35065`, TVM `36b9535ff364c484d04b384555106731049f44cd`; setup requires LLVM 17.0.6. The fixed T-MAC archive was downloaded and inspected locally. No local model file or complete server toolchain was found. The configured GGUF path is a server path; its metadata, shapes and SHA256 remain unverified locally.

Source locations in the pinned T-MAC tree: `python/t_mac/model_utils.py` and `tools/` handle conversion; the llama.cpp fork's `ggml/src/ggml-tmac.cpp` handles loading/packing and task dispatch; `python/t_mac/intrins/lut_ctor.cc` constructs LUTs; `python/t_mac/intrins/tbl.cc` handles lookup, integer partial sums, conversion and scaling; `python/t_mac/ops/qgemm.py` defines CBits, final bit-plane combination and scheduling; `deploy/compile.py` generates kernels. `N=T`, expanded `M=O*bits`. The kfactor filter requires both `(k*g) % act_group_size == 0` and `weight_group_size % (k*g) == 0`; of official candidates 8 and 16 only 8 survives at groups=32,g=4.

B is not implemented. The existing tensorized path converts/scales partial sums and accumulates FP16 CBits per group, followed by an FP32 bit-plane combination. Integer fusion requires a new tensorization contract and grouped output reduction, not a coefficient substitution. C is not implemented. Existing scheduling places CBits under output tile `mo` with inner K traversal; cache-block interleaving requires coordinated runtime task boundaries and a versioned packing layout. Consequently B/AB/C/AC/BC/ABC fail at build/check, and no performance result is accepted. Strict upstream `llama_ref` is also not implemented: the fork's exact upstream ancestor and non-TMAC modifications must be audited before selecting a matching upstream commit. Native CPU optimizations must remain enabled for that future baseline.

Local validation on Apple ARM64: original and A constructors compiled at `-O3 -march=armv8.2-a+fp16`, compared byte/scale/bias equality for 1,600 cases. Generated assembly for both is saved by `check`. This is neither Linux ABI verification nor Kunpeng performance evidence. A introduces transpose instructions and register pressure; store count in source is not a speedup measurement. Existing scalar/setup tests also pass. Generated TVM kernels, actual model metadata/weight equivalence, operator checks, logits checks, all-off equivalence, CPU binding/quota enforcement and formal performance experiments remain pending.

Commands for the implemented portion (replace `SERVER` with your SSH destination; use an independent server directory):

```sh
# Local
python3 tools/kunpeng_exp.py check --output /tmp/kunpeng-local
python3 tools/kunpeng_exp.py bundle --output /tmp/kunpeng-bundle
scp /tmp/kunpeng-bundle/kunpeng-experiment.tar.gz SERVER:~/kunpeng-experiment.tar.gz

# Server: existing project LLVM must be at the configured tmac/source/build path.
mkdir -p ~/kunpeng-round1
cd ~/kunpeng-round1
tar -xzf ~/kunpeng-experiment.tar.gz
python3 tools/kunpeng_exp.py probe --output results/environment
bash tmac_experiment/run.sh setup
python3 tools/kunpeng_exp.py check --source /data/home/2300013234/kunpeng-llama-opt/tmac/source --output results/check
python3 tools/kunpeng_exp.py build --variants tmac_ref,A --output results/build-round1
python3 tools/kunpeng_exp.py build --opts '' --output results/build-control
python3 tools/kunpeng_exp.py summarize --output results

tar -czf ~/kunpeng-round1-results.tar.gz results/environment results/check
# Local
scp SERVER:~/kunpeng-round1-results.tar.gz /tmp/
```

The archived config preserves existing server model/toolchain paths. Override those with `--set paths.model=...`, `--set paths.tmac_root=...`, and `--set paths.results_root=...` consistently for setup and build; setup validates configuration fingerprints. Do not replace system compilers or delete prior baselines/results. Build manifests are under `results/build-round1/variants/*/*/manifest.json`; return these and console logs as well, including failed runs. `check` with no `--source` runs scalar/setup tests only; even with `--source` it does not certify operator/model correctness. No executable formal bench command is supplied because the required gates and other variants are incomplete.
