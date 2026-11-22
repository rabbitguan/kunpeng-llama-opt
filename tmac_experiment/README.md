# Kunpeng T-MAC baseline experiment

## Original T-MAC on the c622f62 baseline

Run `python3 tools/kunpeng_tmac_latest.py --baseline /absolute/path/to/official-Q4-baseline --tmac-baseline /absolute/path/to/basic_baseline_RUN --out /absolute/path/to/new-results`. The official baseline must be a completed generation run from `bench-llama-baseline.py` at `c622f623fca9f0f085eb82f38874d1ab4a1791cd`, containing threads 1,16,32 and pp512, pp2048, tg128, tg128_d1024. Both the auto selector and its recorded official 4x4 selector patch are supported. The old T-MAC directory must be the successful original untuned baseline for the same Q4_0 GGUF. No server dependencies are downloaded or regenerated.

Host changes require `--allow-host-change`. This mode keeps the saved model, source, compiler and workload settings, but selects 32 currently allowed physical cores from the saved NUMA node, with one hardware thread per core. Fewer than 32 available cores fail explicitly. Both native and T-MAC are remeasured on the new host; old throughput is never used in their speedup ratio. The manifest preserves the original profile separately from the execution profile, records the host change, and includes the current CPU topology in the logs archive. Without this option, the saved hostname and CPU binding are required.

The runner archives the exact official commit into separate native and T-MAC source directories, preserves a recorded 4x4 selector patch on both, and applies `patches/0003-c622f62-tmac.patch` only to T-MAC. It verifies the frozen generated kernel hashes before copying them into the new result directory. Compiler versions, compiler environment, ISA request, CPU/NUMA binding, batch/ubatch, repetitions, Flash Attention and FP16 KV settings come from the saved official baseline. Both variants are built and remeasured; the old T-MAC throughput is not used to calculate a new speedup. The old source, build and result directories remain read only.

The original NEON microkernel uses scalar FP16 variables directly in vector arithmetic, which Clang accepts but GCC rejects after scalar promotion. In the new kernel copy, the runner changes exactly seven expressions to explicit `vdupq_n_f16` broadcasts, including the zero-point and single-scale branches that GCC parses even when not instantiated. The AVX2 branch, weights, group boundaries, arithmetic order and kernel schedule are preserved. Unexpected source patterns fail explicitly. The input hashes, compiled kernel hash and `neon-broadcast.patch` are recorded separately. Local ARM64 checks of the pinned microkernel covered 288 cases across three BM sizes, random inputs, signed scales and the normal/zero-point/single-scale branches. Original Clang, adapted Clang and adapted GCC 15 builds produced identical output bytes in these cases; Linux GCC model checks remain pending.

CMake also probes whether the selected C++ compiler supports `_Float16`. If not, every generated `half` typedef uses ARM `__fp16` storage instead, with a two-byte size assertion. Compilers that support `_Float16` keep that type. The selected storage type is recorded in the manifest, and the source diff includes the conditional declarations. Local CMake regression checks compile and run both the native and forced fallback routes. Server operator and model gates remain required before accepting timings.

The CPU_TMAC buffer uses the current extra-buffer and tensor-traits interfaces ahead of CPU_REPACK. Only contiguous two-dimensional Q4_0 weights with a supported kernel configuration are selected. Embeddings and the output head retain the native path. Raw weights remain available for readback alongside the packed weights and scales, so resident weight memory increases. Packing occurs during loading, outside steady-state benchmark timing. Activation conversion, LUT construction, synchronization, all tile calls and output conversion occur inside operator execution. The current frozen Llama-3 preset contains N=1 kernels: prefill schedules token/tile tasks using these same kernels. This is not a reproduction of a separately generated N>1 GEMM configuration. No A/B/C optimizations, tuning, new quantization types or SVE kernel rewrite are included.

Before performance measurement, `tmac-latest-check` verifies a real 4096x4096 generated kernel against an independent Q4_0 mathematical reference at 1/7/32 input rows and 1/16/32 threads. Thread counts must produce identical output bytes. Model smoke checks then compare native and T-MAC logits for prefill, decode and decode after 1024 history tokens at all three thread counts. Each stage separately requires finite values, NMSE <= 0.001 and max-absolute error <= 1.0. These are fixed smoke gates, not a perplexity evaluation. Failures stop the performance comparison. Every timed T-MAC process must confirm execution of every tensor packed by that process; initialization alone is insufficient.

Results include raw benchmark JSON, `summary.csv`, `comparison.json`, correctness records, logits, build metadata and `logs.tar.gz`, including on failures after the result directory is created. `tmac_over_native` divides T-MAC throughput by the contemporaneous native throughput. The original cached baseline is retained as context. Local ARM64 compilation and generated-kernel operator checks do not establish Linux/Kunpeng model correctness or performance; those require running this command on the server. The integration patch was checked against the exact official target files; full c622f62 server builds and model measurements remain pending.

## Retest the earlier v0.6.0 repack baseline

Run `python3 tools/kunpeng_llama_baseline.py`. This short follow-up snapshots the earlier selector commit `a8efd95389352e668d3e234a0243ab08f1d8ba18` from the local project Git history. It is a pinned cross-version engineering comparison, not a latest-upstream claim or a measurement isolating T-MAC alone. Historical GCC 14.3 tg128 results remain background evidence; this run uses the validated project Clang 17.0.6 and matches the T-MAC tg32 protocol.

The build enables runtime CPU repack and OpenMP, requests llamafile, disables BLAS and KleidiAI, and uses explicit `armv8.6-a+sve+dotprod+i8mm+fp16` with `GGML_NATIVE=OFF` so the requested ISA is deterministic. This revision undefines `GGML_USE_LLAMAFILE` when SVE or I8MM is compiled; its actual runtime value is recorded rather than claimed enabled. Actual optimized CPU compile commands, preprocessor feature macros and runtime CPU features must pass checks. The existing runtime selector forces 4x4 and 8x8 in separate processes; 8x8 is skipped explicitly if the current SVE vector length is not 32 bytes. Both model checks and benchmarks must log the selected layout and a CPU_REPACK model buffer, so silent fallback fails.

Each layout first evaluates the same seven fixed tokens against the saved upstream logits, using the existing fixed NMSE <= 0.001 and max-absolute <= 1.0 smoke gates. Both GEMM and GEMV paths occur in that seven-token batch. Then it runs zero-prompt tg32, batch/micro-batch 32, FP16 KV, Flash Attention off, one thread, CPU 0 and node-0 memory binding. Four samples per process are saved; the first is warmup and three are retained. The original T-MAC is not rebuilt or remeasured; its historical result is reported separately with `same_run=false`, without a new T-MAC speedup ratio. No tuning, prefill benchmark, thread sweep or model file conversion is performed; the original GGUF checksum is checked before and after.

Results and failure archives are separate `llama_optimized_baseline_*` directories. They include raw benchmark JSONL, logits, compile commands, CMake cache and before/after CPU/cgroup/frequency/resource snapshots. These snapshots do not provide continuous load or frequency profiling, and CPU binding is not exclusive allocation. Model loading and runtime weight repacking occur outside the reported decode sample interval. Historical cross-run comparisons remain provisional.

## Short R=1 versus R=4 confirmation

Add `--r-list 1,4` to the larger-K reuse command to measure only those two controls. Reuse mode executes orders 1,4 and 4,1, five samples per process, retaining ten samples per configuration. It does not compile or load the full model. The same complete-output bitwise gate is active in each process. The summary now preserves per-pass medians and R=4/R=1 throughput ratios, so an anomalously slow R=1 pass is not hidden by a pooled median. Raw samples are never removed. When R=0 is not selected, `original_reference_measured` is false and `operator_speedups` is empty; this follow-up does not claim a new comparison against original T-MAC. Kernel sources, layout, rounding and installed binaries are unchanged.

## C follow-up on the larger K matrix without rebuilding

Use `python3 tools/kunpeng_c.py --k 14336 --reuse-run /absolute/path/to/completed/C_operator_compare_RUN`. The runner requires a successful C operator manifest with the same baseline, commits, model checksum and header hash; the installed generated-source hash must also match. Missing or mismatched artifacts fail instead of triggering a rebuild. The existing runtime C entry handles the requested K, and the original frozen kcfg is checked for its O=4096,K=14336,bm=256 kernel. The same real-GGUF tensor selection and full-output bitwise check remain active for this matrix. The previous run directory is read only; new raw results and a new manifest are written separately.

This matrix has 56 KiB of pure LUT data versus 16 KiB at K=4096. It is a more relevant additional test for the cache-reuse hypothesis, not a guarantee of benefit. Reuse mode runs R=0,1,2,4 and then 4,2,1,0, five measured rounds per process with three warmups. Each configuration still totals ten samples, and pass identifiers are retained. The summary reports R=2 and R=4 relative to R=1 as well as each variant relative to the original consumer. R=1 has no cross-tile joint reuse, so a gain shared with R=1 cannot by itself establish the intended cache-scheduling benefit. No model inference, compilation, kernel generation or tuning is performed in reuse mode. Model hashing and offline packing still occur.

The K=14336 metadata gate was checked against the actual returned baseline kcfg. Local ARM64 tests also compared the unchanged C kernel with the original tile-first reference for 81 larger-K layout/schedule/input cases. Server correctness and timing for the real larger matrix remain pending. This change updates only the Python runner, tests and documentation; the installed C kernel remains unchanged.

## Minimal C layout/schedule prototype

Run `python3 tools/kunpeng_c.py` on the server after the basic baseline. This is an operator-only C prototype; no A/B, model loading, decode, new thread policy or tuning is added. It compares the frozen tile-first path with R=1,2,4 at BK=256 on one real O=K=4096 Q4_0 tensor, with T=1, one CPU, three warmups and ten rounds. BM=256 is the expanded bit-plane dimension, corresponding to 64 logical output channels per tile. The actual frozen kcfg is checked before building; other layouts fail explicitly.

`optimizations/cache-schedule.h` keeps the original `tbl_g4_int8_float_update_impl<true,8,4,8,false,false,false>` microkernel and its FP16 per-group conversion, scale/bias and accumulation. A new consumer processes output superblocks, then K cache blocks, then output tiles within that superblock. Every output still visits its quantization groups in increasing order. The final bit-plane combination follows the original FP32 expression and converts to FP16. A cache block spans multiple 32-element groups but never merges their scales or changes kfactor=8.

The paired offline packer concatenates original 32-element weight blocks and their corresponding scale blocks in the new consumption order. Microkernel-internal bytes and scale values are copied unchanged and verified. R=1 retains the original byte order. The v1 layout identifier binds K, BK, R and tile count; BM=256, group=32, bits=4 and 64-byte alignment are fixed properties of v1. The consumer rejects a mismatching identifier. The manifest also binds the model checksum, kernel/header hashes and parameters; no persistent weight cache is created. Weight, scale and partial buffers are 64-byte aligned to avoid an unrelated alignment change. The LUT layout and its scale/bias mapping remain original. K cache tails and fewer-than-R output tails are handled explicitly.

The implementation appends a separate cache-schedule entry to a copy of the frozen generated C source. Original qgemm entry points stay intact. The specialized test target links that code with the existing GGUF/operator harness; ordinary operator and model targets remain unchanged. Before timing each C case, every output must be finite and byte-identical to the original generated qgemm consumer on the same real matrix and activation. The existing sampled independent Q4 reference check also remains. Local tests cover 243 schedule/layout cases, including R=1/2/4, K tails, output tails, zero, signed and small-scale inputs, alignment, canaries and wrong layout identifiers. Both the appended generated C and modified harness compile on local ARM64; Linux linking, real-matrix bitwise equality and performance await server verification.

Packing runs once outside timing and is reported in CSV notes. Partial-result resets, all microkernel calls, final output conversion/write, activation conversion and LUT construction are included in full operator timing. Additional partial-buffer bytes are reported; `workspace_bytes` counts explicit buffers, not the original kernel's stack frame. Packed weight and scale allocation sizes are also recorded separately. BK=256 is an initial candidate, not an optimum. Its pure LUT segment is 1024 bytes; full K=4096 tables occupy 16 KiB, excluding scale/bias. Capacity estimates alone do not establish a cache bottleneck or a speedup.

Results appear under `C_operator_compare_*` with raw CSV, source/config metadata, layout checks and an automatic logs archive. This stage does not provide full-model C dispatch or AC/BC/ABC combinations. These are deferred until the pure C operator prototype demonstrates a useful effect. B is currently set aside because all tested implementations were slower than the reference.

## Final small B attempt: INT16 merge

The unrolled B follow-up still took 753.181 us versus 627.8715 us for original T-MAC on the tested 4096x4096 operator. The next small change merges the four plane sums with INT16 multiply-adds and widens only the final result for FP32 conversion. Each sum is bounded by 1024 in magnitude, and the successive weighted bounds are 3072, 7168 and 15360, all below 32768. Products by 2, 4 and 8 also fit; no intermediate can overflow. Thus this integer narrowing preserves the exact merged value and the existing B-v1 FP32 numerical contract. Weight layout, group boundaries, floating-point scale/bias handling and scheduling are unchanged. The manifest labels this implementation `B-v3-int16-merge-unroll8`; generated-source and header hashes distinguish it from prior results.

Local ARM64 checks still pass all 420 cases against the independent scalar FP32 contract and double reference, and the 16 related Python tests pass. Local assembly shows the weighted combination using 16-bit shifts/adds before two final widening operations; Kunpeng performance remains pending. Use the same `python3 tools/kunpeng_a.py --opts B --operator-only` command. Only one real matrix is measured. If this version still has no reliable advantage over the reference, retain B as an unsuccessful experimental branch and stop adding B or full-model runs.

## B loop-unrolling follow-up

The first B server run passed the fixed correctness gates but slowed the 4096x4096 operator from 633.427 us to 897.015 us and decode from 3.3611 to 2.4910 tokens/s. Its probe assembly retained the eight-iteration inner loop and copied the eight LUT vectors to a stack array. The follow-up adds only full-unroll directives to the fixed eight-table load and lookup loops in `bitplane-fuse.h`. It does not change the B-v1 numerical contract, weight layout, scale/bias or group order. Local Apple ARM64 assembly now shows register-resident LUTs and expanded lookups; the FP32 partial-output array is still stack-backed. This is a code-generation observation, not Kunpeng speed evidence.

Run `python3 tools/kunpeng_a.py --opts B --operator-only` for this follow-up. It reuses a reference operator harness from a successful run with matching frozen baseline and commits when available; otherwise it builds one. It builds only the revised optimized operator harness, repeats the 420-case B check, and measures one real 4096x4096 Q4_0 tensor with T=1, three warmups and ten rounds per backend. Ten rounds add only milliseconds of measured work. It skips llama-bench builds, model logits and full-model decode. Results live under `B_operator_compare_*`; `operator_pass_model_pending` explicitly means the full model has not been checked. Existing frozen source and previous result directories are not modified. Return the automatically printed logs archive, which includes probe assembly, correctness check results and raw operator timings. Only if the operator improves should full-model timing be repeated.

## B and AB first implementation

Use the existing minimal runner with `python3 tools/kunpeng_a.py --opts B` to compare B alone against frozen T-MAC. `--opts A,B` selects the single AB combination; it does not run A and B as separate experiments. The default remains A. First run B only; no AB server measurement is required until B correctness and timings have been reviewed.

`optimizations/bitplane-fuse.h` implements the fused lookup/accumulation microkernel. `tools/kunpeng_b.py` specializes the frozen generated-C qgemm entry points, keeping their ABI, bm, kfactor, table ordering, scale mapping and output layout. This first implementation operates at the generated-C boundary rather than adding a TVM tensor intrinsic or changing upstream qgemm.py. It replaces the entire bit-plane reduction body with the explicit fused computation, leaving the original preprocessor and reference source/binary intact. No TVM regeneration or tuning is required. Unsupported quantization, kernel signatures and kcfg layouts fail explicitly; there is no silent B fallback. B and AB generated C compile checks were performed locally; actual Linux model integration remains a server check.

The supported contract is bits=4, g=4, weight/activation groups=32, kfactor=8, simd_n_in=16, simd_n_out=8, T=1, fast aggregation off and extra zero point off. BM is one of the official 4-bit candidates, K is a positive multiple of 32 up to 65536. The original Q4_0 bias remains. Each group has eight INT8 table terms per plane, bounded by 1024 in magnitude. Those partial sums use INT16; the weighted merge is explicitly INT32 with conservative bound 15360. No signed shifts are used for the merge. The kernel evaluates `fma(float(u0+2*u1+4*u2+8*u3), s, beta) * (float(d)*0.5)` per group, accumulates FP32 in original group order, and converts to FP16 once at final output. Contraction is disabled outside the explicit group FMA. This changes the original FP16 intermediate rounding and cross-group accumulation, so B does not claim bitwise equivalence. The group-outer/output-block-inner traversal and original packed byte format are retained; this is not C cache scheduling.

`harness/bitplane-fuse.cpp` independently packs raw 4-bit weights, computes scalar per-plane lookup sums, and compares the NEON result against both a scalar implementation of the FP32 contract (bitwise after final FP16 conversion) and a double mathematical reference. It checks 420 cases across BM=128/256/1024 and K=32/64/96/128/4096/14336, including zero, 1e-4 scales, positive/negative values and INT8 extremes, plus output canaries. The double-reference bound is `abs_error <= 0.0005*abs(reference)+0.0001`; nonfinite outputs fail. Maximum absolute and relative error with a 1e-4 denominator floor are recorded. This is a numerical kernel check, not performance evidence.

Before model timing, B/AB logits are compared with saved original T-MAC logits, using fixed gates NMSE <= 0.001 and maximum absolute error <= 1.0, with no NaN/Inf. Relative error with denominator floor 0.01 is recorded as well. These smoke thresholds are set before the first server result; failures are retained and stop valid timing summaries, not repaired by widening thresholds. The existing sampled real-operator gate remains unchanged. For A alone logits still must match bitwise. Full model quality, unseen inputs and broad shape coverage remain outside this minimal run.

The server runs the B kernel check first and saves `B-check.json` and `B-kernel.s`. After this, it retains the existing minimal one-matrix operator test, model logits check, and single-thread tg32 timing. Separate `B_compare_*` or `AB_compare_*` directories and logs archives isolate artifacts. There is still no speedup claim until the server returns actual results.

## Minimal A comparison

After a successful basic baseline, run `python3 tools/kunpeng_a.py`. It selects the latest completed `basic_baseline_*` directory, or accepts `--baseline /absolute/path`. It checks the model and frozen generated/installed kernel hashes and exact dependency commits. No source, generated file or executable in that baseline is modified. The new A tree copies the baseline T-MAC snapshot, applies `optimizations/A.patch`, and substitutes exactly one complete constructor in the frozen generated C source. Headers and kcfg are copied unchanged, so there is no regeneration, tuning or schedule change. The fork's existing build defaults and native optimizations are retained.

The runner first repeats the 1,600-case byte/scale/bias check. After building A, the fixed-token model logits must be byte-identical to the saved T-MAC baseline and finite before performance results are accepted. It also builds the existing operator harness for reference and A, using one real Q4_0 model tensor with O=4096,K=4096,T=1, two warmups and three rounds. The independent sampled-output reference uses the existing fixed error threshold. It records LUT construction, lookup/compute and full operator timing separately; the full interval includes activation conversion, LUT creation, all output tile calls and readback. Operator initialization and weight packing remain outside steady-state timing.

Decode uses the frozen reference binary and a separately built A binary. Each backend loads once for timing, executes four 32-token samples in that process, and discards the first sample before retaining three. This reduces repeated model loading compared with the initial baseline runner. The official one-token warmup also remains. Raw nanosecond samples are retained through JSONL output instead of rounding throughput to two decimal places. One thread, CPU/NUMA binding, zero prompt, batch sizes and FP16 KV match the basic baseline. A separate A logits invocation still loads the model once; the original baseline logits are reused. On the first server baseline, T-MAC initialization took about five minutes per model invocation, so the three model invocations here can still take roughly fifteen minutes plus builds and operator checks.

The result is an incremental A-versus-T-MAC comparison; the first llama.cpp result is retained without remeasurement. No B/C, multithread scan or prefill is added. Differences in operator-stage timing and decode are preliminary, not a guaranteed speedup. Results and failure logs are packaged automatically under `results/A_compare_*-logs.tar.gz`. Linux ARM builds and operator/full-model A checks remain pending until this command runs on the server.

## Minimal baseline run

After setup and LUT checks have passed, run `python3 tools/kunpeng_baseline.py` from the server Git tree. This builds only strict upstream llama.cpp and original T-MAC, with no A/B/C changes and no autotuning. It uses official T-MAC fallback parameters and the preset Llama-3-8B Q4 shapes; this is a preliminary untuned baseline, not the final tuned reference. Results are timestamped outside the repository and a small logs archive is always produced, including on failure.

The llama.cpp upstream pin is `dca1d4b58a7f1acf1bd253be84e50d6367f492fd`, the merge base of current upstream master and the pinned integrated fork `eb07ecf0172230d58fff5d23a3fd6feebda35065`. `baseline_sources.json` records the read-only GitHub comparison and all 14 fork commits above that base. Source snapshots use `git archive` of exact revisions; only the named Kunpeng target patch is applied to the T-MAC snapshot. LLVM/TVM/Python are reused from the validated setup. T-MAC's official generated-C/no-TVM-threadpool integration is selected; installing its headers and C source uses `TMAC_TVM=OFF` to avoid rebuilding TVM. The generated C source is still compiled into the fork with Clang. Upstream ARM flags and native optimization defaults remain enabled; OpenMP uses each version's defaults. Per-build CMake cache and compile_commands.json preserve the actual options.

The basic gate compares final logits for seven fixed Llama-3 token IDs between the two backends. Both outputs must be finite and their normalized squared error must be at most 0.01, a fixed initial smoke threshold. Raw logits and maximum absolute error are retained; this does not establish general model quality. Failure stops timing. T-MAC initialization and missing-kcfg messages are checked, but per-operator dispatch coverage remains unverified.

Timing uses one allowed physical CPU on the configured NUMA node, one thread, FP16 KV, no prompt, 32 generated tokens, one discarded warmup run and three measured runs per backend. Backends run sequentially. The runner records the same model SHA256, each round, medians, ms/token and preliminary throughput ratio. This is short zero-prompt decode, not long-context inference or prefill. Each invocation loads the model, while llama-bench's internal decode interval supplies the timing. Kernel generation, installation and model loading remain outside that interval. Detailed operator checks, model-derived kernel shapes, quota hierarchy auditing and per-operator coverage are deferred. Do not label the quick run as a fully frozen tuned tmac_ref.

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
