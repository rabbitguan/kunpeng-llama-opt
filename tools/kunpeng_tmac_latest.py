#!/usr/bin/env python3
"""Port frozen original T-MAC kernels to the c622f62 baseline and run its protocol."""
import argparse
import array
import configparser
import copy
import csv
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
PIN = "c622f623fca9f0f085eb82f38874d1ab4a1791cd"
PATCH = ROOT / "tmac_experiment/patches/0003-c622f62-tmac.patch"
CASES = {"pp512": (512, 0, 0), "pp2048": (2048, 0, 0),
         "tg128": (0, 128, 0), "tg128_d1024": (0, 128, 1024)}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run(command, log, *, cwd=None, env=None, output=None):
    command = [str(value) for value in command]
    with log.open("a") as errors:
        print("$ " + shlex.join(command), file=errors, flush=True)
        if output is None:
            subprocess.run(command, cwd=cwd, env=env, stdout=errors, stderr=errors, check=True)
        else:
            with output.open("w") as stream:
                subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=errors, check=True)


def dispatch(log, threads=None):
    text = log.read_text(errors="replace")
    packed = set(re.findall(r"tmac: packed tensor=(\S+)", text))
    computed = set(re.findall(r"tmac: compute tensor=(\S+)", text))
    if not packed or packed != computed:
        raise ValueError(f"Missing T-MAC operator coverage in {log}: {sorted(packed - computed)}")
    if threads is not None and not re.search(rf"tmac: compute .* threads={threads} kernel_N=[12]\b", text):
        raise ValueError(f"T-MAC did not execute at {threads} threads in {log}")
    return sorted(computed)


def compare_logits(reference, actual):
    def read(path):
        data = path.read_bytes()
        if len(data) < 8:
            raise ValueError("Truncated logits")
        magic, count = struct.unpack("=II", data[:8])
        if magic != 0x544d4c47 or not count or count % 9 or len(data) != 8 + count * 4:
            raise ValueError("Invalid logits header or dimensions")
        values = array.array("f")
        values.frombytes(data[8:])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite model logits")
        return values
    ref, got = read(reference), read(actual)
    if len(ref) != len(got):
        raise ValueError("Logits dimensions differ")
    records = []
    width = len(ref) // 9
    for index in range(9):
        a, b = ref[index * width:(index + 1) * width], got[index * width:(index + 1) * width]
        norm = sum(value * value for value in a)
        error = sum((x - y) ** 2 for x, y in zip(a, b))
        maximum = max(abs(x - y) for x, y in zip(a, b))
        nmse = error / norm if norm else (0.0 if not error else math.inf)
        records.append({"threads": (1, 16, 32)[index // 3], "stage": ("prefill7", "decode", "decode_d1024")[index % 3],
                        "nmse": nmse if math.isfinite(nmse) else None, "max_abs": maximum,
                        "status": "pass" if nmse <= 0.001 and maximum <= 1.0 else "failed"})
    return records


def validate_baseline(frozen):
    if frozen.get("protocol") != 1 or frozen.get("mode") != "baseline" or frozen.get("commit") != PIN:
        raise ValueError("Choose a completed official c622f62 baseline")
    profile = frozen["profile"]
    if profile["threads"] != [1, 16, 32] or profile["cases"] != {key: list(value) for key, value in CASES.items()}:
        raise ValueError("Baseline must contain the requested 12 generation cases")
    selector = frozen.get("q4_kernel_requested")
    if selector not in ("auto", "4x4") or bool(frozen.get("upstream_selector_modified")) != (selector == "4x4"):
        raise ValueError("Choose an official auto baseline or its recorded 4x4 selector variant")
    return profile


def select_benchmarks(profile, threads=None, cases=None):
    threads = profile["threads"] if threads is None else threads
    cases = list(profile["cases"]) if cases is None else cases
    if not threads or len(set(threads)) != len(threads) or any(value not in profile["threads"] for value in threads):
        raise ValueError("Select distinct baseline thread counts from 1, 16, 32")
    if not cases or len(set(cases)) != len(cases) or any(value not in profile["cases"] for value in cases):
        raise ValueError("Select distinct baseline case names")
    selected = copy.deepcopy(profile)
    selected["threads"] = [value for value in profile["threads"] if value in threads]
    selected["cases"] = {key: value for key, value in selected["cases"].items() if key in cases}
    return selected


def execution_profile(saved, allow_host_change, numa_node=None):
    profile = copy.deepcopy(saved)
    hostname = platform.node()
    if numa_node is not None and numa_node < 0:
        raise ValueError("NUMA node must be nonnegative")
    if saved["hostname"] != hostname and not allow_host_change:
        raise ValueError(f"Baseline host is {saved['hostname']}; current host is {hostname}. Use --allow-host-change to remeasure both variants here")
    if saved["hostname"] != hostname or numa_node is not None:
        node_selected = saved["node"] if numa_node is None else numa_node
        allowed = os.sched_getaffinity(0)
        topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
        physical = {}
        for line in topology.splitlines():
            if not line or line.startswith("#"):
                continue
            cpu, core, socket, node = line.split(",")
            if node == str(node_selected) and core != "-" and socket != "-" and int(cpu) in allowed:
                key = (socket, core)
                physical[key] = min(physical.get(key, int(cpu)), int(cpu))
        cpus = sorted(physical.values())
        if len(cpus) < 32:
            raise ValueError(f"Current NUMA node {node_selected} has only {len(cpus)} allowed physical cores; 32 required")
        profile.update(hostname=hostname, node=node_selected, cpus=cpus[:32])
    if len(profile["cpus"]) != 32 or not set(profile["cpus"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("Saved physical CPU binding is unavailable")
    return profile


def tmac_package_flags(prefix):
    return [f"-DTMAC_DIR={prefix / 'lib/cmake/t-mac'}"]


def tmac_package_config():
    return (
        'get_filename_component(_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)\n'
        'set(TMAC_KERNELS_SOURCE "${_prefix}/lib/kernels.cc")\n'
        'add_library(t_mac_no_tvm INTERFACE)\n'
        'target_include_directories(t_mac_no_tvm INTERFACE "${_prefix}/include")\n'
        'include(CheckCXXSourceCompiles)\n'
        'check_cxx_source_compiles("int main() { _Float16 value = 0; return sizeof(value) != 2; }" TMAC_HAS_FLOAT16)\n'
        'if(NOT TMAC_HAS_FLOAT16)\n'
        '    target_compile_definitions(t_mac_no_tvm INTERFACE TMAC_HALF_STORAGE_FALLBACK=1)\n'
        'endif()\n')


def half_storage_source(source):
    anchor = "typedef _Float16 half;"
    if not source.count(anchor):
        raise ValueError("Expected generated T-MAC half typedefs")
    return source.replace(anchor,
        "#ifdef TMAC_HALF_STORAGE_FALLBACK\n"
        "typedef __fp16 half;\n"
        "#else\n"
        "typedef _Float16 half;\n"
        "#endif\n"
        'static_assert(sizeof(half) == 2, "T-MAC requires 16-bit half storage");')


def compiled_half_storage(cache_path):
    key = b"TMAC_HAS_FLOAT16:INTERNAL="
    values = [line[len(key):] for line in cache_path.read_bytes().splitlines() if line.startswith(key)]
    if len(values) != 1 or values[0] not in (b"1", b"0", b""):
        raise ValueError(f"Missing or invalid TMAC_HAS_FLOAT16 probe result in {cache_path}")
    return "_Float16" if values[0] == b"1" else "__fp16"


def neon_broadcast_source(source):
    anchor = "inline int32_t tbl_g4_int8_float_update_impl("
    if source.count(anchor) != 1:
        raise ValueError("Expected one original T-MAC lookup microkernel")
    start = source.index("#ifdef __ARM_NEON", source.index(anchor))
    end = source.index("#elif defined __AVX2__", start)
    body = source[start:end]
    replacements = {
        "((vs) * lut_s)": "((vs) * vdupq_n_f16(lut_s))",
        "((vs) * lut_s + lut_b)": "((vs) * vdupq_n_f16(lut_s) + vdupq_n_f16(lut_b))",
        "((cs) + zs * partial_sum)": "((cs) + zs * vdupq_n_f16(partial_sum))",
    }
    for index in range(4):
        replacements[f"vec_c{index} * vec_s)"] = f"vec_c{index} * vdupq_n_f16(vec_s))"
    for original, replacement in replacements.items():
        if body.count(original) != 1:
            raise ValueError(f"Unexpected T-MAC NEON expression: {original}")
        body = body.replace(original, replacement)
    return source[:start] + body + source[end:]


def execute(args):
    frozen = json.loads((args.baseline / "manifest.json").read_text())
    profile = validate_baseline(frozen)
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Run on the Linux AArch64 baseline server")
    profile = execution_profile(profile, args.allow_host_change, getattr(args, "numa_node", None))
    profile = select_benchmarks(profile, getattr(args, "threads", None), getattr(args, "cases", None))
    scope = "full" if profile["threads"] == frozen["profile"]["threads"] and profile["cases"] == frozen["profile"]["cases"] else "subset"
    fused = getattr(args, "fused", False)
    print(f"Benchmark scope: {scope}; threads={profile['threads']}; cases={list(profile['cases'])}. All correctness gates remain enabled.", flush=True)
    if profile["hostname"] != frozen["profile"]["hostname"]:
        print(f"Host changed: {frozen['profile']['hostname']} -> {profile['hostname']}. Both variants will be remeasured; historical throughput is not used.", flush=True)
        print(f"NUMA node {profile['node']}, current physical CPUs: {profile['cpus']}", flush=True)
    model = Path(profile["model"])
    if digest(model) != profile["model_sha256"]:
        raise ValueError("Model checksum differs")
    historical = json.loads((args.tmac_baseline / "manifest.json").read_text())
    if historical.get("status") != "preliminary_pass" or historical.get("opts") != [] or historical.get("model_sha256") != profile["model_sha256"]:
        raise ValueError("Choose the successful original T-MAC baseline for this same model")
    if historical.get("tuning") != "disabled_official_fallback":
        raise ValueError("Expected the original untuned fallback baseline")
    generated = args.tmac_baseline / "generated"
    for name in ("kernels.cc", "kernels.h", "kcfg.ini"):
        if digest(generated / name) != historical["generated_sha256"][name]:
            raise ValueError(f"Frozen generated {name} changed")
    cfg = configparser.ConfigParser()
    cfg.read(generated / "kcfg.ini")
    for m, k in ((4096, 4096), (14336, 4096), (4096, 14336), (1024, 4096)):
        section = f"qgemm_lut_t1_int8_m{m * 4}_k{k}_n1_b4"
        if cfg.getint(section, "group_size") != 32 or cfg.getint(section, "kfactor") != 8 or cfg.getint(section, "lut_scales_size") != k // 32:
            raise ValueError(f"Unsupported frozen kernel configuration: {section}")
    env = os.environ.copy()
    for key, value in profile["toolchain"]["env"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"):
        version = subprocess.check_output([profile["toolchain"][key], "--version"], text=True).splitlines()[0]
        if version != profile["toolchain"][key + "_version"]:
            raise ValueError(f"{key} version changed")
    env.update(TMAC_KCFG_FILE=str(generated / "kcfg.ini"), TVM_NUM_THREADS="1", OMP_DYNAMIC="FALSE",
               GGML_FUSED_CPP_SDPA=frozen.get("fused_sdpa_env", "1"))
    out = args.out
    out.mkdir(parents=True)
    env["GIT_CEILING_DIRECTORIES"] = str(out)
    log = out / "build.log"
    manifest = {"status": "preparing", "commit": PIN, "baseline": str(args.baseline), "profile": profile,
                "baseline_profile": frozen["profile"], "host_changed": profile["hostname"] != frozen["profile"]["hostname"],
                "historical_throughput_used": False, "benchmark_scope": scope,
                "tmac_baseline": str(args.tmac_baseline), "kernel_N": [1, 2] if fused else 1, "tuning": False,
                "patch_sha256": digest(PATCH), "generated_sha256": historical["generated_sha256"],
                "fused_consumer": fused, "fused_header_sha256": digest(ROOT / "ggml/src/ggml-cpu/tmac/fused.h") if fused else None,
                "fused_layout": "output_block_k_v1" if fused else None,
                "numa_node_override": getattr(args, "numa_node", None),
                "limitations": (["fused consumer supports the BM=256,simd_in=16,simd_out=8,kfactor=8 layout; others use N=1 generated kernels"] if fused else ["N=1 generated kernels also process prefill"]) + ["raw and packed weights both resident", "fixed-token smoke is not perplexity evaluation"]}
    selector_patch = args.baseline / "q4-4x4.patch" if frozen.get("upstream_selector_modified") else None
    manifest["baseline_selector_patch_sha256"] = digest(selector_patch) if selector_patch else None
    write(out / "manifest.json", manifest)
    run(["lscpu"], out / "cpu-topology.log", env=env)
    run(["numactl", "--physcpubind=" + ",".join(map(str, profile["cpus"])), f"--membind={profile['node']}", "true"], log, env=env)
    source = Path(frozen["source"])
    archive = out / "source.tar"
    run(["git", "-C", source, "archive", "--format=tar", "--output", archive, PIN], log)
    trees = {}
    for variant in ("native", "tmac"):
        tree = out / (variant + "-source")
        tree.mkdir()
        run(["tar", "-xf", archive, "-C", tree], log)
        if selector_patch:
            run(["git", "apply", selector_patch], log, cwd=tree, env=env)
        if variant == "tmac":
            run(["git", "apply", PATCH], log, cwd=tree, env=env)
        trees[variant] = tree
    archive.unlink()
    prefix = out / "tmac-prefix"
    (prefix / "include/t-mac").mkdir(parents=True)
    (prefix / "lib/cmake/t-mac").mkdir(parents=True)
    header = args.tmac_baseline / "install/include/t-mac/INIReader.h"
    shutil.copy2(header, prefix / "include/t-mac/INIReader.h")
    manifest["inireader_sha256"] = digest(header)
    write(out / "manifest.json", manifest)
    shutil.copy2(generated / "kernels.h", prefix / "include/t-mac/kernels.h")
    original_kernel = (generated / "kernels.cc").read_text()
    compiled_kernel = half_storage_source(neon_broadcast_source(original_kernel))
    (prefix / "lib/kernels.cc").write_text(compiled_kernel)
    (out / "neon-broadcast.patch").write_text("".join(difflib.unified_diff(
        original_kernel.splitlines(keepends=True), compiled_kernel.splitlines(keepends=True),
        fromfile="original/kernels.cc", tofile="compiled/kernels.cc")))
    manifest["kernel_compatibility"] = "explicit_neon_fp16_scalar_broadcasts"
    manifest["half_storage_compatibility"] = "compiler_probe_with_arm_fp16_fallback"
    manifest["compiled_kernel_sha256"] = digest(prefix / "lib/kernels.cc")
    write(out / "manifest.json", manifest)
    shutil.copy2(generated / "kcfg.ini", prefix / "lib/kcfg.ini")
    env["TMAC_KCFG_FILE"] = str(prefix / "lib/kcfg.ini")
    (prefix / "lib/cmake/t-mac/TMACConfig.cmake").write_text(tmac_package_config())
    binaries = {}
    for variant, tree in trees.items():
        build = out / (variant + "-build")
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DGGML_CCACHE=OFF", "-DGGML_OPENMP=ON", "-DGGML_CPU_REPACK=ON", "-DGGML_BLAS=OFF", "-DGGML_CPU_KLEIDIAI=OFF", "-DLLAMA_CURL=OFF"]
        flags += [f"-D{key}={profile['toolchain'][key]}" for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")]
        flags += ["-DGGML_NATIVE=OFF", f"-DGGML_CPU_ARM_ARCH={profile['arm_arch']}"] if profile.get("arm_arch") else ["-DGGML_NATIVE=ON"]
        flags += [f"-DLLAMA_SOURCE={tree}", f"-DGGML_TMAC={'ON' if variant == 'tmac' else 'OFF'}", *tmac_package_flags(prefix)]
        flags += [f"-DTMAC_FUSED={'ON' if fused and variant == 'tmac' else 'OFF'}"]
        run(["cmake", "-S", ROOT / "tmac_experiment/latest", "-B", build, *flags], log, env=env)
        binaries[variant] = build
    manifest["compiled_half_storage"] = compiled_half_storage(binaries["tmac"] / "CMakeCache.txt")
    write(out / "manifest.json", manifest)
    for variant in ("tmac", "native"):
        build = binaries[variant]
        targets = ["llama-bench", "tmac-latest-logits"] + (["tmac-latest-check"] + (["tmac-fused-check"] if fused else []) if variant == "tmac" else [])
        print(f"Building {variant}; log: {log}", flush=True)
        run(["cmake", "--build", build, "--target", *targets, "--parallel", args.jobs], log, env=env)
    def affinity(threads):
        return ["numactl", "--physcpubind=" + ",".join(map(str, profile["cpus"][:threads])), f"--membind={profile['node']}"]
    if fused:
        print("Checking fused kernel correctness", flush=True)
        run([*affinity(1), binaries["tmac"] / "tmac-fused-check"], out / "fused-check.log", env=env, output=out / "fused-check.json")
        if json.loads((out / "fused-check.json").read_text()) != {"status": "pass", "cases": 50, "bitwise_equal": True}:
            raise ValueError("Fused consumer correctness gate failed")
    print("Checking operator correctness", flush=True)
    run([*affinity(32), binaries["tmac"] / "tmac-latest-check"], out / "operator.log", env=env, output=out / "operator.jsonl")
    dispatch(out / "operator.log")
    for variant, build in binaries.items():
        print(f"Checking model logits: {variant}; log: {out / (variant + '-logits.log')}", flush=True)
        run([*affinity(32), build / "tmac-latest-logits", model, out / (variant + "-logits.bin")], out / (variant + "-logits.log"), env=env)
    coverage = dispatch(out / "tmac-logits.log")
    correctness = compare_logits(out / "native-logits.bin", out / "tmac-logits.bin")
    write(out / "correctness.json", correctness)
    if any(row["status"] != "pass" for row in correctness):
        raise ValueError("Model logits gate failed; performance comparison stopped")
    rows = []
    for threads in profile["threads"]:
        env["OMP_NUM_THREADS"] = str(threads)
        for case, (pp, tg, depth) in profile["cases"].items():
            for variant in ("native", "tmac"):
                name = f"{variant}-{case}-t{threads}"
                print(f"Benchmarking {name}", flush=True)
                command = [*affinity(threads), binaries[variant] / "bin/llama-bench", "-m", model,
                           "-ngl", "0", "-fa", profile["flash_attn"], "-ctk", "f16", "-ctv", "f16",
                           "-b", profile["batch"], "-ub", profile["ubatch"], "-t", threads, "-lm", "mmap",
                           "-p", pp, "-n", tg, "-d", depth, "-r", profile["reps"], "-o", "json", "-v"]
                run(command, out / (name + ".log"), env=env, output=out / (name + ".json"))
                if variant == "tmac" and dispatch(out / (name + ".log"), threads) != coverage:
                    raise ValueError("T-MAC tensor coverage changed")
                records = json.loads((out / (name + ".json")).read_text())
                if len(records) != 1 or tuple(records[0][key] for key in ("n_threads", "n_prompt", "n_gen", "n_depth")) != (threads, pp, tg, depth):
                    raise ValueError("Benchmark case mismatch")
                value = float(records[0]["avg_ts"])
                deviation = float(records[0]["stddev_ts"])
                if not math.isfinite(value) or value <= 0 or not math.isfinite(deviation) or deviation < 0:
                    raise ValueError("Invalid throughput")
                rows.append({"variant": variant, "case": case, "threads": threads, "avg_tps": value,
                             "stddev_tps": deviation})
                with (out / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    comparison = []
    for native, tmac in zip(rows[::2], rows[1::2]):
        comparison.append({"case": native["case"], "threads": native["threads"], "native_tps": native["avg_tps"],
                           "tmac_tps": tmac["avg_tps"], "tmac_over_native": float(tmac["avg_tps"]) / float(native["avg_tps"])})
    write(out / "comparison.json", comparison)
    if digest(model) != profile["model_sha256"]:
        raise ValueError("Model changed during measurement")
    manifest.update(status="smoke_and_benchmark_pass", tensor_coverage=coverage,
                    binaries_sha256={key: digest(build / "bin/llama-bench") for key, build in binaries.items()})
    write(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="completed official c622f62 Q4_0 baseline directory")
    parser.add_argument("--tmac-baseline", required=True, type=Path, help="successful original basic_baseline directory")
    parser.add_argument("--out", required=True, type=Path, help="new result directory")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--allow-host-change", action="store_true", help="remeasure both variants on this host using 32 allowed physical cores from the saved NUMA node")
    parser.add_argument("--numa-node", type=int, help="bind both variants' CPUs and memory to this NUMA node")
    parser.add_argument("--fused", action="store_true", help="use the fused one/two-row T-MAC consumer after its bitwise correctness gate")
    parser.add_argument("--threads", nargs="+", type=int, choices=(1, 16, 32), help="benchmark only these thread counts; correctness still covers all three")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), help="benchmark only these baseline cases; correctness gates remain enabled")
    args = parser.parse_args()
    args.baseline, args.tmac_baseline, args.out = (path.resolve() for path in (args.baseline, args.tmac_baseline, args.out))
    if args.out.exists() or args.jobs < 1:
        parser.error("--out must be new and --jobs must be positive")
    try:
        execute(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, configparser.Error) as error:
        if args.out.exists():
            write(args.out / "failure.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
            manifest_path = args.out / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest.update(status="failed", error=str(error))
                write(manifest_path, manifest)
        print(f"Failed: {error}", file=sys.stderr)
        location = traceback.extract_tb(error.__traceback__)[-1]
        print(f"Failure location: {location.filename}:{location.lineno}: {location.line}", file=sys.stderr)
        build_log = args.out / "build.log"
        if isinstance(error, subprocess.CalledProcessError) and build_log.exists():
            lines = build_log.read_text(errors="replace").splitlines()
            matches = [index for index, line in enumerate(lines)
                       if re.search(r"error:|CMake Error|undefined reference|Error:|Killed|No rule to make target", line)]
            if matches:
                index = matches[0]
                print(f"First build error from {build_log}:", file=sys.stderr)
                print("\n".join(lines[max(0, index - 3):index + 10]), file=sys.stderr)
        return 1
    finally:
        if args.out.exists():
            with tarfile.open(args.out / "logs.tar.gz", "w:gz") as archive:
                paths = [path for path in args.out.iterdir() if path.suffix in (".log", ".json", ".jsonl", ".csv", ".bin")]
                for build in (args.out / "native-build", args.out / "tmac-build"):
                    paths.extend(build.rglob("CMakeCache.txt"))
                    paths.extend(build.rglob("compile_commands.json"))
                for path in paths:
                    archive.add(path, arcname=str(path.relative_to(args.out)))
                config_path = args.out / "tmac-prefix/lib/kcfg.ini"
                if config_path.exists():
                    archive.add(config_path, arcname="kcfg.ini")
                compatibility_patch = args.out / "neon-broadcast.patch"
                if compatibility_patch.exists():
                    archive.add(compatibility_patch, arcname=compatibility_patch.name)
                archive.add(PATCH, arcname=PATCH.name)
            print(f"Logs archive: {args.out / 'logs.tar.gz'}")
    print(f"Results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
