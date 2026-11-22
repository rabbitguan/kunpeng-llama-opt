#!/usr/bin/env python3
"""Port frozen original T-MAC kernels to the c622f62 baseline and run its protocol."""
import argparse
import array
import configparser
import copy
import csv
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
    if threads is not None and not re.search(rf"tmac: compute .* threads={threads} kernel_N=1", text):
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


def execution_profile(saved, allow_host_change):
    profile = copy.deepcopy(saved)
    hostname = platform.node()
    if saved["hostname"] != hostname:
        if not allow_host_change:
            raise ValueError(f"Baseline host is {saved['hostname']}; current host is {hostname}. Use --allow-host-change to remeasure both variants here")
        allowed = os.sched_getaffinity(0)
        topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
        physical = {}
        for line in topology.splitlines():
            if not line or line.startswith("#"):
                continue
            cpu, core, socket, node = line.split(",")
            if node == str(saved["node"]) and core != "-" and socket != "-" and int(cpu) in allowed:
                key = (socket, core)
                physical[key] = min(physical.get(key, int(cpu)), int(cpu))
        cpus = sorted(physical.values())
        if len(cpus) < 32:
            raise ValueError(f"Current NUMA node {saved['node']} has only {len(cpus)} allowed physical cores; 32 required")
        profile.update(hostname=hostname, cpus=cpus[:32])
    if len(profile["cpus"]) != 32 or not set(profile["cpus"]).issubset(os.sched_getaffinity(0)):
        raise ValueError("Saved physical CPU binding is unavailable")
    return profile


def execute(args):
    frozen = json.loads((args.baseline / "manifest.json").read_text())
    profile = validate_baseline(frozen)
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Run on the Linux AArch64 baseline server")
    profile = execution_profile(profile, args.allow_host_change)
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
                "historical_throughput_used": False,
                "tmac_baseline": str(args.tmac_baseline), "kernel_N": 1, "tuning": False,
                "patch_sha256": digest(PATCH), "generated_sha256": historical["generated_sha256"],
                "limitations": ["N=1 generated kernels also process prefill", "raw and packed weights both resident", "fixed-token smoke is not perplexity evaluation"]}
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
    shutil.copy2(generated / "kernels.cc", prefix / "lib/kernels.cc")
    shutil.copy2(generated / "kcfg.ini", prefix / "lib/kcfg.ini")
    env["TMAC_KCFG_FILE"] = str(prefix / "lib/kcfg.ini")
    (prefix / "lib/cmake/t-mac/TMACConfig.cmake").write_text(
        'get_filename_component(_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)\n'
        'set(TMAC_KERNELS_SOURCE "${_prefix}/lib/kernels.cc")\n'
        'add_library(t_mac_no_tvm INTERFACE)\n'
        'target_include_directories(t_mac_no_tvm INTERFACE "${_prefix}/include")\n')
    binaries = {}
    for variant, tree in trees.items():
        build = out / (variant + "-build")
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DGGML_CCACHE=OFF", "-DGGML_OPENMP=ON", "-DGGML_CPU_REPACK=ON", "-DGGML_BLAS=OFF", "-DGGML_CPU_KLEIDIAI=OFF", "-DLLAMA_CURL=OFF"]
        flags += [f"-D{key}={profile['toolchain'][key]}" for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")]
        flags += ["-DGGML_NATIVE=OFF", f"-DGGML_CPU_ARM_ARCH={profile['arm_arch']}"] if profile.get("arm_arch") else ["-DGGML_NATIVE=ON"]
        flags += [f"-DLLAMA_SOURCE={tree}", f"-DGGML_TMAC={'ON' if variant == 'tmac' else 'OFF'}", f"-DCMAKE_PREFIX_PATH={prefix}"]
        run(["cmake", "-S", ROOT / "tmac_experiment/latest", "-B", build, *flags], log, env=env)
        targets = ["llama-bench", "tmac-latest-logits"] + (["tmac-latest-check"] if variant == "tmac" else [])
        run(["cmake", "--build", build, "--target", *targets, "--parallel", args.jobs], log, env=env)
        binaries[variant] = build
    def affinity(threads):
        return ["numactl", "--physcpubind=" + ",".join(map(str, profile["cpus"][:threads])), f"--membind={profile['node']}"]
    run([*affinity(32), binaries["tmac"] / "tmac-latest-check"], out / "operator.log", env=env, output=out / "operator.jsonl")
    dispatch(out / "operator.log")
    for variant, build in binaries.items():
        run([*affinity(32), build / "tmac-latest-logits", model, out / (variant + "-logits.bin")], out / (variant + "-logits.log"), env=env)
    coverage = dispatch(out / "tmac-logits.log")
    correctness = compare_logits(out / "native-logits.bin", out / "tmac-logits.bin")
    write(out / "correctness.json", correctness)
    if any(row["status"] != "pass" for row in correctness):
        raise ValueError("Model logits gate failed; performance comparison stopped")
    rows = []
    for threads in (1, 16, 32):
        env["OMP_NUM_THREADS"] = str(threads)
        for case, (pp, tg, depth) in CASES.items():
            for variant in ("native", "tmac"):
                name = f"{variant}-{case}-t{threads}"
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
                           "tmac_tps": tmac["avg_tps"], "tmac_over_native": tmac["avg_tps"] / native["avg_tps"]})
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
    args = parser.parse_args()
    args.baseline, args.tmac_baseline, args.out = (path.resolve() for path in (args.baseline, args.tmac_baseline, args.out))
    if args.out.exists() or args.jobs < 1:
        parser.error("--out must be new and --jobs must be positive")
    try:
        execute(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, configparser.Error) as error:
        if args.out.exists():
            write(args.out / "failure.json", {"status": "failed", "error": str(error)})
            manifest_path = args.out / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest.update(status="failed", error=str(error))
                write(manifest_path, manifest)
        print(f"Failed: {error}", file=sys.stderr)
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
                archive.add(PATCH, arcname=PATCH.name)
            print(f"Logs archive: {args.out / 'logs.tar.gz'}")
    print(f"Results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
