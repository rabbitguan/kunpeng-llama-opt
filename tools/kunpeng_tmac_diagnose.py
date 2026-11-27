#!/usr/bin/env python3
"""Compare short native/T-MAC operator runs using a completed experiment's sources."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

import kunpeng_tmac_latest as latest

ROOT = Path(__file__).resolve().parents[1]


def probe(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return {"command": list(map(str, command)), "exit_code": result.returncode,
                "output": result.stdout.decode("utf-8", errors="replace")}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(map(str, command)), "error": str(error)}


def compare_rows(rows):
    indexed = {(row["variant"], row["rows"], row["threads"]): row for row in rows}
    result = []
    for threads in (1, 16, 32):
        for count in (1, 7, 32):
            native = indexed["native", count, threads]["median_graph_us"]
            tmac = indexed["tmac", count, threads]["median_graph_us"]
            if not math.isfinite(native) or not math.isfinite(tmac) or native <= 0 or tmac <= 0:
                raise ValueError("Invalid operator duration")
            result.append({"rows": count, "threads": threads, "native_us": native,
                           "tmac_us": tmac, "native_us_over_tmac_us": native / tmac})
    return result


def execute(args):
    frozen = json.loads((args.run / "manifest.json").read_text())
    if frozen.get("status") != "smoke_and_benchmark_pass" or frozen.get("commit") != latest.PIN:
        raise ValueError("Choose the completed c622f62 T-MAC run")
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Run on a Linux AArch64 host")
    profile = latest.execution_profile(frozen["profile"], args.allow_host_change)
    host_changed = profile["hostname"] != frozen["profile"]["hostname"]
    if host_changed:
        print(f"Host changed: {frozen['profile']['hostname']} -> {profile['hostname']}. Both operator variants will be measured here; old model throughput is context only.", flush=True)
        print(f"NUMA node {profile['node']}, current physical CPUs: {profile['cpus']}", flush=True)
    prefix = args.run / "tmac-prefix"
    if latest.digest(prefix / "lib/kernels.cc") != frozen["compiled_kernel_sha256"]:
        raise ValueError("Compiled kernel source changed")
    args.out.mkdir(parents=True)
    env = os.environ.copy()
    for name, value in profile["toolchain"]["env"].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    env.update(TMAC_KCFG_FILE=str(prefix / "lib/kcfg.ini"), OMP_DYNAMIC="FALSE", TVM_NUM_THREADS="1")
    hardware = {"lscpu": probe(["lscpu"]), "caches": probe(["lscpu", "-C"]),
                "hostname": profile["hostname"], "completed_run_hostname": frozen["profile"]["hostname"],
                "numa": probe(["numactl", "--hardware"]), "allowed_cpus": sorted(os.sched_getaffinity(0)),
                "compiler": probe([profile["toolchain"]["CMAKE_CXX_COMPILER"], "--version"]),
                "clang_in_path": shutil.which("clang++"), "perf_in_path": shutil.which("perf")}
    historical = Path(frozen["tmac_baseline"])
    for cache in historical.glob("*/CMakeCache.txt"):
        for line in cache.read_bytes().splitlines():
            if line.startswith((b"CMAKE_CXX_COMPILER:FILEPATH=", b"CMAKE_CXX_COMPILER:STRING=")):
                path = Path(os.fsdecode(line.split(b"=", 1)[1]))
                hardware.setdefault("historical_compilers", []).append({"cache": str(cache), "path": str(path),
                                                                         "probe": probe([str(path), "--version"])})
    latest.write(args.out / "hardware.json", hardware)
    for name in ("manifest.json", "comparison.json", "correctness.json"):
        shutil.copy2(args.run / name, args.out / ("completed-" + name))
    rows = []
    for variant in ("native", "tmac"):
        build = args.out / (variant + "-build")
        source = args.run / (variant + "-source")
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DGGML_CCACHE=OFF", "-DGGML_OPENMP=ON", "-DGGML_CPU_REPACK=ON", "-DGGML_BLAS=OFF",
                 "-DGGML_CPU_KLEIDIAI=OFF", "-DLLAMA_CURL=OFF", f"-DLLAMA_SOURCE={source}",
                 f"-DGGML_TMAC={'ON' if variant == 'tmac' else 'OFF'}", *latest.tmac_package_flags(prefix)]
        flags += [f"-D{key}={profile['toolchain'][key]}" for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")]
        flags += ["-DGGML_NATIVE=OFF", f"-DGGML_CPU_ARM_ARCH={profile['arm_arch']}"] if profile.get("arm_arch") else ["-DGGML_NATIVE=ON"]
        log = args.out / (variant + "-build.log")
        print(f"Building the {variant} operator probe; log: {log}", flush=True)
        latest.run(["cmake", "-S", ROOT / "tmac_experiment/latest", "-B", build, *flags], log, env=env)
        latest.run(["cmake", "--build", build, "--target", "tmac-latest-check", "--parallel", str(args.jobs)], log, env=env)
        for threads in (1, 16, 32):
            name = f"{variant}-t{threads}"
            path = args.out / (name + ".jsonl")
            env["OMP_NUM_THREADS"] = str(threads)
            command = ["numactl", "--physcpubind=" + ",".join(map(str, profile["cpus"][:threads])),
                       f"--membind={profile['node']}", str(build / "tmac-latest-check"), "--bench-" + variant, str(threads)]
            print(f"Checking and timing {name}", flush=True)
            latest.run(command, args.out / (name + ".log"), env=env, output=path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if len(records) != 3 or {row["rows"] for row in records} != {1, 7, 32}:
                raise ValueError("Operator probe rows differ")
            for row in records:
                if row["status"] != "pass" or row["threads"] != threads or row["buffer"] != ("CPU_TMAC" if variant == "tmac" else "CPU_REPACK") or (row["K"], row["M"]) != (4096, 4096):
                    raise ValueError("Operator correctness or dispatch check failed")
                if len(row["samples_us"]) != 10 or not all(math.isfinite(value) and value > 0 for value in row["samples_us"]):
                    raise ValueError("Invalid raw operator samples")
                row["variant"] = variant
                rows.append(row)
            if variant == "tmac":
                latest.dispatch(args.out / (name + ".log"), threads)
        objdump = shutil.which("objdump")
        if objdump and variant == "tmac":
            for index, obj in enumerate(build.rglob("kernels.cc.o")):
                with (args.out / f"kernels-{index}.asm").open("wb") as output:
                    subprocess.run([objdump, "-d", "-C", str(obj)], stdout=output, stderr=subprocess.STDOUT, check=True)
    latest.write(args.out / "operator-comparison.json", compare_rows(rows))
    latest.write(args.out / "diagnosis.json", {"status": "pass", "completed_run": str(args.run),
                 "completed_profile": frozen["profile"], "execution_profile": profile, "host_changed": host_changed,
                 "historical_throughput_used": False,
                 "check_source_sha256": latest.digest(ROOT / "tmac_experiment/latest/check.cpp"),
                 "scope": "synthetic 4096x4096 Q4_0, repeated matrix, reusable GGML threadpool, three warmups and ten samples",
                 "limitations": ["operator timing includes graph dispatch and planning", "single repeated matrix has different cache behavior from a full model", "no model performance claim"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--allow-host-change", action="store_true", help="remeasure both operators on this host using current physical cores from the saved NUMA node")
    args = parser.parse_args()
    args.run, args.out = args.run.resolve(), args.out.resolve()
    if args.out.exists() or args.jobs < 1:
        parser.error("Choose a new --out and positive --jobs")
    try:
        execute(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Failed: {error}", file=sys.stderr)
        if args.out.exists():
            latest.write(args.out / "failure.json", {"error": str(error)})
        return 1
    finally:
        if args.out.exists():
            with tarfile.open(args.out / "diagnostics.tar.gz", "w:gz") as archive:
                for path in args.out.iterdir():
                    if path.suffix in (".json", ".jsonl", ".log", ".asm"):
                        archive.add(path, arcname=path.name)
                for path in args.out.glob("*-build/compile_commands.json"):
                    archive.add(path, arcname=str(path.relative_to(args.out)))
                for path in args.out.glob("*-build/CMakeCache.txt"):
                    archive.add(path, arcname=str(path.relative_to(args.out)))
            print(f"Return this archive: {args.out / 'diagnostics.tar.gz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
