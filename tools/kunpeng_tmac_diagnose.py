#!/usr/bin/env python3
"""Compare short native/T-MAC operator runs using a completed experiment's sources."""
import argparse
import difflib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import kunpeng_tmac_latest as latest

ROOT = Path(__file__).resolve().parents[1]


def inspect_host():
    allowed = os.sched_getaffinity(0)
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    cores = {}
    for line in topology.splitlines():
        if line and not line.startswith("#"):
            cpu, core, socket, node = line.split(",")
            if "-" not in (core, socket, node):
                cores.setdefault((int(node), socket, core), []).append(int(cpu))

    def counters():
        result = {}
        for line in Path("/proc/stat").read_text().splitlines():
            fields = line.split()
            if fields and fields[0].startswith("cpu") and fields[0][3:].isdigit():
                result[int(fields[0][3:])] = list(map(int, fields[1:9]))
        return result

    before = counters()
    time.sleep(2)
    after = counters()
    busy = {}
    for cpu in before.keys() & after.keys():
        delta = [b - a for a, b in zip(before[cpu], after[cpu])]
        total = sum(delta)
        if len(delta) == 8 and total > 0 and min(delta) >= 0:
            busy[cpu] = 100 * (total - delta[3] - delta[4]) / total
    result = []
    for node in sorted({key[0] for key in cores}):
        groups = [cpus for key, cpus in cores.items() if key[0] == node and allowed.intersection(cpus)]
        loads = [max(busy[cpu] for cpu in cpus) for cpus in groups if all(cpu in busy for cpu in cpus)]
        result.append({"node": node, "allowed_physical_cores": len(groups), "sampled_physical_cores": len(loads),
                       "cores_with_thread_above_20pct": sum(value > 20 for value in loads),
                       "max_thread_busy_pct": max(loads) if loads else None})
    return result


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


def select_compilers(profile, clang_root):
    if clang_root is None:
        return
    for key, binary in (("CMAKE_C_COMPILER", "clang"), ("CMAKE_CXX_COMPILER", "clang++")):
        path = clang_root / "bin" / binary
        report = probe([str(path), "--version"])
        if report.get("exit_code") != 0:
            raise ValueError(f"Cannot execute {path}: {report}")
        profile["toolchain"][key] = str(path)
        profile["toolchain"][key + "_version"] = report["output"]


def blocked_backend_source(source, block_rows):
    if block_rows not in (1, 4, 8, 16, 32):
        raise ValueError("Unsupported diagnostic row block")
    anchor = ("        for (int64_t task = params->ith; task < int64_t(rows) * c.n_tile_num; task += params->nth) {\n"
              "            const size_t row = task / c.n_tile_num, tile = task % c.n_tile_num;\n")
    if source.count(anchor) != 1:
        raise ValueError("Unrecognized original T-MAC task loop")
    start = source.index(anchor)
    end = source.index("        if (params->ith == 0)", start)
    body = source[start + len(anchor):end]
    if not body.endswith("        }\n") or body.count("qgemm_lut_int8(") != 1:
        raise ValueError("Unrecognized T-MAC tile body")
    body = body[:-len("        }\n")]
    replacement = "        auto compute_tile = [&](size_t row, size_t tile) {\n" + body + "        };\n"
    replacement += (f"        for (int first_row = 0; first_row < rows; first_row += {block_rows}) {{\n"
                    f"            const int last_row = first_row + {block_rows} < rows ? first_row + {block_rows} : rows;\n"
                    "            for (int tile = params->ith; tile < c.n_tile_num; tile += params->nth) {\n"
                    "                for (int row = first_row; row < last_row; ++row) {\n"
                    "                    compute_tile(row, tile);\n"
                    "                }\n            }\n        }\n")
    return source[:start] + replacement + source[end:]


def profiled_backend_source(source):
    changes = [
        ("#include <atomic>\n", "#include <atomic>\n#include <chrono>\n#include <cstdio>\n"),
        ("        for (int row = params->ith; row < rows; row += params->nth) {\n",
         "        using profile_clock = std::chrono::steady_clock;\n"
         "        const auto prepare_begin = profile_clock::now();\n"
         "        for (int row = params->ith; row < rows; row += params->nth) {\n"),
        ("        ggml_barrier(params->threadpool);\n",
         "        const auto prepare_end = profile_clock::now();\n"
         "        ggml_barrier(params->threadpool);\n"
         "        const auto consume_begin = profile_clock::now();\n"),
        ("        if (params->ith == 0) {\n",
         "        const auto consume_end = profile_clock::now();\n"
         "        ggml_barrier(params->threadpool);\n"
         "        const double prepare_us = std::chrono::duration<double, std::micro>(prepare_end - prepare_begin).count();\n"
         "        const double wait_us = std::chrono::duration<double, std::micro>(consume_begin - prepare_end).count();\n"
         "        const double consume_us = std::chrono::duration<double, std::micro>(consume_end - consume_begin).count();\n"
         '        std::fprintf(stderr, "tmac: stage {\\"K\\":%d,\\"M\\":%d,\\"rows\\":%d,\\"threads\\":%d,\\"ith\\":%d,\\"prepare_us\\":%.6f,\\"wait_us\\":%.6f,\\"consume_us\\":%.6f}\\n",\n'
         "                     k, m, rows, params->nth, params->ith, prepare_us, wait_us, consume_us);\n"
         "        if (params->ith == 0) {\n"),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Unrecognized T-MAC profiling anchor")
        source = source.replace(old, new, 1)
    return source


def read_stage_profile(log, threads):
    records = [json.loads(line[len("tmac: stage "):]) for line in log.read_text().splitlines() if line.startswith("tmac: stage ")]
    result = []
    for count in (1, 7, 32):
        selected = [row for row in records if row["threads"] == threads and row["rows"] == count]
        if len(selected) != threads * 14:
            raise ValueError("Expected one correctness compute, three warmups and ten stage samples")
        samples: list[dict] = []
        for first in range(0, len(selected), threads):
            workers = sorted(selected[first:first + threads], key=lambda row: row["ith"])
            if [row["ith"] for row in workers] != list(range(threads)):
                raise ValueError("Incomplete profiling worker group")
            for row in workers:
                if (row["K"], row["M"]) != (4096, 4096) or any(not math.isfinite(row[key]) or row[key] < 0 for key in ("prepare_us", "wait_us", "consume_us")):
                    raise ValueError("Invalid worker stage duration")
            samples.append({"workers": workers, **{"max_" + key: max(row[key] for row in workers) for key in ("prepare_us", "wait_us", "consume_us")}})
        samples = samples[4:]
        result.append({"rows": count, "threads": threads, "samples": samples,
                       **{"median_max_" + key: statistics.median(row["max_" + key] for row in samples) for key in ("prepare_us", "wait_us", "consume_us")}})
    return result


def execute(args):
    frozen = json.loads((args.run / "manifest.json").read_text())
    if frozen.get("status") != "smoke_and_benchmark_pass" or frozen.get("commit") != latest.PIN:
        raise ValueError("Choose the completed c622f62 T-MAC run")
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("Run on a Linux AArch64 host")
    numa_node = getattr(args, "numa_node", None)
    profile = latest.execution_profile(frozen["profile"], args.allow_host_change, numa_node)
    clang_root = getattr(args, "clang_root", None)
    select_compilers(profile, clang_root)
    host_changed = profile["hostname"] != frozen["profile"]["hostname"]
    if host_changed:
        print(f"Host changed: {frozen['profile']['hostname']} -> {profile['hostname']}. All operator variants will be measured here; old model throughput is context only.", flush=True)
    if host_changed or numa_node is not None:
        print(f"NUMA node {frozen['profile']['node']} -> {profile['node']}, current physical CPUs: {profile['cpus']}", flush=True)
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
    if clang_root is not None:
        env["PATH"] = str(clang_root / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(clang_root / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        print(f"Toolchain override: {profile['toolchain']['CMAKE_CXX_COMPILER']}", flush=True)
    hardware: dict = {"lscpu": probe(["lscpu"]), "caches": probe(["lscpu", "-C"]),
                "topology": probe(["lscpu", "-p=CPU,CORE,SOCKET,NODE"]),
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
    block_rows = getattr(args, "row_block", None)
    profile_stages = getattr(args, "profile_stages", False)
    fused = getattr(args, "fused", False)
    if sum((block_rows is not None, profile_stages, fused)) > 1:
        raise ValueError("Choose one of row blocking, stage profiling or the fused consumer")
    candidate = "fused" if fused else "blocked" if block_rows is not None else "profiled" if profile_stages else None
    variants = ("native", "tmac", candidate) if candidate is not None else ("native", "tmac")
    stages = []
    for variant in variants:
        backend = "native" if variant == "native" else "tmac"
        build = args.out / (backend + "-build")
        source = args.run / (backend + "-source")
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DGGML_CCACHE=OFF", "-DGGML_OPENMP=ON", "-DGGML_CPU_REPACK=ON", "-DGGML_BLAS=OFF",
                 "-DGGML_CPU_KLEIDIAI=OFF", "-DLLAMA_CURL=OFF", f"-DLLAMA_SOURCE={source}",
                 f"-DGGML_TMAC={'ON' if backend == 'tmac' else 'OFF'}", *latest.tmac_package_flags(prefix)]
        flags.append(f"-DTMAC_FUSED={'ON' if variant == 'fused' else 'OFF'}")
        if variant == candidate:
            original = (source / "ggml/src/ggml-cpu/tmac/tmac.cpp").read_text()
            if variant == "fused":
                modified = (ROOT / "ggml/src/ggml-cpu/tmac/tmac.cpp").read_text()
                shutil.copy2(ROOT / "ggml/src/ggml-cpu/tmac/fused.h", args.out / "fused.h")
            else:
                modified = blocked_backend_source(original, block_rows) if variant == "blocked" else profiled_backend_source(original)
            override = args.out / f"{candidate}-backend.cpp"
            override.write_text(modified)
            (args.out / f"{candidate}-backend.patch").write_text("".join(difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="original/tmac.cpp", tofile=f"{candidate}/tmac.cpp")))
            flags.append(f"-DTMAC_DIAGNOSTIC_BACKEND_SOURCE={override}")
        flags += [f"-D{key}={profile['toolchain'][key]}" for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")]
        if clang_root is not None:
            flags.append(f"-DCMAKE_PREFIX_PATH={clang_root}")
        flags += ["-DGGML_NATIVE=OFF", f"-DGGML_CPU_ARM_ARCH={profile['arm_arch']}"] if profile.get("arm_arch") else ["-DGGML_NATIVE=ON"]
        log = args.out / (variant + "-build.log")
        print(f"Building the {variant} operator probe; log: {log}", flush=True)
        latest.run(["cmake", "-S", ROOT / "tmac_experiment/latest", "-B", build, *flags], log, env=env)
        targets = ["tmac-latest-check"] + (["tmac-fused-check"] if variant == "fused" else [])
        latest.run(["cmake", "--build", build, "--target", *targets, "--parallel", str(args.jobs)], log, env=env)
        if variant == "fused":
            check = args.out / "fused-check.json"
            latest.run(["numactl", f"--physcpubind={profile['cpus'][0]}", f"--membind={profile['node']}", build / "tmac-fused-check"], args.out / "fused-check.log", env=env, output=check)
            if json.loads(check.read_text()) != {"status": "pass", "cases": 50, "bitwise_equal": True}:
                raise ValueError("Fused consumer correctness gate failed")
        if candidate is not None and variant == "tmac":
            shutil.copy2(build / "compile_commands.json", args.out / "original-tmac-compile_commands.json")
            shutil.copy2(build / "CMakeCache.txt", args.out / "original-tmac-CMakeCache.txt")
        for threads in (1, 16, 32):
            name = f"{variant}-t{threads}"
            path = args.out / (name + ".jsonl")
            env["OMP_NUM_THREADS"] = str(threads)
            command = ["numactl", "--physcpubind=" + ",".join(map(str, profile["cpus"][:threads])),
                       f"--membind={profile['node']}", str(build / "tmac-latest-check"), "--bench-" + backend, str(threads)]
            if candidate is not None and backend == "tmac":
                data = args.out / (name + ".bin")
                command.append(str(data))
            print(f"Checking and timing {name}", flush=True)
            latest.run(command, args.out / (name + ".log"), env=env, output=path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if len(records) != 3 or {row["rows"] for row in records} != {1, 7, 32}:
                raise ValueError("Operator probe rows differ")
            for row in records:
                if row["status"] != "pass" or row["threads"] != threads or row["buffer"] != ("CPU_TMAC" if backend == "tmac" else "CPU_REPACK") or (row["K"], row["M"]) != (4096, 4096):
                    raise ValueError("Operator correctness or dispatch check failed")
                if len(row["samples_us"]) != 10 or not all(math.isfinite(value) and value > 0 for value in row["samples_us"]):
                    raise ValueError("Invalid raw operator samples")
                if not math.isfinite(row["median_graph_us"]) or row["median_graph_us"] <= 0:
                    raise ValueError("Invalid operator median")
                row["variant"] = variant
                rows.append(row)
            if backend == "tmac":
                latest.dispatch(args.out / (name + ".log"), threads)
            if variant == "fused" and not re.search(rf"rows=7 threads={threads} kernel_N=2 .*consumer=fused", (args.out / (name + ".log")).read_text()):
                raise ValueError("Fused two-row consumer did not execute")
            if variant == candidate:
                expected = args.out / f"tmac-t{threads}.bin"
                if data.stat().st_size != 4096 * (1 + 7 + 32) * 4 or data.read_bytes() != expected.read_bytes():
                    raise ValueError("Diagnostic backend changed output bytes")
            if variant == "profiled":
                stages.extend(read_stage_profile(args.out / (name + ".log"), threads))
        objdump = shutil.which("objdump")
        if objdump and variant == "tmac":
            for index, obj in enumerate(build.rglob("kernels.cc.o")):
                with (args.out / f"kernels-{index}.asm").open("wb") as output:
                    subprocess.run([objdump, "-d", "-C", str(obj)], stdout=output, stderr=subprocess.STDOUT, check=True)
        if objdump and variant == "fused":
            for index, obj in enumerate(build.rglob("fused-backend.cpp.o")):
                with (args.out / f"fused-backend-{index}.asm").open("wb") as output:
                    subprocess.run([objdump, "-d", "-C", str(obj)], stdout=output, stderr=subprocess.STDOUT, check=True)
    latest.write(args.out / "operator-comparison.json", compare_rows(rows))
    if fused:
        indexed = {(row["variant"], row["rows"], row["threads"]): row for row in rows}
        comparisons = []
        for threads in (1, 16, 32):
            for count in (1, 7, 32):
                native = indexed["native", count, threads]["median_graph_us"]
                original = indexed["tmac", count, threads]["median_graph_us"]
                modified = indexed["fused", count, threads]["median_graph_us"]
                comparisons.append({"rows": count, "threads": threads, "native_us": native, "original_us": original,
                                    "fused_us": modified, "original_over_fused": original / modified, "native_over_fused": native / modified})
        latest.write(args.out / "fused-comparison.json", comparisons)
    if profile_stages:
        latest.write(args.out / "stage-profile.json", {"records": stages,
                     "scope": "instrumented original T-MAC, input conversion/LUT preparation, LUT barrier wait, qgemm/output conversion",
                     "limitations": ["stage maxima can come from different workers and must not be added as a total", "instrumented graph timing includes an added end barrier and logging; use uninstrumented controls for performance comparisons", "ten samples after one correctness compute and three warmups"]})
    if block_rows is not None:
        indexed = {(row["variant"], row["rows"], row["threads"]): row for row in rows}
        comparisons = []
        for threads in (1, 16, 32):
            for count in (1, 7, 32):
                ref = indexed["tmac", count, threads]["median_graph_us"]
                candidate = indexed["blocked", count, threads]["median_graph_us"]
                comparisons.append({"rows": count, "threads": threads, "original_us": ref,
                                    "blocked_us": candidate, "original_over_blocked": ref / candidate})
        latest.write(args.out / "schedule-comparison.json", comparisons)
    latest.write(args.out / "diagnosis.json", {"status": "pass", "completed_run": str(args.run),
                 "completed_profile": frozen["profile"], "execution_profile": profile, "host_changed": host_changed,
                 "numa_node_override": numa_node, "numa_node_changed": profile["node"] != frozen["profile"]["node"],
                 "historical_throughput_used": False,
                 "toolchain_override": str(clang_root) if clang_root is not None else None,
                 "experimental_row_block": block_rows, "blocked_outputs_bitwise_equal": True if block_rows is not None else None,
                 "blocked_backend_sha256": latest.digest(args.out / "blocked-backend.cpp") if block_rows is not None else None,
                 "stage_profiling": profile_stages, "profiled_outputs_bitwise_equal": True if profile_stages else None,
                 "profiled_backend_sha256": latest.digest(args.out / "profiled-backend.cpp") if profile_stages else None,
                 "fused_consumer": fused, "fused_outputs_bitwise_equal": True if fused else None,
                 "fused_header_sha256": latest.digest(args.out / "fused.h") if fused else None,
                 "check_source_sha256": latest.digest(ROOT / "tmac_experiment/latest/check.cpp"),
                 "scope": "synthetic 4096x4096 Q4_0, repeated matrix, reusable GGML threadpool, three warmups and ten samples",
                 "limitations": ["operator timing includes graph dispatch and planning", "single repeated matrix has different cache behavior from a full model", "toolchain changes can also change the OpenMP runtime; inspect build caches", "no model performance claim"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-host", action="store_true", help="only sample per-NUMA CPU load for two seconds, including SMT siblings; no builds or benchmarks")
    parser.add_argument("--numa-node", type=int, help="remeasure all operators with CPU and memory binding on this NUMA node")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--allow-host-change", action="store_true", help="remeasure both operators on this host using current physical cores from the saved NUMA node")
    parser.add_argument("--clang-root", type=Path, help="optional LLVM prefix; build both native and T-MAC with its clang/clang++")
    parser.add_argument("--row-block", type=int, choices=(1, 4, 8, 16, 32), help="also test experimental weight-tile-first scheduling with this row block")
    parser.add_argument("--profile-stages", action="store_true", help="also measure per-worker preparation, LUT barrier wait and consumption in an instrumented original T-MAC copy")
    parser.add_argument("--fused", action="store_true", help="use the new fused one/two-row consumer after bitwise correctness checks")
    args = parser.parse_args()
    if args.inspect_host:
        try:
            print(json.dumps(inspect_host(), indent=2))
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"Failed: {error}", file=sys.stderr)
            return 1
        return 0
    if args.run is None or args.out is None:
        parser.error("--run and --out are required for an experiment")
    if args.numa_node is not None and args.numa_node < 0:
        parser.error("--numa-node must be nonnegative")
    if sum((args.profile_stages, args.row_block is not None, args.fused)) > 1:
        parser.error("Choose one of --row-block, --profile-stages or --fused")
    args.run, args.out = args.run.resolve(), args.out.resolve()
    if args.clang_root is not None:
        args.clang_root = args.clang_root.resolve()
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
                    if path.suffix in (".json", ".jsonl", ".log", ".asm", ".patch", ".cpp", ".txt", ".h"):
                        archive.add(path, arcname=path.name)
                for path in args.out.glob("*-build/compile_commands.json"):
                    archive.add(path, arcname=str(path.relative_to(args.out)))
                for path in args.out.glob("*-build/CMakeCache.txt"):
                    archive.add(path, arcname=str(path.relative_to(args.out)))
            print(f"Return this archive: {args.out / 'diagnostics.tar.gz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
