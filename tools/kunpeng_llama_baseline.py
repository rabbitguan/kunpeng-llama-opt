#!/usr/bin/env python3
"""Retest the earlier v0.6.0 repack experiment with the T-MAC tg32 protocol."""
import argparse
import datetime
import json
import os
import pathlib
import platform
import resource
import shlex
import statistics
import tarfile
import time

from kunpeng_baseline import UPSTREAM, exp, snapshot, write
from kunpeng_a import b_logits_metrics, bench_samples, find_baseline, verify_artifacts

LLAMA_CPU_COMMIT = "a8efd95389352e668d3e234a0243ab08f1d8ba18"
ARM_ARCH = "armv8.6-a+sve+dotprod+i8mm+fp16"
FEATURES = ("__ARM_NEON", "__ARM_FEATURE_SVE", "__ARM_FEATURE_MATMUL_INT8",
            "__ARM_FEATURE_DOTPROD", "__ARM_FEATURE_FP16_VECTOR_ARITHMETIC")


def json_record(text):
    records = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            records.append(item)
    if len(records) != 1:
        raise RuntimeError("Expected one helper JSON record")
    return records[0]


def repack_cases(probe):
    if any(probe.get(key) != 1 for key in ("neon", "dotprod", "i8mm", "sve", "fp16")):
        raise RuntimeError("Required ARM features are not active: " + str(probe))
    cases = [("llama_repack_4x4", "4x4")]
    skipped = {}
    if probe.get("sve_bytes") == 32:
        cases.append(("llama_repack_8x8", "8x8"))
    else:
        skipped["llama_repack_8x8"] = "8x8 SVE kernels require a 32-byte vector; observed " + str(probe.get("sve_bytes"))
    return cases, skipped


def isa_commands(entries):
    commands = {}
    for name in ("ggml-cpu/ggml-cpu.cpp", "ggml-cpu/repack.cpp", "ggml-cpu/arch/arm/quants.c"):
        matches = [entry for entry in entries if pathlib.Path(entry["file"]).as_posix().endswith("/" + name)]
        if len(matches) != 1:
            raise RuntimeError("Missing or ambiguous compile command for " + name)
        entry = matches[0]
        args = entry.get("arguments") or shlex.split(entry["command"])
        flags = [arg for arg in args if arg.startswith(("-march=", "-mcpu=", "-mtune=", "-D__ARM_", "-U__ARM_"))]
        if not flags or "-O3" not in args or any("GGML_USE_TMAC" in arg for arg in args) or not any("GGML_USE_CPU_REPACK" in arg for arg in args):
            raise RuntimeError("Unexpected optimized upstream build for " + name)
        commands[name] = [args[0], *flags, "-dM", "-E", "-x", "c", "/dev/null"]
    return commands


def check_dispatch(text, kernel):
    if "Q4_0 repack kernel: requested={0} selected={0}".format(kernel) not in text or "CPU_REPACK" not in text:
        raise RuntimeError("Requested repack layout was not confirmed: " + kernel)


def check_macros(text):
    macros = {line.split()[1] for line in text.splitlines() if line.startswith("#define ")}
    missing = set(FEATURES) - macros
    if missing:
        raise RuntimeError("Compiler did not enable required ISA: " + ", ".join(sorted(missing)))


def runtime_state(cpu):
    paths = ["/proc/stat", "/proc/self/cgroup", "/proc/self/mountinfo",
             "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list".format(cpu),
             "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq".format(cpu)]
    membership = pathlib.Path("/proc/self/cgroup")
    if membership.is_file():
        for line in membership.read_text().splitlines():
            _, controllers, group = line.split(":", 2)
            if not controllers:
                bases = [pathlib.Path("/sys/fs/cgroup") / group.lstrip("/")]
            elif "cpu" in controllers.split(","):
                bases = [pathlib.Path("/sys/fs/cgroup") / name / group.lstrip("/") for name in ("cpu", "cpu,cpuacct")]
            else:
                continue
            for base in bases:
                for parent in (base, *base.parents):
                    if str(parent).startswith("/sys/fs/cgroup"):
                        paths.extend(str(parent / name) for name in ("cpu.max", "cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us"))
    result = {"unix_time": time.time(), "loadavg": os.getloadavg(), "children_rusage": list(resource.getrusage(resource.RUSAGE_CHILDREN))}
    for name in dict.fromkeys(paths):
        try:
            result[name] = pathlib.Path(name).read_text()
        except OSError:
            pass
    return result


def benchmark(name, binary, model, output, recorder, affinity, env, kernel=None):
    cpu = int(affinity[2].split(",")[0])
    write(output / (name + "-state-before.json"), runtime_state(cpu))
    command = affinity + [str(binary), "-m", str(model), "-t", "1", "-ngl", "0", "-p", "0", "-n", "32",
                          "-r", "4", "-b", "32", "-ub", "32", "-ctk", "f16", "-ctv", "f16", "-o", "jsonl"]
    if kernel:
        command += ["--repack", "1", "--q4-kernel", kernel, "-fa", "off", "-v"]
    result = recorder.run(command, env=env, capture=True)
    write(output / (name + "-state-after.json"), runtime_state(cpu))
    (output / (name + "-bench.txt")).write_text(result.stdout)
    if kernel:
        check_dispatch(result.stdout, kernel)
        record = json_record(result.stdout)
        if record.get("q4_kernel") != kernel or record.get("repack") not in (True, 1) or record.get("n_threads") != 1:
            raise RuntimeError("Benchmark did not retain requested repack/thread settings")
    return [{"variant": name, "round": i + 1, "tokens_per_second": x} for i, x in enumerate(bench_samples(result.stdout))]


def execute(baseline, output, recorder):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run this comparison on the Linux AArch64 server")
    frozen = json.loads((baseline / "manifest.json").read_text())
    if frozen.get("status") != "preliminary_pass" or frozen.get("opts") != [] or frozen.get("llama_ref_commit") != UPSTREAM:
        raise RuntimeError("A matching completed basic baseline is required")
    config = frozen["config"]
    source = pathlib.Path(config["paths"]["tmac_root"]) / "source"
    if exp.verify_source(config, source) != frozen["commits"]:
        raise RuntimeError("Dependency commits changed")
    verify_artifacts(baseline, frozen["generated_sha256"])
    model = pathlib.Path(frozen["model"])
    if exp.sha256_file(model) != frozen["model_sha256"]:
        raise RuntimeError("Model checksum changed")
    env = os.environ.copy()
    env.update(exp.source_environment(source))
    env.update({"OMP_NUM_THREADS": "1", "TVM_NUM_THREADS": "1"})
    cpus = exp.physical_cpus(config["model_benchmark"]["numa_node"])
    if not cpus or exp.affinity_prefix(cpus[:1], config["model_benchmark"]["numa_node"]) != frozen["affinity"]:
        raise RuntimeError("CPU binding differs from frozen baseline")
    affinity = frozen["affinity"]
    manifest = {"status": "building", "baseline": str(baseline), "llama_ref_commit": LLAMA_CPU_COMMIT, "commits": frozen["commits"],
                "model_sha256": frozen["model_sha256"], "affinity": affinity, "threads": 1,
                "scope": "earlier_v0.6.0_runtime_repack_retest", "latest_upstream": False, "cross_version_comparison": True,
                "tmac_remeasured": False,
                "decode_tokens": 32, "prompt_tokens": 0, "rounds": 3, "warmup_samples": 1,
                "weight_contract": "existing lossless runtime repacker; no requantization or model writes",
                "limitations": ["short single-thread smoke comparison, not model quality evaluation", "binding is not exclusive CPU allocation", "no latest-upstream comparison", "T-MAC result is historical, not contemporaneous", "compile/runtime gates establish eligible kernel paths, not per-op profiling"],
                "source_sha256": {name: exp.sha256_file(exp.EXPERIMENT_DIR / "harness" / name) for name in ("cpu-baseline.cpp", "CMakeLists.txt")},
                "variants": {}}
    write(output / "manifest.json", manifest)
    upstream_repo = pathlib.Path(__file__).resolve().parents[1]
    if not exp.git_has_commit(upstream_repo, LLAMA_CPU_COMMIT):
        raise RuntimeError("Earlier experiment commit is missing from the project repository")
    tree = output / "llama-source"
    snapshot(recorder, upstream_repo, LLAMA_CPU_COMMIT, tree)
    build = output / "harness"
    flags = ["-DLLAMA_SOURCE=" + str(tree), "-DTMAC_CPU_BASELINE=ON", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++",
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DGGML_NATIVE=OFF", "-DGGML_CPU_ARM_ARCH=" + ARM_ARCH,
             "-DGGML_CPU_REPACK=ON", "-DGGML_CPU_KLEIDIAI=OFF", "-DGGML_OPENMP=ON", "-DGGML_LLAMAFILE=ON", "-DGGML_BLAS=OFF",
             "-DGGML_BACKEND_DL=OFF", "-DLLAMA_BUILD_COMMIT=" + LLAMA_CPU_COMMIT[:12], "-DLLAMA_BUILD_NUMBER=0"]
    recorder.run(["cmake", "-S", str(exp.EXPERIMENT_DIR / "harness"), "-B", str(build)] + flags, env=env)
    recorder.run(["cmake", "--build", str(build), "--target", "llama-cpu-smoke", "llama-bench", "-j", "4"], env=env)
    library_dirs = sorted({str(path.parent) for path in build.rglob("lib*.so")})
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs + [env.get("LD_LIBRARY_PATH", "")])
    commands = json.loads((build / "compile_commands.json").read_text())
    for name, command in isa_commands(commands).items():
        result = recorder.run(command, env=env, capture=True)
        (output / (pathlib.Path(name).name + "-macros.txt")).write_text(result.stdout)
        check_macros(result.stdout)
    helper, binary = (exp.find_binary(build, name) for name in ("llama-cpu-smoke", "llama-bench"))
    probe = json_record(recorder.run(affinity + [str(helper), "--probe"], env=env, capture=True).stdout)
    write(output / "cpu-features.json", probe)
    cases, skipped = repack_cases(probe)
    manifest.update({"cpu_features": probe, "skipped": skipped, "cmake_flags": flags,
                     "llamafile_note": "This revision undefines GGML_USE_LLAMAFILE on SVE/I8MM builds; the runtime value is authoritative",
                     "binary_sha256": {p.name: exp.sha256_file(p) for p in (helper, binary)}})
    write(output / "manifest.json", manifest)
    reference_logits = exp.read_logits(baseline / "llama_ref-logits.bin")
    rows = []
    for name, kernel in cases:
        record = {"kernel": kernel, "runtime_repack": True, "status": "checking"}
        manifest["variants"][name] = record
        write(output / "manifest.json", manifest)
        logfile = output / (name + "-logits.bin")
        result = recorder.run(affinity + [str(helper), str(model), str(logfile), kernel], env=env, capture=True)
        (output / (name + "-logits.txt")).write_text(result.stdout)
        check_dispatch(result.stdout, kernel)
        record["correctness"] = b_logits_metrics(reference_logits, exp.read_logits(logfile))
        write(output / "manifest.json", manifest)
        if record["correctness"]["status"] != "pass":
            raise RuntimeError("Optimized llama.cpp logits failed fixed NMSE<=0.001/max_abs<=1.0 gate")
        rows.extend(benchmark(name, binary, model, output, recorder, affinity, env, kernel))
        exp.write_csv(output / "decode.csv", ["variant", "round", "tokens_per_second"], rows)
        record["status"] = "pass"
        write(output / "manifest.json", manifest)
    if exp.sha256_file(model) != frozen["model_sha256"]:
        raise RuntimeError("Original model changed during experiment")
    medians = {name: statistics.median(r["tokens_per_second"] for r in rows if r["variant"] == name) for name in {r["variant"] for r in rows}}
    manifest["status"] = "preliminary_pass"
    write(output / "manifest.json", manifest)
    historical = json.loads((baseline / "summary.json").read_text())["median_tokens_per_second"]["tmac_ref"]
    write(output / "summary.json", {"status": "preliminary_pass", "median_tokens_per_second": medians,
          "historical_tmac_reference": {"baseline": str(baseline), "tokens_per_second": historical, "same_run": False},
          "skipped": skipped, "tmac_remeasured": False})


def main():
    config = exp.load_config(exp.EXPERIMENT_DIR / "config.json", [])
    root = pathlib.Path(config["paths"]["results_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path)
    args = parser.parse_args()
    baseline = (args.baseline or find_baseline(root)).resolve()
    output = root / ("llama_optimized_baseline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    recorder = exp.Recorder(output)
    try:
        execute(baseline, output, recorder)
        print("Completed: " + str(output / "summary.json"))
    except Exception as error:
        recorder.log(str(error))
        write(output / "failure.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        with tarfile.open(str(output) + "-logs.tar.gz", "w:gz") as archive:
            for pattern in ("*.json", "*.csv", "*.log", "*.txt", "*-logits.bin"):
                for path in output.glob(pattern):
                    archive.add(path, arcname=path.name)
            for name in ("harness/compile_commands.json", "harness/CMakeCache.txt"):
                if (output / name).is_file():
                    archive.add(output / name, arcname=name)
        print("Return log archive: " + str(output) + "-logs.tar.gz")


if __name__ == "__main__":
    main()
