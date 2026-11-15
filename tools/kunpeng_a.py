#!/usr/bin/env python3
"""Minimal A versus frozen T-MAC experiment; no kernel tuning."""
import argparse
import datetime
import json
import math
import os
import pathlib
import shutil
import statistics
import sys
import tarfile

from kunpeng_baseline import exp, write
from kunpeng_exp import check_lut


def replace_constructor(kernels, old, new):
    if old == new or kernels.count(old) != 1:
        raise RuntimeError("Expected exactly one original LUT constructor in frozen kernels")
    return kernels.replace(old, new, 1)


def bench_samples(text):
    records = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "samples_ns" in record:
            records.append(record)
    if len(records) != 1:
        raise RuntimeError("Expected exactly one benchmark JSON record")
    record = records[0]
    samples = record["samples_ns"]
    if record.get("n_gen") != 32 or record.get("n_prompt") != 0 or len(samples) != 4:
        raise RuntimeError("Expected four zero-prompt tg32 samples")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples):
        raise RuntimeError("Invalid benchmark duration")
    return [32e9 / duration for duration in samples[1:]]


def find_baseline(root):
    for path in sorted(root.glob("basic_baseline_*/manifest.json"), reverse=True):
        data = json.loads(path.read_text())
        if data.get("status") == "preliminary_pass" and data.get("opts") == []:
            return path.parent
    raise RuntimeError("No completed basic baseline found; pass --baseline explicitly")


def execute(baseline, output, recorder):
    frozen = json.loads((baseline / "manifest.json").read_text())
    if frozen.get("status") != "preliminary_pass" or frozen.get("opts") != []:
        raise RuntimeError("Baseline must be a completed unoptimized run")
    config = frozen["config"]
    source = pathlib.Path(config["paths"]["tmac_root"]) / "source"
    commits = exp.verify_source(config, source)
    if commits != frozen["commits"]:
        raise RuntimeError("Dependency commits changed since baseline")
    fork = source / "3rdparty/llama.cpp"
    recorder.run(["git", "diff", "--exit-code", "HEAD"], cwd=fork)
    model = pathlib.Path(frozen["model"])
    if exp.sha256_file(model) != frozen["model_sha256"]:
        raise RuntimeError("Model checksum changed")
    for name, expected in frozen["generated_sha256"].items():
        if exp.sha256_file(baseline / "generated" / name) != expected or exp.sha256_file(baseline / "install/lib" / name) != expected:
            raise RuntimeError("Frozen generated or installed artifact changed: " + name)
    env = os.environ.copy()
    env.update(exp.source_environment(source))
    env.update({"OMP_NUM_THREADS": "1", "TVM_NUM_THREADS": "1"})
    cpus = exp.physical_cpus(config["model_benchmark"]["numa_node"])
    if not cpus or exp.affinity_prefix(cpus[:1], config["model_benchmark"]["numa_node"]) != frozen["affinity"]:
        raise RuntimeError("CPU binding differs from baseline")
    affinity = frozen["affinity"]
    check_lut(source, output)
    tree = output / "tmac-source"
    shutil.copytree(baseline / "tmac-source", tree, symlinks=True)
    ctor = pathlib.Path("python/t_mac/intrins/lut_ctor.cc")
    old = (tree / ctor).read_text()
    recorder.run(["git", "apply", "--check", str(exp.EXPERIMENT_DIR / "optimizations/A.patch")], cwd=tree)
    recorder.run(["git", "apply", str(exp.EXPERIMENT_DIR / "optimizations/A.patch")], cwd=tree)
    generated = output / "generated"
    generated.mkdir()
    for name in ("kernels.h", "kcfg.ini"):
        shutil.copy2(baseline / "generated" / name, generated / name)
    text = replace_constructor((baseline / "generated/kernels.cc").read_text(), old, (tree / ctor).read_text())
    (generated / "kernels.cc").write_text(text)
    for name in ("kernels.cc", "kernels.h", "kcfg.ini"):
        shutil.copy2(generated / name, tree / "deploy/tuned" / name)
    manifest = {"status": "building", "baseline": str(baseline), "commits": commits, "opts": ["A"], "layout": "upstream",
                "model_sha256": frozen["model_sha256"], "affinity": affinity, "threads": 1,
                "patch_sha256": exp.sha256_file(exp.EXPERIMENT_DIR / "optimizations/A.patch"),
                "generated_sha256": {name: exp.sha256_file(generated / name) for name in ("kernels.cc", "kernels.h", "kcfg.ini")},
                "timing": "one process per backend, four tg32 samples, discard first; one real O=4096 K=4096 T=1 operator, three rounds",
                "limitations": ["short single-thread preliminary comparison", "operator checks sample outputs", "dispatch initialization does not prove full operator coverage"]}
    write(output / "manifest.json", manifest)
    common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    install = output / "install"
    recorder.run(["cmake", "-S", str(tree), "-B", str(output / "tmac-build"), "-DCMAKE_INSTALL_PREFIX=" + str(install), "-DCMAKE_INSTALL_LIBDIR=lib", "-DTMAC_TVM=OFF"] + common, env=env)
    recorder.run(["cmake", "--build", str(output / "tmac-build"), "--target", "install", "-j", "4"], env=env)
    harnesses = {}
    for name, prefix in (("tmac_ref", baseline / "install"), ("A", install)):
        build = output / (name + "-harness")
        flags = common + ["-DLLAMA_SOURCE=" + str(fork), "-DGGML_NATIVE=ON", "-DGGML_TMAC=ON", "-DGGML_TMAC_TVM_THREADPOOL=OFF", "-DGGML_TMAC_RECHUNK=OFF", "-DCMAKE_PREFIX_PATH=" + str(prefix / "lib/cmake/t-mac")]
        recorder.run(["cmake", "-S", str(exp.EXPERIMENT_DIR / "harness"), "-B", str(build)] + flags, env=env)
        targets = ["tmac-operator"] + (["tmac-logits"] if name == "A" else [])
        recorder.run(["cmake", "--build", str(build), "--target"] + targets + ["-j", "4"], env=env)
        harnesses[name] = build
    a_build = output / "A-build"
    recorder.run(["cmake", "-S", str(fork), "-B", str(a_build)] + common + ["-DGGML_NATIVE=ON", "-DGGML_TMAC=ON", "-DGGML_TMAC_TVM_THREADPOOL=OFF", "-DGGML_TMAC_RECHUNK=OFF", "-DCMAKE_PREFIX_PATH=" + str(install / "lib/cmake/t-mac")], env=env)
    recorder.run(["cmake", "--build", str(a_build), "--target", "llama-bench", "-j", "4"], env=env)
    env["TMAC_KCFG_FILE"] = str(generated / "kcfg.ini")
    logfile = output / "A-logits.bin"
    result = recorder.run(affinity + [str(exp.find_binary(harnesses["A"], "tmac-logits")), str(model), str(logfile), "1", "128000,791,4062,14198,39935,35308,13"], env=env, capture=True)
    if "ggml_tmac_init" not in result.stdout or "Failed to find kcfg" in result.stdout:
        raise RuntimeError("A dispatch initialization failed")
    ref = exp.read_logits(baseline / "tmac_ref-logits.bin")
    actual = exp.read_logits(logfile)
    equal = logfile.read_bytes() == (baseline / "tmac_ref-logits.bin").read_bytes()
    finite = all(math.isfinite(x) for x in ref + actual)
    write(output / "correctness.json", {"status": "pass" if equal and finite else "failed", "logits_bitwise_equal": equal, "all_finite": finite})
    if not equal or not finite:
        raise RuntimeError("A logits differ from frozen T-MAC; no valid timing result")
    operators, decode = [], []
    for name, prefix, binary in (("tmac_ref", baseline / "install", exp.find_binary(baseline / "tmac_ref-build", "llama-bench")), ("A", install, exp.find_binary(a_build, "llama-bench"))):
        env["TMAC_KCFG_FILE"] = str(prefix / "lib/kcfg.ini")
        operator_csv = output / (name + "-operator.csv")
        exp.write_csv(operator_csv, exp.operator_fields(), [])
        recorder.run(affinity + [str(exp.find_binary(harnesses[name], "tmac-operator")), str(model), "4096", "4096", "1", "3", "2", "20241025", str(operator_csv)], env=env)
        operator_config = dict(config)
        operator_config["operator"] = dict(config["operator"], shapes=[[4096, 4096]], tokens=[1])
        failures, rows = exp.validate_operator_results(operator_config, operator_csv)
        if len(rows) != 3 or failures:
            raise RuntimeError("Operator check failed for " + name)
        for row in rows:
            if row["status"] != "pass" or any(not math.isfinite(float(row[key])) for key in ("lut_us", "compute_us", "total_us", "implementation_max_abs", "high_precision_max_abs")) or any(float(row[key]) <= 0 for key in ("lut_us", "compute_us", "total_us")):
                raise RuntimeError("Invalid operator result for " + name)
            row["variant"] = name
        operators.extend(rows)
        result = recorder.run(affinity + [str(binary), "-m", str(model), "-t", "1", "-ngl", "0", "-p", "0", "-n", "32", "-r", "4", "-b", "32", "-ub", "32", "-ctk", "f16", "-ctv", "f16", "-o", "jsonl"], env=env, capture=True)
        if "ggml_tmac_init" not in result.stdout or "Failed to find kcfg" in result.stdout:
            raise RuntimeError("Benchmark dispatch initialization failed for " + name)
        (output / (name + "-bench.txt")).write_text(result.stdout)
        decode.extend({"variant": name, "round": i + 1, "tokens_per_second": value} for i, value in enumerate(bench_samples(result.stdout)))
        exp.write_csv(output / "operator.csv", exp.operator_fields(), operators)
        exp.write_csv(output / "decode.csv", ["variant", "round", "tokens_per_second"], decode)
    speeds = {name: statistics.median(row["tokens_per_second"] for row in decode if row["variant"] == name) for name in ("tmac_ref", "A")}
    stages = {name: {key: statistics.median(float(row[key]) for row in operators if row["variant"] == name) for key in ("lut_us", "compute_us", "total_us")} for name in speeds}
    write(output / "summary.json", {"status": "preliminary_pass", "decode_median_tokens_per_second": speeds, "A_over_tmac_decode": speeds["A"] / speeds["tmac_ref"], "operator_median_us": stages, "lut_speedup": stages["tmac_ref"]["lut_us"] / stages["A"]["lut_us"], "operator_speedup": stages["tmac_ref"]["total_us"] / stages["A"]["total_us"]})
    manifest["status"] = "preliminary_pass"
    write(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path)
    args = parser.parse_args()
    config = exp.load_config(exp.EXPERIMENT_DIR / "config.json", [])
    root = pathlib.Path(config["paths"]["results_root"])
    baseline = (args.baseline or find_baseline(root)).resolve()
    output = root / ("A_compare_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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
            for pattern in ("*.json", "*.csv", "*.log", "*.txt", "*.bin", "*.s"):
                for path in output.glob(pattern):
                    archive.add(path, arcname=path.name)
            if (output / "generated/kcfg.ini").is_file():
                archive.add(output / "generated/kcfg.ini", arcname="kcfg.ini")
        print("Return log archive: " + str(output) + "-logs.tar.gz")


if __name__ == "__main__":
    main()
