#!/usr/bin/env python3
"""One real-matrix C schedule experiment; no model runs or A/B changes."""
import argparse
import configparser
import datetime
import json
import math
import os
import pathlib
import platform
import shutil
import statistics
import tarfile

from kunpeng_baseline import exp, write
from kunpeng_a import find_baseline, verify_artifacts

HEADER = exp.EXPERIMENT_DIR / "optimizations/cache-schedule.h"


def validate_config(frozen, text, k=4096):
    if k not in (4096, 14336):
        raise RuntimeError("C experiment supports only K=4096 or 14336")
    quant = frozen["config"]["quantization"]
    expected_quant = {"bits": 4, "weight_group_size": 32, "act_group_size": 32, "lut_group": 4,
                      "fast_aggregation": False, "lut_dtype": "int8", "output_dtype": "float16",
                      "aggregation_dtype": "int32", "zero_point": False}
    if quant != expected_quant:
        raise RuntimeError("C requires the original Q4 groups=32,g=4 configuration")
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    section = "qgemm_lut_t1_int8_m16384_k{}_n1_b4".format(k)
    expected = {"bm": 256, "kfactor": 8, "group_size": 32, "simd_n_in": 16, "simd_n_out": 8,
                "lut_scales_size": k // 32, "scales_size": 4096 * (k // 32), "n_tile_num": 64}
    if section not in cfg or any(cfg[section].getint(key) != v for key, v in expected.items()):
        raise RuntimeError("C prototype requires the frozen bm=256,O=4096 kernel for the requested K")


def reusable_manifest(reuse, baseline, frozen):
    record = json.loads((reuse / "manifest.json").read_text())
    if record.get("status") != "operator_pass_model_pending" or record.get("opts") != ["C"] or record.get("commits") != frozen["commits"] or record.get("baseline") != str(baseline) or record.get("model_sha256") != frozen["model_sha256"]:
        raise RuntimeError("Reusable C run does not match frozen baseline")
    if record.get("header_sha256") != exp.sha256_file(HEADER) or record.get("kernels_sha256") != exp.sha256_file(reuse / "install/lib/kernels.cc"):
        raise RuntimeError("Reusable C kernel source changed")
    return dict(record)


def execute(baseline, output, recorder, k=4096, reuse=None):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run the C experiment on the Linux AArch64 server")
    frozen = json.loads((baseline / "manifest.json").read_text())
    if frozen.get("status") != "preliminary_pass" or frozen.get("opts") != []:
        raise RuntimeError("C needs an unoptimized completed baseline")
    config = frozen["config"]
    source = pathlib.Path(config["paths"]["tmac_root"]) / "source"
    if exp.verify_source(config, source) != frozen["commits"]:
        raise RuntimeError("Source versions differ from frozen baseline")
    verify_artifacts(baseline, frozen["generated_sha256"])
    validate_config(frozen, (baseline / "generated/kcfg.ini").read_text(), k)
    model = pathlib.Path(frozen["model"])
    if exp.sha256_file(model) != frozen["model_sha256"]:
        raise RuntimeError("Model checksum differs")
    fork = source / "3rdparty/llama.cpp"
    recorder.run(["git", "diff", "--exit-code", "HEAD"], cwd=fork)
    env = os.environ.copy()
    env.update(exp.source_environment(source))
    env.update({"OMP_NUM_THREADS": "1", "TVM_NUM_THREADS": "1"})
    cpus = exp.physical_cpus(config["model_benchmark"]["numa_node"])
    affinity = exp.affinity_prefix(cpus[:1], config["model_benchmark"]["numa_node"])
    if not cpus or affinity != frozen["affinity"]:
        raise RuntimeError("CPU binding differs from baseline")
    if reuse is not None:
        manifest = reusable_manifest(reuse, baseline, frozen)
        install = reuse / "install"
        build = reuse / "harness"
        manifest.update({"status": "testing", "K": k, "reused_run": str(reuse),
                         "binary_sha256": exp.sha256_file(exp.find_binary(build, "tmac-cache-operator")),
                         "layout_key": "v1:BM256:O4096:K" + str(k) + ":R{R}:BK256:model=" + frozen["model_sha256"]})
        recorder.log("Reuse compiled C operator; no compile, install or tuning: " + str(build))
    else:
        check = output / "C-check.cpp"
        check.write_text('#include "' + str(baseline / 'tmac-source/python/t_mac/intrins/tbl.cc') + '"\n#define TMAC_CACHE_KERNEL\n#include "' + str(HEADER) + '"\n#include "' + str(exp.EXPERIMENT_DIR / 'harness/cache-schedule.cpp') + '"\n')
        recorder.run(["clang++", "-O3", "-std=c++17", "-march=armv8.2-a+fp16", str(check), "-o", str(output / "check-C")], env=env)
        result = recorder.run([str(output / "check-C")], env=env, capture=True)
        write(output / "layout-check.json", {"status": "pass", "result": result.stdout})
        tree = output / "tmac-source"
        shutil.copytree(baseline / "tmac-source", tree, symlinks=True)
        kernels = (baseline / "generated/kernels.cc").read_text() + "\n#define TMAC_CACHE_KERNEL\n" + HEADER.read_text()
        (tree / "deploy/tuned/kernels.cc").write_text(kernels)
        manifest = {"status": "building", "baseline": str(baseline), "scope": "operator_only", "opts": ["C"],
                    "commits": frozen["commits"], "model_sha256": frozen["model_sha256"], "layout_version": 1, "buffer_alignment": 64,
                    "BM": 256, "O": 4096, "K": k, "T": 1, "R": [1, 2, 4], "BK": 256,
                    "affinity": affinity, "threads": 1, "rounds": 10, "warmups": 3,
                    "layout_key": "v1:BM256:O4096:K" + str(k) + ":R{R}:BK256:model=" + frozen["model_sha256"],
                    "header_sha256": exp.sha256_file(HEADER), "kernels_sha256": exp.sha256_file(tree / "deploy/tuned/kernels.cc"),
                    "model_check": "not_run", "fallback": "none; unsupported configuration fails"}
        write(output / "manifest.json", manifest)
        install = output / "install"
        common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
        recorder.run(["cmake", "-S", str(tree), "-B", str(output / "tmac-build"), "-DTMAC_TVM=OFF", "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_INSTALL_PREFIX=" + str(install)] + common, env=env)
        recorder.run(["cmake", "--build", str(output / "tmac-build"), "--target", "install", "-j", "4"], env=env)
        build = output / "harness"
        recorder.run(["cmake", "-S", str(exp.EXPERIMENT_DIR / "harness"), "-B", str(build), "-DLLAMA_SOURCE=" + str(fork), "-DGGML_NATIVE=ON", "-DGGML_TMAC=ON", "-DGGML_TMAC_TVM_THREADPOOL=OFF", "-DGGML_TMAC_RECHUNK=OFF", "-DTMAC_CACHE_SCHEDULE=ON", "-DCMAKE_PREFIX_PATH=" + str(install / "lib/cmake/t-mac")] + common, env=env)
        recorder.run(["cmake", "--build", str(build), "--target", "tmac-cache-operator", "-j", "4"], env=env)
    env["TMAC_KCFG_FILE"] = str(install / "lib/kcfg.ini")
    manifest["status"] = "testing"
    write(output / "manifest.json", manifest)
    all_rows = []
    orders = ((0, 1, 2, 4), (4, 2, 1, 0)) if reuse is not None else ((0, 1, 2, 4),)
    rounds = 5 if reuse is not None else 10
    manifest["orders"] = orders
    manifest["rounds_per_process"] = rounds
    write(output / "manifest.json", manifest)
    for pass_index, order in enumerate(orders):
        for r in order:
            path = output / ("R{}-pass{}.csv".format(r, pass_index + 1))
            exp.write_csv(path, exp.operator_fields(), [])
            command = affinity + [str(exp.find_binary(build, "tmac-cache-operator")), str(model), "4096", str(k), "1", str(rounds), "3", "20241025", str(path)]
            if r:
                command += [str(r), "256"]
            recorder.run(command, env=env)
            check_config = dict(config)
            check_config["operator"] = dict(config["operator"], shapes=[[4096, k]], tokens=[1])
            failures, rows = exp.validate_operator_results(check_config, path)
            if failures or len(rows) != rounds or any(row["status"] != "pass" or not math.isfinite(float(row["total_us"])) or float(row["total_us"]) <= 0 for row in rows):
                raise RuntimeError("C operator check failed for R=" + str(r))
            for row in rows:
                row["round"] = int(row["round"]) + pass_index * rounds
                row["note"] += ";pass=" + str(pass_index + 1)
            all_rows.extend(rows)
            exp.write_csv(output / "operator.csv", exp.operator_fields(), all_rows)
    medians = {name: statistics.median(float(row["total_us"]) for row in all_rows if row["variant"] == name) for name in sorted({row["variant"] for row in all_rows})}
    write(output / "summary.json", {"status": "operator_pass_model_pending", "total_us_medians": medians,
          "tile_reuse_speedups": {name: medians["C-R1-BK256"] / medians[name] for name in ("C-R2-BK256", "C-R4-BK256")},
          "operator_speedups": {name: medians["tmac_ref"] / value for name, value in medians.items() if name != "tmac_ref"}})
    manifest["status"] = "operator_pass_model_pending"
    write(output / "manifest.json", manifest)


def main():
    config = exp.load_config(exp.EXPERIMENT_DIR / "config.json", [])
    root = pathlib.Path(config["paths"]["results_root"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--k", type=int, choices=[4096, 14336], default=4096)
    parser.add_argument("--reuse-run", type=pathlib.Path, help="reuse a completed C run; never rebuild")
    args = parser.parse_args()
    baseline = (args.baseline or find_baseline(root)).resolve()
    output = root / ("C_operator_compare_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    recorder = exp.Recorder(output)
    try:
        execute(baseline, output, recorder, args.k, args.reuse_run.resolve() if args.reuse_run else None)
        print("Completed: " + str(output / "summary.json"))
    except Exception as error:
        recorder.log(str(error))
        write(output / "failure.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        with tarfile.open(str(output) + "-logs.tar.gz", "w:gz") as archive:
            for pattern in ("*.json", "*.csv", "*.log"):
                for path in output.glob(pattern):
                    archive.add(path, arcname=path.name)
            for name in ("harness/compile_commands.json", "harness/CMakeCache.txt", "tmac-build/CMakeCache.txt"):
                if (output / name).is_file():
                    archive.add(output / name, arcname=name)
        print("Return log archive: " + str(output) + "-logs.tar.gz")


if __name__ == "__main__":
    main()
