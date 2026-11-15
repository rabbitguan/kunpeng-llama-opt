#!/usr/bin/env python3
"""Small, untuned, single-thread decode baseline experiment."""
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
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmac_experiment/scripts"))
import experiment as exp

UPSTREAM = "dca1d4b58a7f1acf1bd253be84e50d6367f492fd"
LOGITS_NMSE_LIMIT = 0.01


def compare_logits(reference, actual):
    if not reference or len(reference) != len(actual):
        raise RuntimeError("logits dimensions differ or are empty")
    finite = all(math.isfinite(x) for x in reference + actual)
    if not finite:
        return {"status": "failed", "nonfinite": True}
    error = sum((a - b) ** 2 for a, b in zip(reference, actual))
    norm = sum(x * x for x in reference)
    nmse = error / norm if norm else (0.0 if error == 0 else math.inf)
    return {"status": "pass" if nmse <= LOGITS_NMSE_LIMIT else "failed", "nmse": nmse if math.isfinite(nmse) else None,
            "max_abs": max(abs(a - b) for a, b in zip(reference, actual)), "nmse_limit": LOGITS_NMSE_LIMIT, "nonfinite": False}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def snapshot(recorder, repository, revision, target):
    target.mkdir(parents=True)
    archive = target.parent / (target.name + ".tar")
    recorder.run(["git", "archive", "--format=tar", "--output", str(archive), revision], cwd=repository)
    recorder.run(["tar", "-xf", str(archive), "-C", str(target)])
    archive.unlink()


def execute(config, output, recorder):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run this baseline on the Linux AArch64 server")
    source = pathlib.Path(config["paths"]["tmac_root"]) / "source"
    commits = exp.verify_source(config, source)
    exp.validate_setup_state(config, commits, source)
    env = os.environ.copy()
    env.update(exp.source_environment(source))
    from kunpeng_exp import probe
    probe(output)
    recorder.run(["clang++", "--version"], env=env)
    model = pathlib.Path(config["paths"]["model"])
    if not model.is_file():
        raise RuntimeError("Configured model does not exist")
    cpus = exp.physical_cpus(config["model_benchmark"]["numa_node"])
    if not cpus:
        raise RuntimeError("No allowed physical CPU on the configured NUMA node")
    quota = pathlib.Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        values = quota.read_text().split()
        if values[0] != "max" and int(values[0]) < int(values[1]):
            raise RuntimeError("CPU quota is less than one CPU; cannot run this baseline")
    affinity = exp.affinity_prefix(cpus[:1], config["model_benchmark"]["numa_node"])
    fork = source / "3rdparty/llama.cpp"
    recorder.run(["git", "diff", "--exit-code", "HEAD"], cwd=fork)
    manifest = {"status": "building", "commits": commits, "llama_ref_commit": UPSTREAM, "model": str(model),
                "model_sha256": exp.sha256_file(model), "config": config, "threads": 1, "affinity": affinity,
                "tuning": "disabled_official_fallback", "opts": [], "layout": "upstream", "decode_tokens": 32,
                "prompt_tokens": 0, "cache_type_k": "f16", "cache_type_v": "f16", "rounds": 3,
                "limitations": ["preliminary untuned baseline", "preset Llama-3-8B kernel shapes", "dispatch evidence is initialization, not per-operator coverage", "fixed-token logits smoke is not a quality evaluation"]}
    write(output / "manifest.json", manifest)
    upstream_repo = fork
    if not exp.git_has_commit(fork, UPSTREAM):
        upstream_repo = output / "upstream-git"
        recorder.run(["git", "init", str(upstream_repo)])
        exp.run_git_network(recorder, ["git", "fetch", "--depth=1", "https://github.com/ggml-org/llama.cpp.git", UPSTREAM], upstream_repo)
    tmac = output / "tmac-source"
    snapshot(recorder, source, commits["tmac"], tmac)
    recorder.run(["git", "apply", str(exp.EXPERIMENT_DIR / "patches/0001-kunpeng-device.patch")], cwd=tmac)
    (tmac / "3rdparty").mkdir(exist_ok=True)
    for name in ("tvm", "llama.cpp"):
        (tmac / "3rdparty" / name).rmdir()
        (tmac / "3rdparty" / name).symlink_to(source / "3rdparty" / name, target_is_directory=True)
    upstream = output / "llama-ref-source"
    snapshot(recorder, upstream_repo, UPSTREAM, upstream)
    generated = output / "generated"
    generation_env = env.copy()
    generation_env["PYTHONPATH"] = str(tmac / "python") + os.pathsep + env.get("PYTHONPATH", "")
    python = source.parent / "venv/bin/python"
    recorder.run([str(python), "compile.py", "-o", str(generated), "-d", "kunpeng", "-tb", "-gc", "-da", "-nt", "1", "-gs", "32", "-ags", "32", "-m", "llama-3-8b-4bit"], cwd=tmac / "deploy", env=generation_env)
    cfg = configparser.ConfigParser()
    cfg.read(generated / "kcfg.ini")
    if not cfg.sections() or any(cfg.getint(section, "kfactor") != 8 for section in cfg.sections()):
        raise RuntimeError("Unexpected generated kfactor; expected 8")
    for name in ("kernels.cc", "kernels.h", "kcfg.ini"):
        shutil.copy2(generated / name, tmac / "deploy/tuned" / name)
    install = output / "install"
    jobs = min(4, len(cpus))
    common = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    recorder.run(["cmake", "-S", str(tmac), "-B", str(output / "tmac-build"), "-DCMAKE_INSTALL_PREFIX=" + str(install), "-DTMAC_TVM=OFF", "-DCMAKE_INSTALL_LIBDIR=lib"] + common, env=env)
    recorder.run(["cmake", "--build", str(output / "tmac-build"), "--target", "install", "-j", str(jobs)], env=env)
    env["TMAC_KCFG_FILE"] = str(generated / "kcfg.ini")
    env["LD_LIBRARY_PATH"] = str(install / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["TVM_NUM_THREADS"] = "1"
    binaries, logits = {}, {}
    for name, tree, enabled in (("llama_ref", upstream, False), ("tmac_ref", fork, True)):
        build = output / (name + "-build")
        flags = common + ["-DGGML_NATIVE=ON"]
        if enabled:
            flags += ["-DGGML_TMAC=ON", "-DGGML_TMAC_TVM_THREADPOOL=OFF", "-DGGML_TMAC_RECHUNK=OFF", "-DCMAKE_PREFIX_PATH=" + str(install / "lib/cmake/t-mac")]
        recorder.run(["cmake", "-S", str(tree), "-B", str(build)] + flags, env=env)
        recorder.run(["cmake", "--build", str(build), "--target", "llama-bench", "-j", str(jobs)], env=env)
        binaries[name] = exp.find_binary(build, "llama-bench")
        harness = output / (name + "-harness")
        recorder.run(["cmake", "-S", str(exp.EXPERIMENT_DIR / "harness"), "-B", str(harness), "-DLLAMA_SOURCE=" + str(tree)] + flags, env=env)
        recorder.run(["cmake", "--build", str(harness), "--target", "tmac-logits", "-j", str(jobs)], env=env)
        logfile = output / (name + "-logits.bin")
        result = recorder.run(affinity + [str(exp.find_binary(harness, "tmac-logits")), str(model), str(logfile), "1", "128000,791,4062,14198,39935,35308,13"], env=env, capture=True)
        if enabled and ("ggml_tmac_init" not in result.stdout or "Failed to find kcfg" in result.stdout):
            raise RuntimeError("T-MAC dispatch initialization was not confirmed")
        logits[name] = exp.read_logits(logfile)
    correctness = compare_logits(logits["llama_ref"], logits["tmac_ref"])
    write(output / "correctness.json", correctness)
    if correctness["status"] != "pass":
        raise RuntimeError("Fixed-token logits check failed; no valid timing comparison")
    rows = []
    for round_index in range(4):
        for name, binary in binaries.items():
            command = affinity + [str(binary), "-m", str(model), "-t", "1", "-ngl", "0", "-p", "0", "-n", "32", "-r", "1", "-b", "32", "-ub", "32", "-ctk", "f16", "-ctv", "f16"]
            result = recorder.run(command, env=env, capture=True)
            parsed = exp.parse_bench(result.stdout)
            if len(parsed) != 1 or not math.isfinite(parsed[0]["tokens_per_second"]) or parsed[0]["tokens_per_second"] <= 0:
                raise RuntimeError("Invalid benchmark output for " + name)
            if name == "tmac_ref" and ("ggml_tmac_init" not in result.stdout or "Failed to find kcfg" in result.stdout):
                raise RuntimeError("T-MAC benchmark dispatch initialization missing")
            if round_index:
                rows.append({"variant": name, "round": round_index, "tokens_per_second": parsed[0]["tokens_per_second"]})
            exp.write_csv(output / "decode.csv", ["variant", "round", "tokens_per_second"], rows)
    medians = {name: statistics.median(row["tokens_per_second"] for row in rows if row["variant"] == name) for name in binaries}
    write(output / "summary.json", {"status": "preliminary", "median_tokens_per_second": medians,
          "tmac_over_llama": medians["tmac_ref"] / medians["llama_ref"], "ms_per_token": {k: 1000 / v for k, v in medians.items()}})
    manifest["status"] = "preliminary_pass"
    manifest["generated_sha256"] = {name: exp.sha256_file(generated / name) for name in ("kernels.cc", "kernels.h", "kcfg.ini")}
    write(output / "manifest.json", manifest)
    print("Completed: " + str(output / "summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(exp.EXPERIMENT_DIR / "config.json"))
    args = parser.parse_args()
    config = exp.load_config(args.config, [])
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = pathlib.Path(config["paths"]["results_root"]) / ("basic_baseline_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    recorder = exp.Recorder(output)
    try:
        execute(config, output, recorder)
    except Exception as error:
        recorder.log(str(error))
        write(output / "failure.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        import tarfile
        with tarfile.open(str(output) + "-logs.tar.gz", "w:gz") as archive:
            for name in ("console.log", "manifest.json", "failure.json", "correctness.json", "decode.csv", "summary.json", "probe.json", "llama_ref-logits.bin", "tmac_ref-logits.bin", "generated/kcfg.ini"):
                if (output / name).is_file():
                    archive.add(output / name, arcname=name)
        print("Return log archive: " + str(output) + "-logs.tar.gz")


if __name__ == "__main__":
    main()
