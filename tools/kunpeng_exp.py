#!/usr/bin/env python3
"""Traceable experimental entry point. Unsupported paths fail closed."""
import argparse
import csv
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "tmac_experiment"
sys.path.insert(0, str(EXP / "scripts"))
import experiment as legacy

VARIANTS = ("llama_ref", "tmac_ref", "A", "B", "AB", "C", "AC", "BC", "ABC")


def selected(args):
    if args.opts is not None:
        opts = args.opts.split(",") if args.opts else []
        if len(set(opts)) != len(opts) or any(x not in "ABC" or len(x) != 1 for x in opts):
            raise ValueError("--opts requires distinct A,B,C names")
        return ["".join(sorted(opts)) or "tmac_off"]
    names = args.variants.split(",")
    if len(set(names)) != len(names) or any(x not in VARIANTS for x in names):
        raise ValueError("unknown or duplicate variant")
    return names


def run(command, **kwargs):
    return subprocess.check_output(list(map(str, command)), text=True, stderr=subprocess.STDOUT, **kwargs)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def probe(output):
    data = {"system": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        data["allowed_cpus"] = sorted(os.sched_getaffinity(0))
    for command in (["lscpu", "-J"], ["numactl", "--hardware"], ["clang", "--version"], ["cmake", "--version"]):
        try:
            data[" ".join(command)] = run(command)
        except (OSError, subprocess.CalledProcessError) as error:
            data[" ".join(command)] = {"unavailable": str(error)}
    for name in ("/proc/self/cgroup", "/proc/self/status", "/proc/cpuinfo", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpuset.cpus.effective"):
        path = pathlib.Path(name)
        if path.is_file():
            data[name] = path.read_text()
    write(output / "probe.json", data)


def check_lut(source, output):
    if platform.machine() not in ("arm64", "aarch64"):
        raise RuntimeError("NEON comparison requires ARM; pending server verification")
    original = source / "python/t_mac/intrins/lut_ctor.cc"
    if legacy.sha256_file(original) != "f03f1fb0e9acd1cc9679fbdb1d47dc2599f97806eba7d5b0bd996fa8da9e386b":
        raise RuntimeError("LUT source differs from pinned original")
    runtime_env = os.environ.copy()
    if (source.parent / "setup-environment.json").is_file():
        runtime_env.update(legacy.source_environment(source))
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        target = work / "python/t_mac/intrins/lut_ctor.cc"
        target.parent.mkdir(parents=True)
        shutil.copy2(original, target)
        run(["git", "apply", "--check", EXP / "optimizations/A.patch"], cwd=work)
        run(["git", "apply", EXP / "optimizations/A.patch"], cwd=work)
        wrappers = []
        for name, path in (("ref", original), ("opt", target)):
            wrapper = work / (name + ".cpp")
            wrapper.write_text('#define partial_max_reset partial_max_reset_' + name + '\n#define partial_max_g4_int8_k8 partial_max_g4_int8_k8_' + name + '\n#include "' + str(path) + '"\nextern "C" void lut_' + name + '(int k, int8_t *q, float16_t *b, float16_t *s, float16_t *bias) { lut_ctor_g4_int8_impl<0,4>(k,q,b,s,bias); }\n')
            wrappers.append(wrapper)
        compiler = shutil.which("clang++", path=runtime_env.get("PATH"))
        if not compiler:
            raise RuntimeError("clang++ missing")
        flags = ["-O3", "-std=c++17", "-march=armv8.2-a+fp16"]
        binary = work / "check-lut"
        run([compiler] + flags + wrappers + [EXP / "harness/lut-store.cpp", "-o", binary], env=runtime_env)
        result = run([binary], env=runtime_env)
        output.mkdir(parents=True, exist_ok=True)
        run([compiler] + flags + ["-S", wrappers[1], "-o", output / "lut-A.s"], env=runtime_env)
        run([compiler] + flags + ["-S", wrappers[0], "-o", output / "lut-ref.s"], env=runtime_env)
        write(output / "lut-check.json", {"status": "pass", "result": result, "compiler": run([compiler, "--version"], env=runtime_env), "patch_sha256": legacy.sha256_file(EXP / "optimizations/A.patch")})
        print(result, end="")


def build(config, names, output):
    source = pathlib.Path(config["paths"]["tmac_root"]) / "source"
    commits = legacy.verify_source(config, source)
    legacy.validate_setup_state(config, commits, source)
    changed = run(["git", "diff", "HEAD", "--name-only"], cwd=source).splitlines()
    if set(changed) - {"python/t_mac/platform.py", ".gitmodules"}:
        raise RuntimeError("baseline contains unexpected source changes: " + str(changed))
    for name in names:
        if name not in ("tmac_ref", "tmac_off", "A"):
            raise RuntimeError(name + " is not implemented; no substitute is permitted")
    for name in names:
        key = legacy.fingerprint(config, commits)[:16] + "-" + legacy.sha256_file(EXP / "optimizations/A.patch")[:12]
        root = output / "variants" / name / key
        copy = root / "source"
        if copy.exists():
            raise RuntimeError("isolated source already exists; use a fresh --output to rebuild: " + str(copy))
        shutil.copytree(source, copy, ignore=shutil.ignore_patterns(".git", "3rdparty", "build", "__pycache__"))
        (copy / "build").symlink_to(source / "build", target_is_directory=True)
        shutil.copy2(source.parent / "setup-environment.json", root / "setup-environment.json")
        (copy / "3rdparty").symlink_to(source / "3rdparty", target_is_directory=True)
        (root / "venv").symlink_to(source.parent / "venv", target_is_directory=True)
        if name == "A":
            run(["git", "apply", EXP / "optimizations/A.patch"], cwd=copy)
        local = json.loads(json.dumps(config))
        local["paths"]["tmac_root"] = str(root)
        recorder = legacy.Recorder(root)
        # Editable installation must resolve the isolated optimization sources first.
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(copy / "python") + os.pathsep + (previous or "")
        try:
            builds, harnesses, seconds, artifacts = legacy.build_baselines(local, recorder, commits)
        finally:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous
        write(root / "manifest.json", {"variant": name, "commits": commits, "config": local, "builds": {k: str(v) for k, v in builds.items()}, "harnesses": {k: str(v) for k, v in harnesses.items()}, "artifacts": artifacts, "tune_seconds": seconds, "status": "built_pending_correctness", "layout": "upstream", "opts": ["A"] if name == "A" else []})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "build", "check", "bench", "summarize", "bundle"))
    parser.add_argument("--config", type=pathlib.Path, default=EXP / "config.json")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--set", action="append", default=[])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--variants", default="tmac_ref,A")
    group.add_argument("--opts")
    args = parser.parse_args()
    names = selected(args)
    config = legacy.load_config(args.config, args.set)
    output = (args.output or pathlib.Path(config["paths"]["results_root"]) / "kunpeng").resolve()
    if args.command == "probe":
        probe(output)
    elif args.command == "check":
        if any(name not in ("tmac_ref", "tmac_off", "A") for name in names):
            raise RuntimeError("requested variant is not implemented")
        output.mkdir(parents=True, exist_ok=True)
        legacy.local_test(config, legacy.Recorder(output))
        if args.source:
            check_lut(args.source.resolve(), output)
        else:
            print("Scalar checks passed. Actual LUT, operator and model checks pending; supply --source for LUT comparison.")
    elif args.command == "build":
        build(config, names, output)
    elif args.command == "bench":
        raise RuntimeError("variant operator/model correctness gates are not connected yet; refusing performance results")
    elif args.command == "summarize":
        rows = []
        for path in output.rglob("end_to_end.csv"):
            for row in csv.DictReader(path.open()):
                row["source"] = str(path)
                rows.append(row)
        write(output / "summary.json", {"raw_rows": rows, "valid_speedups": [], "reason": "legacy rows lack strict llama_ref and per-variant correctness gates"})
    else:
        output.mkdir(parents=True, exist_ok=True)
        files = [p for p in EXP.rglob("*") if p.is_file() and not any(x in p.parts for x in (".local-results", "__pycache__"))]
        files.append(pathlib.Path(__file__).resolve())
        write(output / "bundle-manifest.json", {str(p.relative_to(ROOT)): legacy.sha256_file(p) for p in files})
        with tarfile.open(output / "kunpeng-experiment.tar.gz", "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(ROOT)))
            archive.add(output / "bundle-manifest.json", arcname="bundle-manifest.json")
        print(output / "kunpeng-experiment.tar.gz")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(getattr(error, "output", None) or str(error))
