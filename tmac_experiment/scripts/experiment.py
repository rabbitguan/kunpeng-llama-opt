#!/usr/bin/env python3
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import posixpath
import re
import signal
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import warnings

from reference import self_test


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
LLVM_VERSION = "17.0.6"
LLVM_ARCHIVE_NAMES = {
    ("Linux", "aarch64"): "clang+llvm-17.0.6-aarch64-linux-gnu.tar.xz",
    ("Linux", "x86_64"): "clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz",
    ("Darwin", "arm64"): "clang+llvm-17.0.6-arm64-apple-darwin22.0.tar.xz",
}
VENDOR_DIR = EXPERIMENT_DIR / "vendor"
LLVM_CONFIG_QUERIES = ["--cxxflags", "--ldflags", "--libfiles", "--system-libs", "--targets-built"]
DEPENDENCY_SPECS = {
    "zlib": {"flag": "-lz", "source": "#include <zlib.h>\n#include <zconf.h>\nint main() { return zlibVersion() ? 0 : 1; }\n", "pkg": "zlib"},
    "zstd": {"flag": "-lzstd", "source": "#include <zstd.h>\nint main() { return ZSTD_versionNumber() > 0 ? 0 : 1; }\n", "pkg": "libzstd"},
    "xml2": {"flag": "-lxml2", "source": "#include <libxml/parser.h>\nint main() { xmlDocPtr d = xmlReadMemory(\"<a/>\", 4, 0, 0, 0); if (!d) return 1; xmlFreeDoc(d); return 0; }\n", "pkg": "libxml-2.0"},
    "tinfo": {"flag": "-ltinfo", "source": "#include <curses.h>\n#include <term.h>\nint main() { int error = 0; (void)setupterm(0, 1, &error); return 0; }\n", "pkg": "tinfo"},
}
TINFO_BUILD_RECIPE = {
    "revision": 2,
    "configure": ["--disable-widec", "--with-shared", "--with-termlib=tinfo", "--with-pic", "--without-debug", "--without-ada", "--without-cxx-binding", "--enable-pc-files"],
}


def load_config(path, overrides):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    for item in overrides:
        if "=" not in item:
            raise ValueError("--set requires dotted.path=value")
        dotted, raw = item.split("=", 1)
        cursor = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cursor[parts[-1]] = value
    return config


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fingerprint(config, source):
    payload = {
        "config": config,
        "source": source,
        "patches": {path.name: sha256_file(path) for path in sorted((EXPERIMENT_DIR / "patches").glob("*.patch"))},
        "scripts": {path.name: sha256_file(path) for path in sorted(SCRIPT_DIR.glob("*.py"))},
        "vendor": {path.name: sha256_file(path) for path in sorted(VENDOR_DIR.glob("*")) if path.is_file()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Recorder:
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.log_path = result_dir / "console.log"
        self.commands = []

    def log(self, message):
        line = "[{}] {}".format(datetime.datetime.now().isoformat(timespec="seconds"), message)
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def terminate(self, process):
        if process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()

    def run(self, command, cwd=None, env=None, timeout=None, check=True, capture=False, heartbeat=60):
        rendered = shlex.join([str(value) for value in command])
        self.log("RUN cwd={} command={}".format(cwd or os.getcwd(), rendered))
        started = time.monotonic()
        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        process = subprocess.Popen(command, cwd=cwd, env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, start_new_session=(os.name == "posix"))
        output_parts = []
        timed_out = False
        with open(self.log_path, "a", encoding="utf-8") as handle:
            def consume_output():
                for line in process.stdout:
                    output_parts.append(line)
                    handle.write(line)
                    handle.flush()
                    if not capture:
                        print(line, end="", flush=True)
            reader = threading.Thread(target=consume_output)
            reader.start()
            deadline = started + timeout if timeout is not None else None
            try:
                while process.poll() is None:
                    interval = heartbeat
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            timed_out = True
                            self.terminate(process)
                            break
                        interval = min(interval, remaining)
                    try:
                        process.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        if deadline is not None and time.monotonic() >= deadline:
                            timed_out = True
                            self.terminate(process)
                            break
                        self.log("still running after {:.0f} seconds: {}".format(time.monotonic() - started, rendered))
            except (KeyboardInterrupt, SystemExit):
                self.terminate(process)
                raise
            finally:
                reader.join()
        process.stdout.close()
        returncode = process.returncode
        elapsed = time.monotonic() - started
        output = "".join(output_parts)
        self.commands.append({"command": rendered, "cwd": str(cwd or os.getcwd()), "seconds": elapsed, "exit_code": returncode})
        self.log("DONE exit_code={} elapsed={:.1f}s command={}".format(returncode, elapsed, rendered))
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout, output=output)
        completed = subprocess.CompletedProcess(command, returncode, output, None)
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, command, output=output)
        return completed


def command_version(command, env=None):
    path = shutil.which(command, path=(env or os.environ).get("PATH"))
    if not path:
        return {"path": None, "version": None}
    completed = subprocess.run([path, "--version"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"path": os.path.realpath(path), "version": (completed.stdout or "").splitlines()[:2]}


def gitlink_commit(repository, revision, path):
    output = subprocess.check_output(["git", "ls-tree", revision, "--", path], cwd=repository, text=True).strip()
    fields = output.split(None, 3)
    if len(fields) != 4 or fields[0] != "160000" or fields[1] != "commit":
        raise RuntimeError("{} is not a gitlink in {} at {}".format(path, repository, revision))
    return fields[2]


def verify_submodule_set(repository, revision, required, skipped):
    actual = {}
    skipped_actual = {}
    for path, commit in required.items():
        pinned = gitlink_commit(repository, revision, path)
        if pinned != commit:
            raise RuntimeError("gitlink {} mismatch: expected {}, got {}".format(path, commit, pinned))
        checkout = repository / path
        if not git_has_commit(checkout, commit):
            raise RuntimeError("required submodule {} is not initialized at {}".format(path, commit))
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if value != commit:
            raise RuntimeError("submodule {} mismatch: expected {}, got {}".format(path, commit, value))
        actual[path] = value
    for path, commit in skipped.items():
        pinned = gitlink_commit(repository, revision, path)
        if pinned != commit:
            raise RuntimeError("skipped gitlink {} mismatch: expected {}, got {}".format(path, commit, pinned))
        skipped_actual[path] = {"commit": pinned, "status": "skipped"}
    return actual, skipped_actual


def verify_source(config, source):
    expected = config["source"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != expected["commit"]:
        raise RuntimeError("T-MAC commit mismatch: expected {}, got {}".format(expected["commit"], head))
    actual, skipped = verify_submodule_set(source, head, expected["required_submodules"], expected["skipped_submodules"])
    tvm = source / "3rdparty" / "tvm"
    tvm_expected = expected["tvm_submodules"]
    tvm_actual, tvm_skipped = verify_submodule_set(tvm, expected["required_submodules"]["3rdparty/tvm"], tvm_expected["required"], tvm_expected["skipped"])
    return {"tmac": head, "submodules": actual, "skipped_submodules": skipped, "tvm_submodules": tvm_actual, "skipped_tvm_submodules": tvm_skipped}


def git_has_commit(repository, commit):
    if not repository.is_dir():
        return False
    completed = subprocess.run(["git", "cat-file", "-e", "{}^{{commit}}".format(commit)], cwd=repository, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return completed.returncode == 0


def run_git_network(recorder, command, cwd, attempts=3, timeout=180):
    last_error = None
    for attempt in range(1, attempts + 1):
        recorder.log("git network attempt {}/{}".format(attempt, attempts))
        try:
            completed = recorder.run(command, cwd=cwd, timeout=timeout, check=False)
            if completed.returncode == 0:
                return
            last_error = subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)
        except subprocess.TimeoutExpired as error:
            last_error = error
            recorder.log("git command timed out after {} seconds".format(timeout))
        if attempt < attempts:
            time.sleep(attempt * 2)
    raise RuntimeError("git command failed after {} attempts: {}".format(attempts, last_error))


def prepare_submodule_set(recorder, parent, submodules):
    for relative, commit in submodules.items():
        recorder.run(["git", "submodule", "sync", "--", relative], cwd=parent)
        repository = parent / relative
        if git_has_commit(repository, commit):
            recorder.log("reuse local submodule commit {} at {}".format(commit, relative))
        else:
            recorder.log("submodule commit {} is missing at {}; downloading it".format(commit, relative))
            run_git_network(recorder, ["git", "-c", "http.version=HTTP/1.1", "submodule", "update", "--init", "--", relative], parent)
        if not git_has_commit(repository, commit):
            raise RuntimeError("submodule {} does not contain required commit {}".format(relative, commit))
        recorder.run(["git", "checkout", "--detach", commit], cwd=repository)


def prepare_submodules(config, recorder, source):
    expected = config["source"]
    prepare_submodule_set(recorder, source, expected["required_submodules"])
    for relative, commit in expected["skipped_submodules"].items():
        recorder.log("skip unused submodule {} at pinned gitlink {}".format(relative, commit))
    tvm = source / "3rdparty" / "tvm"
    prepare_submodule_set(recorder, tvm, expected["tvm_submodules"]["required"])
    for relative, commit in expected["tvm_submodules"]["skipped"].items():
        recorder.log("skip unused TVM submodule {} at pinned gitlink {}".format(relative, commit))


def apply_patch(recorder, source):
    for patch in sorted((EXPERIMENT_DIR / "patches").glob("*.patch")):
        reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if reverse.returncode == 0:
            recorder.log("adaptation patch {} is already applied".format(patch.name))
            continue
        recorder.run(["git", "apply", "--check", str(patch)], cwd=source)
        recorder.run(["git", "apply", str(patch)], cwd=source)


def prepend_path(value, current):
    parts = [str(value)]
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def llvm_candidates(source, environ=None):
    environ = os.environ if environ is None else environ
    candidates = []
    configured = environ.get("TMAC_LLVM_ROOT")
    if configured:
        candidates.append(pathlib.Path(configured))
    build = source / "build"
    if build.is_dir():
        candidates.extend(path for path in sorted(build.glob("clang+llvm-17.0.6-*")) if path.is_dir())
    llvm_config = shutil.which("llvm-config", path=environ.get("PATH"))
    if llvm_config:
        candidates.append(pathlib.Path(llvm_config).resolve().parent.parent)
    result = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def command_output(command, env=None):
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.returncode, (completed.stdout or "").strip(), (completed.stderr or "").strip()


def validate_llvm_root(root):
    binary = root / "bin"
    required = [binary / "clang", binary / "clang++", binary / "llvm-config"]
    missing = [str(path) for path in required if not path.is_file()]
    not_executable = [str(path) for path in required if path.is_file() and not os.access(path, os.X_OK)]
    if missing or not_executable:
        details = []
        if missing:
            details.append("missing files: {}".format(", ".join(missing)))
        if not_executable:
            details.append("not executable: {}".format(", ".join(not_executable)))
        return False, "; ".join(details)
    versions = {}
    warnings_by_tool = {}
    for path in required:
        try:
            returncode, stdout, stderr = command_output([str(path), "--version"])
        except OSError as error:
            return False, "cannot execute {}: {}".format(path, error)
        if stderr:
            warnings_by_tool[path.name] = stderr
        if returncode:
            return False, "{} --version failed with exit code {}: stdout={!r}; stderr={!r}".format(path, returncode, stdout, stderr)
        if "error while loading shared libraries" in stderr or "symbol lookup error" in stderr:
            return False, "{} --version reported a dynamic loader failure: {}".format(path, stderr)
        versions[path.name] = stdout
    llvm_version = versions["llvm-config"]
    if llvm_version != LLVM_VERSION:
        return False, "llvm-config stdout {!r} is not the pinned version {!r}; stderr={!r}".format(llvm_version, LLVM_VERSION, warnings_by_tool.get("llvm-config", ""))
    if LLVM_VERSION not in versions["clang"] or LLVM_VERSION not in versions["clang++"]:
        return False, "clang and clang++ do not report pinned version {}: clang={!r}, clang++={!r}".format(LLVM_VERSION, versions["clang"], versions["clang++"])
    versions["stderr"] = warnings_by_tool
    return True, versions


def find_llvm_root(source):
    diagnostics = []
    for candidate in llvm_candidates(source):
        valid, detail = validate_llvm_root(candidate)
        if valid:
            return candidate, detail
        diagnostics.append("{}: {}".format(candidate, detail))
    return None, diagnostics


def llvm_archive_name():
    machine = platform.machine()
    machine = {"arm64": "aarch64", "AMD64": "x86_64", "ARM64": "aarch64"}.get(machine, machine)
    return LLVM_ARCHIVE_NAMES.get((platform.system(), machine))


def llvm_archive_candidates(source, environ=None):
    environ = os.environ if environ is None else environ
    candidates = []
    configured = environ.get("TMAC_LLVM_ARCHIVE")
    if configured:
        candidates.append(pathlib.Path(configured))
    name = llvm_archive_name()
    if name:
        candidates.extend([source / "build" / name, source / name, source.parent / name, source.parent / "downloads" / name])
    result = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def extract_local_llvm_archive(recorder, source, archive):
    official_name = llvm_archive_name()
    expected_name = official_name[:-len(".tar.xz")] if official_name else archive.name[:-len(".tar.xz")]
    recorder.log("validating local LLVM archive {}".format(archive))
    (source / "build").mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:xz") as handle:
            members = handle.getmembers()
            for member in members:
                path = pathlib.PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != expected_name:
                    raise RuntimeError("unsafe or unexpected path in LLVM archive: {}".format(member.name))
                if member.issym() or member.islnk():
                    target = pathlib.PurePosixPath(member.linkname)
                    base = path.parent if member.issym() else pathlib.PurePosixPath()
                    resolved_target = pathlib.PurePosixPath(posixpath.normpath(str(base / target)))
                    if target.is_absolute() or not resolved_target.parts or resolved_target.parts[0] != expected_name:
                        raise RuntimeError("unsafe link in LLVM archive: {} -> {}".format(member.name, member.linkname))
            recorder.log("extracting local LLVM archive over the incomplete directory; existing files are preserved")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                handle.extractall(source / "build")
    except (OSError, tarfile.TarError) as error:
        raise RuntimeError("local LLVM archive is unreadable or incomplete: {}: {}".format(archive, error)) from error
    return source / "build" / expected_name


def prepare_llvm(recorder, source):
    llvm_root, diagnostics = find_llvm_root(source)
    if llvm_root is not None:
        for tool, warning in diagnostics.get("stderr", {}).items():
            recorder.log("{} --version stderr warning: {}".format(tool, warning))
        return llvm_root, diagnostics
    archives = [path for path in llvm_archive_candidates(source) if path.is_file()]
    if archives:
        restored = extract_local_llvm_archive(recorder, source, archives[0])
        valid, detail = validate_llvm_root(restored)
        if not valid:
            raise RuntimeError("LLVM remains invalid after extracting {}: {}".format(archives[0], detail))
        for tool, warning in detail.get("stderr", {}).items():
            recorder.log("{} --version stderr warning: {}".format(tool, warning))
        recorder.log("recovered validated LLVM {} from local archive {}".format(LLVM_VERSION, archives[0]))
        return restored.resolve(), detail
    partial = [path for path in llvm_candidates(source) if path.is_dir() and str(path).startswith(str((source / "build").resolve()))]
    if partial:
        searched = ", ".join(str(path) for path in llvm_archive_candidates(source)) or "no archive name for this platform"
        raise RuntimeError("project-local LLVM is incomplete and will not be downloaded again automatically: {}. Place a complete local archive at one of: {}, or set TMAC_LLVM_ARCHIVE".format("; ".join(diagnostics), searched))
    return None, diagnostics


def llvm_config_report(recorder, llvm_root, tvm=None):
    llvm_config = llvm_root / "bin" / "llvm-config"
    report = {}
    errors = []
    for option in LLVM_CONFIG_QUERIES:
        try:
            returncode, stdout, stderr = command_output([str(llvm_config), option])
        except OSError as error:
            errors.append("llvm-config {} could not run: {}".format(option, error))
            continue
        report[option[2:]] = {"stdout": stdout, "stderr": stderr, "returncode": returncode}
        if stderr:
            recorder.log("llvm-config {} stderr: {}".format(option, stderr))
        if returncode:
            errors.append("llvm-config {} failed with exit code {}: {}".format(option, returncode, stderr or stdout))
    libfiles = shlex.split(report.get("libfiles", {}).get("stdout", ""))
    missing_libraries = [path for path in libfiles if not pathlib.Path(path).is_file()]
    if missing_libraries:
        errors.append("LLVM library files are missing: {}".format(", ".join(missing_libraries)))
    targets = shlex.split(report.get("targets-built", {}).get("stdout", ""))
    if "AArch64" not in targets:
        errors.append("LLVM targets-built does not include AArch64: {}".format(" ".join(targets)))
    if tvm is not None:
        find_llvm = pathlib.Path(tvm) / "cmake" / "utils" / "FindLLVM.cmake"
        required_queries = ["--libfiles", "--system-libs", "--cxxflags", "--targets-built"]
        content = find_llvm.read_text(encoding="utf-8", errors="replace") if find_llvm.is_file() else ""
        missing_queries = [query for query in required_queries if query not in content]
        if missing_queries:
            errors.append("fixed TVM FindLLVM.cmake is missing expected llvm-config queries: {}".format(", ".join(missing_queries)))
        report["findllvm"] = {"path": str(find_llvm), "sha256": sha256_file(find_llvm) if find_llvm.is_file() else None, "required_queries": required_queries}
    if errors:
        raise RuntimeError("LLVM configuration checks failed:\n- " + "\n- ".join(errors))
    return report


def dependency_environment(prefix, base_env=None):
    base_env = os.environ if base_env is None else base_env
    lib_dirs = [prefix / "lib", prefix / "lib64"]
    include_dirs = [prefix / "include", prefix / "include" / "ncurses"]
    values = {
        "CMAKE_PREFIX_PATH": prepend_path(prefix, base_env.get("CMAKE_PREFIX_PATH", "")),
        "CPATH": os.pathsep.join([str(path) for path in include_dirs] + ([base_env["CPATH"]] if base_env.get("CPATH") else [])),
        "LIBRARY_PATH": os.pathsep.join([str(path) for path in lib_dirs] + ([base_env["LIBRARY_PATH"]] if base_env.get("LIBRARY_PATH") else [])),
        "LD_LIBRARY_PATH": os.pathsep.join([str(path) for path in lib_dirs] + ([base_env["LD_LIBRARY_PATH"]] if base_env.get("LD_LIBRARY_PATH") else [])),
        "PKG_CONFIG_PATH": os.pathsep.join([str(prefix / "lib" / "pkgconfig"), str(prefix / "lib64" / "pkgconfig"), str(prefix / "share" / "pkgconfig")] + ([base_env["PKG_CONFIG_PATH"]] if base_env.get("PKG_CONFIG_PATH") else [])),
    }
    return values


def pkg_config_flags(package, env, option="--cflags"):
    if not shutil.which("pkg-config", path=env.get("PATH")):
        return []
    completed = subprocess.run(["pkg-config", option, package], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return shlex.split(completed.stdout) if completed.returncode == 0 else []


def path_is_within(path, directory):
    try:
        pathlib.Path(path).resolve().relative_to(pathlib.Path(directory).resolve())
        return True
    except ValueError:
        return False


def tinfo_prefix_inventory(prefix):
    patterns = ["curses.h", "term.h", "ncurses*.pc", "tinfo*.pc", "libtinfo*"]
    return {pattern: [str(path) for path in sorted(prefix.rglob(pattern))] for pattern in patterns}


def local_pkg_config_environment(prefix, env):
    result = env.copy()
    directories = [prefix / "lib" / "pkgconfig", prefix / "lib64" / "pkgconfig", prefix / "share" / "pkgconfig"]
    value = os.pathsep.join(str(path) for path in directories)
    result["PKG_CONFIG_PATH"] = value
    result["PKG_CONFIG_LIBDIR"] = value
    return result


def tinfo_pkg_config(prefix, env):
    inventory = tinfo_prefix_inventory(prefix)
    local_env = local_pkg_config_environment(prefix, env)
    if not shutil.which("pkg-config", path=local_env.get("PATH")):
        return {"inventory": inventory, "pkg_config": {}, "pc_file": None, "header_dir": None, "cflags": [], "ldflags": [], "errors": ["pkg-config is not available"], "env": local_env}
    records = {}
    for option in ["--variable=pcfiledir", "--cflags", "--libs"]:
        returncode, stdout, stderr = command_output(["pkg-config", option, "tinfo"], env=local_env)
        records[option] = {"returncode": returncode, "stdout": stdout, "stderr": stderr}
    pc_directory = pathlib.Path(records["--variable=pcfiledir"]["stdout"]) if records["--variable=pcfiledir"]["stdout"] else None
    pc_file = pc_directory / "tinfo.pc" if pc_directory else None
    errors = []
    for option, record in records.items():
        if record["returncode"]:
            errors.append("pkg-config {} tinfo failed: {}".format(option, record["stderr"] or record["stdout"]))
    if not pc_file or not pc_file.is_file() or not path_is_within(pc_file, prefix):
        errors.append("project-local tinfo.pc was not selected")
    common_headers = []
    curses_dirs = {pathlib.Path(path).parent.resolve() for path in inventory["curses.h"]}
    term_dirs = {pathlib.Path(path).parent.resolve() for path in inventory["term.h"]}
    for directory in sorted(curses_dirs & term_dirs):
        if path_is_within(directory, prefix):
            common_headers.append(directory)
    if not common_headers:
        errors.append("curses.h and term.h were not found in the same project-local include directory")
    cflags = shlex.split(records["--cflags"]["stdout"])
    if common_headers and "-I{}".format(common_headers[0]) not in cflags:
        cflags.append("-I{}".format(common_headers[0]))
    return {
        "inventory": inventory,
        "pkg_config": records,
        "pc_file": str(pc_file) if pc_file else None,
        "header_dir": str(common_headers[0]) if common_headers else None,
        "cflags": cflags,
        "ldflags": shlex.split(records["--libs"]["stdout"]),
        "errors": errors,
        "env": local_env,
    }


def probe_local_tinfo(recorder, compiler, env, work, prefix):
    metadata = tinfo_pkg_config(prefix, env)
    result = {key: value for key, value in metadata.items() if key != "env"}
    recorder.log("tinfo project-local inventory: {}".format(json.dumps(metadata["inventory"], sort_keys=True)))
    recorder.log("tinfo pkg-config selection: pc_file={} cflags={} libs={}".format(metadata["pc_file"], shlex.join(metadata["cflags"]), shlex.join(metadata["ldflags"])))
    if metadata["errors"]:
        result.update({"status": "failed", "stage": "pkg_config", "compile_output": "; ".join(metadata["errors"])})
        return result
    source_file = work / "probe-tinfo.c"
    binary = work / "probe-tinfo"
    source_file.write_text(DEPENDENCY_SPECS["tinfo"]["source"], encoding="ascii")
    library_dirs = [pathlib.Path(flag[2:]) for flag in metadata["ldflags"] if flag.startswith("-L")]
    local_library_dirs = [path for path in library_dirs if path_is_within(path, prefix)]
    if not local_library_dirs:
        result.update({"status": "failed", "stage": "pkg_config", "compile_output": "tinfo.pc did not provide a project-local library directory"})
        return result
    rpaths = ["-Wl,-rpath,{}".format(path) for path in local_library_dirs]
    command = [str(compiler)] + metadata["cflags"] + [str(source_file)] + metadata["ldflags"] + rpaths + ["-o", str(binary)]
    compiled = recorder.run(command, env=metadata["env"], check=False, capture=True)
    result.update({"compile_command": shlex.join(command), "compile_returncode": compiled.returncode, "compile_output": compiled.stdout})
    if compiled.returncode:
        result.update({"status": "failed", "stage": "compile_link"})
        return result
    linker = ["ldd", str(binary)]
    link_returncode, link_stdout, link_stderr = command_output(linker, env=metadata["env"])
    result["dynamic_libraries"] = {"command": shlex.join(linker), "returncode": link_returncode, "stdout": link_stdout, "stderr": link_stderr}
    tinfo_lines = [line for line in link_stdout.splitlines() if "libtinfo" in line]
    if link_returncode or not tinfo_lines or any(str(prefix.resolve()) not in line for line in tinfo_lines):
        result.update({"status": "failed", "stage": "dynamic_library", "run_output": link_stderr or link_stdout or "project-local libtinfo was not resolved"})
        return result
    run_returncode, run_stdout, run_stderr = command_output([str(binary)], env=metadata["env"])
    if run_stderr:
        recorder.log("tinfo runtime stderr warning: {}".format(run_stderr))
    result.update({"run_returncode": run_returncode, "run_stdout": run_stdout, "run_stderr": run_stderr, "run_output": "\n".join(part for part in [run_stdout, run_stderr] if part), "status": "pass" if run_returncode == 0 else "failed", "stage": "run"})
    return result


def probe_dependency(recorder, name, compiler, env, work, prefix=None):
    if name == "tinfo" and prefix is not None:
        return probe_local_tinfo(recorder, compiler, env, work, pathlib.Path(prefix))
    spec = DEPENDENCY_SPECS[name]
    source_file = work / "probe-{}.c".format(name)
    binary = work / "probe-{}".format(name)
    source_file.write_text(spec["source"], encoding="ascii")
    cflags = pkg_config_flags(spec["pkg"], env)
    ldflags = pkg_config_flags(spec["pkg"], env, "--libs")
    command = [str(compiler)] + cflags + [str(source_file)] + ldflags + ([] if spec["flag"] in ldflags else [spec["flag"]]) + ["-o", str(binary)]
    compiled = recorder.run(command, env=env, check=False, capture=True)
    result = {"compile_command": shlex.join(command), "compile_returncode": compiled.returncode, "compile_output": compiled.stdout, "cflags": cflags, "ldflags": ldflags}
    if compiled.returncode:
        result.update({"status": "failed", "stage": "compile_link"})
        return result
    executed = recorder.run([str(binary)], env=env, check=False, capture=True)
    result.update({"run_returncode": executed.returncode, "run_output": executed.stdout, "status": "pass" if executed.returncode == 0 else "failed", "stage": "run"})
    return result


def vendor_manifest():
    path = VENDOR_DIR / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for name, package in manifest["packages"].items():
        archive = VENDOR_DIR / package["archive"]
        if not archive.is_file():
            errors.append("{} archive is missing: {}".format(name, archive))
        elif sha256_file(archive) != package["sha256"]:
            errors.append("{} archive SHA256 mismatch: {}".format(name, archive))
        else:
            try:
                with tarfile.open(archive, "r:*") as handle:
                    if package["license_file"] not in handle.getnames():
                        errors.append("{} archive does not contain declared license {}".format(name, package["license_file"]))
            except (OSError, tarfile.TarError) as error:
                errors.append("{} archive is unreadable: {}".format(name, error))
    if errors:
        raise RuntimeError("vendored dependency validation failed:\n- " + "\n- ".join(errors))
    return manifest


def extract_vendor_source(package, destination):
    marker = destination / ".extracted"
    if marker.is_file() and marker.read_text(encoding="ascii").strip() == package["sha256"]:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = VENDOR_DIR / package["archive"]
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        roots = {pathlib.PurePosixPath(member.name).parts[0] for member in members if pathlib.PurePosixPath(member.name).parts}
        unsafe = len(roots) != 1 or any(pathlib.PurePosixPath(member.name).is_absolute() or ".." in pathlib.PurePosixPath(member.name).parts for member in members)
        archive_root = next(iter(roots)) if len(roots) == 1 else ""
        for member in members:
            if member.issym() or member.islnk():
                path = pathlib.PurePosixPath(member.name)
                target = pathlib.PurePosixPath(member.linkname)
                base = path.parent if member.issym() else pathlib.PurePosixPath()
                resolved_target = pathlib.PurePosixPath(posixpath.normpath(str(base / target)))
                unsafe = unsafe or target.is_absolute() or not resolved_target.parts or resolved_target.parts[0] != archive_root
        if unsafe:
            raise RuntimeError("unsafe vendor archive layout: {}".format(archive))
        extract_root = destination.parent
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            handle.extractall(extract_root)
        extracted = extract_root / archive_root
    if extracted != destination and not destination.exists():
        extracted.rename(destination)
    marker.write_text(package["sha256"] + "\n", encoding="ascii")
    return destination


def build_vendor_dependency(recorder, name, package, deps_root, prefix, compiler_env):
    source = extract_vendor_source(package, deps_root / "src" / "{}-{}".format(name, package["version"]))
    env = os.environ.copy()
    env.update(compiler_env)
    env.update(dependency_environment(prefix, env))
    env.update({"CC": compiler_env["CC"], "CXX": compiler_env["CXX"], "CFLAGS": "-O2 -fPIC", "CXXFLAGS": "-O2 -fPIC"})
    recorder.log("building project-local dependency {} {}".format(name, package["version"]))
    if name == "zlib":
        recorder.run(["./configure", "--prefix={}".format(prefix)], cwd=source, env=env)
        recorder.run(["make", "-j4"], cwd=source, env=env)
        recorder.run(["make", "install"], cwd=source, env=env)
    elif name == "zstd":
        recorder.run(["make", "-C", "lib", "-j4", "BUILD_SHARED=1", "PIC=1"], cwd=source, env=env)
        recorder.run(["make", "-C", "lib", "install", "BUILD_SHARED=1", "PIC=1", "PREFIX={}".format(prefix)], cwd=source, env=env)
    elif name == "xml2":
        recorder.run(["./configure", "--prefix={}".format(prefix), "--without-python", "--without-lzma", "--without-iconv", "--with-zlib"], cwd=source, env=env)
        recorder.run(["make", "-j4"], cwd=source, env=env)
        recorder.run(["make", "install"], cwd=source, env=env)
    elif name == "tinfo":
        recipe_key = hashlib.sha256(json.dumps(TINFO_BUILD_RECIPE, sort_keys=True).encode()).hexdigest()[:16]
        build = deps_root / "build" / "ncurses-{}".format(recipe_key)
        build.mkdir(parents=True, exist_ok=True)
        configure = [str(source / "configure"), "--prefix={}".format(prefix)] + TINFO_BUILD_RECIPE["configure"] + ["--with-pkg-config-libdir={}".format(prefix / "lib" / "pkgconfig")]
        recorder.run(configure, cwd=build, env=env)
        recorder.run(["make", "-j4"], cwd=build, env=env)
        recorder.run(["make", "install"], cwd=build, env=env)
    else:
        raise RuntimeError("no vendor build recipe for {}".format(name))


def required_llvm_dependencies(report):
    flags = shlex.split(report["system-libs"]["stdout"])
    return [name for name, spec in DEPENDENCY_SPECS.items() if spec["flag"] in flags]


def native_dependency_fingerprint(required, manifest, report, toolchain, recipes=None):
    recipes = {"tinfo": TINFO_BUILD_RECIPE} if recipes is None else recipes
    payload = {
        "required": required,
        "manifest": {name: manifest["packages"][name]["sha256"] for name in required},
        "llvm": report,
        "toolchain": toolchain,
        "recipes": {name: recipes.get(name, {"revision": 1}) for name in required},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def prepare_native_dependencies(recorder, root, llvm_root, report):
    manifest = vendor_manifest()
    compiler_env = {"CC": str(llvm_root / "bin" / "clang"), "CXX": str(llvm_root / "bin" / "clang++")}
    required = required_llvm_dependencies(report)
    toolchain = {}
    for name, path in compiler_env.items():
        tool = pathlib.Path(path)
        stat_result = tool.stat()
        returncode, stdout, stderr = command_output([str(tool), "--version"])
        toolchain[name] = {"path": str(tool.resolve()), "size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns, "returncode": returncode, "stdout": stdout, "stderr": stderr}
    dep_key = native_dependency_fingerprint(required, manifest, report, toolchain)
    deps_root = root / "deps" / dep_key
    prefix = deps_root / "prefix"
    work = deps_root / "checks"
    work.mkdir(parents=True, exist_ok=True)
    system_env = os.environ.copy()
    system_probe = {name: probe_dependency(recorder, name, compiler_env["CC"], system_env, work) for name in required}
    existing_env = os.environ.copy()
    existing_env.update(dependency_environment(prefix, existing_env))
    existing_probe = {name: probe_dependency(recorder, name, compiler_env["CC"], existing_env, work, prefix=prefix) for name in required}
    missing = [name for name in required if existing_probe[name]["status"] != "pass"]
    if missing:
        recorder.log("native dependencies requiring project-local recovery: {}".format(", ".join(missing)))
    recipes = {name: TINFO_BUILD_RECIPE if name == "tinfo" else {"revision": 1} for name in required}
    initial_diagnostic = {"fingerprint": dep_key, "required": required, "recipes": recipes, "system": system_probe, "existing_project": existing_probe, "prefix": str(prefix), "llvm_config": report, "toolchain": toolchain}
    (deps_root / "diagnostics.json").write_text(json.dumps(initial_diagnostic, indent=2) + "\n", encoding="utf-8")
    build_errors = []
    for name in missing:
        try:
            build_vendor_dependency(recorder, name, manifest["packages"][name], deps_root, prefix, compiler_env)
        except Exception as error:
            build_errors.append("{}: {}: {}".format(name, type(error).__name__, error))
    final_env = os.environ.copy()
    final_env.update(dependency_environment(prefix, final_env))
    final = {name: probe_dependency(recorder, name, compiler_env["CC"], final_env, work, prefix=prefix) for name in required}
    failures = build_errors + ["{}: {}: {}".format(name, final[name]["stage"], final[name].get("compile_output") or final[name].get("run_output")) for name in required if final[name]["status"] != "pass"]
    diagnostic = {"fingerprint": dep_key, "required": required, "recipes": recipes, "system": system_probe, "existing_project": existing_probe, "final": final, "prefix": str(prefix), "llvm_config": report, "toolchain": toolchain}
    (deps_root / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("native dependency checks failed after local recovery:\n- " + "\n- ".join(failures))
    return prefix, dep_key, diagnostic


def tvm_build_fingerprint(setup_fingerprint, dep_key, report, llvm_root):
    payload = {"setup": setup_fingerprint, "dependencies": dep_key, "llvm": report, "tvm_config": tvm_config_text(llvm_root / "bin" / "llvm-config"), "cc": str(llvm_root / "bin" / "clang"), "cxx": str(llvm_root / "bin" / "clang++")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def validate_setup_state(config, commits, source):
    root = source.parent
    stamp = root / ".setup-fingerprint"
    state_path = root / "setup-environment.json"
    if not stamp.is_file() or not state_path.is_file():
        raise RuntimeError("setup has not completed successfully; run bash tmac_experiment/run.sh setup")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if stamp.read_text(encoding="ascii").strip() != state.get("fingerprint"):
        raise RuntimeError("setup success marker does not match setup-environment.json")
    current = fingerprint(config, commits)
    if state.get("setup_fingerprint") != current:
        raise RuntimeError("setup inputs changed; rerun bash tmac_experiment/run.sh setup")
    for identity in state.get("dependency_diagnostic", {}).get("toolchain", {}).values():
        path = pathlib.Path(identity["path"])
        if not path.is_file():
            raise RuntimeError("validated setup tool is missing: {}".format(path))
        stat_result = path.stat()
        if stat_result.st_size != identity["size"] or stat_result.st_mtime_ns != identity["mtime_ns"]:
            raise RuntimeError("validated setup tool changed: {}; rerun setup".format(path))
    if not has_tvm_compiler_library(source, pathlib.Path(state["tvm_build"])):
        raise RuntimeError("validated TVM library is missing; rerun setup")
    return state


def validate_llvm_api(recorder, root, llvm_root, report, runtime_env):
    smoke = root / "checks" / "llvm-api"
    smoke.mkdir(parents=True, exist_ok=True)
    source = smoke / "main.cpp"
    binary = smoke / "llvm-api-smoke"
    source.write_text("#include <llvm/MC/TargetRegistry.h>\n#include <llvm/Support/TargetSelect.h>\n#include <string>\nint main() { llvm::InitializeAllTargetInfos(); llvm::InitializeAllTargets(); llvm::InitializeAllTargetMCs(); std::string e; return llvm::TargetRegistry::lookupTarget(\"aarch64\", e) ? 0 : 1; }\n", encoding="ascii")
    command = [str(llvm_root / "bin" / "clang++")] + shlex.split(report["cxxflags"]["stdout"]) + [str(source), "-o", str(binary)] + shlex.split(report["ldflags"]["stdout"]) + shlex.split(report["libfiles"]["stdout"]) + shlex.split(report["system-libs"]["stdout"])
    recorder.run(command, env=runtime_env)
    recorder.run([str(binary)], env=runtime_env)
    return {"source": str(source), "binary": str(binary), "command": shlex.join(command)}


def tvm_library_paths(source):
    build = source / "3rdparty" / "tvm" / "build"
    patterns = ["libtvm.so", "libtvm.dylib", "tvm.dll", "libtvm_runtime.so", "libtvm_runtime.dylib", "tvm_runtime.dll"]
    return [build / name for name in patterns if (build / name).is_file()]


def has_tvm_compiler_library(source, tvm_build=None):
    if tvm_build is not None:
        build = pathlib.Path(tvm_build)
        patterns = ["libtvm.so", "libtvm.dylib", "tvm.dll"]
        paths = [build / name for name in patterns if (build / name).is_file()]
    else:
        paths = tvm_library_paths(source)
    required = "libtvm.so" if platform.system() == "Linux" else "libtvm.dylib" if platform.system() == "Darwin" else "tvm.dll"
    return any(path.name == required for path in paths)


def source_environment(source, llvm_root=None, base_env=None, deps_prefix=None, tvm_build=None):
    base_env = os.environ if base_env is None else base_env
    state_path = source.parent / "setup-environment.json"
    if state_path.is_file() and (deps_prefix is None or tvm_build is None):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        deps_prefix = pathlib.Path(state["deps_prefix"]) if deps_prefix is None else deps_prefix
        tvm_build = pathlib.Path(state["tvm_build"]) if tvm_build is None else tvm_build
    if llvm_root is None:
        llvm_root, diagnostics = find_llvm_root(source)
        if llvm_root is None:
            raise RuntimeError("compatible LLVM 17.0.6 was not found: {}".format("; ".join(diagnostics) or "no candidates"))
    tvm = source / "3rdparty" / "tvm"
    tvm_build = tvm / "build" if tvm_build is None else pathlib.Path(tvm_build)
    merged = base_env.copy()
    if deps_prefix is not None:
        merged.update(dependency_environment(pathlib.Path(deps_prefix), merged))
    values = {
        "TVM_ROOT": str(tvm),
        "PATH": prepend_path(llvm_root / "bin", merged.get("PATH", "")),
        "PYTHONPATH": prepend_path(tvm / "python", merged.get("PYTHONPATH", "")),
        "TMAC_LLVM_ROOT": str(llvm_root),
        "TVM_LIBRARY_PATH": str(tvm_build),
    }
    values.update({name: value for name, value in merged.items() if name in ["CMAKE_PREFIX_PATH", "CPATH", "LIBRARY_PATH", "PKG_CONFIG_PATH"]})
    library_parts = [str(llvm_root / "lib"), str(tvm_build)]
    if merged.get("LD_LIBRARY_PATH"):
        library_parts.append(merged["LD_LIBRARY_PATH"])
    values["LD_LIBRARY_PATH"] = os.pathsep.join(library_parts)
    if platform.system() == "Darwin":
        dyld_parts = [str(llvm_root / "lib"), str(tvm_build)]
        if merged.get("DYLD_LIBRARY_PATH"):
            dyld_parts.append(merged["DYLD_LIBRARY_PATH"])
        values["DYLD_LIBRARY_PATH"] = os.pathsep.join(dyld_parts)
    return values


def validate_tvm_build(recorder, source, python, runtime_env, tvm_build):
    if not has_tvm_compiler_library(source, tvm_build):
        raise RuntimeError("TVM build library is missing under {}".format(tvm_build))
    config = pathlib.Path(tvm_build) / "config.cmake"
    llvm_config = pathlib.Path(runtime_env["TMAC_LLVM_ROOT"]) / "bin" / "llvm-config"
    if not config.is_file() or str(llvm_config) not in config.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("TVM config.cmake does not reference the validated llvm-config: {}".format(llvm_config))
    recorder.run([str(python), "-c", "import t_mac, tvm; print('t_mac=' + t_mac.__file__); print('tvm=' + tvm.__version__); print('tvm_python=' + tvm.__file__)"], cwd=source, env=runtime_env)
    recorder.run([str(python), str(EXPERIMENT_DIR / "scripts" / "smoke_tvm.py")], cwd=source, env=runtime_env)


def tvm_config_text(llvm_config):
    settings = {
        "USE_LLVM": str(llvm_config),
        "USE_LIBBACKTRACE": "OFF",
        "BACKTRACE_ON_SEGFAULT": "OFF",
        "USE_CUDA": "OFF",
        "USE_OPENCL": "OFF",
        "USE_VULKAN": "OFF",
        "USE_CUTLASS": "OFF",
        "USE_FLASHINFER": "OFF",
        "USE_VTA_FSIM": "OFF",
        "USE_VTA_TSIM": "OFF",
        "USE_VTA_FPGA": "OFF",
    }
    return "".join("set({} {})\n".format(name, value) for name, value in settings.items())


def find_project_library(prefix, stem):
    candidates = []
    for directory in [prefix / "lib", prefix / "lib64"]:
        candidates.extend(sorted(directory.glob("lib{}.so".format(stem))))
        candidates.extend(sorted(directory.glob("lib{}.a".format(stem))))
    return candidates[0] if candidates else None


def zlib_cmake_args(deps_prefix, dependency_diagnostic):
    project_library = find_project_library(deps_prefix, "z")
    project_header = deps_prefix / "include" / "zlib.h"
    if project_library and project_header.is_file():
        return ["-DZLIB_ROOT={}".format(deps_prefix), "-DZLIB_INCLUDE_DIR={}".format(project_header.parent), "-DZLIB_LIBRARY={}".format(project_library)]
    probe = dependency_diagnostic.get("final", {}).get("zlib", {})
    include_dirs = [pathlib.Path(flag[2:]) for flag in probe.get("cflags", []) if flag.startswith("-I")]
    library_dirs = [pathlib.Path(flag[2:]) for flag in probe.get("ldflags", []) if flag.startswith("-L")]
    headers = [path / "zlib.h" for path in include_dirs if (path / "zlib.h").is_file() and (path / "zconf.h").is_file()]
    libraries = []
    for directory in library_dirs:
        libraries.extend(path for pattern in ["libz.so", "libz.a", "libz.dylib"] for path in directory.glob(pattern))
    if headers and libraries:
        return ["-DZLIB_INCLUDE_DIR={}".format(headers[0].parent), "-DZLIB_LIBRARY={}".format(libraries[0])]
    return []


def build_tvm_with_existing_llvm(recorder, source, llvm_root, deps_prefix, dependency_diagnostic, build_key):
    tvm = source / "3rdparty" / "tvm"
    build = source.parent / "build" / "tvm" / build_key
    build.mkdir(parents=True, exist_ok=True)
    config = build / "config.cmake"
    llvm_config = llvm_root / "bin" / "llvm-config"
    config.write_text(tvm_config_text(llvm_config), encoding="ascii")
    runtime_env = source_environment(source, llvm_root, deps_prefix=deps_prefix, tvm_build=build)
    command = ["cmake", "-S", str(tvm), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER={}".format(llvm_root / "bin" / "clang"), "-DCMAKE_CXX_COMPILER={}".format(llvm_root / "bin" / "clang++"), "-DCMAKE_PREFIX_PATH={}".format(deps_prefix), "-DUSE_LLVM={}".format(llvm_config), "-DUSE_LIBBACKTRACE=OFF", "-DBACKTRACE_ON_SEGFAULT=OFF", "-DUSE_CUDA=OFF", "-DUSE_OPENCL=OFF", "-DUSE_VULKAN=OFF", "-DUSE_CUTLASS=OFF", "-DUSE_FLASHINFER=OFF", "-DUSE_VTA_FSIM=OFF", "-DUSE_VTA_TSIM=OFF", "-DUSE_VTA_FPGA=OFF"]
    command.extend(zlib_cmake_args(deps_prefix, dependency_diagnostic))
    recorder.log("building pinned TVM with the already validated LLVM 17.0.6 toolchain")
    recorder.run(command, cwd=source, env=runtime_env)
    recorder.run(["cmake", "--build", str(build), "-j", "4"], cwd=source, env=runtime_env)
    return build, runtime_env


def missing_preinstall_commands(path=None):
    return [name for name in ["git", "cmake", "make"] if not shutil.which(name, path=path)]


def setup(config, recorder):
    root = pathlib.Path(config["paths"]["tmac_root"])
    source = root / "source"
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / ".setup-fingerprint"
    state_path = root / "setup-environment.json"
    if stamp.exists():
        stamp.unlink()
    if state_path.exists():
        state_path.unlink()
    recorder.log("setup stage: source verification; log={}".format(recorder.log_path))
    if not source.exists():
        run_git_network(recorder, ["git", "-c", "http.version=HTTP/1.1", "clone", "--no-checkout", config["source"]["url"], str(source)], root)
    target_commit = config["source"]["commit"]
    if git_has_commit(source, target_commit):
        recorder.log("reuse local T-MAC commit {} and skip fetch".format(target_commit))
    else:
        recorder.log("T-MAC commit {} is missing locally; fetching it".format(target_commit))
        run_git_network(recorder, ["git", "-c", "http.version=HTTP/1.1", "fetch", "origin", target_commit], source)
    if not git_has_commit(source, target_commit):
        raise RuntimeError("T-MAC repository does not contain required commit {}".format(target_commit))
    recorder.run(["git", "checkout", "--detach", target_commit], cwd=source)
    prepare_submodules(config, recorder, source)
    commits = verify_source(config, source)
    apply_patch(recorder, source)

    missing = missing_preinstall_commands()
    if sys.version_info < (3, 8) or sys.version_info >= (3, 12):
        missing.append("Python 3.8-3.11 (fixed TVM does not support Python 3.12)")
    if missing:
        raise RuntimeError("missing prerequisites: {}. Install them in a user directory and override PATH; do not replace system libraries".format(", ".join(missing)))

    recorder.log("setup stage: LLVM validation and link metadata")
    llvm_root, llvm_detail = prepare_llvm(recorder, source)
    if llvm_root is None:
        raise RuntimeError("the pinned project-local LLVM 17.0.6 is required; setup will not invoke the official download hook: {}".format("; ".join(llvm_detail) or "no candidates"))
    report = llvm_config_report(recorder, llvm_root, source / "3rdparty" / "tvm")
    recorder.log("setup stage: native dependency compile/link/run probes")
    deps_prefix, dep_key, dependency_diagnostic = prepare_native_dependencies(recorder, root, llvm_root, report)
    setup_fingerprint = fingerprint(config, commits)
    build_key = tvm_build_fingerprint(setup_fingerprint, dep_key, report, llvm_root)
    tvm_build = root / "build" / "tvm" / build_key
    runtime_env = source_environment(source, llvm_root, deps_prefix=deps_prefix, tvm_build=tvm_build)
    recorder.log("setup stage: LLVM API compile/link/run smoke")
    llvm_smoke = validate_llvm_api(recorder, root, llvm_root, report, runtime_env)

    venv = root / "venv"
    if not (venv / "bin" / "python").exists():
        recorder.run([sys.executable, "-m", "venv", str(venv)])
    python = venv / "bin" / "python"
    recorder.log("setup stage: Python dependency validation")
    python_probe = "import attrs, cloudpickle, decorator, ml_dtypes, numpy, psutil, scipy, tornado, typing_extensions"
    imports = recorder.run([str(python), "-c", python_probe], cwd=source, env=runtime_env, check=False, capture=True)
    if imports.returncode:
        recorder.run([str(python), "-m", "pip", "install", "-r", str(EXPERIMENT_DIR / "python-requirements.txt"), "-c", str(EXPERIMENT_DIR / "python-constraints.txt")], cwd=source, env=runtime_env)
    recorder.log("setup stage: T-MAC editable install with MANUAL_BUILD=1")
    recorder.run([str(python), "-m", "pip", "install", "--no-deps", "-e", ".", "-v"], cwd=source, env=dict(runtime_env, MANUAL_BUILD="1"))

    recorder.log("setup stage: fingerprinted TVM configure and build")
    tvm_build, runtime_env = build_tvm_with_existing_llvm(recorder, source, llvm_root, deps_prefix, dependency_diagnostic, build_key)
    for command in ["clang", "clang++", "llvm-config"]:
        recorder.run([command, "--version"], cwd=source, env=runtime_env)
    recorder.log("setup stage: TVM LLVM backend execution smoke")
    validate_tvm_build(recorder, source, python, runtime_env, tvm_build)
    recorder.log("setup stage: T-MAC NEON/FP16 kernel generation and execution smoke")
    tmac_smoke = root / "checks" / "tmac-kernel" / build_key
    tmac_smoke.mkdir(parents=True, exist_ok=True)
    smoke_config = tmac_smoke / "config.json"
    smoke_config.write_text(json.dumps(config["quantization"], indent=2) + "\n", encoding="utf-8")
    recorder.run([str(python), str(EXPERIMENT_DIR / "scripts" / "smoke_tmac.py"), str(tmac_smoke), str(smoke_config)], cwd=source, env=runtime_env)
    packages = recorder.run([str(python), "-m", "pip", "freeze", "--all"], cwd=source, env=runtime_env, capture=True)
    (root / "python-packages.txt").write_text(packages.stdout, encoding="utf-8")
    state = {"fingerprint": build_key, "setup_fingerprint": setup_fingerprint, "deps_fingerprint": dep_key, "deps_prefix": str(deps_prefix), "tvm_build": str(tvm_build), "llvm_root": str(llvm_root), "llvm": report, "dependency_diagnostic": dependency_diagnostic, "llvm_smoke": llvm_smoke, "tmac_smoke": str(tmac_smoke)}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    stamp.write_text(build_key + "\n", encoding="ascii")
    recorder.log("setup completed with exact source commits: {}".format(commits))


def parse_cpu_list(text):
    cpus = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(part))
    return cpus


def physical_cpus(node):
    node_file = pathlib.Path("/sys/devices/system/node/node{}".format(node)) / "cpulist"
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    candidates = parse_cpu_list(node_file.read_text()) if node_file.exists() else sorted(allowed)
    chosen = []
    seen = set()
    for cpu in candidates:
        if cpu not in allowed:
            continue
        topology = pathlib.Path("/sys/devices/system/cpu/cpu{}".format(cpu)) / "topology"
        try:
            key = ((topology / "physical_package_id").read_text().strip(), (topology / "core_id").read_text().strip())
        except OSError:
            key = ("unknown", str(cpu))
        if key not in seen:
            seen.add(key)
            chosen.append(cpu)
    return chosen


def build_baselines(config, recorder, commits):
    root = pathlib.Path(config["paths"]["tmac_root"])
    source = root / "source"
    python = root / "venv" / "bin" / "python"
    build_key = fingerprint(config, commits)[:16]
    generated = root / "generated" / "baseline" / build_key
    generated.mkdir(parents=True, exist_ok=True)
    runtime_env = source_environment(source)
    compile_command = [str(python), "compile.py", "-o", str(generated), "-d", "kunpeng", "-t", "-tb", "-gc", "-da", "-nt", "16", "-gs", "32", "-ags", "32", "-m", "llama-3-8b-4bit", "-v"]
    started = time.monotonic()
    recorder.run(compile_command, cwd=source / "deploy", env=runtime_env)
    tune_seconds = time.monotonic() - started
    for name in ["kernels.cc", "kernels.h", "kcfg.ini"]:
        shutil.copy2(generated / name, source / "deploy" / "tuned" / name)
    kcfg_text = (generated / "kcfg.ini").read_text(encoding="utf-8")
    kfactors = [int(value) for value in re.findall(r"^kfactor\s*=\s*(\d+)\s*$", kcfg_text, re.MULTILINE)]
    if not kfactors or any(value != 8 for value in kfactors):
        raise RuntimeError("generated kcfg does not exclusively use the predeclared legal kfactor=8")

    install = root / "install" / "baseline" / build_key
    tmac_build = root / "build" / "tmac-baseline" / build_key
    tmac_build.mkdir(parents=True, exist_ok=True)
    recorder.run(["cmake", "-S", str(source), "-B", str(tmac_build), "-DCMAKE_INSTALL_PREFIX={}".format(install), "-DCMAKE_BUILD_TYPE=Release"], cwd=source, env=runtime_env)
    recorder.run(["cmake", "--build", str(tmac_build), "--target", "install", "-j"], cwd=source, env=runtime_env)

    builds = {}
    harness_builds = {}
    for variant, enabled in [("B0", True), ("C0", False)]:
        build = root / "build" / "llama-{}".format(variant.lower()) / build_key
        command = ["cmake", "-S", str(source / "3rdparty" / "llama.cpp"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_OPENMP=OFF", "-DGGML_NATIVE=OFF", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"]
        if enabled:
            command.extend(["-DGGML_TMAC=ON", "-DGGML_TMAC_RECHUNK=OFF", "-DCMAKE_PREFIX_PATH={}".format(install / "lib" / "cmake" / "t-mac")])
        else:
            command.append("-DGGML_TMAC=OFF")
        recorder.run(command, cwd=source, env=runtime_env)
        recorder.run(["cmake", "--build", str(build), "--target", "llama-bench", "llama-cli", "llama-perplexity", "-j"], cwd=source, env=runtime_env)
        builds[variant] = build
        harness_build = root / "build" / "harness-{}".format(variant.lower()) / build_key
        harness_command = ["cmake", "-S", str(EXPERIMENT_DIR / "harness"), "-B", str(harness_build), "-DLLAMA_SOURCE={}".format(source / "3rdparty" / "llama.cpp"), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_OPENMP=OFF", "-DGGML_NATIVE=OFF", "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"]
        if enabled:
            harness_command.extend(["-DGGML_TMAC=ON", "-DGGML_TMAC_RECHUNK=OFF", "-DCMAKE_PREFIX_PATH={}".format(install / "lib" / "cmake" / "t-mac")])
        else:
            harness_command.append("-DGGML_TMAC=OFF")
        recorder.run(harness_command, cwd=source, env=runtime_env)
        targets = ["tmac-logits"]
        if enabled:
            targets.append("tmac-operator")
        recorder.run(["cmake", "--build", str(harness_build), "--target"] + targets + ["-j"], cwd=source, env=runtime_env)
        harness_builds[variant] = harness_build
    artifacts = {name: {"path": str(generated / name), "sha256": sha256_file(generated / name)} for name in ["kernels.cc", "kernels.h", "kcfg.ini"]}
    artifacts["operator_harness"] = {"path": str(EXPERIMENT_DIR / "harness" / "operator.cpp"), "sha256": sha256_file(EXPERIMENT_DIR / "harness" / "operator.cpp")}
    artifacts["logits_harness"] = {"path": str(EXPERIMENT_DIR / "harness" / "logits.cpp"), "sha256": sha256_file(EXPERIMENT_DIR / "harness" / "logits.cpp")}
    return builds, harness_builds, tune_seconds, artifacts


def find_binary(build, name):
    candidates = [build / "bin" / name, build / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for candidate in build.rglob(name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("{} not found under {}".format(name, build))


def run_operator_benchmarks(config, recorder, harness_build, result_dir, deadline, kcfg):
    binary = find_binary(harness_build, "tmac-operator")
    model = config["paths"]["model"]
    output = result_dir / "operator.csv"
    failures = []
    prefix = affinity_prefix(physical_cpus(config["model_benchmark"]["numa_node"])[:1], config["model_benchmark"]["numa_node"])
    for output_channels, input_channels in config["operator"]["shapes"]:
        for tokens in config["operator"]["tokens"]:
            if time.monotonic() >= deadline:
                failures.append("operator O={} K={} T={} exceeded measurement budget".format(output_channels, input_channels, tokens))
                continue
            command = prefix + [str(binary), model, str(output_channels), str(input_channels), str(tokens), str(config["operator"]["rounds"]), "2", str(config["operator"]["seed"]), str(output)]
            completed = recorder.run(command, env={"TMAC_KCFG_FILE": str(kcfg)}, check=False, capture=True)
            if completed.returncode:
                reason = (completed.stdout or "operator harness failed").strip().replace("\n", " ")[-500:]
                with open(output, "a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["B0", output_channels, input_channels, tokens, "", "", "", "", "", "", "", "", "failed", "repeated_n1", 0, 0, 0, 0, 0, config["operator"]["seed"], "synthetic_uniform_-4_4", reason])
                failures.append("operator O={} K={} T={} failed: {}".format(output_channels, input_channels, tokens, reason))
    return failures


def validate_operator_results(config, path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    valid_shapes = set()
    for row in rows:
        if row["status"] != "pass":
            continue
        try:
            error = float(row["implementation_nmse"])
        except ValueError:
            row["status"] = "failed"
            row["note"] = "invalid implementation NMSE"
            failures.append(row["note"])
            continue
        if not math.isfinite(error) or error > config["operator"]["max_nmse_lut"]:
            row["status"] = "failed_correctness"
            row["note"] = "implementation_nmse={} exceeds {}".format(error, config["operator"]["max_nmse_lut"])
            failures.append("O={} K={} T={}: {}".format(row["O"], row["K"], row["T"], row["note"]))
        else:
            valid_shapes.add((int(row["O"]), int(row["K"]), int(row["T"])))
    expected = {(o, k, t) for o, k in config["operator"]["shapes"] for t in config["operator"]["tokens"]}
    missing = expected - valid_shapes
    for shape in sorted(missing):
        if not any(row["O"] == str(shape[0]) and row["K"] == str(shape[1]) and row["T"] == str(shape[2]) for row in rows):
            failures.append("missing operator result O={} K={} T={}".format(*shape))
    write_csv(path, rows[0].keys() if rows else operator_fields(), rows)
    return failures, rows


def operator_fields():
    return ["variant", "O", "K", "T", "round", "lut_us", "compute_us", "total_us", "workspace_bytes", "implementation_max_abs", "implementation_nmse", "high_precision_max_abs", "status", "execution_mode", "lut_calls", "compute_calls", "covered_outputs", "tile_count", "sample_count", "seed", "activation_source", "note"]


def read_logits(path):
    data = pathlib.Path(path).read_bytes()
    if len(data) < 8:
        raise RuntimeError("short logits file")
    magic, count = struct.unpack_from("<II", data)
    if magic != 0x544d4c47 or len(data) != 8 + count * 4:
        raise RuntimeError("invalid logits file")
    return struct.unpack_from("<{}f".format(count), data, 8)


def run_logits_check(config, recorder, harness_builds, result_dir, kcfg):
    model = config["paths"]["model"]
    tokens = "128000,791,4062,14198,39935,35308,13"
    paths = {}
    evidence = {}
    for variant in ["B0", "C0"]:
        binary = find_binary(harness_builds[variant], "tmac-logits")
        output = result_dir / "logits-{}.bin".format(variant.lower())
        command = [str(binary), model, str(output), "1", tokens]
        completed = recorder.run(command, env={"TMAC_KCFG_FILE": str(kcfg)}, check=False, capture=True)
        evidence[variant] = {"exit_code": completed.returncode, "tmac_init": "ggml_tmac_init" in completed.stdout, "missing_kcfg": "Failed to find kcfg" in completed.stdout, "tokens": tokens}
        if completed.returncode:
            raise RuntimeError("{} logits harness failed: {}".format(variant, completed.stdout[-500:]))
        paths[variant] = output
    b0 = read_logits(paths["B0"])
    c0 = read_logits(paths["C0"])
    if len(b0) != len(c0):
        raise RuntimeError("B0/C0 vocabulary sizes differ")
    squared_error = sum((a - b) * (a - b) for a, b in zip(b0, c0))
    squared_reference = sum(b * b for b in c0)
    maximum = max(abs(a - b) for a, b in zip(b0, c0))
    for path in paths.values():
        path.unlink()
    return {"level": "model", "case": "fixed_token_logits_b0_vs_c0", "status": "pass" if all(math.isfinite(value) for value in b0) else "fail", "metric": "nmse", "value": squared_error / squared_reference if squared_reference else squared_error, "threshold": "record_only", "note": "max_abs={};tokens={}".format(maximum, tokens)}, evidence


def affinity_prefix(cpus, node):
    prefix = ["taskset", "-c", ",".join(str(cpu) for cpu in cpus)]
    if shutil.which("numactl"):
        prefix.extend(["numactl", "--membind={}".format(node)])
    return prefix


def parse_bench(output):
    rows = []
    for line in output.splitlines():
        if "|" not in line or "\u00b1" not in line:
            continue
        columns = [value.strip() for value in line.strip().strip("|").split("|")]
        match = re.search("([0-9.]+)\\s*\\u00b1\\s*([0-9.]+)", line)
        if match:
            rows.append({"raw_columns": columns, "tokens_per_second": float(match.group(1)), "stddev": float(match.group(2))})
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_model_benchmarks(config, recorder, builds, deadline, kcfg):
    model = pathlib.Path(config["paths"]["model"])
    c1_build = pathlib.Path(config["paths"]["c1_build"])
    if not model.is_file():
        raise FileNotFoundError("model not found: {}".format(model))
    variants = {"B0": find_binary(builds["B0"], "llama-bench"), "C0": find_binary(builds["C0"], "llama-bench")}
    variants["C1"] = find_binary(c1_build, "llama-bench")
    node = config["model_benchmark"]["numa_node"]
    available = physical_cpus(node)
    rows = []
    dispatch = {"policy": "B0 must print ggml_tmac_init and must not report missing kcfg", "runs": []}
    for threads in config["model_benchmark"]["threads"]:
        if len(available) < threads:
            recorder.log("skip {} threads: only {} physical cores are available".format(threads, len(available)))
            continue
        prefix = affinity_prefix(available[:threads], node)
        for round_index in range(config["model_benchmark"]["rounds"]):
            for variant in ["B0", "C0", "C1"]:
                if time.monotonic() >= deadline:
                    return rows, dispatch, True
                binary = variants[variant]
                common = [str(binary), "-m", str(model), "-t", str(threads), "-ngl", "0", "-r", "1"]
                if variant == "C1":
                    common.extend(["--q4-kernel", "4x4"])
                for phase, extra in [("prefill", ["-p", "128", "-n", "0", "-b", "128", "-ub", "128"]), ("decode", ["-p", "512", "-n", "128", "-b", "128", "-ub", "128"])]:
                    env = {"TMAC_KCFG_FILE": str(kcfg)}
                    completed = recorder.run(prefix + common + extra, env=env, check=False, capture=True)
                    parsed = parse_bench(completed.stdout)
                    rows.append({"variant": variant, "phase": phase, "threads": threads, "round": round_index + 1, "exit_code": completed.returncode, "tokens_per_second": parsed[-1]["tokens_per_second"] if parsed else "", "stddev": parsed[-1]["stddev"] if parsed else "", "command": shlex.join(prefix + common + extra)})
                    if variant == "B0":
                        dispatch["runs"].append({"threads": threads, "phase": phase, "round": round_index + 1, "tmac_init": "ggml_tmac_init" in completed.stdout, "missing_kcfg": "Failed to find kcfg" in completed.stdout, "exit_code": completed.returncode})
    return rows, dispatch, False


def environment_text(config, cpus, source=None):
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    lines = ["platform=" + platform.platform(), "python=" + sys.version.replace("\n", " "), "allowed_cpus=" + ",".join(map(str, allowed)), "selected_physical_cpus=" + ",".join(map(str, cpus))]
    runtime_env = None
    if source is not None:
        try:
            runtime_env = os.environ.copy()
            runtime_env.update(source_environment(source))
        except RuntimeError as error:
            lines.append("runtime_environment_error=" + str(error))
    for command in ["cc", "c++", "gcc", "g++", "clang", "clang++", "cmake", "ldd", "numactl", "taskset"]:
        lines.append("{}={}".format(command, json.dumps(command_version(command, runtime_env), ensure_ascii=True)))
    for path in ["/proc/cpuinfo", "/proc/meminfo", "/proc/self/status"]:
        try:
            lines.append("\n===== {} =====\n{}".format(path, pathlib.Path(path).read_text(errors="replace")))
        except OSError:
            pass
    return "\n".join(lines) + "\n"


def write_summary(path, state, rows, dispatch, blocking):
    completed = [name for name, value in state.items() if value]
    failed_dispatch = [row for row in dispatch.get("runs", []) if not row["tmac_init"] or row["missing_kcfg"] or row["exit_code"]]
    lines = ["# T-MAC 鲲鹏 baseline 摘要", "", "## 已完成", "", ", ".join(completed) if completed else "无。", "", "## 实际路径", "", "B0 使用固定 T-MAC fork 的 Q4_0 加载时重排和 N=1 生成内核。T>1 时配套 llama.cpp 逐 token 调用 N=1 内核，不属于原生多 token GEMM。C0 是同 commit 关闭 T-MAC；C1 是既有构建的 4x4 路径，属于跨版本参照。", "", "## 结果", "", "模型逐轮记录数：{}。中位数应从 end_to_end.csv 的有效逐轮数据计算；本文件不填充缺失数据。".format(len(rows)), "", "## 正确性与回退", "", "独立标量参考只验证 Q4_0 nibble 顺序、LUT 索引、bit-plane 系数、scale 和分组边界。服务器生成内核及模型级数值验证必须有实际服务器结果才算完成。", "", "B0 dispatch 异常记录数：{}。".format(len(failed_dispatch)), "", "## 限制和阻塞", ""]
    lines.extend(["- " + item for item in blocking] or ["- 无已知阻塞项。"])
    lines.extend(["", "当前结果只能支持已实际完成且有原始记录的项目，不能把本地测试或缺失测量解释为鲲鹏性能结论。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_results(result_dir):
    output = result_dir / "send_to_chatgpt.tar.gz"
    names = ["manifest.json", "environment.txt", "config.json", "correctness.csv", "operator.csv", "end_to_end.csv", "dispatch.json", "summary.md", "console.log"]
    with tarfile.open(output, "w:gz") as archive:
        for name in names:
            path = result_dir / name
            if path.exists():
                archive.add(path, arcname=name)


def baseline(config, recorder, result_dir):
    root = pathlib.Path(config["paths"]["tmac_root"])
    source = root / "source"
    commits = {"tmac": None, "submodules": {}}
    state = {"源码固定": False, "本地标量参考": False, "内核生成与调优": False, "B0/C0 构建": False, "真实内核算子测量": False, "模型 logits 对照": False, "模型测量": False}
    blocking = []
    correctness = []
    operator_rows = []
    model_rows = []
    dispatch = {"runs": []}
    tune_seconds = None
    generated_artifacts = {}
    timed_out = False
    started = time.monotonic()
    succeeded = True
    run_started_at = datetime.datetime.now().astimezone().isoformat()
    write_csv(result_dir / "operator.csv", operator_fields(), [])
    try:
        commits = verify_source(config, source)
        validate_setup_state(config, commits, source)
        state["源码固定"] = True
        apply_patch(recorder, source)
        reference_result = self_test(config["operator"]["seed"])
        state["本地标量参考"] = True
        correctness.append({"level": "scalar", "case": "q4_pack_lut_bitplane_scale", "status": "pass", "metric": "exact", "value": 0.0, "threshold": 0.0, "note": json.dumps(reference_result, ensure_ascii=True)})
        builds, harness_builds, tune_seconds, generated_artifacts = build_baselines(config, recorder, commits)
        state["内核生成与调优"] = True
        state["B0/C0 构建"] = True
        deadline = time.monotonic() + config["model_benchmark"]["measurement_budget_seconds"]
        kcfg = generated_artifacts["kcfg.ini"]["path"]
        blocking.extend(run_operator_benchmarks(config, recorder, harness_builds["B0"], result_dir, deadline, kcfg))
        validation_failures, operator_rows = validate_operator_results(config, result_dir / "operator.csv")
        blocking.extend(validation_failures)
        state["真实内核算子测量"] = not validation_failures and all(row["status"] == "pass" for row in operator_rows)
        seen_correctness = set()
        for row in operator_rows:
            key = (row["O"], row["K"], row["T"])
            if row["status"] != "pass" or key in seen_correctness:
                continue
            seen_correctness.add(key)
            note = "sample_count={};seed={};execution_mode={};weight={}".format(row["sample_count"], row["seed"], row["execution_mode"], row["note"])
            correctness.append({"level": "kernel", "case": "O{}_K{}_T{}_same_lut".format(*key), "status": "pass", "metric": "nmse", "value": row["implementation_nmse"], "threshold": config["operator"]["max_nmse_lut"], "note": note})
            correctness.append({"level": "kernel", "case": "O{}_K{}_T{}_high_precision".format(*key), "status": "measured", "metric": "max_abs", "value": row["high_precision_max_abs"], "threshold": "record_only", "note": note})
        logits_row, logits_dispatch = run_logits_check(config, recorder, harness_builds, result_dir, kcfg)
        correctness.append(logits_row)
        dispatch["logits"] = logits_dispatch
        if logits_row["status"] != "pass" or not logits_dispatch["B0"]["tmac_init"] or logits_dispatch["B0"]["missing_kcfg"]:
            blocking.append("model logits check failed or did not prove B0 T-MAC dispatch")
        else:
            state["模型 logits 对照"] = True
        model_rows, dispatch, timed_out = run_model_benchmarks(config, recorder, builds, deadline, kcfg)
        dispatch["logits"] = logits_dispatch
        state["模型测量"] = bool(model_rows) and not timed_out
        if not model_rows:
            blocking.append("no end-to-end measurements completed")
        if timed_out:
            blocking.append("正式测量达到预算，已保留完成项并标记未完成。")
        if any(not row["tmac_init"] or row["missing_kcfg"] for row in dispatch["runs"]):
            blocking.append("B0 dispatch 证据不完整或出现缺失 kcfg，相关性能数据无效。")
        recorder.log("T>1 operator measurements use repeated_n1 exactly as the pinned llama.cpp integration")
    except Exception as error:
        succeeded = False
        blocking.append("{}: {}".format(type(error).__name__, error))
        recorder.log(traceback.format_exc())
    finally:
        write_csv(result_dir / "correctness.csv", ["level", "case", "status", "metric", "value", "threshold", "note"], correctness)
        if not (result_dir / "operator.csv").exists():
            write_csv(result_dir / "operator.csv", operator_fields(), operator_rows)
        write_csv(result_dir / "end_to_end.csv", ["variant", "phase", "threads", "round", "exit_code", "tokens_per_second", "stddev", "command"], model_rows)
        (result_dir / "dispatch.json").write_text(json.dumps(dispatch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        cpus = physical_cpus(config["model_benchmark"]["numa_node"])
        (result_dir / "environment.txt").write_text(environment_text(config, cpus, source), encoding="utf-8")
        patch_path = EXPERIMENT_DIR / "patches" / "0001-kunpeng-device.patch"
        manifest = {"status": "partial" if blocking else "complete", "started_at": run_started_at, "elapsed_seconds": time.monotonic() - started, "tuning_seconds": tune_seconds, "timed_out": timed_out, "source": commits, "adaptation": {"reason": "Add a named Linux AArch64 ARMv8.2+FP16 compile target; no algorithm code changes", "patch_sha256": sha256_file(patch_path), "diff": patch_path.read_text(encoding="utf-8")}, "generated_artifacts": generated_artifacts, "model_sha256": sha256_file(config["paths"]["model"]) if pathlib.Path(config["paths"]["model"]).is_file() else None, "dimension_mapping": {"T": "T-MAC N; fixed fork invokes generated N=1 kernel once per token", "O": "logical output channels; compile.py passes O*bits as QGeMMLUTBitsCodegen M", "K": "T-MAC K"}, "repeated_n1_source": {"file": "3rdparty/llama.cpp/ggml/src/ggml.c", "init_loop_line": 12586, "compute_loop_lines": [12613, 12680], "commit": commits.get("submodules", {}).get("3rdparty/llama.cpp")}, "commands": recorder.commands, "blocking": blocking}
        (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary(result_dir / "summary.md", state, model_rows, dispatch, blocking)
        package_results(result_dir)
    return succeeded and not blocking


def local_test(config, recorder):
    result = self_test(config["operator"]["seed"])
    recorder.run([sys.executable, "-m", "unittest", "discover", "-s", str(EXPERIMENT_DIR / "tests"), "-v"])
    if config["quantization"] != {"bits": 4, "weight_group_size": 32, "act_group_size": 32, "lut_group": 4, "fast_aggregation": False, "lut_dtype": "int8", "output_dtype": "float16", "aggregation_dtype": "int32", "zero_point": False}:
        raise AssertionError("baseline quantization configuration changed")
    if config["quantization"]["weight_group_size"] % (8 * config["quantization"]["lut_group"]):
        raise AssertionError("expected legal kfactor=8")
    recorder.log("local tests passed: {}".format(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup", "baseline", "local-test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if args.command == "baseline":
        stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        result_dir = pathlib.Path(config["paths"]["results_root"]) / "tmac_baseline_{}".format(stamp)
    elif args.command == "setup":
        result_dir = pathlib.Path(config["paths"]["tmac_root"]) / "logs" / args.command
    else:
        result_dir = EXPERIMENT_DIR / ".local-results" / args.command
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    recorder = Recorder(result_dir)
    try:
        if args.command == "setup":
            setup(config, recorder)
        elif args.command == "baseline":
            if not baseline(config, recorder, result_dir):
                raise SystemExit(1)
        else:
            local_test(config, recorder)
    except BaseException:
        recorder.log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
