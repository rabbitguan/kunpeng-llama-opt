import json
import os
import pathlib
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from experiment import Recorder, build_tvm_with_existing_llvm, find_llvm_root, git_has_commit, missing_preinstall_commands, prepare_llvm, prepare_submodules, probe_dependency, run_git_network, setup, source_environment, tvm_build_fingerprint, tvm_config_text, validate_llvm_root, vendor_manifest


class RecordingRunner:
    def __init__(self):
        self.commands = []
        self.messages = []

    def run(self, command, cwd=None, **kwargs):
        self.commands.append((list(command), pathlib.Path(cwd)))
        return subprocess.CompletedProcess(command, 0, "", None)

    def log(self, message):
        self.messages.append(message)


class SetupTest(unittest.TestCase):
    def make_tool(self, path, output):
        path.write_text("#!/bin/sh\nprintf '%s\\n' '" + output + "'\n", encoding="ascii")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def test_discovers_pinned_project_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory)
            binary = source / "build" / "clang+llvm-17.0.6-aarch64-linux-gnu" / "bin"
            binary.mkdir(parents=True)
            self.make_tool(binary / "clang", "clang version 17.0.6")
            self.make_tool(binary / "clang++", "clang version 17.0.6")
            self.make_tool(binary / "llvm-config", "17.0.6")
            root, versions = find_llvm_root(source)
            self.assertEqual(root, binary.parent.resolve())
            self.assertEqual(versions["llvm-config"], "17.0.6")

    def test_llvm_config_libtinfo_stderr_warning_is_not_a_version_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory)
            binary = source / "build" / "clang+llvm-17.0.6-aarch64-linux-gnu" / "bin"
            binary.mkdir(parents=True)
            self.make_tool(binary / "clang", "clang version 17.0.6")
            self.make_tool(binary / "clang++", "clang version 17.0.6")
            warning = "/usr/lib64/libtinfo.so.6: no version information available"
            (binary / "llvm-config").write_text("#!/bin/sh\nprintf '%s\\n' '" + warning + "' >&2\nprintf '%s\\n' '17.0.6'\n", encoding="ascii")
            (binary / "llvm-config").chmod((binary / "llvm-config").stat().st_mode | stat.S_IXUSR)
            recorder = RecordingRunner()
            root, versions = prepare_llvm(recorder, source)
            self.assertEqual(root, binary.parent.resolve())
            self.assertEqual(versions["llvm-config"], "17.0.6")
            self.assertEqual(versions["stderr"]["llvm-config"], warning)
            self.assertIn("llvm-config --version stderr warning: " + warning, recorder.messages)

    def test_llvm_config_dynamic_loader_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "bin"
            binary.mkdir(parents=True)
            self.make_tool(binary / "clang", "clang version 17.0.6")
            self.make_tool(binary / "clang++", "clang version 17.0.6")
            (binary / "llvm-config").write_text("#!/bin/sh\nprintf '%s\\n' 'error while loading shared libraries: libLLVM.so' >&2\nexit 127\n", encoding="ascii")
            (binary / "llvm-config").chmod((binary / "llvm-config").stat().st_mode | stat.S_IXUSR)
            valid, detail = validate_llvm_root(root)
            self.assertFalse(valid)
            self.assertIn("exit code 127", detail)
            self.assertIn("error while loading shared libraries", detail)

    def test_incomplete_toolchain_reports_exact_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "bin"
            binary.mkdir(parents=True)
            self.make_tool(binary / "clang", "clang version 17.0.6")
            valid, detail = validate_llvm_root(root)
            self.assertFalse(valid)
            self.assertIn(str(binary / "clang++"), detail)
            self.assertIn(str(binary / "llvm-config"), detail)

    def test_recovers_incomplete_toolchain_from_local_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source"
            build = source / "build"
            name = "clang+llvm-17.0.6-aarch64-linux-gnu"
            partial = build / name
            partial.mkdir(parents=True)
            (partial / "partial.txt").write_text("keep\n", encoding="ascii")
            payload = pathlib.Path(directory) / "payload" / name / "bin"
            payload.mkdir(parents=True)
            self.make_tool(payload / "clang", "clang version 17.0.6")
            self.make_tool(payload / "clang++", "clang version 17.0.6")
            self.make_tool(payload / "llvm-config", "17.0.6")
            archive = build / (name + ".tar.xz")
            with tarfile.open(archive, "w:xz") as handle:
                handle.add(payload.parent, arcname=name)
            recorder = RecordingRunner()
            with mock.patch("experiment.platform.system", return_value="Linux"), mock.patch("experiment.platform.machine", return_value="aarch64"):
                root, versions = prepare_llvm(recorder, source)
            self.assertEqual(root, partial.resolve())
            self.assertEqual(versions["llvm-config"], "17.0.6")
            self.assertTrue((partial / "partial.txt").is_file())
            self.assertIn("recovered validated LLVM 17.0.6", "\n".join(recorder.messages))

    def test_incomplete_toolchain_without_archive_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source"
            binary = source / "build" / "clang+llvm-17.0.6-aarch64-linux-gnu" / "bin"
            binary.mkdir(parents=True)
            self.make_tool(binary / "clang", "clang version 17.0.6")
            recorder = RecordingRunner()
            with mock.patch("experiment.platform.system", return_value="Linux"), mock.patch("experiment.platform.machine", return_value="aarch64"):
                with self.assertRaisesRegex(RuntimeError, "will not be downloaded again automatically") as raised:
                    prepare_llvm(recorder, source)
            self.assertIn(str(binary / "clang++"), str(raised.exception))
            self.assertEqual(recorder.commands, [])

    def test_tvm_build_reuses_llvm_and_existing_build_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source"
            (source / "3rdparty" / "tvm").mkdir(parents=True)
            build = source.parent / "build" / "tvm" / "test-key"
            build.mkdir(parents=True)
            marker = build / "resume-marker"
            marker.write_text("keep\n", encoding="ascii")
            llvm_root = source / "build" / "clang+llvm-17.0.6-aarch64-linux-gnu"
            deps_prefix = source.parent / "deps" / "test" / "prefix"
            recorder = RecordingRunner()
            built, _ = build_tvm_with_existing_llvm(recorder, source, llvm_root, deps_prefix, {"final": {}}, "test-key")
            self.assertEqual(built, build)
            self.assertTrue(marker.is_file())
            self.assertEqual(recorder.commands[0][0][:5], ["cmake", "-S", str(source / "3rdparty" / "tvm"), "-B", str(build)])
            self.assertEqual(recorder.commands[1][0], ["cmake", "--build", str(build), "-j", "4"])
            self.assertIn(str(llvm_root / "bin" / "llvm-config"), (build / "config.cmake").read_text(encoding="utf-8"))

    def test_skipped_libbacktrace_is_disabled_in_tvm_config(self):
        content = tvm_config_text(pathlib.Path("/llvm/bin/llvm-config"))
        self.assertIn("set(USE_LIBBACKTRACE OFF)", content)
        self.assertIn("set(BACKTRACE_ON_SEGFAULT OFF)", content)
        self.assertIn("set(USE_CUDA OFF)", content)
        self.assertIn("set(USE_FLASHINFER OFF)", content)

    def test_dependency_fingerprint_changes_with_toolchain_or_dependencies(self):
        report = {"system-libs": {"stdout": "-lz", "stderr": "", "returncode": 0}}
        first = tvm_build_fingerprint("setup-a", "deps-a", report, pathlib.Path("/llvm-a"))
        self.assertNotEqual(first, tvm_build_fingerprint("setup-a", "deps-b", report, pathlib.Path("/llvm-a")))
        self.assertNotEqual(first, tvm_build_fingerprint("setup-a", "deps-a", report, pathlib.Path("/llvm-b")))

    def test_pkg_config_success_does_not_hide_missing_zlib_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            compiler = root / "cc"
            compiler.write_text("#!/bin/sh\nprintf '%s\\n' 'fatal error: zlib.h: No such file or directory' >&2\nexit 1\n", encoding="ascii")
            compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
            recorder = Recorder(root)
            with mock.patch("experiment.pkg_config_flags", return_value=["-I/fake/pkgconfig/include"]):
                result = probe_dependency(recorder, "zlib", compiler, os.environ.copy(), root)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "compile_link")
            self.assertIn("zlib.h", result["compile_output"])

    def test_zlib_headers_do_not_hide_link_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            include = root / "include"
            include.mkdir()
            (include / "zlib.h").write_text("/* test */\n", encoding="ascii")
            (include / "zconf.h").write_text("/* test */\n", encoding="ascii")
            compiler = root / "cc"
            compiler.write_text("#!/bin/sh\nprintf '%s\\n' 'ld: cannot find -lz' >&2\nexit 1\n", encoding="ascii")
            compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
            recorder = Recorder(root)
            with mock.patch("experiment.pkg_config_flags", side_effect=lambda package, env, option="--cflags": ["-I" + str(include)] if option == "--cflags" else ["-lz"]):
                result = probe_dependency(recorder, "zlib", compiler, os.environ.copy(), root)
            self.assertEqual(result["status"], "failed")
            self.assertIn("cannot find -lz", result["compile_output"])

    def test_vendor_archives_have_expected_hashes(self):
        manifest = vendor_manifest()
        self.assertEqual(set(manifest["packages"]), {"zlib", "zstd", "xml2", "tinfo"})

    def test_setup_does_not_call_official_build_hook(self):
        import inspect
        text = inspect.getsource(setup)
        self.assertNotIn("setup.py", text)
        self.assertNotIn('MANUAL_BUILD": "0"', text)

    def test_setup_failure_removes_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "tmac"
            (root / "source").mkdir(parents=True)
            marker = root / ".setup-fingerprint"
            marker.write_text("old-success\n", encoding="ascii")
            config = {"paths": {"tmac_root": str(root)}, "source": {"commit": "0" * 40, "url": "unused"}}
            recorder = Recorder(root)
            with mock.patch("experiment.git_has_commit", side_effect=RuntimeError("source check failed")):
                with self.assertRaisesRegex(RuntimeError, "source check failed"):
                    setup(config, recorder)
            self.assertFalse(marker.exists())

    def test_preinstall_check_does_not_require_clang(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = missing_preinstall_commands(directory)
            self.assertEqual(missing, ["git", "cmake", "make"])
            self.assertNotIn("clang", missing)
            self.assertNotIn("clang++", missing)

    def test_environment_propagates_toolchain_and_tvm(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory)
            root = source / "llvm"
            (root / "bin").mkdir(parents=True)
            env = source_environment(source, root, {"PATH": "/system/bin", "PYTHONPATH": "/existing/python", "LD_LIBRARY_PATH": "/existing/lib"})
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(root / "bin"))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(source / "3rdparty" / "tvm" / "python"))
            self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep)[:2], [str(root / "lib"), str(source / "3rdparty" / "tvm" / "build")])
            self.assertEqual(env["TMAC_LLVM_ROOT"], str(root))

    def test_recorder_captures_combined_output(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(pathlib.Path(directory))
            completed = recorder.run([sys.executable, "-c", "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)"], capture=True)
            self.assertEqual(completed.returncode, 0)
            self.assertIn("stdout-line", completed.stdout)
            self.assertIn("stderr-line", completed.stdout)
            log = (pathlib.Path(directory) / "console.log").read_text(encoding="utf-8")
            self.assertIn("stdout-line", log)
            self.assertIn("stderr-line", log)

    def test_local_git_commit_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory) / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            (repository / "file.txt").write_text("data\n", encoding="ascii")
            subprocess.run(["git", "add", "file.txt"], cwd=repository, check=True)
            env = os.environ.copy()
            env.update({"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com", "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com"})
            subprocess.run(["git", "commit", "-q", "-m", "test"], cwd=repository, env=env, check=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
            self.assertTrue(git_has_commit(repository, commit))
            self.assertFalse(git_has_commit(repository, "0" * 40))

    def test_git_network_retry_preserves_error_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            counter = root / "counter"
            program = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); print('attempt-error-' + str(n), file=sys.stderr); raise SystemExit(0 if n == 3 else 128)"
            recorder = Recorder(root)
            run_git_network(recorder, [sys.executable, "-c", program, str(counter)], root, attempts=3, timeout=5)
            self.assertEqual(counter.read_text(), "3")
            log = (root / "console.log").read_text(encoding="utf-8")
            self.assertIn("attempt-error-1", log)
            self.assertIn("attempt-error-3", log)

    def test_recorder_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(pathlib.Path(directory))
            with self.assertRaises(subprocess.TimeoutExpired):
                recorder.run([sys.executable, "-c", "import time; print('before-timeout', flush=True); time.sleep(5)"], timeout=0.1, capture=True)
            log = (pathlib.Path(directory) / "console.log").read_text(encoding="utf-8")
            self.assertIn("before-timeout", log)

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_recorder_timeout_terminates_child_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            marker = root / "child-terminated"
            child = "import pathlib,signal,sys,time; p=pathlib.Path(sys.argv[1]); signal.signal(signal.SIGTERM, lambda *_: (p.write_text('terminated'), sys.exit(0))); time.sleep(30)"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); time.sleep(30)"
            recorder = Recorder(root)
            with self.assertRaises(subprocess.TimeoutExpired):
                recorder.run([sys.executable, "-c", parent, child, str(marker)], timeout=0.5, capture=True, heartbeat=0.1)
            for _ in range(20):
                if marker.exists():
                    break
                __import__("time").sleep(0.05)
            self.assertEqual(marker.read_text(), "terminated")

    def test_recorder_interrupt_uses_process_group_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            process = mock.MagicMock()
            process.pid = 123
            process.stdout = []
            process.poll.return_value = None
            process.wait.side_effect = KeyboardInterrupt()
            recorder = Recorder(pathlib.Path(directory))
            with mock.patch("experiment.subprocess.Popen", return_value=process), mock.patch.object(recorder, "terminate") as terminate:
                with self.assertRaises(KeyboardInterrupt):
                    recorder.run(["fake-command"])
            terminate.assert_called_once_with(process)

    def test_only_required_submodules_are_prepared_without_recursion(self):
        config_path = pathlib.Path(__file__).resolve().parents[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        recorder = RecordingRunner()
        source = pathlib.Path("/source")
        with mock.patch("experiment.git_has_commit", return_value=True):
            prepare_submodules(config, recorder, source)
        rendered = [" ".join(command) for command, _ in recorder.commands]
        self.assertFalse(any("--recursive" in command for command in rendered))
        self.assertFalse(any("executorch" in command for command in rendered))
        self.assertFalse(any("FACTO" in command for command in rendered))
        self.assertFalse(any("Vulkan" in command for command in rendered))
        expected = {"3rdparty/llama.cpp", "3rdparty/tvm", "3rdparty/dlpack", "3rdparty/dmlc-core", "3rdparty/rang"}
        synced = {command[-1] for command, _ in recorder.commands if command[:3] == ["git", "submodule", "sync"]}
        self.assertEqual(synced, expected)
        messages = "\n".join(recorder.messages)
        self.assertIn("skip unused submodule t-man/executorch", messages)
        self.assertIn("skip unused TVM submodule 3rdparty/cutlass", messages)
        self.assertIn("skip unused TVM submodule 3rdparty/flashinfer", messages)

    def test_official_build_hook_patch_cannot_download_submodules(self):
        patch_path = pathlib.Path(__file__).resolve().parents[1] / "patches" / "0002-selective-submodules.patch"
        content = patch_path.read_text(encoding="utf-8")
        added = "\n".join(line[1:] for line in content.splitlines() if line.startswith("+") and not line.startswith("+++"))
        self.assertNotIn("submodule update", added)
        self.assertNotIn("--recursive", added)
        self.assertNotIn("executorch", added)
        self.assertNotIn("FACTO", added)
        self.assertNotIn("Vulkan", added)


if __name__ == "__main__":
    unittest.main()
