import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from experiment import Recorder, find_llvm_root, git_has_commit, missing_preinstall_commands, prepare_submodules, run_git_network, source_environment


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
