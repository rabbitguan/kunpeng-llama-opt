import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from experiment import Recorder, find_llvm_root, git_has_commit, missing_preinstall_commands, run_git_network, source_environment


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


if __name__ == "__main__":
    unittest.main()
