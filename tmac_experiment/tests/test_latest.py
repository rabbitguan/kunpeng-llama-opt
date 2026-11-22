import array
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tmac_latest", ROOT / "tools/kunpeng_tmac_latest.py")
latest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latest)


class LatestTest(unittest.TestCase):
    def test_host_change_requires_explicit_option(self):
        profile = {"hostname": "cnkunpengb01", "node": 0, "cpus": list(range(0, 64, 2))}
        with mock.patch.object(latest.platform, "node", return_value="cnkunpengb08"):
            with self.assertRaisesRegex(ValueError, "allow-host-change"):
                latest.execution_profile(profile, False)

    def test_host_change_selects_allowed_physical_cores_on_saved_node(self):
        profile = {"hostname": "cnkunpengb01", "node": 0, "cpus": list(range(0, 64, 2))}
        topology = "# CPU,Core,Socket,Node\n" + "\n".join(
            f"{cpu},{cpu // 2},{cpu // 80},{cpu // 80}" for cpu in range(160))
        with mock.patch.object(latest.platform, "node", return_value="cnkunpengb08"), \
                mock.patch.object(latest.os, "sched_getaffinity", return_value=set(range(1, 160)), create=True), \
                mock.patch.object(latest.subprocess, "check_output", return_value=topology):
            actual = latest.execution_profile(profile, True)
        self.assertEqual(actual["hostname"], "cnkunpengb08")
        self.assertEqual(actual["cpus"], [1] + list(range(2, 64, 2)))
        self.assertEqual(profile["hostname"], "cnkunpengb01")
        self.assertEqual(profile["cpus"], list(range(0, 64, 2)))

    def test_host_change_rejects_insufficient_physical_cores(self):
        profile = {"hostname": "cnkunpengb01", "node": 0, "cpus": list(range(0, 64, 2))}
        topology = "\n".join(f"{cpu},{cpu // 2},0,0" for cpu in range(32))
        with mock.patch.object(latest.platform, "node", return_value="cnkunpengb08"), \
                mock.patch.object(latest.os, "sched_getaffinity", return_value=set(range(32)), create=True), \
                mock.patch.object(latest.subprocess, "check_output", return_value=topology):
            with self.assertRaisesRegex(ValueError, "16 allowed physical cores"):
                latest.execution_profile(profile, True)

    def test_same_host_preserves_saved_binding(self):
        profile = {"hostname": "cnkunpengb01", "node": 0, "cpus": list(range(0, 64, 2))}
        with mock.patch.object(latest.platform, "node", return_value="cnkunpengb01"), \
                mock.patch.object(latest.os, "sched_getaffinity", return_value=set(range(64)), create=True), \
                mock.patch.object(latest.subprocess, "check_output") as probe:
            actual = latest.execution_profile(profile, False)
        self.assertEqual(actual, profile)
        probe.assert_not_called()

    def test_rejects_old_commit_and_changed_protocol(self):
        frozen = {"protocol": 1, "mode": "baseline", "commit": latest.PIN,
                  "q4_kernel_requested": "auto", "upstream_selector_modified": False,
                  "profile": {"threads": [1, 16, 32], "cases": {key: list(value) for key, value in latest.CASES.items()}}}
        self.assertEqual(latest.validate_baseline(frozen), frozen["profile"])
        for key, value in (("commit", "old"), ("upstream_selector_modified", True), ("q4_kernel_requested", "4x4")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                latest.validate_baseline(dict(frozen, **{key: value}))
        self.assertEqual(latest.validate_baseline(dict(frozen, q4_kernel_requested="4x4", upstream_selector_modified=True)), frozen["profile"])

    def test_dispatch_requires_every_packed_tensor_and_requested_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dispatch.log"
            text = "tmac: packed tensor=a\ntmac: packed tensor=b\n"
            path.write_text(text + "tmac: compute tensor=a K=4096 threads=1 kernel_N=1\n")
            with self.assertRaises(ValueError):
                latest.dispatch(path)
            path.write_text(path.read_text() + "tmac: compute tensor=b K=4096 threads=1 kernel_N=1\n")
            self.assertEqual(latest.dispatch(path, 1), ["a", "b"])
            with self.assertRaises(ValueError):
                latest.dispatch(path, 32)

    def test_logits_gates_each_stage_without_pooled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, got = (Path(directory) / name for name in ("ref", "got"))
            values = array.array("f", [1.0] * 90)
            header = struct.pack("=II", 0x544d4c47, len(values))
            ref.write_bytes(header + values.tobytes())
            got.write_bytes(ref.read_bytes())
            self.assertTrue(all(row["status"] == "pass" for row in latest.compare_logits(ref, got)))
            values[-1] = 2.1
            got.write_bytes(header + values.tobytes())
            records = latest.compare_logits(ref, got)
            self.assertTrue(all(row["status"] == "pass" for row in records[:-1]))
            self.assertEqual(records[-1]["status"], "failed")
            values[-1] = float("nan")
            got.write_bytes(header + values.tobytes())
            with self.assertRaises(ValueError):
                latest.compare_logits(ref, got)
            got.write_bytes(ref.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                latest.compare_logits(ref, got)

    def test_patch_contains_the_current_backend_sources(self):
        patch = latest.PATCH.read_text()
        for path in ("ggml/src/ggml-cpu/tmac/tmac.h", "ggml/src/ggml-cpu/tmac/tmac.cpp"):
            start = patch.index("+++ b/" + path + "\n")
            section = patch[start:].split("--- ", 1)[0]
            additions = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++")) + "\n"
            self.assertEqual(additions, (ROOT / path).read_text())


if __name__ == "__main__":
    unittest.main()
