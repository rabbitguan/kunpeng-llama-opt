import array
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import shutil
import subprocess
import platform
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tmac_latest", ROOT / "tools/kunpeng_tmac_latest.py")
assert SPEC is not None and SPEC.loader is not None
latest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latest)
with mock.patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
    DIAG_SPEC = importlib.util.spec_from_file_location("tmac_diagnose", ROOT / "tools/kunpeng_tmac_diagnose.py")
    assert DIAG_SPEC is not None and DIAG_SPEC.loader is not None
    diagnose = importlib.util.module_from_spec(DIAG_SPEC)
    DIAG_SPEC.loader.exec_module(diagnose)


class LatestTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TMAC_LATEST_CHECK") and os.environ.get("TMAC_BLOCKED_CHECK"), "set both compiled ARM64 probes")
    def test_blocked_operator_outputs_match_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for threads in (1, 16, 32):
                outputs = []
                for name in ("TMAC_LATEST_CHECK", "TMAC_BLOCKED_CHECK"):
                    path = root / f"{name}-{threads}.bin"
                    result = subprocess.run([os.environ[name], "--bench-tmac", str(threads), str(path)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outputs.append(path.read_bytes())
                self.assertEqual(len(outputs[0]), 4096 * 40 * 4)
                self.assertEqual(outputs[0], outputs[1])

    def test_blocked_schedule_keeps_original_tile_body(self):
        original = (ROOT / "ggml/src/ggml-cpu/tmac/tmac.cpp").read_text()
        for block in (1, 4, 8, 16, 32):
            modified = diagnose.blocked_backend_source(original, block)
            self.assertIn(f"first_row += {block}", modified)
            self.assertEqual(modified.count("qgemm_lut_int8("), original.count("qgemm_lut_int8("))
            self.assertEqual(modified.count("preprocessor_int8("), original.count("preprocessor_int8("))
        for block in (0, 3, 64):
            with self.assertRaises(ValueError):
                diagnose.blocked_backend_source(original, block)
        with self.assertRaises(ValueError):
            diagnose.blocked_backend_source(original.replace("task += params->nth", "task++"), 8)

    def test_compiler_override_updates_both_compilers_and_checks_availability(self):
        profile = {"toolchain": {"CMAKE_C_COMPILER": "gcc", "CMAKE_CXX_COMPILER": "g++"}}
        with mock.patch.object(diagnose, "probe", return_value={"exit_code": 0, "output": "clang version 17.0.6"}) as probe:
            diagnose.select_compilers(profile, Path("/llvm"))
        self.assertEqual(profile["toolchain"]["CMAKE_C_COMPILER"], "/llvm/bin/clang")
        self.assertEqual(profile["toolchain"]["CMAKE_CXX_COMPILER"], "/llvm/bin/clang++")
        self.assertEqual(probe.call_count, 2)
        with mock.patch.object(diagnose, "probe", return_value={"error": "not found"}):
            with self.assertRaisesRegex(ValueError, "Cannot execute"):
                diagnose.select_compilers(profile, Path("/missing"))

    @unittest.skipUnless(os.environ.get("TMAC_LATEST_CHECK"), "set TMAC_LATEST_CHECK to the compiled ARM64 probe")
    def test_operator_processes_with_individually_selected_thread_counts(self):
        binary = os.environ["TMAC_LATEST_CHECK"]
        for variant in ("native", "tmac"):
            for threads in (1, 16, 32):
                with self.subTest(variant=variant, threads=threads):
                    result = subprocess.run([binary, "--bench-" + variant, str(threads)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    records = [json.loads(line) for line in result.stdout.splitlines()]
                    self.assertEqual({row["rows"] for row in records}, {1, 7, 32})
                    self.assertEqual(len(records), 3)
                    for row in records:
                        self.assertEqual(row["threads"], threads)
                        self.assertEqual(row["status"], "pass")
                        self.assertEqual(len(row["samples_us"]), 10)
                    if variant == "tmac":
                        with tempfile.TemporaryDirectory() as directory:
                            log = Path(directory) / "dispatch.log"
                            log.write_text(result.stderr)
                            latest.dispatch(log, threads)

    def test_diagnosis_host_change_uses_new_binding_and_preserves_old_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "completed"
            run.mkdir()
            kernel = run / "tmac-prefix/lib/kernels.cc"
            kernel.parent.mkdir(parents=True)
            kernel.write_bytes(b"frozen kernel")
            profile = {"hostname": "cnkunpengb08", "cpus": list(range(0, 64, 2)), "node": 0,
                       "toolchain": {"env": {}, "CMAKE_C_COMPILER": "gcc", "CMAKE_CXX_COMPILER": "g++"}}
            frozen = {"status": "smoke_and_benchmark_pass", "commit": latest.PIN, "profile": profile,
                      "tmac_baseline": str(root / "historical"), "compiled_kernel_sha256": latest.digest(kernel)}
            (run / "manifest.json").write_text(json.dumps(frozen))
            for name in ("comparison.json", "correctness.json"):
                (run / name).write_text("[]")
            backend = run / "tmac-source/ggml/src/ggml-cpu/tmac/tmac.cpp"
            backend.parent.mkdir(parents=True)
            backend.write_bytes((ROOT / "ggml/src/ggml-cpu/tmac/tmac.cpp").read_bytes())
            commands = []

            def fake_run(command, log, *, env=None, output=None):
                commands.append(command)
                if output is None:
                    if command[:2] == ["cmake", "--build"]:
                        build = Path(command[2])
                        build.mkdir(parents=True, exist_ok=True)
                        (build / "compile_commands.json").write_text("[]")
                        (build / "CMakeCache.txt").write_bytes(b"// compiler metadata: \xe0\xff\n")
                    return
                index = next(i for i, value in enumerate(command) if str(value).startswith("--bench-"))
                threads = int(command[index + 1])
                tmac = command[index] == "--bench-tmac"
                if len(command) > index + 2:
                    Path(command[index + 2]).write_bytes(bytes(4096 * 40 * 4))
                records = [{"rows": count, "threads": threads, "K": 4096, "M": 4096, "status": "pass",
                            "buffer": "CPU_TMAC" if tmac else "CPU_REPACK", "samples_us": [10.0] * 10,
                            "median_graph_us": 10.0} for count in (1, 7, 32)]
                output.write_text("\n".join(json.dumps(row) for row in records))
                log.write_text(f"tmac: packed tensor=w\ntmac: compute tensor=w threads={threads} kernel_N=1\n")

            topology = "\n".join(f"{cpu},{(cpu - 64) // 2},0,0" for cpu in range(64, 128))
            args = argparse.Namespace(run=run, out=root / "diagnosis", jobs=4, allow_host_change=False)
            with mock.patch.object(diagnose.platform, "system", return_value="Linux"), \
                    mock.patch.object(diagnose.platform, "machine", return_value="aarch64"), \
                    mock.patch.object(diagnose.platform, "node", return_value="cnkunpengb04"), \
                    mock.patch.object(diagnose.os, "sched_getaffinity", return_value=set(range(64, 128)), create=True), \
                    mock.patch.object(diagnose.subprocess, "check_output", return_value=topology), \
                    mock.patch.object(diagnose.latest, "run", side_effect=fake_run), \
                    mock.patch.object(diagnose, "probe", return_value={}), \
                    mock.patch.object(diagnose.shutil, "which", return_value=None), \
                    mock.patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "allow-host-change"):
                    diagnose.execute(args)
                self.assertFalse(args.out.exists())
                args.allow_host_change = True
                diagnose.execute(args)
                args.row_block = 8
                args.out = root / "blocked"
                diagnose.execute(args)
            blocked = json.loads((args.out / "diagnosis.json").read_text())
            self.assertTrue(blocked["blocked_outputs_bitwise_equal"])
            self.assertEqual(blocked["experimental_row_block"], 8)
            self.assertEqual((args.out / "original-tmac-CMakeCache.txt").read_bytes(), b"// compiler metadata: \xe0\xff\n")
            schedule = json.loads((args.out / "schedule-comparison.json").read_text())
            self.assertEqual(len(schedule), 9)
            self.assertTrue(all(row["original_over_blocked"] == 1.0 for row in schedule))
            self.assertEqual(backend.read_bytes(), (ROOT / "ggml/src/ggml-cpu/tmac/tmac.cpp").read_bytes())
            args.out = root / "diagnosis"
            metadata = json.loads((args.out / "diagnosis.json").read_text())
            self.assertTrue(metadata["host_changed"])
            self.assertFalse(metadata["historical_throughput_used"])
            self.assertEqual(metadata["completed_profile"], profile)
            self.assertEqual(metadata["execution_profile"]["hostname"], "cnkunpengb04")
            self.assertEqual(metadata["execution_profile"]["cpus"], list(range(64, 128, 2)))
            measurements = [command for command in commands if command[0] == "numactl"]
            self.assertEqual(len(measurements), 15)
            for command in measurements:
                index = next(i for i, value in enumerate(command) if str(value).startswith("--bench-"))
                expected = list(range(64, 128, 2))[:int(command[index + 1])]
                self.assertEqual(command[1], "--physcpubind=" + ",".join(map(str, expected)))

    def test_diagnostic_ratio_uses_durations_in_the_correct_direction(self):
        rows = [{"variant": variant, "rows": count, "threads": threads, "median_graph_us": value}
                for threads in (1, 16, 32) for count in (1, 7, 32)
                for variant, value in (("native", 10.0), ("tmac", 20.0))]
        result = diagnose.compare_rows(rows)
        self.assertEqual(len(result), 9)
        self.assertTrue(all(row["native_us_over_tmac_us"] == 0.5 for row in result))
        rows[0]["median_graph_us"] = float("nan")
        with self.assertRaises(ValueError):
            diagnose.compare_rows(rows)

    def test_half_probe_reads_ascii_entry_from_non_utf8_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            for value, expected in ((b"1", "_Float16"), (b"", "__fp16"), (b"0", "__fp16")):
                cache.write_bytes(b"// generated binary metadata: \xe0\x80\xff\nTMAC_HAS_FLOAT16:INTERNAL=" + value + b"\n")
                self.assertEqual(latest.compiled_half_storage(cache), expected)
            for data in (b"", b"TMAC_HAS_FLOAT16:INTERNAL=unexpected\n",
                         b"TMAC_HAS_FLOAT16:INTERNAL=1\nTMAC_HAS_FLOAT16:INTERNAL=\n"):
                cache.write_bytes(data)
                with self.assertRaises(ValueError):
                    latest.compiled_half_storage(cache)

    def test_neon_broadcast_changes_only_expected_half_scalar_expressions(self):
        neon = "\n".join(["((vs) * lut_s)", "((vs) * lut_s + lut_b)", "((cs) + zs * partial_sum)"] +
                         [f"vec_c{index} * vec_s)" for index in range(4)])
        prefix = "inline int32_t tbl_g4_int8_float_update_impl(\n#ifdef __ARM_NEON\n"
        suffix = "\n#elif defined __AVX2__\n((vs) * lut_s + lut_b)\n#endif\n"
        source = prefix + neon + suffix
        result = latest.neon_broadcast_source(source)
        self.assertTrue(result.startswith(prefix))
        self.assertTrue(result.endswith(suffix))
        self.assertEqual(result.count("vdupq_n_f16("), 8)
        with self.assertRaisesRegex(ValueError, "Unexpected T-MAC NEON expression"):
            latest.neon_broadcast_source(source.replace("vec_c3 * vec_s)", "changed"))
        with self.assertRaisesRegex(ValueError, "Expected one"):
            latest.neon_broadcast_source(source + source)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is unavailable")
    def test_cmake_finds_tmac_in_hyphenated_package_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "prefix/lib/cmake/t-mac"
            package.mkdir(parents=True)
            (package / "TMACConfig.cmake").write_text(latest.tmac_package_config())
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.22)\nproject(check LANGUAGES CXX)\nfind_package(TMAC REQUIRED CONFIG)\n"
                "add_executable(probe probe.cpp)\ntarget_compile_features(probe PRIVATE cxx_std_17)\n"
                "target_link_libraries(probe PRIVATE t_mac_no_tvm)\n")
            probe = latest.half_storage_source("typedef _Float16 half;\n")
            probe += "int main() { half value = 1; return sizeof(value) != 2 || float(value) != 1; }\n"
            (source / "probe.cpp").write_text(probe)
            modes = [False, True] if platform.machine().lower() in ("arm64", "aarch64") else [False]
            for fallback in modes:
                build = root / ("fallback" if fallback else "native")
                flags = ["-DTMAC_HAS_FLOAT16:INTERNAL="] if fallback else []
                subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                                *latest.tmac_package_flags(root / "prefix"), *flags],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
                result = subprocess.run(["cmake", "--build", str(build)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.assertEqual(result.returncode, 0, result.stdout)
                subprocess.run([str(build / "probe")], check=True)
                self.assertIn(latest.compiled_half_storage(build / "CMakeCache.txt"), ("_Float16", "__fp16"))
                if fallback:
                    self.assertEqual(latest.compiled_half_storage(build / "CMakeCache.txt"), "__fp16")

    def test_half_storage_adapts_each_generated_declaration(self):
        source = "typedef _Float16 half;\n// next kernel\ntypedef _Float16 half;\n"
        result = latest.half_storage_source(source)
        self.assertEqual(result.count("typedef __fp16 half;"), 2)
        self.assertEqual(result.count("typedef _Float16 half;"), 2)
        self.assertIn("// next kernel", result)
        with self.assertRaises(ValueError):
            latest.half_storage_source("typedef float half;")

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
