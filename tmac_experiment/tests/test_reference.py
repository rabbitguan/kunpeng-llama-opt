import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from reference import direct_dot, make_lut, q4_0_pack, q4_0_unpack, scalar_lut_dot


class ReferenceTest(unittest.TestCase):
    def test_q4_round_trip_and_nibble_order(self):
        values = list(range(-8, 8)) * 2
        packed = q4_0_pack(values)
        self.assertEqual(packed[0], 0x00)
        self.assertEqual(q4_0_unpack(packed), values)

    def test_lut_sign_order(self):
        table = make_lut([1.0, 2.0, 4.0, 8.0])[0]
        self.assertEqual(table[0], -15.0)
        self.assertEqual(table[15], 15.0)
        self.assertEqual(table[1], -13.0)

    def test_scalar_reference_group_boundary(self):
        qvalues = [7 if i % 3 else -8 for i in range(64)]
        scales = [0.5, -0.25]
        activation = [(-1.0 if i % 2 else 1.0) * (i + 1) / 64 for i in range(64)]
        self.assertAlmostEqual(scalar_lut_dot(qvalues, scales, activation), direct_dot(qvalues, scales, activation), places=10)

    def test_invalid_lengths(self):
        with self.assertRaises(ValueError):
            q4_0_pack([0])
        with self.assertRaises(ValueError):
            scalar_lut_dot([0] * 32, [1.0], [0.0] * 31)


if __name__ == "__main__":
    unittest.main()


class VariantTest(unittest.TestCase):
    def test_selection_distinguishes_sequence_and_combination(self):
        import argparse
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_exp import selected
        self.assertEqual(selected(argparse.Namespace(opts=None, variants="A,B,AB")), ["A", "B", "AB"])
        self.assertEqual(selected(argparse.Namespace(opts="B,A", variants="")), ["AB"])
        self.assertEqual(selected(argparse.Namespace(opts="", variants="")), ["tmac_off"])
        for opts, variants in [("A,A", ""), ("Z", ""), (None, "unknown"), (None, "A,A")]:
            with self.assertRaises(ValueError):
                selected(argparse.Namespace(opts=opts, variants=variants))

    def test_lut_check_uses_project_toolchain_environment(self):
        import tempfile
        from unittest import mock
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        import kunpeng_exp
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            original = source / "python/t_mac/intrins/lut_ctor.cc"
            original.parent.mkdir(parents=True)
            original.write_text("fixture")
            (root / "setup-environment.json").write_text("{}")
            env = {"PATH": "/project/llvm/bin", "LD_LIBRARY_PATH": "/project/llvm/lib"}
            with mock.patch.object(kunpeng_exp.platform, "machine", return_value="aarch64"), mock.patch.object(kunpeng_exp.legacy, "sha256_file", return_value="f03f1fb0e9acd1cc9679fbdb1d47dc2599f97806eba7d5b0bd996fa8da9e386b"), mock.patch.object(kunpeng_exp.legacy, "source_environment", return_value=env), mock.patch.object(kunpeng_exp.shutil, "which", return_value="/project/llvm/bin/clang++") as which, mock.patch.object(kunpeng_exp, "run", return_value="pass") as run:
                kunpeng_exp.check_lut(source, root / "output")
            which.assert_called_once_with("clang++", path=env["PATH"])
            for call in run.call_args_list[2:]:
                self.assertEqual(call.kwargs["env"]["PATH"], env["PATH"])
                self.assertEqual(call.kwargs["env"]["LD_LIBRARY_PATH"], env["LD_LIBRARY_PATH"])


class BasicBaselineTest(unittest.TestCase):
    def test_logits_gate_rejects_invalid_and_large_errors(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_baseline import compare_logits
        self.assertEqual(compare_logits([1.0, -1.0], [1.0, -1.0])["status"], "pass")
        self.assertEqual(compare_logits([1.0], [2.0])["status"], "failed")
        self.assertEqual(compare_logits([1.0], [float("nan")])["status"], "failed")
        self.assertEqual(compare_logits([float("inf")], [1.0])["status"], "failed")
        self.assertEqual(compare_logits([0.0], [1.0])["status"], "failed")
        self.assertEqual(compare_logits([0.0], [0.0])["status"], "pass")
        with self.assertRaises(RuntimeError):
            compare_logits([1.0], [])

    def test_baseline_source_pin_matches_provenance(self):
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_baseline import UPSTREAM
        provenance = json.loads((pathlib.Path(__file__).resolve().parents[1] / "baseline_sources.json").read_text())
        self.assertEqual(UPSTREAM, provenance["llama_ref"]["commit"])
        self.assertEqual(len(provenance["integrated_fork"]["extra_commits"]), 14)


class AExperimentTest(unittest.TestCase):
    def test_generated_kernel_edit_is_limited_to_constructor(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import replace_constructor
        self.assertEqual(replace_constructor("lookup;OLD;output", "OLD", "NEW"), "lookup;NEW;output")
        for kernel, old, new in [("missing", "OLD", "NEW"), ("OLDOLD", "OLD", "NEW"), ("OLD", "OLD", "OLD")]:
            with self.assertRaises(RuntimeError):
                replace_constructor(kernel, old, new)

    def test_benchmark_samples_discard_first_without_rounding(self):
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import bench_samples
        row = {"n_gen": 32, "n_prompt": 0, "samples_ns": [1000000000, 2000000000, 4000000000, 8000000000]}
        self.assertEqual(bench_samples("initialization log\n" + json.dumps(row)), [16.0, 8.0, 4.0])
        for change in [{"samples_ns": [1, 2, 3]}, {"n_gen": 16}, {"n_prompt": 1}, {"samples_ns": [1, 2, 3, 0]}]:
            with self.assertRaises(RuntimeError):
                bench_samples(json.dumps(dict(row, **change)))

    def test_baseline_discovery_ignores_failed_runs(self):
        import tempfile
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import find_baseline
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name, status in [("basic_baseline_01", "preliminary_pass"), ("basic_baseline_02", "building")]:
                path = root / name
                path.mkdir()
                (path / "manifest.json").write_text(json.dumps({"status": status, "opts": []}))
            self.assertEqual(find_baseline(root), root / "basic_baseline_01")
