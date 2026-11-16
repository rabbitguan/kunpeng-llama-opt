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


    def test_frozen_artifacts_follow_actual_install_layout(self):
        import tempfile
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import verify_artifacts, exp
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            hashes = {}
            for name, installed in [("kernels.cc", "lib/kernels.cc"), ("kernels.h", "include/t-mac/kernels.h"), ("kcfg.ini", "lib/kcfg.ini")]:
                generated = root / "generated" / name
                target = root / "install" / installed
                generated.parent.mkdir(parents=True, exist_ok=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                generated.write_text(name)
                target.write_text(name)
                hashes[name] = exp.sha256_file(generated)
            verify_artifacts(root, hashes)
            header = root / "install/include/t-mac/kernels.h"
            header.write_text("changed")
            with self.assertRaises(RuntimeError):
                verify_artifacts(root, hashes)
            header.unlink()
            with self.assertRaises(RuntimeError):
                verify_artifacts(root, hashes)


class BExperimentTest(unittest.TestCase):
    def test_b_specialization_preserves_other_kernels_and_rejects_wrong_layout(self):
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_b import fuse_kernels
        config = json.loads((pathlib.Path(__file__).resolve().parents[1] / "config.json").read_text())["quantization"]
        cfg = "[qgemm_lut_t1_int8_m1024_k32_n1_b4]\nbm=256\ngroup_size=32\nkfactor=8\nsimd_n_in=16\nsimd_n_out=8\nlut_scales_size=1\nscales_size=256\nn_tile_num=4\n"
        text = "int32_t qgemm_lut_t1_int8_m256_k32_n1_b4(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) { if (true) { old_path(); } return 0; }\nint preprocessor() { return 7; }"
        result = fuse_kernels(text, cfg, config)
        self.assertIn("tmac_bitplane_fuse<256, 32>", result)
        self.assertNotIn("old_path", result)
        self.assertIn("int preprocessor() { return 7; }", result)
        for bad_cfg in [cfg.replace("kfactor=8", "kfactor=16"), cfg.replace("group_size=32", "group_size=128"), cfg.replace("scales_size=256", "scales_size=512")]:
            with self.assertRaises(RuntimeError):
                fuse_kernels(text, bad_cfg, config)
        with self.assertRaises(RuntimeError):
            fuse_kernels(text, cfg, dict(config, fast_aggregation=True))
        with self.assertRaises(RuntimeError):
            fuse_kernels(text.replace("void* C", "void* D"), cfg, config)

    def test_b_model_error_gate_is_fixed_and_rejects_nonfinite(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import b_logits_metrics
        self.assertEqual(b_logits_metrics([1.0, 2.0], [1.001, 2.001])["status"], "pass")
        self.assertEqual(b_logits_metrics([1.0], [1.1])["status"], "failed")
        self.assertEqual(b_logits_metrics([100.0], [102.0])["status"], "failed")
        self.assertEqual(b_logits_metrics([0.0], [0.001])["status"], "failed")
        self.assertEqual(b_logits_metrics([1.0], [float("nan")])["status"], "failed")


class OperatorOnlyTest(unittest.TestCase):
    def test_operator_summary_does_not_claim_model_pass(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import operator_summary
        rows = [{"variant": name, "lut_us": 1, "compute_us": time, "total_us": time + 1}
                for name, time in [("tmac_ref", 10), ("B", 5)]]
        result = operator_summary(rows, "B")
        self.assertEqual(result["status"], "operator_pass_model_pending")
        self.assertAlmostEqual(result["operator_speedup"], 11 / 6)
        self.assertNotIn("decode_median_tokens_per_second", result)

    def test_reference_harness_reuse_is_tied_to_same_baseline(self):
        import json
        import tempfile
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_a import reference_operator_harness
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            baseline = root / "basic_baseline"
            baseline.mkdir()
            run = root / "B_compare_01"
            harness = run / "tmac_ref-harness"
            harness.mkdir(parents=True)
            (harness / "tmac-operator").write_text("fixture")
            manifest = {"status": "preliminary_pass", "baseline": str(baseline), "commits": {"tmac": "fixed"}}
            (run / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(reference_operator_harness(baseline, manifest), harness)
            self.assertIsNone(reference_operator_harness(baseline, {"commits": {"tmac": "changed"}}))


class CacheScheduleTest(unittest.TestCase):
    def test_cache_prototype_rejects_incompatible_frozen_config(self):
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_c import validate_config
        config = json.loads((pathlib.Path(__file__).resolve().parents[1] / "config.json").read_text())
        text = "[qgemm_lut_t1_int8_m16384_k4096_n1_b4]\nbm=256\nkfactor=8\ngroup_size=32\nsimd_n_in=16\nsimd_n_out=8\nlut_scales_size=128\nscales_size=524288\nn_tile_num=64\n"
        validate_config({"config": config}, text)
        for invalid in [text.replace("bm=256", "bm=128"), text.replace("kfactor=8", "kfactor=16"), text.replace("n_tile_num=64", "n_tile_num=32")]:
            with self.assertRaises(RuntimeError):
                validate_config({"config": config}, invalid)
        config["quantization"]["fast_aggregation"] = True
        with self.assertRaises(RuntimeError):
            validate_config({"config": config}, text)

    def test_cache_large_k_requires_matching_frozen_kernel(self):
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_c import validate_config
        config = json.loads((pathlib.Path(__file__).resolve().parents[1] / "config.json").read_text())
        cfg = "[qgemm_lut_t1_int8_m16384_k14336_n1_b4]\nbm=256\nkfactor=8\ngroup_size=32\nsimd_n_in=16\nsimd_n_out=8\nlut_scales_size=448\nscales_size=1835008\nn_tile_num=64\n"
        validate_config({"config": config}, cfg, 14336)
        with self.assertRaises(RuntimeError):
            validate_config({"config": config}, cfg, 4096)

    def test_cache_reuse_rejects_modified_source(self):
        import tempfile
        import json
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_c import reusable_manifest, exp, HEADER
        with tempfile.TemporaryDirectory() as directory:
            run = pathlib.Path(directory)
            source = run / "install/lib/kernels.cc"
            source.parent.mkdir(parents=True)
            source.write_text("frozen")
            frozen = {"commits": {"tmac": "pinned"}, "model_sha256": "model"}
            baseline = run / "baseline"
            record = dict(frozen, status="operator_pass_model_pending", opts=["C"], baseline=str(baseline), header_sha256=exp.sha256_file(HEADER), kernels_sha256=exp.sha256_file(source))
            (run / "manifest.json").write_text(json.dumps(record))
            self.assertEqual(reusable_manifest(run, baseline, frozen), record)
            source.write_text("changed")
            with self.assertRaises(RuntimeError):
                reusable_manifest(run, baseline, frozen)

    def test_cache_two_config_summary_keeps_pass_differences(self):
        import argparse
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
        from kunpeng_c import parse_r_list, comparison_summary
        self.assertEqual(parse_r_list("1,4"), (1, 4))
        for invalid in ["1,1", "4", "1,8", ""]:
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_r_list(invalid)
        rows = [{"variant": name, "total_us": time, "note": ";pass=" + str(p)}
                for p, name, time in [(1, "C-R1-BK256", 6000), (1, "C-R4-BK256", 2000),
                                      (2, "C-R1-BK256", 2000), (2, "C-R4-BK256", 2000)]]
        result = comparison_summary(rows)
        self.assertFalse(result["original_reference_measured"])
        self.assertEqual(result["operator_speedups"], {})
        self.assertEqual(result["pass_tile_reuse_speedups"]["1"]["C-R4-BK256"], 3)
        self.assertEqual(result["pass_tile_reuse_speedups"]["2"]["C-R4-BK256"], 1)
