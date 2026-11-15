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
