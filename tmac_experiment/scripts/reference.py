#!/usr/bin/env python3
import math
import random


def q4_0_pack(values):
    if len(values) % 32:
        raise ValueError("Q4_0 data length must be divisible by 32")
    out = bytearray()
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        for value in block:
            if value < -8 or value > 7:
                raise ValueError("Q4_0 value outside [-8, 7]")
        for i in range(16):
            out.append((block[i] + 8) | ((block[i + 16] + 8) << 4))
    return bytes(out)


def q4_0_unpack(data):
    if len(data) % 16:
        raise ValueError("packed Q4_0 length must be divisible by 16")
    out = []
    for start in range(0, len(data), 16):
        lo = []
        hi = []
        for value in data[start:start + 16]:
            lo.append((value & 15) - 8)
            hi.append((value >> 4) - 8)
        out.extend(lo + hi)
    return out


def make_lut(activation, group=4):
    if len(activation) % group:
        raise ValueError("activation length must be divisible by LUT group")
    result = []
    for start in range(0, len(activation), group):
        values = activation[start:start + group]
        table = []
        for index in range(1 << group):
            table.append(sum(values[i] if index & (1 << i) else -values[i] for i in range(group)))
        result.append(table)
    return result


def scalar_lut_dot(qvalues, scales, activation, weight_group=32, lut_group=4):
    if len(qvalues) != len(activation) or len(qvalues) % weight_group:
        raise ValueError("invalid grouped dot dimensions")
    if len(scales) != len(qvalues) // weight_group:
        raise ValueError("invalid scale count")
    tables = make_lut(activation, lut_group)
    total = 0.0
    for k in range(0, len(qvalues), lut_group):
        bit_sum = 0.0
        coefficients = [0.5, 1.0, 2.0, 4.0]
        for bit in range(4):
            index = sum(((qvalues[k + i] + 8) >> bit & 1) << i for i in range(lut_group))
            bit_sum += tables[k // lut_group][index] * coefficients[bit]
        bit_sum -= 0.5 * sum(activation[k:k + lut_group])
        total += bit_sum * scales[k // weight_group]
    return total


def direct_dot(qvalues, scales, activation, weight_group=32):
    return sum(q * scales[k // weight_group] * activation[k] for k, q in enumerate(qvalues))


def self_test(seed=20241025):
    cases = [
        [0.0] * 64,
        [(-1.0 if i % 2 else 1.0) for i in range(64)],
        [65504.0 if i % 2 else -65504.0 for i in range(64)],
    ]
    rng = random.Random(seed)
    cases.append([rng.uniform(-4.0, 4.0) for _ in range(64)])
    qvalues = [rng.randrange(-8, 8) for _ in range(64)]
    scales = [0.125, -0.25]
    unpacked = q4_0_unpack(q4_0_pack(qvalues))
    if unpacked != qvalues:
        raise AssertionError("Q4_0 packing round trip failed")
    for activation in cases:
        got = scalar_lut_dot(qvalues, scales, activation)
        expected = direct_dot(qvalues, scales, activation)
        if not math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-7):
            raise AssertionError("scalar LUT reference differs from direct dot")
        if not math.isfinite(got):
            raise AssertionError("non-finite scalar result")
    for invalid in [math.nan, math.inf, -math.inf]:
        activation = [0.0] * 64
        activation[31] = invalid
        if math.isfinite(scalar_lut_dot(qvalues, scales, activation)):
            raise AssertionError("NaN or Inf was not detected")
    return {"cases": len(cases), "pack_values": len(qvalues), "status": "pass"}
