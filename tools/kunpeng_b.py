"""B specialization for the pinned generated-C ABI and original weight layout."""
import configparser
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parents[1] / "tmac_experiment"
HEADER = HERE / "optimizations/bitplane-fuse.h"


def fuse_kernels(text, kcfg, quantization):
    expected = {"bits": 4, "weight_group_size": 32, "act_group_size": 32, "lut_group": 4,
                "fast_aggregation": False, "lut_dtype": "int8", "output_dtype": "float16",
                "aggregation_dtype": "int32", "zero_point": False}
    if quantization != expected:
        raise RuntimeError("B requires Q4_0 groups=32,g=4, no fast aggregation or extra zero point")
    config = configparser.ConfigParser()
    config.read_string(kcfg)
    shapes = set()
    for section in config.sections():
        match = re.fullmatch(r"qgemm_lut_t1_int8_m(\d+)_k(\d+)_n1_b4", section)
        if not match:
            raise RuntimeError("Unsupported B configuration section: " + section)
        m, k = map(int, match.groups())
        values = config[section]
        bm = values.getint("bm")
        if bm not in (128, 256, 320, 512, 640, 1024) or m <= 0 or m % bm or k <= 0 or k > 65536 or k % 32 or any(values.getint(key) != val for key, val in {"group_size": 32, "kfactor": 8, "simd_n_in": 16, "simd_n_out": 8, "lut_scales_size": k // 32, "scales_size": m // 4 * (k // 32), "n_tile_num": m // bm}.items()):
            raise RuntimeError("Unsupported B shape or weight layout: " + section)
        shapes.add((bm, k))
    pattern = re.compile(r"int32_t (qgemm_lut_t1_int8_m(\d+)_k(\d+)_n1_b4)\(void\* A, void\* LUT, void\* Scales, void\* LUT_Scales, void\* LUT_Biases, void\* C\)\s*\{")
    matches = list(pattern.finditer(text))
    if not shapes or {(int(m[2]), int(m[3])) for m in matches} != shapes or len(matches) != len(shapes):
        raise RuntimeError("Generated qgemm ABI does not match frozen kcfg")
    for match in reversed(matches):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise RuntimeError("Unbalanced generated qgemm body")
        body = '\n    return tmac_bitplane_fuse<{}, {}>((const uint8_t *) A, (const int8_t *) LUT, (const float16_t *) Scales, (const float16_t *) LUT_Scales, (const float16_t *) LUT_Biases, (float16_t *) C);\n}}'.format(match[2], match[3])
        text = text[:match.end()] + body + text[end:]
    return HEADER.read_text() + "\n" + text


def check_fused(env, output, recorder):
    from kunpeng_baseline import exp, write
    binary = output / "check-bitplane-fuse"
    flags = ["-O3", "-std=c++17", "-march=armv8.2-a+fp16"]
    recorder.run(["clang++"] + flags + [str(HERE / "harness/bitplane-fuse.cpp"), "-o", str(binary)], env=env)
    result = recorder.run([str(binary)], env=env, capture=True)
    import json
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    metrics["header_sha256"] = exp.sha256_file(HEADER)
    write(output / "B-check.json", metrics)
    wrapper = output / "B-assembly.cpp"
    wrapper.write_text('#include "' + str(HEADER) + '"\nextern "C" int b_probe(const uint8_t *w, const int8_t *l, const float16_t *s, const float16_t *ls, const float16_t *b, float16_t *c) { return tmac_bitplane_fuse<256,4096>(w,l,s,ls,b,c); }\n')
    recorder.run(["clang++"] + flags + ["-S", str(wrapper), "-o", str(output / "B-kernel.s")], env=env)
