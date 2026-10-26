#!/usr/bin/env python3
import json
import pathlib
import sys

from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen


output = pathlib.Path(sys.argv[1])
configuration = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
output.mkdir(parents=True, exist_ok=True)
target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+neon"
common = {
    "save_dir": str(output),
    "dtype": configuration["lut_dtype"],
    "target": target,
    "verify": True,
    "tune": False,
    "remote_kwargs": None,
    "bits": configuration["bits"],
    "out_dtype": configuration["output_dtype"],
    "act_group_size": configuration["act_group_size"],
    "cc": "clang++",
    "cc_opts": ["-O3", "-std=c++17", "-march=armv8.2-a+fp16", "-mllvm", "-inline-threshold=10000"],
}
fast_aggregation_k = configuration["act_group_size"] // 4 if configuration["fast_aggregation"] else 0
preprocessor = QGeMMLUTBitsPreprocessorCodegen(name="preprocessor", fast_aggregation_k=fast_aggregation_k, **common)
qgemm = QGeMMLUTBitsCodegen(name="qgemm_lut", group_size=configuration["weight_group_size"], m_groups=-1, aggregation_dtype=configuration["aggregation_dtype"], zero_point=configuration["zero_point"], **common)
preprocessor.num_threads = 1
qgemm.num_threads = 1
preprocessor.evaluate(1, 128, number=1, repeat=1, min_repeat_ms=0)
qgemm.evaluate(128 * configuration["bits"], 1, 128, number=1, repeat=1, min_repeat_ms=0)
for name in ["preprocessor", "qgemm_lut"]:
    assembly = output / name / "src.S"
    llvm_ir = output / name / "src.ll"
    if not assembly.is_file() or assembly.stat().st_size == 0 or not llvm_ir.is_file():
        raise RuntimeError("T-MAC {} did not emit LLVM IR and assembly".format(name))
    ir_text = llvm_ir.read_text(encoding="utf-8", errors="replace")
    if "aarch64-linux-gnu" not in ir_text or "+neon" not in ir_text or "+fullfp16" not in ir_text:
        raise RuntimeError("T-MAC {} LLVM IR does not preserve the required AArch64 NEON/FP16 target".format(name))
print("T-MAC NEON/FP16 setup smoke passed")
