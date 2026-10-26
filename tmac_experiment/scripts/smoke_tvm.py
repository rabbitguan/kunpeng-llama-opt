#!/usr/bin/env python3
import numpy as np
import tvm
from tvm import te


size = 16
input_tensor = te.placeholder((size,), name="input", dtype="float32")
output_tensor = te.compute((size,), lambda i: input_tensor[i] + tvm.tir.const(1.0, "float32"), name="output")
schedule = te.create_schedule(output_tensor.op)
module = tvm.build(schedule, [input_tensor, output_tensor], target="llvm", name="setup_tvm_smoke")
device = tvm.cpu(0)
input_data = np.arange(size, dtype="float32")
output_data = tvm.nd.empty((size,), dtype="float32", device=device)
module(tvm.nd.array(input_data, device), output_data)
np.testing.assert_allclose(output_data.numpy(), input_data + 1.0, rtol=0.0, atol=0.0)
print("TVM LLVM smoke passed")
