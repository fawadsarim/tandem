"""jetson-orinnx/parity_check_orinnx.py -- Gate PB-3 reduced FP-parity check.

jetson/'s documented FP-parity semantics require a true (non-QDQ) FP32 ONNX
reference; that file is a .gitignore'd HOST-export artifact, absent from the
committed tree (see PORT_NOTES.md item 8 / 06_MISSING.md). The full gate as
jetson/ defines it is therefore NOT RUNNABLE with the artifacts available on
this device -- this script runs a reduced, weaker substitute instead, with the
limitation stated explicitly here and in every printed line, not silently
treated as equivalent:

  CHECK A (n197, n69 only -- calibration data exists for these two buckets):
    FP16 engine vs the plain ("FP32-labeled") engine, both built from the
    QDQ-annotated ONNX (so both already contain the same baked-in Q/DQ nodes --
    this is NOT a true unquantized-vs-FP16 comparison), run on REAL calibration
    input instead of random noise. Real input matters: the Q/DQ nodes' scales
    were calibrated against real activation distributions, and random Gaussian
    input previously produced large diffs (0.20-0.29) that most likely reflect
    that mismatch, not a device defect.

  CHECK B (all 5 buckets):
    FP16 engine vs INT8(QDQ) engine, same graph family, same input (random,
    fixed seed -- no calibration data needed for a same-input consistency
    check). This checks TensorRT's precision-tactic selection is internally
    consistent on this device -- it does not answer "does this match the
    source model", which is the question the full gate cannot answer here.
"""
import sys
import numpy as np
import tensorrt as trt
import torch


def run_engine(engine_path, input_array=None, fixed_seed=1234):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, f"Failed to deserialize {engine_path}"
    context = engine.create_execution_context()
    n_bindings = engine.num_bindings if hasattr(engine, "num_bindings") else engine.num_io_tensors
    rng = np.random.RandomState(fixed_seed)
    device_buffers, bindings, out_idx, out_shape = [], [], None, None
    for i in range(n_bindings):
        name = engine.get_binding_name(i) if hasattr(engine, "get_binding_name") else engine.get_tensor_name(i)
        shape = engine.get_binding_shape(i) if hasattr(engine, "get_binding_shape") else engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_binding_dtype(i) if hasattr(engine, "get_binding_dtype")
                            else engine.get_tensor_dtype(name))
        is_input = engine.binding_is_input(i) if hasattr(engine, "binding_is_input") else \
            (engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        size = int(np.prod(shape))
        if is_input:
            if input_array is not None:
                host_mem = input_array.astype(dtype).reshape(-1)
                assert host_mem.size == size, f"input size {host_mem.size} != binding size {size} for {engine_path}"
            else:
                host_mem = rng.randn(size).astype(dtype)
        else:
            host_mem = np.zeros(size, dtype=dtype)
            out_idx, out_shape = i, shape
        device_mem = torch.from_numpy(np.ascontiguousarray(host_mem)).cuda()
        device_buffers.append(device_mem)
        bindings.append(int(device_mem.data_ptr()))
    context.execute_v2(bindings)
    torch.cuda.synchronize()
    out = device_buffers[out_idx].cpu().numpy().reshape(out_shape)
    return out


def check_a():
    print("=== CHECK A: FP16 vs plain-build engine, REAL calibration input (n197, n69 only) ===")
    print("NOTE: both engines share the same baked-in Q/DQ nodes -- this is not a true")
    print("unquantized-vs-FP16 comparison. See this file's module docstring.")
    results = []
    for n in (197, 69):
        calib = np.load(f"out/segments/calib/calib_n{n}.npy")
        sample = calib[0:1]  # (1, n, 384) -- one real calibration sample
        out_plain = run_engine(f"out/engines/n{n}_fp32_ref.engine", input_array=sample)
        out_fp16 = run_engine(f"out/engines/n{n}_fp16.engine", input_array=sample)
        max_abs_diff = float(np.max(np.abs(out_plain.astype(np.float64) - out_fp16.astype(np.float64))))
        results.append((n, max_abs_diff))
        print(f"[n={n}] plain-build vs FP16, real calib input: max_abs_diff={max_abs_diff:.6e}")
    return results


def check_b():
    print()
    print("=== CHECK B: FP16 vs INT8(QDQ) engine, same random input, all 5 buckets ===")
    print("This checks precision-tactic consistency on this device, not source-model fidelity.")
    results = []
    for n in (69, 101, 133, 165, 197):
        out_fp16 = run_engine(f"out/engines/n{n}_fp16.engine", input_array=None, fixed_seed=1234)
        out_int8 = run_engine(f"out/engines/n{n}_int8_qdq.engine", input_array=None, fixed_seed=1234)
        max_abs_diff = float(np.max(np.abs(out_fp16.astype(np.float64) - out_int8.astype(np.float64))))
        results.append((n, max_abs_diff))
        print(f"[n={n}] FP16 vs INT8(QDQ), same random input: max_abs_diff={max_abs_diff:.6e}")
    return results


def main():
    a_results = check_a()
    b_results = check_b()
    print()
    print("=== SUMMARY (informational -- reduced gate, no hard tolerance asserted here; ===")
    print("=== see paste-back for the STOP conditions this was evaluated against)      ===")
    for n, d in a_results:
        print(f"  CHECK A n={n}: {d:.6e}")
    for n, d in b_results:
        print(f"  CHECK B n={n}: {d:.6e}")


if __name__ == "__main__":
    main()
