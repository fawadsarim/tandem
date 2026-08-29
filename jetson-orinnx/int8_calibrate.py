"""jetson-orinnx/int8_calibrate.py -- ON-DEVICE companion to build_engines.sh.

WHY THIS EXISTS (not pure trtexec): trtexec's `--int8`/`--calib=<file>`
flags can WRITE or READ a calibration cache, but trtexec alone has no CLI
mechanism to feed it real, pinned calibration ARRAYS (the `.npy` files
export_segments.py captured from real ImageNet activations) -- its
built-in calibration path uses synthetic/random data unless you supply a
custom `IInt8EntropyCalibrator2`, which requires the TensorRT Python API.
Using random data for calibration would be a real accuracy bug (it gives a
*valid* INT8 engine with poor clipping ranges), so this script exists
specifically to calibrate from the real, pinned data instead. build_engines.sh
calls this ONCE per bucket to produce the `.cache` file and the INT8
`.engine` file directly; the actual latency BENCHMARK step (warmUp,
iterations, logs) still goes through trtexec afterward, per the README's
spec (`trtexec --loadEngine=... --int8 ...`).

Requires the `tensorrt` Python package, provided system-wide by JetPack
(accessed via jetson-orinnx/venv's --system-site-packages, see setup.sh) --
not pip-installable generically, so not in requirements.txt. Also
needs a CUDA device-memory allocator/copy mechanism to feed calibration
batches to TensorRT's C++ calibrator interface -- uses `torch.cuda` for
this (already a real dependency of this venv, per paired_eval.py and
this JetPack's own PyTorch wheel), NOT `pycuda`.

Real bug found and fixed, 2026-07: this script originally required
`pycuda` for exactly that device-memory role, with a docstring/error
message claiming it "ships with JetPack 6.x system-wide" -- true for
JetPack 6.x, NOT for JetPack 7.2 (confirmed on a real device: no
`pycuda`, no `cuda-python`, and the system CUDA toolkit's `nvcc`/headers
aren't even on `PATH` in a plain SSH session, so compiling `pycuda`
from source here would be slow and failure-prone). Switched to
`torch.cuda` instead -- device memory allocation, host->device copy,
and a raw pointer via `.data_ptr()` for TensorRT's calibrator interface
all come from a dependency this venv already has and has already
verified working (torch's CUDA build for this exact JetPack/device),
rather than adding a new, ARM-Jetson-fragile one.

Usage:
    python int8_calibrate.py --onnx out/segments/block_seg_n133.onnx \\
        --calib-npy out/segments/calib/calib_n133.npy \\
        --cache-out out/engines/n133_int8.cache \\
        --engine-out out/engines/n133_int8.engine
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
    import torch
except ImportError as e:
    print(f"FATAL: tensorrt/torch not importable ({e!r}). tensorrt ships with JetPack system-wide -- "
          "ensure jetson-orinnx/venv was created with --system-site-packages (setup.sh does this) and that "
          "you're running inside that venv. torch is a JetPack-specific wheel installed manually per "
          "README.md/setup.sh's own notes -- see there if it's missing.", file=sys.stderr)
    raise

assert torch.cuda.is_available(), (
    "torch.cuda.is_available() is False -- INT8 calibration needs a real CUDA device to allocate "
    "calibration buffers on. If this is the Jetson itself, something is wrong with the torch install "
    "(see README.md); this script should never run anywhere else."
)


class NpyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds export_segments.py's pinned, real-activation .npy array as
    INT8 calibration data, one image at a time. Writes/reads a calibration
    cache so re-runs (e.g. re-building after a clean) skip re-calibrating."""

    def __init__(self, npy_path: str, cache_path: str):
        super().__init__()
        self.data = np.load(npy_path).astype(np.float32)
        assert self.data.ndim == 3, f"expected (N, tokens, d_model), got shape {self.data.shape}"
        self.batch_size = 1
        self.current_index = 0
        self.cache_path = cache_path
        one = np.ascontiguousarray(self.data[0:1])
        # torch.cuda tensor instead of pycuda's cuda.mem_alloc() -- see module docstring. A
        # contiguous CUDA tensor's .data_ptr() is exactly the raw device pointer TensorRT's
        # calibrator interface expects back from get_batch().
        self.device_input = torch.empty(one.shape, dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.data.shape[0]:
            return None
        batch = np.ascontiguousarray(self.data[self.current_index:self.current_index + 1])
        self.device_input.copy_(torch.from_numpy(batch))
        self.current_index += 1
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        p = Path(self.cache_path)
        if p.exists():
            print(f"  Reusing existing calibration cache: {p}")
            return p.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_path).write_bytes(cache)
        print(f"  Wrote calibration cache: {self.cache_path}")


def build_int8_engine(onnx_path: str, calib_npy_path: str, cache_path: str, engine_path: str,
                       workspace_bytes: int = 1 << 30):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = NpyEntropyCalibrator(calib_npy_path, cache_path)
    config.int8_calibrator = calibrator

    print(f"  Building INT8 engine from {onnx_path} (calibrating against {calib_npy_path})...")
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"TensorRT engine build FAILED for {onnx_path}"

    Path(engine_path).write_bytes(serialized)
    # .nbytes, not len() -- real bug found and fixed, 2026-07: TensorRT 10.16.2's
    # build_serialized_network() returns an IHostMemory object that supports the buffer
    # protocol (so write_bytes() above works fine) but does NOT support len() -- confirmed on
    # real hardware: `TypeError: object of type 'tensorrt.tensorrt.IHostMemory' has no len()`,
    # raised AFTER the engine file was already written correctly to disk, so this was a
    # reporting-line bug only, not an actual build failure (the .engine file was fine, this
    # script's own nonzero exit code just made build_engines.sh treat a successful build as a
    # failure and abort). `.nbytes` is IHostMemory's own documented byte-size property.
    print(f"  Wrote INT8 engine: {engine_path} ({serialized.nbytes / 1e6:.1f} MB)")
    return engine_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calib-npy", required=True)
    parser.add_argument("--cache-out", required=True)
    parser.add_argument("--engine-out", required=True)
    args = parser.parse_args()

    assert Path(args.onnx).exists(), f"{args.onnx} not found"
    assert Path(args.calib_npy).exists(), (
        f"{args.calib_npy} not found -- re-run export_segments.py on the HOST with --imagenet-dir "
        "to produce calibration data, then copy jetson-orinnx/out/segments/ here."
    )
    Path(args.engine_out).parent.mkdir(parents=True, exist_ok=True)
    build_int8_engine(args.onnx, args.calib_npy, args.cache_out, args.engine_out)


if __name__ == "__main__":
    main()
