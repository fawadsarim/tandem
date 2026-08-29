"""jetson/measure_lut.py -- ON-DEVICE. Builds latency_lut.json: for every
(token-count bucket, precision) pair produced by build_engines.sh, records
median + p99 latency, temperature before/after, energy per inference, and
full provenance (versions, clock state, timestamp).

LATENCY SOURCE (both, for transparency -- neither silently hidden):
  - PRIMARY: this script's own timed loop, executing the built .engine
    directly via the TensorRT Python API. This is the number the LUT's
    "median"/"p99" fields are built from, and it's the mechanism that lets
    us sample jtop temperature/power CONCURRENTLY with real inference
    (trtexec's own benchmark, run earlier by build_engines.sh, is already
    finished by the time this script runs -- there is nothing left to
    sample power against).
  - CROSS-CHECK: this script also parses build_engines.sh's saved trtexec
    bench log for the same (bucket, precision) pair and records its
    "GPU Compute Time" median/p99 alongside, unasserted (the two
    methodologies differ slightly -- host-loop overhead vs trtexec's own
    internal timing loop -- so a small gap is expected and not a bug).

WHICH ENGINE "int8" MEANS (reviewed, real bug avoided, 2026-07): "int8" in this LUT's
`entries` (what replan.py's DP actually consumes) is sourced from the BEST ACHIEVABLE INT8
implementation -- explicit Q/DQ quantization via NVIDIA ModelOpt
(`int8_qdq_experiment.py`, engine file `n{bucket}_int8_qdq.engine`) -- NOT TensorRT's naive
implicit-calibration path (`int8_calibrate.py`, engine file `n{bucket}_int8.engine`), which
measured ~2-2.5x slower than FP16 due to unfused Reformat overhead (see jetson/README.md's
INT8 findings). Pricing INT8 at its worst would inflate the measured cost of every plan that
uses it, distorting every downstream budget/comparison built on the LUT. The naive engine is
still measured here too -- recorded under the separate "int8_naive_implicit" key for
provenance/comparison, never read by replan.py. See `engine_filename()`/`trtexec_log_filename()`
below for the exact file mapping.

THERMAL GUARD: temperature is checked via jtop before every (bucket,
precision) measurement. If >80C, this script sleeps and re-polls (capped
retries) rather than measuring on a throttling chip -- a throttled
measurement is not a real number and this project does not silently
report one.

Deterministic ordering: buckets ascending, PRECISIONS in the order defined above (fp16, int8,
int8_naive_implicit). Resume-safe: any (bucket, precision) key already present in
latency_lut.json is skipped on re-run unless --force.

ONE-TIME MIGRATION NEEDED before this script's first run after "int8" was redefined to mean
best-achievable (see "WHICH ENGINE 'int8' MEANS" above): any EXISTING latency_lut.json from
before this change has its "n{bucket}_int8" keys holding what are now the NAIVE numbers (that
was the only INT8 path that existed at the time). Resume-safe skip-if-present would otherwise
silently treat that stale naive data as if it were already the new best-achievable
measurement and never re-measure it. Rename those keys to "n{bucket}_int8_naive_implicit"
(and set their "precision" field to match) BEFORE re-running with --int8_qdq engines present,
so "int8" is correctly seen as missing (triggers fresh measurement) while the naive data is
preserved under its new name (skipped, not lost, not redundantly re-measured).

Needs the VENV's python specifically (./venv/bin/python3 measure_lut.py), not bare system
python3 -- see the import block below for why (real bug found and fixed, 2026-07).

Usage: ./venv/bin/python3 measure_lut.py [--force]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402

try:
    import tensorrt as trt
    import torch
except ImportError as e:
    print(f"FATAL: tensorrt/torch not importable ({e!r}). tensorrt ships with JetPack "
          "system-wide, importable from bare `python3`. torch does NOT -- it's installed "
          "only into jetson/venv (a JetPack-specific wheel, not on PyPI, not system-wide) -- "
          "run this script via './venv/bin/python3 measure_lut.py', not bare 'python3'. Real "
          "bug found and fixed, 2026-07: this script used to only need tensorrt (importable "
          "from bare python3) plus pycuda for device-memory management -- pycuda isn't "
          "available on this JetPack (same issue already found and fixed in "
          "int8_calibrate.py, see that file's own comment for the full diagnosis), so this "
          "switched to torch.cuda instead, which pulls in this new venv-python requirement.",
          file=sys.stderr)
    raise

WARMUP_ITERS = 20
MEASURE_ITERS = 200
THERMAL_LIMIT_C = 80.0
THERMAL_COOLDOWN_S = 20
THERMAL_MAX_RETRIES = 15
# "int8" here is what replan.py's DP actually consumes -- BEST ACHIEVABLE (explicit Q/DQ), not
# naive. "int8_naive_implicit" is measured too, for provenance/comparison only -- see module
# docstring's "WHICH ENGINE 'int8' MEANS" section.
PRECISIONS = ("fp16", "int8", "int8_naive_implicit")
NAIVE_INT8_LABEL = "int8_naive_implicit"


def engine_filename(n_tokens: int, precision: str) -> str:
    if precision == "int8":
        return f"n{n_tokens}_int8_qdq.engine"
    if precision == NAIVE_INT8_LABEL:
        return f"n{n_tokens}_int8.engine"
    return f"n{n_tokens}_{precision}.engine"


def trtexec_log_filename(n_tokens: int, precision: str) -> str:
    if precision == "int8":
        return f"n{n_tokens}_int8_qdq_bench.log"
    if precision == NAIVE_INT8_LABEL:
        return f"n{n_tokens}_int8_bench.log"
    return f"n{n_tokens}_{precision}_bench.log"


def get_temp_c(jetson):
    """Highest of the reported thermal zones (matches jtop's own 'temp' summary)."""
    if not jetson.temperature:
        return None
    return max(z.get("temp", 0.0) for z in jetson.temperature.values() if z.get("temp") is not None)


def get_power_mw(jetson):
    tot = jetson.power.get("tot", None) if jetson.power else None
    return tot.get("power") if isinstance(tot, dict) else None


def wait_for_safe_temp(jetson):
    for attempt in range(THERMAL_MAX_RETRIES):
        t = get_temp_c(jetson)
        if t is None or t < THERMAL_LIMIT_C:
            return t
        print(f"  THERMAL GUARD: {t:.1f}C >= {THERMAL_LIMIT_C}C -- cooling down "
              f"{THERMAL_COOLDOWN_S}s (attempt {attempt + 1}/{THERMAL_MAX_RETRIES})...")
        time.sleep(THERMAL_COOLDOWN_S)
    raise RuntimeError(
        f"THERMAL GUARD: still >= {THERMAL_LIMIT_C}C after {THERMAL_MAX_RETRIES} cooldown attempts "
        "-- aborting rather than reporting a throttled measurement. Check airflow/heatsink."
    )


def run_engine_timed(engine_path: str, jetson):
    """Loads and executes a serialized TensorRT engine WARMUP_ITERS +
    MEASURE_ITERS times, recording per-iteration wall-clock latency and
    sampling jtop power at each measured iteration. Returns
    (latencies_ms: list[float], power_samples_mw: list[float]).

    Uses torch.cuda for device memory, not pycuda -- see this file's own
    top-of-file import block for why (real bug found and fixed, 2026-07)."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, f"Failed to deserialize engine: {engine_path}"
    context = engine.create_execution_context()

    n_bindings = engine.num_bindings if hasattr(engine, "num_bindings") else engine.num_io_tensors
    host_buffers, device_buffers, bindings = [], [], []
    for i in range(n_bindings):
        name = engine.get_binding_name(i) if hasattr(engine, "get_binding_name") else engine.get_tensor_name(i)
        shape = engine.get_binding_shape(i) if hasattr(engine, "get_binding_shape") else engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_binding_dtype(i) if hasattr(engine, "get_binding_dtype")
                            else engine.get_tensor_dtype(name))
        size = int(np.prod(shape))
        host_mem = np.random.randn(size).astype(dtype)
        # torch.cuda tensor instead of pycuda's cuda.mem_alloc()/memcpy_htod() -- .data_ptr()
        # gives the raw device pointer TensorRT's positional-bindings execute_v2() expects.
        # Kept alive in device_buffers for this whole function: a bare int pointer doesn't
        # hold a Python reference, so without this the tensor -- and the CUDA memory behind
        # it -- could be garbage collected while `bindings` still points at its address.
        device_mem = torch.from_numpy(np.ascontiguousarray(host_mem)).cuda()
        host_buffers.append(host_mem)
        device_buffers.append(device_mem)
        bindings.append(int(device_mem.data_ptr()))

    for _ in range(WARMUP_ITERS):
        context.execute_v2(bindings)
    torch.cuda.synchronize()

    latencies_ms, power_samples_mw = [], []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        context.execute_v2(bindings)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        p = get_power_mw(jetson)
        if p is not None:
            power_samples_mw.append(p)

    return latencies_ms, power_samples_mw


def parse_trtexec_log(log_path: Path):
    """Cross-check only. Parses trtexec's 'GPU Compute Time' summary line
    (falls back to the top-level 'Latency' line on older trtexec
    versions). Returns (median_ms, p99_ms) or (None, None) if unparsable
    -- unparsable is not fatal, this value is a cross-check, not the LUT's
    source of truth."""
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="ignore")
    m = re.search(r"GPU Compute Time:.*?median\s*=\s*([\d.]+)\s*ms.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms", text)
    if not m:
        m = re.search(r"Latency:.*?median\s*=\s*([\d.]+)\s*ms.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-measure pairs already in latency_lut.json.")
    args = parser.parse_args()

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False -- this script needs a real CUDA device to "
        "allocate inference buffers on. If this is the Jetson itself, something is wrong "
        "with the torch install (see README.md); this script should never run anywhere else."
    )

    from jtop import jtop  # local import: only needed on-device

    # JETSON_ROOT-relative, not CWD-relative -- real latent bug found and fixed, 2026-07 (same
    # class already hit for real in jetson/int8_qdq_experiment.py -- see jetson/README.md).
    # This script is currently always run by the operator with cwd=jetson/, which happens to
    # make bare "out/..." paths resolve correctly today, but that's a fragile coincidence, not
    # a guarantee. ensure_out_dir() below already resolves JETSON_ROOT-relative regardless of
    # cwd; these reads now match that, so the whole file is internally consistent regardless
    # of where it's invoked from.
    manifest = read_json(JETSON_ROOT / "out" / "segments" / "manifest.json")
    buckets = sorted(s["n_tokens"] for s in manifest["segments"])
    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    lut_path = ensure_out_dir("lut") / "latency_lut.json"
    lut = read_json(lut_path) if lut_path.exists() else {"entries": {}}

    with jtop() as jetson:
        for n_tokens in buckets:
            for precision in PRECISIONS:
                key = f"n{n_tokens}_{precision}"
                if key in lut["entries"] and not args.force:
                    print(f"[{key}] already in latency_lut.json -- skipping (use --force to re-measure).")
                    continue

                engine_path = JETSON_ROOT / "out" / "engines" / engine_filename(n_tokens, precision)
                if not engine_path.exists():
                    hint = ("run int8_qdq_experiment.py on a HOST/Colab GPU, copy the ONNX over, "
                             "then trtexec --int8 (no --calib) on-device"
                             if precision == "int8" else "run build_engines.sh first")
                    print(f"[{key}] SKIPPING -- {engine_path} not found ({hint}).")
                    continue

                print(f"=== [{key}] ===")
                temp_before = wait_for_safe_temp(jetson)
                latencies_ms, power_mw = run_engine_timed(str(engine_path), jetson)
                temp_after = get_temp_c(jetson)

                median_ms = float(np.median(latencies_ms))
                p99_ms = float(np.percentile(latencies_ms, 99))
                mean_power_w = (float(np.mean(power_mw)) / 1000.0) if power_mw else None
                joules_per_inference = (mean_power_w * median_ms / 1000.0) if mean_power_w is not None else None

                trtexec_log = JETSON_ROOT / "out" / "logs" / trtexec_log_filename(n_tokens, precision)
                trtexec_median, trtexec_p99 = parse_trtexec_log(trtexec_log)

                lut["entries"][key] = {
                    "n_tokens": n_tokens, "precision": precision,
                    "median_ms": median_ms, "p99_ms": p99_ms,
                    "trtexec_crosscheck_median_ms": trtexec_median, "trtexec_crosscheck_p99_ms": trtexec_p99,
                    "temp_c_before": temp_before, "temp_c_after": temp_after,
                    "mean_power_w": mean_power_w, "joules_per_inference": joules_per_inference,
                    "n_measure_iters": MEASURE_ITERS, "n_warmup_iters": WARMUP_ITERS,
                    "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(f"  median={median_ms:.4f}ms p99={p99_ms:.4f}ms "
                      f"(trtexec cross-check: median={trtexec_median}, p99={trtexec_p99}) "
                      f"temp={temp_before}->{temp_after}C power={mean_power_w}W")
                write_json(lut_path, lut)  # write after every pair -- resume-safe by construction

    print(f"\nWrote {lut_path} ({len(lut['entries'])} entries).")


if __name__ == "__main__":
    main()
    # os._exit(0), not a normal return -- real bug found and fixed, 2026-07: confirmed on real
    # hardware, this process segfaults during Python's own interpreter shutdown, AFTER
    # latency_lut.json is already fully written and correct (the crash happens below the
    # "Wrote ..." print, once main() has returned normally). Root cause is almost certainly
    # native-library cleanup ordering: torch.cuda's CUDA context (see this file's own pycuda ->
    # torch.cuda fix, above) does not carry the same TensorRT-friendly teardown hooks
    # `pycuda.autoinit` used to register, so tearing down torch's CUDA context and TensorRT's
    # own C++ engine/context/runtime objects in whatever order CPython's interpreter shutdown
    # happens to collect them in can crash -- a known class of issue when mixing GPU libraries,
    # not a sign the measurement itself is wrong. Since all real work (the LUT) is already
    # safely on disk by this point, os._exit(0) skips the rest of Python's shutdown sequence
    # (atexit hooks, __del__ calls, garbage collection) entirely rather than trying to make
    # that native cleanup race go away -- no data is at risk, nothing after this point does
    # anything the LUT depends on.
    #
    # Real bug found and fixed, 2026-07 (found while running pilot_b8_measure.py, which copied
    # this exact pattern): os._exit(0) skips the C-level stdio flush a normal exit performs. An
    # SSH invocation's stdout is a pipe, not a tty, so CPython block-buffers it -- os._exit(0)
    # can tear the process down before that buffer ever reaches the pipe, silently dropping all
    # of this run's printed output even though latency_lut.json itself (a synchronous file
    # write, unaffected) is written correctly. Explicit flush closes that gap.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
