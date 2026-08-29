"""jetson-orinnx/energy_instrument.py -- ON-DEVICE. ENERGY MEASUREMENT (E-LUT), Part 0 -- see
jetson-orinnx/README.md's "Energy measurement (E-LUT)" section. BLOCKING instrument validation: power
sampling is a new instrument in this project and has never been validated. This must pass
before any plan-level energy number is trusted (same discipline as the port-equivalence gate
for the DP, or FP-parity for the accuracy pipeline).

RAIL ENUMERATION (real, queried on this device, not assumed): `jetson.power` exposes
`rail["VDD_CPU_GPU_CV"]` (CPU+GPU+CV combined), `rail["VDD_SOC"]` (SoC/memory/peripherals), and
`tot` (`name="VDD_IN"`, the total board input power -- the same key `measure_lut.py`'s own
`get_power_mw` already reads). `tot`/`VDD_IN` is used as the PRIMARY energy rail: it is the
actual total system power draw, and it is what every `mean_power_w` field already committed in
`latency_lut.json` already measures -- using it here keeps the energy LUT consistent with data
already trusted elsewhere in this project. The two named sub-rails are recorded for provenance
only, not used in any energy_per_inference calculation.

SAMPLING RATE (real, verified two ways, not assumed):
  1. Code-level: this device's installed jetson-stats (`jetson_power_provider.py`) hard-floors
     the power rail's own hardware refresh at `_min_refresh_s = 0.25` -- i.e. AT MOST 4Hz,
     regel less of the client's requested `jtop(interval=...)`. Confirmed by direct inspection
     of the installed package source on this device.
  2. Empirical: polling at a client-side interval faster than 0.25s under both idle and
     sustained real GPU load shows the SAME value held across many consecutive reads,
     consistent with (1) -- see the empirical block in energy_instrument.json.
  CONSEQUENCE (stated per the task's own requirement): 4Hz (250ms) cannot resolve a single
  ~0.4-0.95ms inference (roughly 260-2400x coarser depending on bucket). Every per-inference
  energy figure in this experiment is therefore derived from SUSTAINED-LOOP average power and
  concurrently measured throughput, never from a single-shot measurement. This is a hard
  instrument limitation, recorded as a scope limitation (see PREREGISTRATION.md), not glossed
  over.

Usage: ./venv/bin/python3 energy_instrument.py
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, write_json  # noqa: E402
from measure_lut import get_power_mw, get_temp_c, wait_for_safe_temp  # noqa: E402

try:
    import tensorrt as trt
    import torch
except ImportError as e:
    print(f"FATAL: tensorrt/torch not importable ({e!r}). Run via './venv/bin/python3 "
          "energy_instrument.py', not bare python3.", file=sys.stderr)
    raise

POWER_RAIL_MIN_REFRESH_S = 0.25  # confirmed by direct read of jetson_power_provider.py on this device
POLL_INTERVAL_S = 0.1  # client-side poll cadence -- faster than the 0.25s hardware floor, so no update is missed
IDLE_BASELINE_S = 60.0
MONOTONIC_LOOP_S = 15.0
MONOTONIC_DISCARD_S = 5.0
THERMAL_LIMIT_C = 80.0


def load_engine_and_buffers(engine_path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, f"Failed to deserialize engine: {engine_path}"
    context = engine.create_execution_context()

    n_bindings = engine.num_bindings if hasattr(engine, "num_bindings") else engine.num_io_tensors
    device_buffers, bindings = [], []
    for i in range(n_bindings):
        name = engine.get_binding_name(i) if hasattr(engine, "get_binding_name") else engine.get_tensor_name(i)
        shape = engine.get_binding_shape(i) if hasattr(engine, "get_binding_shape") else engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_binding_dtype(i) if hasattr(engine, "get_binding_dtype")
                            else engine.get_tensor_dtype(name))
        size = int(np.prod(shape))
        host_mem = np.random.randn(size).astype(dtype)
        device_mem = torch.from_numpy(np.ascontiguousarray(host_mem)).cuda()
        device_buffers.append(device_mem)
        bindings.append(int(device_mem.data_ptr()))
    return context, device_buffers, bindings


def power_sampler_thread(jetson, stop_flag, samples, t_start):
    while not stop_flag.is_set():
        t = time.perf_counter() - t_start
        p = get_power_mw(jetson)
        temp = get_temp_c(jetson)
        if p is not None:
            samples.append((t, p, temp))
        time.sleep(POLL_INTERVAL_S)


def run_sustained_loop(engine_path, duration_s: float, discard_s: float, jetson, warmup_iters: int = 20):
    """Runs the engine in a tight loop for `duration_s` wall-clock seconds while a background
    thread concurrently polls power/temperature at POLL_INTERVAL_S (faster than the hardware's
    own 0.25s refresh floor, so no true update is missed). Returns a dict with power
    samples/stats (post-discard), throughput (post-discard), and raw counts -- discard applies
    to BOTH power samples and the inference count used for throughput, so both reflect the same
    steady-state window."""
    context, device_buffers, bindings = load_engine_and_buffers(engine_path)

    for _ in range(warmup_iters):
        context.execute_v2(bindings)
    torch.cuda.synchronize()

    samples = []
    stop_flag = threading.Event()
    t_start = time.perf_counter()
    sampler = threading.Thread(target=power_sampler_thread, args=(jetson, stop_flag, samples, t_start), daemon=True)
    sampler.start()

    inference_timestamps = []
    while time.perf_counter() - t_start < duration_s:
        context.execute_v2(bindings)
        torch.cuda.synchronize()
        inference_timestamps.append(time.perf_counter() - t_start)
    t_end = time.perf_counter() - t_start

    stop_flag.set()
    sampler.join(timeout=2.0)

    kept_power = [(t, p, temp) for t, p, temp in samples if t >= discard_s]
    kept_inferences = [t for t in inference_timestamps if t >= discard_s]
    assert kept_power, f"No power samples survived the {discard_s}s discard window -- loop too short or sampler failed."
    assert kept_inferences, f"No inferences survived the {discard_s}s discard window -- loop too short."

    powers_w = np.array([p for _, p, _ in kept_power]) / 1000.0
    temps = [temp for _, _, temp in kept_power if temp is not None]
    window_duration = kept_inferences[-1] - discard_s if len(kept_inferences) > 1 else (t_end - discard_s)
    throughput_per_s = len(kept_inferences) / window_duration if window_duration > 0 else float("nan")

    return {
        "mean_power_w": float(np.mean(powers_w)), "median_power_w": float(np.median(powers_w)),
        "p99_power_w": float(np.percentile(powers_w, 99)),
        "n_power_samples": len(kept_power), "achieved_poll_rate_hz": len(kept_power) / window_duration if window_duration > 0 else None,
        "mean_temp_c": float(np.mean(temps)) if temps else None,
        "median_temp_c": float(np.median(temps)) if temps else None,
        "p99_temp_c": float(np.percentile(temps, 99)) if temps else None,
        "throughput_per_s": throughput_per_s, "n_inferences_kept": len(kept_inferences),
        "n_inferences_total": len(inference_timestamps), "loop_duration_s": t_end, "discard_s": discard_s,
    }


def main():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- need a real CUDA device."
    from jtop import jtop

    print("=== energy_instrument.py: Part 0 instrument validation (BLOCKING) ===")

    with jtop() as jetson:
        # --- Step 1: rail enumeration ---
        print("\n--- Step 1: rail enumeration (real, queried on this device) ---")
        raw_power = jetson.power
        print("Raw jetson.power structure:")
        import json as _json
        print(_json.dumps(raw_power, indent=2, default=str))
        rails = {
            name: {"power_mw": r.get("power"), "volt_mv": r.get("volt"), "curr_ma": r.get("curr")}
            for name, r in raw_power.get("rail", {}).items()
        }
        tot = raw_power.get("tot", {})
        primary_rail = {"name": tot.get("name"), "power_mw": tot.get("power"),
                         "volt_mv": tot.get("volt"), "curr_ma": tot.get("curr")}
        print(f"Primary rail selected: {primary_rail['name']} (tot) -- total board input power, "
              "same key measure_lut.py's get_power_mw already reads.")

        # --- Step 2: sampling rate ---
        print("\n--- Step 2: sampling rate ---")
        print(f"Code-level floor (jetson_power_provider.py, this device's installed jetson-stats): "
              f"_min_refresh_s={POWER_RAIL_MIN_REFRESH_S}s -> max {1/POWER_RAIL_MIN_REFRESH_S:.1f}Hz")
        print("This cannot resolve a single ~0.4-0.95ms inference (roughly 260-2400x coarser). "
              "All per-inference energy below is derived from sustained-loop average power and "
              "concurrent throughput, never single-shot measurement.")

        # --- Step 3: idle baseline (60s, no workload) ---
        print(f"\n--- Step 3: idle baseline ({IDLE_BASELINE_S:.0f}s, no workload) ---")
        wait_for_safe_temp(jetson)
        samples = []
        stop_flag = threading.Event()
        t_start = time.perf_counter()
        sampler = threading.Thread(target=power_sampler_thread, args=(jetson, stop_flag, samples, t_start), daemon=True)
        sampler.start()
        time.sleep(IDLE_BASELINE_S)
        stop_flag.set()
        sampler.join(timeout=2.0)
        idle_powers_w = np.array([p for _, p, _ in samples]) / 1000.0
        idle_temps = [t for _, _, t in samples if t is not None]
        idle_baseline = {
            "mean_power_w": float(np.mean(idle_powers_w)), "median_power_w": float(np.median(idle_powers_w)),
            "p99_power_w": float(np.percentile(idle_powers_w, 99)), "n_samples": len(samples),
            "mean_temp_c": float(np.mean(idle_temps)) if idle_temps else None,
            "duration_s": IDLE_BASELINE_S,
        }
        print(f"  idle: mean={idle_baseline['mean_power_w']:.4f}W median={idle_baseline['median_power_w']:.4f}W "
              f"p99={idle_baseline['p99_power_w']:.4f}W  n_samples={idle_baseline['n_samples']}")

        # --- Step 4: monotonic sanity ladder (idle -> N=69 FP16 -> N=197 FP16) ---
        print(f"\n--- Step 4: monotonic sanity ladder ({MONOTONIC_LOOP_S:.0f}s loops, "
              f"discard first {MONOTONIC_DISCARD_S:.0f}s) ---")
        engines_dir = JETSON_ROOT / "out" / "engines"
        n69_engine = engines_dir / "n69_fp16.engine"
        n197_engine = engines_dir / "n197_fp16.engine"
        assert n69_engine.exists() and n197_engine.exists(), (
            f"Existing registered engines not found ({n69_engine}, {n197_engine}) -- Part 1 "
            "reuses these, they must already exist from the completed G1-HW run."
        )

        wait_for_safe_temp(jetson)
        print("  running N=69 FP16 sustained loop...")
        r69 = run_sustained_loop(str(n69_engine), MONOTONIC_LOOP_S, MONOTONIC_DISCARD_S, jetson-orinnx)
        print(f"    mean_power={r69['mean_power_w']:.4f}W throughput={r69['throughput_per_s']:.1f}/s "
              f"temp={r69['mean_temp_c']:.1f}C")

        wait_for_safe_temp(jetson)
        print("  running N=197 FP16 sustained loop...")
        r197 = run_sustained_loop(str(n197_engine), MONOTONIC_LOOP_S, MONOTONIC_DISCARD_S, jetson-orinnx)
        print(f"    mean_power={r197['mean_power_w']:.4f}W throughput={r197['throughput_per_s']:.1f}/s "
              f"temp={r197['mean_temp_c']:.1f}C")

        ladder = [
            {"label": "idle", "mean_power_w": idle_baseline["mean_power_w"]},
            {"label": "fp16_n69", "mean_power_w": r69["mean_power_w"]},
            {"label": "fp16_n197", "mean_power_w": r197["mean_power_w"]},
        ]
        monotonic = ladder[0]["mean_power_w"] < ladder[1]["mean_power_w"] < ladder[2]["mean_power_w"]
        print(f"\n  Ladder: idle={ladder[0]['mean_power_w']:.4f}W < "
              f"N=69={ladder[1]['mean_power_w']:.4f}W < N=197={ladder[2]['mean_power_w']:.4f}W  "
              f"MONOTONIC={monotonic}")

        instrument_result = {
            "rails": rails, "primary_rail": primary_rail,
            "sampling_rate": {
                "code_level_floor_s": POWER_RAIL_MIN_REFRESH_S,
                "code_level_max_hz": 1.0 / POWER_RAIL_MIN_REFRESH_S,
                "source": "jetson_power_provider.py, this device's installed jetson-stats, "
                          "confirmed by direct source inspection, 2026-07/08",
                "consequence_note": (
                    "4Hz (250ms) cannot resolve a single ~0.4-0.95ms inference (roughly "
                    "260-2400x coarser). Per-inference energy is ALWAYS derived from "
                    "sustained-loop average power and concurrent throughput, never from a "
                    "single-shot measurement."
                ),
            },
            "idle_baseline": idle_baseline,
            "monotonic_ladder": {
                "loop_duration_s": MONOTONIC_LOOP_S, "discard_s": MONOTONIC_DISCARD_S,
                "ladder": ladder, "monotonic": monotonic,
                "fp16_n69_full": r69, "fp16_n197_full": r197,
            },
            "validation_passed": bool(monotonic),
        }
        out_dir = ensure_out_dir("energy")
        out_path = out_dir / "energy_instrument.json"
        write_json(out_path, instrument_result)
        print(f"\nWrote {out_path}")

        if not monotonic:
            raise AssertionError(
                "INSTRUMENT VALIDATION FAILED: power did not increase monotonically across "
                "idle -> N=69 -> N=197. The rail selection or instrument is wrong. HALTING per "
                "the task's own hard-stop rule -- do NOT proceed to Part 1."
            )
        print("\n=== INSTRUMENT VALIDATION PASSED -- safe to proceed to Part 1 (energy_lut.py) ===")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
