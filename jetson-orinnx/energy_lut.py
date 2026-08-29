"""jetson-orinnx/energy_lut.py -- ON-DEVICE. ENERGY MEASUREMENT (E-LUT), Part 1 -- see
jetson-orinnx/README.md's "Energy measurement (E-LUT)" section. Mirrors latency_lut.json's shape:
for each of the 5 committed buckets {197,165,133,101,69} x {FP16, INT8-QDQ}, using the ENGINES
ALREADY BUILT in out/engines/ (nothing new built here), runs a sustained loop (>=30s, first 5s
discarded) with concurrent power sampling (energy_instrument.py's run_sustained_loop, imported
directly -- same Part-0-validated instrument, not reimplemented).

PRIMARY METRIC: energy_per_inference_mJ = mean_power_W / throughput_per_s * 1000. Primary
because the loop's own measured throughput includes real inter-inference gaps (host-side
dispatch, memory transfer, etc.) -- it is what the device ACTUALLY sustains, not an idealized
back-to-back rate. The ALTERNATIVE (mean_power_W * median_latency_s, using the median_ms
ALREADY COMMITTED in out/lut/latency_lut.json for this exact engine -- not re-derived) assumes
zero inter-inference gap, which is not physically what happens in a real sustained loop; it is
recorded for auditability, not as the primary figure.

DYNAMIC ENERGY: dynamic_energy_per_inference_mJ = (mean_power_W - idle_power_W) / throughput *
1000, using Part 0's real measured idle baseline. Reports BOTH total and dynamic -- the gap
between them is the static-power story this whole experiment exists to measure.

Requires: out/energy/energy_instrument.json (Part 0, PASSED) to exist first -- this script
refuses to run otherwise (same "validate the instrument before trusting a number" discipline).

Usage: ./venv/bin/python3 energy_lut.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from energy_instrument import run_sustained_loop  # noqa: E402
from measure_lut import wait_for_safe_temp  # noqa: E402

try:
    import torch
except ImportError as e:
    print(f"FATAL: torch not importable ({e!r}). Run via './venv/bin/python3 energy_lut.py'.",
          file=sys.stderr)
    raise

LOOP_DURATION_S = 30.0
DISCARD_S = 5.0
BUCKETS = (197, 165, 133, 101, 69)
PRECISIONS = ("fp16", "int8")  # "int8" here means best-achievable QDQ, matching latency_lut.json's own convention


def engine_filename(n_tokens: int, precision: str) -> str:
    if precision == "int8":
        return f"n{n_tokens}_int8_qdq.engine"
    return f"n{n_tokens}_{precision}.engine"


def main():
    instrument_path = JETSON_ROOT / "out" / "energy" / "energy_instrument.json"
    assert instrument_path.exists(), (
        f"{instrument_path} not found -- run energy_instrument.py first (Part 0 must PASS "
        "before any energy_lut number is trusted)."
    )
    instrument = read_json(instrument_path)
    assert instrument.get("validation_passed"), (
        "energy_instrument.json exists but validation_passed is not True -- Part 0 did not "
        "pass. HALTING per the task's hard-stop rule."
    )
    idle_power_w = instrument["idle_baseline"]["mean_power_w"]
    print(f"=== energy_lut.py: Part 0 validated (idle={idle_power_w:.4f}W) -- proceeding ===")

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- need a real CUDA device."
    from jtop import jtop

    latency_lut_path = JETSON_ROOT / "out" / "lut" / "latency_lut.json"
    assert latency_lut_path.exists(), f"{latency_lut_path} not found -- this is the committed, registered LUT."
    latency_lut = read_json(latency_lut_path)["entries"]

    engines_dir = JETSON_ROOT / "out" / "engines"
    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    entries = {}
    with jtop() as jetson:
        for n_tokens in BUCKETS:
            for precision in PRECISIONS:
                key = f"n{n_tokens}_{precision}"
                engine_path = engines_dir / engine_filename(n_tokens, precision)
                if not engine_path.exists():
                    print(f"[{key}] SKIPPING -- {engine_path} not found (registered engine missing).")
                    continue

                print(f"\n=== [{key}] sustained loop ({LOOP_DURATION_S:.0f}s, discard {DISCARD_S:.0f}s) ===")
                wait_for_safe_temp(jetson)
                r = run_sustained_loop(str(engine_path), LOOP_DURATION_S, DISCARD_S, jetson-orinnx)

                committed_median_ms = latency_lut.get(key, {}).get("median_ms")
                energy_mj_throughput = r["mean_power_w"] / r["throughput_per_s"] * 1000.0
                energy_mj_latency = (r["mean_power_w"] * committed_median_ms) if committed_median_ms is not None else None
                dynamic_energy_mj = (r["mean_power_w"] - idle_power_w) / r["throughput_per_s"] * 1000.0

                entries[key] = {
                    "n_tokens": n_tokens, "precision": precision,
                    "mean_power_w": r["mean_power_w"], "median_power_w": r["median_power_w"],
                    "p99_power_w": r["p99_power_w"], "n_power_samples": r["n_power_samples"],
                    "achieved_poll_rate_hz": r["achieved_poll_rate_hz"],
                    "mean_temp_c": r["mean_temp_c"], "median_temp_c": r["median_temp_c"], "p99_temp_c": r["p99_temp_c"],
                    "throughput_per_s": r["throughput_per_s"], "n_inferences_kept": r["n_inferences_kept"],
                    "loop_duration_s": LOOP_DURATION_S, "discard_s": DISCARD_S,
                    "energy_per_inference_mJ": energy_mj_throughput,
                    "energy_per_inference_mJ_note": (
                        "PRIMARY -- mean_power_W / throughput_per_s * 1000. Uses this loop's own "
                        "real measured throughput (includes real inter-inference gaps)."
                    ),
                    "energy_per_inference_mJ_latency_form": energy_mj_latency,
                    "energy_per_inference_mJ_latency_form_note": (
                        "ALTERNATIVE, for auditability only -- mean_power_W * committed median "
                        "latency (out/lut/latency_lut.json, not re-derived). Assumes zero "
                        "inter-inference gap, which a real sustained loop does not have."
                    ),
                    "dynamic_energy_per_inference_mJ": dynamic_energy_mj,
                    "idle_power_w_used": idle_power_w,
                    "committed_median_latency_ms_used": committed_median_ms,
                    "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(f"  mean_power={r['mean_power_w']:.4f}W throughput={r['throughput_per_s']:.1f}/s  "
                      f"energy(primary)={energy_mj_throughput:.4f}mJ  "
                      f"energy(dynamic)={dynamic_energy_mj:.4f}mJ  temp={r['mean_temp_c']:.1f}C")

                out_dir = ensure_out_dir("energy")
                write_json(out_dir / "energy_lut.json", {
                    "entries": entries, "instrument": instrument, "idle_power_w": idle_power_w,
                })  # write after every config -- resume-safe by construction, same as measure_lut.py

    print(f"\nWrote {ensure_out_dir('energy') / 'energy_lut.json'} ({len(entries)} entries).")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
