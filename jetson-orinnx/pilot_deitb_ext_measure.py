"""jetson-orinnx/pilot_deitb_ext_measure.py -- ON-DEVICE. DeiT-B crossover characterization, step 2b-4
-- see jetson-orinnx/README.md's "DeiT-B crossover characterization" section.

GRAPH VERIFICATION FIRST, MANDATORY, BEFORE ANY TIMING -- same discipline as
pilot_deitb_measure.py: HALTS if either new bucket's INT8 engine shows Reformat-bracketed
(non-fused) GEMMs, before running a single timed iteration.

Measures the two NEW buckets (N=133, N=101), then COMBINES with the ORIGINAL, IMMUTABLE
out/pilot_deitb/pilot_deitb_result.json (N=197, N=69) -- read-only, never re-measured -- to
build the full four-bucket ratio table and fit the crossover point.

CROSSOVER FIT: simple least-squares LINEAR fit of ratio (INT8/FP16 median) vs N over the four
real measured points (not the original pilot's own naive 2-point linear interpolation, which
only used N=197/N=69 -- this fit uses all four). Reported alongside the naive 2-point estimate
for comparison, not silently replacing it.

PRE-REGISTERED DECISION RULE (written before this script has been run against real numbers --
see jetson-orinnx/README.md's own copy of the identical text): GO (full DeiT-B G1-HW) if INT8 is
faster than FP16 by MORE than 5% (ratio < 0.95) at >=2 of the four buckets -- the precision axis
is then meaningfully live in the reachable plan space. NO-GO otherwise (close device work) --
the crossover finding is reported as a measured hardware property regardless.

Needs the VENV's python, same reason as measure_lut.py.

Usage: ./venv/bin/python3 pilot_deitb_ext_measure.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import JETSON_ROOT, read_json, write_json  # noqa: E402
from measure_lut import get_power_mw, get_temp_c, run_engine_timed, wait_for_safe_temp  # noqa: E402

try:
    import torch
except ImportError as e:
    print(f"FATAL: torch not importable ({e!r}). Run via './venv/bin/python3 pilot_deitb_ext_measure.py'.",
          file=sys.stderr)
    raise

WARMUP_ITERS = 20
MEASURE_ITERS = 200
GO_THRESHOLD_REL = 0.05  # INT8 must be faster than FP16 by MORE than 5% (ratio < 0.95) to count
NEW_BUCKETS = (133, 101)
PRECISIONS = ("fp16", "int8_qdq")
PILOT_DIR = JETSON_ROOT / "out" / "pilot_deitb_ext"
ORIGINAL_RESULT_PATH = JETSON_ROOT / "out" / "pilot_deitb" / "pilot_deitb_result.json"


def engine_path(n_tokens: int, precision: str) -> Path:
    return PILOT_DIR / f"n{n_tokens}_{precision}.engine"


def layerinfo_path(n_tokens: int) -> Path:
    return PILOT_DIR / f"n{n_tokens}_qdq_layerinfo.json"


def check_layer_fusion(n_tokens: int):
    """Same defensive-parse approach as pilot_deitb_measure.py's check_layer_fusion."""
    path = layerinfo_path(n_tokens)
    if not path.exists():
        return {"available": False, "note": f"{path} not found -- pilot_deitb_ext_build.sh did not run or failed."}
    data = read_json(path)
    layers = data.get("Layers", data if isinstance(data, list) else [])
    reformat_layers = [l for l in layers if "reformat" in json.dumps(l).lower()]
    return {
        "available": True, "n_layers": len(layers),
        "n_reformat_layers": len(reformat_layers),
        "reformat_layer_names": [l.get("Name", "?") for l in reformat_layers],
        "fused_clean": len(reformat_layers) == 0,
    }


def main():
    print("=== pilot_deitb_ext_measure.py: DeiT-B crossover characterization ===")

    # --- MANDATORY GRAPH VERIFICATION, BEFORE any timing ---
    print("\n--- Graph verification (mandatory before trusting any timing) ---")
    fusion_checks = {}
    unclean = []
    for n_tokens in NEW_BUCKETS:
        fusion = check_layer_fusion(n_tokens)
        fusion_checks[n_tokens] = fusion
        assert fusion["available"], (
            f"N={n_tokens}: layer info not available ({fusion.get('note')}). "
            "Run pilot_deitb_ext_build.sh first."
        )
        status = "CLEAN (no Reformat nodes)" if fusion["fused_clean"] else \
            f"{fusion['n_reformat_layers']} Reformat node(s): {fusion['reformat_layer_names']}"
        print(f"  N={n_tokens}: {fusion['n_layers']} layers, {status}")
        if not fusion["fused_clean"]:
            unclean.append(n_tokens)
    if unclean:
        raise AssertionError(
            f"GRAPH VERIFICATION FAILED at bucket(s) {unclean}: Reformat-bracketed INT8 GEMMs "
            "found. HALTING before any timing -- do not time a crippled engine."
        )
    print("  Graph verification PASSED at both new buckets -- proceeding to timing.")

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- need a real CUDA device."
    from jtop import jtop

    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    results = {}
    with jtop() as jetson:
        for n_tokens in NEW_BUCKETS:
            print(f"\n--- bucket n_tokens={n_tokens} ---")
            results[n_tokens] = {"layer_fusion_check": fusion_checks[n_tokens]}
            for precision in PRECISIONS:
                eng_path = engine_path(n_tokens, precision)
                if not eng_path.exists():
                    print(f"  [{precision}] SKIPPING -- {eng_path} not found.")
                    continue
                print(f"  [{precision}] measuring ({WARMUP_ITERS} warmup + {MEASURE_ITERS} iters)...")
                temp_before = wait_for_safe_temp(jetson)
                latencies_ms, power_mw = run_engine_timed(str(eng_path), jetson-orinnx)
                temp_after = get_temp_c(jetson)

                median_ms = float(np.median(latencies_ms))
                p99_ms = float(np.percentile(latencies_ms, 99))
                mean_power_w = (float(np.mean(power_mw)) / 1000.0) if power_mw else None
                results[n_tokens][precision] = {
                    "median_ms": median_ms, "p99_ms": p99_ms,
                    "temp_c_before": temp_before, "temp_c_after": temp_after,
                    "mean_power_w": mean_power_w,
                    "n_measure_iters": MEASURE_ITERS, "n_warmup_iters": WARMUP_ITERS,
                }
                print(f"    median={median_ms:.4f}ms p99={p99_ms:.4f}ms temp={temp_before}->{temp_after}C")

    # --- Combine with the ORIGINAL, IMMUTABLE pilot_deitb_result.json (N=197, N=69) ---
    assert ORIGINAL_RESULT_PATH.exists(), f"{ORIGINAL_RESULT_PATH} not found -- this is the original pilot's committed result."
    original = read_json(ORIGINAL_RESULT_PATH)
    original_197 = original["measurements"]["197"]
    original_69 = original["measurements"]["69"]

    four_bucket = {
        197: {"fp16_ms": original_197["fp16"]["median_ms"], "int8_ms": original_197["int8_qdq"]["median_ms"]},
        133: {"fp16_ms": results[133]["fp16"]["median_ms"], "int8_ms": results[133]["int8_qdq"]["median_ms"]},
        101: {"fp16_ms": results[101]["fp16"]["median_ms"], "int8_ms": results[101]["int8_qdq"]["median_ms"]},
        69: {"fp16_ms": original_69["fp16"]["median_ms"], "int8_ms": original_69["int8_qdq"]["median_ms"]},
    }
    for n_tokens, d in four_bucket.items():
        d["ratio"] = d["int8_ms"] / d["fp16_ms"]
        d["rel_diff_pct"] = (d["int8_ms"] - d["fp16_ms"]) / d["fp16_ms"] * 100.0
        d["faster_by_more_than_5pct"] = d["ratio"] < (1.0 - GO_THRESHOLD_REL)

    print("\n=== Four-bucket ratio table (197, 133, 101, 69 -- 197/69 from the original, "
          "immutable pilot_deitb_result.json, NOT re-measured) ===")
    for n_tokens in (197, 133, 101, 69):
        d = four_bucket[n_tokens]
        print(f"  N={n_tokens:3d}: FP16={d['fp16_ms']:.4f}ms  INT8={d['int8_ms']:.4f}ms  "
              f"ratio={d['ratio']:.3f}x  rel_diff={d['rel_diff_pct']:+.2f}%  "
              f"faster_by_>5%={d['faster_by_more_than_5pct']}")

    # --- Crossover fit: least-squares line, ratio vs N, over all 4 real points ---
    Ns = np.array([197, 133, 101, 69], dtype=float)
    ratios = np.array([four_bucket[n]["ratio"] for n in (197, 133, 101, 69)])
    slope, intercept = np.polyfit(Ns, ratios, 1)
    crossover_4pt = (1.0 - intercept) / slope if slope != 0 else float("nan")

    # Naive 2-point linear interpolation (N=197, N=69 only) -- the original ~N=87 estimate,
    # reported for comparison, not silently replaced.
    r197, r69 = four_bucket[197]["ratio"], four_bucket[69]["ratio"]
    slope_2pt = (r69 - r197) / (69 - 197)
    crossover_2pt = 197 + (1.0 - r197) / slope_2pt

    print(f"\n=== Crossover fit ===")
    print(f"  4-point least-squares linear fit: ratio = {slope:.6f}*N + {intercept:.4f}  "
          f"-> crossover at N={crossover_4pt:.1f}")
    print(f"  Naive 2-point interpolation (N=197, N=69 only, the original pilot's own estimate): "
          f"N={crossover_2pt:.1f}")

    # --- PRE-REGISTERED DECISION RULE ---
    n_faster_by_5pct = sum(1 for d in four_bucket.values() if d["faster_by_more_than_5pct"])
    overall = "GO" if n_faster_by_5pct >= 2 else "NO-GO"
    print(f"\n=== DECISION RULE (pre-registered) ===")
    print(f"  Buckets with INT8 faster than FP16 by MORE than 5%: {n_faster_by_5pct}/4")
    print(f"  OVERALL VERDICT: {overall}")
    if overall == "GO":
        print("  -> The precision axis is meaningfully live in the reachable plan space. "
              "Flag for manual review -- full DeiT-B G1-HW spec follows.")
    else:
        print("  -> Close device work. Axis domination holds WITHIN each model's reachable "
              "plan space; the crossover mechanism is identified but the profitable region is "
              "too narrow to exploit on this device. CHARACTERIZATION.md to be updated accordingly.")

    result = {
        "model_name": "deit_base_patch16_224", "d_model": 768, "batch_size": 1,
        "buckets": [197, 133, 101, 69],
        "four_bucket_table": {str(n): four_bucket[n] for n in (197, 133, 101, 69)},
        "new_bucket_measurements": {str(n): results.get(n, {}) for n in NEW_BUCKETS},
        "original_pilot_source_note": (
            "N=197 and N=69 are copied read-only from out/pilot_deitb/pilot_deitb_result.json "
            "(the original, committed, immutable pilot result) -- NOT re-measured here."
        ),
        "crossover_fit_4pt": {"slope": float(slope), "intercept": float(intercept), "crossover_n": float(crossover_4pt)},
        "crossover_naive_2pt": {"crossover_n": float(crossover_2pt)},
        "n_buckets_faster_by_more_than_5pct": n_faster_by_5pct,
        "go_threshold_rel": GO_THRESHOLD_REL,
        "overall_verdict": overall,
        "decision_rule_note": (
            "Pre-registered before any number was measured -- GO if INT8 faster than FP16 by "
            "MORE than 5% at >=2 of the four buckets; NO-GO otherwise. See this file's own "
            "module docstring and jetson-orinnx/README.md for the identical registered text."
        ),
        "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = PILOT_DIR / "pilot_deitb_ext_result.json"
    write_json(out_path, result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
