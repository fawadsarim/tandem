"""jetson/pilot_b8_measure.py -- ON-DEVICE. Batch-8 INT8 pilot, step 3 of 3 -- see
jetson/README.md's "Batch-8 INT8 pilot" section and jetson/CHARACTERIZATION.md for why this
pilot exists. Measures the two engines pilot_b8_build.sh built per bucket (FP16, explicit-QDQ
INT8) under the SAME timed-loop/thermal-guard protocol as measure_lut.py (reused directly, not
reimplemented -- see imports below), confirms the QDQ engine's per-layer graph shows fused INT8
kernels (no Reformat bracketing -- same check as the original N=197 QDQ diagnosis), and applies
the PRE-REGISTERED decision rule to the real numbers. Writes jetson/out/pilot_b8/pilot_b8_result.json
-- does not touch any registered out/ artifact (out/lut/latency_lut.json, out/replan/*, etc.).

DECISION RULE (pre-registered here, before this script has ever been run against a real number --
see jetson/README.md's "Batch-8 INT8 pilot" section, same text, for the registration copy):
  Per bucket, let rel_diff = (int8_median - fp16_median) / fp16_median (positive = INT8 slower).
    |rel_diff| <= 0.05                      -> that bucket's verdict is BORDERLINE
    rel_diff <  0 (INT8 faster, outside +/-5%) -> that bucket's verdict is GO
    rel_diff >= 0 (INT8 slower/equal, outside +/-5%) -> that bucket's verdict is NO-GO
  Overall verdict: BORDERLINE if EITHER bucket is BORDERLINE (too close to call takes
  precedence); else GO only if BOTH buckets are GO; else NO-GO.
  GO -> stop here, flag for manual review (a batch-8 full G1-HW spec would follow, explicit
  sign-off required before running it, same discipline as every other scope expansion in this
  folder). NO-GO -> CHARACTERIZATION.md's axis-domination finding closes the question on this
  device; no further batch sizes or buckets without explicit sign-off either way.

Needs the VENV's python, same reason as measure_lut.py (torch is venv-local, not system-wide).

Usage: ./venv/bin/python3 pilot_b8_measure.py
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
    print(f"FATAL: torch not importable ({e!r}). Run via './venv/bin/python3 pilot_b8_measure.py' "
          "-- same requirement as measure_lut.py.", file=sys.stderr)
    raise

WARMUP_ITERS = 20
MEASURE_ITERS = 200
BORDERLINE_REL_TOL = 0.05
PILOT_BUCKETS = (197, 69)
PRECISIONS = ("fp16", "int8_qdq")


def engine_path(n_tokens: int, precision: str) -> Path:
    return JETSON_ROOT / "out" / "pilot_b8" / f"n{n_tokens}_b8_{precision}.engine"


def layerinfo_path(n_tokens: int) -> Path:
    return JETSON_ROOT / "out" / "pilot_b8" / f"n{n_tokens}_b8_qdq_layerinfo.json"


def check_layer_fusion(n_tokens: int):
    """Confirms the QDQ engine's per-layer graph shows fused INT8 kernels, no separate
    Reformat/copy nodes bracketing them -- same manual check the original N=197 diagnosis did
    (see README.md's INT8 findings, step 2b), now applied to the batch=8 build. Schema is
    whatever this device's real trtexec --exportLayerInfo actually produces (TensorRT
    v101602 on this device, confirmed 2026-07) -- read defensively (multiple plausible key
    names), report the raw facts rather than assume a rigid schema a priori."""
    path = layerinfo_path(n_tokens)
    if not path.exists():
        return {"available": False, "note": f"{path} not found -- pilot_b8_build.sh did not run or failed."}

    data = read_json(path)
    layers = data.get("Layers", data if isinstance(data, list) else [])
    n_layers = len(layers)
    reformat_layers = [
        l for l in layers
        if "reformat" in json.dumps(l).lower()
    ]
    int8_layers = [
        l for l in layers
        if "int8" in json.dumps(l).lower()
    ]
    return {
        "available": True, "n_layers": n_layers,
        "n_reformat_layers": len(reformat_layers),
        "reformat_layer_names": [l.get("Name", "?") for l in reformat_layers],
        "n_layers_mentioning_int8": len(int8_layers),
        "fused_clean": len(reformat_layers) == 0,
        "raw_layer_names": [l.get("Name", "?") for l in layers],
    }


def decide_bucket(fp16_median: float, int8_median: float) -> str:
    rel_diff = (int8_median - fp16_median) / fp16_median
    if abs(rel_diff) <= BORDERLINE_REL_TOL:
        return "BORDERLINE"
    return "GO" if rel_diff < 0 else "NO-GO"


def overall_decision(bucket_verdicts):
    if any(v == "BORDERLINE" for v in bucket_verdicts.values()):
        return "BORDERLINE"
    if all(v == "GO" for v in bucket_verdicts.values()):
        return "GO"
    return "NO-GO"


def main():
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False -- this script needs a real CUDA device, same as "
        "measure_lut.py."
    )
    from jtop import jtop  # local import: only needed on-device, same as measure_lut.py

    pilot_dir = JETSON_ROOT / "out" / "pilot_b8"
    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    print("=== pilot_b8_measure.py: BATCH=8 INT8 pilot (GO/NO-GO) ===")

    results = {}
    with jtop() as jetson:
        for n_tokens in PILOT_BUCKETS:
            print(f"\n--- bucket n_tokens={n_tokens} ---")
            results[n_tokens] = {}
            for precision in PRECISIONS:
                eng_path = engine_path(n_tokens, precision)
                if not eng_path.exists():
                    print(f"  [{precision}] SKIPPING -- {eng_path} not found. Run pilot_b8_build.sh first.")
                    continue
                print(f"  [{precision}] measuring ({WARMUP_ITERS} warmup + {MEASURE_ITERS} iters)...")
                temp_before = wait_for_safe_temp(jetson)
                latencies_ms, power_mw = run_engine_timed(str(eng_path), jetson)
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

            if "int8_qdq" in results[n_tokens]:
                fusion = check_layer_fusion(n_tokens)
                results[n_tokens]["layer_fusion_check"] = fusion
                if fusion["available"]:
                    status = "CLEAN (no Reformat nodes)" if fusion["fused_clean"] else \
                        f"{fusion['n_reformat_layers']} Reformat node(s) found"
                    print(f"  Per-layer graph: {fusion['n_layers']} layers, {status}")

    print("\n=== DECISION RULE (pre-registered, see module docstring) ===")
    bucket_verdicts = {}
    for n_tokens in PILOT_BUCKETS:
        r = results.get(n_tokens, {})
        if "fp16" not in r or "int8_qdq" not in r:
            print(f"  N={n_tokens}: INCOMPLETE -- both fp16 and int8_qdq must be measured. Cannot decide.")
            continue
        fp16_median, int8_median = r["fp16"]["median_ms"], r["int8_qdq"]["median_ms"]
        verdict = decide_bucket(fp16_median, int8_median)
        rel_diff_pct = (int8_median - fp16_median) / fp16_median * 100.0
        bucket_verdicts[n_tokens] = verdict
        print(f"  N={n_tokens}: FP16={fp16_median:.4f}ms  INT8/QDQ={int8_median:.4f}ms  "
              f"rel_diff={rel_diff_pct:+.2f}%  -> {verdict}")

    if len(bucket_verdicts) != len(PILOT_BUCKETS):
        print("\nCannot compute an overall verdict -- not all buckets have both precisions measured.")
        overall = "INCOMPLETE"
    else:
        overall = overall_decision(bucket_verdicts)
        print(f"\nOVERALL VERDICT: {overall}")
        if overall == "GO":
            print("  -> STOP here. Flag for manual review -- a batch-8 full G1-HW spec would follow, "
                  "explicit sign-off required before running it.")
        elif overall == "NO-GO":
            print("  -> CHARACTERIZATION.md's axis-domination finding closes the question on "
                  "this device. No further batch sizes or buckets without explicit sign-off.")
        else:
            print("  -> Report both numbers for manual review; too close to call automatically.")

    result = {
        "batch_size": 8, "buckets": list(PILOT_BUCKETS), "precisions": list(PRECISIONS),
        "borderline_rel_tol": BORDERLINE_REL_TOL,
        "measurements": {str(n): results.get(n, {}) for n in PILOT_BUCKETS},
        "bucket_verdicts": {str(n): v for n, v in bucket_verdicts.items()},
        "overall_verdict": overall,
        "decision_rule_note": (
            "Pre-registered before any number was measured -- see this file's own module "
            "docstring and jetson/README.md's Batch-8 INT8 pilot section for the identical "
            "registered text."
        ),
        "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = pilot_dir / "pilot_b8_result.json"
    write_json(out_path, result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
    # Real bug found and fixed, 2026-07: os._exit(0) skips ALL of Python's normal shutdown --
    # including the C-level stdio flush a normal process exit performs. Confirmed for real: this
    # script's real result was written correctly (pilot_b8_result.json's file write is
    # synchronous, unaffected), but every print() this run produced was silently lost -- an SSH
    # invocation's stdout is a pipe, not a tty, so CPython block-buffers it, and os._exit(0)
    # tears the process down before that buffer is ever flushed to the pipe. Explicit flush here
    # guarantees the buffered output actually reaches the caller before the same segfault
    # workaround measure_lut.py uses (torch.cuda/TensorRT native cleanup ordering at interpreter
    # shutdown -- see that file's own comment) skips the rest of shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
