"""jetson/pilot_deitb_measure.py -- ON-DEVICE. DeiT-B INT8 pilot, Part 2 (steps 7-9) -- see
jetson/README.md's "DeiT-B INT8 pilot" section and jetson/out/pilot_deitb/PREREGISTRATION.md.

GRAPH VERIFICATION FIRST, MANDATORY, BEFORE ANY TIMING (task's own explicit ordering): parses
both buckets' INT8 per-layer graph dumps (pilot_deitb_build.sh's --exportLayerInfo output) and
HALTS -- raises, before running a single timed iteration -- if either shows Reformat-bracketed
(non-fused) INT8 GEMMs. A crippled engine's timing would be meaningless for the pre-registered
decision rule, so this is checked before trusting any number, not alongside it.

Reuses measure_lut.py's run_engine_timed/wait_for_safe_temp/get_temp_c/get_power_mw directly
(same timed-loop/thermal-guard protocol, not reimplemented) -- same jetson-internal-import
convention as pilot_b8_measure.py.

Needs the VENV's python, same reason as measure_lut.py (torch is venv-local).

Usage: ./venv/bin/python3 pilot_deitb_measure.py
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
    print(f"FATAL: torch not importable ({e!r}). Run via './venv/bin/python3 pilot_deitb_measure.py'.",
          file=sys.stderr)
    raise

WARMUP_ITERS = 20
MEASURE_ITERS = 200
BORDERLINE_REL_TOL = 0.05
PILOT_BUCKETS = (197, 69)
PRECISIONS = ("fp16", "int8_qdq")
PILOT_DIR = JETSON_ROOT / "out" / "pilot_deitb"


def engine_path(n_tokens: int, precision: str) -> Path:
    return PILOT_DIR / f"n{n_tokens}_{precision}.engine"


def layerinfo_path(n_tokens: int) -> Path:
    return PILOT_DIR / f"n{n_tokens}_qdq_layerinfo.json"


def check_layer_fusion(n_tokens: int):
    """Same defensive-parse approach as pilot_b8_measure.py's check_layer_fusion (schema is
    whatever this device's real trtexec --exportLayerInfo actually produces, TensorRT v101602,
    already confirmed real on this device during the batch-8 pilot)."""
    path = layerinfo_path(n_tokens)
    if not path.exists():
        return {"available": False, "note": f"{path} not found -- pilot_deitb_build.sh did not run or failed."}
    data = read_json(path)
    layers = data.get("Layers", data if isinstance(data, list) else [])
    reformat_layers = [l for l in layers if "reformat" in json.dumps(l).lower()]
    int8_layers = [l for l in layers if "int8" in json.dumps(l).lower()]
    return {
        "available": True, "n_layers": len(layers),
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
    print("=== pilot_deitb_measure.py: DeiT-B INT8 pilot (GO/NO-GO) ===")

    # --- MANDATORY GRAPH VERIFICATION, BEFORE any timing (see module docstring) ---
    print("\n--- Graph verification (mandatory before trusting any timing) ---")
    fusion_checks = {}
    unclean = []
    for n_tokens in PILOT_BUCKETS:
        fusion = check_layer_fusion(n_tokens)
        fusion_checks[n_tokens] = fusion
        assert fusion["available"], (
            f"N={n_tokens}: layer info not available ({fusion.get('note')}) -- cannot verify "
            "the INT8 engine is clean. Run pilot_deitb_build.sh first."
        )
        status = "CLEAN (no Reformat nodes)" if fusion["fused_clean"] else \
            f"{fusion['n_reformat_layers']} Reformat node(s): {fusion['reformat_layer_names']}"
        print(f"  N={n_tokens}: {fusion['n_layers']} layers, {status}")
        if not fusion["fused_clean"]:
            unclean.append(n_tokens)

    if unclean:
        raise AssertionError(
            f"GRAPH VERIFICATION FAILED at bucket(s) {unclean}: Reformat-bracketed (non-fused) "
            "INT8 GEMMs found -- the INT8 engine is not clean and the FP16-vs-INT8 comparison "
            "would be invalid at this bucket. HALTING before any timing -- per task spec, do "
            "not time a crippled engine. Fix the QDQ export/build and re-run from "
            "pilot_deitb_build.sh."
        )
    print("  Graph verification PASSED at both buckets -- proceeding to timing.")

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False -- this script needs a real CUDA device."
    )
    from jtop import jtop  # local import: only needed on-device

    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    export_manifest_path = PILOT_DIR / "pilot_deitb_export_manifest.json"
    export_manifest = read_json(export_manifest_path) if export_manifest_path.exists() else {}

    results = {}
    with jtop() as jetson:
        for n_tokens in PILOT_BUCKETS:
            print(f"\n--- bucket n_tokens={n_tokens} ---")
            results[n_tokens] = {"layer_fusion_check": fusion_checks[n_tokens]}
            for precision in PRECISIONS:
                eng_path = engine_path(n_tokens, precision)
                if not eng_path.exists():
                    print(f"  [{precision}] SKIPPING -- {eng_path} not found. Run pilot_deitb_build.sh first.")
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

    print("\n=== DECISION RULE (pre-registered in PREREGISTRATION.md, see module docstring) ===")
    bucket_verdicts = {}
    for n_tokens in PILOT_BUCKETS:
        r = results.get(n_tokens, {})
        if "fp16" not in r or "int8_qdq" not in r:
            print(f"  N={n_tokens}: INCOMPLETE -- both fp16 and int8_qdq must be measured. Cannot decide.")
            continue
        fp16_median, int8_median = r["fp16"]["median_ms"], r["int8_qdq"]["median_ms"]
        verdict = decide_bucket(fp16_median, int8_median)
        rel_diff_pct = (int8_median - fp16_median) / fp16_median * 100.0
        ratio = int8_median / fp16_median
        bucket_verdicts[n_tokens] = verdict
        print(f"  N={n_tokens}: FP16={fp16_median:.4f}ms  INT8/QDQ={int8_median:.4f}ms  "
              f"ratio={ratio:.3f}x  rel_diff={rel_diff_pct:+.2f}%  -> {verdict}")

    if len(bucket_verdicts) != len(PILOT_BUCKETS):
        print("\nCannot compute an overall verdict -- not all buckets have both precisions measured.")
        overall = "INCOMPLETE"
    else:
        overall = overall_decision(bucket_verdicts)
        print(f"\nOVERALL VERDICT: {overall}")
        if overall == "GO":
            print("  -> The precision axis is live on DeiT-B. STOP here. Flag for manual review, "
                  "to spec a full DeiT-B G1-HW run.")
        elif overall == "NO-GO":
            print("  -> Axis domination confirmed on a second model scale. CHARACTERIZATION.md "
                  "will be updated with the DeiT-B evidence; device work is CLOSED.")
        else:
            print("  -> Report both numbers for manual review; too close to call automatically.")

    result = {
        "model_name": "deit_base_patch16_224", "d_model": 768, "n_blocks": 12,
        "checkpoint_hash": export_manifest.get("checkpoint_state_hash"),
        "calibration_hash": (export_manifest.get("calibration") or {}).get("calib_hash"),
        "batch_size": 1, "buckets": list(PILOT_BUCKETS), "precisions": list(PRECISIONS),
        "borderline_rel_tol": BORDERLINE_REL_TOL,
        "measurements": {str(n): results.get(n, {}) for n in PILOT_BUCKETS},
        "bucket_verdicts": {str(n): v for n, v in bucket_verdicts.items()},
        "overall_verdict": overall,
        "decision_rule_note": (
            "Pre-registered in jetson/out/pilot_deitb/PREREGISTRATION.md before any number was "
            "measured -- see this file's own module docstring for the identical registered text."
        ),
        "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = PILOT_DIR / "pilot_deitb_result.json"
    write_json(out_path, result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()  # see measure_lut.py's own comment on why this precedes os._exit(0)
    sys.stderr.flush()
    import os
    os._exit(0)
