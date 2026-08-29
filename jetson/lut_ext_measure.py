"""jetson/lut_ext_measure.py -- ON-DEVICE. LUT EXTENSION, step 2b -- see jetson/README.md's
"LUT extension" section. Measures the 3 new FP16 engines (N=53/37/21) under the EXACT same
timed-loop/thermal-guard protocol as measure_lut.py (run_engine_timed/wait_for_safe_temp/
get_temp_c/get_power_mw imported directly, not reimplemented), then merges them with the
ORIGINAL 5-bucket latency_lut.json's entries -- copied verbatim, asserted byte-identical
(serialized-JSON comparison), NEVER re-measured -- into latency_lut_extended.json.

FP16 ONLY -- INT8 is CLOSED per CHARACTERIZATION.md; no int8/int8_naive_implicit entries exist
or are expected for the 3 new buckets.

Needs the VENV's python, same reason as measure_lut.py (torch is venv-local).

Usage: ./venv/bin/python3 lut_ext_measure.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from measure_lut import get_power_mw, get_temp_c, run_engine_timed, wait_for_safe_temp  # noqa: E402

try:
    import torch
except ImportError as e:
    print(f"FATAL: torch not importable ({e!r}). Run via './venv/bin/python3 lut_ext_measure.py' "
          "-- same requirement as measure_lut.py.", file=sys.stderr)
    raise

WARMUP_ITERS = 20
MEASURE_ITERS = 200
NEW_BUCKETS = (53, 37, 21)


def main():
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False -- this script needs a real CUDA device, same as "
        "measure_lut.py."
    )
    from jtop import jtop  # local import: only needed on-device, same as measure_lut.py

    # --- Load the ORIGINAL, registered latency_lut.json -- READ-ONLY, copied verbatim below,
    # never re-measured. This is the immutability guarantee the task itself requires.
    original_lut_path = JETSON_ROOT / "out" / "lut" / "latency_lut.json"
    assert original_lut_path.exists(), f"{original_lut_path} not found -- this is the registered, committed LUT."
    original_lut = read_json(original_lut_path)
    original_entries_json = json.dumps(original_lut["entries"], sort_keys=True)  # for the byte-identical assert below

    lut_ext_dir = JETSON_ROOT / "out" / "lut_ext"
    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    print("=== lut_ext_measure.py: FP16-only timed measurement, N=53/37/21 ===")
    print(f"Loaded {len(original_lut['entries'])} original entries from {original_lut_path} "
          "(read-only -- copied verbatim into the extended LUT, never re-measured).")

    new_entries = {}
    with jtop() as jetson:
        for n_tokens in NEW_BUCKETS:
            key = f"n{n_tokens}_fp16"
            engine_path = lut_ext_dir / "engines" / f"n{n_tokens}_fp16.engine"
            if not engine_path.exists():
                print(f"[{key}] SKIPPING -- {engine_path} not found (run lut_ext_build.sh first).")
                continue

            print(f"=== [{key}] ===")
            temp_before = wait_for_safe_temp(jetson)
            latencies_ms, power_mw = run_engine_timed(str(engine_path), jetson)
            temp_after = get_temp_c(jetson)

            median_ms = float(np.median(latencies_ms))
            p99_ms = float(np.percentile(latencies_ms, 99))
            mean_power_w = (float(np.mean(power_mw)) / 1000.0) if power_mw else None
            joules_per_inference = (mean_power_w * median_ms / 1000.0) if mean_power_w is not None else None

            new_entries[key] = {
                "n_tokens": n_tokens, "precision": "fp16",
                "median_ms": median_ms, "p99_ms": p99_ms,
                "temp_c_before": temp_before, "temp_c_after": temp_after,
                "mean_power_w": mean_power_w, "joules_per_inference": joules_per_inference,
                "n_measure_iters": MEASURE_ITERS, "n_warmup_iters": WARMUP_ITERS,
                "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            print(f"  median={median_ms:.4f}ms p99={p99_ms:.4f}ms temp={temp_before}->{temp_after}C "
                  f"power={mean_power_w}W")

    # --- Build the extended LUT: original entries copied verbatim + the 3 new FP16 entries.
    extended_entries = dict(original_lut["entries"])  # copy, not mutate the loaded dict in place
    extended_entries.update(new_entries)

    # --- BYTE-IDENTICAL ASSERT (task's own immutability requirement): the original 5 buckets'
    # entries must be untouched by this merge -- re-serialize the ORIGINAL-KEYS-ONLY subset of
    # the extended dict and compare against the original file's own serialization.
    extended_original_subset = {k: v for k, v in extended_entries.items() if k in original_lut["entries"]}
    assert json.dumps(extended_original_subset, sort_keys=True) == original_entries_json, (
        "BYTE-IDENTICAL CHECK FAILED: the extended LUT's copy of the original 5-bucket entries "
        "differs from latency_lut.json -- this should be structurally impossible (dict copy, "
        "no mutation) -- real bug, halting rather than silently writing drifted data."
    )
    print(f"\nByte-identical check PASSED: all {len(original_lut['entries'])} original entries "
          "unchanged in the extended LUT.")

    out_dir = ensure_out_dir("lut_ext")
    extended_lut = {
        "entries": extended_entries,
        "extension_note": (
            "N=53/37/21 added, FP16 only -- INT8 is CLOSED per CHARACTERIZATION.md (batch-1 "
            "and batch-8 NO-GO). The original 5 buckets' entries (fp16, int8, "
            "int8_naive_implicit) are copied verbatim from out/lut/latency_lut.json, asserted "
            "byte-identical at write time -- never re-measured, no drift."
        ),
        "new_buckets": list(NEW_BUCKETS), "original_buckets": [69, 101, 133, 165, 197],
    }
    out_path = out_dir / "latency_lut_extended.json"
    write_json(out_path, extended_lut)
    print(f"Wrote {out_path} ({len(extended_entries)} total entries: "
          f"{len(original_lut['entries'])} original + {len(new_entries)} new).")


if __name__ == "__main__":
    main()
    sys.stdout.flush()  # see measure_lut.py's own comment on why this precedes os._exit(0)
    sys.stderr.flush()
    import os
    os._exit(0)
