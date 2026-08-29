"""jetson/powersweep_measure.py -- ON-DEVICE. POWER-MODE SWEEP, Parts 1-2. Re-measures FP16 and
INT8 (best-achievable QDQ) median/p99 latency at the 5 committed buckets, AT WHATEVER nvpmodel
POWER MODE IS CURRENTLY ACTIVE on this device -- this script never switches modes itself (that
is an operator action per the task's PROTOCOL PER MODE: nvpmodel -m <id> -> reboot if required ->
jetson_clocks -> verify -> idle 60s -> only then run this script). It refuses to run if the
queried current mode doesn't match --mode-label, so a forgotten mode-switch cannot silently
mislabel a measurement.

REUSES EXISTING ENGINES ONLY (out/engines/*.engine, built once under MAXN_SUPER) -- nothing is
rebuilt here, per the task's own "reuse the EXISTING engines, build nothing" rule. TensorRT
engines are tied to GPU compute capability + TensorRT/CUDA version, not to nvpmodel's clock/power
state, so they are expected to deserialize identically at every mode; if deserialization ever
fails under a given mode, that specific (bucket, precision) entry is recorded with
rebuild_needed=true and skipped, not silently dropped or worked around.

REUSED, NOT REIMPLEMENTED: run_engine_timed, wait_for_safe_temp, get_temp_c, get_power_mw,
engine_filename, WARMUP_ITERS, MEASURE_ITERS, THERMAL_* constants -- all imported directly from
measure_lut.py, the SAME instrument this project's registered latency_lut.json was built with.
Using the identical instrument is what makes Part 1's reproduction-gate comparison meaningful
(a delta would reflect the power mode, not a different measurement method).

At MAXN_SUPER, this script's own output IS Part 1's reproduction-gate input (see
powersweep_repro_gate.py) -- MAXN_SUPER is simply one of the swept modes, measured with the
exact same protocol as the two lower ones.

Writes out/powersweep/latency_lut_<mode_label>.json (mirrors latency_lut.json's entry schema,
plus nvpmodel_state/jetson_clocks_state/mode_label on every entry, per the task's own
"Record nvpmodel state and jetson_clocks state with every row" requirement).

Usage: ./venv/bin/python3 powersweep_measure.py --mode-label MAXN_SUPER [--force]
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from measure_lut import (  # noqa: E402
    engine_filename, get_power_mw, get_temp_c, MEASURE_ITERS, run_engine_timed,
    wait_for_safe_temp, WARMUP_ITERS,
)

try:
    import tensorrt as trt  # noqa: F401  -- imported by measure_lut's run_engine_timed
    import torch
except ImportError as e:
    print(f"FATAL: tensorrt/torch not importable ({e!r}). Run via "
          "'./venv/bin/python3 powersweep_measure.py', not bare python3.", file=sys.stderr)
    raise

BUCKETS = (197, 165, 133, 101, 69)
PRECISIONS = ("fp16", "int8")  # best-achievable QDQ int8, matching every other LUT in this project
IDLE_STEADY_STATE_S = 60.0


def run(cmd, input_text=None):
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True)


def query_nvpmodel_state():
    r = run(["sudo", "-n", "nvpmodel", "-q", "--verbose"])
    text = r.stdout + r.stderr
    m = re.search(r"Current mode: NV Power Mode:\s*(\S+)\s*\n(\d+)", text)
    assert m, f"could not parse `nvpmodel -q --verbose` output:\n{text}"
    return {"name": m.group(1), "id": int(m.group(2)), "raw": text}


def query_jetson_clocks_state():
    r = run(["sudo", "-n", "jetson_clocks", "--show"])
    assert r.returncode == 0, (
        f"`sudo jetson_clocks --show` failed (exit {r.returncode}): {r.stderr}\n"
        "Passwordless sudo may not be configured for this user -- see self_test.py check 3."
    )
    return r.stdout


def clocks_are_locked(jetson_clocks_text):
    """Same parsing logic as self_test.py's check_clocks_locked: every rail line with
    Min/Max/CurrentFreq must have all three equal (locked), not just present."""
    lines = [l for l in jetson_clocks_text.splitlines()
             if "MinFreq" in l and "MaxFreq" in l and "CurrentFreq" in l]
    if not lines:
        return False, "could not parse any Min/Max/CurrentFreq line"
    for l in lines:
        freqs = re.findall(r"(?:Min|Max|Current)Freq=(\d+)", l)
        if len(set(freqs)) > 1:
            return False, f"unlocked rail: {l.strip()}"
    return True, f"{len(lines)} rail(s) locked"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-label", required=True,
                         help="Expected nvpmodel mode name (e.g. MAXN_SUPER, 25W, 15W). Must "
                              "match the ACTUAL currently-active mode or this script refuses to run.")
    parser.add_argument("--force", action="store_true", help="Re-measure pairs already present.")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- need a real CUDA device."

    nvpmodel_state = query_nvpmodel_state()
    assert nvpmodel_state["name"] == args.mode_label, (
        f"REFUSING TO MEASURE: queried current mode is {nvpmodel_state['name']!r} but "
        f"--mode-label={args.mode_label!r} was given. Switch mode first per the registered "
        "PROTOCOL PER MODE (nvpmodel -m <id> -> reboot if required -> jetson_clocks -> verify), "
        "then re-run with the correct --mode-label. A mismatched label would silently mislabel "
        "every row in this run."
    )
    print(f"Confirmed active mode: {nvpmodel_state['name']} (id={nvpmodel_state['id']})")

    jetson_clocks_text = query_jetson_clocks_state()
    locked, locked_detail = clocks_are_locked(jetson_clocks_text)
    assert locked, (
        f"REFUSING TO MEASURE: jetson_clocks --show reports an unlocked rail ({locked_detail}) -- "
        "run `sudo jetson_clocks` first, per the registered protocol. A measurement under "
        "unlocked/scaling clocks is not a valid per-mode number."
    )
    print(f"Confirmed clocks locked: {locked_detail}")
    assert f"NV Power Mode: {args.mode_label}" in jetson_clocks_text, (
        f"jetson_clocks --show's own reported power mode does not mention {args.mode_label!r}:\n"
        f"{jetson_clocks_text}"
    )

    print(f"Idling {IDLE_STEADY_STATE_S:.0f}s to reach thermal steady state before measuring...")
    time.sleep(IDLE_STEADY_STATE_S)

    from jtop import jtop

    versions_path = JETSON_ROOT / "out" / "versions.json"
    versions = read_json(versions_path) if versions_path.exists() else {}

    out_dir = ensure_out_dir("powersweep")
    lut_path = out_dir / f"latency_lut_{args.mode_label}.json"
    lut = read_json(lut_path) if lut_path.exists() else {"entries": {}}

    with jtop() as jetson:
        for n_tokens in BUCKETS:
            for precision in PRECISIONS:
                key = f"n{n_tokens}_{precision}"
                if key in lut["entries"] and not args.force:
                    print(f"[{key}] already in {lut_path.name} -- skipping (use --force to re-measure).")
                    continue

                engine_path = JETSON_ROOT / "out" / "engines" / engine_filename(n_tokens, precision)
                if not engine_path.exists():
                    print(f"[{key}] SKIPPING -- {engine_path} not found (registered engine missing).")
                    continue

                print(f"=== [{key}] mode={args.mode_label} ===")
                temp_before = wait_for_safe_temp(jetson)
                try:
                    latencies_ms, power_mw = run_engine_timed(str(engine_path), jetson)
                except Exception as e:
                    print(f"  ENGINE LOAD/RUN FAILED under mode={args.mode_label}: {e!r} -- "
                          "recording rebuild_needed=True per the task's own fallback rule "
                          "(engines are expected to be portable across power modes on the same "
                          "GPU/TensorRT/CUDA version; a failure here would be a real, reportable "
                          "finding, not something to silently work around).")
                    lut["entries"][key] = {
                        "n_tokens": n_tokens, "precision": precision, "mode_label": args.mode_label,
                        "rebuild_needed": True, "error": repr(e),
                        "nvpmodel_state": nvpmodel_state, "jetson_clocks_raw": jetson_clocks_text,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    write_json(lut_path, lut)
                    continue
                temp_after = get_temp_c(jetson)

                median_ms = float(np.median(latencies_ms))
                p99_ms = float(np.percentile(latencies_ms, 99))
                mean_power_w = (float(np.mean(power_mw)) / 1000.0) if power_mw else None
                joules_per_inference = (mean_power_w * median_ms / 1000.0) if mean_power_w is not None else None

                lut["entries"][key] = {
                    "n_tokens": n_tokens, "precision": precision, "mode_label": args.mode_label,
                    "rebuild_needed": False,
                    "median_ms": median_ms, "p99_ms": p99_ms,
                    "temp_c_before": temp_before, "temp_c_after": temp_after,
                    "mean_power_w": mean_power_w, "joules_per_inference": joules_per_inference,
                    "n_measure_iters": MEASURE_ITERS, "n_warmup_iters": WARMUP_ITERS,
                    "nvpmodel_state": nvpmodel_state,
                    "jetson_clocks_raw": jetson_clocks_text,
                    "jetson_clocks_locked": locked,
                    "versions": versions, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(f"  median={median_ms:.4f}ms p99={p99_ms:.4f}ms "
                      f"temp={temp_before}->{temp_after}C power={mean_power_w}W")
                write_json(lut_path, lut)  # resume-safe -- write after every pair

    print(f"\nWrote {lut_path} ({len(lut['entries'])} entries) for mode={args.mode_label}.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)  # same TensorRT/torch.cuda shutdown-ordering workaround as measure_lut.py -- see its own comment
