"""jetson-orinnx/verify_orinnx_claims.py -- re-derives every headline number in
FINDINGS.md and validates cross-file consistency from committed artifacts ALONE
(no GPU, no device, no network).
Prints MATCH/MISMATCH per claim, exits non-zero on any mismatch.

Usage: python verify_orinnx_claims.py   (run from jetson-orinnx/ or repo root, any host, no GPU needed)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 1e-3  # relative tolerance for recomputed-vs-stated numeric checks

failures = []


def check(label, actual, expected, tol=TOL, method=""):
    ok = abs(actual - expected) <= tol * max(abs(expected), 1e-12)
    status = "MATCH" if ok else "MISMATCH"
    print(f"[{status}] {label}: recomputed={actual!r} vs claimed={expected!r} {method}")
    if not ok:
        failures.append(label)


def main():
    print("=== verify_orinnx_claims.py: re-deriving FINDINGS.md headline numbers & cross-file consistency ===\n")

    # --- Claim 1: proxy divergence, N=69 vs N=197, fp16, MAXN = 2.2582x ---
    print("--- Claim 1: proxy divergence (FINDINGS.md finding 1) ---")
    with open(ROOT / "out/powersweep/proxy_divergence_by_mode.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["mode"] == "MAXN" and r["n_tokens"] == "69" and r["precision"] == "fp16")
    recomputed_proxy = float(row["latency_ratio"]) / float(row["bops_ratio"])
    check("proxy_error(N=69/fp16, MAXN), recomputed from bops_ratio/latency_ratio in CSV",
          recomputed_proxy, 2.2582, tol=1e-3)
    check("proxy_error, stored proxy_error column vs FINDINGS.md figure",
          float(row["proxy_error"]), 2.2582, tol=1e-3)

    # --- Claim 2: INT8/FP16 precision-axis ratios (Canonical LUT & Powersweep) ---
    print("\n--- Claim 2: INT8/FP16 precision-axis ratios (FINDINGS.md finding 2) ---")
    lut_canonical = json.load(open(ROOT / "out/lut/latency_lut.json"))["entries"]
    expected_lut_ratios = {69: 1.1234, 101: 1.1646, 133: 1.2529, 165: 1.3204, 197: 1.4357}
    for n, exp in expected_lut_ratios.items():
        fp16_med = float(lut_canonical[f"n{n}_fp16"]["median_ms"])
        int8_med = float(lut_canonical[f"n{n}_int8"]["median_ms"])
        ratio = int8_med / fp16_med
        check(f"n={n} canonical LUT int8/fp16 ratio", ratio, exp, tol=1e-3)

    with open(ROOT / "out/powersweep/precision_axis_by_mode.csv", newline="") as f:
        prows = {int(r["n_tokens"]): r for r in csv.DictReader(f)}
    expected_sweep_ratios = {69: 1.1198, 101: 1.1957, 133: 1.2518, 165: 1.3056, 197: 1.4357}
    for n, exp in expected_sweep_ratios.items():
        r = prows[n]
        recomputed = float(r["int8_median_ms"]) / float(r["fp16_median_ms"])
        check(f"n={n} powersweep int8/fp16 ratio", recomputed, exp, tol=1e-3)
    n_slower = sum(1 for r in prows.values() if float(r["int8_median_ms"]) > float(r["fp16_median_ms"]))
    check("count of buckets where INT8 is slower than FP16 (out of 5)", n_slower, 5, tol=0)

    # --- Claim 3: Cross-File Consistency on Latency_id & Achievable Floor ---
    print("\n--- Claim 3: Latency_id & Achievable Floor Cross-File Consistency ---")
    replan_manifest = json.load(open(ROOT / "out/replan/replan_manifest.json"))
    transfer = json.load(open(ROOT / "out/transfer/transfer_experiment_result.json"))

    lat_id_replan = float(replan_manifest["latency_id_ms"])
    lat_id_transfer = float(transfer["latency_id_nx_ms"])
    check("Latency_id (NX) in replan_manifest.json", lat_id_replan, 3.7202, tol=1e-3)
    check("Latency_id (NX) in transfer_experiment_result.json", lat_id_transfer, 3.7202, tol=1e-3)
    check("CROSS-FILE CONSISTENCY: replan Latency_id == transfer Latency_id",
          abs(lat_id_replan - lat_id_transfer), 0.0, tol=1e-6)

    floor_replan = float(replan_manifest["min_achievable_fraction_of_identity"])
    floor_transfer = float(transfer["nx_achievable_range_frac"][0])
    check("Floor fraction in replan_manifest.json", floor_replan, 0.7693, tol=2e-3)
    check("Floor fraction in transfer_experiment_result.json", floor_transfer, 0.7693, tol=2e-3)
    check("CROSS-FILE CONSISTENCY: replan floor == transfer floor",
          abs(floor_replan - floor_transfer), 0.0, tol=1e-4)

    # --- Claim 4: Transfer Experiment (Nano LUT vs NX Envelope) ---
    print("\n--- Claim 4: Transfer experiment, cross-device failure ---")
    check("Latency_id (Nano) in transfer_experiment_result.json", transfer["latency_id_nano_ms"], 4.9668, tol=1e-3)
    lat_ratio = transfer["latency_id_nx_ms"] / transfer["latency_id_nano_ms"]
    check("Latency_id ratio NX/Nano (3.7202 / 4.9668)", lat_ratio, 0.7490, tol=1e-3)
    n_infeasible = sum(1 for r in transfer["rows"] if not r.get("feasible_nano", True))
    check("count of tested NX-range budgets infeasible under Nano table (out of 4)", n_infeasible, 4, tol=0)

    # --- Claim 5: Official Reproduction Gate (10/10 within 3.0%, max delta 1.88%) ---
    print("\n--- Claim 5: Official Reproduction Gate (out/powersweep/reproduction_gate.csv) ---")
    with open(ROOT / "out/powersweep/reproduction_gate.csv", newline="") as f:
        rrows = list(csv.DictReader(f))
    n_within = 0
    max_delta = 0.0
    for r in rrows:
        c, m = float(r["committed_median_ms"]), float(r["remeasured_median_ms"])
        recomputed_pct = (m - c) / c * 100.0
        max_delta = max(max_delta, abs(recomputed_pct))
        within = abs(recomputed_pct) <= 3.0
        if within:
            n_within += 1
    check("count of reproduction-gate rows within 3% (out of 10)", n_within, 10, tol=0)
    check("reproduction-gate maximum absolute delta <= 1.88%", max_delta, 1.884, tol=0.01)

    # --- Claim 6: Energy Instrument Monotonic Ladder ---
    print("\n--- Claim 6: Energy instrument monotonic ladder (PB-5) ---")
    energy_inst = json.load(open(ROOT / "out/energy/energy_instrument.json"))
    ladder = energy_inst["monotonic_ladder"]["ladder"]
    powers = [entry["mean_power_w"] for entry in ladder]
    is_monotonic = all(powers[i] < powers[i + 1] for i in range(len(powers) - 1))
    check("monotonic ladder holds (idle < N=69 < N=197)", float(is_monotonic), 1.0, tol=0)
    check("recorded validation_passed flag", float(energy_inst["validation_passed"]), 1.0, tol=0)

    print(f"\n{'='*70}")
    if failures:
        print(f"VERIFICATION FAILED: {len(failures)} mismatch(es): {failures}")
        sys.exit(1)
    print("ALL CLAIMS & CROSS-FILE CHECKS VERIFIED -- zero staleness, complete consistency.")
    sys.exit(0)


if __name__ == "__main__":
    main()
