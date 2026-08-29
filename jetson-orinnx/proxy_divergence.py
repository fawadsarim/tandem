"""jetson-orinnx/proxy_divergence.py -- device-independent (pure data analysis, no device, no
TensorRT). ENERGY MEASUREMENT (E-LUT), Part 2 -- the paper result. See jetson-orinnx/README.md's
"Energy measurement (E-LUT)" section and out/energy/PREREGISTRATION.md for the interpretation
locked in before Part 1 ran.

MODEL SCOPE: DeiT-S, batch=1 -- the same configuration every registered latency result uses.
Not DeiT-B, not batch=8 (out of scope for this task, per its own standing rules).

STEP 1-2: normalizes BOPs, measured latency, and measured energy to the N=197/FP16 reference,
per bucket/precision, and reports the two divergence factors (latency_ratio/BOPs_ratio,
energy_ratio/BOPs_ratio) so it's visible whether BOPs fails the two real cost axes the SAME way
or DIFFERENTLY. BOPs is computed via replan.py's own block_bops formula (imported, not
re-derived) -- "int8" is represented at bits=8 for this purpose (bits=6 shares the same
hardware precision label but a different simulated BOPs value; documented convention, not a
silent choice).

STEP 3: R^2 of BOPs vs measured energy, and R^2 of measured LATENCY vs measured energy (simple
linear regression, all 10 real measured points) -- which one PREDICTS energy better.

STEP 4: RE-PRICES (does NOT re-plan) the plans already committed in out/replan/plans_hw.csv
(BOPs-selected, JOINT, STAGED -- read-only) under the energy LUT, using the SAME linear
interpolation principle replan.py's own make_block_latency_fn uses for latency (ported here for
energy, not imported, since that function is hardcoded to the latency LUT's own dict shape).
Reports whether the energy-cheapest of the three plans differs from the latency-cheapest at any
budget -- a finding, not something this script acts on (no re-running the DP).

Usage: python proxy_divergence.py   (no device needed -- reads out/energy/energy_lut.json,
out/lut/latency_lut.json, out/replan/plans_hw.csv, all already real/committed)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compute_plan_divergence, ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from replan import block_bops, N_TOKENS_TOTAL, plan_bits, n_tokens_entering  # noqa: E402

BUCKETS = (197, 165, 133, 101, 69)
PRECISIONS = ("fp16", "int8")
BITS_FOR_PRECISION = {"fp16": 16, "int8": 8}  # documented convention -- see module docstring
D_MODEL_DEITS = 384


def r_squared(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def make_energy_lookup(energy_lut_entries):
    """Same linear-interpolation principle as replan.py's make_block_latency_fn, ported here
    (not imported) because that function is hardcoded to read median_ms from the LATENCY LUT's
    dict shape -- this one reads energy_per_inference_mJ from the ENERGY LUT instead."""
    by_precision = {"fp16": {}, "int8": {}}
    for entry in energy_lut_entries.values():
        if entry["precision"] not in by_precision:
            continue
        by_precision[entry["precision"]][entry["n_tokens"]] = entry["energy_per_inference_mJ"]

    def block_energy(n_l: int, bits: int) -> float:
        precision = "fp16" if bits == 16 else "int8"
        table = by_precision[precision]
        buckets_sorted = sorted(table.keys())
        assert buckets_sorted[0] <= n_l <= buckets_sorted[-1], (
            f"n_l={n_l} outside measured energy bucket range [{buckets_sorted[0]}, {buckets_sorted[-1]}]."
        )
        if n_l in table:
            return table[n_l]
        lo = max(b for b in buckets_sorted if b <= n_l)
        hi = min(b for b in buckets_sorted if b >= n_l)
        if lo == hi:
            return table[lo]
        frac = (n_l - lo) / (hi - lo)
        return table[lo] * (1 - frac) + table[hi] * frac

    return block_energy


def predict_plan_energy_mj(plan: dict, block_energy_fn) -> float:
    return sum(block_energy_fn(n_tokens_entering(plan, l), plan_bits(plan, l)) for l in range(1, 13))


def main():
    print("=== proxy_divergence.py: Part 2 -- proxy divergence on the energy axis (DeiT-S, batch=1) ===")

    energy_lut_path = JETSON_ROOT / "out" / "energy" / "energy_lut.json"
    assert energy_lut_path.exists(), f"{energy_lut_path} not found -- run energy_lut.py first."
    energy_lut = read_json(energy_lut_path)["entries"]

    latency_lut_path = JETSON_ROOT / "out" / "lut" / "latency_lut.json"
    assert latency_lut_path.exists(), f"{latency_lut_path} not found -- this is the committed LUT."
    latency_lut = read_json(latency_lut_path)["entries"]

    # === Steps 1-2: normalized ratio table + divergence factors ===
    rows = []
    for n in BUCKETS:
        for prec in PRECISIONS:
            key = f"n{n}_{prec}"
            bits = BITS_FOR_PRECISION[prec]
            bops = block_bops(n, bits, d_model=D_MODEL_DEITS)
            lat_ms = latency_lut[key]["median_ms"]
            energy_mj = energy_lut[key]["energy_per_inference_mJ"]
            rows.append({"n_tokens": n, "precision": prec, "bops": bops, "latency_ms": lat_ms, "energy_mJ": energy_mj})

    df = pd.DataFrame(rows)
    ref = df[(df.n_tokens == 197) & (df.precision == "fp16")].iloc[0]
    df["bops_ratio"] = df["bops"] / ref["bops"]
    df["latency_ratio"] = df["latency_ms"] / ref["latency_ms"]
    df["energy_ratio"] = df["energy_mJ"] / ref["energy_mJ"]
    df["latency_divergence_factor"] = df["latency_ratio"] / df["bops_ratio"]
    df["energy_divergence_factor"] = df["energy_ratio"] / df["bops_ratio"]
    df["divergence_factors_differ"] = (
        (df["energy_divergence_factor"] - df["latency_divergence_factor"]).abs() > 0.02
    )  # >2 percentage points apart -- an explicit, stated threshold, not an implicit eyeball call

    csv_path = ensure_out_dir("energy") / "proxy_divergence.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print("\n=== Normalized ratio table (reference: N=197/FP16) ===")
    for _, r in df.iterrows():
        print(f"  N={r['n_tokens']:3d} {r['precision']:4s}: BOPs_ratio={r['bops_ratio']:.4f}  "
              f"latency_ratio={r['latency_ratio']:.4f}  energy_ratio={r['energy_ratio']:.4f}  "
              f"latency_div={r['latency_divergence_factor']:+.4f}  energy_div={r['energy_divergence_factor']:+.4f}  "
              f"differ={r['divergence_factors_differ']}")
    n_differ = int(df["divergence_factors_differ"].sum())
    print(f"\nDivergence factors differ (>2pp apart) at {n_differ}/{len(df)} configurations.")

    # === Step 3: predictive comparison ===
    r2_bops_energy = r_squared(df["bops"], df["energy_mJ"])
    r2_latency_energy = r_squared(df["latency_ms"], df["energy_mJ"])
    print(f"\n=== Predictive comparison (R^2, all {len(df)} real measured points) ===")
    print(f"  R^2(BOPs -> energy)    = {r2_bops_energy:.4f}")
    print(f"  R^2(latency -> energy) = {r2_latency_energy:.4f}")

    # === Step 4: re-price committed plans (BOPs-selected, JOINT, STAGED) -- does NOT re-plan ===
    plans_hw_path = JETSON_ROOT / "out" / "replan" / "plans_hw.csv"
    assert plans_hw_path.exists(), f"{plans_hw_path} not found -- this is the committed, registered replan.py output."
    plans_hw = pd.read_csv(plans_hw_path)

    block_energy_fn = make_energy_lookup(energy_lut)
    reprice_rows = []
    n_ranking_differs = 0
    print("\n=== Re-pricing committed plans (BOPs-selected, JOINT, STAGED) under the energy LUT ===")
    for _, row in plans_hw.iterrows():
        budget_frac = row["budget_fraction"]
        bops_plan = json.loads(row["bops_plan_json"])
        joint_plan = json.loads(row["plan_json"])
        staged_plan = json.loads(row["staged_plan_json"])

        latencies = {"bops": float(row["bops_plan_latency_ms"]), "joint": float(row["latency_plan_latency_ms"]),
                     "staged": float(row["staged_plan_latency_ms"])}
        energies = {"bops": predict_plan_energy_mj(bops_plan, block_energy_fn),
                    "joint": predict_plan_energy_mj(joint_plan, block_energy_fn),
                    "staged": predict_plan_energy_mj(staged_plan, block_energy_fn)}

        latency_cheapest = min(latencies, key=latencies.get)
        energy_cheapest = min(energies, key=energies.get)
        ranking_differs = latency_cheapest != energy_cheapest
        if ranking_differs:
            n_ranking_differs += 1

        print(f"  budget={budget_frac:.1%}: latency(bops={latencies['bops']:.4f} joint={latencies['joint']:.4f} "
              f"staged={latencies['staged']:.4f})ms -> cheapest={latency_cheapest}   "
              f"energy(bops={energies['bops']:.4f} joint={energies['joint']:.4f} "
              f"staged={energies['staged']:.4f})mJ -> cheapest={energy_cheapest}   "
              f"{'*** RANKING DIFFERS ***' if ranking_differs else 'same ranking'}")

        reprice_rows.append({
            "budget_fraction": budget_frac,
            "bops_latency_ms": latencies["bops"], "joint_latency_ms": latencies["joint"], "staged_latency_ms": latencies["staged"],
            "bops_energy_mJ": energies["bops"], "joint_energy_mJ": energies["joint"], "staged_energy_mJ": energies["staged"],
            "latency_cheapest_plan": latency_cheapest, "energy_cheapest_plan": energy_cheapest,
            "ranking_differs": ranking_differs,
        })

    print(f"\nEnergy-cheapest differs from latency-cheapest at {n_ranking_differs}/{len(reprice_rows)} budgets.")
    if n_ranking_differs:
        print("*** FINDING: plan ranking flips between the latency and energy cost axes at "
              "least one budget. Flagged, not acted on -- any follow-up (e.g. an energy-aware "
              "re-plan) is a separate decision, not run here (RE-PRICE, do not re-plan, per "
              "this task's own scope). ***")

    result = {
        "model_scope": "deit_small_patch16_224, batch=1",
        "r2_bops_vs_energy": r2_bops_energy, "r2_latency_vs_energy": r2_latency_energy,
        "n_divergence_factors_differ": n_differ, "n_total_configs": len(df),
        "reprice": reprice_rows, "n_budgets_ranking_differs": n_ranking_differs,
        "pre_registered_interpretation_ref": "See out/energy/PREREGISTRATION.md, written before Part 1 ran.",
    }
    write_json(ensure_out_dir("energy") / "proxy_divergence_summary.json", result)
    print(f"\nWrote {ensure_out_dir('energy') / 'proxy_divergence_summary.json'}")


if __name__ == "__main__":
    main()
