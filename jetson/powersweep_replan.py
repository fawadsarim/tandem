"""jetson/powersweep_replan.py -- device-independent (pure Python DP + LUT lookup, no TensorRT,
no device -- same as replan.py/energy_replan.py). POWER-MODE SWEEP, Part 4. For EVERY mode's own
out/powersweep/latency_lut_<mode>.json, re-runs the IDENTICAL joint DP and staged
(BEST-SEQUENTIAL, both orders, alpha-grid) arms replan.py already uses, substituting that mode's
own measured latency as the cost axis -- the same substitution principle --verify-port already
proves faithful. DP machinery (enumerate_schedules, build_schedule_pareto_sets,
exact_select_for_budget, best_sequential_for_budget, plan_bits, make_block_latency_fn) is
imported directly from replan.py, unchanged -- this file only supplies a per-mode LUT loader
(load_latency_lut_from) since replan.py's own load_latency_lut() is hardcoded to
out/lut/latency_lut.json, the same "duplicate the one small LUT-path function, import
everything else" principle energy_replan.py/proxy_divergence.py already use for the energy LUT.

BUDGET PAIRING (same registered principle as replan.py's own "Budget pairing" section, per
mode): tau_b = the MEASURED LATENCY, IN THAT MODE, of P2.1's own BOPs-selected plan at budget b
-- guarantees feasibility by construction in every mode, for the same reason replan.py's own
docstring proves it for MAXN_SUPER (the BOPs plan is always a feasible candidate in the identical
85-schedule x BIT_MENU plan space, regardless of which LUT prices it).

NO PAIRED EVALUATION -- answers only: does "BOPs plan costs more than doing nothing" (tau_b >=
Latency_id) and "does joint ever separate from staged" reproduce at other operating points.

Writes out/powersweep/replan_by_mode.csv (one row per mode x budget) and
out/powersweep/replan_by_mode_manifest.json (per-mode summary).

Usage: python powersweep_replan.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compute_plan_divergence, ensure_out_dir, JETSON_ROOT, read_json, read_p2_csv_path, write_json  # noqa: E402
from replan import (  # noqa: E402
    BIT_MENU, best_sequential_for_budget, BUDGET_FRACTIONS, build_schedule_pareto_sets,
    enumerate_schedules, exact_select_for_budget, load_dq_dm_tables, make_block_latency_fn,
    MERGE_BLOCKS, MERGE_RS, MODEL_NAME, N_TOKENS_TOTAL, plan_bits, plan_cost, TOKEN_BUCKETS,
)

PRECISIONS = ("fp16", "int8")


def load_latency_lut_from(path):
    """Same shape-loading logic as replan.py's own load_latency_lut(), just parametrized over
    the source path instead of hardcoding out/lut/latency_lut.json -- see module docstring."""
    entries = read_json(path)["entries"]
    by_precision = {"fp16": {}, "int8": {}}
    for entry in entries.values():
        if entry.get("rebuild_needed") or entry.get("precision") not in by_precision:
            continue
        by_precision[entry["precision"]][entry["n_tokens"]] = entry["median_ms"]
    for prec in PRECISIONS:
        missing = set(TOKEN_BUCKETS) - set(by_precision[prec].keys())
        assert not missing, f"{path}: missing {prec} entries for buckets {missing}"
    return by_precision


def plan_uses_bits_below_16(plan: dict) -> bool:
    return any(plan_bits(plan, l) != 16 for l in range(1, 13))


def main():
    print("=== powersweep_replan.py: Part 4 -- joint + staged DP against each mode's own LUT (no device, no eval) ===")

    powersweep_dir = JETSON_ROOT / "out" / "powersweep"
    mode_files = sorted(powersweep_dir.glob("latency_lut_*.json"))
    assert mode_files, f"no latency_lut_<mode>.json files found in {powersweep_dir} -- run powersweep_measure.py first."

    dq_table, dm_table = load_dq_dm_tables()
    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules, got {len(schedules)}"

    bops_df = pd.read_csv(read_p2_csv_path(MODEL_NAME, "plans.csv"))
    assert len(bops_df) == len(BUDGET_FRACTIONS), (
        f"P2.1 plans.csv has {len(bops_df)} rows, expected {len(BUDGET_FRACTIONS)}"
    )
    bops_plans = [json.loads(row["plan_json"]) for _, row in bops_df.iterrows()]

    all_rows = []
    per_mode_summary = {}
    for f in mode_files:
        mode_label = f.stem.replace("latency_lut_", "")
        print(f"\n--- mode={mode_label} ---")
        lut_by_precision = load_latency_lut_from(f)
        block_latency_fn = make_block_latency_fn(lut_by_precision)

        identity_plan = {"bits": {}, "merges": []}
        latency_id = plan_cost(identity_plan, block_latency_fn)
        print(f"  Latency_id (identity plan) = {latency_id:.4f} ms")

        schedule_pareto_sets = build_schedule_pareto_sets(
            schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_latency_fn
        )
        taus_ms = [plan_cost(p, block_latency_fn) for p in bops_plans]

        n_identical, n_bops_dominated = 0, 0
        for i, budget_frac in enumerate(BUDGET_FRACTIONS):
            tau_ms = taus_ms[i]
            joint_result = exact_select_for_budget(schedule_pareto_sets, tau_ms, raise_on_infeasible=False)
            assert joint_result is not None, (
                f"mode={mode_label} budget_fraction={budget_frac:.1%}: JOINT INFEASIBLE at "
                f"tau_b={tau_ms:.4f}ms -- should be impossible by construction (see module "
                "docstring's BUDGET PAIRING section). Real bug if this ever fires."
            )
            joint_cost, joint_d_pred, joint_plan = joint_result

            staged_result = best_sequential_for_budget(tau_ms, schedules, latency_id, dq_table, dm_table, block_latency_fn)
            staged_plan = staged_result["plan"]

            plans_identical, n_blocks_differing, _ = compute_plan_divergence(joint_plan, staged_plan, N_TOKENS_TOTAL)
            bops_dominated_by_identity = tau_ms >= latency_id - 1e-9
            bits_below_16_anywhere = plan_uses_bits_below_16(joint_plan) or plan_uses_bits_below_16(staged_plan)

            if plans_identical:
                n_identical += 1
            if bops_dominated_by_identity:
                n_bops_dominated += 1

            print(f"  budget={budget_frac:.1%}  tau_b={tau_ms:.4f}ms  tau_b/Latency_id={tau_ms / latency_id:.1%}  "
                  f"bops_dominated_by_identity={bops_dominated_by_identity}  "
                  f"joint_vs_staged_identical={plans_identical}  n_blocks_differing={n_blocks_differing}  "
                  f"bits<16_selected_anywhere={bits_below_16_anywhere}")

            all_rows.append({
                "mode": mode_label, "budget_fraction": budget_frac, "tau_b_ms": tau_ms,
                "latency_id_ms": latency_id, "tau_b_over_latency_id": tau_ms / latency_id,
                "bops_dominated_by_identity": bops_dominated_by_identity,
                "joint_vs_staged_plans_identical": plans_identical,
                "n_blocks_differing": n_blocks_differing,
                "bits_below_16_selected_anywhere": bits_below_16_anywhere,
                "joint_plan_json": json.dumps(joint_plan), "staged_plan_json": json.dumps(staged_plan),
            })

        per_mode_summary[mode_label] = {
            "n_budgets_bops_dominated_by_identity": n_bops_dominated,
            "n_budgets_total": len(BUDGET_FRACTIONS),
            "n_budgets_joint_staged_identical": n_identical,
            "joint_ever_separates_from_staged": n_identical < len(BUDGET_FRACTIONS),
        }

    df = pd.DataFrame(all_rows)
    csv_path = ensure_out_dir("powersweep") / "replan_by_mode.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    print("\n=== SUMMARY, per mode ===")
    for mode_label, s in per_mode_summary.items():
        print(f"  mode={mode_label}: {s['n_budgets_bops_dominated_by_identity']}/{s['n_budgets_total']} "
              f"budgets have tau_b>=Latency_id  |  joint_vs_staged identical at "
              f"{s['n_budgets_joint_staged_identical']}/{s['n_budgets_total']}  |  "
              f"joint_ever_separates_from_staged={s['joint_ever_separates_from_staged']}")

    write_json(ensure_out_dir("powersweep") / "replan_by_mode_manifest.json", {
        "model_name": MODEL_NAME, "budget_fractions": list(BUDGET_FRACTIONS),
        "per_mode_summary": per_mode_summary,
        "no_paired_eval_note": "NO paired accuracy evaluation was computed -- plan selection only, per this task's own scope.",
    })
    print(f"Wrote {ensure_out_dir('powersweep') / 'replan_by_mode_manifest.json'}")


if __name__ == "__main__":
    main()
