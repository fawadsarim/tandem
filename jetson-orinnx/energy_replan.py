"""jetson-orinnx/energy_replan.py -- device-independent (pure Python DP + energy LUT lookup, no
TensorRT, no device -- runs anywhere replan.py's own DP does). ENERGY-COST REPLAN -- see
jetson-orinnx/README.md's "Energy-cost replan" section and out/energy_replan/PREREGISTRATION.md
(written before this script was run).

PURPOSE: re-runs the IDENTICAL joint DP and staged (STOP-2, both orders, alpha-grid) arms
replan.py already uses for latency, substituting out/energy/energy_lut.json for the latency
LUT as the cost axis -- the same substitution principle --verify-port already proves faithful
(every call to plan_cost/objective_d_pred/exact_pareto_bits takes a block_cost_fn parameter; no
other logic touched). DP machinery (enumerate_schedules, build_schedule_pareto_sets,
exact_select_for_budget, best_sequential_for_budget, plan_bits, n_tokens_entering) is imported
directly from replan.py (jetson-orinnx-internal import, unchanged/untouched); the energy-LUT lookup
function is imported from proxy_divergence.py's make_energy_lookup (already written and used
there, not reimplemented a third time).

NO PAIRED EVALUATION, NO ACCURACY NUMBERS -- this answers exactly one question: do joint and
staged select different PLANS when the cost axis is energy instead of latency? Nothing here
touches paired_eval.py, the 20k image set, or any accuracy computation.

BUDGET PAIRING (same registered principle as replan.py's own "Budget pairing" section, just
substituting energy for latency): tau_b = the MEASURED ENERGY (mJ/inference) of P2.1's own
BOPs-selected plan at budget b -- guarantees feasibility by construction, since the BOPs plan is
itself always a valid candidate in the identical 85-schedule x BIT_MENU plan space.

Writes to out/energy_replan/ only -- does not touch out/replan/, out/energy/, or any other
existing artifact.

Usage: python energy_replan.py   (no device needed -- reads out/energy/energy_lut.json and
P2.1's read-only tables, all already real/committed)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compute_plan_divergence, ensure_out_dir, JETSON_ROOT, read_json, read_p2_csv_path, write_json  # noqa: E402
from proxy_divergence import make_energy_lookup  # noqa: E402
from replan import (  # noqa: E402
    BIT_MENU, best_sequential_for_budget, BUDGET_FRACTIONS, build_schedule_pareto_sets,
    enumerate_schedules, exact_select_for_budget, load_dq_dm_tables, MERGE_BLOCKS, MERGE_RS,
    MODEL_NAME, N_TOKENS_TOTAL, plan_bits, plan_cost,
)


def plan_uses_bits_below_16(plan: dict) -> bool:
    return any(plan_bits(plan, l) != 16 for l in range(1, 13))


def main():
    print("=== energy_replan.py: joint + staged DP against the ENERGY LUT (no device, no eval) ===")

    energy_lut_path = JETSON_ROOT / "out" / "energy" / "energy_lut.json"
    assert energy_lut_path.exists(), f"{energy_lut_path} not found -- run energy_lut.py on the Jetson first."
    energy_entries = read_json(energy_lut_path)["entries"]
    block_energy_fn = make_energy_lookup(energy_entries)

    dq_table, dm_table = load_dq_dm_tables()

    identity_plan = {"bits": {}, "merges": []}
    energy_id = plan_cost(identity_plan, block_energy_fn)
    print(f"Energy_id (identity plan, all-FP16, r=0) = {energy_id:.4f} mJ/inference")

    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules, got {len(schedules)}"
    schedule_pareto_sets = build_schedule_pareto_sets(
        schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_energy_fn
    )

    bops_df = pd.read_csv(read_p2_csv_path(MODEL_NAME, "plans.csv"))
    assert len(bops_df) == len(BUDGET_FRACTIONS), (
        f"P2.1 plans.csv has {len(bops_df)} rows, expected {len(BUDGET_FRACTIONS)}"
    )
    bops_plans = [json.loads(row["plan_json"]) for _, row in bops_df.iterrows()]
    taus_mj = [plan_cost(p, block_energy_fn) for p in bops_plans]

    print("\nBudget pairing: tau_b = measured ENERGY of P2.1's own BOPs-selected plan at each budget")
    print(f"{'budget (nominal)':>17}  {'tau_b (mJ)':>11}  {'tau_b / Energy_id':>19}")
    for budget_frac, tau_mj in zip(BUDGET_FRACTIONS, taus_mj):
        print(f"{budget_frac:>16.1%}  {tau_mj:>11.4f}  {tau_mj / energy_id:>18.1%}")

    rows = []
    any_bits_below_16_anywhere = False
    for i, budget_frac in enumerate(BUDGET_FRACTIONS):
        tau_mj = taus_mj[i]
        print(f"\n=== budget_fraction={budget_frac:.1%}  tau_b={tau_mj:.4f}mJ ===")

        joint_result = exact_select_for_budget(schedule_pareto_sets, tau_mj, raise_on_infeasible=False)
        assert joint_result is not None, (
            f"JOINT INFEASIBLE at budget_fraction={budget_frac:.1%} (tau_b={tau_mj:.4f}mJ) -- "
            "should be impossible by construction: P2.1's own BOPs-selected plan at this "
            "budget is itself always a feasible candidate at its own measured energy cost, in "
            "the SAME 85-schedule x BIT_MENU plan space. Real bug if this ever fires."
        )
        joint_cost, joint_d_pred, joint_plan = joint_result

        staged_result = best_sequential_for_budget(tau_mj, schedules, energy_id, dq_table, dm_table, block_energy_fn)
        staged_plan = staged_result["plan"]

        plans_identical, n_blocks_differing, _ = compute_plan_divergence(joint_plan, staged_plan, N_TOKENS_TOTAL)
        joint_uses_int8 = plan_uses_bits_below_16(joint_plan)
        staged_uses_int8 = plan_uses_bits_below_16(staged_plan)
        precision_axis_exercised = joint_uses_int8 or staged_uses_int8
        if precision_axis_exercised:
            any_bits_below_16_anywhere = True

        print(f"  JOINT:  cost={joint_cost:.4f}mJ D_pred={joint_d_pred:.6f}  bits={joint_plan['bits']} "
              f"merges={joint_plan['merges']}  bits<16 anywhere={joint_uses_int8}")
        print(f"  STAGED: cost={staged_result['cost']:.4f}mJ D_pred={staged_result['D_pred']:.6f}  "
              f"winner={staged_result['order']} alpha={staged_result['alpha']}  bits={staged_plan['bits']} "
              f"merges={staged_plan['merges']}  bits<16 anywhere={staged_uses_int8}")
        print(f"  plans_identical={plans_identical}  n_blocks_differing={n_blocks_differing}  "
              f"PRECISION AXIS EXERCISED (bits<16 selected anywhere, either plan)={precision_axis_exercised}")

        assert joint_d_pred <= staged_result["D_pred"] + 1e-9, (
            f"budget_fraction={budget_frac:.1%}: JOINT's D_pred ({joint_d_pred:.6f}) is WORSE "
            f"than STAGED's ({staged_result['D_pred']:.6f}) -- should be structurally "
            "impossible (same guard as replan.py's own latency-axis run). Real bug, halting."
        )

        rows.append({
            "budget_fraction": budget_frac, "tau_mj": tau_mj,
            "joint_cost_mj": joint_cost, "joint_d_pred": joint_d_pred, "joint_plan_json": json.dumps(joint_plan),
            "joint_uses_bits_below_16": joint_uses_int8,
            "staged_cost_mj": staged_result["cost"], "staged_d_pred": staged_result["D_pred"],
            "staged_plan_json": json.dumps(staged_plan), "staged_order": staged_result["order"],
            "staged_alpha": staged_result["alpha"], "staged_uses_bits_below_16": staged_uses_int8,
            "plans_identical": plans_identical, "n_blocks_differing": n_blocks_differing,
            "precision_axis_exercised": precision_axis_exercised,
        })

    out_dir = ensure_out_dir("energy_replan")
    df = pd.DataFrame(rows)
    csv_path = out_dir / "energy_replan_plans.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    n_identical = sum(1 for r in rows if r["plans_identical"])
    print(f"\n=== SUMMARY ===")
    print(f"plans_identical at {n_identical}/{len(rows)} budgets")
    print(f"PRECISION AXIS EXERCISED (bits<16 selected anywhere, any budget, either plan): "
          f"{any_bits_below_16_anywhere}")
    if n_identical == len(rows):
        interpretation = (
            "Plans are identical at all six budgets -- the axis-domination characterization "
            "holds on the energy axis as well. Device work is closed."
        )
    else:
        interpretation = (
            f"Plans differ at {len(rows) - n_identical}/{len(rows)} budget(s) -- joint-vs-staged "
            "is a live test on the energy axis. Whether a paired evaluation is warranted "
            "is a follow-up decision, not made by this script."
        )
    print(f"\n{interpretation}")

    manifest = {
        "model_name": MODEL_NAME, "energy_id_mj": energy_id,
        "budget_fractions": list(BUDGET_FRACTIONS), "taus_mj": taus_mj,
        "n_budgets_identical": n_identical, "n_budgets_total": len(rows),
        "precision_axis_exercised_anywhere": any_bits_below_16_anywhere,
        "interpretation": interpretation,
        "pre_registered_note": (
            "See out/energy_replan/PREREGISTRATION.md, written before this script was run: "
            "'If plans are identical at all six budgets, the axis-domination characterization "
            "holds on the energy axis as well and device work is closed. If plans differ at "
            "any budget, joint-vs-staged is a live test on this axis and whether a paired "
            "evaluation is warranted is a follow-up decision.'"
        ),
        "no_paired_eval_note": "NO paired evaluation, NO accuracy numbers were computed -- plan selection only.",
    }
    write_json(out_dir / "energy_replan_manifest.json", manifest)
    print(f"Wrote {out_dir / 'energy_replan_manifest.json'}")


if __name__ == "__main__":
    main()
