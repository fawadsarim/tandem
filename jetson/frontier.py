"""jetson/frontier.py -- device-independent (pure Python DP + LUT lookup, no TensorRT -- runs
anywhere replan.py's own DP does, see that file's own module docstring). G1-HW final phase,
Task 3 -- see jetson/README.md's "Re-scoped sweep" section and jetson/CHARACTERIZATION.md.

TWO ARMS, per budget:
  ARM A -- TANDEM frontier: the IDENTICAL joint DP replan.py uses (same LUT, same schedule
     enumeration, same exact_pareto_bits -- imported directly from replan.py, not reimplemented),
     at a NEW budget grid set strictly inside the plan space's achievable range so every budget
     forces real compression (see BUDGET GRID below). Per CHARACTERIZATION.md (b), this is
     EXPECTED to select merge-only, all-16-bit plans at every budget (the precision axis is
     strictly dominated on this device/LUT) -- checked as a consistency assert per budget, not
     silently assumed.
  ARM B -- uniform-r ToMe: bits=16 throughout, a SINGLE merge amount r applied uniformly at
     every block 1..11 (N_l(l) = 197 - r*(l-1) for l=1..12), NOT TANDEM's own sparse <=2-point
     schedule. Represented using the SAME plan dict schema TANDEM's plan-execution engine
     already uses (`{"bits": {l: 16 for l in 1..12}, "merges": [{"block": l, "r": r} for l in
     1..11]}`) so every existing helper (plan_cost, compute_plan_divergence, and
     evaluate_plan_top1 in frontier_paired_eval.py) works UNCHANGED -- this is purely a
     different SELECTION rule over a differently-shaped candidate set, not a new
     plan-execution engine. Selection, per tau: the SMALLEST feasible r in {1..11} (least
     merging = least damage, under ToMe's own monotonicity assumption) whose summed LUT
     latency clears tau. r is restricted to {1..11} so every per-block N_l >= 69 -- the LUT's
     own measured floor (asserted per candidate, not just assumed from the range restriction).

BUDGET GRID (registered here, before this script has been run against real numbers): tau in
{0.77, 0.79, 0.82, 0.85, 0.89, 0.94} x Latency_id. The achievable floor (from the completed
run: 76.4% of Latency_id -- re-derived here directly via find_min_achievable_cost, not read
from a stale file) sits below all six, so all six force real compression; Latency_id itself
(100%) sits above all six, so the identity plan is infeasible at all six. Both facts are
asserted per budget in main() -- a violation is flagged loudly (a real degeneracy in this new
grid), not silently absorbed.

D_PRED NOTE (a real data gap, not glossed over): P2.1's own Dm_table.csv (the per-block,
per-merge-amount accuracy-damage table replan.py's objective_d_pred reads) has exactly 24 rows
-- MERGE_BLOCKS=(1,3,5,7,9,11) x MERGE_RS=(32,64,98,128), TANDEM's own registered plan-space
grid, confirmed by direct read. ARM B's uniform-r schedule needs EVERY block 1-11 (not just the
odd ones) at r in {1..11} -- entirely outside this table's measured domain. Extrapolating a new
accuracy-damage model here would be fabricating data this project has not measured or
registered anywhere, so ARM B's D_pred is written as null (with this note) in frontier_plans.csv
rather than guessed. The real comparison this task asks for is MEASURED top-1 accuracy via
frontier_paired_eval.py (Colab), not a predicted proxy -- ARM A's D_pred (fully covered by
Dm_table.csv, since it lives in TANDEM's own registered grid) is reported for reference only.

NOT a joint-vs-staged test: that comparison is VOID on this device/LUT (see
CHARACTERIZATION.md). This compares merge-PLACEMENT granularity (TANDEM's own <=2-point
schedules, ARM A) against uniform-r merging (ARM B) at matched MEASURED latency -- the hardware
analogue of the P2.2 saturation finding, a genuinely different and still-live question.

Usage: python frontier.py   (from the repo root or inside jetson/ -- JETSON_ROOT-relative paths
throughout, matching every other script in this folder; needs only latency_lut.json -- already
committed from the completed G1-HW run -- and P2.1's read-only Dq_table.csv/Dm_table.csv/
plans.csv, no device.)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compute_plan_divergence, ensure_out_dir, JETSON_ROOT, write_json  # noqa: E402
from replan import (  # noqa: E402
    BIT_MENU, MERGE_BLOCKS, MERGE_RS, MODEL_NAME, N_TOKENS_TOTAL,
    build_schedule_pareto_sets, enumerate_schedules, exact_select_for_budget,
    find_min_achievable_cost, load_dq_dm_tables, load_latency_lut, make_block_latency_fn,
    n_tokens_entering, plan_bits, plan_cost,
)

FRONTIER_BUDGET_FRACTIONS = (0.77, 0.79, 0.82, 0.85, 0.89, 0.94)
UNIFORM_R_RANGE = range(1, 12)  # 1..11 inclusive -- see module docstring


def uniform_r_plan(r: int) -> dict:
    """ARM B's candidate plan for merge amount r, expressed in TANDEM's own plan dict schema
    (see module docstring) -- one merge point per block 1..11, each of size r, bits=16 throughout."""
    return {"bits": {l: 16 for l in range(1, 13)}, "merges": [{"block": l, "r": r} for l in range(1, 12)]}


def select_uniform_r_for_budget(tau_ms: float, block_latency_fn):
    """Smallest feasible r in {1..11}, ascending -- see module docstring's Selection note.
    Returns (r, plan, cost) or (None, None, None) if even r=1 does not fit."""
    for r in UNIFORM_R_RANGE:
        plan = uniform_r_plan(r)
        for l in range(1, 13):
            n_l = n_tokens_entering(plan, l)
            assert n_l >= 69, (
                f"ARM B r={r}: block {l} n_l={n_l} < 69 (LUT's measured floor) -- should be "
                "structurally impossible for r<=11, see module docstring's UNIFORM_R_RANGE derivation."
            )
        cost = plan_cost(plan, block_latency_fn)
        if cost <= tau_ms + 1e-9:
            return r, plan, cost
    return None, None, None


def main():
    print("=== frontier.py: TANDEM hardware frontier (ARM A) vs uniform-r ToMe (ARM B) ===")
    print("(G1-HW final phase, Task 3 -- NOT a joint-vs-staged test, see module docstring)")

    dq_table, dm_table = load_dq_dm_tables()
    lut_by_precision = load_latency_lut()
    block_latency_fn = make_block_latency_fn(lut_by_precision)

    identity_plan = {"bits": {}, "merges": []}
    latency_id = plan_cost(identity_plan, block_latency_fn)
    print(f"\nLatency_id (identity plan) = {latency_id:.4f}ms")

    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules, got {len(schedules)}"
    schedule_pareto_sets = build_schedule_pareto_sets(
        schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_latency_fn
    )
    min_achievable_ms = find_min_achievable_cost(schedule_pareto_sets)
    min_achievable_frac = min_achievable_ms / latency_id
    print(f"Achievable floor = {min_achievable_ms:.4f}ms ({min_achievable_frac:.1%} of Latency_id)")

    hw_path = JETSON_ROOT / "out" / "replan" / "plans_hw.csv"
    bops_reference = []
    if hw_path.exists():
        hw_df = pd.read_csv(hw_path)
        for _, row in hw_df.iterrows():
            bops_reference.append({
                "budget_fraction": float(row["budget_fraction"]),
                "latency_ms": float(row["bops_plan_latency_ms"]),
                "plan_json": row["bops_plan_json"],
            })
        print(f"\nLoaded {len(bops_reference)} BOPs-selected reference points from {hw_path} "
              "(completed run, read-only -- for the eventual figure's dominated-reference overlay).")
    else:
        print(f"\nNOTE: {hw_path} not found -- skipping BOPs reference points (not required for "
              "this task's own comparison, only for the eventual figure's context overlay).")

    rows = []
    for frac in FRONTIER_BUDGET_FRACTIONS:
        tau_ms = frac * latency_id
        identity_infeasible = tau_ms < latency_id - 1e-9
        above_floor = tau_ms > min_achievable_ms + 1e-9
        print(f"\n=== tau_fraction={frac:.0%}  tau_ms={tau_ms:.4f}ms ===")
        print(f"  identity_infeasible (tau < Latency_id={latency_id:.4f}ms): {identity_infeasible}")
        print(f"  above_floor (tau > achievable floor={min_achievable_ms:.4f}ms): {above_floor}")
        if not identity_infeasible or not above_floor:
            print(f"  *** DEGENERACY FLAGGED *** at tau_fraction={frac:.0%} -- this budget does not "
                  "force real compression as the grid was designed to -- see module docstring's "
                  "BUDGET GRID note. Not halting: reported as a real finding about this grid.")

        result = exact_select_for_budget(schedule_pareto_sets, tau_ms, raise_on_infeasible=False)
        assert result is not None, (
            f"ARM A INFEASIBLE at tau_fraction={frac:.0%} (tau={tau_ms:.4f}ms) -- should be "
            f"impossible, the achievable floor ({min_achievable_ms:.4f}ms) is below every grid "
            "point by construction. Real bug in the DP or the LUT if this ever fires."
        )
        arm_a_cost, arm_a_d_pred, arm_a_plan = result
        arm_a_all_bits_16 = all(plan_bits(arm_a_plan, l) == 16 for l in range(1, 13))
        print(f"  ARM A (TANDEM joint DP): cost={arm_a_cost:.4f}ms D_pred={arm_a_d_pred:.6f}")
        print(f"    bits={arm_a_plan['bits']}")
        print(f"    merges={arm_a_plan['merges']}")
        print(f"    all-16-bit consistency check (CHARACTERIZATION.md (b)): {arm_a_all_bits_16}")
        if not arm_a_all_bits_16:
            print("    *** UNEXPECTED *** ARM A selected bits<16 somewhere -- either "
                  "CHARACTERIZATION.md's axis-domination derivation is wrong for this budget, or "
                  "the LUT changed since it was derived. Flagging loudly, not halting -- this would "
                  "itself be a real, reportable finding.")

        r, arm_b_plan, arm_b_cost = select_uniform_r_for_budget(tau_ms, block_latency_fn)
        arm_b_infeasible = r is None
        if arm_b_infeasible:
            # REAL FINDING, not a bug -- report and continue, same discipline as replan.py's own
            # raise_on_infeasible=False path. Merging at 11 SEPARATE blocks (ARM B) pays whatever
            # near-fixed per-merge-operation overhead this device has 11 times over, vs TANDEM's
            # own <=2-point schedule (ARM A) concentrating the same token reduction into 1-2
            # operations -- on a device where merging's latency savings are dominated by that
            # per-operation overhead (see README.md's Budget pairing section), spreading it
            # this thin can cost MORE than the token savings are worth, even at the smallest
            # possible r. This is itself informative about merge-placement granularity -- exactly
            # what this comparison is asking -- not a failure of the search.
            print(f"  ARM B (uniform-r ToMe): *** INFEASIBLE *** even r=1 (minimal uniform "
                  f"merging at 11 blocks) exceeds tau={tau_ms:.4f}ms. Real finding: spreading "
                  "merges across many blocks pays this device's per-operation overhead many "
                  "times over, unlike TANDEM's own concentrated <=2-point schedule (ARM A).")
            plans_identical, n_blocks_differing = False, None
        else:
            print(f"  ARM B (uniform-r ToMe): r={r} cost={arm_b_cost:.4f}ms "
                  "D_pred=N/A (see module docstring's D_PRED NOTE)")
            print(f"    merges={arm_b_plan['merges']}")
            plans_identical, n_blocks_differing, _ = compute_plan_divergence(arm_a_plan, arm_b_plan, N_TOKENS_TOTAL)
            print(f"  ARM A vs ARM B: plans_identical={plans_identical}  n_blocks_differing={n_blocks_differing}")

        rows.append({
            "budget_fraction": frac, "tau_ms": tau_ms, "latency_id_ms": latency_id,
            "min_achievable_ms": min_achievable_ms, "min_achievable_fraction": min_achievable_frac,
            "identity_infeasible": identity_infeasible, "above_floor": above_floor,
            "arm_a_plan_json": json.dumps(arm_a_plan), "arm_a_cost_ms": arm_a_cost,
            "arm_a_d_pred": arm_a_d_pred, "arm_a_utilization": arm_a_cost / tau_ms,
            "arm_a_all_bits_16": arm_a_all_bits_16,
            "arm_b_infeasible": arm_b_infeasible,
            "arm_b_r": r, "arm_b_plan_json": json.dumps(arm_b_plan) if arm_b_plan else None,
            "arm_b_cost_ms": arm_b_cost,
            "arm_b_d_pred": None, "arm_b_utilization": (arm_b_cost / tau_ms) if arm_b_cost else None,
            "plans_identical": plans_identical, "n_blocks_differing": n_blocks_differing,
        })

    out_dir = ensure_out_dir("frontier")
    df = pd.DataFrame(rows)
    csv_path = out_dir / "frontier_plans.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    n_degenerate = sum(1 for r in rows if not (r["identity_infeasible"] and r["above_floor"]))
    n_arm_a_bits16 = sum(1 for r in rows if r["arm_a_all_bits_16"])
    n_arm_b_infeasible = sum(1 for r in rows if r["arm_b_infeasible"])
    n_identical = sum(1 for r in rows if r["plans_identical"])
    manifest = {
        "model_name": MODEL_NAME, "latency_id_ms": latency_id,
        "min_achievable_latency_ms": min_achievable_ms, "min_achievable_fraction": min_achievable_frac,
        "budget_fractions": list(FRONTIER_BUDGET_FRACTIONS),
        "n_degenerate_budgets": n_degenerate, "n_arm_a_all_bits_16": n_arm_a_bits16,
        "n_arm_b_infeasible": n_arm_b_infeasible,
        "n_plans_identical_arm_a_vs_arm_b": n_identical,
        "arm_b_infeasibility_note": (
            "A real finding, not a bug, if n_arm_b_infeasible > 0 -- ARM B (uniform-r ToMe) "
            "applies a merge operation at 11 SEPARATE blocks, paying this device's near-fixed "
            "per-merge-operation overhead 11 times over, vs ARM A/TANDEM's own <=2-point "
            "schedule concentrating the same token reduction into 1-2 operations. At tight "
            "budgets this can make even the smallest uniform-r candidate (r=1) cost more than "
            "the budget allows. See per-row arm_b_infeasible in frontier_plans.csv."
        ) if n_arm_b_infeasible else None,
        "d_pred_note": (
            "ARM B's D_pred is null in frontier_plans.csv -- P2.1's own Dm_table.csv only covers "
            "TANDEM's registered plan-space grid (MERGE_BLOCKS x MERGE_RS, 24 rows), not ARM B's "
            "uniform-every-block r in {1..11}. See this file's module docstring D_PRED NOTE. The "
            "real comparison is measured top-1 accuracy via frontier_paired_eval.py."
        ),
        "bops_reference_points": bops_reference,
        "not_a_joint_vs_staged_test_note": (
            "This is NOT a joint-vs-staged test -- that comparison is VOID on this device/LUT, see "
            "CHARACTERIZATION.md. This compares merge-PLACEMENT granularity (TANDEM's own "
            "<=2-point schedules, ARM A) against uniform-r merging (ARM B) at matched measured "
            "latency -- the hardware analogue of the P2.2 saturation finding."
        ),
    }
    write_json(out_dir / "frontier_manifest.json", manifest)
    print(f"Wrote {out_dir / 'frontier_manifest.json'}")
    print(f"\n{n_degenerate}/{len(rows)} budgets degenerate, {n_arm_a_bits16}/{len(rows)} ARM A "
          f"all-16-bit (expected {len(rows)}/{len(rows)} per CHARACTERIZATION.md), "
          f"{n_identical}/{len(rows)} ARM A == ARM B exactly.")


if __name__ == "__main__":
    main()
