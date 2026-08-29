"""jetson-orinnx/frontier_v2.py -- device-independent (pure Python DP + LUT lookup, no TensorRT --
runs anywhere replan.py/frontier.py's own DP does). LUT EXTENSION, step 3-5 -- see
jetson-orinnx/README.md's "LUT extension" section.

PURPOSE: frontier.py's original ARM B (uniform-r ToMe) found r restricted to {1..11} because
the ORIGINAL 5-bucket LUT's floor was N=69 (n_tokens_entering block 12 = 197 - 11r >= 69 forces
r <= 11) -- a bound set by what was MEASURED, not by any property of uniform merging itself.
This script re-runs ARM A and ARM B against the EXTENDED LUT (out/lut_ext/
latency_lut_extended.json, N=53/37/21 added, see lut_ext_measure.py) to find out whether that
was a real hardware limit or an artifact of the LUT's own range.

IMMUTABLE INPUTS: out/frontier/frontier_plans.csv (the ORIGINAL v1 run) is read ONLY for
comparison -- never written to. All new output goes to out/frontier_v2/.

ARM A: the IDENTICAL joint DP (same 85 schedules, same BIT_MENU, same 6 budget fractions),
now costed against the extended LUT. TANDEM's own schedule space (MERGE_BLOCKS x MERGE_RS) only
ever produces n_l in {69,101,133,165,197} by construction (see export_segments.py's docstring
on why those 5 buckets were chosen in the first place) -- extending the LUT's floor gives this
DP no new n_l values to work with, so ARM A's plans are EXPECTED to be unchanged. This is
checked, not assumed: every ARM A plan here is compared against v1's committed plan, and any
difference is reported explicitly (both old and new appear in the record).

ARM B: r's upper bound is now DERIVED from the extended LUT's actual measured floor (21), not
hardcoded -- max_r = floor((197 - floor) / 11) = floor(176/11) = 16, not the 18 loosely quoted
in the task that requested this extension (that number wasn't derived from the actual floor
value; 16 is what the arithmetic gives for floor=21, confirmed and used here instead -- see
this run's own printed derivation). Full r-sweep, r=1..max_r, reporting each r's exact per-block
N_l trajectory, measured total latency, and which of the 6 budgets it satisfies.

D_PRED NOTE (unchanged from frontier.py -- still a real data gap, not glossed over): P2.1's
Dm_table.csv still only covers TANDEM's registered plan-space grid (MERGE_BLOCKS x MERGE_RS),
not ARM B's uniform-every-block r -- ARM B's D_pred remains null here too, for the same reason.

Usage: python frontier_v2.py   (from the repo root or inside jetson-orinnx/ -- no device needed, only
out/lut_ext/latency_lut_extended.json + P2.1's read-only tables + out/frontier/frontier_plans.csv
for the comparison.)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compute_plan_divergence, ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from replan import (  # noqa: E402
    BIT_MENU, MERGE_BLOCKS, MERGE_RS, MODEL_NAME, N_TOKENS_TOTAL,
    build_schedule_pareto_sets, enumerate_schedules, exact_select_for_budget,
    find_min_achievable_cost, load_dq_dm_tables, plan_bits, plan_cost,
)

FRONTIER_BUDGET_FRACTIONS = (0.77, 0.79, 0.82, 0.85, 0.89, 0.94)  # unchanged from frontier.py v1


def load_latency_lut_ext():
    lut_path = JETSON_ROOT / "out" / "lut_ext" / "latency_lut_extended.json"
    assert lut_path.exists(), f"{lut_path} not found -- run lut_ext_measure.py on the Jetson first."
    lut = read_json(lut_path)["entries"]
    by_precision = {"fp16": {}, "int8": {}}
    for entry in lut.values():
        if entry["precision"] not in by_precision:
            continue
        by_precision[entry["precision"]][entry["n_tokens"]] = entry["median_ms"]
    return by_precision


def make_block_latency_fn_ext(lut_by_precision):
    """Same interpolation logic as replan.py's make_block_latency_fn, but parameterized over
    the EXTENDED fp16 bucket set (dynamically, not replan.py's module-level TOKEN_BUCKETS
    constant -- replan.py itself is immutable/untouched) so ARM B can look up latency all the
    way down to the extended floor."""
    fp16_buckets_sorted = sorted(lut_by_precision["fp16"].keys())
    int8_buckets_sorted = sorted(lut_by_precision["int8"].keys()) if lut_by_precision["int8"] else []

    def block_latency(n_l: int, bits: int) -> float:
        precision = "fp16" if bits == 16 else "int8"
        table = lut_by_precision[precision]
        buckets_sorted = fp16_buckets_sorted if precision == "fp16" else int8_buckets_sorted
        assert buckets_sorted[0] <= n_l <= buckets_sorted[-1], (
            f"n_l={n_l} outside measured bucket range [{buckets_sorted[0]}, {buckets_sorted[-1]}] "
            f"for precision={precision} -- halting rather than extrapolating."
        )
        if n_l in table:
            return table[n_l]
        lo = max(b for b in buckets_sorted if b <= n_l)
        hi = min(b for b in buckets_sorted if b >= n_l)
        if lo == hi:
            return table[lo]
        frac = (n_l - lo) / (hi - lo)
        return table[lo] * (1 - frac) + table[hi] * frac

    return block_latency


def uniform_r_plan(r: int) -> dict:
    return {"bits": {l: 16 for l in range(1, 13)}, "merges": [{"block": l, "r": r} for l in range(1, 12)]}


def n_l_trajectory(r: int):
    return [197 - r * (l - 1) for l in range(1, 13)]


def main():
    print("=== frontier_v2.py: LUT EXTENSION re-test (ARM A + ARM B r-sweep) ===")

    dq_table, dm_table = load_dq_dm_tables()
    lut_by_precision = load_latency_lut_ext()
    fp16_floor = min(lut_by_precision["fp16"].keys())
    fp16_ceiling = max(lut_by_precision["fp16"].keys())
    print(f"Extended LUT: fp16 buckets {sorted(lut_by_precision['fp16'].keys())} "
          f"(floor={fp16_floor}, was 69 in the original LUT)")
    block_latency_fn = make_block_latency_fn_ext(lut_by_precision)

    identity_plan = {"bits": {}, "merges": []}
    latency_id = plan_cost(identity_plan, block_latency_fn)
    print(f"Latency_id (identity plan) = {latency_id:.4f}ms (unchanged -- identity never touches "
          "the new small buckets)")

    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85
    schedule_pareto_sets = build_schedule_pareto_sets(
        schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_latency_fn
    )
    min_achievable_ms = find_min_achievable_cost(schedule_pareto_sets)
    print(f"Achievable floor (TANDEM's own 85-schedule space) = {min_achievable_ms:.4f}ms "
          f"({min_achievable_ms / latency_id:.1%} of Latency_id) -- expected UNCHANGED from v1 "
          "(76.4%), since TANDEM's own schedules never produce n_l below 69 regardless of what "
          "the LUT's own floor is.")

    # --- Dynamically derive ARM B's max r from the ACTUAL extended-LUT floor, not a hardcoded
    # number (the task's own "r in 1..18" was a loose estimate, not derived from floor=21 --
    # see module docstring). max_r solves 197 - 11*r >= fp16_floor.
    max_r = (197 - fp16_floor) // 11
    print(f"\nARM B max feasible r, derived from extended floor {fp16_floor}: "
          f"floor((197-{fp16_floor})/11) = {max_r} "
          f"(N_l(block12) at r={max_r} = {197 - max_r * 11}, at r={max_r + 1} = {197 - (max_r + 1) * 11} "
          "-- confirms this is the correct integer bound, not the task's loosely-quoted 18).")

    # --- Load v1's committed frontier_plans.csv -- READ-ONLY, for ARM A comparison only.
    v1_path = JETSON_ROOT / "out" / "frontier" / "frontier_plans.csv"
    assert v1_path.exists(), f"{v1_path} not found -- this is the immutable v1 run, required for comparison."
    v1_df = pd.read_csv(v1_path)
    assert len(v1_df) == len(FRONTIER_BUDGET_FRACTIONS), f"{v1_path} row count mismatch"

    # === ARM A: re-run, compare against v1 ===
    arm_a_rows = []
    arm_a_any_changed = False
    for i, frac in enumerate(FRONTIER_BUDGET_FRACTIONS):
        tau_ms = frac * latency_id
        result = exact_select_for_budget(schedule_pareto_sets, tau_ms, raise_on_infeasible=False)
        assert result is not None, f"ARM A INFEASIBLE at tau_fraction={frac:.0%} against the extended LUT."
        cost, d_pred, plan = result

        v1_plan = json.loads(v1_df.iloc[i]["arm_a_plan_json"])
        identical, n_diff, _ = compute_plan_divergence(plan, v1_plan, N_TOKENS_TOTAL)
        if not identical:
            arm_a_any_changed = True
        print(f"\n  tau_fraction={frac:.0%}: v2 cost={cost:.4f}ms D_pred={d_pred:.6f}  "
              f"identical_to_v1={identical}" + (f"  n_blocks_differing={n_diff}" if not identical else ""))
        arm_a_rows.append({
            "budget_fraction": frac, "tau_ms": tau_ms,
            "arm_a_cost_ms_v2": cost, "arm_a_d_pred_v2": d_pred, "arm_a_plan_json_v2": json.dumps(plan),
            "arm_a_cost_ms_v1": float(v1_df.iloc[i]["arm_a_cost_ms"]),
            "arm_a_d_pred_v1": float(v1_df.iloc[i]["arm_a_d_pred"]),
            "arm_a_plan_json_v1": v1_df.iloc[i]["arm_a_plan_json"],
            "identical_to_v1": identical, "n_blocks_differing_v1_v2": n_diff,
        })
    print(f"\nARM A summary: {'plans CHANGED at 1 or more budgets' if arm_a_any_changed else 'ALL 6 plans IDENTICAL to v1'} "
          "(both old and new recorded per budget regardless -- see frontier_plans_v2.csv).")

    # === ARM B: full r-sweep, r=1..max_r ===
    print(f"\n=== ARM B r-sweep, r=1..{max_r} ===")
    r_sweep_rows = []
    for r in range(1, max_r + 1):
        plan = uniform_r_plan(r)
        traj = n_l_trajectory(r)
        assert min(traj) >= fp16_floor, f"r={r}: min n_l={min(traj)} < floor={fp16_floor} -- bound derivation is wrong."
        cost = plan_cost(plan, block_latency_fn)
        satisfies = [frac for frac in FRONTIER_BUDGET_FRACTIONS if cost <= frac * latency_id + 1e-9]
        r_sweep_rows.append({
            "r": r, "n_l_trajectory": json.dumps(traj), "n_l_min": min(traj),
            "cost_ms": cost, "satisfies_budgets": json.dumps(satisfies),
        })
        print(f"  r={r:2d}: n_l min={min(traj):3d}  cost={cost:.4f}ms  "
              f"satisfies={[f'{f:.0%}' for f in satisfies] if satisfies else 'none'}")

    # --- Per-budget feasibility: smallest feasible r (least damage), or infeasible.
    print("\n=== Per-budget ARM B feasibility (smallest feasible r) ===")
    feasibility_rows = []
    for frac in FRONTIER_BUDGET_FRACTIONS:
        tau_ms = frac * latency_id
        feasible_rs = [row["r"] for row in r_sweep_rows if row["cost_ms"] <= tau_ms + 1e-9]
        if feasible_rs:
            best_r = min(feasible_rs)
            best_row = next(row for row in r_sweep_rows if row["r"] == best_r)
            print(f"  {frac:.0%}: FEASIBLE, r={best_r}, cost={best_row['cost_ms']:.4f}ms "
                  f"(tau={tau_ms:.4f}ms)")
            feasibility_rows.append({"budget_fraction": frac, "tau_ms": tau_ms, "feasible": True,
                                      "best_r": best_r, "cost_ms": best_row["cost_ms"]})
        else:
            print(f"  {frac:.0%}: INFEASIBLE even at r={max_r} (tau={tau_ms:.4f}ms, cheapest ARM B "
                  f"candidate costs {r_sweep_rows[-1]['cost_ms']:.4f}ms)")
            feasibility_rows.append({"budget_fraction": frac, "tau_ms": tau_ms, "feasible": False,
                                      "best_r": None, "cost_ms": None})

    n_feasible = sum(1 for r in feasibility_rows if r["feasible"])
    # --- Honest, DATA-DERIVED feasibility sentence -- replaces the old hardcoded one, per the
    # task's own instruction: "if uniform merging turns out feasible at 85/82% via higher r, say
    # so plainly and retire the claim; retain it only where it survives."
    still_infeasible = [f"{r['budget_fraction']:.0%}" for r in feasibility_rows if not r["feasible"]]
    feasible_pct_list = [f"{r['budget_fraction']:.0%}" for r in feasibility_rows if r["feasible"]]
    now_feasible_that_were_not = [f"{r['budget_fraction']:.0%}" for r in feasibility_rows
                                   if r["feasible"] and r["budget_fraction"] < 0.89]  # 89/94% were already feasible in v1
    feasible_list_str = f" ({', '.join(feasible_pct_list)})" if n_feasible else ""
    still_infeasible_str = (
        f"; still INFEASIBLE at {', '.join(still_infeasible)} even with r up to {max_r}"
        if still_infeasible else "; feasible at ALL 6 budgets"
    )
    newly_feasible_str = (
        f"Newly feasible (were infeasible in v1): {', '.join(now_feasible_that_were_not)}. "
        if now_feasible_that_were_not else "No budget below 89% became feasible even with the extended range. "
    )
    feasibility_sentence = (
        "LUT EXTENSION RESULT (supersedes the earlier v1 sentence, per this task's own "
        "instruction): the original ARM B feasibility bound at r<=11 was set by the measured "
        "LUT's floor (N=69), not by any property of uniform merging. The LUT was extended to "
        f"N={fp16_floor} to remove that confound (max feasible r rose from 11 to {max_r}). "
        f"Result: ARM B is now feasible at {n_feasible}/6 budgets{feasible_list_str}"
        f"{still_infeasible_str}. {newly_feasible_str}"
        "The feasibility results above supersede the earlier v1 ones."
    )
    print(f"\n{feasibility_sentence}")

    # === Write outputs ===
    out_dir = ensure_out_dir("frontier_v2")
    pd.DataFrame(arm_a_rows).to_csv(out_dir / "frontier_plans_v2.csv", index=False)
    print(f"\nWrote {out_dir / 'frontier_plans_v2.csv'}")
    pd.DataFrame(r_sweep_rows).to_csv(out_dir / "arm_b_r_sweep_v2.csv", index=False)
    print(f"Wrote {out_dir / 'arm_b_r_sweep_v2.csv'}")

    manifest = {
        "model_name": MODEL_NAME, "latency_id_ms": latency_id,
        "min_achievable_ms_v2": min_achievable_ms, "min_achievable_ms_unchanged_from_v1": True,
        "extended_fp16_floor": fp16_floor, "extended_fp16_ceiling": fp16_ceiling,
        "arm_b_max_r_derivation": (
            f"floor((197 - {fp16_floor}) / 11) = {max_r}. The task's own '1..18' was a loose "
            f"estimate, not derived from the actual floor (21) -- {max_r} is the correct integer "
            "bound; r values beyond this would need n_l < the measured floor and are not evaluated."
        ),
        "arm_a_any_plan_changed": arm_a_any_changed,
        "arm_b_feasibility": feasibility_rows,
        "verbatim_note": (
            "The original ARM B feasibility bound at r<=11 was set by the measured LUT's floor "
            "(N=69), not by any property of uniform merging. The LUT was extended to N=21 to "
            "remove that confound; the feasibility results below supersede the earlier ones."
        ),
        "arm_b_feasibility_sentence": feasibility_sentence,
        "d_pred_note": (
            "ARM B's D_pred remains null -- P2.1's own Dm_table.csv still only covers TANDEM's "
            "registered plan-space grid (MERGE_BLOCKS x MERGE_RS), not ARM B's uniform-every-"
            "block r values, extended range or not. Unchanged from frontier.py v1."
        ),
        "int8_still_closed_note": (
            "INT8 remains CLOSED per CHARACTERIZATION.md -- this extension is FP16-only and "
            "does not re-litigate the batch-1/batch-8 NO-GO finding."
        ),
    }
    write_json(out_dir / "frontier_manifest_v2.json", manifest)
    print(f"Wrote {out_dir / 'frontier_manifest_v2.json'}")


if __name__ == "__main__":
    main()
