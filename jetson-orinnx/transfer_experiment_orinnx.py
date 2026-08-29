"""jetson-orinnx/transfer_experiment_orinnx.py -- PB-6 TRANSFER EXPERIMENT, reshaped
(no jetson/ equivalent -- new science, per the campaign's own reuse of replan.py's
DP machinery). ImageNet unavailable this campaign (see 06_MISSING.md) so the
accuracy-gap version of this experiment cannot run; this is the cost-side
substitute, per explicit ruling.

Definition:
  - Select plans for a budget expressed in ORIN NX MILLISECONDS, using the Orin
    Nano's OWN cost table (read-only, from the frozen jetson/ mirror -- never
    written to).
  - Measure what those Nano-table-selected plans ACTUALLY cost when evaluated
    against the NX's OWN measured table.
  - Separately select plans using the NX's own table for the same budget.
  - Report the MEASURED COST GAP and the PLAN-SELECTION DIVERGENCE (blocks
    differing) between the two plan sets.

BUDGET CHOICE: the pre-registered BUDGET_FRACTIONS (0.15-0.80) do not reach this
device's achievable range (real range: 76.7%-100% of Latency_id -- see PORT_NOTES.md
§12). Budgets here are chosen INSIDE the NX's own achievable range instead, so the
allocator faces a real decision -- these are explicitly NOT the pre-registered
fractions. This substitution is itself part of the finding (see PORT_NOTES.md,
FINDINGS.md).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import compute_plan_divergence, ensure_out_dir, read_json, write_json  # noqa: E402
from replan import (  # noqa: E402
    BIT_MENU, MERGE_BLOCKS, MERGE_RS, N_TOKENS_TOTAL,
    build_schedule_pareto_sets, enumerate_schedules, exact_select_for_budget,
    find_min_achievable_cost, load_dq_dm_tables, make_block_latency_fn,
    objective_d_pred, plan_cost,
)

NANO_LUT_PATH = Path("/home/tandem-jetson/jetson/out/lut/latency_lut.json")  # READ ONLY, frozen mirror
if not NANO_LUT_PATH.exists():
    NANO_LUT_PATH = Path(__file__).resolve().parent.parent / "jetson/out/lut/latency_lut.json"


def load_lut_by_precision(path: Path):
    entries = read_json(path)["entries"]
    by_precision = {"fp16": {}, "int8": {}}
    for entry in entries.values():
        if entry["precision"] in by_precision:
            by_precision[entry["precision"]][entry["n_tokens"]] = entry["median_ms"]
    return by_precision


def main():
    assert NANO_LUT_PATH.exists(), f"{NANO_LUT_PATH} not found -- frozen jetson/ mirror missing."
    print(f"Nano LUT (READ ONLY): {NANO_LUT_PATH}")

    lut_nx = load_lut_by_precision(Path("out/lut/latency_lut.json"))
    lut_nano = load_lut_by_precision(NANO_LUT_PATH)
    block_fn_nx = make_block_latency_fn(lut_nx)
    block_fn_nano = make_block_latency_fn(lut_nano)

    dq_table, dm_table = load_dq_dm_tables()

    identity_plan = {"bits": {}, "merges": []}
    latency_id_nx = plan_cost(identity_plan, block_fn_nx)
    latency_id_nano = plan_cost(identity_plan, block_fn_nano)
    print(f"Latency_id (NX, this device's own table)   = {latency_id_nx:.4f} ms")
    print(f"Latency_id (Nano's table, read-only)        = {latency_id_nano:.4f} ms")

    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules, got {len(schedules)}"

    print("\nBuilding Pareto sets under NX's own table...")
    pareto_nx = build_schedule_pareto_sets(schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_fn_nx)
    print("Building Pareto sets under Nano's table (read-only)...")
    pareto_nano = build_schedule_pareto_sets(schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_fn_nano)

    min_achievable_nx = find_min_achievable_cost(pareto_nx)
    min_achievable_frac_nx = min_achievable_nx / latency_id_nx
    print(f"\nNX achievable range: [{min_achievable_frac_nx:.1%}, 100%] of Latency_id "
          f"({min_achievable_nx:.4f}ms - {latency_id_nx:.4f}ms)")

    # Budgets INSIDE the NX achievable range -- NOT the pre-registered BUDGET_FRACTIONS.
    budget_fracs = [0.80, 0.85, 0.90, 0.95]
    print(f"Budget fractions used (of NX's own Latency_id, NOT pre-registered BUDGET_FRACTIONS): "
          f"{budget_fracs}")

    rows = []
    for bf in budget_fracs:
        budget_ms = bf * latency_id_nx
        print(f"\n=== budget_fraction(NX)={bf:.0%}  budget={budget_ms:.4f}ms ===")

        result_nano = exact_select_for_budget(pareto_nano, budget_ms, raise_on_infeasible=False)
        result_nx = exact_select_for_budget(pareto_nx, budget_ms, raise_on_infeasible=False)

        if result_nano is None:
            print(f"  Nano-table selection: INFEASIBLE at this budget under Nano's own table.")
        if result_nx is None:
            print(f"  NX-table selection: INFEASIBLE at this budget under NX's own table (should not "
                  f"happen -- budget chosen inside NX's own achievable range).")
        if result_nano is None or result_nx is None:
            rows.append({"budget_fraction_nx": bf, "budget_ms": budget_ms,
                         "feasible_nano": result_nano is not None, "feasible_nx": result_nx is not None})
            continue

        nano_predicted_cost, nano_predicted_dpred, plan_nano_selected = result_nano
        nx_native_cost, nx_native_dpred, plan_nx_selected = result_nx

        # MEASURE what the Nano-table-selected plan ACTUALLY costs on the NX's own table.
        real_cost_on_nx = plan_cost(plan_nano_selected, block_fn_nx)
        real_dpred_of_nano_plan = objective_d_pred(plan_nano_selected, dq_table, dm_table)

        cost_gap_ms = real_cost_on_nx - nx_native_cost
        cost_gap_pct = (cost_gap_ms / nx_native_cost) * 100.0
        over_budget = real_cost_on_nx > budget_ms

        plans_identical, n_blocks_differing, per_block_diff = compute_plan_divergence(
            plan_nano_selected, plan_nx_selected, N_TOKENS_TOTAL
        )

        print(f"  Nano-table PREDICTED cost for its own selected plan: {nano_predicted_cost:.4f}ms "
              f"(D_pred={nano_predicted_dpred:.6f})")
        print(f"  That SAME plan's REAL cost measured via NX's own table: {real_cost_on_nx:.4f}ms "
              f"(D_pred={real_dpred_of_nano_plan:.6f})  {'[OVER BUDGET]' if over_budget else '[within budget]'}")
        print(f"  NX-native selection at the same budget: {nx_native_cost:.4f}ms (D_pred={nx_native_dpred:.6f})")
        print(f"  MEASURED COST GAP (Nano-plan-real-cost minus NX-native-cost): "
              f"{cost_gap_ms:+.4f}ms ({cost_gap_pct:+.2f}%)")
        print(f"  PLAN-SELECTION DIVERGENCE: plans_identical={plans_identical}  "
              f"n_blocks_differing={n_blocks_differing}/12")

        rows.append({
            "budget_fraction_nx": bf, "budget_ms": budget_ms,
            "feasible_nano": True, "feasible_nx": True,
            "nano_predicted_cost_ms": nano_predicted_cost, "nano_predicted_dpred": nano_predicted_dpred,
            "nano_plan_real_cost_on_nx_ms": real_cost_on_nx, "nano_plan_real_dpred": real_dpred_of_nano_plan,
            "nano_plan_over_budget": over_budget,
            "nx_native_cost_ms": nx_native_cost, "nx_native_dpred": nx_native_dpred,
            "cost_gap_ms": cost_gap_ms, "cost_gap_pct": cost_gap_pct,
            "plans_identical": plans_identical, "n_blocks_differing": n_blocks_differing,
            "nano_plan_json": json.dumps(plan_nano_selected), "nx_plan_json": json.dumps(plan_nx_selected),
        })

    out_dir = ensure_out_dir("transfer")
    write_json(out_dir / "transfer_experiment_result.json", {
        "definition": "cost-side transfer experiment (accuracy unavailable -- see 06_MISSING.md)",
        "budget_fractions_used_of_nx_latency_id": budget_fracs,
        "note": "NOT the pre-registered BUDGET_FRACTIONS -- those do not reach this device's "
                "achievable range. See PORT_NOTES.md / FINDINGS.md.",
        "latency_id_nx_ms": latency_id_nx, "latency_id_nano_ms": latency_id_nano,
        "nx_achievable_range_frac": [min_achievable_frac_nx, 1.0],
        "rows": rows,
    })
    print(f"\nWrote {out_dir / 'transfer_experiment_result.json'}")


if __name__ == "__main__":
    main()
