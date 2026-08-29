"""E5/G1 headline-claims verifier -- READ-ONLY, no model, no GPU, no re-sweep.

Recomputes each claim below directly from committed artifacts under
experiments/phase1_E1-E5_E4a/results/E5/ and experiments/phase1_E1-E5_E4a/results/E4/, and checks the recomputed
value against what experiments/phase1_E1-E5_E4a/E5_allocator_G1.ipynb itself stored. This is
the standing pre-presentation verifier for E5/G1 (verdict
V2_E5_G1_kill_test) -- the sixth independent check in the verification
ledger (production run; a full bit-for-bit rerun; Claude Code's from-scratch
solver reimplementation; a separate artifact analysis; a separate
frontier-mechanism check; and now this script), on top of the frontier
regression assert already living inside the notebook itself.

Claims verified (all pure arithmetic on committed CSV/JSON -- no torch, no
timm, no CUDA):
  1. Plans identical: joint plan == best_sequential plan at all 6 budgets
     (E5_plans.csv), cross-checked against E5_summary.json's own
     `plans_identical` field.
  2. Exact tie: the G1 kill test's mean delta_top1 == 0.0 with a zero-width
     CI (E5_summary.json) -- the NECESSARY CONSEQUENCE of (1), not an
     independent re-derivation (this script never touches G1's raw
     20k-image data; there is nothing to re-sweep here).
  3. Frontier mechanism: recompute the token damage/savings frontier from
     E4_raw.csv's F_S family + the kappa correction, INDEPENDENT of
     E5_schedule_token_frontier.csv (the schedule-space enumeration, cost
     model, and Pareto-frontier rule are re-implemented from scratch here,
     verbatim from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py, not imported from it).
     Assert 26/85 schedules are frontier-efficient and all 6 joint
     schedules are among them; then cross-check every one of the 85 rows
     against the stored CSV.
  4. Family-X cross-axis fit (V1_E5_cross_axis_additivity): recompute the
     through-origin slope/R^2 on the 10 menu-consistent E5_familyX.csv
     rows, match the summary to 1e-6.
  5. Corrected pooled fidelity fit (pred_vs_measured_fidelity_all_plans):
     recompute on the 22-point pool (10 menu-consistent Family-X rows +
     12 plan-audit rows; the 2 retained WA4 evidence rows stay excluded,
     per F-X1), match the summary to 1e-6.

Usage: python scripts/verify_g1_claims.py   (run from anywhere -- paths are
resolved relative to this file, not the caller's cwd)
"""
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
E5_DIR = REPO_ROOT / "experiments" / "phase1_E1-E5_E4a" / "results" / "E5"
E4_DIR = REPO_ROOT / "experiments" / "phase1_E1-E5_E4a" / "results" / "E4"
LAB_LOG_PATH = REPO_ROOT / "LAB_LOG.md"

# --- schedule-space / cost-model constants, verbatim copies from
# experiments/phase1_E1-E5_E4a/E5_allocator_G1.py (not an import -- this verifier must have
# zero dependency on the notebook's own runtime state to be a genuinely
# independent check).
D_MODEL = 384
N_TOKENS_TOTAL = 197
N_PATCHES = 196
MERGE_BLOCKS = (1, 3, 5, 7, 9, 11)
MERGE_RS = (32, 64, 98, 128)
KAPPA = {0: 1.0, 1: 1.0, 2: 1.15}


def enumerate_schedules(merge_blocks, merge_rs, max_points=2):
    """Verbatim from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's enumerate_schedules:
    empty + single-point + feasibility/regime-capped two-point schedules."""
    schedules = [[]]
    schedules += [[{"block": b, "r": r}] for b in merge_blocks for r in merge_rs]
    if max_points >= 2:
        for b1, b2 in combinations(merge_blocks, 2):
            for r1 in merge_rs:
                for r2 in merge_rs:
                    feasible = r2 <= N_PATCHES - r1 - 1
                    in_regime = r1 + r2 <= max(merge_rs)
                    if feasible and in_regime:
                        schedules.append([{"block": b1, "r": r1}, {"block": b2, "r": r2}])
    return schedules


def n_tokens_entering(merges, l):
    return N_TOKENS_TOTAL - sum(m["r"] for m in merges if m["block"] < l)


def linear_macs(n_l):
    return 12 * n_l * D_MODEL ** 2


def attention_macs(n_l):
    return 2 * n_l ** 2 * D_MODEL


def block_bops(n_l, bits):
    return linear_macs(n_l) * (bits * bits) + attention_macs(n_l) * (16 * 16)


def plan_cost(bits, merges):
    return sum(block_bops(n_tokens_entering(merges, l), bits.get(l, 16)) for l in range(1, 13))


def pareto_frontier(points, tol=1e-12):
    """Verbatim algorithm from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's
    pareto_frontier: sort by (cost, damage) ascending, keep a point iff its
    damage is strictly less than the best damage seen so far."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier, best_d = [], float("inf")
    for p in pts:
        if p[1] < best_d - tol:
            frontier.append(p)
            best_d = p[1]
    return frontier


def fit_through_origin(x, y):
    """Verbatim formula from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's fit_through_origin."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = float((x * y).sum() / (x ** 2).sum())
    ss_res = float(((y - s * x) ** 2).sum())
    ss_tot = float((y ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return s, r2, len(x)


def build_dm_table(e4_raw_df):
    """Verbatim from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's build_dm_table: Dm[(l,r)]
    = measured D_mse of Ward-merging r tokens at block l ALONE (E4's F_S family)."""
    fs = e4_raw_df[e4_raw_df["family"] == "F_S"]
    table = {}
    for _, row in fs.iterrows():
        (pt,) = json.loads(row["points_json"])
        table[(int(pt["block"]), int(pt["r"]))] = float(row["D_mse"])
    expected = {(l, r) for l in MERGE_BLOCKS for r in MERGE_RS}
    missing = expected - table.keys()
    assert not missing, f"Dm table missing entries from E4_raw.csv: {sorted(missing)}"
    return table


def load_artifacts():
    with open(E5_DIR / "E5_summary.json") as f:
        summary = json.load(f)
    plans_df = pd.read_csv(E5_DIR / "E5_plans.csv")
    familyx_df = pd.read_csv(E5_DIR / "E5_familyX.csv")
    frontier_df = pd.read_csv(E5_DIR / "E5_schedule_token_frontier.csv")
    e4_raw_df = pd.read_csv(E4_DIR / "E4_raw.csv")
    return summary, plans_df, familyx_df, frontier_df, e4_raw_df


def claim_1_plans_identical(summary, plans_df):
    joint = plans_df[plans_df["method"] == "joint"].set_index("budget")
    seq = plans_df[plans_df["method"] == "best_sequential"].set_index("budget")
    assert set(joint.index) == set(seq.index) and len(joint) == 6, (
        f"expected 6 budgets for both methods, got joint={len(joint)} best_sequential={len(seq)}"
    )
    recomputed_per_budget = {}
    for budget in sorted(joint.index):
        j_plan = json.loads(joint.loc[budget, "plan_json"])
        s_plan = json.loads(seq.loc[budget, "plan_json"])
        recomputed_per_budget[budget] = (j_plan == s_plan)
    recomputed_all = all(recomputed_per_budget.values())

    stored_per_budget = summary["plans_identical"]
    stored_all = all(stored_per_budget.values())

    mismatches = []
    for budget, val in recomputed_per_budget.items():
        key = f"{budget:.4e}"
        if key not in stored_per_budget:
            mismatches.append((key, "missing from summary.plans_identical"))
        elif bool(stored_per_budget[key]) != val:
            mismatches.append((key, "recomputed/stored value mismatch"))

    match = recomputed_all and stored_all and not mismatches
    return {
        "name": "1. plans_identical (joint == best_sequential, all 6 budgets)",
        "recomputed": f"all_identical={recomputed_all}",
        "stored": f"all_identical={stored_all}",
        "match": match,
        "detail": {"recomputed_per_budget": {f"{b:.4e}": v for b, v in recomputed_per_budget.items()},
                    "mismatches": mismatches},
    }, recomputed_all


def claim_2_exact_tie(summary, claim1_passed):
    kt = summary["verdicts"]["V2_E5_G1_kill_test"]
    stored_mean = kt["mean_delta_top1_pp"]
    stored_ci_lo = kt["mean_ci_low_pp"]
    stored_ci_hi = kt["mean_ci_high_pp"]
    is_exact_zero = (stored_mean == 0.0 and stored_ci_lo == 0.0 and stored_ci_hi == 0.0)
    match = is_exact_zero and claim1_passed
    return {
        "name": "2. exact tie (mean delta_top1_pp == 0.0, zero-width CI) -- necessary consequence of (1)",
        "recomputed": f"consequence_of_claim1_holding={claim1_passed}",
        "stored": f"mean_delta_top1_pp={stored_mean}, ci=[{stored_ci_lo}, {stored_ci_hi}]",
        "match": match,
        "detail": {"note": "NOT independently re-derived from G1's raw 20k-image data (no GPU, no "
                            "re-sweep, by design) -- verified as the logically necessary consequence "
                            "of bit-for-bit identical plans (claim 1): identical plans must score "
                            "identically on any dataset, so a nonzero delta here with claim 1 holding "
                            "would itself be a contradiction worth halting on."},
    }


def claim_3_frontier(e4_raw_df, plans_df, frontier_df):
    dm_table = build_dm_table(e4_raw_df)
    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules, enumerated {len(schedules)}"
    assert len(frontier_df) == 85, f"expected 85 rows in E5_schedule_token_frontier.csv, found {len(frontier_df)}"

    c_id = plan_cost({}, [])
    points = []
    for s in schedules:
        kappa = KAPPA.get(len(s), KAPPA[2])
        damage = kappa * sum(dm_table[(m["block"], m["r"])] for m in s) if s else 0.0
        cost = plan_cost({}, s)
        points.append((cost, damage, s))
    frontier_pts = pareto_frontier(points)
    frontier_schedules = [p[2] for p in frontier_pts]
    n_frontier = len(frontier_schedules)

    joint_schedules = [json.loads(pj)["merges"] for pj in plans_df[plans_df["method"] == "joint"]["plan_json"]]
    assert len(joint_schedules) == 6, f"expected 6 joint schedules, found {len(joint_schedules)}"
    all_on_frontier = all(s in frontier_schedules for s in joint_schedules)

    mismatches = []
    stored_by_schedule = {row["schedule"]: row for _, row in frontier_df.iterrows()}
    for cost, damage, s in points:
        key = json.dumps(s)
        stored_row = stored_by_schedule.get(key)
        if stored_row is None:
            mismatches.append((key, "missing from stored CSV"))
            continue
        recomputed_on_frontier = s in frontier_schedules
        if bool(stored_row["on_token_frontier"]) != recomputed_on_frontier:
            mismatches.append((key, "on_token_frontier mismatch"))
        if abs(float(stored_row["merge_damage_kappa"]) - damage) > 1e-9:
            mismatches.append((key, f"damage mismatch (recomputed={damage}, stored={stored_row['merge_damage_kappa']})"))
        if abs(float(stored_row["fp16_costed_savings"]) - (c_id - cost)) > 1e-3:
            mismatches.append((key, "savings mismatch"))

    passed = (n_frontier == 26) and all_on_frontier and not mismatches
    return {
        "name": "3. frontier mechanism (26/85 frontier-efficient, all 6 joint schedules on it)",
        "recomputed": f"n_frontier={n_frontier}/85, all_6_joint_on_frontier={all_on_frontier}, "
                       f"csv_cross_check_mismatches={len(mismatches)}/85",
        "stored": "26/85 (E5_summary.json note); on_token_frontier column in E5_schedule_token_frontier.csv",
        "match": passed,
        "detail": {"c_id_recomputed": c_id, "mismatches": mismatches[:10]},
    }


def claim_4_family_x_fit(summary, familyx_df):
    menu_consistent = familyx_df[~familyx_df["excluded_out_of_menu"]]
    assert len(menu_consistent) == 10, f"expected 10 menu-consistent rows, got {len(menu_consistent)}"
    slope, r2, n = fit_through_origin(menu_consistent["D_pred"], menu_consistent["D_mse"])
    stored = summary["verdicts"]["V1_E5_cross_axis_additivity"]
    match = (abs(slope - stored["slope"]) < 1e-6 and abs(r2 - stored["r2"]) < 1e-6 and n == stored["n_points"])
    return {
        "name": "4. Family-X cross-axis fit (V1_E5_cross_axis_additivity)",
        "recomputed": f"slope={slope!r}, r2={r2!r}, n={n}",
        "stored": f"slope={stored['slope']!r}, r2={stored['r2']!r}, n={stored['n_points']}",
        "match": match, "detail": {},
    }


def claim_5_pooled_fidelity_fit(summary, familyx_df, plans_df):
    menu_consistent = familyx_df[~familyx_df["excluded_out_of_menu"]][["D_pred", "D_mse"]]
    plan_audit = plans_df[["D_pred"]].assign(D_mse=plans_df["D_mse_5k"])
    pooled = pd.concat([menu_consistent, plan_audit], ignore_index=True)
    assert len(pooled) == 22, f"expected 22 pooled points (10 menu-consistent + 12 plan-audit), got {len(pooled)}"
    slope, r2, n = fit_through_origin(pooled["D_pred"], pooled["D_mse"])
    stored = summary["pred_vs_measured_fidelity_all_plans"]
    match = (abs(slope - stored["slope"]) < 1e-6 and abs(r2 - stored["r2"]) < 1e-6 and n == stored["n_points"])
    return {
        "name": "5. corrected pooled fidelity fit (pred_vs_measured_fidelity_all_plans)",
        "recomputed": f"slope={slope!r}, r2={r2!r}, n={n}",
        "stored": f"slope={stored['slope']!r}, r2={stored['r2']!r}, n={stored['n_points']}",
        "match": match, "detail": {},
    }


def append_lab_log_line_once():
    """Idempotent: only ever appends once, guarded by a marker string --
    re-running this verifier on later dates must not spam LAB_LOG.md."""
    marker = "verify_g1_claims.py"
    if LAB_LOG_PATH.exists() and marker in LAB_LOG_PATH.read_text(encoding="utf-8"):
        return False
    line = (
        f"\nE5/G1 standing pre-presentation verifier (scripts/verify_g1_claims.py) created and "
        f"passed on first run ({datetime.now().strftime('%Y-%m-%d')}): read-only, no GPU, no "
        "re-sweep -- recomputes plans-identical, the exact-tie consequence, the 26/85 token "
        "frontier (independent of the stored CSV), the Family-X fit, and the corrected pooled "
        "fidelity fit directly from committed E4/E5 artifacts, all MATCH. Sixth independent "
        "verification of the E5/G1 headline result; standing check for future re-presentation.\n"
    )
    with open(LAB_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    return True


def main():
    summary, plans_df, familyx_df, frontier_df, e4_raw_df = load_artifacts()

    c1, c1_passed = claim_1_plans_identical(summary, plans_df)
    c2 = claim_2_exact_tie(summary, c1_passed)
    c3 = claim_3_frontier(e4_raw_df, plans_df, frontier_df)
    c4 = claim_4_family_x_fit(summary, familyx_df)
    c5 = claim_5_pooled_fidelity_fit(summary, familyx_df, plans_df)
    claims = [c1, c2, c3, c4, c5]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 100)
    print(f"E5/G1 HEADLINE CLAIMS -- READ-ONLY VERIFICATION -- {now}")
    print("(no model, no GPU, no re-sweep -- recomputed from experiments/phase1_E1-E5_E4a/results/{E4,E5}/ artifacts)")
    print("=" * 100)
    all_passed = True
    for c in claims:
        status = "MATCH" if c["match"] else "FAIL"
        if not c["match"]:
            all_passed = False
        print(f"\n[{status}] {c['name']}")
        print(f"  recomputed: {c['recomputed']}")
        print(f"  stored:     {c['stored']}")
        if not c["match"]:
            print(f"  detail:     {c['detail']}")

    print("\n" + "=" * 100)
    if all_passed:
        print("ALL CLAIMS VERIFIED")
        wrote = append_lab_log_line_once()
        print("LAB_LOG.md: appended first-successful-run line." if wrote
              else "LAB_LOG.md: marker already present -- no line added (not the first successful run).")
    else:
        print("SOME CLAIMS FAILED -- see [FAIL] lines above. Do not present this result until diagnosed.")
    print("=" * 100)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
