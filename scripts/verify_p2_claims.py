"""P2 headline-claims verifier -- READ-ONLY, no model, no GPU, no re-sweep.

Recomputes each claim below directly from committed artifacts under
experiments/phase2_1_fleet/results/P2/<model>/ for all four models
(deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224,
vit_base_patch16_224), and checks the recomputed value against what
P2_<model>_summary.json itself stored. This is the standing pre-share
verifier for P2.1 -- run this before floating any P2 table/plot/verdict.

Claims verified per model (all pure arithmetic on committed CSV/JSON -- no
torch, no timm, no CUDA):
  1. Family-X fit (family_x_gate): recompute the through-origin slope/R^2
     on familyX.csv, match the summary to 1e-6; recompute `passed` from
     the 0.8<=slope<=1.2 and r2>=0.85 rule and match the stored bool.
  2. S8 sanity gate (sanity_gate_s8): recompute |measured - published|
     and match diff_pct_points; recompute `passed` (<=0.5pp, or the
     waiver/no-reference tri-state resolved the same way the notebook's
     own gate resolves it) and match the stored bool.
  3. Plan D_pred values: recompute objective_d_pred(plan, Dq, Dm, kappa)
     for every row of BOTH plans.csv and familyX.csv, independently from
     Dq_table.csv/Dm_table.csv, and match each row's own stored D_pred.
  4. C_id/C_min: recompute the BOPs cost model (verbatim from
     P2_tandem_pipeline.py's plan_cost/block_bops) using THIS model's own
     recorded dims (model_dims.d/n_tokens_total), enumerate all 85
     schedules, and match the summary's c_id/c_min.
  5. Per-budget fidelity ratios (s7_binding_gate.per_budget_ratios):
     recompute D_mse_5k/D_pred per plans.csv row and match every stored
     per-budget value.

Usage: python scripts/verify_p2_claims.py   (run from anywhere -- paths
are resolved relative to this file, not the caller's cwd)
"""
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
P2_DIR = REPO_ROOT / "experiments" / "phase2_1_fleet" / "results" / "P2"
MODELS = (
    "deit_tiny_patch16_224", "deit_small_patch16_224",
    "deit_base_patch16_224", "vit_base_patch16_224",
)

# --- cost-model / plan constants, verbatim copies from
# experiments/phase2_1_fleet/P2_tandem_pipeline.py (not an import -- this verifier must have
# zero dependency on the notebook's own runtime state to be a genuinely
# independent check). D_MODEL/N_TOKENS_TOTAL are NOT fixed here (unlike
# E5's single-model verify_g1_claims.py) -- P2 spans four different
# architectures, so those are read per-model from each summary's own
# model_dims and threaded through explicitly below.
MERGE_BLOCKS = (1, 3, 5, 7, 9, 11)
MERGE_RS = (32, 64, 98, 128)
QUANT_BITS = (6, 8)
BIT_MENU = (16, 8, 6)
KAPPA = {0: 1.0, 1: 1.0, 2: 1.15}
N_PATCHES = 196  # fixed 14x14 grid, patch16_224, all four models


def enumerate_schedules(merge_blocks, merge_rs, max_points=2):
    """Verbatim from P2_tandem_pipeline.py's enumerate_schedules: empty +
    single-point + feasibility/regime-capped two-point schedules."""
    schedules = [[]]
    schedules += [[{"block": b, "r": r}] for b in merge_blocks for r in merge_rs]
    if max_points >= 2:
        for b1, b2 in itertools.combinations(merge_blocks, 2):
            for r1 in merge_rs:
                for r2 in merge_rs:
                    feasible = r2 <= N_PATCHES - r1 - 1
                    in_regime = r1 + r2 <= max(merge_rs)
                    if feasible and in_regime:
                        schedules.append([{"block": b1, "r": r1}, {"block": b2, "r": r2}])
    return schedules


def n_tokens_entering(merges, l, n_tokens_total):
    return n_tokens_total - sum(m["r"] for m in merges if m["block"] < l)


def linear_macs(n_l, d_model):
    return 12 * n_l * d_model ** 2


def attention_macs(n_l, d_model):
    return 2 * n_l ** 2 * d_model


def block_bops(n_l, bits, d_model):
    return linear_macs(n_l, d_model) * (bits * bits) + attention_macs(n_l, d_model) * (16 * 16)


def plan_bits(bits_dict, l):
    return bits_dict.get(str(l), bits_dict.get(l, 16))


def plan_cost(plan, d_model, n_tokens_total):
    """Verbatim from P2_tandem_pipeline.py's plan_cost."""
    total = 0.0
    for l in range(1, 13):
        n_l = n_tokens_entering(plan["merges"], l, n_tokens_total)
        total += block_bops(n_l, plan_bits(plan["bits"], l), d_model)
    return total


def objective_d_pred(plan, dq_table, dm_table):
    """Verbatim from P2_tandem_pipeline.py's objective_d_pred."""
    d_quant = sum(
        dq_table[(l, plan_bits(plan["bits"], l))]
        for l in range(1, 13) if plan_bits(plan["bits"], l) != 16
    )
    d_merge = sum(dm_table[(m["block"], m["r"])] for m in plan["merges"])
    kappa = KAPPA.get(len(plan["merges"]), KAPPA[2])
    return d_quant + kappa * d_merge


def fit_through_origin(x, y):
    """Verbatim formula from P2_tandem_pipeline.py's fit_through_origin."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = float((x * y).sum() / (x ** 2).sum())
    ss_res = float(((y - s * x) ** 2).sum())
    ss_tot = float((y ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return s, r2, len(x)


def load_model_artifacts(model_name):
    model_dir = P2_DIR / model_name
    with open(model_dir / f"P2_{model_name}_summary.json") as f:
        summary = json.load(f)
    dq_df = pd.read_csv(model_dir / "Dq_table.csv")
    dm_df = pd.read_csv(model_dir / "Dm_table.csv")
    dq_table = {(int(r["block"]), int(r["bits"])): float(r["D_mse"]) for _, r in dq_df.iterrows()}
    dm_table = {(int(r["block"]), int(r["r"])): float(r["D_mse"]) for _, r in dm_df.iterrows()}
    plans_df = pd.read_csv(model_dir / "plans.csv")
    familyx_df = pd.read_csv(model_dir / "familyX.csv")
    return summary, dq_table, dm_table, plans_df, familyx_df


def claim_family_x(model_name, summary, familyx_df):
    slope, r2, n = fit_through_origin(familyx_df["D_pred"], familyx_df["D_mse"])
    passed_recomputed = bool(0.8 <= slope <= 1.2 and r2 >= 0.85)
    stored = summary["family_x_gate"]
    match = (
        abs(slope - stored["slope"]) < 1e-6 and abs(r2 - stored["r2"]) < 1e-6
        and n == stored["n_points"] and passed_recomputed == bool(stored["passed"])
    )
    return {
        "name": f"[{model_name}] 1. Family-X fit (family_x_gate)",
        "recomputed": f"slope={slope!r}, r2={r2!r}, n={n}, passed={passed_recomputed}",
        "stored": f"slope={stored['slope']!r}, r2={stored['r2']!r}, n={stored['n_points']}, "
                  f"passed={stored['passed']}",
        "match": match, "detail": {},
    }


def claim_s8_gate(model_name, summary):
    s8 = summary["sanity_gate_s8"]
    published = s8.get("published_top1_pct")
    measured = s8["measured_top1_pct"]
    if published is not None:
        diff = abs(measured - published)
        passed_recomputed = bool(diff <= 0.5)
        diff_match = abs(diff - (s8.get("diff_pct_points") or float("nan"))) < 1e-9
        recomputed_desc = f"diff={diff!r}pp, passed={passed_recomputed}"
    elif s8.get("waived"):
        # No published reference, but an explicit waiver -- the
        # notebook's own gate resolves this to passed=True (never a bare
        # null), diff undefined/not applicable.
        passed_recomputed = True
        diff_match = True
        recomputed_desc = f"no published reference, waived={s8.get('waiver_reason')!r}, passed=True"
    else:
        # No published reference, no waiver -- the notebook's own gate
        # resolves this to passed=False (never a bare null).
        passed_recomputed = False
        diff_match = True
        recomputed_desc = "no published reference, no waiver, passed=False"
    match = diff_match and passed_recomputed == bool(s8["passed"])
    return {
        "name": f"[{model_name}] 2. S8 sanity gate (sanity_gate_s8)",
        "recomputed": recomputed_desc,
        "stored": f"published={published!r}, measured={measured!r}, "
                  f"diff_pct_points={s8.get('diff_pct_points')!r}, waived={s8.get('waived')}, "
                  f"passed={s8['passed']}",
        "match": match, "detail": {},
    }


def claim_d_pred(model_name, dq_table, dm_table, plans_df, familyx_df):
    mismatches = []
    for label, df, dpred_col in (("plans.csv", plans_df, "D_pred"), ("familyX.csv", familyx_df, "D_pred")):
        for _, row in df.iterrows():
            plan = json.loads(row["plan_json"])
            recomputed = objective_d_pred(plan, dq_table, dm_table)
            stored = float(row[dpred_col])
            if abs(recomputed - stored) > 1e-9:
                mismatches.append((label, row.get("budget", row.get("config_id")), recomputed, stored))
    n_checked = len(plans_df) + len(familyx_df)
    match = not mismatches
    return {
        "name": f"[{model_name}] 3. Plan D_pred values (from Dq/Dm tables + kappa)",
        "recomputed": f"{n_checked - len(mismatches)}/{n_checked} rows match exactly",
        "stored": "D_pred column in plans.csv and familyX.csv",
        "match": match, "detail": {"mismatches": mismatches[:10]},
    }


def claim_c_id_c_min(model_name, summary, dq_table):
    d_model = summary["model_dims"]["d"]
    n_tokens_total = summary["model_dims"]["n_tokens_total"]

    c_id_recomputed = plan_cost({"bits": {}, "merges": []}, d_model, n_tokens_total)
    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    n_schedules_ok = len(schedules) == 85
    cheapest_bits = min(BIT_MENU)
    all_cheapest = {l: cheapest_bits for l in range(1, 13)}
    costs = [plan_cost({"bits": all_cheapest, "merges": s}, d_model, n_tokens_total) for s in schedules]
    c_min_recomputed = min(costs)

    c_id_stored, c_min_stored = summary["c_id"], summary["c_min"]
    match = (
        n_schedules_ok
        and abs(c_id_recomputed - c_id_stored) / c_id_stored < 1e-6
        and abs(c_min_recomputed - c_min_stored) / c_min_stored < 1e-6
    )
    return {
        "name": f"[{model_name}] 4. C_id/C_min (BOPs formula, d={d_model}, n_tokens_total={n_tokens_total})",
        "recomputed": f"n_schedules={len(schedules)}, C_id={c_id_recomputed!r}, C_min={c_min_recomputed!r}",
        "stored": f"C_id={c_id_stored!r}, C_min={c_min_stored!r}",
        "match": match, "detail": {},
    }


def claim_fidelity_ratios(model_name, summary, plans_df):
    stored_ratios = summary.get("s7_binding_gate", {}).get("per_budget_ratios", {})
    mismatches = []
    n_checked = 0
    for _, row in plans_df.iterrows():
        budget = float(row["budget"])
        recomputed = float(row["D_mse_5k"]) / float(row["D_pred"])
        key = f"{budget:.4e}"
        n_checked += 1
        stored = stored_ratios.get(key)
        if stored is None or abs(recomputed - stored) > 1e-9:
            mismatches.append((key, recomputed, stored))
    match = not mismatches and n_checked == len(stored_ratios)
    return {
        "name": f"[{model_name}] 5. Per-budget fidelity ratios (s7_binding_gate.per_budget_ratios)",
        "recomputed": f"{n_checked - len(mismatches)}/{n_checked} budgets match exactly",
        "stored": f"{len(stored_ratios)} budgets recorded",
        "match": match, "detail": {"mismatches": mismatches[:10]},
    }


def verify_model(model_name):
    summary, dq_table, dm_table, plans_df, familyx_df = load_model_artifacts(model_name)
    return [
        claim_family_x(model_name, summary, familyx_df),
        claim_s8_gate(model_name, summary),
        claim_d_pred(model_name, dq_table, dm_table, plans_df, familyx_df),
        claim_c_id_c_min(model_name, summary, dq_table),
        claim_fidelity_ratios(model_name, summary, plans_df),
    ]


def main():
    all_claims = []
    for model_name in MODELS:
        model_dir = P2_DIR / model_name
        if not (model_dir / f"P2_{model_name}_summary.json").exists():
            print(f"SKIPPING {model_name}: no summary found at {model_dir} (model not yet run/pushed).")
            continue
        all_claims.extend(verify_model(model_name))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 100)
    print(f"P2 HEADLINE CLAIMS -- READ-ONLY VERIFICATION -- {now}")
    print("(no model, no GPU, no re-sweep -- recomputed from experiments/phase2_1_fleet/results/P2/<model>/ artifacts)")
    print("=" * 100)
    all_passed = True
    for c in all_claims:
        status = "MATCH" if c["match"] else "FAIL"
        if not c["match"]:
            all_passed = False
        print(f"\n[{status}] {c['name']}")
        print(f"  recomputed: {c['recomputed']}")
        print(f"  stored:     {c['stored']}")
        if not c["match"]:
            print(f"  detail:     {c['detail']}")

    print("\n" + "=" * 100)
    if not all_claims:
        print("NO MODELS VERIFIED -- no P2 summaries found under experiments/phase2_1_fleet/results/P2/.")
        all_passed = False
    elif all_passed:
        print(f"ALL CLAIMS VERIFIED ({len(all_claims)} checks across "
              f"{len({c['name'].split(']')[0] for c in all_claims})} models)")
    else:
        print("SOME CLAIMS FAILED -- see [FAIL] lines above. Do not present this result until diagnosed.")
    print("=" * 100)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
