"""Baselines (B1-B5) headline-claims verifier -- READ-ONLY, no GPU, no
subprocess, no re-run.

Recomputes each check below directly from committed artifacts under
experiments/phase2_2_baselines/ (results.csv, equivalence_fingerprint.json if present,
pareto_comparison.csv if present) using the SAME formulas as
common/repro_gate.py and compare_pareto.py -- reimplemented here verbatim,
not imported, so this verifier has zero dependency on the runtime state of
the code it's checking. This is the standing pre-share verifier for
B1-B5 -- run this before floating any baseline table or B5 comparison.

As of this writing, experiments/phase2_2_baselines/ has not yet been run for real:
results.csv has its header row only, and no equivalence_fingerprint.json /
pareto_comparison.csv exist yet. Every check below handles that
gracefully (reports "NO DATA YET", does not fail) -- this script is meant
to be run again, for real, once B1-B4 (and B5) have actually executed and
pushed results, at which point every check either MATCHes or FAILs on
real numbers.

Checks:
  1. Reproduction gate (results.csv): for every row with a located
     published reference (gate_status in {pass, fail}), recompute
     diff = abs(top1_50k - published_top1_pct) and status
     (<=0.3pp -> pass) via common/repro_gate.py's exact rule, and match
     both against the row's own gate_status/gate_diff_pct_points.
  2. Schema completeness (results.csv): every row has exactly the columns
     in common/results_io.py's RESULTS_COLUMNS, no missing/unexpected
     columns, and every row-defining key field (method, model, config,
     row_kind, checkpoint_tag) is non-empty.
  3. config_hash structural consistency (results.csv): the same
     (method, model, config, row_kind, checkpoint_tag) combination must
     never appear with two different config_hash values -- that would
     mean two runs claiming to be the same measurement actually used
     different pinned inputs.
  4. Equivalence fingerprint (equivalence_fingerprint.json, if present):
     no baseline/host entry recorded as halted (requirements_hash
     mismatch or delta-too-large) -- see common/equivalence_gate.py.
  5. B5 Pareto comparison (pareto_comparison.csv, if present): recompute
     matched_cost_delta_pp via linear interpolation of each model's own P2
     plans.csv (numpy.interp, verbatim from compare_pareto.py's
     interpolate_tandem_top1_pct), independent of the stored CSV, and
     match every row. NOTE: compare_pareto.py's standalone
     pareto_comparison.csv output was superseded by b5_final_closure.py's
     richer compare_summary.json before either was ever run for real in
     this project -- this check will therefore always report "NO DATA
     YET" and that is expected, not a gap; see check 6 for the artifact
     that actually exists.
  6. B5 final closure (compare_summary.json, if present): recompute every
     comparison_table row's delta_tandem_minus_baseline_pp the same way
     (independent numpy.interp against each model's own plans.csv), and
     independently recompute the DeiT-S pooled-baseline Pareto front
     (same dominance definition as b5_final_closure.py's
     pooled_pareto_front, reimplemented verbatim here) and match the
     stored front exactly.

Usage: python scripts/verify_baselines_claims.py   (run from anywhere --
paths are resolved relative to this file, not the caller's cwd)
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINES_ROOT = REPO_ROOT / "experiments" / "phase2_2_baselines"
P2_RESULTS_ROOT = REPO_ROOT / "experiments" / "phase2_1_fleet" / "results" / "P2"

# --- verbatim copies of the formulas being checked (not imports -- this
# verifier must have zero dependency on the code under test to be a
# genuinely independent check).
GATE_TOLERANCE_PCT = 0.3  # common/repro_gate.py

RESULTS_COLUMNS = [
    "method", "model", "config", "checkpoint_tag", "row_kind", "setting", "top1_50k", "n_images",
    "bops_ours", "repo_commit", "published_top1_pct", "gate_status", "gate_diff_pct_points",
    "calib_images", "host", "gpu_name", "torch_version", "cuda_version", "config_hash",
    "pip_freeze_hash", "notes",
]  # common/results_io.py's RESULTS_COLUMNS -- kept in sync by hand (this verifier must have zero
# import dependency on the code under test); last synced 2026-07 after "setting" (B1 gate
# correction) and "pip_freeze_hash" (P2.2 finalization audit item #4) were added there. A real
# drift here (missing "setting"/"pip_freeze_hash") was caught and fixed during the P2.2 reviewer-
# handoff pass -- see REPRODUCIBILITY.md.


def check_gate(measured_top1_pct, published_top1_pct):
    """Verbatim from common/repro_gate.py's check_gate."""
    if published_top1_pct is None:
        return "no_published_reference", None
    diff = abs(measured_top1_pct - published_top1_pct)
    status = "pass" if diff <= GATE_TOLERANCE_PCT else "fail"
    return status, diff


def load_results_df():
    results_csv = BASELINES_ROOT / "results.csv"
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv, dtype=str)
    return df


def claim_reproduction_gate(df):
    if df is None or len(df) == 0:
        return {
            "name": "1. Reproduction gate (results.csv, common/repro_gate.py's +-0.3pp rule)",
            "recomputed": "N/A", "stored": "N/A", "match": True,
            "detail": {"note": "NO DATA YET -- results.csv has no rows. Not a failure; nothing to check."},
        }
    mismatches = []
    n_checked = 0
    for _, row in df.iterrows():
        published = row.get("published_top1_pct")
        if published in (None, "", "nan") or pd.isna(published):
            continue  # no_published_reference rows: nothing to recompute against
        n_checked += 1
        measured = float(row["top1_50k"])
        recomputed_status, recomputed_diff = check_gate(measured, float(published))
        stored_status = row.get("gate_status")
        stored_diff = row.get("gate_diff_pct_points")
        stored_diff_f = float(stored_diff) if stored_diff not in (None, "", "nan") else None
        ok = (recomputed_status == stored_status
              and stored_diff_f is not None and abs(recomputed_diff - stored_diff_f) < 1e-9)
        if not ok:
            mismatches.append({
                "method": row.get("method"), "model": row.get("model"), "config": row.get("config"),
                "recomputed_status": recomputed_status, "stored_status": stored_status,
                "recomputed_diff": recomputed_diff, "stored_diff": stored_diff_f,
            })
    return {
        "name": "1. Reproduction gate (results.csv, common/repro_gate.py's +-0.3pp rule)",
        "recomputed": f"{n_checked - len(mismatches)}/{n_checked} gated rows match exactly",
        "stored": "gate_status/gate_diff_pct_points columns",
        "match": not mismatches, "detail": {"mismatches": mismatches[:10]},
    }


def claim_schema(df):
    if df is None:
        return {
            "name": "2. Schema completeness (results.csv columns)",
            "recomputed": "N/A", "stored": "N/A", "match": True,
            "detail": {"note": "NO DATA YET -- results.csv does not exist."},
        }
    actual_columns = list(df.columns)
    columns_ok = actual_columns == RESULTS_COLUMNS
    key_fields = ("method", "model", "config", "row_kind", "checkpoint_tag")
    missing_key_field_rows = []
    for i, row in df.iterrows():
        for k in key_fields:
            v = row.get(k)
            if v in (None, "", "nan") or pd.isna(v):
                missing_key_field_rows.append((i, k))
    match = columns_ok and not missing_key_field_rows
    return {
        "name": "2. Schema completeness (results.csv columns)",
        "recomputed": f"columns_match_RESULTS_COLUMNS={columns_ok}, "
                       f"missing_key_fields={len(missing_key_field_rows)} across {len(df)} row(s)",
        "stored": f"{len(RESULTS_COLUMNS)}-column schema, common/results_io.py",
        "match": match, "detail": {"missing_key_field_rows": missing_key_field_rows[:10],
                                    "actual_columns": actual_columns if not columns_ok else None},
    }


def claim_config_hash_consistency(df):
    if df is None or len(df) == 0:
        return {
            "name": "3. config_hash structural consistency (results.csv)",
            "recomputed": "N/A", "stored": "N/A", "match": True,
            "detail": {"note": "NO DATA YET -- results.csv has no rows."},
        }
    key_fields = ("method", "model", "config", "row_kind", "checkpoint_tag")
    seen = {}
    conflicts = []
    for _, row in df.iterrows():
        key = tuple(row.get(k) for k in key_fields)
        h = row.get("config_hash")
        if key in seen and seen[key] != h:
            conflicts.append({"key": dict(zip(key_fields, key)), "hash_a": seen[key], "hash_b": h})
        seen[key] = h
    return {
        "name": "3. config_hash structural consistency (results.csv)",
        "recomputed": f"{len(seen)} distinct (method,model,config,row_kind,checkpoint_tag) keys, "
                       f"{len(conflicts)} hash conflicts",
        "stored": "config_hash column",
        "match": not conflicts, "detail": {"conflicts": conflicts[:10]},
    }


def claim_equivalence_fingerprint():
    fp_path = BASELINES_ROOT / "equivalence_fingerprint.json"
    if not fp_path.exists():
        return {
            "name": "4. Equivalence fingerprint (no halted host mismatches)",
            "recomputed": "N/A", "stored": "N/A", "match": True,
            "detail": {"note": f"NO DATA YET -- {fp_path.name} does not exist (no equivalence gate run yet)."},
        }
    with open(fp_path) as f:
        fp = json.load(f)
    halted = []
    for key, entry in fp.items():
        if isinstance(entry, dict) and entry.get("halted"):
            halted.append({"key": key, "reason": entry.get("halt_reason")})
    return {
        "name": "4. Equivalence fingerprint (no halted host mismatches)",
        "recomputed": f"{len(fp)} entries, {len(halted)} halted",
        "stored": str(fp_path.name),
        "match": not halted, "detail": {"halted": halted[:10]},
    }


def load_tandem_pareto(model_name):
    """Verbatim from compare_pareto.py's load_tandem_pareto."""
    plans_csv = P2_RESULTS_ROOT / model_name / "plans.csv"
    if not plans_csv.exists():
        return None
    df = pd.read_csv(plans_csv)
    df = df.sort_values("C").reset_index(drop=True)
    df["top1_50k_pct"] = df["top1_50k"] * 100
    return df


def interpolate_tandem_top1_pct(pareto_df, cost):
    """Verbatim from compare_pareto.py's interpolate_tandem_top1_pct."""
    costs = pareto_df["C"].to_numpy()
    top1s_pct = pareto_df["top1_50k_pct"].to_numpy()
    extrapolated = bool(cost < costs.min() or cost > costs.max())
    interpolated_pct = float(np.interp(cost, costs, top1s_pct))
    return interpolated_pct, extrapolated


def claim_pareto_comparison():
    out_path = BASELINES_ROOT / "pareto_comparison.csv"
    if not out_path.exists():
        return {
            "name": "5. B5 Pareto comparison (pareto_comparison.csv)",
            "recomputed": "N/A", "stored": "N/A", "match": True,
            "detail": {"note": f"NO DATA YET -- {out_path.name} does not exist (B5 not run yet)."},
        }
    stored_df = pd.read_csv(out_path)
    mismatches = []
    n_checked = 0
    for _, row in stored_df.iterrows():
        pareto_df = load_tandem_pareto(row["model"])
        stored_delta = row.get("matched_cost_delta_pp")
        if pareto_df is None or pd.isna(stored_delta) or stored_delta == "":
            continue  # "no TANDEM P2 result found" rows: nothing to recompute
        n_checked += 1
        recomputed_pct, recomputed_extrap = interpolate_tandem_top1_pct(pareto_df, float(row["bops_ours"]))
        recomputed_delta = recomputed_pct - float(row["baseline_top1_50k_pct"])
        ok = abs(recomputed_delta - float(stored_delta)) < 1e-6
        if not ok:
            mismatches.append({
                "method": row.get("method"), "model": row.get("model"),
                "recomputed_delta": recomputed_delta, "stored_delta": float(stored_delta),
            })
    return {
        "name": "5. B5 Pareto comparison (pareto_comparison.csv)",
        "recomputed": f"{n_checked - len(mismatches)}/{n_checked} matched-cost rows match exactly",
        "stored": str(out_path.name),
        "match": not mismatches, "detail": {"mismatches": mismatches[:10]},
    }


def pooled_pareto_front_configs(points):
    """Verbatim reimplementation of b5_final_closure.py's pooled_pareto_front
    dominance definition (a point is dominated if another point exists with
    <= cost AND >= accuracy, strictly better in at least one), returning
    just the frontier's set of configs for comparison against the stored
    front."""
    pts = sorted(points, key=lambda p: p["bops"])
    frontier = []
    for p in pts:
        is_dominated = any(
            q is not p and q["bops"] <= p["bops"] and q["top1"] >= p["top1"] and
            (q["bops"] < p["bops"] or q["top1"] > p["top1"])
            for q in pts
        )
        if not is_dominated:
            frontier.append(p)
    return frontier


def claim_compare_summary():
    out_path = BASELINES_ROOT / "compare_summary.json"
    if not out_path.exists():
        return {
            "name": "6. B5 final closure (compare_summary.json)",
            "recomputed": "N/A", "stored": "N/A", "match": True,
            "detail": {"note": f"NO DATA YET -- {out_path.name} does not exist (B5 closure not run yet)."},
        }
    with open(out_path) as f:
        summary = json.load(f)

    mismatches = []
    n_checked = 0
    points_by_model = {}
    for row in summary["comparison_table"]:
        pareto_df = load_tandem_pareto(row["model"])
        if pareto_df is None:
            continue
        n_checked += 1
        recomputed_pct, recomputed_extrap = interpolate_tandem_top1_pct(pareto_df, float(row["baseline_bops"]))
        recomputed_delta = recomputed_pct - float(row["baseline_top1_pct"])
        stored_delta = float(row["delta_tandem_minus_baseline_pp"])
        if abs(recomputed_delta - stored_delta) > 1e-4:
            mismatches.append({
                "method": row.get("method"), "model": row.get("model"), "config": row.get("config"),
                "recomputed_delta": recomputed_delta, "stored_delta": stored_delta,
            })
        points_by_model.setdefault(row["model"], []).append(
            {"label": f"{row['method']}/{row['config']}", "bops": float(row["baseline_bops"]),
             "top1": float(row["baseline_top1_pct"])}
        )

    front_mismatches = []
    deit_small_pts = points_by_model.get("deit_small_patch16_224", [])
    if deit_small_pts:
        # r0 (the identity/off-the-shelf anchor) is intentionally excluded from comparison_table
        # (b5_final_closure.py's own main() skips it -- "r0 has no TANDEM-curve counterpart row").
        # The stored pooled front DOES include it (as "r0"), so it's added back here the same way
        # b5_final_closure.py's build_model_points does: from results.csv directly.
        results_df = load_results_df()
        if results_df is not None:
            r0_rows = results_df[(results_df["method"] == "tome") & (results_df["model"] == "deit_small_patch16_224") &
                                  (results_df["config"] == "r0") & (results_df["row_kind"] == "reproduction")]
            if len(r0_rows) == 1:
                c_id = None
                p2_summary_path = P2_RESULTS_ROOT / "deit_small_patch16_224" / "P2_deit_small_patch16_224_summary.json"
                if p2_summary_path.exists():
                    c_id = json.loads(p2_summary_path.read_text())["c_id"]
                deit_small_pts = deit_small_pts + [{
                    "label": "r0", "bops": c_id if c_id is not None else float(r0_rows.iloc[0]["bops_ours"]),
                    "top1": float(r0_rows.iloc[0]["top1_50k"]),
                }]
        recomputed_front = sorted(p["label"] for p in pooled_pareto_front_configs(deit_small_pts))
        stored_front = sorted(summary["pooled_pareto_fronts"]["deit_small_patch16_224"]["frontier"])
        if recomputed_front != stored_front:
            front_mismatches.append({"recomputed": recomputed_front, "stored": stored_front})

    match = not mismatches and not front_mismatches
    return {
        "name": "6. B5 final closure (compare_summary.json)",
        "recomputed": f"{n_checked - len(mismatches)}/{n_checked} comparison-table deltas match; "
                       f"DeiT-S pooled front {'matches' if not front_mismatches else 'MISMATCHES'} stored",
        "stored": str(out_path.name),
        "match": match, "detail": {"mismatches": mismatches[:10], "front_mismatches": front_mismatches},
    }


def main():
    df = load_results_df()
    claims = [
        claim_reproduction_gate(df),
        claim_schema(df),
        claim_config_hash_consistency(df),
        claim_equivalence_fingerprint(),
        claim_pareto_comparison(),
        claim_compare_summary(),
    ]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 100)
    print(f"BASELINES (B1-B5) HEADLINE CLAIMS -- READ-ONLY VERIFICATION -- {now}")
    print("(no GPU, no subprocess, no re-run -- recomputed from experiments/phase2_2_baselines/ artifacts)")
    print("=" * 100)
    # Deliberately NOT inferred from each claim's "recomputed" text (a
    # passing schema check on a header-only, zero-row results.csv would
    # otherwise count as "real data" and misleadingly print ALL CLAIMS
    # VERIFIED) -- explicitly requires actual measurement rows/artifacts,
    # not just a well-formed empty file.
    any_real_data = bool(
        (df is not None and len(df) > 0)
        or (BASELINES_ROOT / "equivalence_fingerprint.json").exists()
        or (BASELINES_ROOT / "pareto_comparison.csv").exists()
    )
    all_passed = True
    for c in claims:
        status = "MATCH" if c["match"] else "FAIL"
        if not c["match"]:
            all_passed = False
        print(f"\n[{status}] {c['name']}")
        print(f"  recomputed: {c['recomputed']}")
        print(f"  stored:     {c['stored']}")
        if c["detail"].get("note"):
            print(f"  note:       {c['detail']['note']}")
        elif not c["match"]:
            print(f"  detail:     {c['detail']}")

    print("\n" + "=" * 100)
    if not any_real_data:
        print("NO REAL DATA YET -- experiments/phase2_2_baselines/ has not been run. This is expected before the first real "
              "B1-B4 run; re-run this script once results.csv/equivalence_fingerprint.json/"
              "pareto_comparison.csv have real content.")
    elif all_passed:
        print("ALL CLAIMS VERIFIED")
    else:
        print("SOME CLAIMS FAILED -- see [FAIL] lines above. Do not present this result until diagnosed.")
    print("=" * 100)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
