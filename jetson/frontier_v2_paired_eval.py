"""jetson/frontier_v2_paired_eval.py -- HOST/Colab, deliberately NOT the Jetson (same reason as
paired_eval.py -- pure PyTorch, no TensorRT). LUT EXTENSION, step 6 -- see
jetson/README.md's "LUT extension" section and jetson/CHARACTERIZATION.md.

Evaluates ARM A vs ARM B, paired, at EVERY budget where frontier_v2.py found BOTH now feasible
against the extended LUT -- 82%, 85%, 89%, 94% (real result: 77%/79% remain infeasible for ARM B
even at the extended range's max r=16; see frontier_manifest_v2.json's arm_b_feasibility).
Descriptive per-budget + pooled mean paired Delta-top1 with a 95% CI, NO gate (none was
registered for this comparison, same as frontier_paired_eval.py v1).

REUSES paired_eval.py's plan-execution engine directly (evaluate_plan_top1,
calibrate_activation_scales, hash_state_dict, load_partitioned_dataset, normalize_batch,
paired_ci) and frontier_v2.py's uniform_r_plan -- jetson-internal imports, same established
exception _common.py's own docstring documents.

CROSS-CHECK, not re-derivation: ARM A's plans at ALL 6 budgets are IDENTICAL to v1 (frontier.py
already confirmed this structurally -- TANDEM's own schedule space never touches n_l below 69
regardless of the LUT's floor), and ARM B's plans at 89%/94% use the SAME r as v1 (r=11, r=8 --
the smallest-feasible-r selection rule doesn't change when a wider range becomes available if
the original smallest-feasible r was already within that range). This script therefore
evaluates all 4 budgets fresh (never trusts "should be the same" without checking), then
explicitly compares the 89%/94% results against v1's already-measured, committed numbers as a
reproducibility check -- same real images, same plans, deterministic evaluation, so these
should match closely; a mismatch would indicate a real bug, not measurement noise.

Genuinely NEW plans being measured for the first time: ARM B at 82% (r=16) and 85% (r=14) --
these never existed in v1 (both budgets were ARM-B-infeasible there).

Usage: python frontier_v2_paired_eval.py --imagenet-dir <dir>
(same materialized 20,512-image directory paired_eval.py/frontier_paired_eval.py already use)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from frontier_v2 import uniform_r_plan  # noqa: E402
from paired_eval import (  # noqa: E402
    BATCH_SIZE, calibrate_activation_scales, CALIB_N, evaluate_plan_top1, hash_state_dict,
    load_partitioned_dataset, MODEL_NAME, paired_ci, PAIRED_EVAL_N, RESOLVED_TAG,
)

VERBATIM_NOTE = (
    "The original ARM B feasibility bound at r<=11 was set by the measured LUT's floor (N=69), "
    "not by any property of uniform merging. The LUT was extended to N=21 to remove that "
    "confound; the feasibility results below supersede the earlier ones."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"=== frontier_v2_paired_eval.py: device={device} (TF32 explicitly disabled) ===")
    print(VERBATIM_NOTE)

    manifest_path = JETSON_ROOT / "out" / "frontier_v2" / "frontier_manifest_v2.json"
    assert manifest_path.exists(), f"{manifest_path} not found -- run frontier_v2.py first (no device needed)."
    manifest = read_json(manifest_path)
    jointly_feasible = [r for r in manifest["arm_b_feasibility"] if r["feasible"]]
    jointly_feasible_pct = [f"{r['budget_fraction']:.0%}" for r in jointly_feasible]
    print(f"\nJointly feasible budgets (per frontier_v2.py's real result): {jointly_feasible_pct}")

    print(f"\nLoading + partitioning {args.imagenet_dir} ({CALIB_N} calib + {PAIRED_EVAL_N} "
          "paired-eval, disjoint) -- same manifest/split as paired_eval.py...")
    calib_images, eval_images, eval_labels, manifest_hash = load_partitioned_dataset(args.imagenet_dir)
    print(f"Paired-eval manifest hash: {manifest_hash}")

    import timm
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True)
    except Exception:
        model = timm.create_model(RESOLVED_TAG, pretrained=True)
    model = model.eval().to(device).requires_grad_(False)
    n_blocks = len(model.blocks)
    checkpoint_hash = hash_state_dict(model)
    print(f"Loaded {RESOLVED_TAG}, checkpoint_hash={checkpoint_hash}")

    print(f"Calibrating activation scales from {CALIB_N} local images...")
    amax_lookup = calibrate_activation_scales(model, calib_images, BATCH_SIZE, n_blocks, device)

    v2_path = JETSON_ROOT / "out" / "frontier_v2" / "frontier_plans_v2.csv"
    assert v2_path.exists(), f"{v2_path} not found -- run frontier_v2.py first."
    v2_df = pd.read_csv(v2_path)

    per_budget_results = []
    all_diffs = []
    for r in jointly_feasible:
        budget_frac = r["budget_fraction"]
        row = v2_df[np.isclose(v2_df["budget_fraction"], budget_frac)].iloc[0]
        arm_a_plan = json.loads(row["arm_a_plan_json_v2"])
        arm_b_plan = uniform_r_plan(r["best_r"])
        print(f"\n=== budget_fraction={budget_frac:.0%}, ARM B r={r['best_r']} ===")
        print(f"  ARM A plan: bits={arm_a_plan['bits'] if any(b != 16 for b in arm_a_plan['bits'].values()) else 'all-16'} "
              f"merges={arm_a_plan['merges']}")
        print(f"  ARM B plan: merges={arm_b_plan['merges']}")

        arm_a_correct = evaluate_plan_top1(model, eval_images, eval_labels, arm_a_plan, amax_lookup, BATCH_SIZE, device)
        arm_b_correct = evaluate_plan_top1(model, eval_images, eval_labels, arm_b_plan, amax_lookup, BATCH_SIZE, device)
        arm_a_top1 = float(arm_a_correct.mean() * 100)
        arm_b_top1 = float(arm_b_correct.mean() * 100)
        diff_pp = (arm_a_correct.astype(np.float64) - arm_b_correct.astype(np.float64)) * 100.0
        all_diffs.append(diff_pp)

        per_budget_results.append({
            "budget_fraction": budget_frac, "arm_b_r": r["best_r"],
            "arm_a_top1_pct": arm_a_top1, "arm_b_top1_pct": arm_b_top1,
            "mean_diff_ab_pp": float(diff_pp.mean()),
            "newly_feasible_in_v2": budget_frac < 0.89,  # 82%/85% were infeasible in v1; 89%/94% weren't
        })
        print(f"  ARM A top1={arm_a_top1:.4f}%  ARM B top1={arm_b_top1:.4f}%  diff(A-B)={diff_pp.mean():+.4f}pp")

    print("\n=== Pooled (all jointly-feasible budgets) ===")
    if all_diffs:
        pooled_diffs = np.concatenate(all_diffs)
        mean_pp, ci_low, ci_high = paired_ci(pooled_diffs)
        print(f"Pooled mean paired delta-top1 (ARM A - ARM B): {mean_pp:+.4f}pp  "
              f"95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]pp  (n={len(pooled_diffs)})")
    else:
        mean_pp = ci_low = ci_high = None
        print("No jointly-feasible budgets -- nothing to pool.")

    # --- Reproducibility cross-check against v1's already-committed numbers at 89%/94% (same
    # plans, proven identical by frontier_v2.py -- see module docstring). Not required for the
    # result to be valid, but a real, useful consistency check on real hardware/data.
    v1_result_path = JETSON_ROOT / "out" / "frontier" / "frontier_result.json"
    cross_check = []
    if v1_result_path.exists():
        v1_result = read_json(v1_result_path)
        v1_paired_by_budget = {p["budget_fraction"]: p for p in v1_result["paired_comparison"]["per_budget"]}
        for r in per_budget_results:
            if r["newly_feasible_in_v2"]:
                continue
            v1_row = v1_paired_by_budget.get(r["budget_fraction"])
            if v1_row is None:
                continue
            a_diff = abs(r["arm_a_top1_pct"] - v1_row["arm_a_top1_pct"])
            b_diff = abs(r["arm_b_top1_pct"] - v1_row["arm_b_top1_pct"])
            match = a_diff < 1e-6 and b_diff < 1e-6
            cross_check.append({
                "budget_fraction": r["budget_fraction"], "match": match,
                "v1_arm_a_top1_pct": v1_row["arm_a_top1_pct"], "v2_arm_a_top1_pct": r["arm_a_top1_pct"],
                "v1_arm_b_top1_pct": v1_row["arm_b_top1_pct"], "v2_arm_b_top1_pct": r["arm_b_top1_pct"],
            })
            print(f"\nCross-check budget={r['budget_fraction']:.0%} vs v1 (same plans, same images): "
                  f"{'MATCH' if match else 'MISMATCH -- real bug, investigate'} "
                  f"(v1 ARM A={v1_row['arm_a_top1_pct']:.4f}% v2 ARM A={r['arm_a_top1_pct']:.4f}%, "
                  f"v1 ARM B={v1_row['arm_b_top1_pct']:.4f}% v2 ARM B={r['arm_b_top1_pct']:.4f}%)")

    result = {
        "model_name": MODEL_NAME, "checkpoint_hash": checkpoint_hash, "manifest_hash": manifest_hash,
        "calib_n": CALIB_N, "paired_eval_n": PAIRED_EVAL_N,
        "verbatim_note": VERBATIM_NOTE,
        "jointly_feasible_budgets": [r["budget_fraction"] for r in jointly_feasible],
        "per_budget": per_budget_results,
        "pooled_mean_diff_ab_pp": mean_pp, "pooled_ci_low_pp": ci_low, "pooled_ci_high_pp": ci_high,
        "n_pooled_pairs": int(sum(len(d) for d in all_diffs)) if all_diffs else 0,
        "no_gate_registered_note": (
            "Descriptive comparison only -- no pass/fail gate was registered for ARM A vs ARM B."
        ),
        "v1_reproducibility_cross_check": cross_check,
        "arm_a_frontier_unchanged_note": (
            "ARM A's plans at all 6 budgets are unchanged from v1 (frontier.py confirmed this "
            "structurally) -- see out/frontier/frontier_result.json's arm_a_frontier for the "
            "full 6-budget accuracy curve, not re-measured here."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = ensure_out_dir("frontier_v2") / "frontier_result_v2.json"
    write_json(out_path, result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
