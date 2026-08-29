"""jetson/frontier_paired_eval.py -- HOST/Colab, deliberately NOT the Jetson (same reason as
paired_eval.py -- pure PyTorch, no TensorRT). G1-HW final phase, Task 3's accuracy comparison
-- see jetson/README.md's "Re-scoped sweep" section and jetson/CHARACTERIZATION.md.

REDUCED SCOPE (directed, 2026-07, superseding this file's original all-6-budgets-if-
feasible design): frontier.py found ARM B (uniform-r ToMe) LATENCY-INFEASIBLE at 4 of the 6
frontier budgets (77/79/82/85% of identity latency) -- a real, measured hardware finding (see
LATENCY_INFEASIBLE_NOTE below), not a bug. Since ARM B does not exist at those 4 budgets, a
paired accuracy comparison there would compare a plan against nothing -- the DP's own
infeasibility result already settles them. This script now:
  1. Evaluates ARM A (TANDEM's own joint DP plan) at ALL SIX budgets, against the FP-identity
     plan for context -- this is THE frontier curve, reported regardless of ARM B's feasibility.
  2. Evaluates ARM B, and the ARM-A-vs-ARM-B PAIRED comparison, ONLY at the 89% and 94% budgets
     -- the two where ARM B is feasible. Pooled mean paired Delta-top1 with a 95% CI, per-budget
     and pooled, NO pass/fail gate (none was registered for this comparison).
  3. Writes the LATENCY_INFEASIBLE_NOTE and the D_PRED coverage-limitation note into
     frontier_result.json verbatim, so both are visible in the result artifact itself, not just
     this file's docstring.

REUSES paired_eval.py's plan-execution engine directly (evaluate_plan_top1,
calibrate_activation_scales, hash_state_dict, load_partitioned_dataset, normalize_batch,
paired_ci) via import, same jetson-internal-only exception _common.py's own docstring
documents -- not reimplemented a third time.

NOT A JOINT-VS-STAGED TEST: that comparison is VOID on this device/LUT (see
CHARACTERIZATION.md). This compares merge-PLACEMENT granularity (ARM A) against uniform-r
merging (ARM B) at matched MEASURED latency -- the hardware analogue of the P2.2 saturation
finding.

Usage: python frontier_paired_eval.py --imagenet-dir <dir>
(same materialized 20,512-image directory paired_eval.py already uses -- no new materialization)
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
from _common import ensure_out_dir, JETSON_ROOT, write_json  # noqa: E402
from paired_eval import (  # noqa: E402
    BATCH_SIZE, calibrate_activation_scales, CALIB_N, evaluate_plan_top1, hash_state_dict,
    load_partitioned_dataset, MODEL_NAME, paired_ci, PAIRED_EVAL_N, RESOLVED_TAG,
)

PRE_REGISTERED_INTERPRETATION = (
    "This compares merge-PLACEMENT granularity (TANDEM's <=2-point schedules) against "
    "uniform-r merging at matched MEASURED latency -- the hardware analogue of the P2.2 "
    "saturation finding. It is NOT a joint-vs-staged test (void per CHARACTERIZATION.md)."
)

PAIRED_COMPARISON_BUDGETS = (0.89, 0.94)  # the only budgets where ARM B is latency-feasible

LATENCY_INFEASIBLE_NOTE = (
    "ARM B (uniform-r ToMe, r in 1..11) is LATENCY-INFEASIBLE at 77/79/82/85% of identity "
    "latency under the measured LUT: uniform merging pays this device's fixed per-merge "
    "overhead at 11 sites, whereas TANDEM's concentrated <=2-point schedules reach 76.5% of "
    "identity. This is a measured hardware finding about merge PLACEMENT granularity, "
    "independent of accuracy."
)

D_PRED_LIMITATION_NOTE = (
    "ARM B's D_pred is not reported (null in frontier_plans.csv) -- P2.1's own Dm_table.csv "
    "only covers TANDEM's registered plan-space grid (MERGE_BLOCKS=(1,3,5,7,9,11) x "
    "MERGE_RS=(32,64,98,128), 24 rows total, confirmed by direct read), not ARM B's "
    "uniform-every-block r in {1..11}. Extrapolating a new accuracy-damage model here would be "
    "fabricating data this project has not measured; the real comparison is the MEASURED top-1 "
    "accuracy reported below, not a predicted proxy."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"=== frontier_paired_eval.py: device={device} (TF32 explicitly disabled) ===")
    print("REDUCED SCOPE (directed, 2026-07): ARM A evaluated at ALL 6 budgets (the "
          "frontier curve). ARM A vs ARM B paired comparison restricted to the 89% and 94% "
          "budgets ONLY -- the two where ARM B is latency-feasible. See LATENCY_INFEASIBLE_NOTE.")
    print(f"NOT a joint-vs-staged test -- {PRE_REGISTERED_INTERPRETATION}")

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

    frontier_path = JETSON_ROOT / "out" / "frontier" / "frontier_plans.csv"
    assert frontier_path.exists(), f"{frontier_path} not found -- run frontier.py first (no device needed)."
    df = pd.read_csv(frontier_path)
    assert len(df) == 6, f"{frontier_path} has {len(df)} rows, expected 6 -- re-run frontier.py."

    identity_plan = {"bits": {}, "merges": []}
    print("\nEvaluating FP identity plan (context reference for the ARM A frontier curve)...")
    identity_correct = evaluate_plan_top1(model, eval_images, eval_labels, identity_plan, amax_lookup, BATCH_SIZE, device)
    identity_top1 = float(identity_correct.mean() * 100)
    print(f"  identity top1={identity_top1:.4f}%")

    arm_a_frontier = []
    paired_per_budget = []
    all_diffs = []

    for _, row in df.iterrows():
        budget_frac = float(row["budget_fraction"])
        tau_ms = float(row["tau_ms"])
        arm_a_plan = json.loads(row["arm_a_plan_json"])
        print(f"\n=== budget_fraction={budget_frac:.0%}, tau_b={tau_ms:.4f}ms ===")
        print(f"  ARM A plan: bits={arm_a_plan['bits']} merges={arm_a_plan['merges']}")

        arm_a_correct = evaluate_plan_top1(model, eval_images, eval_labels, arm_a_plan, amax_lookup, BATCH_SIZE, device)
        arm_a_top1 = float(arm_a_correct.mean() * 100)
        arm_a_frontier.append({
            "budget_fraction": budget_frac, "tau_ms": tau_ms,
            "arm_a_cost_ms": float(row["arm_a_cost_ms"]), "arm_a_d_pred": float(row["arm_a_d_pred"]),
            "arm_a_top1_pct": arm_a_top1, "arm_a_vs_identity_pp": arm_a_top1 - identity_top1,
        })
        print(f"  ARM A top1={arm_a_top1:.4f}%  (identity={identity_top1:.4f}%, "
              f"delta={arm_a_top1 - identity_top1:+.4f}pp)")

        is_paired_budget = any(abs(budget_frac - b) < 1e-6 for b in PAIRED_COMPARISON_BUDGETS)
        arm_b_infeasible = bool(row["arm_b_infeasible"]) or pd.isna(row["arm_b_plan_json"])
        if not is_paired_budget:
            print("  ARM B: not evaluated at this budget (outside the reduced-scope paired set "
                  "{89%, 94%} -- see LATENCY_INFEASIBLE_NOTE if also infeasible here).")
            continue
        assert not arm_b_infeasible, (
            f"budget_fraction={budget_frac:.0%} is in PAIRED_COMPARISON_BUDGETS but "
            "frontier.py marked ARM B infeasible there -- frontier_plans.csv disagrees with "
            "this script's own hardcoded scope, halting rather than silently skipping."
        )

        arm_b_plan = json.loads(row["arm_b_plan_json"])
        print(f"  ARM B plan: r={int(row['arm_b_r'])} merges={arm_b_plan['merges']}")
        arm_b_correct = evaluate_plan_top1(model, eval_images, eval_labels, arm_b_plan, amax_lookup, BATCH_SIZE, device)
        arm_b_top1 = float(arm_b_correct.mean() * 100)
        diff_pp = (arm_a_correct.astype(np.float64) - arm_b_correct.astype(np.float64)) * 100.0
        all_diffs.append(diff_pp)
        paired_per_budget.append({
            "budget_fraction": budget_frac, "tau_ms": tau_ms,
            "arm_a_top1_pct": arm_a_top1, "arm_b_top1_pct": arm_b_top1,
            "arm_b_vs_identity_pp": arm_b_top1 - identity_top1,
            "mean_diff_ab_pp": float(diff_pp.mean()),
        })
        print(f"  ARM B top1={arm_b_top1:.4f}%  diff(A-B)={diff_pp.mean():+.4f}pp")

    print("\n=== ARM A frontier curve (all 6 budgets, vs FP identity) ===")
    for r in arm_a_frontier:
        print(f"  budget={r['budget_fraction']:.0%}: ARM A top1={r['arm_a_top1_pct']:.4f}%  "
              f"delta_vs_identity={r['arm_a_vs_identity_pp']:+.4f}pp")

    print("\n=== ARM A vs ARM B paired comparison (89%/94% ONLY -- descriptive, no gate) ===")
    if all_diffs:
        pooled_diffs = np.concatenate(all_diffs)
        mean_pp, ci_low, ci_high = paired_ci(pooled_diffs)
        print(f"Pooled mean paired delta-top1 (ARM A - ARM B): {mean_pp:+.4f}pp  "
              f"95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]pp  (n={len(pooled_diffs)})")
    else:
        mean_pp = ci_low = ci_high = None
        print("No paired budgets were evaluated -- check PAIRED_COMPARISON_BUDGETS vs frontier_plans.csv.")

    print(f"\n{LATENCY_INFEASIBLE_NOTE}")
    print(f"\n{D_PRED_LIMITATION_NOTE}")
    print(f"\n{PRE_REGISTERED_INTERPRETATION}")

    result = {
        "model_name": MODEL_NAME, "checkpoint_hash": checkpoint_hash, "manifest_hash": manifest_hash,
        "calib_n": CALIB_N, "paired_eval_n": PAIRED_EVAL_N,
        "scope_note": (
            "REDUCED SCOPE, directed 2026-07: ARM A evaluated at all 6 budgets (the "
            "frontier curve, see arm_a_frontier below); ARM A vs ARM B paired comparison "
            "restricted to budget_fraction in {0.89, 0.94}, the only budgets where ARM B is "
            "latency-feasible -- see arm_b_latency_infeasible_note."
        ),
        "identity_top1_pct": identity_top1,
        "arm_a_frontier": arm_a_frontier,
        "paired_comparison": {
            "budgets": list(PAIRED_COMPARISON_BUDGETS),
            "per_budget": paired_per_budget,
            "pooled_mean_diff_ab_pp": mean_pp, "pooled_ci_low_pp": ci_low, "pooled_ci_high_pp": ci_high,
            "n_pooled_pairs": int(sum(len(d) for d in all_diffs)) if all_diffs else 0,
            "no_gate_registered_note": (
                "Descriptive comparison only -- no pass/fail gate was registered for ARM A vs "
                "ARM B, unlike paired_eval.py's BOPs-vs-JOINT/JOINT-vs-STAGED comparisons."
            ),
        },
        "arm_b_latency_infeasible_note": LATENCY_INFEASIBLE_NOTE,
        "arm_b_d_pred_limitation_note": D_PRED_LIMITATION_NOTE,
        "not_a_joint_vs_staged_test": True,
        "pre_registered_interpretation": PRE_REGISTERED_INTERPRETATION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = ensure_out_dir("frontier") / "frontier_result.json"
    write_json(out_path, result)  # written unconditionally, same discipline as paired_eval.py
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
