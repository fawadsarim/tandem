"""B5: reads experiments/phase2_2_baselines/results.csv (B1-B4's measurements) and TANDEM's own
P2 Pareto curves (results/P2/<model>/plans.csv, from
experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb), linearly interpolates TANDEM's own
Pareto curve AT each baseline's `bops_ours` cost, and reports the
matched-cost delta (TANDEM interpolated top-1 minus the baseline's
measured top-1). No conclusions drawn here -- numbers only, per spec;
`pareto_comparison.csv` is the entire output.

Unit note: P2's plans.csv stores `top1_50k` as a FRACTION (0-1) -- see
P2_tandem_pipeline.py's `run_plan` (`"top1_pert": n_correct_pert / n`).
This project's own results.csv stores `top1_50k` as a PERCENTAGE (0-100)
-- see e.g. b1_tome/run.py (`top1_pct = top1 * 100`) and B2/B3's parsed
`Prec@1` (already a percentage per RepQ-ViT's/FQ-ViT's own `accuracy()`
helpers). Converted explicitly below, not left as an implicit assumption.

Row filter (P2.2 audit item #8, "training-free parity rule enforced in
code" -- FIXED 2026-07, found during the audit, was previously UNFILTERED):
only `row_kind="comparison"` rows (Step B, TANDEM's own PINNED checkpoint
-- the ONLY rows measured on the same checkpoint identity as TANDEM's own
P2 Pareto curve) with `setting` in {"", "off_the_shelf"} are eligible.
Two independent reasons a naive "every row" join would be wrong, not just
imprecise:
  (a) `row_kind="reproduction"` rows (Step A, the PAPER's own checkpoint)
      exist to gate against a published number, not to plot against
      TANDEM's Pareto curve -- for ViT-B specifically, Step A's checkpoint
      (augreg_in21k_ft_in1k, FP 84.534) is a DIFFERENT identity than
      TANDEM's pinned Step B checkpoint (augreg2_in21k_ft_in1k, FP
      85.108) that P2's own plans.csv was measured on (see
      common/checkpoint_pin.py) -- including both row_kinds would not
      only double-count every method+model but compare Step A's number
      against a Pareto curve measured on different weights entirely.
  (b) `setting="trained_with_tome"` rows (B1's r=13 gate against ToMe's
      OWN shipped checkpoint, fine-tuned WITH merging during training --
      see b1_tome/run.py's module docstring and ../deviations.md's
      setting-mismatch correction entry) are not a training-free/
      off-the-shelf measurement at all -- joining one into this table
      would compare TANDEM's training-free Pareto curve against a
      DIFFERENT method's trained checkpoint, the exact class of
      apples-to-oranges error this project's whole B1 correction was
      about, just recurring here in the comparison step instead of the
      gate step.
`row_kind="reproduction"` rows remain fully visible in results.csv itself
(the gate citation and setting-mismatch audit trail need them) -- they are
excluded from THIS join only.

Orchestration environment only (pandas/numpy, no torch/timm) -- see
requirements.txt and deviations.md #1.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASELINES_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_ROOT.parent.parent  # experiments/phase2_2_baselines/ now nests one level deeper, under experiments/
# P2.2 audit item #17 (FIXED 2026-07): was `REPO_ROOT / "results" / "P2"`,
# a path that has never existed -- the real canonical P2 artifacts live
# under experiments/phase2_1_fleet/results/P2/ (verified directly: `ls
# experiments/phase2_1_fleet/results/P2` exists, `ls results/P2` does not).
# load_tandem_pareto()'s `.exists()` check silently swallowed this the
# entire time -- every row fell into the "no TANDEM P2 result found on
# disk" branch, meaning this join has never actually performed a real
# interpolation since this file was written.
P2_RESULTS_ROOT = REPO_ROOT / "experiments" / "phase2_1_fleet" / "results" / "P2"


def load_tandem_pareto(model_name):
    """Returns a sorted-by-cost DataFrame of TANDEM's own (C, top1_50k_pct)
    points for `model_name`, or None if no P2 result exists yet for it."""
    plans_csv = P2_RESULTS_ROOT / model_name / "plans.csv"
    if not plans_csv.exists():
        return None
    df = pd.read_csv(plans_csv)
    assert "C" in df.columns and "top1_50k" in df.columns, (
        f"{plans_csv} is missing expected columns C/top1_50k -- check it was produced by "
        "P2_tandem_pipeline.ipynb's S9 artifact stage (Section 22)."
    )
    df = df.sort_values("C").reset_index(drop=True)
    df["top1_50k_pct"] = df["top1_50k"] * 100  # fraction -> percentage, see module docstring
    return df


def interpolate_tandem_top1_pct(pareto_df, cost):
    """Linear interpolation of TANDEM's own Pareto curve at `cost`.
    Returns (interpolated_top1_pct, extrapolated: bool) --
    extrapolated=True means `cost` fell outside TANDEM's own measured cost
    range for this model, so the value is a flat-extrapolation beyond the
    nearest edge point (numpy.interp's default behavior), not a genuine
    interpolation between two measured plans -- flagged, not hidden."""
    costs = pareto_df["C"].to_numpy()
    top1s_pct = pareto_df["top1_50k_pct"].to_numpy()
    extrapolated = bool(cost < costs.min() or cost > costs.max())
    interpolated_pct = float(np.interp(cost, costs, top1s_pct))
    return interpolated_pct, extrapolated


def comparison_eligible_rows(results_df):
    """Filters to the rows this join is actually allowed to use -- see
    module docstring's "Row filter" section. Prints what got excluded and
    why, so a caller can see the filter did something rather than
    silently trusting an empty diff."""
    row_kind_ok = results_df["row_kind"] == "comparison"
    # "setting" may be "" (every non-B1 row, or B1's own off_the_shelf
    # rows written before this column existed) or "off_the_shelf" -- both
    # eligible; only "trained_with_tome" is excluded.
    setting_ok = results_df["setting"].fillna("").isin(["", "off_the_shelf"])
    eligible = results_df[row_kind_ok & setting_ok]
    excluded = results_df[~(row_kind_ok & setting_ok)]
    if len(excluded):
        print(f"Excluded {len(excluded)}/{len(results_df)} row(s) from the comparison join "
              "(row_kind != 'comparison' and/or setting == 'trained_with_tome'):")
        print(excluded[["method", "model", "config", "row_kind", "setting"]].to_string(index=False))
    return eligible


def main():
    results_csv = BASELINES_ROOT / "results.csv"
    assert results_csv.exists(), f"{results_csv} not found -- run at least one baseline (B1-B4) first."
    results_df = pd.read_csv(results_csv)
    assert len(results_df), f"{results_csv} has a header but no rows yet -- nothing to compare."
    results_df = comparison_eligible_rows(results_df)
    assert len(results_df), (
        f"{results_csv} has {sum(1 for _ in open(results_csv)) - 1} row(s) but NONE are "
        "row_kind=comparison with an off-the-shelf/training-free setting -- nothing eligible to compare."
    )

    rows = []
    for _, r in results_df.iterrows():
        model_name = r["model"]
        pareto_df = load_tandem_pareto(model_name)
        base_row = {
            "method": r["method"], "model": model_name, "config": r["config"],
            "baseline_top1_50k_pct": r["top1_50k"], "bops_ours": r["bops_ours"],
        }
        if pareto_df is None:
            rows.append({
                **base_row, "tandem_top1_at_matched_cost_pct": "", "matched_cost_delta_pp": "",
                "extrapolated": "", "note": "no TANDEM P2 result found on disk for this model yet",
            })
            continue

        cost = float(r["bops_ours"])
        tandem_top1_pct, extrapolated = interpolate_tandem_top1_pct(pareto_df, cost)
        baseline_top1_pct = float(r["top1_50k"])
        delta_pp = tandem_top1_pct - baseline_top1_pct
        rows.append({
            **base_row,
            "tandem_top1_at_matched_cost_pct": tandem_top1_pct,
            "matched_cost_delta_pp": delta_pp,
            "extrapolated": extrapolated,
            "note": "extrapolated beyond TANDEM's own measured cost range for this model" if extrapolated else "",
        })

    out_df = pd.DataFrame(rows)
    out_path = BASELINES_ROOT / "pareto_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
