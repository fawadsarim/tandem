"""B4 DeiT-S post-hoc analysis (2026-07 follow-up, items #2-5):
structural cost-split finding, attention-skip-bias quantification, and
staged Pareto-frontier dominance analysis against TANDEM's own joint
curve. Torch-free (pandas/numpy only, same orchestration-environment
constraint as compare_pareto.py -- see its own module docstring and
../deviations.md #1) -- run via:

    python experiments/phase2_2_baselines/b4_sequential_composites/analyze_deit_small.py

from experiments/phase2_2_baselines/, no GPU needed. Operates entirely on already-recorded
`results.csv` rows plus P2.1's real `plans.csv`, read programmatically
(spec item #5 -- never typed/hardcoded values). Writes
`B4_deit_small_summary.json` (the "summary field" the original spec
asks for, per-item) alongside this file, and prints everything to stdout
for direct paste-back.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bops import cost_from_trajectory, tome_schedule_n_l  # noqa: E402
from compare_pareto import load_tandem_pareto, interpolate_tandem_top1_pct  # noqa: E402

BASELINES_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "deit_small_patch16_224"
D_MODEL = 384
N_BLOCKS = 12
N_TOKENS_TOTAL = 197


def structural_cost_split(canonical_c_id):
    """Item #2: derive DeiT-S's linear/attention BOPs split from the two
    ALREADY-ESTABLISHED r=0 bit-width fractions (W8A8@r0, W6A6@r0),
    algebraically -- `common/bops.py`'s own cost model prices attention
    matmuls at a FIXED 16x16 regardless of (w_bits, a_bits), so the
    attention term is a CONSTANT fraction of C_id and the linear term
    scales exactly with bw*ba: L(bw,ba) = L(16,16) * (bw*ba)/256. Solving
    L8 + A = frac(8,8) and 4*L8 + A = 1.0 (since L(16,16) = 4*L(8,8),
    as 256/64=4) gives L8 and A directly -- verified against a DIRECT,
    independent computation of every fraction below (not just the two
    used to derive the split), so this is proof, not just algebra."""
    n_l0 = tome_schedule_n_l(0, N_BLOCKS, N_TOKENS_TOTAL)

    def frac(w, a):
        return cost_from_trajectory(n_l0, [(w, a)] * N_BLOCKS, D_MODEL) / canonical_c_id

    w8a8_frac, w6a6_frac, w8a16_frac = frac(8, 8), frac(6, 6), frac(8, 16)
    linear8_frac = (1.0 - w8a8_frac) / 3
    attention_frac = w8a8_frac - linear8_frac
    linear16_frac = 4 * linear8_frac

    # Independent cross-checks: the derived (linear16_frac, attention_frac)
    # split must reproduce w6a6_frac and w8a16_frac exactly when combined
    # with the OTHER bit-width ratios -- these were NOT used to derive the
    # split, so agreement here is real proof, not circular.
    predicted_w6a6 = (36 / 256) * linear16_frac + attention_frac
    predicted_w8a16 = (128 / 256) * linear16_frac + attention_frac
    assert abs(predicted_w6a6 - w6a6_frac) < 1e-9, (
        f"cross-check failed: predicted W6A6@r0={predicted_w6a6}, direct calc={w6a6_frac}"
    )
    assert abs(predicted_w8a16 - w8a16_frac) < 1e-9, (
        f"cross-check failed: predicted W8A16@r0={predicted_w8a16}, direct calc={w8a16_frac}"
    )

    result = {
        "linear_frac_pct": round(linear16_frac * 100, 4),
        "attention_frac_pct": round(attention_frac * 100, 4),
        "w8a8_at_r0_pct": round(w8a8_frac * 100, 4),
        "w6a6_at_r0_pct": round(w6a6_frac * 100, 4),
        "w8a16_at_r0_pct": round(w8a16_frac * 100, 4),
    }
    print("=== Item #2: DeiT-S structural cost split (derived, cross-checked) ===")
    print(f"  linear fraction (W16A16@r0)   = {result['linear_frac_pct']:.2f}%")
    print(f"  attention fraction (constant) = {result['attention_frac_pct']:.2f}%")
    print(f"  W8A8@r0  = {result['w8a8_at_r0_pct']:.2f}%")
    print(f"  W6A6@r0  = {result['w6a6_at_r0_pct']:.2f}%")
    print(f"  W8A16@r0 = {result['w8a16_at_r0_pct']:.2f}%  (asymmetric, illustrative only -- excluded, see below)")
    print("  Cross-checks (W6A6, W8A16 predicted from the derived split vs. computed directly): PASS")
    return result


def attention_skip_bias(results_df):
    """Item #3: the observed floor of B4's attention-quantization-skip
    bias (item #38 in deviations.md) -- B4's r2/w8a8 comparison row vs
    B2's plain w8a8 comparison row, SAME model, SAME nominal bit-width,
    the only difference being B4 leaves attention unquantized. The
    estimated total range (0.08-0.15pp) is a documented estimate,
    not independently re-derived here -- recorded as given, not asserted
    as proven, since it accounts for foregone r=2 merge damage that this
    script cannot isolate from results.csv alone."""
    b4_row = results_df[(results_df["method"] == "sequential_composite") &
                         (results_df["config"] == "r2_w8a8_best") &
                         (results_df["row_kind"] == "comparison")]
    b2_row = results_df[(results_df["method"] == "repq_vit") &
                         (results_df["model"] == MODEL_NAME) &
                         (results_df["config"] == "w8a8") &
                         (results_df["row_kind"] == "comparison")]
    assert len(b4_row) == 1, f"expected exactly 1 B4 r2_w8a8_best comparison row, found {len(b4_row)}"
    assert len(b2_row) == 1, f"expected exactly 1 B2 w8a8 comparison row, found {len(b2_row)}"
    b4_top1 = float(b4_row.iloc[0]["top1_50k"])
    b2_top1 = float(b2_row.iloc[0]["top1_50k"])
    floor_pp = round(b4_top1 - b2_top1, 4)

    result = {"floor_pp": floor_pp, "est_range_pp": [0.08, 0.15],
              "b4_r2_w8a8_top1": b4_top1, "b2_w8a8_top1": b2_top1}
    print("\n=== Item #3: attention-skip bias ===")
    print(f"  B4 r2/w8a8 (comparison) = {b4_top1}%")
    print(f"  B2 plain w8a8 (comparison) = {b2_top1}%")
    print(f"  Observed floor = {floor_pp:+.4f}pp")
    print(f"  Estimated total range (documented estimate, includes foregone r=2 merge damage): "
          f"{result['est_range_pp'][0]}-{result['est_range_pp'][1]}pp")
    print("  composite_attention_skip_bias_pp = "
          f"{{floor: {floor_pp}, est_range: {result['est_range_pp']}}}")
    return result


def pareto_dominance_analysis(results_df, pareto_df, canonical_c_id):
    """Item #4: build the staged Pareto set from B1's r=0 anchor (no
    merge, no quant -- the trivial upper-left anchor of any Pareto
    curve) plus the 6 real B4 comparison rows, then assert which points
    are DOMINATED (a cheaper point exists with >= accuracy)."""
    r0_row = results_df[(results_df["method"] == "tome") & (results_df["model"] == MODEL_NAME) &
                         (results_df["config"] == "r0") & (results_df["row_kind"] == "reproduction")]
    assert len(r0_row) == 1, f"expected exactly 1 B1 r0 row, found {len(r0_row)}"
    r0_top1 = float(r0_row.iloc[0]["top1_50k"])
    # Use canonical_c_id, not the row's own bops_ours (stored at reduced
    # precision, "1.16E+12", in results.csv for this specific row) --
    # r=0/no-merge/no-quant IS the identity plan by construction, so its
    # true BOPs is exactly C_id, not an approximation of it.
    r0_bops = canonical_c_id

    b4_rows = results_df[(results_df["method"] == "sequential_composite") &
                          (results_df["row_kind"] == "comparison")].copy()
    assert len(b4_rows) == 6, f"expected exactly 6 B4 comparison rows, found {len(b4_rows)}"

    points = [{"label": "r0 (FP anchor)", "bops": r0_bops, "top1": r0_top1, "config": "r0"}]
    for _, r in b4_rows.iterrows():
        points.append({"label": r["config"], "bops": float(r["bops_ours"]), "top1": float(r["top1_50k"]),
                        "config": r["config"]})
    points.sort(key=lambda p: p["bops"])

    dominated = []
    frontier = []
    for i, p in enumerate(points):
        is_dominated = False
        dominated_by = None
        for q in points:
            if q is p:
                continue
            if q["bops"] <= p["bops"] and q["top1"] >= p["top1"] and (q["bops"] < p["bops"] or q["top1"] > p["top1"]):
                is_dominated = True
                dominated_by = q["config"]
                break
        if is_dominated:
            dominated.append({"config": p["config"], "bops": p["bops"], "top1": p["top1"], "dominated_by": dominated_by})
        else:
            frontier.append(p["config"])

    print("\n=== Item #4: staged Pareto-frontier dominance analysis ===")
    print(f"  Points (sorted by BOPs): {[(p['config'], round(p['bops'],4), p['top1']) for p in points]}")
    print(f"  Staged Pareto FRONTIER (non-dominated): {frontier}")
    print(f"  DOMINATED points:")
    for d in dominated:
        print(f"    {d['config']} (BOPs={d['bops']:.4e}, top1={d['top1']}%) dominated by {d['dominated_by']}")

    expected_dominated = {"r22_w16a16_best", "r15_w16a16_best", "r7_w16a16_best"}
    actual_dominated = {d["config"] for d in dominated}
    assert actual_dominated == expected_dominated, (
        f"Expected dominated set {expected_dominated}, got {actual_dominated} -- the pre-registered "
        "finding ('41/57.2/80% points are all dominated by the 29.3% config') does not hold as stated; "
        "reporting the ACTUAL result, not silently reconciling it."
    )

    # The per-point scan above reports the FIRST dominator found (e.g.
    # r22 is reported as dominated by r17, since r17 is cheaper and also
    # beats it) -- that's a real dominator, but not necessarily the
    # SINGLE point the pre-registered claim names. Separately, explicitly
    # verify the STRONGER, more specific claim: r2_w8a8_best (29.3%)
    # ALONE dominates all three simultaneously (a single cheaper
    # reference point beats each of them, not just "some point or other
    # does") -- this is what "the staged frontier saturates at 29.3%"
    # actually asserts.
    r2_point = next(p for p in points if p["config"] == "r2_w8a8_best")
    for target_config in expected_dominated:
        target = next(p for p in points if p["config"] == target_config)
        assert r2_point["bops"] <= target["bops"] and r2_point["top1"] >= target["top1"], (
            f"r2_w8a8_best does NOT single-handedly dominate {target_config} "
            f"(r2: BOPs={r2_point['bops']:.4e} top1={r2_point['top1']}%; "
            f"{target_config}: BOPs={target['bops']:.4e} top1={target['top1']}%)"
        )
    print(f"  ASSERTION PASSED (per-point scan): {sorted(expected_dominated)} are each dominated by "
          "SOME cheaper-and-at-least-as-accurate point on the frontier.")
    print(f"  ASSERTION PASSED (specific pre-registered claim): r2_w8a8_best (29.3%) ALONE dominates ALL "
          "THREE simultaneously -- \"the published-methods staged frontier saturates at ~29% of C_id.\"")

    return {"frontier": frontier, "dominated": [d["config"] for d in dominated], "points": points,
            "staged_frontier_saturation": {
                "saturating_config": "r2_w8a8_best", "saturating_bops_frac_pct": round(
                    points[[p["config"] for p in points].index("r2_w8a8_best")]["bops"] / r0_bops * 100, 2),
                "dominated_configs": sorted(actual_dominated),
            }}


def generate_figure(pareto_df, pareto_result, out_path):
    """Item #5's figure: TANDEM's joint curve, staged points split into
    frontier (non-dominated) vs dominated (per item #4), on one axis.
    Matches this project's existing P2.1 figure style (experiments/phase2_1_fleet/
    P2_tandem_pipeline.py's cross_model_pareto plot) for visual
    consistency: figsize=(9,6), "o-" markers, dpi=150."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(pareto_df["C"], pareto_df["top1_50k_pct"], "o-", color="#1f77b4", label="TANDEM (joint)", zorder=3)

    points = pareto_result["points"]
    frontier_pts = [p for p in points if p["config"] in pareto_result["frontier"]]
    dominated_pts = [p for p in points if p["config"] in pareto_result["dominated"]]

    ax.scatter([p["bops"] for p in frontier_pts], [p["top1"] for p in frontier_pts],
               marker="s", s=90, color="#d62728", edgecolors="black", linewidths=0.8, zorder=4,
               label="Staged (best-of-both), Pareto frontier")
    ax.scatter([p["bops"] for p in dominated_pts], [p["top1"] for p in dominated_pts],
               marker="x", s=90, color="#7f7f7f", linewidths=2.0, zorder=4,
               label="Staged (best-of-both), DOMINATED")

    for p in points:
        ax.annotate(p["config"].replace("_best", "").replace("_w16a16", "\nw16a16").replace("_w8a8", "\nw8a8"),
                    (p["bops"], p["top1"]), fontsize=7, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Cost C(plan) [BOPs, common/bops.py convention]")
    ax.set_ylabel("Measured top-1 accuracy (50k)")
    ax.set_title("DeiT-S: TANDEM (joint) vs B4 staged composites (best-of-both-orders)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


def programmatic_comparison_table(results_df, pareto_df):
    """Item #5: comparison table with BOTH sides' ACTUAL measured BOPs
    (TANDEM's own `C` column from its real plans.csv -- the budget x
    utilization actually achieved, not the nominal budget target), read
    programmatically, never typed. Bias-flag on the 3 genuine-composite
    (w8a8, attention-skip-bias-affected) staged rows."""
    b4_rows = results_df[(results_df["method"] == "sequential_composite") &
                          (results_df["row_kind"] == "comparison")].copy()
    rows_out = []
    for _, r in b4_rows.iterrows():
        bops = float(r["bops_ours"])
        staged_top1 = float(r["top1_50k"])
        tandem_top1, extrapolated = interpolate_tandem_top1_pct(pareto_df, bops)
        delta = tandem_top1 - staged_top1
        bias_flagged = "w8a8" in r["config"]  # the 3 genuine composites (15/21/29.3%) quantize; the 3 pure-ToMe don't
        rows_out.append({
            "config": r["config"], "staged_bops": bops, "staged_top1_pct": staged_top1,
            "tandem_bops_note": "interpolated at staged's exact BOPs (TANDEM's own measured C values, "
                                 "programmatically loaded from plans.csv)",
            "tandem_top1_pct": round(tandem_top1, 4), "delta_tandem_minus_staged_pp": round(delta, 4),
            "extrapolated": extrapolated,
            "attention_skip_bias_flag": bias_flagged,
        })
    rows_out.sort(key=lambda r: r["staged_bops"])

    print("\n=== Item #5: programmatic comparison table (TANDEM's real plans.csv, read programmatically) ===")
    print(f"{'config':<18}{'staged_bops':>14}{'staged_top1':>13}{'tandem_top1':>13}{'delta(T-S)':>12}"
          f"{'bias_flag':>11}{'extrap':>8}")
    for r in rows_out:
        print(f"{r['config']:<18}{r['staged_bops']:>14.4e}{r['staged_top1_pct']:>13.3f}"
              f"{r['tandem_top1_pct']:>13.3f}{r['delta_tandem_minus_staged_pp']:>+12.3f}"
              f"{str(r['attention_skip_bias_flag']):>11}{str(r['extrapolated']):>8}")

    print("\n  TANDEM's own real Pareto points (loaded programmatically from plans.csv, not typed):")
    print(f"  {pareto_df[['C', 'top1_50k_pct']].to_string(index=False)}")
    return rows_out


def main():
    results_csv = BASELINES_ROOT / "results.csv"
    assert results_csv.exists(), f"{results_csv} not found."
    results_df = pd.read_csv(results_csv)

    pareto_df = load_tandem_pareto(MODEL_NAME)
    assert pareto_df is not None, f"No TANDEM P2.1 Pareto data found for {MODEL_NAME}."

    # canonical_c_id comes from the canonical summary directly -- NOT from
    # pareto_df's own max(C). An earlier version of this script assumed
    # those were the same (the "identity/no-reduction plan" reasoning);
    # they are NOT: P2.1's own highest-budget plan (nominally 80% of C_id)
    # only achieved 96.21% utilization of that budget in practice
    # (budget_utilization in the canonical summary), so pareto_df's real
    # max C sits at ~77% of true C_id, not 100%. Caught by an assertion
    # that fired on this exact mismatch during this session's own
    # verification -- corrected here rather than silently trusting the
    # wrong assumption. This is also independent, direct confirmation of
    # why B4's r=7 point (79.34% of C_id) is flagged `extrapolated=True`
    # in the comparison table below: it genuinely exceeds TANDEM's own
    # highest REAL measured point on this curve.
    summary_path = (BASELINES_ROOT.parent.parent / "experiments" / "phase2_1_fleet" / "results" / "P2" / MODEL_NAME /
                     f"P2_{MODEL_NAME}_summary.json")
    canonical_summary = json.loads(summary_path.read_text())
    canonical_c_id = canonical_summary["c_id"]
    print(f"canonical_c_id (from canonical summary) = {canonical_c_id:.6e}")
    print(f"TANDEM plans.csv's own max measured C    = {float(pareto_df['C'].max()):.6e} "
          f"({float(pareto_df['C'].max())/canonical_c_id*100:.2f}% of C_id -- NOT 100%, see comment above)")

    cost_split = structural_cost_split(canonical_c_id)
    bias = attention_skip_bias(results_df)
    pareto = pareto_dominance_analysis(results_df, pareto_df, canonical_c_id)
    table = programmatic_comparison_table(results_df, pareto_df)
    fig_path = Path(__file__).parent / "B4_deit_small_pareto.png"
    generate_figure(pareto_df, pareto, fig_path)

    summary = {
        "model": MODEL_NAME, "c_id": canonical_c_id,
        "structural_cost_split": cost_split,
        "composite_attention_skip_bias_pp": {"floor": bias["floor_pp"], "est_range": bias["est_range_pp"]},
        "staged_frontier_saturation": pareto["staged_frontier_saturation"],
        "staged_pareto_frontier": pareto["frontier"],
        "comparison_table": table,
        "figure_path": str(fig_path.name),
    }
    out_path = Path(__file__).parent / "B4_deit_small_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
