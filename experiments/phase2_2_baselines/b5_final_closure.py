"""B5 (compare_pareto) final P2.2 closure pass (2026-07 spec).
Torch-free (pandas/numpy/matplotlib only, same orchestration-environment
constraint as compare_pareto.py -- see its own docstring and
deviations.md #1). Run via:

    python experiments/phase2_2_baselines/b5_final_closure.py

from experiments/phase2_2_baselines/, no GPU needed. Reuses compare_pareto.py's existing
`comparison_eligible_rows`/`load_tandem_pareto`/`interpolate_tandem_top1_pct`
unchanged (no forked re-implementation of the join/interpolation logic
this project already has and has already validated) -- this script adds
the POOLED cross-method Pareto-front analysis, per-model 4-panel figure,
flagged comparison table, and `compare_summary.json`/reviewer-release
packaging on top of that existing foundation.
"""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_pareto import comparison_eligible_rows, load_tandem_pareto, interpolate_tandem_top1_pct  # noqa: E402

BASELINES_ROOT = Path(__file__).resolve().parent
P2_RESULTS_ROOT = BASELINES_ROOT.parent.parent / "experiments" / "phase2_1_fleet" / "results" / "P2"
MODELS = ["deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224", "vit_base_patch16_224"]

# Row filter assertion targets (spec item B.3) -- real counts confirmed
# against actual results.csv this session (2026-07), not assumed.
EXPECTED_COMPARISON_COUNTS = {
    "deit_tiny_patch16_224": 3, "deit_small_patch16_224": 12,
    "deit_base_patch16_224": 3, "vit_base_patch16_224": 3,
}

# DeiT-S pooled-baseline Pareto front, as pre-registered (spec
# item C.6) -- asserted against the real computation below, not assumed.
EXPECTED_DEIT_SMALL_FRONT = ["r17_w8a8_best", "w6a6", "r10_w8a8_best", "r2_w8a8_best", "r0"]


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def assert_row_filter(results_df):
    """Spec item B.3."""
    comp = comparison_eligible_rows(results_df)
    counts = comp.groupby("model").size().to_dict()
    print("=== Item B.3: row-filter assertions ===")
    print(f"  Per-model row_kind=comparison counts: {counts}")
    for model, expected in EXPECTED_COMPARISON_COUNTS.items():
        actual = counts.get(model, 0)
        assert actual == expected, f"{model}: expected {expected} comparison rows, found {actual}"
    print(f"  ASSERTION PASSED: counts match {EXPECTED_COMPARISON_COUNTS} exactly.")

    all_comp = results_df[results_df["row_kind"] == "comparison"]
    n_trained_with_tome = (all_comp["setting"] == "trained_with_tome").sum()
    n_per_order = all_comp["config"].str.contains("merge_first|quant_first", na=False).sum()
    print(f"  setting=trained_with_tome rows within row_kind=comparison: {n_trained_with_tome}")
    print(f"  per-order composite rows (merge_first/quant_first) within row_kind=comparison: {n_per_order}")
    assert n_trained_with_tome == 0, "a trained_with_tome row leaked into row_kind=comparison"
    assert n_per_order == 0, "a per-order composite row leaked into row_kind=comparison"
    print("  ASSERTION PASSED: both are structurally impossible -- trained_with_tome and per-order "
          "composite rows are ALWAYS row_kind=reproduction, verified by direct query above, not "
          "just by construction.")
    return comp


def pooled_pareto_front(model_rows, r0_point=None):
    """Pooled Pareto front across ALL methods' comparison rows for one
    model (spec item C.6), same dominance definition as
    b4_sequential_composites/analyze_deit_small.py's single-method
    version, generalized to an arbitrary set of (label, bops, top1)
    points."""
    points = list(model_rows)
    if r0_point is not None:
        points = points + [r0_point]
    points.sort(key=lambda p: p["bops"])
    frontier, dominated = [], []
    for p in points:
        is_dominated = any(
            q is not p and q["bops"] <= p["bops"] and q["top1"] >= p["top1"] and
            (q["bops"] < p["bops"] or q["top1"] > p["top1"])
            for q in points
        )
        (dominated if is_dominated else frontier).append(p)
    return frontier, dominated, points


def build_model_points(results_df, model_name):
    """One (label, bops, top1, method, config) dict per comparison row for
    `model_name`, plus B1's r0 anchor if it exists for this model (only
    DeiT-S has one -- B1 is DeiT-S only)."""
    comp = comparison_eligible_rows(results_df)
    rows = comp[comp["model"] == model_name]
    points = []
    for _, r in rows.iterrows():
        label = f"{r['method']}/{r['config']}"
        points.append({"label": label, "method": r["method"], "config": r["config"],
                        "bops": float(r["bops_ours"]), "top1": float(r["top1_50k"])})

    r0_point = None
    r0_rows = results_df[(results_df["method"] == "tome") & (results_df["model"] == model_name) &
                          (results_df["config"] == "r0") & (results_df["row_kind"] == "reproduction")]
    if len(r0_rows) == 1:
        pareto_df = load_tandem_pareto(model_name)
        c_id = None
        summary_path = P2_RESULTS_ROOT / model_name / f"P2_{model_name}_summary.json"
        if summary_path.exists():
            c_id = json.loads(summary_path.read_text())["c_id"]
        r0_point = {"label": "tome/r0", "method": "tome", "config": "r0",
                    "bops": c_id if c_id is not None else float(r0_rows.iloc[0]["bops_ours"]),
                    "top1": float(r0_rows.iloc[0]["top1_50k"])}
    return points, r0_point


def load_attention_skip_bias_floor_pp():
    """Read B4's measured attention-skip-bias floor programmatically from
    b4_sequential_composites/B4_deit_small_summary.json (item #40's
    composite_attention_skip_bias_pp field) -- never re-typed."""
    path = BASELINES_ROOT / "b4_sequential_composites" / "B4_deit_small_summary.json"
    assert path.exists(), f"{path} not found -- run analyze_deit_small.py first."
    bias = json.loads(path.read_text())["composite_attention_skip_bias_pp"]
    return bias["floor"]


def main():
    results_csv = BASELINES_ROOT / "results.csv"
    assert results_csv.exists(), f"{results_csv} not found."
    results_df = pd.read_csv(results_csv)
    attention_skip_bias_floor_pp = load_attention_skip_bias_floor_pp()

    comp = assert_row_filter(results_df)

    # --- Item B.4: TANDEM curves read programmatically, per model ---
    print("\n=== Item B.4: TANDEM's real Pareto curves (read programmatically from plans.csv) ===")
    pareto_dfs = {}
    plans_provenance = {}
    for model in MODELS:
        pdf = load_tandem_pareto(model)
        assert pdf is not None, f"No TANDEM P2.1 plans.csv found for {model}."
        pareto_dfs[model] = pdf
        plans_path = P2_RESULTS_ROOT / model / "plans.csv"
        plans_provenance[model] = {
            "path": plans_path.relative_to(BASELINES_ROOT.parent).as_posix(),
            "sha256_16": sha256_of(plans_path),
        }
    print("\nDeiT-S TANDEM table (verbatim, for independent verification):")
    print(pareto_dfs["deit_small_patch16_224"][["C", "top1_50k_pct"]].to_string(index=False))

    # --- Item C.6: pooled-baseline Pareto front per model ---
    print("\n=== Item C.6: pooled-baseline Pareto front (all methods, per model) ===")
    fronts, all_points = {}, {}
    for model in MODELS:
        pts, r0 = build_model_points(results_df, model)
        frontier, dominated, points = pooled_pareto_front(pts, r0)
        fronts[model] = {"frontier": [p["label"] if p["config"] != "r0" else "r0" for p in frontier],
                          "dominated": [p["label"] if p["config"] != "r0" else "r0" for p in dominated]}
        all_points[model] = points
        # Full label (method/config), not bare config -- repq_vit and
        # fq_vit both use "w8a8" as their config string, so bare config
        # is ambiguous whenever two methods share a config name (which
        # they routinely do). The underlying dominance computation
        # already uses the full label internally (see build_model_points/
        # pooled_pareto_front) -- this is a print-clarity fix only.
        frontier_labels = [p["label"] if p["config"] != "r0" else "r0" for p in frontier]
        dominated_labels = [p["label"] if p["config"] != "r0" else "r0" for p in dominated]
        print(f"  {model}: frontier={frontier_labels}  dominated={dominated_labels}")

    actual_front_configs = sorted(p["config"] for p in all_points["deit_small_patch16_224"]
                                   if (p["label"] if p["config"] != "r0" else "r0") in fronts["deit_small_patch16_224"]["frontier"])
    expected_configs = sorted(EXPECTED_DEIT_SMALL_FRONT)
    print(f"\n  DeiT-S pooled front (by config): {actual_front_configs}")
    print(f"  Pre-registered front (by config): {expected_configs}")
    assert actual_front_configs == expected_configs, (
        f"DeiT-S pooled front mismatch: computed {actual_front_configs}, pre-registered {expected_configs}"
    )
    saturating = next(p for p in all_points["deit_small_patch16_224"] if p["config"] == "r2_w8a8_best")
    c_id_deit_small = json.loads((P2_RESULTS_ROOT / "deit_small_patch16_224" /
                                   "P2_deit_small_patch16_224_summary.json").read_text())["c_id"]
    saturation_frac = saturating["top1"]
    print(f"  ASSERTION PASSED: DeiT-S pooled front == pre-registered set exactly.")
    print(f"  Saturation ceiling: {saturating['top1']}% at {saturating['bops']/c_id_deit_small*100:.2f}% of C_id "
          f"(below 100% C_id: {saturating['bops'] < c_id_deit_small})")
    assert saturating["top1"] < 100.0 and saturating["bops"] < c_id_deit_small, "saturation ceiling check failed"

    # --- Item C.7: comparison table with flags ---
    print("\n=== Item C.7: per-model comparison table (flagged) ===")
    table_rows = []
    for model in MODELS:
        pdf = pareto_dfs[model]
        for p in all_points[model]:
            if p["config"] == "r0":
                continue  # r0 has no TANDEM-curve counterpart row to compare against; anchor only
            tandem_top1, extrapolated = interpolate_tandem_top1_pct(pdf, p["bops"])
            delta = tandem_top1 - p["top1"]
            label = p["label"] if p["config"] != "r0" else "r0"
            dominated_flag = label in fronts[model]["dominated"]
            attn_skip_flag = p["method"] == "sequential_composite" and "w8a8" in p["config"]
            cost_undercredited_flag = p["method"] == "fq_vit"
            # Item 3 of the 2026-07 record patch: a composite row's
            # apparent loss to TANDEM is not a real algorithmic loss if it's
            # smaller in magnitude than B4's own measured attention-skip-bias
            # FLOOR (item #40/#38) -- the entire observed delta is then fully
            # explained by the known measurement bias, not by TANDEM losing
            # on the merits. Only applies where attn_skip_flag is set (the
            # bias only exists for staged w8a8 composite rows).
            tie_within_bias_flag = attn_skip_flag and abs(delta) <= attention_skip_bias_floor_pp
            table_rows.append({
                "model": model, "method": p["method"], "config": p["config"],
                "baseline_bops": p["bops"], "baseline_top1_pct": p["top1"],
                "tandem_top1_pct": round(tandem_top1, 4), "delta_tandem_minus_baseline_pp": round(delta, 4),
                "extrapolated": extrapolated,
                "attention_skip_bias_flag": attn_skip_flag,
                "cost_under_credited_flag": cost_undercredited_flag,
                "dominated_flag": dominated_flag,
                "tie_within_measurement_bias_flag": tie_within_bias_flag,
            })
    table_df = pd.DataFrame(table_rows)
    print(table_df.to_string(index=False))
    tie_rows = [r for r in table_rows if r["tie_within_measurement_bias_flag"]]
    print(f"\n  Rows reclassified as 'tie within measurement bias' (|delta| <= "
          f"{attention_skip_bias_floor_pp}pp floor): "
          f"{[(r['model'], r['config'], r['delta_tandem_minus_baseline_pp']) for r in tie_rows]}")

    # --- Item 2 of the 2026-07 record patch: quantizer_orthogonality note ---
    # RepQ-ViT's two matched-BOPs cost points (w6a6, w8a8) beat TANDEM at
    # every model in the real comparison table (delta_tandem_minus_baseline_pp
    # < 0 in all 8 cases below) -- computed here from the same table_rows
    # built above, never re-typed.
    repq_rows = [r for r in table_rows if r["method"] == "repq_vit"]
    assert len(repq_rows) == 8, f"expected 8 repq_vit comparison rows (4 models x 2 cost points), found {len(repq_rows)}"
    deltas = [r["delta_tandem_minus_baseline_pp"] for r in repq_rows]
    max_delta_row = min(repq_rows, key=lambda r: r["delta_tandem_minus_baseline_pp"])  # most negative = biggest RepQ win
    print("\n=== Item 2 (record patch): quantizer_orthogonality ===")
    print(f"  RepQ-ViT matched-BOPs delta range across all 4 models' 2 cost points: "
          f"[{min(deltas)}, {max(deltas)}] pp")
    print(f"  Largest RepQ-ViT win: {max_delta_row['model']}/{max_delta_row['config']} "
          f"({max_delta_row['delta_tandem_minus_baseline_pp']}pp)")
    quantizer_orthogonality = {
        "note": (
            "RepQ-ViT's matched-BOPs wins over TANDEM at its two published cost points "
            "(w6a6, w8a8; all 4 models) reflect a quantizer-quality difference -- "
            "RepQ-ViT's reparameterized PTQ pipeline vs TANDEM's own simple per-block "
            "symmetric quantizer -- not allocation regret. Uniform-bit, r=0 configurations "
            "already lie inside TANDEM's own (r, per-block-bits) plan search space, so "
            "TANDEM's allocator is not structurally prevented from reaching this region; "
            "RepQ-ViT simply quantizes better at a given nominal bit-width than TANDEM's "
            "current quantizer does."
        ),
        "delta_range_pp": [min(deltas), max(deltas)],
        "max_delta_point": {"model": max_delta_row["model"], "config": max_delta_row["config"],
                             "delta_tandem_minus_baseline_pp": max_delta_row["delta_tandem_minus_baseline_pp"]},
        "stated_limitation": (
            "TANDEM's current quantizer (static per-tensor symmetric weight/activation "
            "quantization) is less sophisticated than RepQ-ViT's reparameterized, "
            "scale-search PTQ pipeline -- this is a quantizer-implementation limitation of "
            "TANDEM's current codebase, not evidence against TANDEM's cost-allocation search."
        ),
        "future_work": (
            "Composability: pairing TANDEM's own (r, bits)-allocation search with a "
            "stronger per-block quantizer (e.g. RepQ-ViT's reparameterization) is left as "
            "future work; the two contributions are orthogonal."
        ),
    }

    # --- Item C.5: 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    method_colors = {"tome": "#1f77b4", "repq_vit": "#2ca02c", "fq_vit": "#9467bd", "sequential_composite": "#d62728"}
    for ax, model in zip(axes.flat, MODELS):
        pdf = pareto_dfs[model]
        ax.plot(pdf["C"], pdf["top1_50k_pct"], "o-", color="black", label="TANDEM (joint)", zorder=2, linewidth=1.5)
        for p in all_points[model]:
            label = p["label"] if p["config"] != "r0" else "r0"
            is_dominated = label in fronts[model]["dominated"]
            marker = "x" if is_dominated else "o"
            color = "#7f7f7f" if is_dominated else method_colors.get(p["method"], "#333333")
            ax.scatter([p["bops"]], [p["top1"]], marker=marker, s=70, color=color,
                       edgecolors="black" if not is_dominated else None, linewidths=0.8, zorder=3)
            if "23" in p["config"] or "22" in p["config"]:
                ax.annotate(p["config"].replace("_best", ""), (p["bops"], p["top1"]), fontsize=6,
                           xytext=(4, 4), textcoords="offset points", color="#7f7f7f")
        # pooled-baseline Pareto front, as a step line
        frontier_pts = sorted([p for p in all_points[model]
                                if (p["label"] if p["config"] != "r0" else "r0") in fronts[model]["frontier"]],
                               key=lambda p: p["bops"])
        if len(frontier_pts) > 1:
            ax.step([p["bops"] for p in frontier_pts], [p["top1"] for p in frontier_pts],
                    where="post", color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
                    label="Pooled baseline Pareto front", zorder=1)
        if model == "deit_small_patch16_224":
            ax.annotate(f"saturation: {saturating['top1']}% @ {saturating['bops']/c_id_deit_small*100:.1f}% C_id",
                       (saturating["bops"], saturating["top1"]), fontsize=8, xytext=(10, -15),
                       textcoords="offset points", arrowprops=dict(arrowstyle="->", color="black"))
        ax.set_xscale("log")
        ax.set_xlabel("Cost C(plan) [BOPs, log scale]")
        ax.set_ylabel("Measured top-1 accuracy (50k)")
        ax.set_title(model)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("TANDEM (joint) vs published-method baselines, all 4 models -- P2.2 final closure")
    fig.tight_layout()
    fig_path = BASELINES_ROOT / "B5_all_models_pareto.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")

    # --- Item C.8: compare_summary.json + reviewer-release folder ---
    manifest_path = BASELINES_ROOT / "val_manifest_canonical.json"
    manifest_hash = json.loads(manifest_path.read_text())["manifest_hash"] if manifest_path.exists() else None

    summary = {
        "row_filter_assertions": {"counts": EXPECTED_COMPARISON_COUNTS, "trained_with_tome_leaked": 0,
                                   "per_order_leaked": 0},
        "pooled_pareto_fronts": fronts,
        "deit_small_saturation": {"config": "r2_w8a8_best", "top1_pct": saturating["top1"],
                                   "bops_frac_pct": round(saturating["bops"] / c_id_deit_small * 100, 2)},
        "comparison_table": table_rows,
        "attention_skip_bias_floor_pp": attention_skip_bias_floor_pp,
        "quantizer_orthogonality": quantizer_orthogonality,
        "manifest_hash": manifest_hash,
        "plans_csv_provenance": plans_provenance,
        "figure": str(fig_path.name),
    }
    summary_path = BASELINES_ROOT / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {summary_path}")

    release_dir = BASELINES_ROOT / "reviewer_release"
    release_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(fig_path, release_dir / fig_path.name)
    shutil.copy2(results_csv, release_dir / "results.csv")
    shutil.copy2(summary_path, release_dir / "compare_summary.json")
    b4_fig = BASELINES_ROOT / "b4_sequential_composites" / "B4_deit_small_pareto.png"
    if b4_fig.exists():
        shutil.copy2(b4_fig, release_dir / b4_fig.name)
    print(f"Copied figures + results.csv + compare_summary.json into {release_dir}")


if __name__ == "__main__":
    main()
