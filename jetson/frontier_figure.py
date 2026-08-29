"""jetson/frontier_figure.py -- device-independent (matplotlib only, no device, no TensorRT --
runs anywhere frontier.py does). G1-HW final phase, Task 3's headline output -- see
jetson/README.md's "Re-scoped sweep" section.

Plots real measured latency (x) vs real measured top-1 accuracy (y):
  - The 6 BOPs-selected plans from the COMPLETED, real paired_eval.py run (dominated reference
    points -- read from out/paired_eval/paired_eval_result.json, already real, already
    committed).
  - The FP identity point (same source -- any budget row whose joint_plan is literally the
    identity plan reports the identical top1 by construction; asserted, not assumed).
  - ARM A / ARM B (this task's own two arms, from frontier.py's plan selection): if
    out/frontier/frontier_result.json exists (produced by frontier_paired_eval.py, Colab-only --
    see jetson/README.md), plots their real measured (latency, top1) points too. If it does NOT
    exist yet, plots their real MEASURED LATENCY only (from frontier_plans.csv, computed
    locally, no device needed) as ticks along the bottom, explicitly labeled "accuracy pending
    Colab run" -- real x-position, no fabricated y-value. This script never invents a number
    that hasn't been measured.

Copies the figure + frontier_result.json (if present) into out/reviewer_release/, per this
task's own output spec.

Usage: python frontier_figure.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json  # noqa: E402


def main():
    print("=== frontier_figure.py: accuracy vs measured latency (G1-HW final phase, Task 3) ===")

    paired_eval_path = JETSON_ROOT / "out" / "paired_eval" / "paired_eval_result.json"
    assert paired_eval_path.exists(), (
        f"{paired_eval_path} not found -- this is the completed, registered G1-HW run's real "
        "result (BOPs reference points + identity accuracy). Cannot plot without it."
    )
    paired_eval = read_json(paired_eval_path)

    bops_points = [(r["budget_fraction"], r["tau_ms"], r["bops_top1_pct"]) for r in paired_eval["per_budget"]]

    # Identify identity via the plan embedded in plans_hw.csv (bits all-16, merges=[]) rather
    # than guessing from budget -- read plans_hw.csv directly for the plan JSON.
    hw_path = JETSON_ROOT / "out" / "replan" / "plans_hw.csv"
    identity_top1 = None
    identity_latency_ms = None
    if hw_path.exists():
        hw_df = pd.read_csv(hw_path)
        found = []
        for i, row in hw_df.iterrows():
            plan = json.loads(row["plan_json"])
            is_identity = plan["merges"] == [] and all(b == 16 for b in plan["bits"].values())
            if is_identity:
                pe_row = paired_eval["per_budget"][i]
                found.append((float(row["latency_plan_latency_ms"]), pe_row["joint_top1_pct"]))
        if found:
            latencies = {round(f[0], 6) for f in found}
            accs = {round(f[1], 6) for f in found}
            assert len(accs) == 1, f"identity plan's measured top1 disagrees across budgets: {found}"
            identity_latency_ms, identity_top1 = found[0][0], found[0][1]
            print(f"Identity point (real, from {len(found)} agreeing budget row(s)): "
                  f"latency={identity_latency_ms:.4f}ms top1={identity_top1:.4f}%")
    if identity_top1 is None:
        print("NOTE: could not identify a real identity-plan row in plans_hw.csv -- identity "
              "point will be omitted from the figure (not fabricated).")

    frontier_plans_path = JETSON_ROOT / "out" / "frontier" / "frontier_plans.csv"
    assert frontier_plans_path.exists(), f"{frontier_plans_path} not found -- run frontier.py first."
    frontier_df = pd.read_csv(frontier_plans_path)

    frontier_result_path = JETSON_ROOT / "out" / "frontier" / "frontier_result.json"
    frontier_result = read_json(frontier_result_path) if frontier_result_path.exists() else None
    if frontier_result is None:
        print(f"NOTE: {frontier_result_path} not found -- ARM A/ARM B accuracy is not measured "
              "yet (needs frontier_eval_colab.ipynb on a Colab GPU). Plotting their real "
              "measured LATENCY only, as ticks, explicitly labeled pending -- no accuracy "
              "value is invented.")

    fig, ax = plt.subplots(figsize=(9, 6))

    bops_x = [p[1] for p in bops_points]
    bops_y = [p[2] for p in bops_points]
    ax.scatter(bops_x, bops_y, marker="x", color="gray", s=70, label="BOPs-selected (completed run, dominated reference)", zorder=3)
    for frac, x, y in bops_points:
        ax.annotate(f"{frac:.0%}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color="gray")

    if identity_top1 is not None:
        ax.scatter([identity_latency_ms], [identity_top1], marker="*", color="black", s=200,
                   label="FP identity (all-16-bit, no merges) -- real, measured", zorder=4)

    if frontier_result is not None and frontier_result.get("arm_a_frontier"):
        # REDUCED SCOPE (directed, 2026-07): ARM A has a real point at all 6 budgets
        # (frontier_result["arm_a_frontier"]); ARM B only has real points at the 2 budgets where
        # it's latency-feasible (frontier_result["paired_comparison"]["per_budget"]). Use each
        # arm's own measured cost from frontier_plans.csv (matched by budget_fraction), not tau
        # -- tau is the ceiling, cost is what was actually measured.
        cost_by_budget = {round(float(r["budget_fraction"]), 6): r for _, r in frontier_df.iterrows()}
        arm_a_pts = []
        for r in frontier_result["arm_a_frontier"]:
            key = round(float(r["budget_fraction"]), 6)
            row = cost_by_budget.get(key)
            if row is not None:
                arm_a_pts.append((float(row["arm_a_cost_ms"]), r["arm_a_top1_pct"]))
        arm_b_pts = []
        for r in frontier_result.get("paired_comparison", {}).get("per_budget", []):
            key = round(float(r["budget_fraction"]), 6)
            row = cost_by_budget.get(key)
            if row is not None:
                arm_b_pts.append((float(row["arm_b_cost_ms"]), r["arm_b_top1_pct"]))
        if arm_a_pts:
            ax.plot(*zip(*sorted(arm_a_pts)), marker="o", color="tab:blue",
                    label="ARM A -- TANDEM frontier (real, measured, all 6 budgets)", zorder=5)
        if arm_b_pts:
            ax.scatter(*zip(*arm_b_pts), marker="s", color="tab:orange", s=90,
                       label="ARM B -- uniform-r ToMe (real, measured, 2/6 feasible budgets)", zorder=6)
    else:
        arm_a_x = frontier_df["arm_a_cost_ms"].tolist()
        arm_b_x = frontier_df.loc[~frontier_df["arm_b_infeasible"], "arm_b_cost_ms"].tolist()
        ymin, _ = ax.get_ylim()
        y_tick = min(bops_y + ([identity_top1] if identity_top1 is not None else [])) - 1.0
        ax.scatter(arm_a_x, [y_tick] * len(arm_a_x), marker="|", color="tab:blue", s=200,
                   label="ARM A latency (6 budgets) -- accuracy PENDING Colab run", zorder=2)
        if arm_b_x:
            ax.scatter(arm_b_x, [y_tick - 0.6] * len(arm_b_x), marker="|", color="tab:orange", s=200,
                       label=f"ARM B latency ({len(arm_b_x)}/6 feasible budgets) -- accuracy PENDING Colab run", zorder=2)

    ax.set_xlabel("Measured latency (ms)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("G1-HW: accuracy vs measured latency (deit_small_patch16_224, Jetson Orin Nano Super)\n"
                  "NOT a joint-vs-staged test -- see CHARACTERIZATION.md", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = ensure_out_dir("frontier")
    fig_path = out_dir / "frontier_figure.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nWrote {fig_path}")

    release_dir = ensure_out_dir("reviewer_release")
    shutil.copy(fig_path, release_dir / fig_path.name)
    print(f"Copied to {release_dir / fig_path.name}")
    if frontier_result_path.exists():
        shutil.copy(frontier_result_path, release_dir / frontier_result_path.name)
        print(f"Copied to {release_dir / frontier_result_path.name}")
        # Real bug found and fixed, 2026-07: the PENDING placeholder this function writes
        # below (while frontier_result.json doesn't exist yet) was never cleaned up once the
        # real result landed -- confirmed for real, it silently lingered in reviewer_release/
        # alongside the actual result, misleadingly implying the run was still pending.
        stale_pending = release_dir / "frontier_result.PENDING.txt"
        if stale_pending.exists():
            stale_pending.unlink()
            print(f"Removed stale {stale_pending} (real result now present).")
    else:
        note_path = release_dir / "frontier_result.PENDING.txt"
        note_path.write_text(
            "frontier_result.json does not exist yet -- run frontier_eval_colab.ipynb on a "
            "Colab GPU (ARM A vs ARM B paired accuracy eval), then re-run frontier_figure.py "
            "to complete this release folder for real.\n"
        )
        print(f"Wrote {note_path} (frontier_result.json not yet produced -- see note).")


if __name__ == "__main__":
    main()
