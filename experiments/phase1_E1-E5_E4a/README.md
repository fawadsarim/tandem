# Phase 1 — E1 through E5, plus E4a

The foundational experiments: build and validate the per-block distortion
model `D ~ Sum_l S_l * eps_l` that every later phase (P2.1's fleet
allocator, G1-HW's hardware validation) depends on, then run the "kill
test" — does a JOINT (bits+tokens) allocator beat the best STAGED
allocator at matched simulated cost?

Each experiment is a jupytext-paired `.py`/`.ipynb` (the `.py` is the
source of truth; `.ipynb` is regenerated from it, never hand-edited) that
runs top-to-bottom on a single Colab GPU (T4 is what these were built and
measured against — see each notebook's own GPU-check cell). Committed
output lives in `results/<experiment>/` — raw CSVs, summary JSONs, and
figures, all real, not regenerated for this repo.

## Run order and what each one proves

| # | Notebook | Proves | Headline result |
|---|---|---|---|
| **E1** | `E1_sensitivity_linearity.py` | Per-block sensitivity constants `S_l`: distortion grows linearly in injected Gaussian noise energy in each block's own regime — the load-bearing assumption the whole distortion model rests on. | Per-block `S_l`/`m_l` fitted, R² per block in the 0.9995–0.99998 range (`results/E1/E1_summary.json`). |
| **E2** | `E2_additivity.py` | Additivity: distortion from perturbing SETS of blocks simultaneously equals the sum of single-block contributions — what lets the allocator score a whole plan by summing independent per-block terms. | Predicted-vs-measured MSE fit: slope=1.0068, R²=0.9987, n=129 (`results/E2/E2_summary.json`), PASSED against the registered acceptance criterion (slope in [0.85,1.15], R²≥0.9). |
| **E3** | `E3_quantization_curves.py` | Gaussian-calibrated sensitivities transfer to REAL quantization damage, and additivity still holds under real quantization (up to 11 simultaneous blocks). | Quantization-axis fit: slope=1.118, R²=0.995, n=8 (`results/E3/E3_summary.json`, `q3_fit`); `overall_pass=True`. |
| **E4** | `E4_token_merge_curves.py` | Per-block token-merge rate-distortion curves, the merge-shaped transfer test, and additivity of merge damage across a multi-point schedule. | `overall_pass=True`; the measured-damage-table predictor (not the original Gaussian-sensitivity predictor — see note below) fits at slope=1.149, R²=0.9993, n=4 (`results/E4/E4b_additivity_reanalysis.json`). |
| **E4a** | `E4a_ward_verification.py` | The mathematical foundation E4 depends on: size-weighted-mean (Ward) merging injects distortion energy equal to the exact Ward two-point cost — a closed-form algebraic identity, not an approximation. | All 3 claims (two-point identity, telescoping SSE, greedy==Ward equivalence) verified numerically to ~1e-15 relative error (`results/E4a/E4a_verification.json`, `all_passed=True`). |
| **E5 (G1)** | `E5_allocator_G1.py` | **The central Phase-1 claim**: build the JOINT (bits+tokens) allocator from E1–E4's measured damage tables and test it against the best STAGED/sequential baseline at matched simulated (BOPs) cost — the "kill test" the whole program feeds. | See "The E5 kill test result" below — this is more subtle than a plain PASS. |

## A real nuance in E3/E4: two verdicts per experiment, not one

Both E3 and E4 originally tested a **Gaussian-sensitivity predictor**
(built purely from E1's own fitted `S_l`) against the REAL, measured
damage from quantization (E3) / merging (E4). That specific transfer test
(`V2_transfer` in E3, `V2_E4` in E4) does **not** pass — and is explicitly
marked `"gating": false` in both summary JSONs, with a `note` explaining
why: **the allocator was never built on that predictor**. It consumes
measured damage tables (`Dq_table.csv`/`Dm_table.csv`, built directly from
E3/E4's own real quantization/merge measurements) directly, not a
Gaussian-sensitivity extrapolation. The Gaussian-predictor transfer test
characterizes cross-mechanism anisotropy (a real, reported finding — see
each summary's own `V2_transfer`/`V2_E4` fields) but was never meant to
gate anything, and the actual gating predictor (`V3_quant_additivity` in
E3, the E4b measured-damage refit in E4) does pass. Read each summary
JSON's own `verdicts` field before citing a single "E3 passed" / "E4
passed" headline — there are two different questions being answered, and
only one of them was ever registered as a gate.

## The E5 kill test result — read this before citing "E5 passed/failed"

E5's `overall_pass` in `results/E5/E5_summary.json` is **`False`** — and
that is not a defect, nor does it mean the allocator doesn't work. Every
individual correctness gate in E5 (artifact loading, `m_l` consistency,
identity-plan reproduction, table fidelity, executor reproduction, solver
soundness/hull completeness, mechanism toggle) **passes**
(`gate_results`, all `passed: true`). The one verdict that reads `false`
is `V2_E5_G1_kill_test`:

```
"mean_delta_top1_pp": 0.0, "mean_ci_low_pp": 0.0, "mean_ci_high_pp": 0.0,
"passed": false, "mechanism": "exact_tie_via_plan_collapse"
```

The kill test was pre-registered to require the JOINT allocator to
**strictly beat** the best STAGED baseline (a confidence interval
excluding zero on the positive side) at matched simulated (BOPs) cost.
What actually happens, at all 6 tested budgets, is an **exact tie**
(`plans_identical=True` in every budget) — JOINT and STAGED select the
literal same plan, because `D_pred` is additively separable in
(schedule, bits) whenever the joint-optimal schedule is also
token-frontier-efficient (26/85 schedules are; all 6 joint schedules are
among them — see `results/E5/E5_schedule_token_frontier.csv`). This is a
real, structural, independently-reproducible finding — **staging
collapse** — not noise and not a bug: `scripts/verify_g1_claims.py`
re-derives it from scratch (not by reading the stored CSV) and gets the
same 26/85 count and the same exact tie.

Read literally against its own pre-registered criterion, the kill test
did not pass under the simulated BOPs cost proxy. That is exactly what
motivated re-testing the same joint-vs-staged question against **real,
measured hardware cost** instead of a simulated proxy — see
`jetson/README.md` and `docs/CHARACTERIZATION.md` for how that turned
out (axis domination: on real Jetson latency, the precision axis
contributes negligibly, which is a different and independently
interesting resolution of this same tie, not a contradiction of it).

## Files

```
E1_sensitivity_linearity.{py,ipynb}   Per-block Gaussian sensitivity constants
E2_additivity.{py,ipynb}              Multi-block additivity of Gaussian-injected distortion
E3_quantization_curves.{py,ipynb}     Real quantization damage + additivity under real quantization
E4_token_merge_curves.{py,ipynb}      Real token-merge damage curves + schedule additivity
E4a_ward_verification.{py,ipynb}      Closed-form proof of the Ward-merge energy identity (no GPU)
E5_allocator_G1.{py,ipynb}            The joint allocator + the staging-collapse kill test
results/{E1..E5,E4a}/                 Committed raw CSVs, summary JSONs, and figures per experiment
```

## Reproducing

No-GPU verification: `python scripts/verify_g1_claims.py` (from repo
root) independently recomputes E5's headline claims (plans_identical, the
exact tie, the 26/85 frontier mechanism, the Family-X and pooled fidelity
fits) from committed CSV/JSON, and diffs against what's stored here — see
`docs/REPRODUCING.md` Tier 1. Full GPU re-runs: `docs/REPRODUCING.md`
Tier 2 / `docs/REVIEWER_RUNBOOK.md` (one notebook per experiment, T4 GPU,
minutes each; E4a needs no GPU at all).
