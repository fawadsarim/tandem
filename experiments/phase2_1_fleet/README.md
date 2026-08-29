# P2.1 — the fleet: TANDEM's allocator, run for real on 4 architectures

Takes E5's joint (bits+tokens) Pareto-DP allocator — validated on one
model in Phase 1 — and runs it for real, end-to-end, on four different
Vision Transformer architectures: DeiT-Tiny, DeiT-Small, DeiT-Base, and
ViT-Base. Builds each model's own per-block damage tables from real
quantization/merge measurements (not reused from E3/E4 — those were
DeiT-Small only), runs the exact same Pareto-DP allocator per model, and
audits predicted-vs-measured accuracy at 6 budget fractions of each
model's own identity cost `C_id`.

This is the artifact every later phase reads from, read-only:
`experiments/phase2_2_baselines/` (P2.2) gates every baseline's
comparison point against these summaries, and `jetson/` (G1-HW) reads
DeiT-Small's own `Dq_table.csv`/`Dm_table.csv`/`plans.csv` directly to
build its hardware-cost allocator (`jetson/replan.py`'s
`--verify-port` mode proves that port faithful against this phase's own
`plans.csv`, 6/6 MATCH).

## One notebook, one model per run

`P2_tandem_pipeline.py` (jupytext-paired with the `.ipynb`) runs the full
pipeline for whichever model is set in `CONFIG["MODEL_NAME"]` — edit that
one line and re-run for each of the 4 architectures. T4 GPU, ~20–40
minutes per model. Output lands in `results/P2/<model>/`.

## Per-model results

| Model | FP top-1 (measured / published) | C_id (identity BOPs) | C_min (min-BOPs plan) | Family-X fit (slope / R², n=8) |
|---|---|---|---|---|
| DeiT-Tiny | 72.166% / 72.19% (S8 diff 0.024pp) | 3.135e11 | 2.419e10 | 0.838 / 0.987 |
| DeiT-Small | 79.848% / 79.856% (S8 diff 0.008pp) | 1.162e12 | 7.882e10 | 0.983 / 0.997 |
| DeiT-Base | 81.97% / 81.98% (S8 diff 0.010pp) | 4.467e12 | 2.794e11 | 0.830 / 0.993 |
| ViT-Base | 85.10% / 85.108% (S8 diff 0.008pp) | 4.467e12 | 2.794e11 | 0.956 / 0.994 |

**S8 sanity gate**: every model's own measured full-precision top-1 is
within 0.5pp of its published reference before any compressed-plan result
for that model is trusted (all 4 pass here, comfortably, at ≤0.024pp).
**Family-X fit**: predicted-vs-measured distortion on 8 combined-axis
(bits+tokens) held-out configurations per model, the direct test of
whether the additively-separable `D_pred` model (proven in Phase 1)
transfers to a real joint bits+tokens configuration on each architecture
— all 4 pass the registered [0.8,1.2]-slope / R²≥0.85 criterion by a wide
margin.

Each model's plans are selected at the same 6 budget fractions of its own
`C_id`: 15%, 21%, 29.3%, 41%, 57.2%, 80% — the same plan-space constants
(`BIT_MENU`, `MERGE_BLOCKS`, `MERGE_RS`) as E5, reproduced verbatim
wherever they're needed downstream (`jetson/replan.py`,
`scripts/verify_p2_claims.py`), never re-derived.

## Files

```
P2_tandem_pipeline.{py,ipynb}   The pipeline: per-model damage-table construction,
                                 the Pareto-DP allocator, plan audit, cross-model comparison
results/P2/<model>/             Per-model: Dq_table.csv, Dm_table.csv (raw + fitted),
                                 plans.csv, familyX.csv, calibration_indices.json,
                                 plan_composition.txt, P2_<model>_summary.json
results/P2/cross_model_comparison.csv,
results/P2/cross_model_fidelity.csv,
results/P2/cross_model_pareto.png   Fleet-level rollups across all 4 models
```

## Reproducing

No-GPU verification: `python scripts/verify_p2_claims.py` (from repo
root) independently recomputes, per model, the Family-X fit, the S8
sanity gate, every plan's `D_pred`, `C_id`/`C_min`, and the per-budget
fidelity ratios directly from committed CSV/JSON, and diffs against each
model's stored summary — 20/20 checks (5 per model × 4 models). See
`docs/REPRODUCING.md` Tier 1. Full GPU re-run: `docs/REPRODUCING.md`
Tier 2 / `docs/REVIEWER_RUNBOOK.md`'s "P2.1" section — run once per model,
recommended order deit_small (fastest, most-referenced-downstream) first.
