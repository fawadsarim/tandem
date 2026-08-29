# results/ — pointers only, no duplicated data

This directory intentionally contains no result files. The canonical
result tables already live next to the code that produced them, and are
read directly from there by every verification script in this repo
(`scripts/verify_*_claims.py`, `jetson/replan.py`, etc.) — duplicating
them here would create a second copy that could silently drift out of
sync with the one everything actually reads.

| What you're looking for | Where it actually lives |
|---|---|
| Phase-1 (E1-E5, E4a) raw results | `experiments/phase1_E1-E5_E4a/results/{E1..E5,E4a}/` |
| P2.1 fleet (per-model plans, damage tables, summaries) | `experiments/phase2_1_fleet/results/P2/<model>/` |
| P2.2 baselines (unified results table, comparison summary) | `experiments/phase2_2_baselines/{results.csv,compare_summary.json}` |
| G1-HW hardware validation (LUT, replan, paired eval, energy, power sweep) | `jetson/out/` — already self-contained, see `jetson/README.md` |
| G1-HW-2 hardware validation (Orin NX LUT, replan, paired eval, energy, transfer) | `jetson-orinnx/out/` — already self-contained, see `jetson-orinnx/README.md` |
| Full manifest, one row per file, with hashes and the claim each supports | `evidence/00_MANIFEST.md` |

See `docs/RESULTS.md` for the annotated version of this same index, with
figure-regeneration status and what a user must supply themselves.
