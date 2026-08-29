# tandem/ — method map (no code lives here)

This directory is a signpost, not a package. TANDEM's method — the
distortion model, the cost rulers, the Pareto DP allocator, and the plan
space it searches — is **not extracted into an importable package** as
part of the public-release reorganization: it lives in its original,
actively-used, gate-protected locations, and copying or refactoring it out
would be a rewrite, not a rename (this repo's own stated principle for
this reorg — see the top-level `README.md`). Extracting working, tested,
independently-verified allocator code into a new location without
byte-for-byte re-verification is exactly the kind of change that risks
silently altering a result no one would notice until much later.

If you're looking for the actual implementation:

| Method concept | Real location |
|---|---|
| Distortion model (`D ~ Sum_l S_l * eps_l`, per-block sensitivity) | `experiments/phase1_E1-E5_E4a/E1_sensitivity_linearity.py` (derivation), `experiments/phase1_E1-E5_E4a/results/E1/E1_summary.json` (fitted constants) |
| Cost ruler (BOPs formula) | `experiments/phase2_2_baselines/common/bops.py` — the canonical implementation, imported directly by every P2.2 baseline and unit-checked against P2.1's own formula in `bops_identity_check.py` |
| Pareto DP allocator (the joint bits+tokens optimizer) | `experiments/phase2_1_fleet/P2_tandem_pipeline.py`, sections 8-11 (plan-execution engine) and 10, 18-19 (the allocator itself) |
| Plan space (bit menu, merge blocks, merge-r grid, budget fractions) | Same file, section 10 — the constants (`BIT_MENU`, `MERGE_BLOCKS`, `MERGE_RS`, `BUDGET_FRACTIONS`) are reproduced verbatim (not re-derived) everywhere else they're needed: `jetson/replan.py`, `scripts/verify_p2_claims.py`, `scripts/verify_g1_claims.py` |
| The allocator, ported to real hardware cost | `jetson/replan.py` (Orin Nano) and `jetson-orinnx/replan.py` (Orin NX) — the identical DP, with `block_bops(n_l, bits)` replaced by measured `block_latency(n_l, bits)`; `--verify-port` proves the port is faithful (6/6 MATCH against P2.1's own committed plans) |
| Independent from-scratch reimplementations (not imports — the actual proof these numbers are real) | `scripts/verify_p2_claims.py`, `scripts/verify_g1_claims.py` |

See `docs/GLOSSARY.md` for what each of these terms means, and
`README.md`'s repository map for the full directory layout.
