# Reproducing TANDEM

Two tiers. **Tier 1** verifies every headline claim from committed
artifacts — no GPU, no device, minutes. **Tier 2** re-runs the actual
experiments that produced those artifacts — GPU and, for the final phase,
a physical Jetson Orin Nano Super. Ground truth is what's committed; a
Tier-2 rerun is a *check against* it, not a replacement for it.

## Gate semantics — read this before Tier 2

Two different kinds of gate appear throughout this codebase, and they
behave differently on purpose:

- **FP-parity gate failure HALTS.** If a baseline's freshly-loaded model
  doesn't reproduce its own published (or pinned) full-precision
  accuracy, the run stops before spending any GPU time on a number built
  on a broken foundation. Nothing downstream is trusted.
- **Reproduction-gate failure is RECORDED, and the run continues.** If a
  reproduced number falls outside its tolerance band, that deviation is
  written to `results.csv`/the relevant summary JSON and the run keeps
  going — the deviation itself becomes part of the record (see
  `docs/deviations.md`), not a silent retry or a crash.

Knowing which is which matters when Tier 2 doesn't finish cleanly: a HALT
means stop and diagnose before trusting anything; a RECORDED deviation
means read what was recorded before assuming something is broken.

---

## Tier 1 — verify from committed artifacts (no GPU, ~5 minutes)

```bash
git clone https://github.com/fawadsarim/tandem.git
cd tandem
pip install pandas numpy
bash scripts/run_reviewer.sh
```

Runs 7 steps in order, stops at the first failure:

1. `python scripts/verify_p2_claims.py` — P2.1 headline claims, all 4 models
2. `python scripts/verify_g1_claims.py` — E1-E5/E4a (Phase 1, the joint-vs-staged "kill test")
3. `python scripts/verify_baselines_claims.py` — P2.2 (baselines B1-B5) headline claims
4. `(cd experiments/phase2_2_baselines && python -m common.bops_identity_check)` — cost-ruler unit checks
5. `(cd experiments/phase2_2_baselines/b4_sequential_composites && python analyze_deit_small.py)` — B4 closure re-derivation
6. `(cd experiments/phase2_2_baselines && python b5_final_closure.py)` — B5 final closure re-derivation
7. `(cd jetson && python self_test.py)` — G1-HW static self-test (11 checks, including the real port-equivalence gate)

**Expected output** — this is real, captured output from an actual run on
a clean checkout, not a mock-up (excerpted; each step's own full output is
considerably longer):

```
======================================================================
TIER 1 -- verify from artifacts (no GPU)
======================================================================

--- [1/8] P2.1 headline claims ---
====================================================================================================
P2 HEADLINE CLAIMS -- READ-ONLY VERIFICATION -- ...
====================================================================================================
[MATCH] [deit_tiny_patch16_224] 1. Family-X fit (family_x_gate)
  recomputed: slope=0.8376682792752251, r2=0.9868636079613713, n=8, passed=True
  stored:     slope=0.8376682792752252, r2=0.9868636079613714, n=8, passed=True
... (5 checks x 4 models) ...
====================================================================================================
ALL CLAIMS VERIFIED (20 checks across 4 models)
====================================================================================================

--- [2/8] E1-E5/E4a (Phase 1, G1 kill test) ---
====================================================================================================
[MATCH] 1. plans_identical (joint == best_sequential, all 6 budgets)
[MATCH] 2. exact tie (mean delta_top1_pp == 0.0, zero-width CI)
[MATCH] 3. frontier mechanism (26/85 frontier-efficient, all 6 joint schedules on it)
[MATCH] 4. Family-X cross-axis fit (V1_E5_cross_axis_additivity)
[MATCH] 5. corrected pooled fidelity fit (pred_vs_measured_fidelity_all_plans)
====================================================================================================
ALL CLAIMS VERIFIED
====================================================================================================

--- [3/8] P2.2 (baselines B1-B5) headline claims ---
[... 6 checks, ALL CLAIMS VERIFIED ...]

--- [4/8] Cost-ruler unit checks ---
[... BOPs identity-config equivalence checks, all pass ...]

--- [5/8] B4 closure re-derivation ---
[... structural cost split, attention-skip bias, Pareto dominance, all recomputed and printed ...]

--- [6/8] B5 final closure re-derivation ---
[... cross-baseline comparison table regenerated, compare_summary.json + figure rewritten. Expected: the
     JSON's numbers are byte-identical to the committed file; the PNG bytes vary with matplotlib version ...]

--- [7/8] G1-HW (jetson/) static self-test ---
=== jetson/self_test.py (HOST dry-run) ===
  [PASS] 1. Checkpoint pinning consistency: ...
  [PASS] 2. Versions recorded (setup.sh output): ...
  [PASS] 3. Clocks locked (jetson_clocks): N/A on this host -- not running on L4T
  [PASS] 4. TF32 n/a / explicitly disabled note: ...
  [PASS] 5. Manifest hashes present and well-formed: ...
  [PASS] 6. LUT schema: 15 entries, schema OK
  [PASS] 7. Resume behavior (measure_lut.py skip-if-present logic): ...
  [PASS] 8. Secrets scan (every file in jetson/): zero hits ...
  [PASS] 9. Read-only guard on P2.1 paths: ...
  [PASS] 10. Real port-equivalence gate (replan.py --verify-port): 6/6 MATCH
  [PASS] 11. Gate ordering (bring-up -> engines -> LUT -> replan -> eval): dependency chain intact

11/11 checks passed.
ALL CHECKS PASSED.

--- [8/8] G1-HW-2 (jetson-orinnx/) headline re-derivation + static self-test ---
=== verify_orinnx_claims.py: re-deriving FINDINGS.md headline numbers & cross-file consistency ===
[MATCH] proxy_error(N=69/fp16, MAXN) ... 2.2582
[MATCH] n=69..197 canonical LUT int8/fp16 ratios ... 1.1234 / 1.1646 / 1.2529 / 1.3204 / 1.4357
[MATCH] Latency_id (NX) ... 3.7202   [MATCH] floor fraction ... 0.7693
[MATCH] Nano-table infeasible at NX budgets ... 4/4   [MATCH] reproduction gate ... 10/10, max 1.88%
[MATCH] energy instrument monotonic ladder ... holds
ALL CLAIMS & CROSS-FILE CHECKS VERIFIED -- zero staleness, complete consistency.
=== jetson-orinnx/self_test.py (HOST dry-run) ===  [... 11/11 checks passed ...]

======================================================================
TIER 1: ALL 8 STEPS PASSED.
======================================================================

```

**On a host machine (no physical Jetson), step 11's checks 2 and 3 report
"N/A on this host" rather than PASS/FAIL** — that's expected, not a gap;
those checks need real device state. Everything else, including the real
port-equivalence gate (check 10) and the gate-ordering check (check 11),
is a genuine, GPU-free assertion against committed data and passes for
real.

A failure anywhere in Tier 1 means: **do not cite any number from this
repo until diagnosed** — that is the script's own final message on
failure, verbatim.

---

## Tier 2 — full re-measurement (GPU / device required)

Stage order matters: **E1 → E2 → E3 → E4 → E4a → E5 → P2.1 → B1 → B2 → B3
→ B4 → B5 → jetson**. Each later stage reads committed output from an
earlier one (e.g. P2.1's per-model summaries are read by every baseline's
FP-parity gate; jetson/'s replan reads P2.1's plans.csv directly).

| Stage | Environment | Command | Runtime |
|---|---|---|---|
| E1–E5, E4a | Colab (GPU) | `experiments/phase1_E1-E5_E4a/E*.ipynb`, Runtime → Run all | minutes each; E4a needs no GPU |
| P2.1 | Colab (GPU), once per model | `experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb`, edit `CONFIG["MODEL_NAME"]` | ~20–40 min/model |
| B1 (ToMe) | isolated venv, `torch==1.12.1` | `experiments/phase2_2_baselines/b1_tome/run.py --imagenet-dir $IMAGENET_DIR` | ~1–2 hr (full 50k, all r values) |
| B2 (RepQ-ViT) | isolated venv + orchestrator venv | `experiments/phase2_2_baselines/b2_repq_vit/run.py --imagenet-dir ... --python ...` | ~1–2 hr |
| B3 (FQ-ViT) | isolated venv, **`torch==1.7.1`, Python ≤3.8** | same shape as B2 | ~1–3 hr (older stack is slower) |
| B4 (composites) | isolated venv, `torch==1.12.1` | `experiments/phase2_2_baselines/b4_sequential_composites/run.py ...` | ~2–4 hr (both orders, both stages) |
| B5 (compare) | no GPU | `experiments/phase2_2_baselines/compare_pareto.py` then `b5_final_closure.py` | seconds |
| jetson/ (G1-HW) | physical Jetson Orin Nano Super | see `jetson/README.md` in full | real device time — hours across the full harness |
| jetson-orinnx/ (G1-HW-2) | physical Jetson Orin NX 8GB | see `jetson-orinnx/README.md` in full | real device time — hours across the full harness |

Full per-stage detail (exact secrets/env vars, resume behavior, expected
numbers with tolerance class, and what a legitimate deviation looks like
versus what should be reported as a bug) is in
[`docs/REVIEWER_RUNBOOK.md`](REVIEWER_RUNBOOK.md) — this table is the
map, that document is the manual. `bash scripts/run_reviewer.sh --tier2
--stage <name>` (or `--stage list`) drives what it safely can and prints
exact instructions for the rest.

**B3 (FQ-ViT) needs an older torch and is intentionally isolated** — see
`requirements/README.md` — never install its requirements into the same
venv as any other baseline.

### Legitimate deviation vs. bug

- **Legitimate**: a reproduction-gate deviation recorded in `results.csv`
  with a `gate_status`/`gate_diff_pct_points` entry, cross-referenced in
  `docs/deviations.md`. These are expected and documented; the number is
  still trustworthy, with its caveat attached.
- **Bug**: an FP-parity HALT, an unexplained crash, or a number that
  differs from the committed one by more than its documented tolerance
  *without* a corresponding `docs/deviations.md` entry. Report these.

### Resume behavior

B1/B4 checkpoint every `save_every_n_batches` batches within a run and
resume from that checkpoint on restart (config-hash-guarded — a resume
attempt against a different configuration starts fresh rather than
silently reusing stale state). B2/B3 resume at the `results.csv` row
level (`find_existing_row`) — an already-completed `(model, config,
row_kind)` cell is skipped, not re-run.

## No-GPU verification depth

Everything in Tier 1, plus: `jetson/replan.py --verify-port` (6/6 MATCH,
proves the ported DP allocator agrees with P2.1's own allocator under
identical cost semantics — the same command self_test.py's own check 10
already runs), and `python -m experiments.phase2_2_baselines.common.bops_identity_check`
from `experiments/phase2_2_baselines/` (cost-ruler formula equivalence).
Neither needs a model, a GPU, or network access.
