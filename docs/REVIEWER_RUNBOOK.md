# Reviewer Runbook

Two tiers. **Tier 1** checks that every headline number in this repo is
recomputable from what's already committed — no GPU, minutes, and it's
what `bash scripts/run_reviewer.sh` (no flags) does. **Tier 2** re-measures
everything from scratch on real hardware — hours, a GPU, and real
ImageNet/HF access. Start with Tier 1 regardless of whether you intend to
run Tier 2.

See `REPRODUCIBILITY.md` for the pin table (commits, checkpoints, hashes,
tolerance registry) referenced throughout this document, and the
top-level `README.md` for what each phase proves.

---

## TIER 1 — Verify from artifacts (no GPU, ~minutes)

Requirements: Python 3.10+, `pip install pandas numpy` (nothing else —
these scripts are deliberately dependency-light; no torch, no timm, no
GPU). Run from the repo root.

```
bash scripts/run_reviewer.sh              # runs everything below, in order, stops at the first failure
```

or run each step individually:

### 1. P2.1 headline claims

```
python scripts/verify_p2_claims.py
```
Recomputes, for all 4 models, independently from committed CSVs/JSON: the
Family-X through-origin fit, the S8 sanity gate, every plan's `D_pred`
(from `Dq_table.csv`/`Dm_table.csv` + kappa), `C_id`/`C_min` (BOPs formula
+ 85-schedule enumeration), and every per-budget fidelity ratio.

**Expected output** (verbatim tail):
```
====================================================================================================
ALL CLAIMS VERIFIED (20 checks across 4 models)
====================================================================================================
```

### 2. E1-E5/E4a (Phase 1, the G1 kill test)

```
python scripts/verify_g1_claims.py
```
Recomputes: joint plan == best-sequential plan at all 6 budgets (the
kill-test claim itself), the necessary zero-width-CI consequence, the
frontier mechanism (26/85 schedules token-frontier-efficient), the
cross-axis Family-X fit, and the pooled fidelity fit.

**Expected output** (verbatim tail):
```
====================================================================================================
ALL CLAIMS VERIFIED
====================================================================================================
```
(This script also appends a one-time marker line to `LAB_LOG.md` on its
first successful run in a given checkout — the only Tier-1 script with
any write side effect, and it's idempotent/non-destructive.)

### 3. P2.2 (baselines B1-B5) headline claims

```
python scripts/verify_baselines_claims.py
```
Recomputes: the reproduction gate's ±0.30pp rule for every gated
`results.csv` row, schema completeness (21 columns), `config_hash`
structural consistency (no two runs claiming to be the same measurement
with different pinned inputs), the equivalence fingerprint (if present),
and — independently — every B5 comparison-table delta plus the DeiT-S
pooled Pareto front, both recomputed from `results.csv`/`plans.csv` and
checked against `compare_summary.json`.

**Expected output** (verbatim tail):
```
====================================================================================================
ALL CLAIMS VERIFIED
====================================================================================================
```

### 4. Cost-ruler unit checks (BOPs formula, all 3 model shapes + composite reductions)

```
cd experiments/phase2_2_baselines && python -m common.bops_identity_check
```
**Expected**: identity-config `C_id` for all 3 `MODEL_DIMS` shapes matches
canonical summaries bit-for-bit; the composite-at-r=0-reduces-to-pure-PTQ
and composite-at-w16a16-reduces-to-pure-ToMe identities both hold exactly
(no message means pass — each `assert` prints its own confirmation line
and raises on failure; look for `PASSED` on every printed line, no
traceback).

### 5. B4 closure re-derivation (structural cost split, attention-skip bias, Pareto dominance)

```
cd experiments/phase2_2_baselines/b4_sequential_composites && python analyze_deit_small.py
```
**Expected** (key lines): `linear fraction (W16A16@r0) = 92.1232%`,
`attention fraction (constant) = 7.8768%`, `W8A8@r0 = 30.9076%`; both
Pareto-dominance assertions print `ASSERTION PASSED`.

### 6. B5 final closure re-derivation (pooled fronts, saturation, flagged comparison table)

```
cd experiments/phase2_2_baselines && python b5_final_closure.py
```
**Expected** (key lines):
```
ASSERTION PASSED: counts match {'deit_tiny_patch16_224': 3, 'deit_small_patch16_224': 12, 'deit_base_patch16_224': 3, 'vit_base_patch16_224': 3} exactly.
DeiT-S pooled front (by config): ['r0', 'r10_w8a8_best', 'r17_w8a8_best', 'r2_w8a8_best', 'w6a6']
ASSERTION PASSED: DeiT-S pooled front == pre-registered set exactly.
Saturation ceiling: 79.704% at 28.78% of C_id (below 100% C_id: True)
```
(Regenerates `compare_summary.json` and `B5_all_models_pareto.png` in
place — both are committed artifacts, so a clean checkout should produce
byte-for-byte-equivalent JSON content, modulo the `timestamp` field.)

### 7. G1-HW (jetson/) static self-test

```
cd jetson && python self_test.py
```
**Expected** (host dry-run, no Jetson/GPU present):
```
11/11 checks passed.
ALL CHECKS PASSED.
```
Most checks report `N/A on this host` for anything requiring real Jetson
hardware (`jetson_clocks`, `out/versions.json`) — that is the correct,
expected result off-device, not a failure being masked.

### What a Tier-1 FAILURE means

Any `[FAIL]` line, non-`0` exit code, or missing `ALL CLAIMS VERIFIED` /
`PASSED` banner above means a committed artifact no longer matches its
own recomputation — **stop and diagnose before citing any number from
this repo**. This is different from Tier 2's "legitimate deviation" cases
(see below) — Tier 1 never touches a GPU or re-measures anything, so
there is no floating-point jitter to absorb a mismatch; a Tier-1 failure
is either a real regression or a stale artifact.

---

## TIER 2 — Full re-measurement (GPU, real ImageNet/HF access required)

Exact order matters where noted — later stages consume earlier stages'
committed artifacts. Time estimates are for a **T4** (16GB); scale down
for an A100/H100, up for anything smaller. Where this project has a real
measured wall-clock time, it's cited as such; everything else is a
reasoned estimate, labeled.

### Stage order

```
E1 -> E2 -> E3 -> E4 -> E4a -> E5           (Phase 1, experiments/phase1_E1-E5_E4a/)
P2.1: deit_small -> deit_tiny -> deit_base -> vit_base   (experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb, 4 runs)
B1 (ToMe) -> B2 (RepQ-ViT) -> B3 (FQ-ViT) -> B4 (sequential composites)   (experiments/phase2_2_baselines/)
B5 (compare_pareto)                          (experiments/phase2_2_baselines/, no GPU — could run any time after B1-B4)
[optional] G1-HW                             (jetson/, real Jetson Orin Nano Super hardware)
[optional] G1-HW-2                           (jetson-orinnx/, real Jetson Orin NX 8GB hardware)
```

E4a has no ordering dependency on E1-E4 (it's a pure numerical proof, not
a measurement) but is listed after E4 because E4's own docstring cites
E4a's theorem as its foundation — read/run it there for context.
`E1_sensitivity_linearity` is DeiT-Small only throughout Phase 1 (E1-E5);
the 4-model fleet only begins at P2.1.

### Phase 1: E1 — per-block sensitivity + linearity

- **Command**: Colab, `experiments/phase1_E1-E5_E4a/E1_sensitivity_linearity.ipynb`, Runtime
  → Run all. (Or locally: `jupyter nbconvert --to notebook --execute
  experiments/phase1_E1-E5_E4a/E1_sensitivity_linearity.ipynb`, same environment-detection
  code path either way.)
- **Secrets/env**: `HF_TOKEN` (real ImageNet-1k access via HF — no
  Imagenette fallback).
- **Expected**: writes `experiments/phase1_E1-E5_E4a/results/E1/{E1_raw.csv,E1_summary.json,*.png}`;
  per-block sensitivity constants `S_l` fit with the linearity claim
  holding across blocks. Tolerance class: **stochastic-band** (GPU
  fp16/fp32 nondeterminism; compare against the committed summary within
  ordinary run-to-run jitter, not bit-for-bit).
- **Real measured runtime this project has seen**: ~19.7 min
  (`wall_clock_seconds_this_run=1182.06` in the committed summary).
- **Resume**: none — single top-to-bottom run; a failure partway requires
  a fresh Run All.
- **Legitimate deviation vs. report**: small (≤ a few %) shifts in
  individual `S_l` values or R² from GPU/library-version drift are
  legitimate — reported science, not a bug, as long as the LINEARITY
  claim (the thing E1 actually tests) still holds. A qualitatively
  different linearity verdict (the fit breaking down) is not a legitimate
  deviation — stop and report it as a possible regression.

### Phase 1: E2 — additivity

- **Command**: Colab, `experiments/phase1_E1-E5_E4a/E2_additivity.ipynb`, Run all. Requires
  E1's `results/E1/*summary*.json` + `*raw*.csv` already present (local
  repo path, else Drive mount, else manual upload).
- **Secrets/env**: `HF_TOKEN`.
- **Expected**: `experiments/phase1_E1-E5_E4a/results/E2/{E2_raw.csv,E2_summary.json,*.png}`;
  distortion from perturbing block SETS equals the sum of single-block
  contributions. Stochastic-band tolerance.
- **Real measured runtime**: ~13.1 min (`wall_clock_seconds_this_run=783.36`).
- **Resume**: none.
- **Legitimate deviation**: small interaction-term magnitude drift is
  expected/reported; a qualitative additivity failure is not.

### Phase 1: E3 — quantization curves + Gaussian-to-quant transfer

- **Command**: Colab, `experiments/phase1_E1-E5_E4a/E3_quantization_curves.ipynb`, Run all.
  Does not modify E1/E2; copies their already-verified machinery in.
- **Secrets/env**: `HF_TOKEN`.
- **Expected**: `experiments/phase1_E1-E5_E4a/results/E3/`; transfer + additivity under real
  quantization (Q3 required, Q1's −2 scaling law informative-only).
  Stochastic-band tolerance.
- **Real measured runtime**: ~25.9 min (`wall_clock_seconds_this_run=1552.21`
  — the longest of the four Phase-1 sweeps).
- **Resume**: none.
- **Legitimate deviation**: transfer-ratio magnitude drift within band is
  expected; Q3 additivity failing is not.

### Phase 1: E4 — token-merge curves + merge-shaped transfer

- **Command**: Colab, `experiments/phase1_E1-E5_E4a/E4_token_merge_curves.ipynb`, Run all.
- **Secrets/env**: `HF_TOKEN`.
- **Expected**: `experiments/phase1_E1-E5_E4a/results/E4/`; per-block merge rate-distortion
  curves, Q2 (anisotropy — informative, not required to pass), Q3
  (schedule additivity — required). Stochastic-band tolerance.
- **Real measured runtime**: ~5.7 min (`wall_clock_seconds_this_run=344.85`).
- **Resume**: none.
- **Legitimate deviation**: Q2 anisotropy pattern shifting is itself a
  finding, not a failure (see the notebook's own docstring); Q3
  additivity failing is not legitimate.

### Phase 1: E4a — Ward-linkage theorem verification

- **Command**: Colab or **local, no GPU needed** — pure numerical proof.
  `experiments/phase1_E1-E5_E4a/E4a_ward_verification.ipynb`, Run all (or `python
  experiments/phase1_E1-E5_E4a/E4a_ward_verification.py` directly).
- **Secrets/env**: none.
- **Expected**: `experiments/phase1_E1-E5_E4a/results/E4a/E4a_verification.json`; the
  telescoping-cost theorem verified to **1e-16** (this is the one Phase-1
  stage with a **deterministic-exact** tolerance class — pure arithmetic,
  no model, no data, no GPU jitter possible).
- **Resume**: none; also the cheapest stage by far (seconds).
- **Legitimate deviation**: none at the 1e-16 level — any failure here is
  a real numerical/implementation bug, not measurement noise.

### Phase 1: E5 — joint allocator + G1 kill test

- **Command**: Colab, `experiments/phase1_E1-E5_E4a/E5_allocator_G1.ipynb`, Run all. Does not
  modify E1-E4; reuses their already-verified machinery + measured damage
  tables.
- **Secrets/env**: `HF_TOKEN` (for the 50k audit pass).
- **Expected**: `experiments/phase1_E1-E5_E4a/results/E5/`; joint allocator plan == best
  staged/sequential plan at all 6 budgets (the kill test), exact tie
  (`delta_top1=0.0`, zero-width CI) as the necessary consequence. This
  specific claim (plan identity) is **deterministic-exact** given the
  same damage tables; the underlying 50k accuracy MEASUREMENTS feeding
  those tables are stochastic-band. Verify with `scripts/verify_g1_claims.py`
  (Tier 1) after this run.
- **Real measured runtime**: the solver itself is near-instant
  (`wall_clock_seconds.joint_solver=1.71s`, `sequential_solver=0.19s`) —
  the notebook's OWN total runtime (including the 50k audit pass, not
  separately recorded) is materially longer; budget similarly to E3 as a
  starting estimate.
- **Resume**: none.
- **Legitimate deviation**: none on the plan-identity claim itself (it's
  deterministic given the tables) — a different plan selected at any of
  the 6 budgets is a real regression, report it, do not attribute it to
  jitter.

### P2.1 — TANDEM 4-model fleet

- **Command**: Colab, `experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb`. Run **once per
  model**, editing `CONFIG["MODEL_NAME"]` in Section 2, Runtime → Run all.
  Recommended order: `deit_small_patch16_224` first (runs in ARCHIVED
  mode — S0/S1/S8 only, reusing Phase-1's already-validated E1-E5 tables —
  fastest, good smoke test), then `deit_tiny_patch16_224` (smallest fresh
  full pipeline), then `deit_base_patch16_224`, then `vit_base_patch16_224`.
- **Secrets/env**: `HF_TOKEN`. `TANDEM_ALLOW_ENV_DRIFT=1` if intentionally
  running on a different torch/timm/datasets version than the canonical
  summary (uncertified against these tables if set). `TANDEM_INTERACTIVE=1`
  to restore the original hard-halting review checkpoints (default:
  non-interactive, prints the same review block, continues automatically).
- **Expected**: writes `experiments/phase2_1_fleet/results/P2/<model>/` (`Dq_table.csv`,
  `Dm_table.csv`, `plans.csv`, `familyX.csv`, `P2_<model>_summary.json`,
  `calibration_indices.json`, figures). Tables/`D_pred`/`C_id`/`C_min`/gate
  verdicts: **deterministic-exact** given pinned calibration+environment
  (see `REPRODUCIBILITY.md`). Measured accuracies: **stochastic-band**,
  ±0.15pp. Verify with `scripts/verify_p2_claims.py` (Tier 1) after each
  model.
- **Real measured runtime** (S2 quant-table + S3 merge-table sweeps only —
  the largest single-notebook time sinks; total wall-clock including
  model load/calibration/S7 5k audit/S8 50k eval is larger and not
  separately recorded): DeiT-Tiny ~3.4 min, DeiT-Small ~6.3 min, DeiT-Base
  ~15.9 min, ViT-Base ~17.3 min. Budget at least 2-3x these figures for a
  full Run All per model as a starting estimate (the S8 50k evaluation —
  6 plans × 50,000 images each — is the largest unmeasured component;
  B4's own real 50k single-pass measurements this project has seen run
  ~7 minutes each on a T4, giving a rough sense of scale).
- **Resume**: the notebook's own checkpoint/reuse mechanism (S2/S3 skip
  already-computed rows unless `FORCE_FRESH_RUN=True`); a killed Colab
  session can be restarted and Run All re-issued — already-computed table
  rows are not recomputed.
- **Legitimate deviation**: ±0.15pp accuracy jitter, and the
  `phase1_regression` bridge fingerprint landing anywhere in the
  documented 46/48-in-band / median-1.0031 neighborhood (DeiT-S only).
  NOT legitimate: any gate verdict flipping, a `D_pred`/`C_id`/`C_min`
  mismatch against a fresh recomputation (these are deterministic), or a
  bridge fingerprint materially worse than 46/48 in-band.

### B1 (ToMe)

- **Command**:
  ```
  cd experiments/phase2_2_baselines/b1_tome && python -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python run.py --imagenet-dir $IMAGENET_DIR --host-tag <university_gpu|colab>
  ```
- **Secrets/env**: `IMAGENET_DIR` (local ImageFolder layout — NOT
  HF-streamed, unlike Phase 1/P2.1; see `deviations.md` #2). No HF token
  needed for B1 itself (no HF-hub checkpoint pull beyond timm's own
  cache).
- **Gate order** (registered, enforced by `run.py`'s own control flow —
  do not run these out of order): (1) startup checks (dual-host audit
  items present), (2) host-equivalence fingerprint (256-image smoke set;
  first host writes the canonical fingerprint, every later host/run
  compares — **HALTS**, no override, on a mismatch beyond ±1 image or a
  `requirements_hash` drift), (3) FP-parity gate (0.15pp vs P2.1's own
  S8), (4) Step A (`row_kind=reproduction`, gated ±0.30pp against ToMe's
  own published number), (5) Step B (`row_kind=comparison`, TANDEM's
  pinned checkpoint — coincides with Step A for DeiT-S, so not re-run).
- **FP-parity halts vs. reproduction-gate semantics — the distinction
  that matters**: FP-parity gate FAILURE means the HARNESS itself is
  untrustworthy (wrong preprocessing, wrong checkpoint, eval-loop bug) —
  it halts immediately, before any method-specific number is built on a
  broken foundation, and nothing is recorded for that run. Reproduction
  gate FAILURE means the harness is fine but the MEASURED number misses
  its PUBLISHED reference — the row is still written (`gate_status=fail`,
  `gate_diff_pct_points` populated) before the runner halts, so the
  miss is on record, not silently swallowed. `gate_status=no_published_reference`
  is a third, non-failing state (reported, never gated against a guessed
  number) and does not halt anything.
- **Expected**: 8 rows in `results.csv` (r0/r8/r13/r16 off-the-shelf ×
  2 row_kinds where they coincide, minus the r13 trained_with_tome
  addition — see `deviations.md` #16 for the exact count derivation).
  Cost-ruler worked example: `ToMe r=8, DeiT-S: BOPs_ours = 8.887083e+11`
  (deterministic-exact — pure arithmetic, verify independently via Tier
  1's cost-ruler check).
- **Real measured runtime this project has seen**: ~28.6 min total across
  the 4 timed rows (off-the-shelf configs; `wall_clock_s` in each row's
  own `notes` field).
- **Resume**: per-batch checkpoint (`common/checkpoint.py`), real
  interrupt/resume self-test passes (uninterrupted == interrupted-then-
  resumed, byte-for-byte); a killed run resumes from its last checkpointed
  batch, not from scratch.
- **Legitimate deviation**: reproduction-gate diff moving within the
  0.30pp band on a rerun (stochastic). NOT legitimate: an equivalence-
  fingerprint halt (investigate the host/environment before proceeding,
  never override), an FP-parity failure (harness bug, fix before
  continuing), or a reproduction-gate diff that FAILS where it previously
  passed (report, do not silently accept the new host's number).

### B2 (RepQ-ViT)

- **Command**:
  ```
  cd experiments/phase2_2_baselines/b2_repq_vit && python -m venv .venv && .venv/bin/pip install -r requirements.txt
  cd .. && python -m venv .venv_orch && .venv_orch/bin/pip install -r requirements.txt
  .venv_orch/bin/python b2_repq_vit/run.py --imagenet-dir $IMAGENET_DIR \
    --python b2_repq_vit/.venv/bin/python --host-tag <university_gpu|colab>
  ```
- **Secrets/env**: `IMAGENET_DIR`.
- **Gate order**: same shape as B1 minus B1's training-setting split
  (RepQ-ViT is pure PTQ, every row is off-the-shelf by construction, no
  `trained_with_tome`-style alternate setting): startup checks →
  equivalence fingerprint → FP-parity → Step A → Step B (ViT-B does NOT
  coincide with Step A here — a real second pass runs with the checkpoint
  override active; DeiT-T/S/B do coincide).
- **Expected**: 16 rows (3 coinciding models × 2 configs × 2 row_kinds +
  ViT-B × 2 configs × 2 row_kinds, both real passes).
- **Real measured runtime**: not separately recorded in `results.csv`'s
  `notes` field for this baseline (unlike B1/B4) — budget comparably to
  B1 as a starting estimate (simpler design: no merge sweep, just
  calibration + a handful of forward passes per config).
- **Resume**: coarse, results.csv-row-level (`find_existing_row` against
  `(method,model,config,row_kind,checkpoint_tag)`) — no B2 does not have
  the per-batch checkpoint B1/B4 have (documented, already-audited
  deviation, `deviations.md` #14); a killed run's already-WRITTEN rows
  are not recomputed, but a row that was mid-flight when killed is not
  partially recorded.
- **Legitimate deviation**: same shape as B1.

### B3 (FQ-ViT)

- **Command**: same shape as B2, pointing at `b3_fq_vit/run.py`.
- **Secrets/env**: `IMAGENET_DIR`. **Known hardware risk (pre-existing,
  documented, not new)**: `torch==1.7.1`/`cudatoolkit=10.1` predates most
  current GPU driver stacks — the notebook's own "STOP AND CHECK" cell
  verifies `torch.cuda.is_available()` and refuses to proceed silently on
  CPU-only; if it fails here, that is an environment problem to solve
  before continuing, not a code defect to work around.
- **Gate order**: same shape as B2.
- **Expected**: 8 rows (3 coinciding models × 1 config × 2 row_kinds +
  ViT-B × 1 config × 2 row_kinds, both real).
- **Resume**: same coarse mechanism as B2.
- **Legitimate deviation**: same shape as B1/B2.

### B4 (sequential composites: ToMe → RepQ-ViT)

- **Command**:
  ```
  cd experiments/phase2_2_baselines/b4_sequential_composites && python -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python run.py --imagenet-dir $IMAGENET_DIR --host-tag <university_gpu|colab>
  ```
  (6 operating-point cells + 1 audit-point cell in the Colab notebook,
  `--only-budget-fraction X.XX` / `--only-audit-r 23` locally.)
- **Secrets/env**: `IMAGENET_DIR`.
- **Gate order**: startup checks → equivalence fingerprint → cost-ruler
  unit checks (printed, not a halting gate) → FP-parity → per-operating-
  point BEST-OF-BOTH-ORDERS (both `merge_first`/`quant_first` run and
  recorded as `row_kind=reproduction`; the max is written as
  `row_kind=comparison` — B4 has **no** reproduction gate at all,
  `gate_status` is hardcoded `not_applicable`, since no published
  reference exists for this composite; see `deviations.md` #19).
- **Expected**: 18 core rows (6 budgets × 3 rows each) + 2 audit rows
  (r=23, order-degenerate, `row_kind=reproduction` only, deliberately no
  comparison row) = 20 total. Structural finding (deterministic-exact,
  pure algebra on the BOPs formula):
  `linear fraction (W16A16@r0) = 92.1232%`, `attention fraction = 7.8768%`.
  Pareto saturation (from real 50k measurements, stochastic-band):
  staged frontier saturates at ~29.3% of `C_id` — this is B4's own
  *nominal* budget-fraction label (`run.py`'s config row), not the same
  quantity as the **28.78%** figure quoted elsewhere (README.md,
  `docs/RESULTS.md`, this file's own §116) for the same saturation point
  — that 28.78% is the *realized* BOPs fraction actually measured for
  that config. The two numbers describe the same real result at two
  different points in the pipeline (nominal target vs. measured
  outcome), not a contradiction.
- **Real measured runtime this project has seen**: ~100 min total across
  the 14 timed reproduction rows (note: 3 of the 6 operating points are
  order-degenerate — w16a16, `merge_first`/`quant_first` mathematically
  identical, ONE GPU pass records both row labels — so true GPU time is
  less than this raw sum suggests, roughly ~70 min of actual distinct
  passes).
- **Resume**: per-batch checkpoint, same mechanism as B1 (real
  interrupt/resume self-test passes).
- **Legitimate deviation**: measured top-1 moving within ordinary GPU
  jitter; the audit point's exact number moving slightly while staying
  well below the 41%-budget operating point's accuracy (the qualitative
  finding — r=22/r=23 both collapse — is what must hold, not the exact
  digit). NOT legitimate: the best-of-both-orders selection choosing a
  different order at a given operating point on a rerun without a real
  environment change, or a different Pareto-frontier/saturation
  structure.

### B5 (compare_pareto / b5_final_closure)

- **Command**: no GPU, no subprocess — orchestration environment only.
  ```
  cd experiments/phase2_2_baselines && python compare_pareto.py       # numbers only, per original spec
  python b5_final_closure.py    # the final closure pass: pooled fronts, flagged table, figure, JSON
  ```
- **Secrets/env**: none.
- **Expected**: see Tier 1 step 6 above — identical expected output,
  since this Tier-2 entry and the Tier-1 re-derivation run the exact same
  script. The only difference in a genuine Tier-2 context is that B1-B4
  were just freshly re-measured, so B5 is now summarizing NEW numbers
  rather than the committed ones — re-run `scripts/verify_baselines_claims.py`
  afterward to confirm the new `compare_summary.json` is internally
  consistent with the new `results.csv`.
- **Resume**: not applicable (single fast pass, re-run freely).
- **Legitimate deviation**: comparison-table deltas shifting within the
  stochastic band inherited from B1-B4's fresh measurements. NOT
  legitimate: the DeiT-S pooled-front SET changing, or the saturation
  point moving to a different config.

### [Optional] G1-HW (jetson/)

Real Jetson Orin Nano Super hardware, not a T4 — no comparable time
estimate on this project's reference GPU. See `jetson/README.md` in full
(Day-0 bring-up, harness usage, expected runtimes on an Orin Nano Super
specifically, thermal guard, resume behavior per stage) — it is written
to the same standard as this section and is not restated here.

### [Optional] G1-HW-2 (jetson-orinnx/)

Real Jetson Orin NX 8GB hardware on loan, not a T4. See `jetson-orinnx/README.md`
and `jetson-orinnx/FINDINGS.md` in full (Day-0 bring-up, power-mode masking audit,
achieved-clock sysfs gate, canonical Run 3 LUT, cross-device transfer experiment,
and energy instrument validation). The automated verifier is run host-side in Tier 1
via `(cd jetson-orinnx && python verify_orinnx_claims.py)`.

---

## Summary: tolerance classes used throughout this document

- **Deterministic-exact**: pure arithmetic on already-measured inputs
  (BOPs formulas, `D_pred`, plan selection given fixed damage tables, the
  Ward theorem). A mismatch is always a real bug, never noise.
- **Stochastic-band**: involves a fresh GPU forward pass (fp16/fp32
  non-associativity, cuDNN algorithm selection). A mismatch within the
  documented band (±0.15pp for FP-parity/P2.1 accuracy, ±0.30pp for the
  reproduction gate) is expected and not investigated further; outside
  the band is a real finding to report, not silently absorbed.
