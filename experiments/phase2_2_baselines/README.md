# TANDEM Phase-2 baselines (P2.2)

**What this is**: P2.1 (`experiments/phase2_1_fleet/`) already established
TANDEM's own joint allocator's accuracy-vs-BOPs Pareto curve, per
architecture. P2.2 (this directory) exists to answer "compared to what?"
-- it reproduces four **published external compression methods**, on
TANDEM's own models/protocol, so their accuracy-vs-BOPs points can be
placed on the same matched-cost axis as TANDEM's own curve:

- **B1 -- ToMe** (`b1_tome/`): Token Merging (Bolya et al.), a published
  training-free token-reduction method (no quantization).
- **B2 -- RepQ-ViT** (`b2_repq_vit/`): a published post-training
  quantization method for ViTs (no token reduction).
- **B3 -- FQ-ViT** (`b3_fq_vit/`): a second published post-training
  quantization method for ViTs, reproduced independently of B2 as a
  second quantization reference point.
- **B4 -- the staged composite** (`b4_sequential_composites/`): no
  published paper of its own -- ToMe (token merging) applied first, then
  RepQ-ViT (quantization) applied on top, as the strongest *sequential*
  (do-one-axis-then-the-other) baseline available, run on TANDEM's own
  protocol. Not to be confused with TANDEM's own bits-vs-tokens "Staged /
  BEST-SEQUENTIAL" comparison in Phase 1/G1-HW -- same English word,
  unrelated baseline; see `docs/GLOSSARY.md`.
- **B5** (`compare_pareto.py` + `b5_final_closure.py`): no new
  measurement -- pools B1-B4's own results against TANDEM's P2.1 Pareto
  curve into one matched-cost comparison table, figure, and JSON (see
  "Running a baseline" below for the real final artifacts this produces).

Runs on the university 24GB GPU as the primary path; Colab notebooks
(`*_colab.ipynb`, one per baseline) are thin wrappers around the exact
same code for when university GPU access is unavailable -- see the
pre-run audit checklists in `docs/archive/baselines_AUDIT_REPORT_pre-run.md`
and `docs/archive/P22_AUDIT_REPORT_pre-run.md` for the single-source-of-truth
design and the host-equivalence gate that keeps the two hosts numerically
comparable (a post-run `AUDIT_REPORT.md` was planned but never produced --
see `evidence/00_MISSING.md` item 2; these pre-run documents are the real
audit trail that exists). Purpose: reproduce published external baselines
on our models/protocol so TANDEM's own Pareto curve (from
`experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb`) can be compared at
matched cost, on TANDEM's own BOPs axis (see `docs/GLOSSARY.md` for BOPs).

## Fairness rules (binding on every runner in this directory)

1. **Run each method at its documented settings only.** No hyperparameter
   tuning, no cherry-picking a better config than the one the upstream repo
   publishes. Every deliberate choice (which flag combination, which
   calibration size) is the one THAT REPO documents as its own default or
   headline result -- traced to source in `docs/deviations.md`, not assumed.
2. **Record repo commit, environment, and any unavoidable deviation.**
   `commits.md` pins exact SHAs; each baseline's `requirements.txt` pins its
   environment; `docs/deviations.md` records every place a clean reproduction
   wasn't fully achievable and why.
3. **Every number carries a reproduction gate.** Every measured top-1 is
   checked against a published reference at ±0.3%, where one exists. Where
   none was located (documented explicitly in `docs/deviations.md`, never
   silently skipped), the row is still written to `results.csv` with
   `gate_status=no_published_reference` -- reported, not asserted. **A gate
   FAILURE (not "no_published_reference") HALTS the runner** -- a measured
   number that misses its published reference is never silently recorded
   as if nothing were wrong. **Documented exception**: B3 (FQ-ViT)'s 4
   W8A8 reproduction rows (one per model) DO read `gate_status=fail` in
   `results.csv` and were explicitly, deliberately **not** halted -- root
   cause traced (`docs/deviations.md` #34-35) to FQ-ViT's own `--lis` flag
   forcing 4-bit attention regardless of the requested bit-width, so these
   numbers actually match the paper's unpublished "8/8/4" configuration,
   not the "8/8/8" row they were gated against. An explicit, documented ruling
   ("record-not-halt" for these 4 specific rows) is the one documented
   exception to this rule -- every other `gate_status=fail` in this
   project's history halts as stated.
4. **Hosts must be numerically equivalent.** The university GPU and Colab
   run the identical code (single source of truth -- see
   `docs/archive/baselines_AUDIT_REPORT_pre-run.md`)
   under the identical pinned environment and identical determinism
   settings (TF32 disabled, fixed seeds). The host-equivalence gate
   (`common/equivalence_gate.py`) verifies this empirically before any
   full run.
5. **Checkpoint identity is two-step, not assumed.** A method's own
   reproduction gate (Step A) runs on the checkpoint ITS paper actually
   used; a separate comparison row (Step B) runs the same method settings
   on TANDEM's PINNED checkpoint -- these coincide for every DeiT model
   but NOT for ViT-B (RepQ-ViT/FQ-ViT's own `vit_base` resolves to a
   different, lower-accuracy augreg checkpoint than TANDEM's pinned tag --
   see `common/checkpoint_pin.py` and `docs/deviations.md` #16). Every
   `results.csv` row records which (`checkpoint_tag`, `row_kind`).
6. **A cheap FP-parity gate runs before any comparison-checkpoint number
   is trusted.** The plain (unquantized) model, through that baseline's
   own harness, must land within 0.15pp of P2.1's own S8 measurement on
   the full 50k set (`common/fp_parity_gate.py`) -- catches a harness-level
   eval-loop/transform drift before any method-specific number is built on
   top of it. See `docs/deviations.md` #19. Each baseline's `run.py`
   accepts a `--skip-fp-parity` escape hatch for this specific gate (not
   the reproduction gate, which cannot be skipped): it prints a loud
   `**** WARNING: FP-PARITY GATE SKIPPED ****` banner and writes
   `skip_fp_parity=true` into that run's result row instead of running
   the check. Intended for a harness you've already separately verified,
   not routine use -- the gate stays ON by default.
7. **Every runner self-checks its own prerequisites at startup.**
   `common/startup_checks.py` asserts the dual-host audit items (pinning,
   TF32-off, calibration-index capture, equivalence fingerprint, unified
   schema, resume) are actually present in this checkout and aborts with a
   named list of what's missing, rather than failing confusingly deep into
   a run or silently proceeding without one of them.

## Layout

```
experiments/phase2_2_baselines/
  requirements.txt          orchestration env (common/, compare_pareto.py) -- no torch/timm, exact-pinned
  commits.md                pinned upstream commit SHAs
  results.csv                unified schema across all runners/hosts (see common/results_io.py)
  equivalence_fingerprint.json  canonical host-equivalence record (created on first gate run)
  compare_summary.json        B5's final rollup: pooled fronts, flagged comparison table (b5_final_closure.py)
  val_manifest_canonical.json  B5's canonical validation-set manifest (b5_final_closure.py)
  B5_all_models_pareto.png     B5's cross-model Pareto figure (b5_final_closure.py)
  reviewer_release/            curated copy of the 4 files above, regenerated on every b5_final_closure.py
                               run -- convenience bundle, not read by any script in this repo
  common/                    shared utilities, framework-free where possible
    bops.py                  TANDEM's own BOPs cost model, ported from P2 (model-parameterized)
    bops_identity_check.py    unit-checks bops.py against P2's own C_id for the identity config
    imagenet.py               local ImageFolder val loader (matches TANDEM's preprocessing exactly)
    eval_utils.py              batched top-1 eval loop (plain + per-batch-checkpointed)
    checkpoint.py               per-batch checkpoint/resume (B1/B4)
    determinism.py               TF32 disable, seeding, host/GPU recording, sitecustomize injection
    calib_capture.py              reconstructs RepQ-ViT's/FQ-ViT's exact calibration indices
    equivalence_gate.py            the cross-host smoke-eval equivalence check
    verify_colab_thin_wrappers.py   grep-assert: no eval logic duplicated into a notebook
    repo_utils.py               clone-and-pin-at-exact-commit helper (asserts the checkout at runtime)
    results_io.py                results.csv schema + append (incl. checkpoint_tag/row_kind)
    repro_gate.py                 the +-0.3% published-vs-measured check
    checkpoint_pin.py               two-step checkpoint-identity policy + pinned-tag override
    fp_parity_gate.py                plain-model-vs-P2.1-S8 harness-soundness check
    startup_checks.py                 dual-host audit item presence check, run at every runner's startup
  external/                  upstream repos cloned here by repo_utils (git-ignored, not committed)
  b1_tome/            requirements.txt + run.py + published.json + *_colab.ipynb
  b2_repq_vit/        requirements.txt + run.py + published.json + *_colab.ipynb
  b3_fq_vit/          requirements.txt + run.py + published.json + *_colab.ipynb
                       + diagnose_vitb.py (one-off, 2026-07 investigation of the ViT-B
                       Step-B FP-parity gap -- not part of the regular pipeline; its
                       findings are permanently wired into run.py's Step-B flow, see
                       docs/deviations.md #36)
  b4_sequential_composites/  requirements.txt + run.py + *_colab.ipynb (no published.json -- no reference exists)
  compare_pareto.py   B5 step 1: interpolates TANDEM's P2 Pareto at each baseline's BOPs_ours,
                       matched-cost delta -- numbers only, no conclusions, per original spec
  b5_final_closure.py  B5 step 2, the real final closure pass: pools compare_pareto.py's output
                       across all models into compare_summary.json, the flagged comparison table,
                       B5_all_models_pareto.png, and the reviewer_release/ bundle (see above)
  B5_compare_pareto_colab.ipynb
```

## Prerequisites (read before running anything)

- **A local ImageNet-1k ILSVRC2012 copy**, standard `ImageFolder` layout,
  at `$IMAGENET_DIR/val/<wnid>/*.JPEG` (and `$IMAGENET_DIR/train/...` for
  B2's 32-image calibration pull). This differs from Phase 1/P2's Colab
  HF-streaming approach -- see `docs/deviations.md` #2 for why. Set:
  ```bash
  export IMAGENET_DIR=/path/to/imagenet
  ```
- **Four separate Python environments**, one per baseline (`b1_tome/`,
  `b2_repq_vit/`, `b3_fq_vit/`, `b4_sequential_composites/`, each with its
  own `requirements.txt`) plus one lightweight orchestration environment
  (`experiments/phase2_2_baselines/requirements.txt`) for `compare_pareto.py`. These are
  **not interchangeable** -- see `docs/deviations.md` #1 for the concrete,
  verified version conflicts between the upstream repos (FQ-ViT pins
  `torch==1.7.1`; ToMe requires `torch>=1.12.1`; there is no torch version
  satisfying both).
- **TANDEM's own P2 results already on disk** at
  `experiments/phase2_1_fleet/results/P2/*/P2_*_summary.json` and
  `experiments/phase2_1_fleet/results/P2/*/plans.csv` (from `experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb`)
  -- `compare_pareto.py` reads these directly; it does not re-derive
  TANDEM's Pareto curve itself.

## Host-equivalence gate (run this FIRST, on every host)

Before any full 50k run, on EITHER the university GPU or Colab, run the
cheap host-equivalence gate: a smoke evaluation on a fixed 256-image
subset (indices recorded, deterministic) for one representative config
per baseline, using each baseline's own `run.py` completely unmodified.
The FIRST host to run it writes `experiments/phase2_2_baselines/equivalence_fingerprint.json`
as the canonical record; every subsequent run (any host) recomputes and
compares -- an exact match means that host will produce numbers directly
comparable to the canonical ones; a mismatch beyond +-1 image, or a
`requirements_hash` drift, HALTS with a full diagnostic before you waste
GPU time on a full run that won't be comparable. See
`common/equivalence_gate.py`'s module docstring for the full design
(TF32, calibration-index determinism, etc.) and
`docs/archive/baselines_AUDIT_REPORT_pre-run.md` for what this gate was
designed to catch (the pre-run audit checklist -- see the note at the top
of this file for why there is no post-run version).

```bash
# from the orchestration environment, pointing at each baseline's own venv python
.venv_orch/bin/python -m common.equivalence_gate \
  --imagenet-dir $IMAGENET_DIR --host-tag university_gpu \
  --tome-python b1_tome/.venv/bin/python \
  --repq-vit-python b2_repq_vit/.venv/bin/python \
  --fq-vit-python b3_fq_vit/.venv/bin/python \
  --sequential-composite-python b4_sequential_composites/.venv/bin/python
```

(Colab notebooks call the equivalent with `--host-tag colab` automatically
-- see each `*_colab.ipynb`.)

## Running a baseline

Each baseline is a single runner script, self-contained (clones/pins its
own repo on first run via `common.repo_utils`, writes directly to
`../results.csv`, records the resolved host/GPU/torch/CUDA in every row
via `--host-tag`). Two different execution patterns, because of how each
upstream repo is actually built (see `docs/deviations.md`):

- **B1 (ToMe) and B4 (sequential composites)** import the upstream repo(s)
  as libraries IN-PROCESS (`import tome`, and for B4 also RepQ-ViT's
  `quant`/`utils` modules) so they can hook into the live model (token
  trajectory measurement, mid-pipeline patching). These MUST run inside
  their own pinned venv directly:
  ```bash
  cd experiments/phase2_2_baselines/b1_tome && python -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python run.py --imagenet-dir $IMAGENET_DIR --host-tag university_gpu
  ```
- **B2 (RepQ-ViT) and B3 (FQ-ViT)** wrap the upstream repo's own
  standalone `test_quant.py` as a SUBPROCESS (neither exposes a composable
  library API) -- `run.py` itself has no torch/timm dependency, so it can
  run from the lightweight orchestration environment, but needs `--python`
  pointing at the target venv's interpreter for the actual subprocess call:
  ```bash
  cd experiments/phase2_2_baselines/b2_repq_vit && python -m venv .venv && .venv/bin/pip install -r requirements.txt
  cd .. && python -m venv .venv_orch && .venv_orch/bin/pip install -r requirements.txt
  .venv_orch/bin/python b2_repq_vit/run.py --imagenet-dir $IMAGENET_DIR \
    --python b2_repq_vit/.venv/bin/python --host-tag university_gpu
  ```

(repeat per baseline). Then, from the orchestration environment, run
**both** B5 steps in order -- `compare_pareto.py` alone does not produce
the shipped comparison artifacts:

```bash
cd experiments/phase2_2_baselines
.venv_orch/bin/python compare_pareto.py       # step 1: numbers only, no conclusions, per original spec
.venv_orch/bin/python b5_final_closure.py     # step 2: the real final closure pass (see below)
```

`compare_pareto.py` writes `pareto_comparison.csv` (an intermediate,
per-model file -- not itself committed to the repo, since
`b5_final_closure.py` consumes and regenerates it on every run).
`b5_final_closure.py` is the actual final step: it pools that output
across all 4 models into the artifacts this repo's root `README.md` and
`docs/RESULTS.md` treat as the real shipped B5 deliverable --
`compare_summary.json`, `B5_all_models_pareto.png`, and a curated copy of
both plus `results.csv` in `reviewer_release/` (see the Layout section
above).
