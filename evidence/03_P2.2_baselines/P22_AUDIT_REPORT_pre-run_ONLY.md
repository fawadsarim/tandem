> **ARCHIVED — historical, pre-run snapshot.** Written before
> any real B1-B4 GPU run existed; every "FLAGGED for review" item below
> has since been run and ruled on. For current, authoritative status see
> `../../REPRODUCIBILITY.md` and `../../baselines/deviations.md`. Kept for
> process history only — do not cite numbers from this file.

# P2.2 Baseline Code Audit — B1 (ToMe), B2 (RepQ-ViT), B3 (FQ-ViT), B4 (sequential composites)

Static analysis + cheap dynamic self-tests only (no 50k evals — no GPU in this
environment). Every check below was either (a) actually executed (framework-free
Python, real repo greps, real JSON/CSV reads against the real canonical P2.1
artifacts) or (b) verified by direct re-read of the real source at the pinned
commit — never assumed. Scientific/design judgment calls are FLAGGED for
review, not resolved silently. Mechanical defects were fixed in place; the fix
is described under "What changed," and every fix was re-verified after the edit.

**Two real, previously-undetected bugs were found and fixed this pass**, both in
`compare_pareto.py` (B5) — see Cross-cutting fix #6. Everything else was either
already correct or a smaller, well-scoped gap.

---

## Cross-cutting fixes (shared infrastructure — benefits all 4 baselines)

1. **`common/bops_identity_check.py`** — `MODEL_DIMS` was missing
   `deit_tiny_patch16_224` (d=192) even though B2/B3 actively use it; its C_id
   was never unit-checked. Added. Also added
   `check_c_id_matches_canonical_summary()`, which reads the REAL
   `notebooks/results/P2/<model>/P2_<model>_summary.json` and asserts
   bit-for-bit equality against our computed C_id — previously the only check
   was two independent formula re-implementations cross-checked against each
   other (a shared misunderstanding in both would have passed silently). Ran
   and verified against all 4 real summary files:
   `3.134950e+11 / 1.162418e+12 / 4.466548e+12 / 4.466548e+12` — exact match.
2. **`common/checkpoint_pin.py`** — `PINNED_TAGS`' FP references
   (72.166/79.848/81.97/85.10) were hardcoded mirrored constants, never
   asserted against the live canonical summaries. Added
   `assert_fp_references_match_canonical_summaries()`, wired into the
   self-test. Verified: all 4 match exactly.
3. **`common/notebook_self_test.py`** — `REQUIRED_RESULT_COLUMNS` was missing
   `"setting"` (added to `results.csv`'s real schema in the prior B1-correction
   pass, but never synced into this list). Added. `check_cost_ruler` (item 6)
   now also runs both new canonical cross-checks above.
4. **`common/results_io.py`** — added a `pip_freeze_hash` column (item #4:
   "pip-freeze hash recorded into results" was previously only written to the
   smoke-fingerprint file, never to `results.csv` itself).
5. **All 4 runners** (`b1_tome/run.py`, `b2_repq_vit/run.py`,
   `b3_fq_vit/run.py`, `b4_sequential_composites/run.py`) — compute
   `pip_freeze_hash` once per invocation (reusing the existing
   `equivalence_gate._pip_freeze_hash` helper — B1/B4 via `sys.executable`,
   since they run IN-PROCESS inside their own pinned venv; B2/B3 via
   `args.python`, the target venv) and thread it into every `results.csv` row,
   the same pattern already used for host/gpu/torch/cuda.
6. **`baselines/compare_pareto.py` (B5) — two real, independent bugs, both
   fixed and re-verified end-to-end:**
   - **(a) No filter on `row_kind`/`setting`.** Every row in `results.csv` —
     including `row_kind="reproduction"` (Step A, the paper's own checkpoint —
     for ViT-B a genuinely DIFFERENT checkpoint identity than TANDEM's Pareto
     curve was measured on) and `setting="trained_with_tome"` (B1's r=13 gate
     against ToMe's own fine-tuned checkpoint) — was being joined straight
     into the comparison table. This is the exact class of setting-mismatch
     bug the B1 gate correction (this session, earlier) fixed at the gate
     level, recurring independently at the plotting/comparison level. Fixed:
     added `comparison_eligible_rows()`, restricting the join to
     `row_kind=="comparison"` with `setting` in `{"", "off_the_shelf"}`.
   - **(b) `P2_RESULTS_ROOT` pointed at a path that has never existed.** It was
     `<repo_root>/results/P2/`; the real canonical P2 artifacts live at
     `<repo_root>/notebooks/results/P2/` (verified directly — `ls results/P2`
     fails, `ls notebooks/results/P2` succeeds). `load_tandem_pareto()`'s own
     `.exists()` guard silently swallowed this every single time, so the join
     has **never** performed a real interpolation since this file was written
     — every row fell into the "no TANDEM P2 result found on disk" branch.
     Fixed the path. **Verified end-to-end**: a dry run with 4 synthetic rows
     (clearly labeled `TEST_SYNTHETIC`, real DeiT-S cost values) against the
     REAL `notebooks/results/P2/deit_small_patch16_224/plans.csv` correctly
     excluded the 2 ineligible rows and produced genuine interpolated
     TANDEM-vs-baseline deltas for the 2 eligible ones.
7. **`b1_tome/published.json`, `b2_repq_vit/published.json`,
   `b3_fq_vit/published.json`** — added `setting_verbatim_quote` fields with
   text actually fetched from each upstream repo's real README/paper at the
   pinned commit (not paraphrased), confirming each reference's setting
   matches what our harness runs. B4 correctly has no `published.json` (no
   published reference exists for either composite).

All of the above were re-verified after editing: `python -m
common.bops_identity_check`, `python -m common.verify_colab_thin_wrappers`,
`ast.parse` on every touched `.py` file, `json.load` on every touched `.json`
file, and the `compare_pareto.py` dry run — all green.

---

## B1 (ToMe)

| # | Item | Status | Note |
|---|------|--------|------|
| 1 | Thin wrappers, scanned | PASS | `verify_colab_thin_wrappers.py`: 0 violations |
| 2 | B4 imports not duplicates | N/A | (B4-specific item; B1 itself imports `tome` from the pinned clone, not a copy — 0 grep matches for merge internals outside `external/`) |
| 3 | No P2.1 writes; refs read/asserted from canonical | FIXED | C_id + FP-reference now cross-checked live against real summary JSON (cross-cutting #1, #2) |
| 4 | Pinning: commit+assert, ==-pinned reqs, pip-freeze hash recorded | FIXED | Commit pin/assert and ==-pinning already correct; pip-freeze hash was missing from `results.csv` — fixed (cross-cutting #4, #5) |
| 5 | TF32 disabled both flags; host/gpu/torch/cuda recorded; seeds pinned | PASS | `disable_tf32()` sets both flags; `resolved_host_info` records all 4; `seed_everything(SEED=0)` |
| 6 | Zero credential matches | PASS | Full-repo grep, 0 matches |
| 7 | published.json setting field, verbatim | FIXED | Added `setting_verbatim_quote` to every reference (r0_baseline, r13 both settings, r8, r16) — real quotes fetched from ToMe's README + paper appendix |
| 8 | Training-free parity rule enforced in code | FIXED | `results.csv` already had `setting`; `compare_pareto.py` now actually filters on it (cross-cutting #6a) |
| 9 | Binding gates (FP-parity, identity-config, gate ordering) | PASS | FP-parity (±0.15pp vs S8) AND r0_baseline (±0.3pp vs ToMe's own stated 79.8) both binding — the corrected design from this session's earlier B1 pass. Gate ordering (fingerprint→FP-parity→Step-A→Step-B) verified by direct re-read of `run.py`'s control flow |
| 10 | Two-step policy, checkpoint_tag, DeiT coincidence not double-run | PASS | DeiT-S coincides (Step A==B) — verified, one measurement two rows, no re-run |
| 11 | Manifest identity; calibration protocol; no-calib skips w/ note | PASS / FLAGGED | B1 has no calibration — `BASELINES_NEEDING_CALIBRATION` correctly excludes "tome", no fabricated index file. Manifest identity to P2.1's own S8 is INDIRECT only — see FLAG below (applies to all 4 baselines identically, not B1-specific) |
| 12 | Resume: hash-guarded, simulated-interrupt self-test passes | PASS | `evaluate_top1_resumable` + `notebook_self_test._resume_simulation_inprocess`: uninterrupted==interrupted-then-resumed verified, wrong-hash-starts-fresh verified |
| 13 | BOPs formula, dims derived, unit checks vs canonical C_id | FIXED | Now bit-for-bit verified against the real canonical summary (cross-cutting #1) |
| 14 | ToMe trajectory + cap, r=0→C_id, r=8 worked example = 8.887e11 | PASS | From this session's earlier correction: real `min(r,floor((N-1)/2))` cap implemented; verified r=8 DeiT-S gives `8.887083e+11`, matching the verified value |
| 16 | Unified schema, corrected row counts | FIXED | Schema now has `setting`+`pip_freeze_hash`; row count traced by hand against every `append_result` call site: 3 off-the-shelf r × 2 row_kinds (6) + r0_baseline (1) + r13 trained_with_tome (1) = **8**, matches `EXPECTED_ROWS` in the notebook |
| 18 | deviations.md entry, B1 as template | PASS | Item #31 (setting-mismatch correction) and #32 (BOPs cap) already present from the prior task this session |

**FLAGGED for review (B1):**
- Manifest byte-identity to P2.1's own S8 manifest is asserted **indirectly**
  (via the FP-parity 0.15pp gate), not as a literal hash diff — P2.1's own
  pipeline never exported a manifest hash to diff against, and touching
  `P2_tandem_pipeline.py` was ruled out of scope in an earlier session. Is the
  indirect proof sufficient, or should P2.1 export a manifest hash?
- The trained-checkpoint (`deit_S_r13.pth`) loading code assumes the standard
  `{"model": state_dict, ...}` DeiT training-script checkpoint shape — not
  independently confirmed (no GPU here to load the actual file). `strict=True`
  will fail loudly if wrong, not silently misload — but this is the one
  genuinely untested code path in B1.

**B1 GO/NO-GO: GO.** All binding gates correctly scoped, cost ruler verified
bit-for-bit, mechanical gaps closed. The trained-checkpoint cell is the only
truly new, GPU-unverified code path — treat its first run as a validation
pass for the checkpoint-loading assumption specifically, not the whole
baseline.

---

## B2 (RepQ-ViT)

| # | Item | Status | Note |
|---|------|--------|------|
| 1 | Thin wrappers, scanned | PASS | 0 violations |
| 2 | No duplicated upstream logic | PASS | 0 grep matches for RepQ/MatMul internals outside `external/` |
| 3 | No P2.1 writes; refs asserted from canonical | FIXED | Same cross-cutting fix (C_id + FP-ref now verified) — B2 is torch-free so never wrote to P2.1 anyway |
| 4 | Pinning + pip-freeze hash recorded | FIXED | Commit pin/assert + ==-pinning already correct (verified: `torch==1.12.1`/`timm==0.4.12`, matches README); pip-freeze hash now wired into every row |
| 5 | TF32 disabled; host/gpu/torch/cuda recorded; seeds pinned | PASS | `subprocess_env_with_tf32_disabled` sets both flags via injected `sitecustomize.py`; `get_host_info` subprocess records all 4; `--seed 0` passed explicitly (their own default, verified from source) |
| 6 | Zero credential matches | PASS | — |
| 7 | published.json setting field, verbatim | FIXED | Added `setting_verbatim_quote`: repo title verbatim-confirms "Post-Training Quantization"; classification/README's own results-table caption quoted |
| 8 | Training-free parity rule enforced | PASS | RepQ-ViT is pure PTQ — every row is legitimately off-the-shelf/training-free by construction, `setting` column correctly `""` (no per-config setting variation exists for B2, unlike B1) |
| 9 | Binding gates (FP-parity, identity-config, gate ordering) | PASS / **FLAGGED** | FP-parity (±0.15pp) is implemented and binding for every model+step. **No identity-config gate exists for B2** — `original_top1_pct` (e.g. 79.85 for DeiT-S) is stored in `published.json` but never gated against, documentation-only. This is the exact same shape of gap B1 had before this session's correction added `r0_baseline`. See FLAG below. Gate ordering (FP-parity→Step-A→Step-B) verified by direct re-read of `run.py`. |
| 10 | Two-step policy, checkpoint_tag, coincidence not double-run | PASS | DeiT-T/S/B coincide (verified — reused, not re-run); ViT-B does NOT coincide — verified `run.py` performs a REAL second pass with the checkpoint override active, not a relabeling |
| 11 | Manifest identity; calibration protocol; disjoint train/val | PASS / see B1's manifest FLAG (applies identically) | Calibration = 1 batch of 32 (their own default, verified from source); indices reconstructed via `common/calib_capture.py` (provably-exact `RandomSampler` replay) and recorded per-config |
| 12 | Resume (coarse, since no per-batch checkpoint exists for B2) | PASS-by-design | Documented, already-audited deviation (`deviations.md` #14) — per-(model,config,row_kind,checkpoint) resume via `find_existing_row`, self-tested via `check_resume_csv_mechanism` |
| 13 | BOPs formula + unit checks vs canonical C_id (now incl. deit_tiny) | FIXED | B2 is the baseline that actually USES all 4 model shapes — the missing `deit_tiny` in `MODEL_DIMS` (cross-cutting #1) was found because of B2/B3's own `MODELS` dict |
| 16 | Unified schema, corrected row count | PASS | Traced by hand against every `append_result` call site: 3 coinciding models × (2 configs × 2 row_kinds = 4) + ViT-B × (2 configs × 2 row_kinds, both REAL passes = 4) = **16**, matches `EXPECTED_ROWS` |
| 18 | deviations.md entries | PASS | #5 (no W8A8 reference), #24 (prior DeiT-Base OOM) already present |

**FLAGGED for review (B2):**
- **No identity-config binding gate for B2** (mirrors the gap B1 had before
  this session's r0_baseline addition). `published.json`'s `original_top1_pct`
  values (72.21/79.85/81.80/84.54) sit unused. Should B2 gain an equivalent
  binding gate (plain-model FP-parity measurement vs RepQ-ViT's own stated
  `original_top1_pct`, ±0.3pp), the same shape as B1's `r0_baseline`? This is
  a real design decision (a new gate that could halt runs more often) — not
  implemented here, since it changes behavior rather than fixing a defect.
- Manifest indirect-identity FLAG (see B1) applies identically.

**B2 GO/NO-GO: GO.** Correct on every item this audit can mechanically fix.
Has never been run to completion on GPU in this project (a prior DeiT-Base OOM
is already documented) — the identity-config gate question above is a
judgment call for review, not a blocker.

---

## B3 (FQ-ViT)

| # | Item | Status | Note |
|---|------|--------|------|
| 1 | Thin wrappers, scanned | PASS | 0 violations |
| 2 | No duplicated upstream logic | PASS | — |
| 3 | No P2.1 writes; refs asserted from canonical | FIXED | Same as B2 |
| 4 | Pinning + pip-freeze hash recorded | FIXED | `torch==1.7.1`/`torchvision==0.8.2` verified against README's exact `conda install` line; pip-freeze hash now wired in |
| 5 | TF32 disabled; host/gpu/torch/cuda recorded; seeds pinned | PASS | Same mechanism as B2; FQ-ViT's own `seed()` already sets `CUBLAS_WORKSPACE_CONFIG` (verified from source), `subprocess_env_with_tf32_disabled` sets the same value if absent — no-op, not a conflict |
| 6 | Zero credential matches | PASS | — |
| 7 | published.json setting field, verbatim | FIXED | Added `setting_verbatim_quote`: repo title verbatim-confirms "Post-Training Quantization"; README's own protocol sentence quoted ("We compare our methods with several post-training quantization strategies...") |
| 8 | Training-free parity rule enforced | PASS | Pure PTQ, same reasoning as B2 |
| 9 | Binding gates | PASS / **FLAGGED** | Same finding as B2 — FP-parity binding, no identity-config gate exists. Same FLAG applies. |
| 10 | Two-step policy | PASS | DeiT-T/S/B coincide (their loader uses the byte-identical `dl.fbaipublicfiles.com/deit/...` URL, verified from source); ViT-B does not — real second pass verified |
| 11 | Manifest identity; calibration; disjoint train/val | PASS / see B1's FLAG | Calibration = 100×10=1,000 images (their own default); indices reconstructed and recorded, same mechanism as B2 |
| 12 | Resume (coarse) | PASS-by-design | Same as B2 |
| 13 | BOPs formula + unit checks vs canonical C_id | FIXED | Same as B2 |
| 16 | Unified schema, corrected row count | PASS | Traced: 3 coinciding models × (1 config × 2 row_kinds = 2) + ViT-B × (1 config × 2 row_kinds, both real = 2) = **8**, matches `EXPECTED_ROWS` |
| 18 | deviations.md entries | PASS | #9 ("Ours" flags, not the MinMax competitor row) already present |

**FLAGGED for review (B3):**
- Same identity-config-gate design question as B2.
- Manifest indirect-identity FLAG (see B1) applies identically.
- **Pre-existing, already-documented hardware risk, not a new finding**:
  `torch==1.7.1`/`cudatoolkit=10.1` predates most current GPU driver stacks.
  Both `requirements.txt` and the Colab notebook's own "STOP AND CHECK" cell
  already handle this explicitly (checks `torch.cuda.is_available()`, refuses
  to proceed silently on CPU-only). Not a code defect — flagging only so the
  reader knows this is a live, hardware-dependent risk on B3's real run, same
  severity `deviations.md` #1 already assigns it.

**B3 GO/NO-GO: GO**, with the same caveats as B2 plus the pre-existing
hardware-compatibility risk (already documented, not new, not a code defect).

---

## B4 (sequential composites: ToMe → RepQ-ViT)

| # | Item | Status | Note |
|---|------|--------|------|
| 1 | Thin wrappers, scanned | PASS | 0 violations |
| 2 | B4 imports B1's/B2's machinery, no duplicates | PASS | Grep-verified 0 matches for `bipartite_soft_matching`/`ToMeAttention`/`apply_patch`/`MatMul`/`quant_model`/`set_quant_state` outside `external/` — B4 genuinely imports `tome` and `quant.quant_model` from the pinned clones |
| 3 | No P2.1 writes; refs asserted from canonical | FIXED | Same cross-cutting fix |
| 4 | Pinning + pip-freeze hash recorded | FIXED | Both ToMe AND RepQ-ViT commits pinned+asserted; pip-freeze hash now wired in |
| 5 | TF32 disabled; host/gpu/torch/cuda recorded; seeds pinned | PASS | In-process, same mechanism as B1 |
| 6 | Zero credential matches | PASS | — |
| 7 | published.json | N/A (correct) | No published reference exists for either composite (neither upstream paper reports this combination) — no file, correctly |
| 8 | Training-free parity rule | PASS | Both composites are training-free (off-the-shelf checkpoint, post-hoc merge+quant); `setting` correctly `""` |
| 9 | Binding gates | PASS | FP-parity (plain DeiT-S) binding. No identity-config gate — correctly N/A, B4 has no published original-accuracy reference of its own to gate against (unlike B1/B2/B3, which measure a checkpoint a paper actually reports on) |
| 10 | Two-step policy | PASS | DeiT-S coincides — verified |
| 11 | Manifest identity; calibration; disjoint train/val | PASS / see B1's FLAG | B4 calibrates via RepQ's OWN `build_dataset` train_loader directly, in-process — NOT the `calib_capture.py` reconstruction path (correctly N/A per B4's own module docstring: "the index-reconstruction mechanism... doesn't apply here") |
| 12 | Resume: hash-guarded, simulated-interrupt self-test | PASS | Same mechanism as B1 — real interrupt/resume self-test verified |
| 13 | BOPs formula + unit checks | FIXED | Same cross-cutting fix; B4's own N_l trajectory comes from the ToMe stage only (verified: `tome_schedule_n_l(r, ...)`, cross-checked against a live forward-hook measurement on the ALREADY-QUANTIZED composite model — the docstring's own reasoning, "quantization changes VALUES, never token COUNTS," verified true by inspection of `measure_token_trajectory`, which hooks token *counts*, not values) |
| 14 | N/A (ToMe-specific worked example) | N/A | Covered under B1 |
| 15 | **Both merge-then-quant AND quant-then-merge orders implemented** | **FLAGGED — not implemented** | See below |
| 16 | Unified schema, corrected row count | PASS | Traced: 2 composites × 2 row_kinds (written together, DeiT-S coincides) = **4**, matches `EXPECTED_ROWS` |
| 18 | deviations.md entries | PASS | #10-11 (Attention.forward conflict resolution, attention left unquantized) already present, matches `run.py`'s own module docstring exactly |

**FLAGGED for review (B4) — the one substantive gap found this audit:**

- **Item 15 requires BOTH merge-then-quant and quant-then-merge orders,
  taking the sequential baseline as the best of the two. B4 currently
  implements ONLY merge-then-quant (ToMe first, RepQ second).** This is not
  an oversight — `run.py`'s own module docstring explains the reverse order
  is **structurally blocked** under the CURRENT integration strategy: both
  repos monkey-patch `Attention.forward` for incompatible purposes (ToMe
  reassigns `__class__` to return a tuple `(x, metric)`; RepQ's `build_model`
  installs `MatMul` submodules and an instance-level `forward` override
  returning a single tensor), and whichever patch is applied second wins,
  breaking the other's mechanism. The current resolution — skip RepQ's
  `build_model()` entirely, apply only ToMe's patch, let `quant_model()` wrap
  whatever Linear layers it finds — is compatible specifically *because* it
  quantizes AFTER merging. A quant-then-merge order would need a genuinely
  different integration (e.g., manually installing RepQ's `MatMul` wrapping
  onto ToMe's own attention class) — new, untested, GPU-unverified design
  work, not a mechanical fix.
- **Open question**: is merge-then-quant-only acceptable as B4's reported
  "sequential baseline" number (current design), or does the thesis require
  both orders (with the harder integration problem above solved first)? If
  both are required, this should be scoped as its own design/implementation
  task, not folded into a mechanical-fix pass.
- Manifest indirect-identity FLAG (see B1) applies identically.

**B4 GO/NO-GO: CONDITIONAL GO.** `run.py`'s own docstring already labels this
"the HIGHEST-RISK RUNNER IN THIS PROJECT" and states it has never been
executed — that remains true (no GPU here). Recommend proceeding with a real
run to validate the current merge-then-quant order produces non-degenerate
accuracy (exactly what the docstring itself recommends), but do **not** treat
the resulting number as a settled "sequential baseline" for thesis claims
until item 15 is ruled on.

---

## Item 17 — comparison-ready join (B5, cross-cutting)

Already covered under Cross-cutting fix #6: two real bugs found and fixed,
verified end-to-end with a dry run against real P2.1 data.

One naming note for the reader, not a defect: the audit spec names
`cross_model_comparison.csv` (a coarse, 4-row, per-model summary table with
only two budget-level reference points) as the join target.
`compare_pareto.py` instead joins against each model's full `plans.csv` (the
complete measured Pareto curve, enabling genuine linear interpolation at the
baseline's exact matched cost, not a coarse budget-bucket match) — this
predates this audit and matches the file's own module docstring reasoning. I
judge this the better design and left it as-is, but flagging the naming
discrepancy in case the coarser file is specifically wanted
instead (e.g. for a simpler sanity-check plot).

---

## Recommended run order

**B1 gates → B2 → B3 → B4, DeiT-S first throughout**, per the spec's own
ordering:

1. **B1 first** — cheapest, most mature at this point (already through one
   full real-Colab debugging cycle this session; its binding-gate design is
   now the template `deviations.md` #31 documents for the other baselines).
   Run the notebook's cells in order: self-test → r=8 → r=13 off-the-shelf →
   r=16 → r=13 trained_with_tome → completeness check. The trained-checkpoint
   cell is the one genuinely new code path — watch its output closely for the
   `strict=True` load succeeding.
2. **B2 next** — single-repo PTQ, no cross-repo integration risk. DeiT-Tiny →
   DeiT-Small → DeiT-Base → ViT-B, one Colab cell per model (session-limit
   hardened, see `deviations.md` #24's prior OOM).
3. **B3 next** — same shape as B2, plus the pre-existing torch==1.7.1/CUDA
   compatibility risk (already gated by the notebook's own STOP-AND-CHECK
   cell). Same model order.
4. **B4 last** — depends conceptually on B1 (ToMe) and B2 (RepQ-ViT quant
   mechanics) both already being validated; it is the composite, highest-risk
   runner. Run `tome_r13_then_repq_w8a8` first (smaller reduction, likely
   higher/safer accuracy), then `tome_r16_then_repq_w6a6`. Treat the FIRST
   number either composite produces as a validation pass per its own
   docstring, not yet a trusted thesis number, pending the item-15 ruling.

After all four: run `compare_pareto.py` (now fixed) to produce
`pareto_comparison.csv` and confirm the join actually finds real TANDEM
Pareto points (it will now print an "Excluded N/M rows" line if the filter
drops anything — expected for any `row_kind=reproduction` or
`setting=trained_with_tome` row).

---

## Paste-back lines expected

The four below require a real GPU run (none possible from this environment) —
shown as the exact format each will print, per the real, unmodified source
(`common/repro_gate.format_gate_banner`, `common/fp_parity_gate.check_fp_parity`,
`common/equivalence_gate.run_gate`), so they're recognizable when they land:

**Equivalence fingerprint** (`equivalence_gate.py`, one line per baseline):
```
HOST-EQUIVALENCE GATE: ALL CHECKED BASELINES PASSED.
tome/deit_small_patch16_224/r13: <correct_count>/<n_images> correct on the 256-image smoke set
```

**FP-parity ×4** (`common/fp_parity_gate.check_fp_parity`, one per model,
tolerance 0.15pp, reference = P2.1's own S8 measurement):
```
[FP-parity / deit_tiny_patch16_224 / <harness>] harness=<X.XXXX>%  P2.1_S8=72.1660%  diff=<d>pp (tol=0.15pp) -> PASS
[FP-parity / deit_small_patch16_224 / <harness>] harness=<X.XXXX>%  P2.1_S8=79.8480%  diff=<d>pp (tol=0.15pp) -> PASS
[FP-parity / deit_base_patch16_224 / <harness>] harness=<X.XXXX>%  P2.1_S8=81.9700%  diff=<d>pp (tol=0.15pp) -> PASS
[FP-parity / vit_base_patch16_224 / <harness>] harness=<X.XXXX>%  P2.1_S8=85.1000%  diff=<d>pp (tol=0.15pp) -> PASS
```

**Each baseline's Step-A gate line(s), with citation**:
```
[tome / deit_small_patch16_224 / r0]   measured=<X.XX>%  published=79.80%  diff=<d>pp (tol=0.3pp) -> PASS   (source: ToMe README 'original acc' column)
[tome / deit_small_patch16_224 / r13]  measured=<X.XX>%  published=79.36%  diff=<d>pp (tol=0.3pp) -> PASS   (source: ToMe README DeiT table, trained_with_tome setting, checkpoint=deit_S_r13.pth)
[repq_vit / deit_small_patch16_224 / w6a6] measured=<X.XX>%  published=78.90%  diff=<d>pp (tol=0.3pp) -> PASS  (source: RepQ-ViT classification/README.md)
[fq_vit / deit_small_patch16_224 / w8a8]   measured=<X.XX>%  published=79.17%  diff=<d>pp (tol=0.3pp) -> PASS  (source: FQ-ViT README.md, 'Ours' row)
```

**The two cost-ruler worked examples** (these ARE real — computed and
verified this session, framework-free, no GPU needed):
```
Identity-config C_id (per model, matches canonical notebooks/results/P2/<model>/*_summary.json exactly):
  deit_tiny_patch16_224:  3.134950e+11
  deit_small_patch16_224: 1.162418e+12
  deit_base_patch16_224:  4.466548e+12
  vit_base_patch16_224:   4.466548e+12

ToMe r=8, DeiT-S: BOPs_ours = 8.887083e+11  (verified expected: 8.887e11 -- MATCH)
```
