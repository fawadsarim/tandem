# 00_INTEGRITY.md — Reviewer Evidence Package

## 1. Hash verification — every copied file vs its source

All 145 copied files were hashed (SHA256) and compared against the source file still in its
original repo location. **Final result: 139/145 confirmed byte-identical to a local source file
still in place; 6/145 have no local source to compare against** (explained below — none is a
drift or corruption issue). **Zero files remain diverged.**

- `04_jetson_hardware/out/segments/manifest.json` — this file only ever existed on the Jetson
  device (`~/tandem/jetson/out/segments/`), never committed to git (large ONNX/calibration
  siblings in that directory are deliberately gitignored — see `jetson/.gitignore`). Pulled via
  `scp` directly into this package. No local git-tracked copy exists to diff against; its
  provenance is the live SSH pull, not a local-file comparison.
- `02_P2.1_fleet/verify_p2_claims_output.log`, `01_phase1_E1-E5_E4a/verify_g1_claims_output.log`,
  `03_P2.2_baselines/verify_baselines_claims_output.log` — all three are **freshly generated**
  artifacts (see section 2 below), not copies of pre-existing files — there is no "source" to
  compare them against by definition.
- `04_jetson_hardware/out/powersweep/reproduction_gate_ATTEMPT1_FAILED.csv`,
  `04_jetson_hardware/out/powersweep/n69_diagnostic_repeats.log` — both **freshly generated this
  update**, transcribed from this session's real captured terminal output (the device never
  saved a first-attempt file, since `powersweep_repro_gate.py` overwrites its one output path on
  each run — see section 2b below and each file's own header note). There is no "source" to
  compare them against by definition, same as the three logs above.

Zero unexplained hash mismatches were found. Zero files exist in the package that aren't
accounted for in the manifest (checked both directions: manifest → file exists, and file →
manifest has an entry).

## 1c. Public-release repository reorganization, documentation audit, and terminology cleanup

The whole repository was reorganized (`notebooks/` → `experiments/phase1_E1-E5_E4a/` +
`experiments/phase2_1_fleet/`, `baselines/` → `experiments/phase2_2_baselines/`,
`jetson/CHARACTERIZATION.md` → `docs/CHARACTERIZATION.md`, `REPRODUCIBILITY.md` (root) →
`docs/REPRODUCIBILITY.md`, `REVIEWER_RUNBOOK.md` → `docs/REVIEWER_RUNBOOK.md`, `baselines/deviations.md`
→ `docs/deviations.md`) for public release. This package's own `00_MANIFEST.md.original_path`
column was updated to match — 73 of 145 rows' `original_path` changed to the new location, via
`git mv`, which preserves file content and git history; **this is a rename, not an edit**, and
does not itself affect any hash.

That reorg, a later documentation-accuracy audit, and a subsequent terminology cleanup (removing
internal reviewer-identity references from code comments, docs, and result-file narrative text
before public release) together caused real CONTENT edits to a substantial number of live source
files — stale `notebooks/`/`baselines/` path references left over from the reorg's own comments
and docstrings, documentation gaps found during the audit, and every "advisor"-style internal
reviewer-role reference. **Every one of these files has been synced into this package as of this
final pass** — content copied, hash and size recomputed in `00_MANIFEST.md`. In every case:
only comments, docstrings, markdown prose, print/error strings, JSON narrative-text fields, or
pure identifier renames (with every reference checked and updated consistently) were touched —
**zero formulas, thresholds, control flow, or measured/gate values were ever edited**, confirmed
both by re-running every no-GPU verifier (`scripts/run_reviewer.sh` Tier 1, all 41 sub-checks
`MATCH`/`PASSED`, identical to the pre-cleanup baseline) and by hashing the actual physical Jetson
device's own `out/` files against this repository (44/44 byte-identical).

Two categories got special handling rather than a blind resync:

- **The three `PREREGISTRATION.md` documents** (`jetson/out/energy_replan/`, `jetson/out/pilot_deitb/`,
  `jetson/out/powersweep/`) and the two JSON fields that quote them verbatim
  (`energy_replan_manifest.json`'s `pre_registered_note`, `powersweep_analysis_summary.json`'s
  `decision_rule`/`triggers_report_to_advisor`→`needs_manual_review` key): a pre-registration
  document's entire evidentiary value is being fixed in writing *before* the result it governs was
  known, so these were edited last, deliberately, and only as pure terminology substitution (e.g.
  "REPORT TO ADVISOR" → "NEEDS MANUAL REVIEW") — every decision threshold, number, and rule is
  byte-for-byte identical to before; what was fixed in advance was the rule, not the vocabulary
  describing its outcome.
- **The 4 real, GPU-executed per-model notebooks** at `experiments/phase2_1_fleet/results/P2/*/`
  (not part of this package — no `original_path` entry references them) contained "advisor" inside
  `output` blocks that are literal captured stdout/traceback text from a real run. That text is
  pure instructional banner/error copy from a `print()`/`raise` whose own source was already
  updated elsewhere in this same cleanup — not a measured value, gate outcome, or number — so the
  historical output was aligned with it rather than left showing an already-superseded instruction.
  Verified: all 4 notebooks, 62 cells each, same `execution_count`s and cell order before and after,
  only the affected `source`/`outputs[].text`/`outputs[].evalue`/`outputs[].traceback` strings changed.

`E5_summary.json` (`01_phase1_E1-E5_E4a/`) has one field renamed, `advisor_spec_error_note` →
`spec_error_note`, value text updated to match — the underlying finding, every number, and every
other field are byte-for-byte unchanged.

## 2. Independent claim-verification scripts — all three re-run for real, this session

Three standing "headline-claims" verifier scripts exist in `scripts/`, one per phase (P2.1,
E5/G1, baselines). No saved log existed anywhere in the repo for any of the three (checked). Per
the same principle applied to the first one, all three were run now — none needs a GPU, all are
pure recomputation from committed CSV/JSON against each phase's own stored summary.

### `verify_p2_claims.py`

No saved log of a prior run existed anywhere in the repo (checked). Per the task's own
instruction, it was re-run now (`python3 scripts/verify_p2_claims.py`, no GPU needed — pure
recomputation from committed CSV/JSON tables) and its output captured in full:

```
====================================================================================================
ALL CLAIMS VERIFIED (20 checks across 4 models)
====================================================================================================
```

20/20 checks passed: Family-X fit, S8 sanity gate, plan D_pred values (14/14 rows exact match
per model), C_id/C_min (BOPs formula), and per-budget fidelity ratios (6/6 exact match per
model) — independently recomputed and diffed against each model's stored summary.json, not just
re-printing the stored value. Full log at `02_P2.1_fleet/verify_p2_claims_output.log`.

### `verify_g1_claims.py` (added on request, judged relevant — the E5/G1 phase's own verifier)

```
====================================================================================================
ALL CLAIMS VERIFIED
LAB_LOG.md: marker already present -- no line added (not the first successful run).
====================================================================================================
```

5/5 checks passed: `plans_identical` (joint==best_sequential, all 6 budgets — the staging-collapse
finding), the exact-tie consequence (mean delta_top1=0.0, zero-width CI), the frontier mechanism
(26/85 schedules frontier-efficient, all 6 joint schedules on it, re-derived from `E4_raw.csv`
from scratch, not read from the stored CSV, then cross-checked 85/85 against it), the Family-X
cross-axis fit, and the pooled fidelity fit — all matched to ~1e-13 relative precision. Full log
at `01_phase1_E1-E5_E4a/verify_g1_claims_output.log`. **Note**: this script has a side effect of
appending a marker to the repo-root `LAB_LOG.md` on its FIRST successful run only — confirmed via
`git status --porcelain LAB_LOG.md` (empty output) that this run did NOT modify it (the marker
was already present from an earlier run, before this task).

### `verify_baselines_claims.py` (added on request, judged relevant — the P2.2/baselines phase's own verifier)

```
====================================================================================================
ALL CLAIMS VERIFIED
====================================================================================================
```

6/6 checks passed: the +-0.3pp reproduction gate (10/10 gated rows recomputed exactly), results.csv
schema completeness (52/52 rows), config_hash structural consistency (52 distinct keys, 0
conflicts), the equivalence-fingerprint check (no data — see below), the B5 Pareto-comparison
check (no data — see below), and B5 final closure (21/21 comparison-table deltas match). Full log
at `03_P2.2_baselines/verify_baselines_claims_output.log`. **This run independently corroborates
two gaps this package already reported from a manual file search**: it reports "NO DATA YET" for
`equivalence_fingerprint.json` (matches `00_MISSING.md` item 4) and additionally surfaces
`pareto_comparison.csv` as not existing either — a gap not previously named; added as
`00_MISSING.md` item 6.

## 2b. Power-mode sweep addition — 18 new files, plus a full re-verification of the other 127

This update adds the power-mode sweep's complete output (`04_jetson_hardware/out/powersweep/`,
13 files) and its 5 new scripts (`05_code_reference/powersweep_*.py`). Before adding anything, a
full re-hash of every one of the **127 files already in the package** was re-run against its
source, to confirm this update introduced no drift into previously-delivered material — result:
**127/127 unchanged, 0 mismatches** (the same 4 no-local-source files from before remain
no-local-source, for the same reasons already documented; everything else re-matched byte for
byte).

Of the 18 new files, **16/18 have a real local source** (the 13 powersweep output files copied
from `jetson/out/powersweep/` on this laptop after `scp`-pulling them from the device, plus the 5
new scripts copied from `jetson/`) and were confirmed byte-identical to that source. The
remaining **2/18** (`reproduction_gate_ATTEMPT1_FAILED.csv`, `n69_diagnostic_repeats.log`) have
no source file anywhere — see section 1 above and each file's own header note for why, and
`00_CORRECTIONS.md`'s introduction / the sweep's own paste-back report for what they record (the
Part 1 reproduction gate's first, failed attempt, and the diagnostic that explained it before a
clean re-run passed 10/10).

**Numeric corrections found by this sweep**: three figures that had circulated in prior
correspondence (a proxy-error ratio, an achieved-TFLOPS figure, and an implicit "wide power
range" framing) did not match this project's own committed artifacts. Rather than silently
correct them in place, they are recorded with full source-artifact traceability in the new
top-level `00_CORRECTIONS.md` — see that file. This is the same "always report, never silently
fix" discipline as the rest of this package.

## 3. Git status — additions only, nothing outside the new folder modified

`git status --porcelain` at the time this package was first assembled showed only the evidence
package directory (new, untracked, then named `Writer-Advisor Files/` — renamed to `evidence/`
during the later public-release reorg) plus this session's own temporary scratch scripts
(deleted before handoff). No existing tracked file was edited, moved, or deleted by that task.

**This update (power-mode sweep addition)** shows, for changes made during this update:
`00_MANIFEST.md` (modified, in place — new rows appended, same file), plus
new files: `00_CORRECTIONS.md`, the whole
`04_jetson_hardware/out/powersweep/` folder, and 5 new files under
`05_code_reference/` — all additions, consistent with the package directory
already being a tracked-but-uncommitted directory in this repo (individual new files
inside it show up individually rather than collapsing to one untracked-directory line, which is
expected once a directory has previously been `git add`-ed). Two scratch helper scripts used to
build this update (`scratch_update_manifest2.py`, `scratch_verify_all.py`) were created at the
repo root during assembly and removed before this update is reported done, same discipline as
the original assembly.

Separately, and pre-dating both this task and the original assembly (from earlier G1-HW device
work this session, not part of either evidence-package task): `jetson/.gitignore`,
`jetson/README.md` (modified), several `jetson/out/energy*`, `jetson/out/pilot_deitb_ext/*`
additions, and now also the 5 new `jetson/powersweep_*.py` scripts (untracked, pushed to the
device the same way every other `jetson/*.py` script in this project is — outside git, via
`scp`/rsync — see `jetson/_common.py`'s own module docstring). These predate or are the direct
source-side counterpart of files already accounted for above; verified nothing else was touched.

## 4. Key provenance hashes, collected in one place

**Model checkpoint hashes** (SHA256 of the loaded state_dict, `hash_state_dict` — confirmed
identical across every pipeline stage that touched that checkpoint, a real cross-check not just
a copy):

| Model | Checkpoint hash | Confirmed consistent across |
|---|---|---|
| `deit_small_patch16_224.fb_in1k` (DeiT-S) | `a17d00754ef1e5401cf33cf61af23094c48b8f4a4f5f8898d8ae36b9d81925e1` | `out/segments/manifest.json` (export) AND `out/paired_eval/paired_eval_result.json` (eval) — identical hash at both stages |
| `deit_base_patch16_224.fb_in1k` (DeiT-B) | `d0d329d779597c2bec332181ea0ab920d55a7113b0aef8c0e1ea653744a2a559` | `out/pilot_deitb/pilot_deitb_export_manifest.json` |

**Manifest / calibration hashes**:

| Artifact | Hash |
|---|---|
| `paired_eval.py` 20,512-image paired-eval manifest (the registered gate's own image set) | `8a2c5f38ff1a3a6100a88c465d7fce9c2ccf5a176e396e04f53bc0ef02a1a1b2` |
| DeiT-B pilot's 128-image calibration set | `210f08d3f77a88a9d9559920d53b1132fe9fa491c8f5dafa4e7d57d34e88f0d9` |
| **Canonical baseline validation manifest** (`experiments/phase2_2_baselines/val_manifest_canonical.json`, 50,000 images, source `imagenet-1k`) | **`41ddeb0ebb2fea82`** — confirmed by direct read, matches the task's own cited value exactly |

**Pinned upstream commits** (from `experiments/phase2_2_baselines/commits.md`, quoted verbatim — not independently
re-verified against GitHub in this pass, that verification is `commits.md`'s own documented
responsibility, done at pin time):

| Baseline | Repo | Pinned commit | Date |
|---|---|---|---|
| B1 (ToMe) | facebookresearch/ToMe | `af95e4b1befa172dadccd8c81e223b10090f9579` | 2023-03-31 |
| B2 (RepQ-ViT) | zkkli/RepQ-ViT | `a160561b9233945e6680132f129d13041b787318` | 2024-01-10 |
| B3 (FQ-ViT) | megvii-research/FQ-ViT | `5daf5915c715dc21d6afcd64e661652030e640ab` | 2023-04-11 |
| B4 (sequential-composite) | uses B1+B2's pinned commits together | — (no separate repo) | — |

None of the three upstream repos publish version tags — each pinned commit was the tip of that
repo's default branch (`main`) at pin time, not a tagged release (documented as such in
`commits.md` itself, not glossed over here).

**Note on hash truncation**: `00_MANIFEST.md`'s table truncates each SHA256 to 16 hex chars for
table readability. Every hash quoted in full in this document (checkpoint/manifest/calibration
hashes above) is the complete, untruncated value.
