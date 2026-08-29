# Reproducibility — Baselines (B1-B5)

What reproduces, at what tolerance, and how to check it yourself. Same
purpose as `../../docs/REPRODUCIBILITY.md` (the project's consolidated pin
table and top-level index) — this file is the detailed, baseline-specific
narrative it points to.

**Status (P2.2 closed): all of B1-B5 have real data.** B1 (ToMe,
8 rows), B2 (RepQ-ViT, 16 rows), B3 (FQ-ViT, 8 rows), B4 (sequential
composites, 20 rows incl. the r=23 audit point) — 52 rows total in
`results.csv` — plus B5's closure (`compare_summary.json`,
`b5_final_closure.py`). `scripts/verify_baselines_claims.py` reports **ALL
CLAIMS VERIFIED** as of this writing (6 checks, including an independent
recomputation of every B5 comparison-table delta and the DeiT-S pooled
Pareto front). Re-run that verifier before presenting any table from this
directory.

**Not "uniformly clean" — one real, documented gate failure.** B3
(FQ-ViT)'s 4 W8A8 reproduction rows (one per model) read
`gate_status=fail` in `results.csv`, genuinely outside the ±0.3pp
reproduction tolerance below. This is not silently passed or hidden:
`docs/deviations.md` #34-35 traces the root cause to FQ-ViT's own `--lis`
flag forcing 4-bit attention regardless of the requested bit-width, so
these 4 measured numbers actually match the paper's unpublished "8/8/4"
configuration, not the "8/8/8" row they were gated against — an explicit
documented ruling records these 4 rows as "record-not-halt," the one
documented exception to this project's normal gate-failure-halts rule
(see `experiments/phase2_2_baselines/README.md`'s Fairness Rule #3).

## Verify before you present anything

```
python scripts/verify_baselines_claims.py
```

Read-only, no GPU, no subprocess, no re-run. Recomputes the reproduction
gate (`common/repro_gate.py`'s exact +-0.3pp rule) for every `results.csv`
row with a located published reference, checks schema completeness against
`common/results_io.py`'s `RESULTS_COLUMNS`, checks that no
(method, model, config, row_kind, checkpoint_tag) key ever carries two
different `config_hash` values, checks `equivalence_fingerprint.json` for
any halted host, and independently recomputes B5's matched-cost Pareto
deltas via linear interpolation of each model's own P2 `plans.csv`. Exits
nonzero on any real mismatch.

One thing this verifier already caught, before any real data existed:
`results.csv`'s committed header was stale (17 columns, missing
`checkpoint_tag`/`row_kind` — added to the schema after that file was last
touched). Fixed directly since the file held no data yet; flagged here
because it's exactly the kind of thing that would have silently corrupted
the very first real run's output (a 19-value row appended under a
17-column header).

## What reproduces exactly

**Every `results.csv` row's inputs are pinned via `config_hash`**
(`common/determinism.py`): the exact upstream commit SHA(s) (see
`commits.md`), the config dict, the seed (0 — matches RepQ-ViT's and
FQ-ViT's own `test_quant.py` default, pinned explicitly rather than
silently inherited), and the calibration source/indices. Two rows sharing
a `config_hash` are claiming to be the exact same measurement; the
verifier's check 3 catches it if that's ever violated.

**Calibration (B2/B3, PTQ methods)**: not cache-and-verify the way P2.1's
is — `common/calib_capture.py` *deterministically reconstructs* RepQ-ViT's
and FQ-ViT's own internal calibration image selection by replaying their
exact `seed()` -> `ImageFolder(train)` -> `RandomSampler` call sequence, with
a documented equivalence proof (see that module's docstring for why this
is provably exact, not just plausible). Same theoretical exposure as
P2.1's `datasets`-library-version risk: a future `torch` release changing
`RandomSampler`'s internals would silently reconstruct a different
sequence. Not currently pinned-and-asserted the way P2.1's
`calibration_indices.json` is — worth the same treatment if this ever
becomes a live concern, not done here since it wasn't asked for and the
existing mechanism has a real correctness proof behind it already.

**Determinism (all methods)**: TF32 disabled on every GPU op
(`common/determinism.disable_tf32` for in-process runners B1/B4;
`subprocess_env_with_tf32_disabled`'s injected `sitecustomize.py` for
subprocess runners B2/B3) — the #1 silent cross-host numeric difference
this project has hit (Ampere-class GPUs default TF32 on; Colab's T4 has no
TF32 path, so it's already a no-op there — same code, different hosts,
different logits, unless disabled everywhere explicitly).

**Environment**: `common/equivalence_gate.py` hashes the full `pip freeze`
output for each baseline's pinned venv, first-run-canonical, and **halts**
(no override) on any drift, printing a full pip-freeze diff. Unlike P2.1
there is no `TANDEM_ALLOW_ENV_DRIFT`-style escape hatch here, and that's
deliberate (confirmed, not an oversight) — a baseline comparison's
fairness depends on the SAME environment across methods and hosts in a
way P2.1's own pipeline doesn't; a silent or loudly-tolerated drift here
could change which method looks better, not just move a number within a
model's own gate band.

## Two gates, two different tolerances, two different questions

- **FP-parity gate** (`common/fp_parity_gate.py`, **0.15pp**): does this
  baseline's own unquantized/plain forward pass match P2.1's own S8
  measurement for the exact same checkpoint? Catches *harness* drift
  (wrong preprocessing, wrong checkpoint resolved, wrong eval loop) --
  answers "is our own measurement machinery trustworthy," independent of
  whether the METHOD reproduces its paper.
- **Reproduction gate** (`common/repro_gate.py`, **±0.3%**): does this
  baseline's measured top-1 match the number its own paper/repo published,
  for `row_kind=reproduction` configs where a published number was
  located? Answers "did we reproduce the paper," not "is our harness
  correct" -- the two are deliberately separate checks (`docs/deviations.md`
  item 19's B4 note explains why conflating them would be wrong for B4,
  which has no reproduction gate at all — its `gate_status` is hardcoded
  `not_applicable`). Each `run.py` also accepts a `--skip-fp-parity` flag
  that skips only the FP-parity gate specifically (prints a loud warning
  banner and writes `skip_fp_parity=true` into that row instead of
  running the check) — the reproduction gate below has no equivalent
  skip; it always runs and a failure is always recorded.
- Where no published number exists for an exact config, `gate_status` is
  `no_published_reference` — reported, never silently gated against a
  guessed number (`docs/deviations.md` records exactly which configs this
  applies to and why).

### Tolerance standardization: 0.30 percentage points, flat

The reproduction gate's own module docstring (`common/repro_gate.py`)
reads "+-0.3%", inherited verbatim from this project's original spec
wording — that phrasing is ambiguous between two different readings:

- **Flat/absolute** (what is actually implemented): `abs(measured_pct -
  published_pct) <= 0.3` — a straight percentage-POINT difference,
  regardless of the published value's own magnitude. `GATE_TOLERANCE_PCT
  = 0.3` in `common/repro_gate.py`'s `check_gate()` is this reading; every
  gate line this project has ever printed uses it.
- **Relative** (NOT implemented): `abs(measured_pct - published_pct) /
  published_pct <= 0.003` — 0.3% *of* the published number, which shrinks
  the effective bar for any published value below 100 (i.e. every config
  this project runs).

The two readings agree almost everywhere in this project's results, but
diverge for one real row: `repq_vit / deit_tiny_patch16_224 / w6a6`
(measured diff = 0.228pp against published 70.76%). Under the flat
reading it passes (0.228 <= 0.3). Under the relative reading it would
fail (0.228 / 70.76 = 0.322%, which exceeds 0.3%). This project has
always reported and gated it as a PASS (flat reading, as implemented and
printed since B1) — recorded here explicitly so the ambiguity in the
original spec text is resolved on paper, not just in code, and so this
one row's status is traceable to a stated, deliberate reading rather than
an unexamined default.

## Baseline-to-P2.1 eval-set identity (no literal manifest-hash diff, and why)

P2.1's own pipeline (`experiments/phase2_1_fleet/P2_tandem_pipeline.py`) exports no hash of
the 50k ImageNet-1k val manifest itself — verified via exhaustive grep of
the full pipeline file and every artifact under `experiments/phase2_1_fleet/results/P2/`:
state-dict hashes (`hash_state_dict`, weight-restoration verification),
`config_hash` (per-plan identity), and `calib_hash` (calibration-index
pinning) all exist; nothing hashes the val SET. There is therefore nothing
to read and assert byte-for-byte equality against for the
baseline-to-P2.1 link.

**P2.1 itself is deliberately NOT being modified to add one.** It is a
finalized, verifier-certified package (`scripts/verify_p2_claims.py`
reports 20/20 MATCH as of this writing) — adding a manifest export would
require re-running that entire pipeline for a change that alters no
number in any shipped table. That cost is not justified by a check whose
real job is already covered by two other mechanisms:

1. **Identical source and count.** Both P2.1 (`config.N_IMAGES_FINAL=50000`,
   `data_source_50k="imagenet-1k"`, in every canonical summary) and every
   baseline (`common.startup_checks.assert_val_set_matches_canonical_p2`,
   run at each baseline's own startup, reads these two fields LIVE from a
   canonical summary rather than hardcoding them) are evaluating the SAME
   thing: the entire canonical ImageNet-1k validation split, all 50,000
   images, no subset. There is only one such split — source and count
   together identify it completely for a full (non-subset) evaluation.
2. **The per-model FP-parity gate** (`common/fp_parity_gate.py`, 0.15pp
   tolerance) is the real, measured proof: it re-evaluates each baseline's
   own plain/unpatched model through that baseline's own harness on
   whatever images `--imagenet-dir` actually points at, and checks the
   result against P2.1's own S8 measurement on the SAME checkpoint. A
   materially different image set (wrong images, an order difference
   that mattered, a corrupted subset) would very likely fail that gate —
   it is a stronger, measured check than a manifest hash would be, not a
   weaker substitute for one. On the real B1 run: **0.022pp** on
   DeiT-S (harness 79.826% vs P2.1's own S8 79.848%) — see `results.csv`'s
   `r0` row.

**The baseline-to-BASELINE link** (does B2 evaluate the same 50k images B1
did, on whichever host) IS checked by literal hash, since this one has no
"P2.1 exports nothing" obstacle — every baseline runs through this
project's own code, which can export and check a real hash.
`experiments/phase2_2_baselines/val_manifest_canonical.json` freezes it, from B1's real Colab
run (`41ddeb0ebb2fea82`), asserted for equality by every baseline's
self-test item 5 (`common/notebook_self_test.check_data_determinism`) —
the old "no canonical yet, recording this run" first-run path no longer
exists once that file is committed; a mismatch is always a real problem
from here on, never a legitimate first run.

## Reproducing from scratch

Each baseline (B1-B4) runs once per its own pinned venv; B5
(`compare_pareto.py`) reads all of B1-B4's `results.csv` rows plus P2.1's
own `plans.csv` per model, orchestration-environment only (pandas/numpy,
no torch). See `experiments/phase2_2_baselines/README.md` for the per-baseline run instructions
and `commits.md` for the exact pinned upstream commit SHAs. There is no
`TANDEM_INTERACTIVE`-style checkpoint to worry about here — experiments/phase2_2_baselines/ has
no manual-review pause anywhere (confirmed by grep across every
`raise RuntimeError` in the non-vendored baseline code): every halt in
this codebase is a real correctness/environment guard, not a process
pause, so there was nothing to convert for reviewer release.
