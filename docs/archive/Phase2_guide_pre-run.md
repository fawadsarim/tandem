> **ARCHIVED — historical, pre-run planning doc.** Written
> before P2.1 or P2.2 had been executed. Both are now complete; for the
> current, authoritative run order and expected outputs see
> `../../REVIEWER_RUNBOOK.md` (Tier 2) and `../../REPRODUCIBILITY.md`.
> Kept for process history only.

# Phase 2 — Running Guide

Two deliverables, two environments. Neither has been executed yet (no
GPU access from where they were built) -- this guide is what to actually
do with them.

| Deliverable | Environment | Purpose |
|---|---|---|
| `notebooks/P2_tandem_pipeline.ipynb` | **Colab** | TANDEM's own per-model damage tables, allocator, audited accuracy |
| `baselines/` | **University 24GB GPU** (not Colab) | Reproduce external published baselines at matched cost for comparison |

Run Part A (Colab) and Part B (GPU) independently, in either order --
`baselines/compare_pareto.py` just needs at least one completed P2 model
on disk before it produces anything meaningful, but nothing about running
the baselines themselves depends on P2 having finished first.

---

## Part A: `P2_tandem_pipeline.ipynb` on Colab

### Which models, and in what order

Run the notebook **four times**, once per model, by editing
`CONFIG["MODEL_NAME"]` in Section 2 and doing Runtime > Run all. Recommended
order (cheapest/lowest-risk first, so a Colab or protocol problem shows up
early and cheaply rather than after the most expensive run):

1. **`deit_small_patch16_224` first** -- this is the one case that runs in
   ARCHIVED mode (Phase 1's E1-E5 already validated it): only S0 (setup),
   S1 (calibration), S8 (50k re-scoring of the 6 archived E5 joint plans),
   and S9 (artifacts) execute. It's the fastest run by a wide margin and a
   good smoke test that the notebook, your Colab environment, and your
   HF_TOKEN access all work before committing to the three full runs.
2. **`deit_tiny_patch16_224`** -- smallest full pipeline (d=192), cheapest
   of the three fresh-table models, good second checkpoint that S2/S3
   (damage-table sweeps) and S6/S7 (allocator + audit) work end-to-end.
3. **`deit_base_patch16_224`**
4. **`vit_base_patch16_224`**

(3) and (4) are both d=768 -- similarly costed, roughly ~4x deit_small's
per-block linear-term BOPs, so expect them to be the slowest of the four
runs. No real wall-clock numbers exist yet for any of the three fresh
models (Phase 1's timings are DeiT-S-only, not transferable) -- rather
than guess, trust the notebook's own **dry-run projections**: Section 12
times one real config from S2 and S3 before committing to the full sweep
and prints the projected total; if S2+S3 together project over 60 minutes,
or S8 projects over 90 minutes, the notebook halts (`raise RuntimeError`)
with a recommendation rather than silently burning your session on an
underestimated run. That halt is not a failure -- re-run cells manually
from that point if you've decided the projection is acceptable, or move
that model to the university GPU instead.

### Before you start

- **HF_TOKEN via Colab Secrets only**, never pasted into a cell. Colab's
  left sidebar > key icon > add secret named `HF_TOKEN`, toggle notebook
  access on. Section 1 checks for it automatically
  (`userdata.get("HF_TOKEN")`) and falls back to the Imagenette tier with a
  loud warning if it's missing -- **the S8 FINAL 50k stage hard-requires
  real ImageNet-1k access and will assert-fail on the Imagenette fallback**
  (it can't supply 50k labeled images), so don't skip this.
- **GPU runtime**: Runtime > Change runtime type > T4 GPU (or better; a
  more powerful GPU only helps the runtime-discipline dry-run projections
  come in lower).
- Section 1b greps the notebook's own source for a literal-looking HF
  token before proceeding, as a standing defensive check -- if you ever
  paste a token directly into a cell instead of using Secrets, this should
  catch it. Don't rely on it as your only safeguard, though: see the git
  hygiene note at the end of this guide.

### What you'll interact with mid-run

- **S4 (STOP-P2a)** -- fresh models only (skipped for `deit_small`'s
  archived mode). Unconditional halt: prints both measured damage tables,
  FP top-1, and any gate anomalies, then always raises. This is a
  mandatory pause for manual review, not a failure -- paste the printed
  block for review, get GO, then re-run cells from the one *after*
  this checkpoint (not the whole notebook again).
- **S5 (mini Family-X gate)** -- conditional halt: only stops if the
  8-config cross-axis additivity fit fails (slope outside [0.8,1.2] or
  R²<0.85). A pass proceeds automatically, no pause.
- **Runtime-discipline halts** (Section 12, and again before S8) --
  conditional on the dry-run projection exceeding 60 min (S2+S3) or 90 min
  (S8); see above.

### What lands on disk

`results/P2/<MODEL_NAME>/`: `Dq_table.csv`, `Dm_table.csv`, `familyX.csv`,
`plans.csv` (predicted + 5k + 50k accuracy per budget), `pareto_<model>.png`,
`plan_composition.txt`, `P2_<model>_summary.json`. Download this whole
folder (or sync it back into the repo) after each model -- Colab sessions
are ephemeral, this is not.

### Cross-model comparison

The notebook's last cell (Section 24) reads `results/P2/*/P2_*_summary.json`
from disk and produces `results/P2/cross_model_pareto.png` +
`cross_model_comparison.csv`. It needs >=2 completed models to do anything
(prints a message and exits cleanly otherwise) and is safe to re-run after
each additional model finishes -- it doesn't depend on notebook state from
earlier cells, only on what's actually on disk.

---

## Part B: `baselines/` on the university GPU

Read `baselines/README.md` and `baselines/deviations.md` once before your
first run -- this section is the condensed action sequence; those two
files are the reference for *why* each step is shaped the way it is
(exact commit pins, exact published numbers and their sources, the
real version conflicts between the three upstream repos, and B4's
documented integration risk).

### Prerequisites (do these once)

1. **A local ImageNet-1k ILSVRC2012 copy**, standard `ImageFolder` layout:
   ```
   $IMAGENET_DIR/
     train/<wnid>/*.JPEG   (needed for B2's and B4's calibration batches)
     val/<wnid>/*.JPEG     (the full 50,000-image standard val split)
   ```
   ```bash
   export IMAGENET_DIR=/path/to/imagenet
   ```
2. **Four separate virtual environments** -- not optional, not a
   preference: ToMe requires `torch>=1.12.1`, FQ-ViT pins `torch==1.7.1`,
   there is no version satisfying both (verified from each repo's own
   install docs, see `deviations.md` #1).
   ```bash
   cd baselines
   python -m venv .venv_orch && .venv_orch/bin/pip install -r requirements.txt

   cd b1_tome && python -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
   cd b2_repq_vit && python -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
   cd b3_fq_vit && python -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
   cd b4_sequential_composites && python -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
   ```
   `b3_fq_vit/requirements.txt` has a header comment about `cudatoolkit=10.1`
   likely needing a newer CUDA build on current hardware -- read it before
   `pip install` fails partway through.

### Running order

Each baseline is one command. Commit pinning, repo cloning, and
`results.csv` writing all happen automatically inside `run.py` -- nothing
to set up per-baseline beyond the venv.

```bash
# B1 -- ToMe, DeiT-S, r in {8,13,16}. In-process (imports tome directly),
# must run inside its own venv.
cd baselines/b1_tome
.venv/bin/python run.py --imagenet-dir $IMAGENET_DIR
cd ../..

# B2 -- RepQ-ViT, W8A8+W6A6, DeiT-S/DeiT-B/ViT-B. Subprocess wrapper --
# run.py itself is lightweight (orchestration env is fine); --python
# points at the venv that actually executes their test_quant.py.
cd baselines
.venv_orch/bin/python b2_repq_vit/run.py --imagenet-dir $IMAGENET_DIR \
    --python b2_repq_vit/.venv/bin/python
cd ..

# B3 -- FQ-ViT, W8A8, same three models. Same subprocess pattern as B2.
cd baselines
.venv_orch/bin/python b3_fq_vit/run.py --imagenet-dir $IMAGENET_DIR \
    --python b3_fq_vit/.venv/bin/python
cd ..

# B4 -- sequential composites (ToMe(r=13)->RepQ-W8A8, ToMe(r=16)->RepQ-W6A6),
# DeiT-S only. In-process, own venv. HIGHEST-RISK RUNNER -- read
# b4_sequential_composites/run.py's docstring before trusting its output;
# treat the first run as a validation pass (does it produce sane,
# non-collapsed accuracy?), not a trusted number.
cd baselines/b4_sequential_composites
.venv/bin/python run.py --imagenet-dir $IMAGENET_DIR
cd ../..
```

Recommended order is B1 -> B2 -> B3 -> B4: B4 reuses mechanisms from B1
(ToMe patching) and B2 (RepQ calibration/reparam), so confirming those work
individually first makes it much easier to tell, if B4 misbehaves, whether
the problem is environmental (bad ImageNet path, missing CUDA) or the
actual novel glue code.

Every run appends to `baselines/results.csv` (never overwrites) and prints
a reproduction-gate banner per config: `PASS`/`FAIL` against a located
published number, or `NO_PUBLISHED_REFERENCE` where none exists (see
`deviations.md` for exactly which configs fall into that last bucket --
ToMe's r=8/r=16, RepQ-ViT's W8A8 configs, and both B4 composites, none of
which have a published number to check against by construction).

### Then, the comparison (B5)

```bash
cd baselines
.venv_orch/bin/python compare_pareto.py
```

Reads `results.csv` + whatever's in `results/P2/*/plans.csv` (Part A's
output) and writes `baselines/pareto_comparison.csv`: for every baseline
row, TANDEM's own Pareto curve interpolated at that row's `bops_ours`
cost, and the matched-cost delta. Re-runnable any time either side gains
new rows. **No conclusions in the file** -- it's numbers only, by design;
reading the deltas is a manual step, not something the script
does for you. Rows flagged `extrapolated=True` mean the baseline's cost
fell outside TANDEM's own measured range for that model -- treat those
deltas as less trustworthy than an interpolated one.

---

## Git hygiene (done this session)

- **Found and fixed:** `.env` (containing a live-looking HF token) was
  tracked in git and pushed to `github.com/fawadsarim/tandem`. It's now
  untracked (`git rm --cached`, staged, not yet committed -- commit this
  yourself when ready) and `.gitignore` now excludes `.env`/`.env.*`
  going forward. **The token itself should still be rotated at
  huggingface.co** -- untracking the file doesn't scrub git history, and a
  revoked token is harmless even if it remains visible in old commits,
  which is the simpler fix than rewriting history.
- Root `.gitignore` extended: LaTeX build artifacts (`*.aux`/`*.log`/
  `*.synctex.gz`/etc. -- sources and PDFs stay tracked, intermediates
  don't), OS/editor cruft (`.DS_Store`, `Thumbs.db`, `.idea/`), and
  `.venv_orch/` alongside the existing `.venv/`.
- `baselines/.gitignore` (new, scoped to that directory) excludes
  `external/` (cloned third-party repos -- large, separately licensed,
  never TANDEM code; `commits.md` pins what to re-clone), all four
  per-baseline `.venv/` dirs, and any local `imagenet/` copy.
- Nothing was deleted. `baselines/` itself is untracked and unstaged --
  add and commit it whenever you're ready; I didn't stage or commit
  anything beyond the `.env` fix, per your repo's standing "don't commit
  unless explicitly asked" convention.
