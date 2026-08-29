# REMOVAL_CANDIDATES.md

Everything below was considered for deletion during public-release
cleanup and deliberately **not** deleted, because its provenance or
continued relevance could not be fully confirmed. Per this reorg's own
"correctness over tidiness" principle: if unsure, list it here rather than
delete it. Nothing in this list has been touched.

## 1. `experiments/phase2_2_baselines/reviewer_release/`

**What it is**: a 4-file subfolder (`results.csv`, `compare_summary.json`,
`B4_deit_small_pareto.png`, `B5_all_models_pareto.png`) that is, verified
directly (`diff -q`, re-checked after the reorg), **byte-identical** to
the live versions of those same 4 files one level up.

**Why it wasn't deleted**: nothing in this repository's own code or
scripts (`scripts/run_reviewer.sh`, `docs/REVIEWER_RUNBOOK.md`, any `.py`
file) reads from this subfolder — it appears to be a "curated snapshot"
folder that `b5_final_closure.py` itself populates on every run (see that
script's own `item C.8` step, which copies the live files into
`reviewer_release/` as its last action) as a deliberate, small
convenience bundle for handing to a reviewer without navigating the full
directory tree. It may be actively relied on by an external reader who
was pointed at this exact path in prior correspondence outside this
repo — something I have no visibility into.
**If confirmed unused, this whole subfolder can be deleted** (it will
regenerate automatically the next time `b5_final_closure.py` runs, so
deleting it loses nothing that isn't trivially reproducible).

## 2. ~~The "this repo is PRIVATE" / `GITHUB_TOKEN` clone flow~~ — FIXED, no longer a candidate

**Originally flagged here as deferred; fixed in a follow-up pass** once
the repo was actually made public. All 19 live Colab notebook scripts
(`experiments/phase1_E1-E5_E4a/*.py`, `experiments/phase2_1_fleet/P2_tandem_pipeline.py`,
5 files under `experiments/phase2_2_baselines/`, 7 `jetson/*_colab.py`)
had their clone cell simplified to a plain, unauthenticated `git clone`
(a public repo needs no token), and the stale "this repo is PRIVATE"
markdown removed — while preserving each file's separate, still-real
`HF_TOKEN` instructions where present, since those are unrelated and
still needed. Every `.ipynb` twin was regenerated via `jupytext` from its
edited `.py` source. Verified: `common.verify_colab_thin_wrappers` still
reports all 5 baseline notebooks as thin wrappers, and Tier 1 still passes
7/7 end-to-end after the change.

**A second, independent bug was found and fixed in the same pass**: 7 of
these files (`E1`–`E5`, `E4a`, `P2_tandem_pipeline.py`) `%cd`'d into
`{REPO_DIR}/notebooks` after cloning — a path that no longer exists after
this same reorg moved those experiments to `experiments/phase1_E1-E5_E4a/`
and `experiments/phase2_1_fleet/`. This was missed by the original
path-reference sweep because the string appeared as an f-string suffix
(`f'{REPO_DIR}/notebooks'`) with no trailing slash, which the sweep's
`"notebooks/"` pattern didn't match. Fixed to `%cd` into the correct new
directory for each file.

## 3. `experiments/phase2_2_baselines/b1_tome/requirements.txt` etc. — the live requirements.txt files themselves

Not removed, but flagged for visibility: `requirements/` (new, this
reorg) holds **reference copies** of these files, not the live ones. The
live files (read by every real `pip install -r requirements.txt`) remain
exactly where they were. This is not a duplication concern in the
"stale copy" sense since `requirements/README.md` states plainly which
copy is live — listed here only so a future maintainer knows both exist
and why, if they ever wonder whether to delete one.

## 4. `docs/archive/`, `LAB_LOG.md`, `docs/deviations.md`'s older entries

Not removal candidates — explicitly **kept as historical record, prose
mentions of pre-reorg paths (`notebooks/`, `baselines/`) left as
originally written** rather than edited to describe a directory structure
that didn't exist when they were written. See `docs/deviations.md`'s own
top-of-file note for the full reasoning. Listed here only for
completeness, since a future pass might otherwise wonder why these
weren't touched during the path-reference sweep.
