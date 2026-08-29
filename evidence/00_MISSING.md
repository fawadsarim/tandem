# 00_MISSING.md — gaps, named and classified

Exhaustive as of this package's assembly. Every item below was actually searched for (not
assumed absent) — see the specific search performed. Nothing was substituted, reconstructed, or
approximated in the package itself; where regeneration was possible with no GPU, it was done
now (item 1) and the result is already in `02_P2.1_fleet/`.

---

## 1. `verify_p2_claims.py`'s output log — was missing, REGENERATED NOW

**What it was**: the 20/20 MATCH confirmation log for P2.1's fleet claims.
**Claim it supports**: every P2.1 fleet headline number (Family-X fit, S8 sanity gate, D_pred
values, C_id/C_min, per-budget fidelity) is independently re-derivable from raw tables, not just
asserted.
**Search performed**: `find . -iname "*verify_p2*log*"`, `-iname "*20_20*"`, `-iname
"*20-20*"` across the whole repo — nothing found.
**Classification**: **(a) regenerable locally, no GPU** — `scripts/verify_p2_claims.py` is pure
recomputation from committed CSV/JSON tables.
**Action taken**: re-ran it this session (`python3 scripts/verify_p2_claims.py`), captured the
full output. Result: **ALL CLAIMS VERIFIED, 20/20**. Saved at
`02_P2.1_fleet/verify_p2_claims_output.log`.

---

## 2. `AUDIT_REPORT.md` (P2.2 baselines) — does not exist under that name

**What it was asked for**: `experiments/phase2_2_baselines/AUDIT_REPORT.md`, implying a post-run audit of the
completed baseline reproductions.
**Claim it would support**: an independent audit trail that the baseline reproduction gates were
actually checked, not just reported.
**Search performed**: `find . -iname "*AUDIT_REPORT*"` across the whole repo.
**What actually exists**: `docs/archive/baselines_AUDIT_REPORT_pre-run.md` (and a sibling
`P22_AUDIT_REPORT_pre-run.md`) — both explicitly named "pre-run": these are PRE-REGISTRATION
audit checklists written **before** the baseline runs happened, not a final results audit. A
different document, for a different purpose, than what was requested.
**Classification**: **(c) genuinely does not exist as specified.** The pre-run version was
included anyway (`03_P2.2_baselines/AUDIT_REPORT_pre-run_ONLY.md`, name deliberately marked
`_ONLY` so it is not mistaken for the requested file) since it is the closest real document and
still useful (it shows the audit criteria were fixed before results were known). If a genuine
post-run audit is needed for the write-up, it does not currently exist and would need to be
newly written by comparing `deviations.md` + `results.csv` against the pre-run checklist — that
is authorship, not file recovery, and was not done here.

---

## 3. Baseline calibration index files (`_calib_indices/`) — not present locally

**What it was asked for**: "The calibration index files (`_calib_indices/`) for every baseline
run."
**Claim it supports**: exact reproducibility of which images were used for RepQ-ViT's and
FQ-ViT's activation calibration (a documented, non-trivial reconstruction — see
`experiments/phase2_2_baselines/common/calib_capture.py`'s own docstring on why this needed careful re-implementation
of each upstream repo's own seeding behavior).
**Search performed**: `find experiments/phase2_2_baselines -iname "*calib_indices*"` — nothing found locally.
**What the code confirms**: `experiments/phase2_2_baselines/b2_repq_vit/run.py:287` and
`experiments/phase2_2_baselines/b3_fq_vit/run.py:346` both construct
`calib_indices_path = .../b{2,3}_.../_calib_indices/{model}_{config}_{row_kind}.json` and call
`capture_calib_indices(...)` as a real step of each baseline's own reproduction run (the SAME
run that produced the numbers already in `results.csv`) — so this file was almost certainly
written once, on Colab, during that real run, and simply never downloaded back into this git
checkout (this directory is not committed — likely gitignored or just never pulled).
**Note**: B1 (ToMe) and B4 (sequential-composite) do NOT use this mechanism at all (checked —
no `calib_capture`/`capture_calib` references in either `run.py`) — ToMe needs no activation
calibration (token-merging only), and B4 is compositional. So this gap applies to B2/B3 only.
**Classification**: **(b) needs a Colab/GPU run** — specifically, re-running
`experiments/phase2_2_baselines/b2_repq_vit/B2_repq_vit_colab.ipynb` and `experiments/phase2_2_baselines/b3_fq_vit/B3_fq_vit_colab.ipynb`'s
existing run cell (the one that already invokes `run.py`) would regenerate
`_calib_indices/*.json` as a side effect, using whatever `--imagenet-dir` that Colab session has
materialized. Not done here — this laptop has no materialized ImageNet directory and the
capture function needs one (`torch`+dataset-length dependent, not GPU-compute dependent per se,
but practically tied to the Colab environment where the data lives).

---

## 4. `equivalence_fingerprint.json` — not found

**What it was asked for**: the FP-parity smoke-eval fingerprint `equivalence_gate.py`'s
`run_gate()` produces (256-image smoke eval, first-run-canonical).
**Claim it supports**: that each baseline's Colab environment reproduces the same FP16 baseline
numbers session-to-session / host-to-host, a determinism cross-check.
**Search performed**: `find . -iname "*equivalence_fingerprint*"` across the whole repo.
**Classification**: **(b) needs a Colab/GPU run.** `experiments/phase2_2_baselines/common/equivalence_gate.py`'s
`FINGERPRINT_PATH` is written on first real run of any baseline's Colab notebook and would need
an actual GPU forward pass (a real 256-image eval) — not something derivable from static files.
Would be produced automatically the next time any `B{1,2,3,4}_*_colab.ipynb` is run from a fresh
session (it's the FIRST thing each notebook's self-test does per `notebook_self_test.py`'s own
design, per this project's memory of that module).

---

## 5a. Cross-check: both new verifier scripts corroborate items above, first-hand

`verify_g1_claims.py` and `verify_baselines_claims.py` (added to this package on request — see
`00_INTEGRITY.md` section 2) were run for real this session. `verify_baselines_claims.py`'s own
output independently re-confirms item 4 (`equivalence_fingerprint.json`) is genuinely absent —
its check 4 reports "NO DATA YET" rather than silently skipping — and separately surfaces item 6
below, a gap not in the original task's explicit list. `verify_g1_claims.py`'s 5/5 pass has no
missing-file findings of its own (E5/G1's raw data is fully present); it is listed here only as
the source of the newly-added `01_phase1_E1-E5_E4a/verify_g1_claims_output.log`.

---

## 6. `pareto_comparison.csv` (B5) — not found, discovered via `verify_baselines_claims.py`

**What it was asked for**: not itself named in the task's original list — this gap was discovered
as a side effect of adding `verify_baselines_claims.py` to the package (see item 5a above), not
from a manual search against the original spec. Included here anyway per the task's own
instruction to be exhaustive about gaps.
**Claim it would support**: `verify_baselines_claims.py`'s check 5, a direct numeric comparison of
each baseline's Pareto-frontier position against the others, read from a dedicated CSV rather than
recomputed from `results.csv` + `compare_summary.json` each time.
**Search performed**: `find . -iname "*pareto_comparison*"` across the whole repo — nothing found;
also confirmed via `verify_baselines_claims.py`'s own check 5, which looks for this exact filename
and reports "NO DATA YET" rather than erroring, i.e. its absence is an anticipated, handled state
in the verifier itself, not a broken path.
**What exists instead**: `B5_all_models_pareto.png` (the rendered plot, already in this package)
and `compare_summary.json` (the underlying comparison numbers in JSON form, already in this
package) — the claim itself IS supported by files already present; only this specific
intermediate CSV artifact is missing.
**Classification**: **(b) needs a Colab/GPU run** — `pareto_comparison.csv` is written by
`experiments/phase2_2_baselines/compare_pareto.py` as part of its own comparison pipeline; regenerating it does
not itself need a GPU forward pass (it's a comparison over already-computed numbers in
`results.csv`), but it does need `B5_compare_pareto_colab.ipynb`'s existing run cell invoked in an
environment with the repo checked out, i.e. the same Colab session pattern as the rest of B5 — not
attempted here since it is a strictly redundant artifact (see above) and out of scope for this
package's raw-data-preservation goal.

---

## 7. `out/segments/manifest.json` — existed only on the Jetson device, not in git

**What it was asked for**: not itself named in the task's list, but relevant to "the int8 QDQ
experiment outputs" bullet.
**Classification**: **not actually missing** — pulled directly from the device via `scp` this
session (device state, not local-repo state) and included at
`04_jetson_hardware/out/segments/manifest.json`. Flagged here only because its hash-verification
story differs from every other file (see `00_INTEGRITY.md` section 1) — there is no local git
copy to diff against, by design (`jetson/.gitignore` excludes the rest of `out/segments/`).

---

## 8. Energy-per-mode (power-mode sweep Part 5) — DELIBERATELY SKIPPED, not an omission

**What it was**: Part 5 of the power-mode sweep task — re-running the validated energy protocol
(sustained-loop, throughput-derived mJ/inference, idle-subtracted dynamic variant) at each of the
3 power modes, FP16 and INT8, all 5 buckets, plus a fresh idle baseline and monotonic sanity
ladder per mode.
**Why it doesn't exist**: the task's own instructions explicitly authorized skipping Part 5 "if
short on time." Parts 1–4 consumed substantially more device time than planned — the Part 1
reproduction-gate retry (see item below), a genuine GPU firmware fault at 15W (ACR/RISC-V
bootstrap failure, recovered by a clean reboot), discovering and temporarily disabling
`maxn-super-autofix.service` (which was silently reverting every mode switch back to
MAXN_SUPER), and the GPU/EMC real-ceiling investigation now recorded in `00_CORRECTIONS.md` item
c — five device reboots total before Part 5 was even reached.
**Classification**: **not a gap — a scoping decision, made after both pre-registered questions
were already answered.** The sweep's PRIMARY question (does proxy divergence persist across
power modes?) and SECONDARY question (does the precision axis revive at reduced clocks?) were
both closed cleanly by Parts 1–4 (`04_jetson_hardware/out/powersweep/powersweep_analysis_summary.json`,
`replan_by_mode.csv`) — divergence persists at 2.19–2.25x across all 3 modes, and INT8 stays
slower than FP16 in all 15/15 mode-bucket cells. Energy-per-mode would only generalize a result
that is already closed, not open a new question. If it's later wanted, it needs a live Jetson
session: `python energy_instrument.py` (Part 0, per mode) then `python energy_lut.py` (per mode),
same protocol as the original single-mode `out/energy/` artifacts already in this package —
**(b) needs Jetson/GPU hardware**, not Colab (jtop/TensorRT power sampling is device-specific).

---

## 9. Reproduction-gate first attempt — not a gap, but recorded as a near-miss, not smoothed over

**What happened**: the power-mode sweep's own Part 1 reproduction gate FAILED on its first
attempt (9/10 rows within 3%; `bucket=69/precision=int8` at +3.19%, just over the pre-registered
3% hard-stop). Per that rule, the sweep halted before any lower-mode measurement was trusted.
**What was done**: a non-destructive 5-trial diagnostic remeasurement of just that config showed
tight clustering (CV=0.29%, mean within 0.78% of committed) — consistent with ordinary
measurement noise at the smallest/fastest bucket, not drift. A full, clean OFFICIAL re-run (all
10 rows, not just the failing one) was then performed and passed 10/10.
**Classification**: **not missing — both attempts are included**, specifically so this halt-and-
diagnose episode is part of the record rather than tidied away:
`04_jetson_hardware/out/powersweep/reproduction_gate_ATTEMPT1_FAILED.csv` (first, failed attempt;
no device-side source file exists for it — see its own header note and `00_INTEGRITY.md` section
1) and `n69_diagnostic_repeats.log` (the diagnostic), alongside the official passing
`reproduction_gate.csv`.

---

## Summary table

| # | Item | Claim it supports | Classification | Status |
|---|---|---|---|---|
| 1 | `verify_p2_claims.py` output log | P2.1 fleet 20/20 verification | (a) no-GPU regenerable | **Done — regenerated, included** |
| 2 | `AUDIT_REPORT.md` (post-run) | Independent audit of P2.2 gates | (c) genuinely doesn't exist | Pre-run version included, clearly labeled |
| 3 | `_calib_indices/` (B2/B3) | Exact calibration image reproducibility | (b) needs Colab (materialized ImageNet) | Not regenerated — needs a live Colab run |
| 4 | `equivalence_fingerprint.json` | Cross-session FP determinism | (b) needs Colab/GPU | Not regenerated — needs a live Colab run |
| 6 | `pareto_comparison.csv` (B5) | B5 Pareto-frontier comparison detail | (b) needs Colab/GPU | Redundant with included PNG + JSON — not regenerated |
| 7 | `out/segments/manifest.json` | INT8 QDQ export provenance | N/A — pulled from device | **Done — included** |
| 8 | Energy-per-mode (sweep Part 5) | Would generalize the already-closed energy R² finding to 3 modes | N/A — deliberate scoping decision | Not attempted — both pre-registered questions already answered by Parts 1-4 |
| 9 | Reproduction-gate first (failed) attempt | Part 1's halt-and-diagnose record | N/A — not missing, included on purpose | **Done — both attempts included, see item 9 above** |

*(Item 5 is the cross-check note above, not a gap of its own — no table row.)*

## Second device (Jetson Orin NX 8GB)

Gaps specific to the second device are listed in
`06_orinnx_hardware/06_MISSING.md` and, in full, in
`../jetson-orinnx/FINDINGS.md` §7 and `../jetson-orinnx/PORT_NOTES.md`:
one operating point only; frontier-vs-ToMe arm and energy re-plan not run;
engine-level FP16/INT8 numerical parity unresolved (§9); ambient
temperature not measured; joint-vs-staged untested (all pre-registered
budgets degenerate). The NX energy proxy-divergence artifacts were produced
by the repo's own `jetson-orinnx/proxy_divergence.py` on the committed
`energy_lut.json` (no device), and are committed.

