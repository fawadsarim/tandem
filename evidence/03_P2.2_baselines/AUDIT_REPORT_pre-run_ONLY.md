> **ARCHIVED — historical, pre-run snapshot.** Written before
> any real B1-B4 GPU run existed. For current, authoritative status see
> `../../REPRODUCIBILITY.md` and `../../baselines/deviations.md`. Kept for
> process history only — do not cite numbers from this file.

# P2.2 Baselines — Correctness/Finality Audit + Host-Equivalence Report

Scope: make the university-GPU and Colab paths for B1 (ToMe), B2 (RepQ-ViT),
B3 (FQ-ViT), B4 (sequential composites) IDENTICAL by construction. No
baseline's scientific settings (which flags, which calibration size, which
preprocessing, which published number) were changed anywhere in this pass
— every change below is either a correctness/finality fix (something that
was wrong or missing) or an infrastructure addition (determinism, host
recording, resume, cross-host verification). Where I could not
execution-verify something (no GPU in this environment), that is stated
explicitly rather than implied.

Method: every claim about RepQ-ViT's/FQ-ViT's actual behavior below was
checked against their real source at the pinned commits (fetched live from
GitHub), not assumed from README prose or prior notes. Package-version
claims were checked against real PyPI/torchvision release metadata. Where
I built new logic (calibration-index reconstruction, the BOPs identity
check, the smoke-subset builder), I ran it standalone against synthetic
inputs and confirmed the results — reported per-item below.

---

## Part A — Correctness & finality audit

### A.1 Pinning

| Item | B1 (ToMe) | B2 (RepQ-ViT) | B3 (FQ-ViT) | B4 (composites) |
|---|---|---|---|---|
| Commit hash in code (not README-only) | Already correct — `TOME_COMMIT` constant in `run.py` | Already correct — `REPQ_COMMIT` | Already correct — `FQVIT_COMMIT` | Already correct — both constants |
| Runtime assert: checkout == pinned hash | Already correct — `common.repo_utils.ensure_repo` hard-checks-out and asserts `HEAD == pinned_commit` before returning, for every baseline | same | same | same |
| requirements.txt: every direct dep `==` | **FIXED** — was `torch>=1.12.1`, `torchvision` (unpinned), `numpy>=1.24`, `tqdm`/`Pillow`/`scipy` unpinned | **FIXED** — was `torch>=1.10,<2.0`, rest unpinned | **FIXED** — `torch==1.7.1` was already exact; `torchvision`/`numpy`/`tqdm`/`Pillow` were unpinned | **FIXED** — same pattern as B1 |

**Finding A.1-1 (fixed):** none of the four `requirements.txt` files exact-pinned every dependency — most used `>=` floors or no pin at all. This is a real cross-host risk (a newer torchvision/PIL/numpy resolved at install time on one host vs. the other can silently change interpolation defaults, JPEG decode behavior, or floating-point reduction order). Fixed by pinning every package to a specific, real, PyPI-verified version:
- **B1/B4**: `torch==1.12.1` (ToMe's own documented floor — chosen deliberately over a newer release, since it's the version closest to what ToMe's authors actually tested; a newer torch I cannot execution-verify here is a bigger fidelity risk than staying at the documented floor) / `torchvision==0.13.1` (confirmed torch 1.12.1's official companion release: both published within the same minute on PyPI, 2022-08-05) / `timm==0.4.12` (unchanged, already pinned) / `scipy==1.9.3` / `numpy==1.23.5` / `tqdm==4.64.1` / `Pillow==9.2.0`.
- **B2**: same torch/torchvision/timm pins as B1/B4 (RepQ-ViT documents no exact torch pin at all — chose the same era as ToMe for consistency, flagged in `deviations.md` #1 as not RepQ-ViT's own verified pin) / `numpy==1.23.5` / `tqdm==4.64.1` / `Pillow==9.2.0`.
- **B3**: `torch==1.7.1` (unchanged) / `torchvision==0.8.2` (confirmed 1.7.1's official companion, same-minute PyPI publish, 2020-12-10) / `numpy==1.19.5` / `tqdm==4.62.3` / `Pillow==8.0.1` (all era-appropriate for Dec 2020, not TANDEM's modern floors — a numpy this old codebase was never tested against is a real, not theoretical, risk).
- **Orchestration** (`baselines/requirements.txt`): `pandas==2.1.4` / `numpy==1.26.4` / `matplotlib==3.8.2` (our own code, no legacy floor to respect).

**Caveat, stated plainly:** these exact pins are real, PyPI-verified-to-exist versions with a defensible selection rationale (official torch/torchvision companion pairs, era-matched secondary packages) — but they are **not individually execution-verified** against ToMe/RepQ-ViT/FQ-ViT's actual code, because this environment has no GPU/torch. The host-equivalence gate (Part B.2) is the real, empirical check once these run for real; if a pin turns out to be wrong, it will fail loudly there (import error or a `requirements_hash` mismatch against whatever gets installed), not silently.

### A.2 Protocol fidelity

All four baselines' actual settings were re-verified against real upstream source at the pinned commits (not just README prose, not just prior notes):
- **Preprocessing** (RepQ-ViT `utils/build_dataset.py`, FQ-ViT `test_quant.py`): confirmed byte-for-byte identical between the two repos — DeiT gets `mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`, `crop_pct=0.875`, bicubic; ViT gets `mean=std=(0.5,0.5,0.5)`, `crop_pct=0.9`, bicubic. This independently cross-validates the exact preprocessing bug found and fixed in P2's own pipeline for `vit_base_patch16_224` (same 0.5-norm/crop_pct=0.9 recipe). Already correctly reflected in `run.py`'s docstrings — no change needed.
- **Calibration mechanics** (both repos): `train_loader = DataLoader(ImageFolder(train), batch_size=calib_batchsize, shuffle=True, num_workers=16, drop_last=True)`, `--seed` default 0, their own `seed()` called first in `main()`. Confirmed already correctly described.
- **CLI flags** (`--w_bits`/`--a_bits` plural, not the README's singular `--w_bit`/`--a_bit`; FQ-ViT's `--quant --ptf --lis --quant-method minmax`): confirmed correct against real source, already correctly implemented.

**Finding A.2-1 (fixed) — reproduction gates existed but did not halt on failure.** `check_gate()` correctly computed `"pass"`/`"fail"`/`"no_published_reference"`, and the banner printed it — but no `run.py` actually stopped when `gate_status == "fail"`; a failed reproduction was recorded to `results.csv` exactly like a passing one. Fixed: B1/B2/B3 now `assert gate_status != "fail"` immediately after the gate check, with a message citing the measured/published/diff values and the published.json source, HALTING before any further config runs (and before `compare_pareto.py` could ever treat an unreproduced number as trustworthy). B4 correctly has no gate at all (`gate_status="not_applicable"`, since no published reference exists for either composite) — left unchanged, this was already correct.

**Finding A.2-2 (already correct) — published-number citations.** Re-checked every number in all three `published.json` files against the real source cited in each file's own `"source"`/`"source_url"` fields (ToMe README DeiT table, RepQ-ViT README W6A6 table, FQ-ViT README "Ours" row) — all four numbers (79.36/78.90/81.27/83.62/79.17/81.20/83.31) match what those tables actually publish, correctly restricted to the exact config each gate applies to (documented already in `deviations.md` #3/#5/#9). No changes needed.

### A.3 Determinism

**Finding A.3-1 (fixed) — TF32 was not disabled anywhere.** This is the audit's explicitly flagged #1 risk, and it was real: neither `run.py` nor RepQ-ViT's/FQ-ViT's own `seed()` (checked in their real source) ever touches `torch.backends.cuda.matmul.allow_tf32` / `torch.backends.cudnn.allow_tf32`. Left at PyTorch's default (TF32 **on** on Ampere+ GPUs, and TF32 simply absent as hardware on Colab's T4), the same code would silently produce different logits on the two hosts. Fixed via `common/determinism.py`:
- B1/B4 (in-process): `disable_tf32()` called first thing in `main()`.
- B2/B3 (subprocess of an unmodified upstream script): a `sitecustomize.py` is generated and injected onto the subprocess's `PYTHONPATH` (`subprocess_env_with_tf32_disabled`) — Python's `site` module auto-imports it at interpreter startup, before `test_quant.py`'s own `import torch` runs, disabling TF32 for that whole subprocess without touching their file on disk. Verified this mechanism is sound: neither repo's `seed()` ever touches these two flags (confirmed from source), so nothing later in their run re-enables TF32.

**Finding A.3-2 (fixed) — seeds were inherited implicitly, not pinned in our own code.** Both upstream scripts default `--seed` to 0 and are otherwise deterministic (see A.3-4) — our `run.py` files never passed `--seed` explicitly, silently relying on their default never changing. Fixed: `common.determinism.SEED = 0` is now the single source of truth, passed explicitly to both subprocess calls and used by `seed_everything()` for B1/B4's in-process seeding (same recipe as RepQ-ViT's `seed()`, verified from source).

**Finding A.3-3 (already correct) — eval order.** Confirmed from real source: both RepQ-ViT's and FQ-ViT's own `val_loader` use `shuffle=False`; B1/B4's `common/imagenet.py` also uses `shuffle=False`. Full-50k eval order is deterministic on all four baselines already — no change needed.

**Finding A.3-4 (fixed) — calibration subset was neither recorded nor independently verified deterministic.** RepQ-ViT's and FQ-ViT's calibration `train_loader` uses `shuffle=True` with no explicit `generator=` — I traced this through PyTorch's actual `RandomSampler.__iter__` source (torch 1.12.1 tag) to confirm this is in fact fully deterministic given a fixed global seed (`generator=None` draws one seed value from the *already-seeded* global RNG, then uses that to seed a fresh local generator — reproducible run to run, and unaffected by `num_workers` since index *selection* happens in the sampler in the main process, not in the worker processes that only *fetch* by index). Built `common/calib_capture.py`, which replicates each repo's exact `seed()` → `ImageFolder(train)` → `RandomSampler` call sequence as a **separate, cheap side computation** (not a modification of their file) and reads off the identical index sequence directly, writing `{index, path, class_idx}` records to `_calib_indices/<config>.json` before every real B2/B3 run. Ran this standalone against a synthetic directory tree — confirmed the captured indices exactly match a `_sorted_samples()` reconstruction of what `ImageFolder` + `RandomSampler` would select. **Caveat:** the equivalence between the capture script and the real subprocess's actual behavior rests on a source-level proof (verified against real GitHub source, cited above), not a side-by-side execution against their actual code (no GPU here) — the first real B2/B3 run is the empirical confirmation.

**Finding A.3-5 (fixed) — host/GPU/torch/CUDA were not recorded anywhere.** `common.determinism.resolved_host_info()` (in-process, B1/B4) and a `python -m common.determinism host-info` subprocess call (B2/B3, whose `run.py` is deliberately torch-free) now populate `host`/`gpu_name`/`torch_version`/`cuda_version` in every `results.csv` row.

### A.4 Our-cost (BOPs) attachment

**Finding A.4-1 (already correct, now unit-checked) —** `common/bops.py`'s docstring claimed its formula was "ported verbatim" from P2's own `block_bops`/`plan_cost`. I re-derived P2's exact formula independently (from `notebooks/P2_tandem_pipeline.py` Section 10) and compared algebraically: identical. Built `common/bops_identity_check.py`, which computes the IDENTITY-config (FP16 everywhere, no merge) cost via `common.bops.cost_from_trajectory` and via an independent re-implementation of P2's own `C_id` formula, and asserts exact equality. **Ran this for real** (pure Python, no GPU needed):

```
deit_small_patch16_224: identity-config BOPs = 1.162418e+12
deit_base_patch16_224:  identity-config BOPs = 4.466548e+12
vit_base_patch16_224:   identity-config BOPs = 4.466548e+12
All identity-config equivalence checks PASSED.
```

`deit_small_patch16_224`'s value (`1.1624e12`) matches P2's own archived `C_id` for that model exactly, cross-validating both the formula and this check.

### A.5 Schema

**Finding A.5-1 (fixed) — results.csv columns were not unified with what the audit specifies, and lacked host/determinism fields entirely.** `common/results_io.py`'s `RESULTS_COLUMNS` extended: `commit` → `repo_commit`; added `n_images`, `host`, `gpu_name`, `torch_version`, `cuda_version`, `config_hash`. `published_top1_pct`/`gate_status`/`gate_diff_pct_points`/`calib_images` were kept (not in the audit's literal column list, but dropping them would lose the reproduction-gate audit trail the fairness rules require — kept as an intentional superset, not a deviation). `results.csv` itself had zero data rows recorded yet, so its header was rewritten directly (no migration needed). **Verified all four `run.py`'s `append_result()` calls against the real `RESULTS_COLUMNS`** (pure Python, no GPU needed) — all 17 columns set explicitly by every runner, zero unexpected keys (a typo here would have raised at runtime, mid-run, on real GPU time):

```
b1_tome: 17 keys passed, unexpected=[], not-explicitly-set=[]
b2_repq_vit: 17 keys passed, unexpected=[], not-explicitly-set=[]
b3_fq_vit: 17 keys passed, unexpected=[], not-explicitly-set=[]
b4_sequential_composites: 17 keys passed, unexpected=[], not-explicitly-set=[]
ALL FOUR run.py append_result() calls match RESULTS_COLUMNS exactly (no typos).
```

---

## Part B — Single source of truth + host equivalence

### B.1 Refactor: notebooks as thin wrappers

**Already correct — verified, not assumed.** The five `*_colab.py` notebook sources (B1–B4 + B5/compare_pareto) already contained zero eval/preprocessing/calibration logic: each one clones the repo at a pinned ref, `pip install`s the SAME `requirements.txt` the university GPU uses, resolves `HF_TOKEN` from Colab Secrets only, and shells out to the SAME `run.py`/`compare_pareto.py` entry points, unmodified. Formalized this as a real, re-runnable check rather than an ad-hoc grep: `common/verify_colab_thin_wrappers.py` scans every `*_colab.py` for a blocklist of patterns that would only appear in real eval/model/transform/calibration code (`def `/`class `, `torch.no_grad`, `.argmax(`, `nn.Module`, `CenterCrop`/`Normalize`, `ImageFolder(`, `quant_model`, `.forward(`, `DataLoader(`). **Ran it for real:**

```
Checked 5 Colab notebook source(s): ['B1_tome_colab.py', 'B2_repq_vit_colab.py', 'B3_fq_vit_colab.py', 'B4_sequential_composites_colab.py', 'B5_compare_pareto_colab.py']
All are thin wrappers -- no eval/preprocessing/calibration logic found.
```

Updated: each notebook's `run.py` invocation now passes `--host-tag colab` explicitly (matching the university path's new `--host-tag university_gpu` convention), and all four `.ipynb` files were regenerated from their `.py` sources and re-validated (`nbformat.validate`). HF token handling was already secrets-only (`google.colab.userdata.get("HF_TOKEN")`) — confirmed, no change needed.

### B.2 Host-equivalence gate

Built `common/equivalence_gate.py` (framework-free, runs from the orchestration environment):

1. **Fixed 256-image subset, recorded indices.** `build_smoke_subset()` symlinks (falls back to copy if symlinks are unavailable) the first 256 val images and first 2,000 train images, in the SAME sorted order `torchvision.datasets.ImageFolder` itself uses internally (sorted class dirs, sorted filenames) — so index N in the smoke subset is provably index N in a real ImageFolder built from the same directory, not a separately-chosen sample. **Verified for real** against a synthetic directory tree: exact image count, exact determinism match against an independent `ImageFolder`-order reconstruction, and idempotency (a second call skips rather than duplicates) — all passed.
2. **Reuses each baseline's real `run.py`, unmodified**, pointed at the smoke subset via `--imagenet-dir`, with results routed to a separate CSV (`TANDEM_RESULTS_CSV_OVERRIDE`, a new environment-variable override in `common/results_io.results_csv_path()`) so a smoke run can never pollute the real `results.csv`. This means the equivalence gate exercises the EXACT SAME code path (TF32 disable, seeding, gates, calibration capture) as a real run — not a separate, potentially-drifting implementation.
3. **Canonical fingerprint.** `equivalence_fingerprint.json` records, per baseline's representative config (B1: `r13`; B2: `w6a6`; B3: `w8a8`; B4: `tome_r13_then_repq_w8a8`), the correct-count on the 256-image subset (recovered exactly from `top1_50k × n_images`, verified round-trip-exact against a synthetic CSV row), the repo commit, GPU/torch/host, and a `requirements_hash` (`sha256` of that venv's actual `pip freeze` output — the empirical ground truth, not just the `requirements.txt` file). The FIRST host to run the gate writes the canonical record; every later run (any host) compares.
4. **Comparison policy, exactly as specified:** exact correct-count match → pass; ±1 → tolerated but logged with both hosts' GPU/arch (the documented, narrow "borderline-logit-flip across GPU architectures" failure mode — floating-point reduction order in cuDNN/cuBLAS kernels is not guaranteed bit-identical across GPU *generations*); larger delta → `AssertionError`, halting before any full run; `requirements_hash` mismatch → halts with the full `pip freeze` output printed for diagnosis.

**Caveat, stated plainly:** the gate's *mechanism* is built and its pure-Python logic (subset construction, determinism, result-parsing, idempotency) is verified for real. The gate has **not been run against real GPUs on either host** (no GPU in this environment) — there is no canonical `equivalence_fingerprint.json` yet. The first person to run `common/equivalence_gate.py` on either host creates it.

### B.3 Resume

**B1/B4 (our own in-process eval loops — fully implemented and reusable):** `common/checkpoint.py` (atomic-write, config-hash-guarded, mirrors P2's own `_load_checkpoint` pattern) + `common.eval_utils.evaluate_top1_resumable()` (per-batch checkpoint, saves every 10 batches, resumes by re-iterating the deterministic `shuffle=False` val loader and skipping — not re-scoring — already-completed batches). Wired into both B1 (per r-value) and B4 (per composite), each with its own `config_hash`-tagged checkpoint file under `_checkpoints/`.

**B2/B3 — structurally cannot get the same treatment, documented rather than silently gapped (`deviations.md` #14).** Their measurement is delegated entirely to an UNMODIFIED upstream `test_quant.py` subprocess; its own `validate()` loop has no checkpoint hook, and adding one requires either modifying their file (violates the fairness rule) or fully reimplementing their quantization+eval pipeline ourselves (defeats the purpose of running their own code, and risks silently diverging from their exact protocol — the opposite of what this audit is for). A Colab disconnect mid-run restarts B2/B3 from scratch; partial mitigation noted: their single `validate()` pass is structurally shorter than B1's 3-way sweep, so the exposure window is smaller, not eliminated. This is a real, load-bearing limitation, not an oversight — flagged here explicitly per the instruction that no behavioral gap goes unlisted.

---

## Summary of every file touched

**New:** `common/determinism.py`, `common/calib_capture.py`, `common/bops_identity_check.py`, `common/checkpoint.py`, `common/equivalence_gate.py`, `common/verify_colab_thin_wrappers.py`, `AUDIT_REPORT.md`.

**Modified:** `common/results_io.py` (schema), `common/eval_utils.py` (added `evaluate_top1_resumable`), `b1_tome/run.py`, `b2_repq_vit/run.py`, `b3_fq_vit/run.py`, `b4_sequential_composites/run.py` (all four: TF32/seed/host-recording/gate-halt/config_hash; B1/B4 additionally: checkpoint/resume; B2/B3 additionally: sitecustomize injection, explicit `--seed`, calibration-index capture), `requirements.txt` ×5 (exact pins), `results.csv` (header only — zero data rows existed), `deviations.md` (+4 entries, #12–15), `README.md` (host-equivalence gate section, `--host-tag` examples, updated layout), all 4 `*_colab.py`/`.ipynb` (added `--host-tag colab`).

**Unchanged (confirmed correct, not just left alone):** `commits.md`, `published.json` ×3, `common/bops.py`, `common/repo_utils.py`, `common/repro_gate.py`, `common/imagenet.py`, `compare_pareto.py`, `B5_compare_pareto_colab.py/.ipynb`, every scientific setting (bit configs, calibration sizes, preprocessing, published-number citations, ToMe's `prop_attn=True`, B4's RepQ-`build_model`-skip resolution).

## What this audit does NOT claim

No baseline has been executed in this pass — this environment has no GPU. Every fix above is either (a) verified against real, live-fetched upstream source and real PyPI metadata, or (b) verified standalone via pure-Python/synthetic-input tests where the logic doesn't require a GPU (BOPs identity check, schema key-matching, smoke-subset determinism, calibration-index-capture logic). The exact-pin version choices and the calibration-index-capture equivalence proof are the two places with the most residual, honestly-flagged uncertainty — both are designed so that if either is wrong, the failure is loud (an import error, or a host-equivalence gate mismatch with a full diagnostic printed), never silent.
