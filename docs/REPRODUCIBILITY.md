# Reproducibility

What reproduces, at what tolerance, and how to check it yourself — for
the whole project (Phase 1's E1-E5/E4a, P2.1, P2.2/baselines, and
G1-HW/jetson). This is the consolidated pin table; each phase's own
detailed reproducibility narrative lives alongside its code
(`experiments/phase2_2_baselines/REPRODUCIBILITY.md`, `jetson/README.md`) and is linked from
here rather than duplicated.

**Ground truth is what's committed.** A rerun is a *check against*
already-committed artifacts, not a replacement for them. See
`REVIEWER_RUNBOOK.md` for the full two-tier verification path — this
document is the reference table that runbook's checks are built from.

## Pin table

### Repository

| What | Value |
|---|---|
| TANDEM repo, `p22-closed` tag | see `git tag -l p22-closed` / `git log -1 p22-closed` after tagging |
| B1-B5 val-set manifest hash (baseline-to-baseline eval-set identity, `experiments/phase2_2_baselines/val_manifest_canonical.json`, from B1's real run) | `41ddeb0ebb2fea82` |

### Upstream (third-party baseline) repo commits — `experiments/phase2_2_baselines/commits.md`

| Baseline | Repo | Pinned commit (full SHA) |
|---|---|---|
| B1 | facebookresearch/ToMe | `af95e4b1befa172dadccd8c81e223b10090f9579` |
| B2 | zkkli/RepQ-ViT | `a160561b9233945e6680132f129d13041b787318` |
| B3 | megvii-research/FQ-ViT | `5daf5915c715dc21d6afcd64e661652030e640ab` |
| B4 | (uses B1 + B2's pinned commits together; no separate repo) | — |

`common/repo_utils.ensure_repo(...)` clones/fetches and hard-checks-out
the exact SHA above, and asserts the checkout matches before any run
proceeds — a commit bump without updating this table is caught at
runtime, not silently absorbed.

### Checkpoint tags (resolved, verified identical across P2.1 and baselines)

| Model | Resolved pretrained tag | P2.1 FP top-1 reference |
|---|---|---|
| `deit_tiny_patch16_224` | `deit_tiny_patch16_224.fb_in1k` | 72.166% |
| `deit_small_patch16_224` | `deit_small_patch16_224.fb_in1k` | 79.848% |
| `deit_base_patch16_224` | `deit_base_patch16_224.fb_in1k` | 81.970% |
| `vit_base_patch16_224` | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 85.100% |

Cross-checked, exact match, both directions: P2.1's own
`P2_<model>_summary.json.resolved_pretrained_tag` vs. baselines'
`common/checkpoint_pin.py.PINNED_TAGS` (`scripts/verify_baselines_claims.py`
check via `common/bops_identity_check.check_c_id_matches_canonical_summary`
+ `assert_fp_references_match_canonical_summaries`, both bit-for-bit).
**ViT-B is the one model where a method's own unmodified checkpoint
resolution does NOT coincide** with this pinned tag (RepQ-ViT/FQ-ViT's
own loaders resolve to `vit_base_patch16_224.augreg_in21k_ft_in1k` — no
`2` — a different, lower-accuracy augreg checkpoint) — this is why every
`results.csv` row carries its own `checkpoint_tag` and a `row_kind`
(`reproduction` = paper's own checkpoint, `comparison` = this pinned tag);
see `common/checkpoint_pin.py`'s module docstring and `deviations.md` #16
for the full two-step policy.

A raw model state-dict hash is computed and asserted at runtime in two
places (P2.1's `hash_state_dict`, used for weight-restoration
verification after every plan execution; `jetson/export_segments.py`'s
`checkpoint_state_hash`, written into `jetson/out/segments/manifest.json`
once run) — neither is persisted into a committed summary artifact today,
so it is not restated as a fixed value here; the resolved TAG above is
the persisted, checkable identity.

### Calibration index files

| Model | Source | Count | `calib_hash` |
|---|---|---|---|
| `deit_tiny_patch16_224` | imagenet-1k-train | 512 | `98f8b1e277d0` |
| `deit_small_patch16_224` | imagenet-1k-train | 512 | `98f8b1e277d0` |
| `deit_base_patch16_224` | imagenet-1k-train | 512 | `98f8b1e277d0` |
| `vit_base_patch16_224` | imagenet-1k-train | 512 | `98f8b1e277d0` |

(All four share one hash — the same 512 image identifiers were drawn once
at the dataset level, independent of model architecture.) Recorded in
`experiments/phase2_1_fleet/results/P2/<model>/calibration_indices.json`. Baselines B2/B3
do not cache-and-verify the same way — `common/calib_capture.py`
*deterministically reconstructs* RepQ-ViT's/FQ-ViT's own internal
calibration selection by replaying their exact `seed()` →
`ImageFolder(train)` → `RandomSampler` sequence (proof of exactness in
that module's own docstring); B4 calibrates via RepQ-ViT's own
`build_dataset` train loader directly, in-process. See
`experiments/phase2_2_baselines/REPRODUCIBILITY.md` for the full per-baseline calibration
narrative.

### Tolerance registry

| Gate | Tolerance | Reading | Answers |
|---|---|---|---|
| P2.1 S8 sanity gate | 0.5pp | flat | Does this run's FP top-1 match the published reference? |
| Baselines FP-parity gate | 0.15pp | flat | Does this baseline's own plain forward pass match P2.1's own S8 measurement (harness soundness)? |
| **Baselines reproduction gate** | **0.30pp** | **flat/absolute** (see ruling below) | Did we reproduce the paper's published number? |
| P2.1 accuracy run-to-run jitter | ±0.15pp | (not a gate — measured GPU nondeterminism band) | — |
| Phase-1↔P2.1 Dq/Dm bridge fingerprint | ratio ∈ [0.9, 1.1] | 46/48 in-band, median ratio 1.0031 | Does the P2.1 toolchain reproduce Phase-1's own measurements? |

**Tolerance standardization: 0.30 percentage points, flat.**
`common/repro_gate.py`'s module docstring reads "±0.3%",
inherited verbatim from the project's original spec wording — ambiguous
between **flat/absolute** (`abs(measured_pct - published_pct) <= 0.3`,
what is actually implemented: `GATE_TOLERANCE_PCT = 0.3` in `check_gate()`)
and **relative** (`abs(measured_pct - published_pct) / published_pct <=
0.003`, NOT implemented). The two readings diverge for one real row
(`repq_vit / deit_tiny_patch16_224 / w6a6`, diff=0.228pp vs published
70.76%: PASSES flat (0.228≤0.3), would FAIL relative (0.322%>0.3%)) —
this project has always gated and reported it as a PASS (flat reading),
recorded here so the ambiguity in the original spec text is resolved on
paper, not just in code. Full derivation: `experiments/phase2_2_baselines/REPRODUCIBILITY.md`.

### Cost-ruler correction: the merge cap

`common/bops.py`'s (and its ports in `scripts/verify_p2_claims.py`,
`jetson/replan.py`) per-block token-merge cap is
**`min(r, floor((N-1)/2))`** — the requested merge count `r`, capped by
the maximum pairs `bipartite_soft_matching`/the Ward merge can physically
form once the CLS token is excluded from pairing (`N-1` mergeable tokens,
`floor((N-1)/2)` pairs). An earlier spot-check had proposed the
CLS-unaware variant `floor(N/2)` (no `-1`); that variant is superseded —
the code was always correct (verified against a live forward-hook
cross-check on real 50k GPU runs at r=8/13/16), only the stated formula
needed correcting for the record. Full derivation: `docs/deviations.md`
item #32's addendum.

## Per-phase verification (Tier 1 — no GPU)

| Phase | Command | Expected |
|---|---|---|
| E1-E5/E4a (Phase 1, G1 kill test) | `python scripts/verify_g1_claims.py` | ALL CLAIMS VERIFIED (5 checks) |
| P2.1 (4-model fleet) | `python scripts/verify_p2_claims.py` | **20/20 MATCH** (5 checks × 4 models) |
| P2.2 (B1-B5) | `python scripts/verify_baselines_claims.py` | ALL CLAIMS VERIFIED (6 checks) |
| G1-HW (Device A static self-test) | `(cd jetson && python self_test.py)` | 11/11 checks passed |
| G1-HW-2 (Device B headline + static) | `(cd jetson-orinnx && python verify_orinnx_claims.py && python self_test.py)` | ALL CLAIMS MATCH + 11/11 passed |

Run all three (plus the cost-ruler unit checks and B4/B5 closure
re-derivations) via `bash scripts/run_reviewer.sh` — see `REVIEWER_RUNBOOK.md`.

## P2.1 detail: what reproduces exactly, and the Phase-1 bridge fingerprint

**Tables (Dq/Dm/plans/familyX, `D_pred`, `C_id`/`C_min`, gate verdicts)**:
exact (bit-for-bit on deterministic quantities), given pinned calibration
(the 512-image set above, reused/asserted by default) and pinned
environment (`torch`/`timm`/`datasets` versions matching the canonical
summary — `assert_environment_matches_canonical`, overridable via
`TANDEM_ALLOW_ENV_DRIFT=1` for deliberate exploration, which marks the
run explicitly uncertified rather than silently different).

**Without pinned calibration (F-calib, documented and closed)**: a
different 512-image draw moves calibration-sensitive Dq entries by up to
~1.8x relative to the pinned value — a real, characterized sensitivity of
per-block activation-scale calibration, not a bug (it's exactly why
`calib_hash` is folded into every config's `config_hash`). Plan-level
per-budget fidelity ratios are calibration-*robust* (reproduce to 3-4
decimal places across different calibration subsets).

**Accuracies (measured top-1)**: reproduce to **±0.15pp** under pinned
calibration+environment — ordinary GPU floating-point nondeterminism
(non-associative fp16/fp32 reduction order, cuDNN algorithm selection),
not a reproducibility defect. Every gate's own tolerance band sits well
outside this jitter specifically so it never flips a verdict.

**Gate verdicts**: identical, always, under pinned calibration+environment.
A different verdict on a pinned rerun is a real regression, not expected
noise.

**Phase-1↔P2.1 bridge fingerprint** (`deit_small_patch16_224`'s
`phase1_regression` field, comparing P2.1's freshly-measured Dq/Dm tables
against Phase 1's own independently-measured E3/E4 raw CSVs): **46/48
entries in-band** (`[0.9,1.1]` ratio), **median ratio 1.0031**. Two named
out-of-band cells (`block=1,bits=8`: 1.1180; `block=2,bits=6`: 0.7954) are
interpolation-sensitive single-cell outliers, reported not hidden. If a
rerun produces a materially different picture, stop and investigate
before trusting anything else from that run.

**Reproducing from scratch (not the default)**: `FORCE_FRESH_RUN = True`
ignores existing checkpoints; delete `calibration_indices.json` for a
fresh calibration draw; `TANDEM_ALLOW_ENV_DRIFT=1` / `TANDEM_INTERACTIVE=1`
as above. None of these change any gate's pass/fail logic.

## P2.2 (baselines) and G1-HW (jetson) detail

See **`experiments/phase2_2_baselines/REPRODUCIBILITY.md`** for: the two-gate design
(FP-parity vs reproduction), determinism (TF32 disabled on every GPU op,
both flags, every runner), environment hashing (`equivalence_gate.py`,
no override — deliberately stricter than P2.1's own), and the
baseline-to-P2.1 eval-set identity argument (no literal manifest hash
exported by P2.1 itself — two other mechanisms cover it instead, detailed
there).

See **`jetson/README.md`** and **`jetson-orinnx/README.md`** for G1-HW and
G1-HW-2 pinning (checkpoint tag + state-dict hash, ONNX opset, TensorRT/CUDA/L4T
versions via `out/versions.json`) and the documented bits→hardware-precision
mapping (bits∈{8,6} both execute as measured INT8 latency — no native INT6
datapath on this hardware).
