# TANDEM

Training-free joint token and bit-width allocation for Vision Transformers.
Target venue: CVPR 2027.

TANDEM builds a theoretical model of end-to-end distortion for compressed
ViTs, `D ~ Sum_l S_l * eps_l`, and uses it to jointly allocate per-block
bit-widths and token-merge budgets without any fine-tuning. **Headline
finding**: allocating jointly (bits and tokens together) beats every
tested alternative — a matched-cost proxy allocator, four published
external methods, and a strong staged/sequential baseline — and the
advantage survives intact when cost is switched from a simulated proxy to
latency measured on real edge hardware -- on two independently measured
Jetson Orin modules (Orin Nano Super and Orin NX 8GB), whose measured
proxy error agrees within 1%.

## Results

Every number below is reproducible from a committed artifact — see
`docs/REPRODUCING.md` Tier 1 for the exact command and expected output.

| Claim | Value | Proof |
|---|---|---|
| Registered hardware gate (BOPs-selected plan vs JOINT allocator, measured latency) | **+0.88pp** [95% CI +0.78, +0.98], PASS, n=120,000 paired predictions | [`jetson/out/paired_eval/paired_eval_result.json`](jetson/out/paired_eval/paired_eval_result.json) |
| TANDEM vs uniform-r ToMe, matched hardware cost | **+0.31pp** [95% CI +0.18, +0.44], n=80,000 paired predictions | [`jetson/out/frontier_v2/frontier_result_v2.json`](jetson/out/frontier_v2/frontier_result_v2.json) |
| BOPs-to-latency proxy error (N=69 vs N=197, FP16, MAXN_SUPER) | **2.235x** — persists (2.19-2.25x) across all 3 tested power modes | [`jetson/out/energy/proxy_divergence.csv`](jetson/out/energy/proxy_divergence.csv), [`evidence/00_CORRECTIONS.md`](evidence/00_CORRECTIONS.md) |
| Staged-composite (B4) accuracy saturation | **79.704%** top-1 at only **28.78%** of identity BOPs | [`experiments/phase2_2_baselines/compare_summary.json`](experiments/phase2_2_baselines/compare_summary.json) (`deit_small_saturation`) |
| Cost-proxy predictive power for energy | R²(BOPs->energy) = **0.061** vs R²(latency->energy) = **0.939** | [`jetson/out/energy/proxy_divergence_summary.json`](jetson/out/energy/proxy_divergence_summary.json) |
| INT8 precision axis, all 3 power modes, all 5 token buckets | INT8 slower than FP16 in **15/15** cells — axis domination confirmed | [`jetson/out/powersweep/precision_axis_by_mode.csv`](jetson/out/powersweep/precision_axis_by_mode.csv) |
| **Second device** — registered gate reproduced on Jetson Orin NX 8GB | **+0.93pp** [95% CI +0.82, +1.03], PASS, n=120,000; every measured-cost plan is identity (all 6 BOPs plans slower than doing nothing) — a proxy-divergence result, per pre-registration | [`jetson-orinnx/out/paired_eval/paired_eval_result.json`](jetson-orinnx/out/paired_eval/paired_eval_result.json) |
| Second device — proxy error reproduces | **2.252x** (canonical LUT; 2.258x on independent remeasurement) vs 2.235x on the Nano — within 1% | [`jetson-orinnx/out/energy/proxy_divergence.csv`](jetson-orinnx/out/energy/proxy_divergence.csv), [`jetson-orinnx/FINDINGS.md`](jetson-orinnx/FINDINGS.md) |
| Second device — INT8 slower 5/5; energy R²(BOPs) = **0.024** vs R²(latency) = **0.992**; INT8 costs 1.16–1.64x the energy | [`jetson-orinnx/out/lut/latency_lut.json`](jetson-orinnx/out/lut/latency_lut.json), [`jetson-orinnx/out/energy/proxy_divergence_summary.json`](jetson-orinnx/out/energy/proxy_divergence_summary.json) | |
| Cost tables do not transfer between devices | Achievable ranges disjoint (NX 2.86–3.72 ms vs Nano 3.80–4.97 ms); Nano-table plans infeasible at **4/4** NX budgets; only the cost table needs re-measuring | [`jetson-orinnx/out/transfer/transfer_experiment_result.json`](jetson-orinnx/out/transfer/transfer_experiment_result.json) |

Every row is independently re-derivable with no GPU via
`scripts/verify_p2_claims.py` (20/20 MATCH), `scripts/verify_g1_claims.py`
(5/5 MATCH), `scripts/verify_baselines_claims.py` (6/6 MATCH),
`jetson/replan.py --verify-port` (6/6 MATCH), and, for the second device,
`jetson-orinnx/verify_orinnx_claims.py` (all headline numbers and cross-file
consistency MATCH) — see `docs/REPRODUCING.md`.

## Repository map

```
README.md              this file
LICENSE                Apache 2.0
CITATION.cff           how to cite this work
requirements/          one reference copy of each environment's pinned deps (NOT the live
                        install source -- see requirements/README.md for which file is live)
tandem/                pointer/map from method concept -> real implementation (no code lives
                        here; see tandem/README.md)
experiments/
  phase1_E1-E5_E4a/     Phase 1: per-block sensitivity (E1), additivity (E2), real-quantization
                        transfer (E3), token-merge damage (E4), the Ward-merge proof (E4a), and
                        the joint-vs-staged "kill test" (E5/G1) -- one script/notebook pair +
                        results/ per experiment (jupytext-paired .py/.ipynb, .py is the source)
                        -- see phase1_E1-E5_E4a/README.md for the E5 kill-test nuance in particular
  phase2_1_fleet/       P2.1: TANDEM's own allocator run for real on 4 architectures
                        (DeiT-Tiny/Small/Base, ViT-Base) -- P2_tandem_pipeline.py + results/P2/
                        -- see phase2_1_fleet/README.md for per-model headline numbers
  phase2_2_baselines/   P2.2: 4 published external methods (ToMe, RepQ-ViT, FQ-ViT, a staged
                        ToMe->RepQ-ViT composite) reproduced on TANDEM's own protocol and
                        matched-cost BOPs axis, plus the cross-baseline comparison (B5)
                        -- see phase2_2_baselines/README.md
jetson/                 G1-HW: Jetson Orin Nano Super hardware validation -- self-contained,
                        real measured-latency replan + paired evaluation + a 3-power-mode sweep
                        -- see jetson/README.md
jetson-orinnx/          G1-HW-2: the same campaign mirrored byte-for-byte onto a Jetson Orin NX
                        8GB (README.md there is deliberately the verbatim Nano copy; start at
                        README_ORINNX.md, DEVICE_PROFILE.md, FINDINGS.md, PORT_NOTES.md).
                        Adds the cross-device transfer experiment and verify_orinnx_claims.py
evidence/               the manifest/hash/claim-to-file mapping that makes every number above
                        auditable -- see evidence/00_MANIFEST.md
docs/                   REPRODUCING.md, RESULTS.md, GLOSSARY.md, deviations.md,
                        CHARACTERIZATION.md, REPRODUCIBILITY.md, REVIEWER_RUNBOOK.md, archive/
results/                pointers to the canonical result tables above (no duplicated data)
scripts/                repo-wide, read-only, no-GPU verification scripts + run_reviewer.sh
LAB_LOG.md              running experiment log, chronological, unedited
```

## Quickstart (no GPU, ~5 minutes)

```bash
git clone https://github.com/fawadsarim/tandem.git
cd tandem
pip install pandas numpy
bash scripts/run_reviewer.sh          # Tier 1: 8 read-only checks against committed artifacts
```

Expect `TIER 1: ALL 8 STEPS PASSED.` See `docs/REPRODUCING.md` for the
exact expected output of each step and what a real failure vs. a
legitimate deviation looks like.

## Full reproduction (GPU / hardware required)

`docs/REPRODUCING.md` Tier 2 gives the exact stage order (E1..E4 -> E4a ->
E5 -> P2.1 -> B1..B5 -> jetson), per-stage commands, environments, expected
numbers with tolerance, and realistic runtimes (minutes for the no-GPU
stages, hours for a full baseline sweep, real device time for jetson/).
Gate semantics are central and documented there explicitly: an **FP-parity
gate failure HALTS** a run before any real number is trusted; a
**reproduction-gate failure is RECORDED and the run continues** (the
deviation itself becomes part of the record, not a silent retry).

## Hardware scope

All hardware claims in this repo were measured on **two** Jetson Orin
modules of one product family, both on JetPack 7.2 (L4T R39.2), TensorRT
10.16.2, CUDA 13.2:

- **Jetson Orin Nano Super Developer Kit** (`jetson/`), at up to **three**
  nvpmodel operating points (15W / 25W / MAXN_SUPER). This unit's real
  GPU/EMC clock ceilings are narrower than those mode names imply — see
  [`evidence/00_CORRECTIONS.md`](evidence/00_CORRECTIONS.md) item c.
- **Jetson Orin NX 8GB** (`jetson-orinnx/`, on the same carrier), at
  **one** operating point (MAXN). Its lower modes could not be characterised
  cleanly because the vendor tool's configured GPU power-gating mask is not
  applied and drifts — a measured device property, documented in
  [`jetson-orinnx/DEVICE_PROFILE.md`](jetson-orinnx/DEVICE_PROFILE.md) and
  `PORT_NOTES.md` §6/§16. Every NX measurement is gated on *achieved*
  clocks and mask state. On the NX the frontier-vs-ToMe arm and the energy
  re-plan were not run, engine-level FP16/INT8 numerical parity is recorded
  unresolved (`PORT_NOTES.md` §9), and joint-vs-staged is untested because
  every pre-registered budget is degenerate there (`FINDINGS.md` §6).

No claim of a wide power range or clock-ratio sweep should be inferred
beyond what those documents state. The divergence *mechanism* reproduced
across the two modules within 1%; magnitudes on other hardware require
re-measuring the cost table, which is by design the only device-specific
artifact (`jetson-orinnx/FINDINGS.md` §4).


## Known limitations

Every documented deviation from a published baseline's exact protocol,
with justification, is in [`docs/deviations.md`](docs/deviations.md).
Numeric corrections found during the hardware power-mode sweep (a proxy-
error figure and a TFLOPS figure that had circulated with the wrong
denominator) are in [`evidence/00_CORRECTIONS.md`](evidence/00_CORRECTIONS.md).
Gaps in the evidence package — what's missing, why, and how to regenerate
it — are in [`evidence/00_MISSING.md`](evidence/00_MISSING.md).

## LLM usage statement

AI coding assistants (Claude Code) were used to implement experiment
scripts and analysis code from detailed human-written specifications; AI
assistance was also used for editing and phrasing of the manuscript. All
research questions, algorithms, experimental designs, acceptance criteria,
and scientific rulings are the authors'. Every reported number was
independently verified: headline fits recomputed from raw artifacts,
allocator outputs reproduced by an independent implementation, and the
full experimental suite re-run end-to-end under supervisor direction. A
development-time defect ledger (issues caught by verification gates, none
reaching a reported number) is documented throughout `LAB_LOG.md` and
`docs/deviations.md`. The authors take full responsibility for all
content.

## How to cite

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{sarim2026tandem,
  author = {Sarim, Muhammad Fawad Ul Hassan},
  title = {{TANDEM: Training-Free Joint Token and Bit-Width Allocation for Vision Transformers}},
  year = {2026},
  url = {https://github.com/fawadsarim/tandem},
  license = {Apache-2.0}
}
```
