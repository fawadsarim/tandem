# Glossary

Terms used throughout this repo's code, docs, and commit history, in the
sense specific to this project.

**S_l, eps_l** — the per-block sensitivity constant (`S_l`) and injected
distortion-energy term (`eps_l`) in the project's foundational model,
`D ~ Sum_l S_l * eps_l` (README.md's opening formula): `S_l` is fitted in
E1 from how much output distortion a unit of injected Gaussian noise
energy produces in block `l`'s own regime (linear in that regime — E1's
own headline finding); `eps_l` is the actual distortion energy a given
compression choice (a bit-width, a merge schedule) injects into block
`l`. Summing `S_l * eps_l` over blocks is what E2 validates as additive —
see `experiments/phase1_E1-E5_E4a/README.md`.

**Block** — one of the 12 transformer blocks (LayerNorm → Attention →
residual → LayerNorm → MLP → residual) in a DeiT/ViT model. TANDEM
allocates bit-width and token count independently per block.

**BOPs (Bit-Operations)** — the simulated compute-cost proxy used
throughout Phase 1/P2.1/P2.2:
`12*N_l*d²*(bits²) + 2*N_l²*d*(16²)` per block (linear-layer term +
attention term, attention always priced at FP16 regardless of the block's
own bit-width). Defined in `experiments/phase2_2_baselines/common/bops.py`
and ported verbatim into `jetson/replan.py`'s `--verify-port` mode. Not a
real hardware measurement — see **proxy divergence** below for how far it
diverges from one.

**Plan** — a full allocation: a per-block bit-width choice (from the bit
menu, typically `{16, 8, 6}`) plus a token-merge schedule (which blocks
merge how many tokens). The unit the Pareto DP allocator selects.

**D_pred** — the predicted total distortion of a plan,
`D_quant + kappa * D_merge`, built from per-block damage tables
(`Dq_table.csv`, `Dm_table.csv`) established in Phase 1 (E1-E4). The
allocator's objective function — never the real measured accuracy loss
directly, always a prediction validated against it downstream.

**C_id** — the BOPs cost of the identity plan (all-FP16, no merging) —
the most expensive point in the plan space, used to define budget
fractions (`budget = fraction * C_id`).

**Pareto DP** — the dynamic-programming allocator that, for a given cost
budget, finds the plan minimizing `D_pred` across the full joint
(bit-width x token-schedule) plan space — not a greedy or per-axis search.
Ported from `experiments/phase2_1_fleet/P2_tandem_pipeline.py` into
`jetson/replan.py` (cost axis substituted from BOPs to measured latency)
and independently re-derived from scratch in `scripts/verify_g1_claims.py`
/ `scripts/verify_p2_claims.py`.

**Staged / BEST-SEQUENTIAL** — the strongest sequential baseline the
joint allocator is measured against: pick tokens first then bits, or bits
first then tokens (whichever order+threshold combination performs best),
rather than optimizing both axes together. The comparison the entire
project's central claim rests on. **Not the same thing as "the staged
composite" (B4)** in P2.2 (`experiments/phase2_2_baselines/`) — B4 is an
unrelated, specific two-method pipeline (ToMe token-merging applied, then
RepQ-ViT quantization applied on top) reproduced from published baselines,
not TANDEM's own bits-vs-tokens ordering baseline. Both use the English
word "staged" for "one axis/method applied after another," but they are
different baselines answering different questions — see
`experiments/phase2_2_baselines/README.md` for B4 specifically.

**Staging collapse** — the Phase-1 (E5/G1) finding that, under simulated
BOPs cost, the JOINT allocator's plan is *identical* to the best staged
plan at every tested budget (`plans_identical=True`, mechanism:
`exact_tie_via_plan_collapse`) — a real, structural consequence of
`D_pred` being additively separable whenever the joint-optimal schedule is
also token-frontier-efficient. Investigated further on real hardware in
G1-HW (see **axis domination**).

**Axis domination** — the G1-HW finding (formalized in
`docs/CHARACTERIZATION.md`) that on measured latency, the precision axis
(bit-width choice) contributes negligibly to real cost savings compared to
the token-merge axis — INT8 measures *slower* than FP16 in every tested
configuration on this hardware (15/15 mode-bucket cells in the power-mode
sweep), so a joint allocator's advantage over a token-only staged
allocator collapses to near-zero on this device, not because joint
allocation doesn't help in principle, but because one of its two axes is
structurally dead on this specific hardware/precision-kernel combination.

**Proxy divergence** — how far a simulated cost proxy (BOPs) diverges
from a real, measured cost (latency, energy) as a *ratio*, not just in
absolute terms: `(measured_ratio) / (BOPs_ratio)`, relative to the
N=197/FP16 reference point. Measured at 2.235x for latency (N=69 vs N=197)
at MAXN_SUPER, persisting (2.19–2.25x) across all three tested Jetson
power modes.

**Registered gate** — a pre-declared pass/fail comparison whose criteria
were fixed before the measurement that resolves it, e.g. G1-HW's
BOPs-selected-vs-JOINT paired comparison (registered: mean delta ≥ +0.4pp
AND CI excluding zero → PASS — the threshold is `GATE_MEAN_THRESHOLD_PP`
in `jetson/paired_eval.py`, also stored in the result JSON's own
`gate_mean_threshold_pp` field). Distinct from a **reproduction gate**
(compares a fresh measurement against a published/prior number within a
tolerance) — see `docs/REPRODUCING.md`'s gate-semantics section for how
the two behave differently on failure.

**FP-parity gate** — the check that a freshly-loaded baseline model
reproduces its own published (or TANDEM-pinned) full-precision accuracy
before any compressed number from that load is trusted. Failure HALTS the
run — see `docs/REPRODUCING.md`.

**Reproduction gate** — per-config-fraction, the `results.csv` check that
compares a baseline's own reproduced number against its documented
published reference. Failure is **recorded**, not halted — the deviation
becomes part of the record (`docs/deviations.md`), not a crash.

**G1-HW** — this project's hardware-validation phase (the `jetson/`
folder): does the plan selection made under simulated BOPs cost still win
once cost is *measured* latency on a real edge device (Jetson Orin Nano
Super)?

**G1-HW-2** — this project's second-device hardware-validation phase (the
`jetson-orinnx/` folder): replicating the proxy divergence, INT8 latency/energy
slowdown, and cross-device cost transferability on physical Jetson Orin NX 8GB hardware.

**S8 sanity gate** — a per-model check that P2.1's own measured
full-precision top-1 accuracy is within tolerance of that model's
published reference accuracy, before any compressed-plan result for that
model is trusted.

**B1–B5** — the five items in P2.2 (`experiments/phase2_2_baselines/`),
each reproducing a published external method on TANDEM's own protocol
and matched-cost BOPs axis, plus a final cross-baseline comparison:
- **B1 — ToMe**: Token Merging (Bolya et al.), a published training-free
  token-reduction method — this project's own uniform-r ToMe comparison
  point (see README's "TANDEM vs uniform-r ToMe" result row).
- **B2 — RepQ-ViT**: a published post-training quantization method for
  ViTs.
- **B3 — FQ-ViT**: a second published post-training quantization method
  for ViTs.
- **B4 — the staged composite**: ToMe (token merging) applied first,
  then RepQ-ViT (quantization) applied on top — a sequential two-method
  pipeline, not TANDEM's own bits-vs-tokens **Staged / BEST-SEQUENTIAL**
  baseline (see that entry above for the disambiguation).
- **B5 — the cross-baseline comparison**: pulls B1-B4's own results
  together (`compare_pareto.py` → `b5_final_closure.py` →
  `compare_summary.json`) into one matched-cost comparison against
  TANDEM's own P2.1 allocator — no new measurement, a rollup.

**Quantizer-orthogonality finding** — the P2.2/B4 observation (see
`docs/deviations.md` item 2 and `experiments/phase2_2_baselines/compare_summary.json`)
that RepQ-ViT's own quantization damage, measured standalone vs. measured
after ToMe's token-merging has already been applied, is close enough to
call the two compression axes (merging, quantization) independent in
their effect on accuracy for this composite — i.e., quantization damage
doesn't materially change once tokens are already merged. A supporting
finding for B4's staged-composite result, not a headline claim on its own.

**Paired predictions / paired comparison** — this project's statistical
comparison method for two plans (e.g. BOPs-selected vs JOINT, or TANDEM
vs uniform-r ToMe): both plans are evaluated on the *same* held-out image
at the *same* budget, so `delta_top1` is computed per-image (plan A
correct/incorrect vs plan B correct/incorrect on that one image) before
being pooled and averaged — reducing variance versus comparing two
independent accuracy numbers, and what makes a tight confidence interval
possible at these sample sizes (n=120,000 / n=80,000 pooled pairs). See
`jetson/paired_eval.py`'s own docstring for the exact procedure.
