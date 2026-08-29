# TANDEM Lab Log

Running log of experiments. One entry per run. Append-only — do not edit past
entries except to fix typos. Config hash = short hash of the CONFIG dict used
for that run (printed by each notebook's setup cell).

---

## Phase 1: Foundations & Analytical Proofs (E1–E5, E4a)

### E1: Sensitivity & Linearity on DeiT-Small | PASS
Ran: E1 sensitivity/linearity, DeiT-S, imagenette-full fallback (3,925 imgs), T4, cfg 5cb154827e20, 12.6 min.
- S_l (logit-MSE) spans 0.376–3.90; block 12 most / block 4 least sensitive; R² ≥ 0.99 all blocks (bar was 0.9).
- Linear to eps=0.03 for 11/12 blocks; block 1 superlinear past 0.003; block-12 KL saturates past 0.003.
- Advisor findings: canonical currency = ABSOLUTE energy, S_abs = S_rel/m_l ≈ [2.02,0.81,0.68,0.60,0.62,0.65,0.71,0.68,0.59,0.54,0.62,0.81]
  → near-flat 0.66±0.09 for blocks 2–12, block 1 is 3x outlier (hypothesis H-flat, to be tested on ImageNet + other models).
  Design rule: block 1 conservative in all plans. Primary distortion metric confirmed: logit-MSE.
- Open: E1b re-run on ImageNet-val when access granted. Code verified by advisor (gates/assertions present and correct).
- Next: E4a Ward derivation (pen & paper) + E2 additivity.

### E1b: ImageNet-1k Validation Rerun | PASS
Progress report v1.0 (E1/E1b/E2) compiled and archived; advisor-verified numbers.
- E1b (ImageNet) PASS; E1b constants established as canonical.
- H-flat refined to "flat band 0.76±0.09 with shallow-U edges, block 1 ≈2.6×"; sensitivity level noted distribution-dependent (±15%) → calibration-per-deployment framing added.

### E2: Multi-Block Additivity on ImageNet | PASS
Ran: E2 additivity, DeiT-S, ImageNet-1k val, 129 multi-block perturbation combinations, T4.
- E2 PASS (slope 1.007, R² 0.9987, ImageNet). T1 foundation complete.
- Allocator spec amended: per-block regime caps as hard constraints; mandatory one-pass plan audit.
- Second-order structure recorded: adjacency-localized coupling (7× effect).

### E2: Notebook Bug Fix & In-Artifact Correction
- E2 rerun: fixed silent all-NaN in summary 'correlations' (one NaN input poisoned np.corrcoef; now drops NaN rows first).
- New values 0.621/-0.359/-0.073 match the advisor's prior manual recompute exactly.
- Acceptance fit, interactions, config hash unchanged — science unaffected; artifact now self-consistent (no manual-correction step needed).

### E3: Real Quantization Curves & Cross-Block Transfer | PARTIAL
Ran: E3 quantization curves + transfer, DeiT-S ImageNet val, 158 configs, T4, cfg 2324ce485d52, 30.1 min.
- Activation quantizer = per-tensor absmax (percentile-clip design rejected, see prior). Gates a/b/c/d all PASS (c on A_6 mean metric, top1_agreement 0.9996).
- V1 (scaling) PASS: weight-only slope −2.0..−2.1, R²≥0.999 (clean c·2^-2b). Activation/WA slope −1.6..−1.9 (heavier-tailed — real ViT-outlier finding).
- V2 (transfer) 9/12 in band (bar 10/12): mean log-T −0.057 → Gaussian-calibrated S_abs predicts QUANT damage to ~5% median bias. Spread std 0.27; blocks 7/8/9 (W) out of band → block-dependent transfer correction needed, not model failure.
- V3 (additivity) slope 0.118: NOT a bug (D_pred verified to use joint energies). Initial DISCOVERY interpretation: single-block quant energies appeared NON-COMPOSABLE (upstream quant inflates downstream injected energy up to ~30x; sum-of-singles over-predicts joint distortion 5–10x).
- Consequence: allocator (E5) must model cross-block coupling; cannot assume single-block additivity of quant error. Follow-ups queued.

### E3: Metric Audit & Resolution | PASS
- E3 audit: V3 metric mis-specified (accumulated vs injected energy, advisor spec bug); corrected V3 PASSES (damage additivity under real quant, slope 1.118/R² 0.995, 11-block joint within 1.6%).
- v1.5's "non-composability" interpretation retracted.
- Propagation-amplification finding (30–86×, damage-invariant) added.
- Allocator quant axis switched to measured D_l(b) tables; E4 gains a mandatory merge-transfer test; E3b run cancelled as redundant.
- E3 closed PASS (V2→informative per measured-D_l(b) design).

### E4a: Machine-Verified Ward Merge Distance Bound | PASS
- E4a theorem package accepted (machine-verified incl. blind 0.5 scipy constant).
- notes.tex two fixes mandated (amssymb; stepwise-vs-global optimality wording).
- E4 launched with measured-damage-table gating and marginal-injection predictors. Ownership Q1–Q3 gate E4 review.

### E4: Retrospective & Issue Resolution | PASS
- E4 retrospective: 5 issues resolved — 2 coding bugs (state leak at short batch; autocast fp16 downcast), 1 coverage gap (bipartite guard), 1 advisor spec error (197→196), 1 scientific finding (merge-transfer anisotropy).
- Zero contaminated numbers; all intercepted at gates.
- Advisor's bipartite-telescoping statement corrected on record.
- E5 scope note: token schedules restricted to measured F_S grid.

### E4 / E4b: Schedule Additivity & Anisotropic Transfer | PASS
- E4 closed PASS via E4b: schedule additivity slope 1.149/R² 0.999 from measured damages, advisor-verified.
- V2→informative: Gaussian→merge transfer is anisotropic (deep merges benign, ≤8× vs isotropic).
- Both axes now use measured damage tables (DD7). Unified distortion model validated end-to-end across synthetic/quant/merge.
- Super-additivity correction: introduced kappa = 1.15 for multi-site merging (M=2).

### E5: Pre-Flight & Ownership Checkpoints | PASS
- E5 prompt issued (allocator + Family X cross-axis + G1 kill test; Kim-et-al-compliant sequential baselines; 2 STOP checkpoints).
- L0.1 accepted; rulings: max-over-orders implemented, Def.6 absorbed for §6.8, L0.2 (LampQ) assigned.
- Q1–Q3 ownership PASSED (advisor: no follow-up; two defense refinements gifted).

### E5: STOP-1 Clearance & Constraint Formulation | PASS
- STOP-1: 7/7 gates pass incl. executor reproduction; advisor caught unconstrained schedule space (infeasible/over-regime 2-point combos) via C_min=0.87% anomaly → Σr≤128 + feasibility patch, 265→85 schedules, budgets expected unchanged.
- STOP-1 cleared: advisor independently reproduced C_min and C_id from BOPs formulas to 5 sig figs (cost model + constraints verified exact); budgets unchanged; GO for Family X.

### E5: Gate (e) Duality Gap & Hull-Restricted Resolution
- Gate (e) duality gap: advisor spec error #3 (unsatisfiable exactness assertion); Claude Code's hull-restricted resolution ratified (Everett/Shoham–Gersho).
- Gate re-scoped to soundness+hull-completeness (both asserted); symmetric hull restriction keeps G1 fair, bias conservative for H1.
- Utilization reporting + pre-committed exactness trigger added. GO for Family X.

### E5: STOP-2 Clearance & Menu Restriction | PASS
- STOP-2: V1_E5 initial FAIL traced to WA4 (out-of-regime 12/12 blocks, eps_eff 5–230× over boundary; advisor spec error #4 violating own D4).
- Advisor refit on legal menu: slope 0.962, R² 0.997 → additivity holds across axes.
- Menu restricted to {FP16,WA8,WA6}; D4 amended to validated-composition-territory criterion with evidenced WA6 exemption.
- Findings F-X1 (regime enforcement) + F-X2 (benign token-removal sub-additivity, anti-QUOTA sign, r=−0.64).

### E5: Exact Pareto Dynamic Program Mandate | PASS
- Gate-(e) re-scope ratified; utilization data (89–95% at 4/6 budgets) → exactness trigger pulled forward pre-emptively.
- Exact per-schedule Pareto DP mandated on BOTH arms (isolates staging as the sole joint-vs-sequential difference; closes duality gap; gate (e) gains full-frontier assertion).
- Consolidated with the Family-X menu ruling into one patch; kill test held pending STOP-2 re-verdict.

### E5: Pre-Flight Security Audit & Credential Hygiene
- E5 pre-flight: kill-test code audited — paired eval, shared-index bootstrap, both-orders×α sequential, source/label asserts, registered bar verbatim: all verified.
- SECURITY: hardcoded HF token found in cell 3 (second occurrence) → revoked, rotated to Secrets, override line deleted, repo grepped. Conditional GO issued.

### E5 / G1: Simulated-Cost Post-Mortem & Staging-Collapse Result | PASS (STAGING COLLAPSE)
- E5 post-mortem complete: three independent verifications (advisor CSV, CC solver reproduction, frontier-condition computation — True at 6/6 budgets).
- Staging-collapse result named and formalized as a CONDITIONAL finding: under additive BOPs cost, the joint optimum and best sequential baseline coincide exactly (plans identical 6/6, delta top-1 = 0.000pp).
- Corrected fidelity 0.9688/0.9971 (n=22).
- Ruling: no simulated-space enlargement; G1-HW on measured Jetson cost is the registered escalation — collapse result makes any HW margin causally attributable to cost structure.
- Phase-1 final report v3.0 compiled 0-error, advisor-verified.

---

## Phase 2: Fleet Scaling & Baseline Reproduction (P2.1 & P2.2)

### Jetson Day-0 Bring-Up & Setup
- JetPack 7.x flashed to NVMe (not SD), jtop/trtexec/power-modes verified, MAXN SUPER + locked clocks confirmed, toy engine timed, SSD image backed up. Versions recorded. Ready for G1-HW LUT harness.

### Phase 2 Launch: Multi-Model Fleet & Baseline Track
- Phase 2 launched in parallel with Jetson track: P2.1 multi-model pipeline (Colab) + P2.2 external baselines (uni GPU) issued; two-comparison framing recorded (literature-practice vs oracle-staged — never conflate).
- E5 patched rerun: bit-for-bit reproduction confirmed (plans identical 6/6, frontier 26/85, fits unchanged); no change to any Phase-1 conclusion.

### P2.1 Fleet: DeiT-Tiny Shakedown Run | PASS
- P2.1 code audited (cross-model dim-derivation clean; 196/12-block asserts are a scope-limit for the 4 target checkpoints, logged).
- DeiT-Tiny designated shakedown run on Colab — machinery verified empirically via its STOP-P2a + mini-X gate before scaling to DeiT-B/ViT-B on uni GPU.
- DeiT-Tiny STOP-P2a cleared: FP sanity 0.7274≈published; block-1 outlier, 2^-2b law (median 15.6x), and merge anisotropy all REPLICATE DeiT-S findings → parameterization verified on 2nd architecture.

### P2.1 Colab OOM Triage & Streaming-Loop Patch
- P2.1 50k stage OOM on Colab (and would OOM on 16GB) — root cause: full-split preload, not hardware.
- Streaming-loop patch issued (one-pass all-plan eval, batch 64). Tiny finishing on local 1060 6GB (mini-X gate + patched 50k); DeiT-B/ViT-B reserved for uni GPU with the fix.
- DeiT-Tiny FULL PASS advisor-verified: Family-X refit exact (6/8 ratios in [0.90,1.04]); cost model 4th hand-reproduction, first at d=192; plan-audit 0.98–1.09 all budgets; 5k≈50k fidelity validates working-set economy.

### P2.1 Fleet: S5 Rulings & Documented Exemptions | PASS
- S5 rulings: ViT-B PASS 1.002/0.990 → GO.
- DeiT-B 0.789/0.992 → documented exemption (6/8 ratios in [0.91,1.01]; X5 carries 77% fit weight; F-X2 at scale, conservative direction; WA6-exemption precedent) with S7 binding gate [0.85,1.15] as certification.
- Supplementary patch: run_status tri-state (pass / pass_with_exemption / fail) replaces overloaded overall_pass bool.

### P2.1 Credential Triage & Gated Authorization
- ImageNet-1k 403 = gated-dataset authorization (not token auth) — auth_check/whoami to confirm license re-acceptance; resume fleet on restore.
- THIRD credential exposure — token revoked, standing Secrets-only rule reaffirmed.
- P2.1 regenerated notebook audited and cleared: exemption machinery + run_status tri-state verified airtight.

### P2.1: DeiT-B S7 Binding Gate Diagnosis & Refinement Attempt
- DeiT-B S7 binding gate FAIL (1.216 @21%) — exemption void per its own terms, no bar moved.
- Advisor diagnosis arithmetic-exact: 15%/21% plans differ by block-1 WA8 alone (ΔD_pred=0.073142=table entry); in-context block-1 contribution 46% (outlier dilution by merges, F-X2 family) + κ under-correction at Σr=128 on Base (+21.6% naked).
- Measured-refinement protocol issued (3 probes → κ_B_128 + c1_merge, DeiT-B only).

### P2.1: ViT-Base Preprocessing Mismatch & Recovery | PASS
- ViT-B: S8 gate null-passed with no reference; measured 81.09 vs ~85 for augreg2 tag → preprocessing hypothesis (0.5-norm/bicubic/crop_pct 0.9 vs hardcoded DeiT transform).
- Model-derived transforms + loud-fail S8 patched; plain-model accuracy recovered: 81.09% -> 85.10% (vs. 85.108% published).

### P2.1: DeiT-B Refinement Re-Audit & Limitation Verdict | MEASURED-ONLY TIER
- DeiT-B refinement FAILED at re-audit (21% ratio 1.255) — root cause: c1 and κ are Σr-context-dependent, not constants (κ_B_128=1.598 measured vs advisor-predicted 1.2–1.3, prediction miss owned).
- One-round stopping rule executed → MEASURED-ONLY tier per pre-registration.
- Refined 15% plan audits 0.996 (mechanisms real); refined 21% plan +2.2pp better measured (direction right).
- Named limitation finding recorded: additive-objective fidelity is scale-sensitive on merge×quant coupling; measured allocation valid everywhere. G1-HW/DeiT-S unaffected.

### P2.1: Pipeline Unification & Standardized S0–S9 Execution
- Pipeline unification ruling: archived-model path deleted — one standard S0-S9 path for all four models (uniform provenance, thesis-aligned "one dial per model").
- Phase-1 bridge preserved as read-only regression diagnostic (fresh-vs-archived ratios, [0.9,1.1] flags); Phase-1 record itself untouched/frozen.

### P2.1: Calibration Drift Diagnosis & F-Calib Finding
- Rerun STOP-P2a (Tiny+Small): Differential diagnosis — Dq drift (block-2 +38/+79% at outlier blocks) vs Dm reproduction (±1.5%) diagnosed as unpinned calibration subset (F-calib finding).
- Calibration-pinning patch issued: pinned 512-image calibration set to eliminate resampling noise across runs.

### P2.1: Phase-1 Bridge & Fleet Certification | PASS
- Phase-1 bridge (Small, rerun): ACCEPTED — 46/48 in-band, median ~1.00; the two OOB entries reproduce IDENTICALLY across independent fresh runs (b1/WA8 1.118, b2/WA6 0.795) → reclassified as deterministic toolchain+bicubic fingerprint, not calibration noise.
- P2.1 FINAL fleet ACCEPTED: slopes identical to accepted ×4, S8 ×4 pass incl. augreg2/85.108, FP deltas ≤0.01pp, single snapshot, calibration pinning shipped.
- `verify_p2_claims.py`: 20/20 MATCH exit 0 ×4 models (Family-X, S8, 14 D_pred rows/model, C_id/C_min, per-budget fidelity). PHASE 1 + P2.1 FORMALLY FINALIZED.

### P2.2: Baseline B1 (ToMe) Setting Triage & Resolution | PASS
- P2.2 B1 (ToMe) started: off-the-shelf variant only; calibration checks N/A for B1; N_l = 197−r·(l−1) with flooring.
- B1 gate failure triaged: r13 −0.42pp traced to setting mismatch in published.json (79.36 was trained-with-ToMe checkpoint; off-the-shelf measured 78.94 stands as honest value).
- published.json adjudicated INCORRECT: corrected json issued; trained checkpoint reproduced at 79.230 vs 79.36 (0.130pp PASS). Fine-tuning recovery margin 0.292pp at r13 documented.

### P2.2: B1 Header Defect Fix & Manifest Freeze
- B1 ACCEPTED (advisor-verified): BOPs ×4 recomputed exact incl. r0 == C_id; N_l closed-form confirmed; gates FP-parity 0.022pp, r0-vs-79.8 0.026pp, trained-ckpt r13 79.230 vs 79.36 = 0.130pp all PASS; self-test 10/10, resume proved.
- DEFECT: committed results.csv stale 19-col header over 21-field rows → pandas silent MultiIndex shift, B5 would plot corrupted axes; migration+guard+manifest-equality-assert prompt issued.
- B1 hash `41ddeb0ebb2fea82` promoted to canonical for B2-B4 equality asserts.

### P2.2: Baseline B2 (RepQ-ViT) Reproduction | PASS
- B2 ACCEPTED: W6A6 gates 4/4 pass (0.026-0.228pp), W8A8 reported-not-gated, ViT-B two-checkpoint divergence executed per policy (augreg StepA / augreg2 StepB, parity pre-validated).
- Tolerance ruling: 0.30pp flat registered (spec said 0.3%; Tiny W6A6 marginal under relative — documented).

### P2.2: Baseline B3 (FQ-ViT) Checkpoint Loader Resolution | PASS
- B3 FP-parity halt DEFENDED: RepQ-pass/FQ-fail contrast on same checkpoint isolated FQ-ViT loader transform.
- B3 (FQ-ViT) ACCEPTED: ViT-B blocker resolved — root cause was WRONG BUNDLED CHECKPOINT (augreg-v1 npz loaded regardless of tag), NOT transform; checkpoint_pin override fixes it, plain-model 84.466→85.154, FP-parity 0.054pp PASS, 152/152 keys clean, corrected W8A8 83.526 internally consistent.
- Cost-ruler ViT-B rows correct (0.28 = W8A8 linears + FP16 attention, not naive 0.25). Disclosed 8/8/4 attention quantization deviation.

### P2.2: Baseline B4 (Sequential Staged Composites) Pre-Run Validation | PASS
- B4 composites prompt issued: single-source, DEFINING requirement = best-over-both-orders (merge-first AND quant-first, max at matched BOPs, both stored, comparison=best).
- B4 operating-point ruling: ACCEPT AS-IS (bit-widths {16,8,6} only).
- B4 pre-run deliverable VERIFIED: three cost-ruler reductions all MATCH (identity→C_id; r0-composite→pure-PTQ 3.59e11=B3; w16a16-composite→pure-ToMe 7.24e11=B1).

### P2.2: Baseline B4 15% Point & Order Flipping Finding | PASS
- B4 15% point COMPLETE: best-over-both-orders working — merge_first 77.526 / quant_first 77.404 both stored, best=merge_first, 0.122pp order-sensitivity.
- Composite sanity PASS (77.53 = −2.3pp at 15%, below either single axis = correct stacked-damage shape).
- Staged 77.53 vs TANDEM-joint 77.37 at 15% BOPs — staged marginally ahead, CONSISTENT with staging-collapse.

### P2.2: Baseline B4 Acceptance & Pathological 41% Point Analysis | PASS
- B4 ACCEPTED: both pre-registered checks PASS. Finding: optimal order FLIPS with budget (merge-first @15/21%, quant-first @29.3%).
- ANOMALY: 41% point r=22 pathological (late blocks at 4 tokens, 70.206 = 8.1pp below 57.2% point) → diagnosed to uniform-bit staged quantization lever spanning only 0–31% C_id (W8A8@r0=30.91%; W6A6=20.83%), forcing FP16+uniform-r merging at >=31%.
- Attention-skip flattery QUANTIFIED: B4 r2/w8a8 79.704 > B2 plain w8a8 79.652 despite more compression → >= +0.052pp floor bias favoring staged; disclosed.

### P2.2: Baseline B4 Staged Saturation Discovery | PASS
- Staged Pareto SATURATES at 29.3%/79.704 — 41/57.2/80% points all dominated by staged's own cheaper config (verified); supersedes per-budget deltas as the honest framing.
- Structural finding sharpened w/ derived 92.1/7.9 linear/attention split + W8A16=53.9% exclusion arithmetic.

### P2.2: Baseline B5 (Cross-Baseline Comparison) & Suite Closure | PASS
- P2.2 FORMALLY CLOSED: all section-D items verified (interp deltas 4-decimal match, r=23 BOPs exact + structural confirm at 68.096, fronts/counts/filters/provenance all pass).
- Cap-formula false alarm self-corrected (⌊(N−1)/2⌋ CLS-protected is correct).
- Quantizer-orthogonality finding recorded: RepQ beats TANDEM at matched BOPs at both its points ×4 models (quantizer-quality, orthogonal — limitation + composability note mandated).
- `verify_baselines_claims.py`: 6/6 MATCH. Phase 2.2 fully closed.

---

## Phase 3: Hardware Validation — Device A (Jetson Orin Nano Super)

### G1-HW: Jetson Package Audit & Port-Equivalence Proof | PASS
- jetson/ package audited (13 files, 2216 lines). Gaps caught and fixed: (1) Pareto DP port-equivalence proof, (2) plan-divergence logging.
- Port-equivalence gate VERIFIED GENUINE: compares parsed plan_json from real plans.csv on structural identity + C to 1e-6 rel + D_pred to 1e-9 abs. 6/6 MATCH → live subprocess check 10/11. PACKAGE CLEARED FOR HARDWARE.

### G1-HW: Hardware Bring-Up & On-Device Patching | PASS
- Jetson bring-up COMPLETE: MAXN-drop installer bug + fail-closed gate bug found on-device and fixed.
- Hardware configuration: Jetson Orin Nano Super Developer Kit (P3768 carrier, JetPack 7.x, L4T R39.2, TensorRT 10.16.2, CUDA 13.2, locked clocks: GPU 624.75 MHz, EMC 2133 MHz).

### G1-HW: First Latency LUT & Proxy Divergence Discovery | PASS
- FP16 token-axis non-additivity MEASURED: BOPs→latency proxy error grows to 1.84× at N=69 (later 2.235x on full sweep).
- INT8 2.0–2.5× slower than FP16 ×5 buckets due to fallback churn; RULED inadmissible pending strong-baseline remediation.

### G1-HW: INT8 Remediation Ladder (2a & 2b) | PASS (AXIS DOMINATION)
- 2a (--best search) abandoned INT8 onto FP16; ruled not sufficient to discharge strawman risk.
- 2b (ModelOpt explicit Q/DQ graphs): 4/4 GEMMs on INT8 tensor cores, 0 reformat layers, verified fusion.
- Finding ACCEPTED: INT8 remained 1.05x-1.34x slower than FP16 across all 15 mode x token cells. Quantize/dequantize traffic around small (197x384) matrices fails to amortize.

### G1-HW: Hardware Budget Pairing & Proxy-Inversion Finding | PASS
- Budget pairing rule: tau_b set to measured LUT latency of the BOPs plan (96.6–131.6% of Latency_id), exposing proxy failure where BOPs plans are slower than doing nothing.
- Root cause diagnosed: DeiT-S @ batch-1 achieves ~2.5 TFLOPS ≈ 15% of Orin FP16 peak; 34% of wall time is enqueue — device not compute-bound.

### G1-HW: Registered Hardware Gate Passed | PASS (+0.88 pp)
- REGISTERED GATE PASSED: BOPs-vs-JOINT +0.88pp, CI [0.78, 0.98], 6/6 divergent, +2.60 at 15% budget (120,000 paired ImageNet predictions).
- Characterization Theorem: PROVED that with INT8 strictly slower, plan space collapses to 85 merge schedules ⇒ JOINT == STAGED by necessity in the achievable band. Joint beats staged iff cost is non-additive AND both axes profitable.

### G1-HW: Batch-8 Pilot Evaluation | NO-GO
- Batch-8 pilot: NO-GO on compute-bound INT8 revival (INT8 still 1.62×/1.26× slower at both token extremes).
- os._exit stdio-flush bug found and fixed in pilot + measure_lut (silent print loss, JSON unaffected).

### G1-HW: Hardware Frontier vs. ToMe (V2) | PASS (+0.31 pp)
- Evaluated TANDEM concentrated merging (<= 2 early sites) vs uniform-rate ToMe (12 sites) on hardware (n = 80,000 paired evaluations).
- TANDEM wins by +0.31pp pooled (CI [0.178, 0.442]), widening to +0.560pp at tight budgets.
- Lower latency floor unlocked: TANDEM reaches 76.4% of identity latency (3.80 ms) where uniform ToMe cannot descend below 81.7% (4.06 ms) due to 12x kernel dispatch overhead.

### G1-HW: DeiT-Base Token-Coupling Crossover Discovery | PASS
- DeiT-B pilot & extension: Discovered non-monotone INT8/FP16 ratio (1.193 -> 1.116 -> 0.917 -> 0.969), crossing unity at N ~ 101.5.
- Proves coupling: token reduction dictates quantization profitability. 4-point crossover fit N≈101.5 (2-point fit mislocated it at 86.6).

### G1-HW: Energy LUT & Monotonic Ladder | PASS
- Energy LUT ACCEPTED: Monotonic ladder passed (idle 5.276 W -> N=69 -> N=197).
- R²(latency->energy) = 0.939 vs R²(BOPs->energy) = 0.061. Power scales 9.84 W -> 8.25 W as tokens fall 197 -> 69.
- Energy replan: plans identical 6/6, bits<16 never selected. Characterization holds on both latency and energy axes.

---

## Phase 4: Hardware Validation — Device B (Jetson Orin NX 8GB)

### G1-HW-2: Module Arrival, Carrier Integration & Directory Mirroring | PASS
- Target hardware: NVIDIA Jetson Orin NX 8GB module on loan, installed on P3768 carrier with stock heatsink/fan and verified 45W (19V/2.37A) power supply.
- Flashed clean JetPack 7.2 bootloader and rootfs via SDK Manager (L4T R39.2.1, TensorRT 10.16.2 / `v101602`, CUDA 13.2).
- Directory isolation: `jetson/` campaign mirrored byte-for-byte into `jetson-orinnx/` (132/132 files verified, `MIRROR_AUDIT.md`).
- Artifact quarantine: All Orin Nano output files strictly quarantined to prevent any cross-device artifact contamination.

### G1-HW-2: Power-Mode Tooling Audit & Sysfs Mask Finding | PASS (DEVICE FINDING)
- Discovered vendor tooling anomaly: `nvpmodel -q --verbose` reports configured GPU power-gating mask as NOT applied (`TPC_PG_MASK` REAL 0 vs CONF 240 in MAXN).
- State instability: Cleanly applied 20W preset reverts autonomously within 13–20 minutes without user command.
- Boot-to-boot drift: Mask value varies across reboots (0 vs. 15), causing ~10% measured latency swings.
- Driver crash: 10W preset crashes `nvpmodel.service` at boot (NVIDIA acknowledged known issue 6236259).
- Protocol amendment: Multi-mode sweep abandoned; MAXN operating point established as sole clean baseline.
- `measure_lut.py` gate enhanced: Added hard sysfs read of achieved CPU/GPU/EMC clocks (6x1984 MHz, 765 MHz, 3199 MHz) and active power-gating mask before every single measurement cell. Refuses session if unconfirmed.

### G1-HW-2: Latency Table Measurement & Superseeded Run Audit | PASS
- Audit caught that the first-pass measurement gate checked configured ceilings rather than achieved sysfs clocks.
- First-pass table (PB-3) superseded, quarantined, and archived in `out/lut/archive/` with full diagnostic report.
- Canonical Latency LUT (Run 3) measured under strict achieved-state gate: 15 TensorRT engines (FP16 and explicit-Q/DQ INT8) timed on-device (20 warm-up discarded, median of 200 runs, temperature guarded).
- Run-to-run consistency: Back-to-back measurement runs agree within 0.52%.
- Identity latency established: L_id = 3.7202 ms (NX/Nano speed ratio 0.749). Achievable range [2.8618, 3.7202] ms (76.93% - 100% of L_id).

### G1-HW-2: Reproduction Gate & Proxy Error Replication | PASS (10/10 MATCH)
- Official reproduction gate (`out/powersweep/reproduction_gate.csv`): 10/10 cells reproduced within 3% tolerance (max absolute delta 1.88%).
- Proxy divergence replicated: Cutting tokens 197 -> 69 gives 2.252x divergence factor on canonical LUT (2.258x on remeasurement), agreeing with Orin Nano (2.235x) within 1.0%.
- Confirms proxy failure is an intrinsic property of transformer launch overhead, not a single-device artifact.
- Precision axis replicated: INT8 is slower than FP16 across 5/5 token buckets, strictly monotone in N (1.123x -> 1.165x -> 1.253x -> 1.320x -> 1.436x).

### G1-HW-2: Registered Hardware Gate on Orin NX | PASS (+0.93 pp)
- Evaluated pre-registered budgets on 120,000 paired ImageNet predictions (`paired_eval_result.json`).
- REGISTERED GATE PASSED: TANDEM achieves +0.93 pp pooled top-1 gain (95% CI [+0.824, +1.029]).
- Proxy-inversion result: Because Orin NX is 25% faster, all 6 BOPs-selected plans took 3.85–5.21 ms (103.5%–139.9% of L_id). TANDEM correctly recognized that every compression plan was slower than uncompressed, selecting identity at 6/6 budgets and winning by +2.875 pp at the 15% budget.
- Characterization Branch D: Joint-vs-staged recorded as UNTESTED on Device B due to degenerate budgets (tau >= L_id), rather than claimed as an unearned confirmation.

### G1-HW-2: Cross-Device Cost Transfer Experiment | PASS (DISJOINT ENVELOPES)
- Tested cost portability between Nano and NX (`transfer_experiment_result.json`).
- Disjoint achievable ranges: Orin NX range [2.8618, 3.7202] ms vs. Orin Nano range [3.7958, 4.9668] ms.
- 4/4 NX-budget plans (80%, 85%, 90%, 95% of NX L_id) are physically infeasible under the Nano cost table.
- Core scientific conclusion: Distortion tables port across hardware without re-measurement; cost tables, budgets, and hardware plans are device-bound. Re-measuring the single-device cost table (a few device-hours) is the complete porting cost.

### G1-HW-2: Energy Proxy Divergence & Monotonic Ladder | PASS
- On-device power sampled on VDD_IN rail with jtop instrument (`energy_instrument.json`).
- Monotonic ladder verified: idle (7.296 W) < N=69 load (10.86 W) < N=197 load (12.55 W). Full model inference energy = ~45 mJ at FP16.
- Energy predictability: Latency predicts energy with R² = 0.9921; BOPs predicts energy with R² = 0.0243.
- INT8 energy penalty: INT8 draws more power while running longer, costing 1.16x–1.64x more energy than FP16 across all 5 token buckets (47.5–71.4 mJ for BOPs plans vs. 45.1 mJ identity).
- Automated Verifier: `verify_orinnx_claims.py` created and integrated into `run_reviewer.sh` as Step 8/8. All headline numbers, reproduction gates, and cross-file consistency checks pass 100%.

### G1-HW-2: Disclosed Engine Parity Limitation
- Engine-level FP16-vs-INT8 numerical agreement (0.20–0.35 max-abs output differences across layer boundaries, harness self-consistent) recorded as unresolved in `PORT_NOTES.md §9`.
- Accuracy figures come from verified shared simulation pipeline; latency/energy tables come from executed engines.

---

## Phase 5: Final Submission, Verification & Public Release

### Tier-1 Reviewer Verification & Provenance Closure | PASS (8/8 STEPS)
- Top-level repository aligned to two-device scope (`README.md`, `CITATION.cff`, `docs/RESULTS.md`, `evidence/00_MANIFEST.md`).
- Provenance hashes in `compare_summary.json` synchronized from committed files to prevent clean-clone diffs.
- `scripts/run_reviewer.sh` executed: ALL 8 STEPS PASSED with zero GPU requirement.

### CVPR 2027 Submission Package Finalization | PASS
- CVPR 2027 v2 submission package (`tempdoc/TANDEM_CVPR_submission_v2_FINAL/`) compiled (8 pages, 0 errors, 18 references, 4 figures).
- Table 1 configured as two-column-span side-by-side dual-device comparison.
- Figure 1 updated to dual-device proxy divergence and INT8 penalty plot.

### Provisional Patent & Standing Verification Suite | PASS
- Provisional patent application documentation and thesis review finalized.
- Standing verifiers (`verify_p2_claims.py`, `verify_g1_claims.py`, `verify_baselines_claims.py`, `verify_orinnx_claims.py`) certified for ongoing public audit.
