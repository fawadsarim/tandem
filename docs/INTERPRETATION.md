# Current interpretation of the saved TANDEM results

Updated 9 November 2026. This note supersedes broader narrative claims in
historical experiment logs, copied device instructions and evidence manifests.
Those original records and their numerical contents are preserved.

1. **Cost-model substitution, not universal joint-search superiority.** The
   mean +0.880pp Nano and +0.927pp NX gains compare BOPs-selected and
   device-calibrated selections. Identity is selected at five and six of the
   six ceilings. Same-menu joint and staged comparisons tie; other reproduced
   baselines can be stronger. No matched POMT evaluation was performed.
2. **Block calibration, not complete deployment validation.** Hardware times
   representative transformer blocks. Plan costs sum those entries and omit
   merging, embedding, classifier heads and cross-block effects. Accuracy is
   measured in software. Plain NX ONNX export checks pass, but do not establish
   numerical agreement of the final compiled classifier. Both NX engine labels
   use Q/DQ sources. Six-bit settings have simulated accuracy and INT8 cost.
3. **Unique images, not pooled independence.** Six comparisons reuse 20,000
   images, producing 120,000 image-budget pairs. Four scheduling comparisons
   reuse the same-size image set, producing 80,000 pairs. They are not 120,000
   or 80,000 independent examples. Current conclusions are descriptive means;
   historical pooled confidence intervals do not establish population precision.
4. **Ward schedule comparison.** Concentrated and uniform-rate arms use the
   same Ward operator. The +0.31pp result is not a hardware comparison against
   the original ToMe bipartite-matching implementation. The lowest-cost plan
   loses substantial accuracy, so reduced predicted cost alone is insufficient.
5. **Dataset boundaries.** Four-model scores on 50,000 validation images include
   the 5,000 distortion-probe images. The 20,000-image subset is path-ordered,
   not stratified; disjointness from earlier distortion probes is unestablished.
6. **Solver guarantee.** Pareto dynamic programming and schedule enumeration
   solve the finite additive surrogate (with numerical comparison tolerance in
   code). This is not a guarantee of optimal accuracy, polynomial complexity or
   optimal measured end-to-end latency. Precision dominance is conditional.
7. **Device independence and transfer.** The two device campaigns were executed
   independently. Both timing remeasurements pass their 3% gates. The later NX
   evidence package copies final NX outputs for preservation; this is not reuse
   of Nano measurements. Absolute-budget feasibility requires device calibration;
   disjoint tested intervals do not prove all transfer impossible.
8. **Research ownership and assistance.** The author reports performing the
   research independently. The previous README's supervisor-directed rerun
   wording was unsupported and has been removed. AI assistance is disclosed in
   the current README; no model is an author.

Numerical gate checks remain useful for detecting inconsistencies in saved
artifacts. They do not validate every historical label, causal interpretation,
confidence interval or deployment claim.
