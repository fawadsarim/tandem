# TANDEM

Training-Free Token and Precision Allocation with Device-Calibrated Cost.

TANDEM studies how the cost model changes compression choices for a fixed
Vision Transformer. It measures token-merging and quantization damage in
classifier-logit mean squared error, then uses a Pareto dynamic program over
85 merge schedules and per-block precisions. The same finite search accepts
analytical bit-operations (BOPs), device-calibrated block latency, or block
energy without retraining model weights.

The central observation is that arithmetic savings do not reliably predict
execution savings on the evaluated devices. Replacing BOPs with measured
block-cost tables often selects the uncompressed model, recovering accuracy
lost by unnecessary compression. This is a controlled allocation study;
it does not establish superiority over every baseline or end-to-end speedup.

## Results and interpretation

| Finding | Recorded result | Scope |
|---|---|---|
| Cost-model substitution, Orin Nano Super | Mean +0.880 percentage points in top-1 accuracy | Six budget comparisons on the same 20,000 images; identity selected at five ceilings |
| Cost-model substitution, Orin NX 8GB | Mean +0.927 percentage points | Six comparisons on the same 20,000 images; identity selected at all six ceilings |
| Concentrated vs uniform-rate Ward merging | Mean +0.31 percentage points on Nano | Four shared ceilings; same Ward operator; 20,000 distinct images, not 80,000 independent samples |
| FP16 token count 197 to 69 | BOPs fall 66.8%; block latency falls 25.7% on Nano and 25.2% on NX | Measured representative blocks, not full classifiers |
| Independent timing remeasurement | All 10 main cells within 3% on each device | Maximum differences: Nano 2.67%, NX 1.88% |
| Finite-surrogate solver | Reduced 162-plan case matches brute force at 20 budgets | Exactness concerns the additive finite surrogate, not true accuracy |

Accuracy is evaluated in software; hardware-linked plan costs sum measured
representative-block entries. Repeated image-budget pairs do not constitute
independent images. Historical pooled confidence intervals are not used to
claim population confidence in the current interpretation.

[Results index](docs/RESULTS.md) maps these findings to saved artifacts.
[Interpretation and limitations](docs/INTERPRETATION.md) explains the boundaries
and supersedes broader prose in historical logs and evidence manifests.
Original experimental records remain unchanged.

## Repository map

- `experiments/phase1_E1-E5_E4a/`: distortion, merge, quantization and solver studies.
- `experiments/phase2_1_fleet/`: analytical-cost evaluation on four architectures.
- `experiments/phase2_2_baselines/`: reproduced baselines and comparison artifacts.
- `jetson/`: standalone Orin Nano measurement and evaluation package.
- `jetson-orinnx/`: independently executed Orin NX package. Start with
  `README_ORINNX.md`, `DEVICE_PROFILE.md` and `FINDINGS.md`; some copied early
  instructions refer to Nano.
- `evidence/`: archived artifacts, hashes and original claim-to-file mappings.
- `docs/`: reproduction instructions, current interpretation and historical notes.
- `scripts/`: artifact verification and reproduction entry points.
- `requirements/`: environment maps; follow its README for live dependency files.
- `tandem/`, `results/`: pointers to implementations and canonical result tables.
- `LAB_LOG.md`: historical experiment log.

Private manuscript drafts, faculty materials, patent documents, generated
packages and temporary files are excluded from ordinary Git staging.

## Reproduction

See [REPRODUCING.md](docs/REPRODUCING.md) and
[REVIEWER_RUNBOOK.md](docs/REVIEWER_RUNBOOK.md) for commands, dependencies and
expected values. The existing Tier 1 entry point is:

```bash
git clone https://github.com/fawadsarim/tandem.git
cd tandem
pip install pandas numpy
bash scripts/run_reviewer.sh
```

Tier 1 uses saved artifacts without a GPU. Some steps regenerate plots or
analysis outputs; run it in a disposable checkout if you need a byte-preserved
working tree. Numerical checks validate artifact consistency, not every
historical narrative claim. Full inference reproduction requires the specified
model checkpoints, ImageNet access, environments and GPU or Jetson hardware.

## Evidence boundaries

- Joint and same-menu staged allocation tie in reported comparisons. Some
  specialist quantizers/composites outperform TANDEM at reported analytical
  operating points. POMT was not reproduced.
- Block sums omit merge execution, embedding, the classifier head and
  cross-block effects. Selected compiled classifiers were not validated for
  end-to-end performance or numerical agreement with software accuracy.
- Six-bit accuracy is simulated; INT8 timing is assigned as a cost assumption.
  Both NX engine labels use Q/DQ sources, limiting a pure-precision comparison.
- Full-validation accuracy includes distortion-probe images. The hardware-linked
  subset is path-ordered and its separation from earlier probes is unestablished.
- Hardware conclusions concern two Orin modules and the recorded stack and
  harness. Accuracy under hardware schedule sweeps was not evaluated on NX.

## AI assistance

Anthropic Claude assisted with implementation coding. OpenAI ChatGPT and Codex
assisted with manuscript drafting and revision, reference checks, analysis of
saved artifacts, and preparation scripts. Measurements come from recorded
research runs. The human author remains responsible for accuracy, originality
and reproducibility.

## Citation and license

See [CITATION.cff](CITATION.cff) for software citation metadata and
[LICENSE](LICENSE) for the Apache 2.0 license. A repository citation does not
imply peer-reviewed publication or acceptance.
