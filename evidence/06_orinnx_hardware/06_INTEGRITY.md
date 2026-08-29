# 06_INTEGRITY.md -- Orin NX Evidence Package Verification

## Summary & Verification Status
- Target Device: NVIDIA Jetson Orin NX (8GB module, JetPack 7.x, L4T R39.2.1, TensorRT 10.16.2 (`v101602`), CUDA 13.2)
- Operating Power Mode: MAXN (Mode 0, locked clocks: CPU 6x1984 MHz, GPU 765 MHz, EMC 3199 MHz)
- Self-Test Verification: `self_test.py` passed 11/11 hard assertions on hardware.
- Reproduction Gate: `powersweep_repro_gate.py` passed 10/10 rows within 3% tolerance (max delta: 1.88%).
- Paired Evaluation Gate: `paired_eval_result.json` achieved +0.93 pp Top-1 gain (120,000 paired sample evaluations), passing the registered +0.40 pp gate.
- Phase 3 Transfer Experiment: `transfer_experiment_result.json` verified cost-table non-transferability (Nano table fails 4/4 budgets under NX latency envelope).

## Key Artifacts in Evidence Package
1. `out/lut/latency_lut.json`: Full 15-cell measured latency lookup table.
2. `out/replan/plans_hw.csv`: Re-planned Pareto hardware plans under Orin NX cost.
3. `out/paired_eval/paired_eval_result.json`: Statistically significant paired ImageNet accuracy verification (+0.93 pp).
4. `out/powersweep/powersweep_analysis_summary.json`: AXIS_DOMINATION_CONFIRMED_ACROSS_OPERATING_POINTS.
5. `out/transfer/transfer_experiment_result.json`: Headline transfer experiment results.
6. `out/versions.json`: Cryptographic environment fingerprint.

## Cryptographic Hashes
All hashes are computed via SHA-256 and matched in `06_MANIFEST.md`.
