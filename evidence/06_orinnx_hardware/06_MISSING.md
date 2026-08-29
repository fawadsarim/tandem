# 06_MISSING.md — Status & Limitations Record

## Resolved Stages (Completed in Full Campaign)

### 1. Calibration Data & Full Engine Coverage — COMPLETED
- **Status**: All 5 token buckets ($N \in \{69, 101, 133, 165, 197\}$) have complete calibration arrays (`calib_n*.npy`), plain ONNX segments (`block_seg_n*.onnx`), and explicit QDQ graphs (`block_seg_n*_qdq.onnx`).
- **Engine Compilation**: All 15 TensorRT engines (5 FP16 + 5 INT8 QDQ + 5 Naive INT8) were successfully compiled and benchmarked under locked clocks on the Jetson Orin NX.

### 2. Paired ImageNet-1k Evaluation — COMPLETED
- **Status**: Executed on Google Colab (Standard Runtime 2026.4) across 20,000 validation images (120,000 paired sample evaluations across 6 budgets).
- **Result**: **$+0.93\text{ pp}$ Top-1 accuracy gain** (95% CI: $[+0.82, +1.03]\text{ pp}$), decisively passing the registered $+0.40\text{ pp}$ gate and replicating the Orin Nano result ($+0.88\text{ pp}$).
- **Artifact**: Recorded in `out/paired_eval/paired_eval_result.json`.

---

## Retained Hardware Observations & Scope Notes

### Ambient Temperature — Not Measured
- No ambient probe was available during sustained-load runs. Peak SoC thermal zone temperatures were measured via jtop under the strict $80^\circ\text{C}$ guard ($62.7^\circ\text{C}$ steady state).

### Power Mode Operating Scope
- On this Orin NX module under JetPack, hardware TPC power-gating (10W/15W) disables GPU devfreq sysfs nodes, triggering low-level CUDA initialization faults.
- Benchmarking and verification are conducted in **MAXN (Mode 0)**, where all 8 GPU TPCs, 6 CPU cores, and the 3199 MHz EMC bus run at unconstrained silicon limits.
