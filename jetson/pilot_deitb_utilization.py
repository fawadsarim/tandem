"""jetson/pilot_deitb_utilization.py -- device-independent (pure arithmetic on already-measured
JSON, no device, no TensorRT). DeiT-B INT8 pilot, Part 3 -- see jetson/README.md's "DeiT-B INT8
pilot" section.

PURPOSE: explain WHY the Part 2 verdict came out as it did, by testing the hypothesis directly
(PREREGISTRATION.md): "utilization, not precision arithmetic, is what killed INT8 on DeiT-S -- a
4x larger model should reverse it if that diagnosis is right." Computes ACHIEVED TFLOPS for both
models at N=197, FP16, batch=1 -- from a stated analytic FLOP formula and REAL measured medians
-- and the cross-model ACHIEVED-TFLOPS RATIO, which needs no assumption about the device's peak
throughput at all (the peak cancels out in a same-hardware ratio).

FLOP FORMULA (one ViT/DeiT transformer block, forward pass, batch=1, N tokens, d_model=d,
mlp_ratio=4 -- standard for DeiT-S/B): per-block MACs = 12*N*d^2 + 2*N^2*d (QKV: 3Nd^2, output
proj: Nd^2, MLP fc1+fc2: 8Nd^2 combined -- (3+1+8)=12 Nd^2 term; attention scores + attn@V:
2*N^2*d, summed across heads since head_dim*n_heads=d). FLOPs = 2*MACs (1 multiply + 1 add per
MAC), the standard convention.

PEAK/UTILIZATION -- HONESTLY UNCERTAIN, presented as such (not asserted as fact): converting
achieved TFLOPS to a utilization PERCENTAGE needs this device's peak FP16 tensor-core TFLOPS.
Two independent derivations were tried and DISAGREE by roughly 3x (bottom-up from this device's
own REAL measured GPU clock -- see versions.json's jetson_clocks_show, 624.75MHz -- times
NVIDIA's published Orin Nano tensor-core count (32) times a commonly-cited but NOT
independently-verified-for-this-exact-die Ampere per-cycle throughput figure; vs. top-down from
NVIDIA's published "67 TOPS" Orin Nano Super headline figure divided by commonly-assumed
sparse/INT8-to-FP16 conversion ratios). Rather than silently pick one and present a possibly-
wrong percentage as fact, BOTH are reported with their derivations and the disagreement is
stated plainly. The RATIO (DeiT-B TFLOPS / DeiT-S TFLOPS, both real, both measured, no peak
assumption) is the rigorous, headline number this script actually stands behind.

Usage: python pilot_deitb_utilization.py   (reads out/pilot_deitb/pilot_deitb_result.json and
out/lut/latency_lut.json -- both already real/committed, no device needed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402

N_TOKENS = 197
D_DEITS = 384
D_DEITB = 768

# Real, on-device measured GPU clock at MAXN_SUPER (see versions.json's jetson_clocks_show) --
# NOT a marketing/boost figure.
GPU_CLOCK_HZ_MEASURED = 624_750_000
N_TENSOR_CORES_ORIN_NANO = 32  # NVIDIA's published Orin Nano module spec
ASSUMED_FP16_FLOPS_PER_CYCLE_PER_TENSOR_CORE = 256  # commonly-cited Ampere 3rd-gen figure, NOT independently verified for this die
ORIN_NANO_SUPER_HEADLINE_TOPS = 67  # NVIDIA's published Orin Nano Super headline (INT8, sparse)


def block_flops(n_tokens: int, d: int) -> float:
    macs = 12 * n_tokens * d**2 + 2 * n_tokens**2 * d
    return 2.0 * macs


def achieved_tflops(n_tokens: int, d: int, median_ms: float) -> float:
    flops = block_flops(n_tokens, d)
    seconds = median_ms / 1000.0
    return flops / seconds / 1e12


def main():
    print("=== pilot_deitb_utilization.py: FLOP/utilization analysis (Part 3) ===")

    deitb_path = JETSON_ROOT / "out" / "pilot_deitb" / "pilot_deitb_result.json"
    assert deitb_path.exists(), f"{deitb_path} not found -- run pilot_deitb_measure.py first."
    deitb_result = read_json(deitb_path)

    lut_path = JETSON_ROOT / "out" / "lut" / "latency_lut.json"
    assert lut_path.exists(), f"{lut_path} not found -- this is the original, committed, batch=1 LUT."
    lut = read_json(lut_path)["entries"]

    deits_fp16_ms = lut["n197_fp16"]["median_ms"]
    deits_int8_ms = lut["n197_int8"]["median_ms"]
    deitb_fp16_ms = deitb_result["measurements"]["197"]["fp16"]["median_ms"]
    deitb_int8_ms = deitb_result["measurements"]["197"]["int8_qdq"]["median_ms"]

    deits_flops = block_flops(N_TOKENS, D_DEITS)
    deitb_flops = block_flops(N_TOKENS, D_DEITB)
    print(f"\nAnalytic per-block FLOPs at N={N_TOKENS}, batch=1: "
          f"DeiT-S (d={D_DEITS}) = {deits_flops:.4e}, DeiT-B (d={D_DEITB}) = {deitb_flops:.4e} "
          f"({deitb_flops / deits_flops:.2f}x)")

    deits_tflops_fp16 = achieved_tflops(N_TOKENS, D_DEITS, deits_fp16_ms)
    deitb_tflops_fp16 = achieved_tflops(N_TOKENS, D_DEITB, deitb_fp16_ms)
    deits_tflops_int8 = achieved_tflops(N_TOKENS, D_DEITS, deits_int8_ms)
    deitb_tflops_int8 = achieved_tflops(N_TOKENS, D_DEITB, deitb_int8_ms)

    ratio = deitb_tflops_fp16 / deits_tflops_fp16
    print(f"\n=== ACHIEVED TFLOPS (rigorous -- real FLOP formula / real measured median, no peak assumption) ===")
    print(f"  DeiT-S FP16 @ N=197: {deits_fp16_ms:.4f}ms -> {deits_tflops_fp16:.4f} TFLOPS achieved")
    print(f"  DeiT-B FP16 @ N=197: {deitb_fp16_ms:.4f}ms -> {deitb_tflops_fp16:.4f} TFLOPS achieved")
    print(f"  RATIO (DeiT-B / DeiT-S achieved FP16 TFLOPS): {ratio:.2f}x -- this is the headline "
          "number this script stands behind; no peak-TFLOPS assumption needed for a ratio on "
          "the same hardware.")

    # --- Peak/utilization -- HONESTLY UNCERTAIN, see module docstring ---
    peak_bottom_up = N_TENSOR_CORES_ORIN_NANO * GPU_CLOCK_HZ_MEASURED * ASSUMED_FP16_FLOPS_PER_CYCLE_PER_TENSOR_CORE / 1e12
    peak_top_down = ORIN_NANO_SUPER_HEADLINE_TOPS / 2.0 / 2.0  # /2 sparse->dense, /2 int8->fp16, both assumed standard Ampere ratios

    print(f"\n=== Peak TFLOPS -- TWO independent derivations, DISAGREE, both reported (see module docstring) ===")
    print(f"  Bottom-up (measured clock {GPU_CLOCK_HZ_MEASURED/1e6:.2f}MHz x {N_TENSOR_CORES_ORIN_NANO} "
          f"tensor cores x {ASSUMED_FP16_FLOPS_PER_CYCLE_PER_TENSOR_CORE} FLOPs/cycle, the last figure "
          f"NOT independently verified for this die): {peak_bottom_up:.2f} TFLOPS")
    print(f"  Top-down ({ORIN_NANO_SUPER_HEADLINE_TOPS} TOPS published headline / 2 sparse->dense / "
          f"2 int8->fp16, both assumed standard ratios): {peak_top_down:.2f} TFLOPS")
    print(f"  Disagreement: {peak_top_down / peak_bottom_up:.1f}x -- NOT resolved here. Utilization %% "
          "below is reported under BOTH, neither asserted as ground truth.")

    deits_util_bottom_up = deits_tflops_fp16 / peak_bottom_up * 100
    deits_util_top_down = deits_tflops_fp16 / peak_top_down * 100
    deitb_util_bottom_up = deitb_tflops_fp16 / peak_bottom_up * 100
    deitb_util_top_down = deitb_tflops_fp16 / peak_top_down * 100

    print(f"\n  DeiT-S utilization: {deits_util_bottom_up:.1f}% (bottom-up peak) / "
          f"{deits_util_top_down:.1f}% (top-down peak)")
    print(f"  DeiT-B utilization: {deitb_util_bottom_up:.1f}% (bottom-up peak) / "
          f"{deitb_util_top_down:.1f}% (top-down peak)")
    print("  (Task's own WHY section cited '~15%' for DeiT-S -- compare against both estimates "
          "above; neither derivation here was tuned to hit that number.)")

    print(f"\n=== Cross-scale INT8/FP16 ratio table (batch=1, N=197) ===")
    print(f"  DeiT-S: FP16={deits_fp16_ms:.4f}ms  INT8={deits_int8_ms:.4f}ms  ratio={deits_int8_ms/deits_fp16_ms:.3f}x")
    print(f"  DeiT-B: FP16={deitb_fp16_ms:.4f}ms  INT8={deitb_int8_ms:.4f}ms  ratio={deitb_int8_ms/deitb_fp16_ms:.3f}x")

    result = {
        "n_tokens": N_TOKENS, "d_deits": D_DEITS, "d_deitb": D_DEITB,
        "deits_flops_per_block": deits_flops, "deitb_flops_per_block": deitb_flops,
        "deits_fp16_median_ms": deits_fp16_ms, "deitb_fp16_median_ms": deitb_fp16_ms,
        "deits_int8_median_ms": deits_int8_ms, "deitb_int8_median_ms": deitb_int8_ms,
        "deits_achieved_tflops_fp16": deits_tflops_fp16, "deitb_achieved_tflops_fp16": deitb_tflops_fp16,
        "deits_achieved_tflops_int8": deits_tflops_int8, "deitb_achieved_tflops_int8": deitb_tflops_int8,
        "achieved_tflops_ratio_deitb_over_deits": ratio,
        "peak_tflops_bottom_up": peak_bottom_up, "peak_tflops_top_down": peak_top_down,
        "peak_derivation_disagreement_note": (
            "Two independent peak-TFLOPS derivations disagree by "
            f"{peak_top_down / peak_bottom_up:.1f}x -- neither is asserted as ground truth. "
            "The achieved-TFLOPS RATIO above needs no peak assumption and is the number this "
            "script stands behind."
        ),
        "deits_utilization_pct_bottom_up": deits_util_bottom_up, "deits_utilization_pct_top_down": deits_util_top_down,
        "deitb_utilization_pct_bottom_up": deitb_util_bottom_up, "deitb_utilization_pct_top_down": deitb_util_top_down,
        "deits_int8_fp16_ratio": deits_int8_ms / deits_fp16_ms,
        "deitb_int8_fp16_ratio": deitb_int8_ms / deitb_fp16_ms,
    }
    out_path = ensure_out_dir("pilot_deitb") / "pilot_deitb_utilization.json"
    write_json(out_path, result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
