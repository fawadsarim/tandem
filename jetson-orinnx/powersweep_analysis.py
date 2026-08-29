"""jetson-orinnx/powersweep_analysis.py -- device-independent (pure data analysis, no device, no
TensorRT). POWER-MODE SWEEP, Part 3 -- the actual result. Reads every
out/powersweep/latency_lut_<mode>.json produced by powersweep_measure.py and computes, PER MODE
(normalized to that mode's own N=197/FP16 reference, never cross-mode):

  1. PROXY DIVERGENCE: BOPs ratio (via replan.py's own block_bops, imported not re-derived) vs
     measured latency ratio, at every bucket -- same "latency_divergence_factor" definition
     jetson-orinnx/proxy_divergence.py already uses, just recomputed per mode instead of once at
     MAXN_SUPER. Headline: the N=69 proxy error per mode, against the committed MAXN_SUPER
     reference read directly from jetson-orinnx/out/energy/proxy_divergence.csv's own
     n_tokens=69/precision=fp16 row (2.235x -- NOT the "~1.84x" figure this task's own prompt
     cited, which does not match that file; likely confused with the achieved-TFLOPS figure
     below, 1.83, a plausible transcription mix-up between two adjacent numbers in this
     project's own prior output. Corrected here to the real committed value, read from the
     file rather than hardcoded, so this cannot silently drift out of sync again).

  2. PRECISION AXIS: INT8/FP16 median ratio at every bucket, every mode. Applies the
     pre-registered decision rule from out/powersweep/PREREGISTRATION.md verbatim: INT8 faster
     than FP16 by MORE than 5% (ratio <= 0.95) at >=2 of the 5 buckets, at ANY mode ->
     NEEDS_MANUAL_REVIEW. Otherwise -> AXIS_DOMINATION_CONFIRMED. The verdict is computed
     automatically here, not eyeballed.

  3. UTILIZATION CONTEXT: achieved TFLOPS at FP16/N=197 per mode. Uses the REAL (unweighted)
     MAC count for one DeiT-S transformer block -- 12*n*d^2 (linear) + 2*n^2*d (attention), the
     same two terms replan.py's block_bops formula uses, but WITHOUT block_bops's bits^2/16^2
     weighting (that weighting is a simulated accuracy-cost proxy from P2.1's own allocator, not
     real hardware FLOPs -- see replan.py's own module docstring). TFLOPS = 2 * real_MACs /
     median_latency_s / 1e12 (2 FLOPs/MAC, standard convention). This formula and its FLOPs
     count (756782592.0 per block at N=197) is an EXACT match to the value already committed in
     jetson-orinnx/out/pilot_deitb/pilot_deitb_utilization.json's own deits_flops_per_block field
     (written earlier this project by a different script, independently) -- confirming the
     formula, not assuming it. That file's own deits_achieved_tflops_fp16 (1.83, at MAXN_SUPER)
     is the real committed reference, read directly from the file below -- NOT the "~2.5
     TFLOPS" figure this task's own prompt cited, which does not match any committed artifact.

Usage: python powersweep_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json, write_json  # noqa: E402
from replan import block_bops  # noqa: E402

BUCKETS = (197, 165, 133, 101, 69)
PRECISIONS = ("fp16", "int8")
BITS_FOR_PRECISION = {"fp16": 16, "int8": 8}  # same documented convention as proxy_divergence.py
D_MODEL_DEITS = 384
INT8_FASTER_THRESHOLD_PCT = 5.0
INT8_FASTER_MIN_BUCKETS = 2


def real_macs_per_block(n_l: int, d_model: int = D_MODEL_DEITS) -> float:
    """Real (unweighted) MAC count for one DeiT-S transformer block -- see module docstring.
    Deliberately NOT block_bops (that formula's bits^2/16^2 terms are a simulated cost-proxy
    weighting, not real hardware FLOPs)."""
    linear_macs = 12 * n_l * d_model ** 2
    attention_macs = 2 * n_l ** 2 * d_model
    return linear_macs + attention_macs


def main():
    print("=== powersweep_analysis.py: Part 3 -- proxy divergence + precision axis + utilization, per mode ===")

    powersweep_dir = JETSON_ROOT / "out" / "powersweep"
    mode_files = sorted(powersweep_dir.glob("latency_lut_*.json"))
    assert mode_files, f"no latency_lut_<mode>.json files found in {powersweep_dir} -- run powersweep_measure.py first."

    # Real committed references, READ from the actual files (never hardcoded) -- see module
    # docstring for why these correct the task's own "~1.84x" / "~2.5 TFLOPS" citations.
    established_proxy_error_path = JETSON_ROOT / "out" / "energy" / "proxy_divergence.csv"
    established_proxy_error_n69 = None
    if established_proxy_error_path.exists():
        established_df = pd.read_csv(established_proxy_error_path)
        row = established_df[(established_df.n_tokens == 69) & (established_df.precision == "fp16")]
        if not row.empty:
            established_proxy_error_n69 = float(row.iloc[0]["latency_divergence_factor"])
    established_tflops_path = JETSON_ROOT / "out" / "pilot_deitb" / "pilot_deitb_utilization.json"
    established_tflops_n197 = None
    if established_tflops_path.exists():
        established_tflops_n197 = read_json(established_tflops_path).get("deits_achieved_tflops_fp16")
    print(f"Real committed references: proxy_error(N=69/fp16, MAXN_SUPER)="
          f"{established_proxy_error_n69} (from {established_proxy_error_path.name}), "
          f"achieved_tflops(N=197/fp16, MAXN_SUPER)={established_tflops_n197} "
          f"(from {established_tflops_path.name})")

    modes_data = {}
    for f in mode_files:
        mode_label = f.stem.replace("latency_lut_", "")
        entries = read_json(f)["entries"]
        modes_data[mode_label] = entries
    print(f"Found {len(modes_data)} mode(s): {sorted(modes_data.keys())}")

    # === Part 1: proxy divergence per mode ===
    divergence_rows = []
    for mode_label, entries in modes_data.items():
        rows = []
        for n in BUCKETS:
            for prec in PRECISIONS:
                key = f"n{n}_{prec}"
                if key not in entries or entries[key].get("rebuild_needed"):
                    continue
                bits = BITS_FOR_PRECISION[prec]
                bops = block_bops(n, bits, d_model=D_MODEL_DEITS)
                lat_ms = entries[key]["median_ms"]
                rows.append({"n_tokens": n, "precision": prec, "bops": bops, "latency_ms": lat_ms})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        ref_rows = df[(df.n_tokens == 197) & (df.precision == "fp16")]
        if ref_rows.empty:
            print(f"  [{mode_label}] SKIPPING -- no N=197/fp16 reference row present.")
            continue
        ref = ref_rows.iloc[0]
        df["mode"] = mode_label
        df["bops_ratio"] = df["bops"] / ref["bops"]
        df["latency_ratio"] = df["latency_ms"] / ref["latency_ms"]
        df["proxy_error"] = df["latency_ratio"] / df["bops_ratio"]
        divergence_rows.append(df)

    div_df = pd.concat(divergence_rows, ignore_index=True)
    div_csv = ensure_out_dir("powersweep") / "proxy_divergence_by_mode.csv"
    div_df.to_csv(div_csv, index=False)
    print(f"\nWrote {div_csv}")

    print("\n=== Headline: proxy error at N=69/fp16 relative to N=197/fp16, per mode ===")
    n69_rows = div_df[(div_df.n_tokens == 69) & (div_df.precision == "fp16")]
    headline_by_mode = {}
    for _, r in n69_rows.iterrows():
        headline_by_mode[r["mode"]] = float(r["proxy_error"])
        ref_str = f"{established_proxy_error_n69:.4f}x" if established_proxy_error_n69 is not None else "n/a"
        print(f"  mode={r['mode']:>10s}: proxy_error(N=69 vs N=197, fp16) = {r['proxy_error']:.4f}x "
              f"(real committed MAXN_SUPER reference: {ref_str})")

    # === Part 2: precision axis per mode ===
    print("\n=== INT8/FP16 ratio, every bucket, every mode ===")
    precision_rows = []
    needs_manual_review = False
    verdict_detail_by_mode = {}
    for mode_label, entries in modes_data.items():
        n_buckets_int8_faster = 0
        for n in BUCKETS:
            fp16_key, int8_key = f"n{n}_fp16", f"n{n}_int8"
            if fp16_key not in entries or int8_key not in entries:
                continue
            if entries[fp16_key].get("rebuild_needed") or entries[int8_key].get("rebuild_needed"):
                continue
            fp16_ms = entries[fp16_key]["median_ms"]
            int8_ms = entries[int8_key]["median_ms"]
            ratio = int8_ms / fp16_ms
            pct_faster = (1.0 - ratio) * 100.0  # positive means INT8 is faster
            int8_faster_over_threshold = pct_faster > INT8_FASTER_THRESHOLD_PCT
            if int8_faster_over_threshold:
                n_buckets_int8_faster += 1
            precision_rows.append({
                "mode": mode_label, "n_tokens": n, "fp16_median_ms": fp16_ms, "int8_median_ms": int8_ms,
                "int8_fp16_ratio": ratio, "pct_int8_faster": pct_faster,
                "int8_faster_over_5pct": int8_faster_over_threshold,
            })
            print(f"  mode={mode_label:>10s} N={n:>3d}: int8/fp16={ratio:.4f}  "
                  f"int8_faster={pct_faster:+.2f}%  over_5pct_threshold={int8_faster_over_threshold}")
        mode_triggers = n_buckets_int8_faster >= INT8_FASTER_MIN_BUCKETS
        verdict_detail_by_mode[mode_label] = {
            "n_buckets_int8_faster_over_5pct": n_buckets_int8_faster,
            "needs_manual_review": mode_triggers,
        }
        if mode_triggers:
            needs_manual_review = True

    prec_df = pd.DataFrame(precision_rows)
    prec_csv = ensure_out_dir("powersweep") / "precision_axis_by_mode.csv"
    prec_df.to_csv(prec_csv, index=False)
    print(f"\nWrote {prec_csv}")

    verdict = "NEEDS_MANUAL_REVIEW" if needs_manual_review else "AXIS_DOMINATION_CONFIRMED_ACROSS_OPERATING_POINTS"
    print(f"\n=== PRE-REGISTERED DECISION RULE VERDICT: {verdict} ===")
    for mode_label, d in verdict_detail_by_mode.items():
        print(f"  mode={mode_label}: {d['n_buckets_int8_faster_over_5pct']}/5 buckets INT8>5% faster "
              f"(threshold >=2 triggers NEEDS_MANUAL_REVIEW) -> triggers={d['needs_manual_review']}")

    # === Part 3: utilization context (achieved TFLOPS, FP16, N=197, per mode) ===
    print("\n=== Utilization context: achieved TFLOPS (FP16, N=197), per mode ===")
    tflops_by_mode = {}
    real_macs_n197 = real_macs_per_block(197)
    real_flops_n197 = 2.0 * real_macs_n197
    for mode_label, entries in modes_data.items():
        key = "n197_fp16"
        if key not in entries or entries[key].get("rebuild_needed"):
            continue
        median_s = entries[key]["median_ms"] / 1000.0
        tflops = real_flops_n197 / median_s / 1e12
        tflops_by_mode[mode_label] = tflops
        print(f"  mode={mode_label:>10s}: {tflops:.4f} TFLOPS  (real committed MAXN_SUPER reference, "
              f"out/pilot_deitb/pilot_deitb_utilization.json: {established_tflops_n197})")

    result = {
        "model_scope": "deit_small_patch16_224, batch=1, one representative transformer block",
        "modes_analyzed": sorted(modes_data.keys()),
        "proxy_divergence_headline_n69_fp16_by_mode": headline_by_mode,
        "established_maxn_super_reference_proxy_error_n69": established_proxy_error_n69,
        "established_reference_source_note": (
            "Both established_* reference values below are read directly from this project's "
            "own prior committed artifacts (out/energy/proxy_divergence.csv, "
            "out/pilot_deitb/pilot_deitb_utilization.json), NOT from this task's own prompt "
            "text -- that prompt cited '1.84x' and '~2.5 TFLOPS', neither of which matches the "
            "real committed record (2.235x and 1.83 TFLOPS respectively). Likely a transcription "
            "mix-up between these two adjacent numbers from prior output; corrected here."
        ),
        "precision_axis_verdict": verdict,
        "precision_axis_verdict_detail_by_mode": verdict_detail_by_mode,
        "decision_rule": (
            "INT8 faster than FP16 by MORE than 5% at >=2 of the 5 buckets, at ANY mode -> "
            "NEEDS_MANUAL_REVIEW, do not proceed further. Otherwise -> axis domination confirmed "
            "across operating points, closes the question permanently. (Verbatim from "
            "out/powersweep/PREREGISTRATION.md.)"
        ),
        "achieved_tflops_fp16_n197_by_mode": tflops_by_mode,
        "established_maxn_super_reference_tflops": established_tflops_n197,
    }
    write_json(ensure_out_dir("powersweep") / "powersweep_analysis_summary.json", result)
    print(f"\nWrote {ensure_out_dir('powersweep') / 'powersweep_analysis_summary.json'}")


if __name__ == "__main__":
    main()
