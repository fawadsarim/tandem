"""jetson-orinnx/int8_qdq_experiment.py -- HOST-SIDE ONLY (Colab/workstation GPU), NOT the Jetson.

G1-HW's PRODUCTION INT8 export path (promoted from a single-bucket diagnostic to the standard
path for all 5 buckets, per review -- see jetson-orinnx/README.md's INT8 findings section for
the full ladder that led here). Steps 1, 2a, 2b were run for real on the Jetson at N=197 only,
while this was still a diagnosis:
  - Step 1: forced-INT8 (TensorRT's IMPLICIT calibration path, via int8_calibrate.py) only
    actually quantizes 3 of the block's 4 GEMMs (qkv, mlp.fc1, mlp.fc2 -- NOT attn.proj),
    leaving both LayerNorms, both attention matmuls, softmax, and GELU in FP32, with explicit
    FP32<->INT8 Reformat nodes bracketing each quantized GEMM. Consistent with INT8 measuring
    ~2-2.5x SLOWER than FP16 on this hardware -- overhead without full quantization.
  - Step 2a (trtexec --best, letting TensorRT pick precision per-layer freely): landed at
    FP16-equivalent speed by ABANDONING INT8 entirely (zero nodes touching INT8). Confirmed the
    diagnosis; not a usable path.
  - Step 2b (this script, N=197 only at first): EXPLICIT Q/DQ quantization narrowed the gap
    from ~2.4x to ~1.5x slower -- still doesn't beat FP16, so the "INT8 is slower on this
    hardware" finding stands (full ladder as evidence), BUT this is the BEST ACHIEVABLE INT8
    implementation found -- textbook-clean, all 4 GEMMs correctly fused, no Reformat overhead.

DECISION (reviewed): the LUT's INT8 entries must always be sourced from the BEST
ACHIEVABLE INT8 path (this script), never the naive implicit-calibration path
(int8_calibrate.py) -- pricing INT8 at its worst inflates the measured latency of any plan
that uses it (including P2.1's own BOPs-selected plans, several of which use bits=8/6), which
distorts every downstream budget/comparison built on top of the LUT. This is why this script
is now generalized to all 5 TOKEN_BUCKETS, not just N=197 -- it produces the INT8 numbers the
LUT actually uses, not just a diagnostic data point. int8_calibrate.py's implicit-calibration
engines/numbers are NOT deleted -- kept and recorded alongside the explicit-QDQ ones for
provenance (see jetson-orinnx/README.md), just no longer what the LUT itself reports.

MECHANISM (unchanged from the original N=197 diagnosis, applies per-bucket): NVIDIA ModelOpt
applies Q/DQ (quantize/dequantize) nodes directly to all four GEMMs' weights AND input
activations -- qkv, attn.proj, mlp.fc1, mlp.fc2. These are the ONLY nn.Linear submodules
inside a timm ViT block (LayerNorm/softmax/GELU/the attention matmuls are never nn.Linear),
so restricting quantization to parent_class=nn.Linear via an explicit, hand-built quant_cfg
(not a stock preset like INT8_DEFAULT_CFG) naturally targets exactly these four and nothing
else -- it cannot accidentally touch anything the registered protocol excludes. Each bucket is
calibrated on ITS OWN pinned calib_n{bucket}.npy real-activation data export_segments.py
already captured (not new data, not a new calibration run) -- a fresh model/block instance is
quantized per bucket (mtq.quantize mutates in place, and each bucket's calibration statistics
and dummy-input shape differ). The resulting ONNX graph carries its own baked-in scales --
TensorRT builds it in EXPLICIT quantization mode (no separate calibrator/cache needed), which
lets it fuse Q/DQ directly into the GEMM kernels rather than inserting Reformat nodes.

SCOPE LIMITATION, recorded here per the registered protocol regardless of outcome: this
experiment (like the rest of this folder's LUT) measures a BLOCK-SEGMENT engine -- the
correct instrument for replan.py's DP, which prices blocks, not a whole network. The
INT8-vs-FP16 verdict at block granularity may not generalize to a fully-fused whole-network
engine, where Reformat/reconversion overhead amortizes very differently across many more
INT8 ops in sequence. This is a scope limitation of what this finding tells you about
deployment more broadly, not a limitation of the experiment itself -- the experiment answers
exactly the question replan.py's DP actually needs answered (per-block cost).

This is genuinely new, unverified-beyond-N=197 code -- unlike the rest of jetson-orinnx/, the 4
non-197 buckets have not been execution-tested against a real GPU (no GPU available where
this was written; N=197 WAS execution-tested for real and confirmed working, see above).
Every stage that depends on an uncertain library-API assumption is guarded by an assertion
that fails loudly and specifically rather than silently producing a wrong ONNX file; read the
printed diagnostics if anything trips.

Needs `nvidia-modelopt`, deliberately NOT added to requirements-host.txt -- installed in its
own cell in the Colab notebooks, same as before.

Usage (on a HOST/Colab GPU, same venv export_segments.py already uses, plus modelopt; run
from the REPO ROOT, same as export_segments.py/paired_eval.py -- this script's own default
paths are repo-root-relative, not relative to jetson-orinnx/):
    pip install nvidia-modelopt
    python jetson-orinnx/int8_qdq_experiment.py
    # produces jetson-orinnx/out/segments/block_seg_n{197,165,133,101,69}_qdq.onnx -- copy ALL 5 to
    # the Jetson, then on-device, per bucket:
    #   trtexec --onnx=out/segments/block_seg_n{N}_qdq.onnx --int8 \
    #       --saveEngine=out/engines/n{N}_int8_qdq.engine --profilingVerbosity=detailed \
    #       --dumpLayerInfo --exportLayerInfo=/tmp/n{N}_qdq_layerinfo.json
    #   (no --calib flag -- the graph carries its own scales, this is explicit-quantization mode)
    #   trtexec --loadEngine=out/engines/n{N}_int8_qdq.engine --int8 --warmUp=500 --iterations=200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_segments import BLOCK_IDX_FOR_EXPORT, load_model, TOKEN_BUCKETS  # noqa: E402

try:
    import modelopt.torch.quantization as mtq
except ImportError as e:
    print(f"FATAL: nvidia-modelopt not importable ({e!r}). pip install nvidia-modelopt into "
          "this same host venv (not requirements-host.txt -- see this file's own docstring "
          "for why).", file=sys.stderr)
    raise

CALIB_BATCH_SIZE = 16


def build_quant_config():
    """Explicit, hand-built config rather than a stock preset (e.g. INT8_DEFAULT_CFG) --
    restricting both quantizers to parent_class=nn.Linear, which inside a timm ViT block is
    EXACTLY {qkv, attn.proj, mlp.fc1, mlp.fc2} and nothing else. Per-channel weight
    quantization (axis=0, one scale per output feature) + per-tensor activation quantization
    (axis=None) -- the same values NVIDIA's own INT8_DEFAULT_CFG uses, just scoped narrower
    via parent_class so this can't accidentally touch anything the registered protocol
    excludes (LayerNorm, softmax, GELU, both attention matmuls -- none of these are
    nn.Linear, so none of them can match this filter no matter what)."""
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "parent_class": "nn.Linear",
             "enable": True, "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*input_quantizer", "parent_class": "nn.Linear",
             "enable": True, "cfg": {"num_bits": 8, "axis": None}},
        ],
        "algorithm": "max",
    }


def quantize_and_export_one_bucket(n_tokens: int, calib_npy_path: Path, onnx_out_path: Path, device: str):
    """Fresh model load per bucket -- deliberate, not wasteful: mtq.quantize() mutates its
    input module in place, and each bucket needs its own calibration statistics (different
    real-activation data) and its own dummy-input token count, so reusing a single
    already-quantized block across buckets would either error or silently carry the wrong
    bucket's calibration into another bucket's export."""
    assert calib_npy_path.exists(), (
        f"{calib_npy_path} not found -- run export_segments.py with --imagenet-dir first "
        f"(this script reuses that run's pinned N={n_tokens} calibration data, not new data)."
    )

    model = load_model()
    # Same fix as export_segments.py, same reason (real bug, see that file's own comment):
    # forces the non-fused (manual matmul) attention path so ONNX export traces cleanly.
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    model = model.to(device).eval()
    block = model.blocks[BLOCK_IDX_FOR_EXPORT]
    d_model = model.embed_dim

    calib_data = np.load(calib_npy_path).astype(np.float32)
    assert calib_data.shape[1] == n_tokens, (
        f"{calib_npy_path} has n_tokens={calib_data.shape[1]}, expected {n_tokens} -- wrong "
        "calibration file for this bucket."
    )
    calib_tensor = torch.from_numpy(calib_data).to(device)
    print(f"  Loaded {calib_tensor.shape[0]} real calibration activations from {calib_npy_path}")

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, calib_tensor.shape[0], CALIB_BATCH_SIZE):
                m(calib_tensor[i:i + CALIB_BATCH_SIZE])

    print("  Quantizing (Q/DQ on qkv/attn.proj/mlp.fc1/mlp.fc2 weights+activations, INT8, "
          "calibrated on the pinned real-activation set)...")
    block = mtq.quantize(block, build_quant_config(), forward_loop)

    quantized_names = [name for name, m in block.named_modules()
                        if hasattr(m, "weight_quantizer") or hasattr(m, "input_quantizer")]
    print(f"  Quantized Linear submodules: {quantized_names}")
    EXPECTED = {"qkv", "proj", "fc1", "fc2"}
    found = {n.rsplit(".", 1)[-1] for n in quantized_names}
    assert EXPECTED <= found, (
        f"Expected quantizers on {EXPECTED}, found suffixes {found} from {quantized_names} -- "
        "quant_cfg's parent_class filter did not match as expected, do not proceed to export."
    )

    dummy = torch.randn(1, n_tokens, d_model, device=device)
    with torch.no_grad():
        _ = block(dummy)  # confirm the quantized block still runs cleanly before exporting

    onnx_out_path.parent.mkdir(parents=True, exist_ok=True)
    block_cpu = block.cpu()
    dummy_cpu = dummy.cpu()
    torch.onnx.export(
        block_cpu, dummy_cpu, str(onnx_out_path),
        input_names=["tokens_in"], output_names=["tokens_out"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None, dynamo=False,
        # dynamo=False: same legacy exporter as export_segments.py -- fused_attn=False above
        # already avoids the is_causal/tracer bug that forced that choice there. Confirmed
        # working for real at N=197; if a DIFFERENT bucket's Q/DQ nodes don't survive this
        # exporter (see the assertion right after this call), the fallback to try is
        # dynamo=True with `pip install onnxscript`.
    )
    print(f"  Wrote {onnx_out_path}")

    # Ground-truth check: does the exported graph actually contain QuantizeLinear/
    # DequantizeLinear nodes? If constant-folding or the exporter silently dropped them,
    # TensorRT would just build a plain FP32/FP16 engine and this whole export would be a
    # false negative without this catching it first.
    onnx_model = onnx.load(str(onnx_out_path))
    qdq_ops = [n.op_type for n in onnx_model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    print(f"  QuantizeLinear/DequantizeLinear nodes in exported graph: {len(qdq_ops)}")
    assert qdq_ops, (
        f"No QuantizeLinear/DequantizeLinear nodes in the exported ONNX graph for N={n_tokens} "
        "-- the Q/DQ export did not survive (try dynamo=True + `pip install onnxscript` "
        "instead of dynamo=False above). Do NOT copy this ONNX to the Jetson as-is -- it would "
        "just build a plain (non-quantized) engine and silently corrupt the LUT."
    )
    return len(qdq_ops)


def main():
    # Defaults are relative to the REPO ROOT, not to this file's own directory -- real bug
    # found and fixed, 2026-07: this script is invoked as `{VENV_PYTHON} jetson-orinnx/int8_qdq_
    # experiment.py` from the Colab notebook's CWD (the cloned repo root, /content/tandem),
    # exactly like export_segments.py/paired_eval.py -- NOT with CWD=jetson-orinnx/, despite this
    # file's own usage docstring implying that. export_segments.py's actual calibration
    # output path is resolved via _common.py's OUT_ROOT, which is `__file__`-relative (always
    # <repo_root>/jetson-orinnx/out/, regardless of CWD) -- so the real files live under
    # jetson-orinnx/out/segments/calib/, not out/segments/... (which this script's defaults
    # originally said, and which resolved to a nonexistent path one directory level off).
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", default="jetson-orinnx/out/segments",
                         help="Where export_segments.py wrote calib/calib_n{bucket}.npy and "
                              "where the {bucket}_qdq.onnx outputs are written.")
    args = parser.parse_args()
    segments_dir = Path(args.segments_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU -- this needs one, same as export_segments.py."

    print(f"=== int8_qdq_experiment.py: explicit Q/DQ INT8 export, all {len(TOKEN_BUCKETS)} buckets ===")
    results = {}
    for n_tokens in TOKEN_BUCKETS:
        print(f"\n--- bucket n_tokens={n_tokens} ---")
        calib_npy = segments_dir / "calib" / f"calib_n{n_tokens}.npy"
        onnx_out = segments_dir / f"block_seg_n{n_tokens}_qdq.onnx"
        n_qdq_ops = quantize_and_export_one_bucket(n_tokens, calib_npy, onnx_out, device)
        results[n_tokens] = {"onnx_path": str(onnx_out), "qdq_ops": n_qdq_ops}

    print(f"\nAll {len(TOKEN_BUCKETS)} buckets exported and verified (each contains real "
          "QuantizeLinear/DequantizeLinear nodes).")
    print("\nNext: copy ALL of these ONNX files to the Jetson, then on-device build + "
          "benchmark each (--int8, no --calib flag -- explicit-quantization mode):")
    for n_tokens, r in results.items():
        print(f"  scp {r['onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/segments/")
    print("\nOn the Jetson, per bucket:")
    print("  trtexec --onnx=out/segments/block_seg_n{N}_qdq.onnx --int8 \\")
    print("      --saveEngine=out/engines/n{N}_int8_qdq.engine --profilingVerbosity=detailed \\")
    print("      --dumpLayerInfo --exportLayerInfo=/tmp/n{N}_qdq_layerinfo.json")
    print("  trtexec --loadEngine=out/engines/n{N}_int8_qdq.engine --int8 --warmUp=500 --iterations=200")


if __name__ == "__main__":
    main()
