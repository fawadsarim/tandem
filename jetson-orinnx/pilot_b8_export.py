"""jetson-orinnx/pilot_b8_export.py -- HOST-SIDE ONLY (Colab/workstation GPU), NOT the Jetson.
Same reason as export_segments.py/int8_qdq_experiment.py -- see those files' own docstrings.

BATCH-8 INT8 PILOT (GO/NO-GO), step 1 of 3 -- see jetson-orinnx/README.md's "Batch-8 INT8 pilot"
section for the full purpose and jetson-orinnx/CHARACTERIZATION.md for why this pilot exists at all:
every ONNX export/engine/LUT entry elsewhere in this folder is BATCH=1 by construction (see
export_segments.py's fixed torch.randn(1, n_tokens, d_model) dummy input), which is a
memory-bandwidth-bound regime -- exactly where INT8's bandwidth-halving advantage is smallest
relative to its Q/DQ conversion overhead (see README.md's INT8 findings). This script exports
the SAME two artifact types export_segments.py/int8_qdq_experiment.py already produce (a plain
FP-target ONNX and an explicit-QDQ INT8-target ONNX), just at BATCH_SIZE=8 instead of 1, and
restricted to PILOT_BUCKETS=(197, 69) -- the two extremes of the measured token-count range --
to keep this pilot to roughly an afternoon rather than repeating the full 5-bucket sweep.

REUSES existing calibration data, no new capture: calib_n197.npy/calib_n69.npy were already
captured by export_segments.py (128 real images, N_CALIB_IMAGES) and already used once by
int8_qdq_experiment.py for the batch=1 QDQ export. Calibration batch size (CALIB_BATCH_SIZE=16,
forward-pass mini-batching over those 128 images to gather quantization statistics) is
independent of THIS script's export batch size (8, the model's fixed INPUT shape) -- the same
independence export_segments.py's own N_CALIB_IMAGES vs per-bucket export shape already relies
on. No new HF access, no new calibration run.

VERIFICATION (same tolerance as export_segments.py, same structural check as
int8_qdq_experiment.py): the FP-target export is checked against the SAME PyTorch block on the
SAME random batch-8 input via onnxruntime (max-abs-diff <= MAX_ABS_DIFF_TOL); the QDQ-target
export is checked for real QuantizeLinear/DequantizeLinear nodes actually surviving the
exporter (same assertion int8_qdq_experiment.py already uses, same reasoning: a silently
un-quantized export would build a plain FP engine and corrupt the pilot without this catching
it first).

Needs `nvidia-modelopt` (same as int8_qdq_experiment.py, deliberately not in
requirements-host.txt -- install separately, see that file's own docstring) for the QDQ half
only; the FP half needs only what export_segments.py already needs.

Usage (from the REPO ROOT, same convention as export_segments.py/int8_qdq_experiment.py/
paired_eval.py -- this script's own default paths are repo-root-relative):
    pip install nvidia-modelopt
    python jetson-orinnx/pilot_b8_export.py
    # produces jetson-orinnx/out/pilot_b8/block_seg_n{197,69}_b8_fp.onnx
    #      and jetson-orinnx/out/pilot_b8/block_seg_n{197,69}_b8_qdq.onnx
    # copy both pairs to the Jetson (out/pilot_b8/), then run pilot_b8_build.sh on-device.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, write_json  # noqa: E402
from export_segments import BLOCK_IDX_FOR_EXPORT, load_model  # noqa: E402
from int8_qdq_experiment import build_quant_config, CALIB_BATCH_SIZE  # noqa: E402

try:
    import modelopt.torch.quantization as mtq
except ImportError as e:
    print(f"FATAL: nvidia-modelopt not importable ({e!r}). pip install nvidia-modelopt into "
          "this same host venv -- see this file's own docstring.", file=sys.stderr)
    raise

BATCH_SIZE = 8
PILOT_BUCKETS = (197, 69)  # the two extremes of the measured token-count range -- see docstring
MAX_ABS_DIFF_TOL = 1e-4
ONNX_OPSET = 17


def export_and_verify_fp(block, d_model: int, n_tokens: int, out_dir: Path, device: str):
    """BATCH_SIZE-8 analogue of export_segments.py's export_and_verify_one_bucket -- same
    verification discipline (onnxruntime max-abs-diff vs the same PyTorch block on the same
    input), different batch dimension only."""
    torch.manual_seed(0)
    dummy = torch.randn(BATCH_SIZE, n_tokens, d_model, dtype=torch.float32, device=device)

    block_cpu = block.cpu().eval()
    dummy_cpu = dummy.cpu()
    with torch.no_grad():
        pt_out = block_cpu(dummy_cpu).numpy()

    onnx_path = out_dir / f"block_seg_n{n_tokens}_b8_fp.onnx"
    torch.onnx.export(
        block_cpu, dummy_cpu, str(onnx_path),
        input_names=["tokens_in"], output_names=["tokens_out"],
        opset_version=ONNX_OPSET, do_constant_folding=True,
        dynamic_axes=None,  # FIXED shape (batch=8 AND n_tokens), deliberately -- pilot is a
        # single fixed-shape GO/NO-GO probe, not a general dynamic-batch deployment path.
        dynamo=False,  # same legacy-exporter reason as export_segments.py -- see that file's
        # own comment (fused_attn=False below already avoids the is_causal/tracer bug).
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["tokens_out"], {"tokens_in": dummy_cpu.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
    assert max_abs_diff <= MAX_ABS_DIFF_TOL, (
        f"FP export/PyTorch mismatch at n_tokens={n_tokens}, batch={BATCH_SIZE}: "
        f"max_abs_diff={max_abs_diff:.3e} > {MAX_ABS_DIFF_TOL:.0e} -- aborting."
    )
    return onnx_path, max_abs_diff


def export_and_verify_qdq(d_model: int, n_tokens: int, calib_npy_path: Path, out_dir: Path, device: str):
    """BATCH_SIZE-8 analogue of int8_qdq_experiment.py's quantize_and_export_one_bucket -- same
    quant_cfg (parent_class=nn.Linear, i.e. qkv/attn.proj/mlp.fc1/mlp.fc2 only), same calibration
    data, same QuantizeLinear/DequantizeLinear survival assertion. Fresh model load, same reason
    as the original: mtq.quantize() mutates its input module in place."""
    assert calib_npy_path.exists(), (
        f"{calib_npy_path} not found -- this pilot reuses export_segments.py's existing "
        f"calibration capture for N={n_tokens}; if missing, copy it from the Jetson's "
        "out/segments/calib/ (already captured there by the completed G1-HW run) rather than "
        "re-streaming ImageNet."
    )
    model = load_model()
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    model = model.to(device).eval()
    block = model.blocks[BLOCK_IDX_FOR_EXPORT]

    calib_data = np.load(calib_npy_path).astype(np.float32)
    assert calib_data.shape[1] == n_tokens, (
        f"{calib_npy_path} has n_tokens={calib_data.shape[1]}, expected {n_tokens}."
    )
    calib_tensor = torch.from_numpy(calib_data).to(device)

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, calib_tensor.shape[0], CALIB_BATCH_SIZE):
                m(calib_tensor[i:i + CALIB_BATCH_SIZE])

    print(f"  Quantizing (batch={BATCH_SIZE} pilot, same quant_cfg as int8_qdq_experiment.py)...")
    block = mtq.quantize(block, build_quant_config(), forward_loop)

    quantized_names = [name for name, m in block.named_modules()
                        if hasattr(m, "weight_quantizer") or hasattr(m, "input_quantizer")]
    EXPECTED = {"qkv", "proj", "fc1", "fc2"}
    found = {n.rsplit(".", 1)[-1] for n in quantized_names}
    assert EXPECTED <= found, f"Expected quantizers on {EXPECTED}, found suffixes {found}."

    dummy = torch.randn(BATCH_SIZE, n_tokens, d_model, device=device)
    with torch.no_grad():
        _ = block(dummy)  # confirm the quantized block runs cleanly at this batch size

    onnx_path = out_dir / f"block_seg_n{n_tokens}_b8_qdq.onnx"
    block_cpu, dummy_cpu = block.cpu(), dummy.cpu()
    torch.onnx.export(
        block_cpu, dummy_cpu, str(onnx_path),
        input_names=["tokens_in"], output_names=["tokens_out"],
        opset_version=ONNX_OPSET, do_constant_folding=True,
        dynamic_axes=None, dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    qdq_ops = [n.op_type for n in onnx_model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    assert qdq_ops, (
        f"No QuantizeLinear/DequantizeLinear nodes survived export at N={n_tokens}, batch={BATCH_SIZE} "
        "-- do NOT copy this ONNX to the Jetson, it would silently build a plain (non-quantized) engine."
    )
    return onnx_path, len(qdq_ops)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU -- this needs one, same as export_segments.py."

    print(f"=== pilot_b8_export.py: BATCH={BATCH_SIZE} FP + explicit-QDQ INT8 export, "
          f"buckets={PILOT_BUCKETS} ===")

    segments_dir = Path("jetson-orinnx/out/segments")  # repo-root-relative, matching
    # int8_qdq_experiment.py's own default (see that file's comment on why: this script is
    # invoked from the repo root, same CWD as every other host-side script in this folder).
    out_dir = ensure_out_dir("pilot_b8")

    model = load_model()
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    model = model.eval().requires_grad_(False)
    d_model = model.embed_dim
    block = model.blocks[BLOCK_IDX_FOR_EXPORT]

    results = {}
    for n_tokens in PILOT_BUCKETS:
        print(f"\n--- bucket n_tokens={n_tokens} ---")
        fp_path, fp_diff = export_and_verify_fp(block, d_model, n_tokens, out_dir, device)
        print(f"  FP:  {fp_path}  max_abs_diff={fp_diff:.3e}  OK")

        calib_npy = segments_dir / "calib" / f"calib_n{n_tokens}.npy"
        qdq_path, n_qdq_ops = export_and_verify_qdq(d_model, n_tokens, calib_npy, out_dir, device)
        print(f"  QDQ: {qdq_path}  qdq_ops={n_qdq_ops}  OK")

        results[n_tokens] = {
            "fp_onnx_path": str(fp_path), "fp_max_abs_diff": fp_diff,
            "qdq_onnx_path": str(qdq_path), "qdq_ops": n_qdq_ops,
        }

    manifest = {
        "batch_size": BATCH_SIZE, "buckets": list(PILOT_BUCKETS),
        "max_abs_diff_tol": MAX_ABS_DIFF_TOL, "results": results,
        "note": "Pilot export only -- see jetson-orinnx/README.md's Batch-8 INT8 pilot section and "
                 "jetson-orinnx/CHARACTERIZATION.md. Does not touch any registered G1-HW out/ artifact.",
    }
    write_json(out_dir / "pilot_b8_export_manifest.json", manifest)
    print(f"\nWrote {out_dir / 'pilot_b8_export_manifest.json'}")
    print(f"\nAll {len(PILOT_BUCKETS)} buckets exported and verified (FP: onnxruntime max-abs-diff; "
          "QDQ: real QuantizeLinear/DequantizeLinear nodes confirmed).")
    print("Next: copy jetson-orinnx/out/pilot_b8/*.onnx to the Jetson, then run pilot_b8_build.sh on-device.")
    for n_tokens, r in results.items():
        print(f"  scp {r['fp_onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/pilot_b8/")
        print(f"  scp {r['qdq_onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/pilot_b8/")


if __name__ == "__main__":
    main()
