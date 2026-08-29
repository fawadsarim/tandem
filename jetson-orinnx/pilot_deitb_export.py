"""jetson-orinnx/pilot_deitb_export.py -- HOST-SIDE ONLY (Colab/workstation GPU), NOT the Jetson.
DeiT-B INT8 PILOT, Part 1 -- see jetson-orinnx/README.md's "DeiT-B INT8 pilot" section and
jetson-orinnx/out/pilot_deitb/PREREGISTRATION.md for the hypothesis and decision rule.

WHY HOST-SIDE, CONFIRMED NOT ON-DEVICE (not re-tried here): `nvidia-modelopt`, needed for the
QDQ half, cannot import on this Jetson's JetPack-specific torch build -- `modelopt.torch`
unconditionally imports `torch.distributed.tensor`, which needs `torch._C._distributed_c10d`, a
compiled symbol this device's `torch==2.12.0` (built with `USE_DISTRIBUTED=0`, standard for
edge-device wheels) does not have. Confirmed for real during the batch-8 pilot (see
jetson-orinnx/README.md's "Batch-8 INT8 pilot" section for the full investigation) -- this is a known,
documented, structural blocker, not re-investigated here.

MODEL: deit_base_patch16_224.fb_in1k, PINNED tag (same pinning discipline as
export_segments.py) -- d_model=768, 12 blocks, N=197 tokens, asserted after loading (halts
rather than silently proceeding if timm/checkpoint drift changes these).

BATCH=1 (not batch=8 -- this pilot tests a DIFFERENT axis, model scale/compute-intensity, not
batch size; batch=1 matches the ORIGINAL G1-HW/CHARACTERIZATION.md regime so the two pilots are
comparable on the SAME axis they each vary).

TWO BUCKETS ONLY, N=197 and N=69 -- the same two extremes used in the batch-8 pilot, so results
are directly comparable across model scales (see README.md's Part 3 "cross-scale ratio table").

CALIBRATION: DeiT-B is a DIFFERENT model from DeiT-S (different weights, different activation
distributions) -- the existing calib_n197.npy/calib_n69.npy (captured from DeiT-S) CANNOT be
reused. This script captures its own 128-image calibration set, same mechanism as
export_segments.py's capture_calibration_activations (real activations at N=197, strided-
subsampled down to N=69 -- see that function's own docstring for the full reasoning),
reimplemented here (not imported) because the original hardcodes DeiT-S's TOKEN_BUCKETS;
folded into this single script rather than a separate capture step, since this pilot is scoped
to exactly 2 buckets, not the full 5-bucket production grid.

REUSES export_segments.py's export_and_verify_one_bucket/hash_state_dict (fully generic, no
DeiT-S-specific hardcoding) and int8_qdq_experiment.py's build_quant_config (same quant_cfg:
Q/DQ on qkv/attn.proj/mlp.fc1/mlp.fc2 only, nothing else) directly -- jetson-orinnx-internal imports,
same established exception _common.py's own docstring documents. The QDQ export/quantize flow
itself is reimplemented (mirroring int8_qdq_experiment.py's quantize_and_export_one_bucket
closely) because that function hardcodes export_segments.py's load_model (DeiT-S).

Usage (from the REPO ROOT, same convention as export_segments.py/int8_qdq_experiment.py):
    pip install nvidia-modelopt
    python jetson-orinnx/pilot_deitb_export.py --imagenet-dir <local ImageFolder dir>
    # produces jetson-orinnx/out/pilot_deitb/block_seg_n{197,69}.onnx (FP)
    #      and jetson-orinnx/out/pilot_deitb/block_seg_n{197,69}_qdq.onnx (INT8)
    # copy all 4 to the Jetson, then run pilot_deitb_build.sh on-device.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, sha256_of_json_obj, write_json  # noqa: E402
from export_segments import export_and_verify_one_bucket, hash_state_dict  # noqa: E402
from int8_qdq_experiment import build_quant_config  # noqa: E402

try:
    import modelopt.torch.quantization as mtq
except ImportError as e:
    print(f"FATAL: nvidia-modelopt not importable ({e!r}). pip install nvidia-modelopt into "
          "this same host venv -- see this file's own docstring.", file=sys.stderr)
    raise

MODEL_NAME = "deit_base_patch16_224"
EXPECTED_RESOLVED_TAG = "deit_base_patch16_224.fb_in1k"
BLOCK_IDX_FOR_EXPORT = 0
PILOT_BUCKETS = (197, 69)
N_CALIB_IMAGES = 128
CALIB_BATCH_SIZE = 16
MAX_ABS_DIFF_TOL = 1e-4
ONNX_OPSET = 17


def load_model():
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True)
    except Exception:
        model = timm.create_model(EXPECTED_RESOLVED_TAG, pretrained=True)
    return model.eval().requires_grad_(False)


def capture_calibration_activations(model, d_model: int, n_tokens_total: int, imagenet_dir: str, out_dir: Path):
    """See module docstring's CALIBRATION note -- same mechanism as export_segments.py's own
    function, reimplemented for PILOT_BUCKETS=(197,69) only."""
    import torchvision.datasets as tvd
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = tvd.ImageFolder(imagenet_dir, transform=tf)
    assert len(ds) >= N_CALIB_IMAGES, f"{imagenet_dir} has only {len(ds)} images, need >= {N_CALIB_IMAGES}"
    indices = list(range(0, len(ds), max(1, len(ds) // N_CALIB_IMAGES)))[:N_CALIB_IMAGES]
    identifiers = [ds.samples[i][0] for i in indices]
    calib_hash = sha256_of_json_obj(sorted(identifiers))

    captured = {}

    def hook(module, args):
        captured["x"] = args[0].detach().clone()

    h = model.blocks[BLOCK_IDX_FOR_EXPORT].register_forward_pre_hook(hook)
    batch = torch.stack([ds[i][0] for i in indices], dim=0)
    with torch.no_grad():
        model.forward_features(batch) if hasattr(model, "forward_features") else model(batch)
    h.remove()
    assert "x" in captured, "calibration forward hook never fired -- model API mismatch"
    full_activations = captured["x"].numpy()
    assert full_activations.shape == (len(indices), n_tokens_total, d_model)

    calib_dir = ensure_out_dir("pilot_deitb", "calib")
    calib_paths = {}
    for n_tokens in PILOT_BUCKETS:
        if n_tokens == n_tokens_total:
            sub = full_activations
        else:
            n_patch_tokens = n_tokens - 1
            stride = (n_tokens_total - 1) / n_patch_tokens
            patch_idx = np.unique(np.round(np.arange(n_patch_tokens) * stride).astype(int))
            patch_idx = patch_idx[patch_idx < n_tokens_total - 1][:n_patch_tokens]
            keep = np.concatenate([[0], 1 + patch_idx])
            sub = full_activations[:, keep, :]
        path = calib_dir / f"calib_n{n_tokens}.npy"
        np.save(path, sub.astype(np.float32))
        calib_paths[n_tokens] = str(path)

    return {"n_calib_images": len(indices), "calib_hash": calib_hash, "calib_paths": calib_paths}


def quantize_and_export_one_bucket(n_tokens: int, calib_npy_path: Path, onnx_out_path: Path, device: str):
    """Mirrors int8_qdq_experiment.py's quantize_and_export_one_bucket exactly (same quant_cfg,
    same calibrate-then-export flow, same QuantizeLinear/DequantizeLinear survival assertion) --
    not imported, since that function hardcodes export_segments.py's load_model/
    BLOCK_IDX_FOR_EXPORT (DeiT-S). Fresh model load, same reason as the original: mtq.quantize()
    mutates its input module in place."""
    model = load_model().to(device)
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    block = model.blocks[BLOCK_IDX_FOR_EXPORT]
    d_model = model.embed_dim

    calib_data = np.load(calib_npy_path).astype(np.float32)
    assert calib_data.shape[1] == n_tokens, (
        f"{calib_npy_path} has n_tokens={calib_data.shape[1]}, expected {n_tokens}."
    )
    calib_tensor = torch.from_numpy(calib_data).to(device)
    print(f"  Loaded {calib_tensor.shape[0]} real calibration activations from {calib_npy_path}")

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, calib_tensor.shape[0], CALIB_BATCH_SIZE):
                m(calib_tensor[i:i + CALIB_BATCH_SIZE])

    print("  Quantizing (Q/DQ on qkv/attn.proj/mlp.fc1/mlp.fc2, INT8, same quant_cfg as "
          "int8_qdq_experiment.py, calibrated on DeiT-B's own real-activation set)...")
    block = mtq.quantize(block, build_quant_config(), forward_loop)

    quantized_names = [name for name, m in block.named_modules()
                        if hasattr(m, "weight_quantizer") or hasattr(m, "input_quantizer")]
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
    block_cpu, dummy_cpu = block.cpu(), dummy.cpu()
    torch.onnx.export(
        block_cpu, dummy_cpu, str(onnx_out_path),
        input_names=["tokens_in"], output_names=["tokens_out"],
        opset_version=ONNX_OPSET, do_constant_folding=True, dynamic_axes=None, dynamo=False,
    )
    print(f"  Wrote {onnx_out_path}")

    onnx_model = onnx.load(str(onnx_out_path))
    qdq_ops = [n.op_type for n in onnx_model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    print(f"  QuantizeLinear/DequantizeLinear nodes in exported graph: {len(qdq_ops)}")
    assert qdq_ops, (
        f"No QuantizeLinear/DequantizeLinear nodes in the exported ONNX graph for N={n_tokens} "
        "-- the Q/DQ export did not survive. Do NOT copy this ONNX to the Jetson as-is."
    )
    return len(qdq_ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True,
                         help="Local ImageFolder-layout directory for this pilot's OWN "
                              "128-image calibration capture (DeiT-B needs its own, cannot "
                              "reuse DeiT-S's calib_n*.npy -- see module docstring).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU -- this needs one, same as export_segments.py."

    print(f"=== pilot_deitb_export.py: {MODEL_NAME}, FP + explicit-QDQ INT8, batch=1, "
          f"buckets={PILOT_BUCKETS} ===")
    model = load_model()
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    d_model = model.embed_dim
    n_blocks = len(model.blocks)
    n_patches = model.patch_embed.num_patches
    n_tokens_total = n_patches + 1
    checkpoint_hash = hash_state_dict(model)
    resolved_tag = (getattr(model, "pretrained_cfg", {}) or {}).get("tag", None)
    resolved_full = f"{MODEL_NAME}.{resolved_tag}" if resolved_tag else EXPECTED_RESOLVED_TAG
    print(f"Loaded {resolved_full}, checkpoint_hash={checkpoint_hash}")
    print(f"d_model={d_model} n_blocks={n_blocks} n_tokens_total={n_tokens_total} (expect 768, 12, 197)")
    assert d_model == 768 and n_blocks == 12 and n_tokens_total == 197, (
        f"DeiT-B dims unexpected: d_model={d_model} n_blocks={n_blocks} n_tokens_total={n_tokens_total} "
        "-- halting rather than proceeding with a model that doesn't match the pinned architecture."
    )

    block = model.blocks[BLOCK_IDX_FOR_EXPORT]
    out_dir = ensure_out_dir("pilot_deitb")

    print("\n--- Calibration capture (this pilot's own, DeiT-B-specific -- see module docstring) ---")
    calibration = capture_calibration_activations(model, d_model, n_tokens_total, args.imagenet_dir, out_dir)
    print(f"Captured {calibration['n_calib_images']} images, calib_hash={calibration['calib_hash']}")

    results = {}
    for n_tokens in PILOT_BUCKETS:
        print(f"\n--- bucket n_tokens={n_tokens} ---")
        fp_path, fp_diff = export_and_verify_one_bucket(block, d_model, n_tokens, out_dir)
        print(f"  FP:  {fp_path}  max_abs_diff={fp_diff:.3e}  OK")

        qdq_path = out_dir / f"block_seg_n{n_tokens}_qdq.onnx"
        calib_npy = Path(calibration["calib_paths"][n_tokens])
        n_qdq_ops = quantize_and_export_one_bucket(n_tokens, calib_npy, qdq_path, device)
        print(f"  QDQ: {qdq_path}  qdq_ops={n_qdq_ops}  OK")

        results[n_tokens] = {
            "fp_onnx_path": str(fp_path), "fp_max_abs_diff": fp_diff,
            "qdq_onnx_path": str(qdq_path), "qdq_ops": n_qdq_ops,
        }

    manifest = {
        "model_name": MODEL_NAME, "resolved_pretrained_tag": resolved_full,
        "checkpoint_state_hash": checkpoint_hash, "d_model": d_model, "n_blocks": n_blocks,
        "n_tokens_total": n_tokens_total, "buckets": list(PILOT_BUCKETS),
        "calibration": {"n_calib_images": calibration["n_calib_images"], "calib_hash": calibration["calib_hash"]},
        "max_abs_diff_tol": MAX_ABS_DIFF_TOL, "results": results,
    }
    write_json(out_dir / "pilot_deitb_export_manifest.json", manifest)
    print(f"\nWrote {out_dir / 'pilot_deitb_export_manifest.json'}")
    print(f"\nAll {len(PILOT_BUCKETS)} buckets exported and verified (FP: onnxruntime max-abs-diff; "
          "QDQ: real QuantizeLinear/DequantizeLinear nodes confirmed).")
    print("Next: copy jetson-orinnx/out/pilot_deitb/*.onnx to the Jetson, then run pilot_deitb_build.sh on-device.")
    for n_tokens, r in results.items():
        print(f"  scp {r['fp_onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/pilot_deitb/")
        print(f"  scp {r['qdq_onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/pilot_deitb/")


if __name__ == "__main__":
    main()
