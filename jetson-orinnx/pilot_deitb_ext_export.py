"""jetson-orinnx/pilot_deitb_ext_export.py -- HOST-SIDE ONLY (Colab/workstation GPU), NOT the Jetson.
DeiT-B CROSSOVER CHARACTERIZATION, step 1 -- see jetson-orinnx/README.md's "DeiT-B crossover
characterization" section. Fills the gap between the DeiT-B pilot's two buckets (N=197: clear
NO-GO, +19.3%; N=69: BORDERLINE, -3.09%) with TWO more buckets, N=133 and N=101, around the
naive 2-point linear-interpolation crossover estimate (~N=87), so the crossover can be located
from FOUR real measured points instead of extrapolated from two.

SAME reasons/mechanism as pilot_deitb_export.py (not repeated in full -- see that file's own
docstring): HOST-side because nvidia-modelopt cannot import under this device's JetPack torch
(confirmed during the batch-8 pilot); DeiT-B needs its own calibration capture (cannot reuse
DeiT-S's calib_n*.npy); same explicit-QDQ path (qkv/attn.proj/fc1/fc2 only); same
export_and_verify_one_bucket/hash_state_dict/build_quant_config reuse.

Writes to out/pilot_deitb_ext/ -- does NOT touch out/pilot_deitb/ (the original pilot's
committed, immutable result).

Usage (from the REPO ROOT):
    pip install nvidia-modelopt
    python jetson-orinnx/pilot_deitb_ext_export.py --imagenet-dir <local ImageFolder dir>
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
          "this same host venv.", file=sys.stderr)
    raise

MODEL_NAME = "deit_base_patch16_224"
EXPECTED_RESOLVED_TAG = "deit_base_patch16_224.fb_in1k"
BLOCK_IDX_FOR_EXPORT = 0
EXT_BUCKETS = (133, 101)  # fills the gap around the ~N=87 naive crossover estimate
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
    """Same mechanism as pilot_deitb_export.py's own function -- reimplemented here (not
    imported) because it hardcodes that file's PILOT_BUCKETS=(197,69), not this file's (133,101)."""
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

    calib_dir = ensure_out_dir("pilot_deitb_ext", "calib")
    calib_paths = {}
    for n_tokens in EXT_BUCKETS:
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
    """Mirrors pilot_deitb_export.py's own function exactly -- see that file's docstring."""
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
          "int8_qdq_experiment.py, calibrated on this run's own real-activation set)...")
    block = mtq.quantize(block, build_quant_config(), forward_loop)

    quantized_names = [name for name, m in block.named_modules()
                        if hasattr(m, "weight_quantizer") or hasattr(m, "input_quantizer")]
    EXPECTED = {"qkv", "proj", "fc1", "fc2"}
    found = {n.rsplit(".", 1)[-1] for n in quantized_names}
    assert EXPECTED <= found, (
        f"Expected quantizers on {EXPECTED}, found suffixes {found} from {quantized_names}."
    )

    dummy = torch.randn(1, n_tokens, d_model, device=device)
    with torch.no_grad():
        _ = block(dummy)

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
        f"No QuantizeLinear/DequantizeLinear nodes for N={n_tokens} -- do not copy to the Jetson."
    )
    return len(qdq_ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True,
                         help="Local ImageFolder-layout directory for this run's OWN 128-image "
                              "calibration capture (cannot reuse pilot_deitb_export.py's).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "No GPU."

    print(f"=== pilot_deitb_ext_export.py: {MODEL_NAME}, FP + explicit-QDQ INT8, batch=1, "
          f"buckets={EXT_BUCKETS} (crossover characterization) ===")
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
    assert d_model == 768 and n_blocks == 12 and n_tokens_total == 197, (
        f"DeiT-B dims unexpected: d_model={d_model} n_blocks={n_blocks} n_tokens_total={n_tokens_total}."
    )

    block = model.blocks[BLOCK_IDX_FOR_EXPORT]
    out_dir = ensure_out_dir("pilot_deitb_ext")

    print("\n--- Calibration capture (this run's own -- see module docstring) ---")
    calibration = capture_calibration_activations(model, d_model, n_tokens_total, args.imagenet_dir, out_dir)
    print(f"Captured {calibration['n_calib_images']} images, calib_hash={calibration['calib_hash']}")

    results = {}
    for n_tokens in EXT_BUCKETS:
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
        "n_tokens_total": n_tokens_total, "buckets": list(EXT_BUCKETS),
        "calibration": {"n_calib_images": calibration["n_calib_images"], "calib_hash": calibration["calib_hash"]},
        "max_abs_diff_tol": MAX_ABS_DIFF_TOL, "results": results,
        "note": "Crossover characterization -- fills the gap between the original pilot's "
                "N=197/N=69 buckets. Does not touch out/pilot_deitb/'s committed result.",
    }
    write_json(out_dir / "pilot_deitb_ext_export_manifest.json", manifest)
    print(f"\nWrote {out_dir / 'pilot_deitb_ext_export_manifest.json'}")
    print(f"\nAll {len(EXT_BUCKETS)} buckets exported and verified.")
    print("Next: copy jetson-orinnx/out/pilot_deitb_ext/*.onnx to the Jetson, then run pilot_deitb_ext_build.sh.")
    for n_tokens, r in results.items():
        print(f"  scp {r['fp_onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/pilot_deitb_ext/")
        print(f"  scp {r['qdq_onnx_path']} jetson-orinnx:~/tandem/jetson-orinnx/out/pilot_deitb_ext/")


if __name__ == "__main__":
    main()
