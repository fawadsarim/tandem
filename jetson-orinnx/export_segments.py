"""jetson-orinnx/export_segments.py -- HOST-SIDE ONLY. Run this on a workstation
or cloud GPU with PyTorch installed (`pip install -r requirements-host.txt`
in your OWN venv, not jetson-orinnx/venv) -- it never runs on the Jetson itself.
The Jetson only ever sees the ONNX files this script produces (copy
jetson-orinnx/out/segments/ to the device, or re-clone the repo after committing
them).

WHAT THIS EXPORTS AND WHY (per-block-segment latency profiling)
-----------------------------------------------------------------
G1-HW needs a latency LUT keyed by (token-count bucket, precision) -- see
measure_lut.py. A DeiT/ViT transformer block (LayerNorm -> Attention ->
residual -> LayerNorm -> MLP -> residual) has IDENTICAL architecture
(dims, head count, MLP ratio) across all 12 blocks of deit_small_patch16_224
-- only the trained WEIGHT VALUES differ block-to-block, and weight values
do not affect latency (latency depends on tensor shapes and the op graph,
not on the numbers inside the tensors). This script therefore exports ONE
representative block (block index BLOCK_IDX_FOR_EXPORT, arbitrary choice --
see the constant below) at each of the 5 token-count buckets TANDEM's own
plan space actually produces for deit_small (see replan.py's docstring for
the derivation of these 5 buckets from MERGE_BLOCKS/MERGE_RS), instead of
exporting 12 blocks x 5 buckets = 60 near-identical graphs.

Each exported ONNX graph is verified against the SAME PyTorch block on the
SAME random input before being trusted: max-abs-diff must be <= 1e-4, or
this script aborts (never silently ships a graph that doesn't match).

Checkpoint pinning: `deit_small_patch16_224.fb_in1k` (the exact tag P2.1
resolved and pinned -- cross-checked against experiments/phase2_1_fleet/results/P2/
deit_small_patch16_224/P2_deit_small_patch16_224_summary.json's
`resolved_pretrained_tag` field, READ-ONLY, never written to). The loaded
model's own state-dict hash is also recorded in the manifest, same
mechanism as experiments/phase2_1_fleet/P2_tandem_pipeline.py's `hash_state_dict` (ported
here, not imported -- see jetson-orinnx/_common.py's module docstring for why
jetson-orinnx/ cannot import experiments/ code).

HF auth: via the HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) environment
variable ONLY, read automatically by huggingface_hub/timm -- this script
contains no token literal, and jetson-orinnx/self_test.py's secrets scan covers
every file in this folder including this one.

Usage (on the HOST, in your own venv):
    python export_segments.py
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, read_p2_summary, sha256_of_json_obj, write_json  # noqa: E402

MODEL_NAME = "deit_small_patch16_224"
EXPECTED_RESOLVED_TAG = "deit_small_patch16_224.fb_in1k"
BLOCK_IDX_FOR_EXPORT = 0  # arbitrary -- see module docstring (architecture is block-homogeneous)
TOKEN_BUCKETS = (197, 165, 133, 101, 69)  # see replan.py's docstring for the derivation
MAX_ABS_DIFF_TOL = 1e-4
ONNX_OPSET = 17
N_CALIB_IMAGES = 128


def hash_state_dict(model) -> str:
    """Ported verbatim from experiments/phase2_1_fleet/P2_tandem_pipeline.py (section
    'hash_state_dict') -- see this file's module docstring for why it is
    duplicated here rather than imported."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_model():
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True)
    except Exception:
        model = timm.create_model(EXPECTED_RESOLVED_TAG, pretrained=True)
    model = model.eval().requires_grad_(False)
    resolved_tag = getattr(getattr(model, "pretrained_cfg", None), "get", lambda *_: None)("tag") \
        if hasattr(model, "pretrained_cfg") else None
    resolved_tag = (model.pretrained_cfg or {}).get("architecture", None) if resolved_tag is None else resolved_tag
    return model


def derive_dims(model):
    d_model = model.embed_dim
    n_blocks = len(model.blocks)
    n_patches = model.patch_embed.num_patches
    n_tokens_total = n_patches + 1  # + CLS
    return d_model, n_blocks, n_patches, n_tokens_total


def export_and_verify_one_bucket(block: nn.Module, d_model: int, n_tokens: int, out_dir: Path):
    torch.manual_seed(0)
    dummy = torch.randn(1, n_tokens, d_model, dtype=torch.float32)

    with torch.no_grad():
        pt_out = block(dummy).numpy()

    onnx_path = out_dir / f"block_seg_n{n_tokens}.onnx"
    # dynamo=False: real bug found and fixed, 2026-07. As of torch 2.9, `torch.onnx.export`
    # defaults to the newer dynamo/torch.export-based exporter (`dynamo=True`) when not told
    # otherwise, and torch 2.11 (this project's pin) removed the automatic fallback to the
    # legacy exporter that used to mask this on failure -- the new path also needs the
    # `onnxscript` package, which is not in requirements-host.txt, so this call failed here
    # (before writing manifest.json, hence a downstream FileNotFoundError on that file, not
    # an error at this line). This script's own hand-rolled onnxruntime verification step
    # right below already re-checks the export is numerically correct, and every input here
    # is a fixed shape (`dynamic_axes=None`, deliberately, per this function's whole design) --
    # there is nothing to gain from the new exporter for this use case, so pin to the legacy
    # TorchScript-based one explicitly rather than take on a new dependency for no benefit.
    torch.onnx.export(
        block, dummy, str(onnx_path),
        input_names=["tokens_in"], output_names=["tokens_out"],
        opset_version=ONNX_OPSET, do_constant_folding=True,
        dynamic_axes=None,  # FIXED shape per bucket, deliberately -- see module docstring
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["tokens_out"], {"tokens_in": dummy.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
    assert max_abs_diff <= MAX_ABS_DIFF_TOL, (
        f"ONNX vs PyTorch mismatch at n_tokens={n_tokens}: max_abs_diff={max_abs_diff:.3e} > "
        f"{MAX_ABS_DIFF_TOL:.0e} -- aborting, refusing to ship a segment that doesn't match its "
        "PyTorch reference."
    )
    return onnx_path, max_abs_diff


def capture_calibration_activations(model, d_model: int, n_tokens_total: int, imagenet_dir: str, out_dir: Path):
    """Captures REAL activations entering block BLOCK_IDX_FOR_EXPORT, from
    N_CALIB_IMAGES real ImageNet images, for TensorRT INT8 range
    calibration (int8_calibrate.py, run later on-device). Random/synthetic
    calibration data would give a valid-but-poor INT8 range estimate --
    this project's standing principle is to calibrate from real, pinned
    data (same discipline as experiments/phase2_1_fleet/P2_tandem_pipeline.py's own
    `resolve_calibration`/`pin_calibration_indices`, not imported, ported
    in spirit only: local-directory-based here, since jetson-orinnx/ cannot
    depend on experiments/phase2_2_baselines/'s HF-streaming loader either).

    Local-directory-only (torchvision ImageFolder layout) is a deliberate
    design choice for an edge deployment: a Jetson calibration run is not
    expected to stream from HF hub the way the Colab notebooks do; a small
    (~128-image) local copy is pinned by content hash instead.

    STATED APPROXIMATION (documented, not hidden): buckets below 197
    tokens correspond to Ward-merged states this script does not simulate
    (that requires the full merge machinery, ported separately in
    paired_eval.py, which is NOT used here to keep export_segments.py
    torch-only-inference and dependency-light). Instead, real activations
    captured at the full 197-token width are deterministically
    strided-subsampled (position 0 = CLS always kept) down to each smaller
    bucket's token count. Ward-merged tokens are weighted averages of
    similar real tokens, so their value distribution is close in scale to
    a subsample of the originals -- adequate for INT8 RANGE calibration
    (which only needs approximately-correct clipping thresholds, not
    exact activation values), but this is an approximation and is
    recorded as such in the manifest.
    """
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
    full_activations = captured["x"].numpy()  # (N_CALIB_IMAGES, 197, d_model)
    assert full_activations.shape == (len(indices), n_tokens_total, d_model)

    calib_dir = ensure_out_dir("segments", "calib")
    calib_paths = {}
    for n_tokens in TOKEN_BUCKETS:
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
        calib_paths[n_tokens] = str(path.relative_to(out_dir.parent.parent)).replace("\\", "/")

    return {
        "n_calib_images": len(indices), "calib_hash": calib_hash, "imagenet_dir": str(imagenet_dir),
        "approximation_note": (
            "Sub-197-token buckets are strided subsamples of the real 197-token activations, not "
            "true Ward-merge outputs -- adequate for INT8 range calibration only. See "
            "export_segments.py's capture_calibration_activations docstring."
        ),
        "calib_paths": calib_paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", default=None,
                         help="Local ImageFolder-layout ImageNet directory for INT8 calibration "
                              "capture. If omitted, ONNX export/verification still runs, but "
                              "calibration data is skipped -- build_engines.sh's INT8 step will "
                              "fail loudly until this is provided and export_segments.py is re-run.")
    args = parser.parse_args()

    print("=== export_segments.py (HOST-SIDE, not for the Jetson) ===")
    model = load_model()

    # Force the non-fused (manual matmul) attention path -- real bug found and fixed, 2026-07:
    # timm's fused_attn path calls F.scaled_dot_product_attention(..., is_causal=is_causal), and
    # under torch.jit.trace (the legacy ONNX exporter this script deliberately uses -- see
    # export_and_verify_one_bucket's dynamo=False), that is_causal argument gets boxed as a
    # 0-dim Tensor by the tracer instead of staying a plain Python bool, which SDPA's C++
    # binding rejects outright: `TypeError: scaled_dot_product_attention(): argument
    # 'is_causal' must be bool, not Tensor`. The non-fused path (matmul + softmax + matmul, no
    # is_causal kwarg at all) traces cleanly and is mathematically identical up to float
    # precision -- well within this script's own MAX_ABS_DIFF_TOL=1e-4 verification -- so
    # disabling the fused kernel here costs nothing but a little export-time speed. Set
    # directly on each already-constructed Attention submodule, not via
    # timm.layers.set_fused_attn() (which only affects modules constructed AFTER it's called --
    # this model is already built above, so that global toggle would be a no-op here).
    for _m in model.modules():
        if hasattr(_m, "fused_attn"):
            _m.fused_attn = False

    resolved_tag = (getattr(model, "pretrained_cfg", {}) or {}).get("tag", None)
    resolved_full = f"{MODEL_NAME}.{resolved_tag}" if resolved_tag else EXPECTED_RESOLVED_TAG
    print(f"Loaded {MODEL_NAME}, resolved pretrained tag: {resolved_full}")

    d_model, n_blocks, n_patches, n_tokens_total = derive_dims(model)
    print(f"Derived dims: d_model={d_model} n_blocks={n_blocks} n_patches={n_patches} "
          f"n_tokens_total={n_tokens_total}")

    # Cross-check against P2.1's canonical summary -- READ-ONLY reference value, per the top-level
    # instruction ("reference values read from summaries at runtime"). This is a cross-check, not a
    # source of truth: the dims above are derived fresh from the loaded model, independent of P2.1.
    p2_summary = read_p2_summary(MODEL_NAME)
    p2_dims = p2_summary["model_dims"]
    assert d_model == p2_dims["d"] and n_patches == p2_dims["n_patches"] and \
        n_tokens_total == p2_dims["n_tokens_total"], (
        f"Model dims derived here ({d_model}, {n_patches}, {n_tokens_total}) disagree with P2.1's "
        f"pinned reference ({p2_dims}) -- checkpoint/timm-version drift since P2.1 ran. Halting "
        "rather than exporting segments for a possibly-different architecture."
    )
    assert p2_summary["resolved_pretrained_tag"] == EXPECTED_RESOLVED_TAG, (
        f"P2.1's pinned resolved_pretrained_tag ({p2_summary['resolved_pretrained_tag']!r}) does not "
        f"match this script's EXPECTED_RESOLVED_TAG ({EXPECTED_RESOLVED_TAG!r})."
    )
    print(f"Cross-check vs P2.1 canonical summary (read-only): dims and resolved tag MATCH.")

    checkpoint_hash = hash_state_dict(model)
    print(f"Checkpoint state-dict hash: {checkpoint_hash}")

    block = model.blocks[BLOCK_IDX_FOR_EXPORT]
    out_dir = ensure_out_dir("segments")

    segments = []
    for n_tokens in TOKEN_BUCKETS:
        onnx_path, max_abs_diff = export_and_verify_one_bucket(block, d_model, n_tokens, out_dir)
        rel_path = onnx_path.relative_to(out_dir.parent.parent)
        print(f"  n_tokens={n_tokens:3d}: {rel_path}  max_abs_diff={max_abs_diff:.3e}  OK")
        segments.append({
            "n_tokens": n_tokens, "onnx_path": str(rel_path).replace("\\", "/"),
            "max_abs_diff": max_abs_diff, "verified": True,
        })

    calibration = None
    if args.imagenet_dir:
        calibration = capture_calibration_activations(model, d_model, n_tokens_total, args.imagenet_dir, out_dir)
        print(f"\nCaptured calibration activations: {calibration['n_calib_images']} images, "
              f"calib_hash={calibration['calib_hash']}")
    else:
        print("\n--imagenet-dir not given -- SKIPPING calibration capture (INT8 build will not be "
              "possible until this is re-run with --imagenet-dir).")

    manifest = {
        "model_name": MODEL_NAME,
        "resolved_pretrained_tag": resolved_full,
        "checkpoint_state_hash": checkpoint_hash,
        "block_idx_exported": BLOCK_IDX_FOR_EXPORT,
        "block_homogeneity_assumption": (
            "All 12 blocks of deit_small_patch16_224 share identical architecture (dims, head count, "
            "MLP ratio); only trained weight VALUES differ, which do not affect latency. One block "
            "(index above) is exported and assumed representative of all 12 for LATENCY purposes only "
            "-- this manifest is never used for accuracy, only for the latency LUT."
        ),
        "d_model": d_model, "n_blocks": n_blocks, "n_patches": n_patches, "n_tokens_total": n_tokens_total,
        "onnx_opset": ONNX_OPSET, "max_abs_diff_tol": MAX_ABS_DIFF_TOL,
        "segments": segments,
        "calibration": calibration,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"\nWrote {manifest_path}")
    print(f"\nAll {len(TOKEN_BUCKETS)} buckets exported and verified (max-abs-diff <= {MAX_ABS_DIFF_TOL:.0e}).")
    print(f"Copy {out_dir} to the Jetson (or commit + re-clone) before running build_engines.sh.")


if __name__ == "__main__":
    main()
