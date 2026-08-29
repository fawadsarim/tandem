"""jetson-orinnx/lut_ext_export.py -- ON-DEVICE (deliberate deviation from export_segments.py's
HOST/Colab-only convention -- see DEVIATION note below). LUT EXTENSION, step 1 -- see
jetson-orinnx/README.md's "LUT extension" section. Exports plain FP16-target block-segment ONNX at
THREE new token buckets, N=53/37/21, extending the achievable-latency floor below the original
5-bucket LUT's N=69 minimum. FP16 ONLY -- INT8 is CLOSED per CHARACTERIZATION.md's batch-1/
batch-8 NO-GO ladder; this script does not touch quantization at all, so that finding is not
being re-litigated here.

DEVIATION from export_segments.py's HOST-only convention, justified: this export needs only
torch+timm+onnx+onnxruntime (no calibration capture, no nvidia-modelopt -- FP16 needs neither),
and this device's own venv already has torch/timm/onnx proven working throughout this project.
`onnxruntime` was the one missing piece -- checked for real, 2026-07: unlike nvidia-modelopt
(which pulled a conflicting torch/CUDA tree, see the batch-8 pilot's README.md section), it has
a clean aarch64 wheel with only `flatbuffers` genuinely missing (numpy/packaging/protobuf were
already satisfied at compatible versions) -- installed into this venv with zero risk to the
existing torch/CUDA setup (confirmed: torch.cuda.is_available() still True afterward). Running
this export here avoids an unnecessary Colab round-trip for what turned out to be a safe,
lightweight dependency.

Reuses export_segments.py's export_and_verify_one_bucket/load_model/BLOCK_IDX_FOR_EXPORT
directly (jetson-orinnx-internal import, same established exception _common.py's own docstring
documents) -- same MAX_ABS_DIFF_TOL=1e-4 onnxruntime-based verification as the original 5
buckets, not a different/weaker check.

Usage (on-device, in jetson-orinnx/venv): ./venv/bin/python3 lut_ext_export.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, write_json  # noqa: E402
from export_segments import (  # noqa: E402
    BLOCK_IDX_FOR_EXPORT, derive_dims, export_and_verify_one_bucket, hash_state_dict, load_model,
)

NEW_BUCKETS = (53, 37, 21)

# Real environment finding, 2026-07 (not a code change needed, recorded for provenance): on
# this device's torch==2.12.0, export_and_verify_one_bucket's dynamo=False no longer avoids the
# dynamo/torch.export-based exporter path the way it did on torch==2.11.0 (the Colab host pin,
# see export_segments.py's own comment on why dynamo=False was originally added) -- confirmed
# by the exporter's own log lines ("[torch.onnx] Obtain model graph... with torch.export.
# export(..., strict=False)") appearing regardless. This is why onnxscript ended up a real,
# unavoidable dependency here (see module docstring) rather than an optional one -- not a
# regression in this script, a torch-version behavior difference. Verification (max-abs-diff)
# still passed cleanly at all 3 buckets, so the exported graphs are trustworthy either way.


def main():
    print(f"=== lut_ext_export.py (ON-DEVICE, FP16-only -- see module docstring's DEVIATION "
          f"note): buckets={NEW_BUCKETS} ===")
    model = load_model()
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    model = model.eval().requires_grad_(False)
    d_model, n_blocks, n_patches, n_tokens_total = derive_dims(model)
    checkpoint_hash = hash_state_dict(model)
    print(f"Loaded model, checkpoint_hash={checkpoint_hash}, d_model={d_model}")

    block = model.blocks[BLOCK_IDX_FOR_EXPORT]
    out_dir = ensure_out_dir("lut_ext")

    segments = []
    for n_tokens in NEW_BUCKETS:
        onnx_path, max_abs_diff = export_and_verify_one_bucket(block, d_model, n_tokens, out_dir)
        rel_path = onnx_path.relative_to(out_dir.parent.parent)
        print(f"  n_tokens={n_tokens:3d}: {rel_path}  max_abs_diff={max_abs_diff:.3e}  OK")
        segments.append({
            "n_tokens": n_tokens, "onnx_path": str(rel_path).replace("\\", "/"),
            "max_abs_diff": max_abs_diff, "verified": True,
        })

    manifest = {
        "checkpoint_state_hash": checkpoint_hash, "block_idx_exported": BLOCK_IDX_FOR_EXPORT,
        "d_model": d_model, "new_buckets": list(NEW_BUCKETS), "segments": segments,
        "fp16_only_note": (
            "INT8 is CLOSED per CHARACTERIZATION.md (batch-1 and batch-8 NO-GO, clean fused "
            "Q/DQ kernels confirmed, penalty worsens not improves with batch -- see "
            "jetson-orinnx/README.md's Batch-8 INT8 pilot section). This LUT extension is FP16-only "
            "and does not re-litigate that finding."
        ),
    }
    write_json(out_dir / "lut_ext_export_manifest.json", manifest)
    print(f"\nWrote {out_dir / 'lut_ext_export_manifest.json'}")
    print(f"All {len(NEW_BUCKETS)} buckets exported and verified (max-abs-diff <= 1e-4, same "
          "tolerance/mechanism as the original 5 buckets).")
    print("Next: bash lut_ext_build.sh && ./venv/bin/python3 lut_ext_measure.py")


if __name__ == "__main__":
    main()
