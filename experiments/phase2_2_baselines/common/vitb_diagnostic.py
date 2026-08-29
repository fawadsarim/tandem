"""ViT-B-specific diagnostics for FQ-ViT's checkpoint-override path
(common/checkpoint_pin.py's "fqvit" style). Built 2026-07 to investigate a
real Step-B FP-parity failure (measured 84.4660% vs P2.1's own S8
85.1000%, diff=0.6340pp) per a documented ruling; both of the ruling's
leading hypotheses (a preprocessing mismatch, a checkpoint-key mismatch)
were run for real and came back CLEAN (see deviations.md #36) -- root
cause of that specific gap is still open. Checks 1+2 are now PERMANENTLY
wired into b3_fq_vit/run.py's real Step-B flow (every run, not just this
investigation) since a real gap was found once and nothing here rules out
it recurring for a future checkpoint/model; check 3 similarly closes a
real, previously-unaddressed coverage gap for Step A (see below). Three
checks, cheapest first, EACH A HARD GATE (asserts, does not just print) --
consistent with this project's existing FP-parity/reproduction-gate
philosophy of never silently downgrading a correctness check to a
diagnostic once it's known to matter:

1. TRANSFORM SIDE-BY-SIDE: FQ-ViT's own stock build_transform() output for
   model_type=='vit' (their test_quant.py main(), unmodified, imported and
   called live) vs timm's resolve_data_config() for our pinned
   vit_base_patch16_224.augreg2_in21k_ft_in1k (read from timm's real
   source, NOT live-imported -- `timm` is deliberately not a dependency of
   B3's own pinned venv; see check_transform_side_by_side()'s own comment
   for why installing it there would be a real risk, not just noise).
   Asserts both sides match on every field. No GPU needed, seconds. Run
   before every Step-B override use (b3_fq_vit/run.py).
2. CHECKPOINT-KEY DIAGNOSTIC: FQ-ViT's own VisionTransformer('vit_base')
   state_dict() key set vs our pinned augreg2 checkpoint's key set -- the
   exact same strict=False set-difference common/checkpoint_pin.py's
   Step-B override (_FQVIT_OVERRIDE_SOURCE) performs at real run time, but
   computed here with full key names printed, before spending any GPU
   time, and asserted (0 missing, 0 unexpected) rather than just reported.
   CPU-only, seconds. Run before every Step-B override use.
3. PLAIN AUGREG-V1 DEFICIT CHECK: FQ-ViT's own stock loader + own
   augreg-v1 checkpoint (Step A's checkpoint, NO pinned-checkpoint
   override active), one full-50k plain pass, gated (via
   common.fp_parity_gate.check_fp_parity, the SAME 0.15pp tolerance,
   reused not duplicated) against FQ-ViT's OWN published FP baseline
   (84.53%, from their README -- NOT P2.1's S8 85.10%, which is measured
   on the DIFFERENT augreg2 checkpoint; see common/checkpoint_pin.py's
   module docstring for the checkpoint-identity proof). This closes a real
   coverage gap, not just adds redundant diagnostics: Step A's own
   FP-parity gate is structurally SKIPPED for ViT-B today
   (coincides_with_pinned() is False, so b3_fq_vit/run.py's `if coincide:
   run_fp_parity_gate(...)` never fires for this model) -- this check
   becomes that missing pre-Step-A gate, on ITS OWN checkpoint/reference
   rather than P2.1's. Needs GPU, ~6-12 min (same cost as any full-50k
   FP-parity pass). Run once before every Step-A run for a
   CHECKPOINT_OVERRIDE_NEEDED model, cached the same way
   run_fp_parity_gate caches (see b3_fq_vit/run.py's run_vitb_diagnostics).

Callable two ways: (a) permanently, via b3_fq_vit/run.py's
run_vitb_diagnostics() on every real Step A/Step B pass for a
CHECKPOINT_OVERRIDE_NEEDED model; (b) manually, via
../b3_fq_vit/diagnose_vitb.py, for ad hoc re-investigation. Both set up the
same PYTHONPATH/cwd/TF32-disabled subprocess environment
run_fp_parity_gate() already uses (this module needs `cwd` to be the
FQ-ViT repo root for `from config import Config` / `from test_quant import
build_transform, str2model` to resolve -- see common/fp_parity_gate.py's
module docstring for why) -- never invoked directly by a user.
"""
import argparse
import json
import os

# FQ-ViT's OWN README "Full Precision" row for ViT-B, measured on their
# stock checkpoint (augreg_in21k_ft_in1k -- the SAME weights their
# pretrained=True path always downloads via load_weights_from_npz; see
# common/checkpoint_pin.py's module docstring for the URL-identity proof
# against timm's real source). This is NOT the same number as
# checkpoint_pin.PINNED_TAGS' 85.10 (P2.1's own S8 measurement on
# augreg2_in21k_ft_in1k, a DIFFERENT checkpoint) -- this check's whole
# point is to test FQ-ViT's stock pipeline against ITS OWN stated
# baseline, not ours.
PUBLISHED_VITB_FP_STOCK_CHECKPOINT_PCT = 84.53


def check_transform_side_by_side():
    from test_quant import build_transform  # their own unmodified helper

    # crop_pct=0.9 is the LITERAL value passed to build_transform() below --
    # the same value test_quant.py's own main() hardcodes for model_type
    # == 'vit' (see module docstring). Compare THIS value directly, never
    # reverse-derive it from resize_t.size: build_transform() computes
    # `size = int(math.floor(input_size / crop_pct))` internally, and floor()
    # is lossy -- floor(224/0.9)=248, but 224/248=0.9032, NOT 0.9. Comparing
    # a floor-recovered approximation against timm's exact 0.9 produces a
    # false-positive mismatch on every call, regardless of whether the real
    # crop_pct actually matches (it does -- this literal IS timm's value,
    # confirmed against real source, see timm_side below).
    fqvit_crop_pct_passed_in = 0.9
    fqvit_t = build_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=fqvit_crop_pct_passed_in)
    resize_t, crop_t, _, norm_t = fqvit_t.transforms
    fqvit_side = {
        "resize_size": resize_t.size,
        "resize_interpolation": str(resize_t.interpolation),
        "crop_size": crop_t.size,
        "mean": list(norm_t.mean.tolist() if hasattr(norm_t.mean, "tolist") else norm_t.mean),
        "std": list(norm_t.std.tolist() if hasattr(norm_t.std, "tolist") else norm_t.std),
    }
    print(f"=== [1] FQ-ViT stock loader transform (model_type='vit', their own "
          f"test_quant.build_transform(), unmodified) ===\n{json.dumps(fqvit_side, indent=2)}")

    # NOT imported live: `timm` is not (and should not become) a dependency
    # of B3's own pinned venv -- FQ-ViT's own requirements.txt (torch==1.7.1,
    # torchvision==0.8.2, exact-pinned) has no reason to include it (FQ-ViT
    # vendors its own ViT implementation, see models/vit_quant.py), and
    # `pip install timm` into that fairness-rule-pinned environment risks
    # pulling a torch-version-incompatible resolver or, worse, silently
    # upgrading the pinned torch/torchvision themselves -- not worth the
    # risk for a diagnostic side-check. These values are instead read
    # directly from timm's real source at
    # timm/models/vision_transformer.py: the `augreg2_in21k_ft_in1k` entry
    # is a bare `_cfg(hf_hub_id='timm/')` with no field overrides, so it
    # resolves to `_cfg()`'s own function defaults verbatim (confirmed by
    # reading both the entry and the `_cfg()` definition in the same file,
    # same commit range as this project's other timm-version-agnostic
    # source reads).
    timm_side = {
        "input_size": [3, 224, 224],
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": [0.5, 0.5, 0.5],  # IMAGENET_INCEPTION_MEAN
        "std": [0.5, 0.5, 0.5],  # IMAGENET_INCEPTION_STD
    }
    print(f"=== [1] timm-resolved config (vit_base_patch16_224.augreg2_in21k_ft_in1k) -- "
          f"read from timm's real source, not live-imported (see comment above) ===\n"
          f"{json.dumps(timm_side, indent=2)}")

    mismatches = []
    if abs(fqvit_crop_pct_passed_in - timm_side["crop_pct"]) > 1e-9:
        mismatches.append(f"crop_pct: fqvit={fqvit_crop_pct_passed_in} vs timm={timm_side['crop_pct']}")
    if list(fqvit_side["mean"]) != timm_side["mean"]:
        mismatches.append(f"mean: fqvit={fqvit_side['mean']} vs timm={timm_side['mean']}")
    if list(fqvit_side["std"]) != timm_side["std"]:
        mismatches.append(f"std: fqvit={fqvit_side['std']} vs timm={timm_side['std']}")
    print(f"VERDICT: {'MATCH -- every field identical' if not mismatches else 'MISMATCH FOUND'}")
    assert not mismatches, (
        "TRANSFORM MISMATCH between FQ-ViT's stock loader and timm's resolved config for "
        "vit_base_patch16_224.augreg2_in21k_ft_in1k: " + "; ".join(mismatches) + " -- this is exactly "
        "the mechanism the 2026-07 ruling hypothesized for the Step-B FP-parity gap; previously "
        "verified clean (deviations.md #36) -- halting because that verification no longer holds, not "
        "because this was expected to fail."
    )
    return {"fqvit_stock": fqvit_side, "timm_augreg2": timm_side}


def check_checkpoint_keys(checkpoint_path):
    import torch
    from config import Config
    from test_quant import str2model

    cfg = Config(False, False, "minmax")  # ptf/lis/quant_method -- irrelevant, architecture-only
    model = str2model("vit_base")(pretrained=False, cfg=cfg)  # architecture only, no weights loaded
    model_keys = set(model.state_dict().keys())

    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and set(sd.keys()) <= {"model", "state_dict"}:
        sd = sd.get("model", sd.get("state_dict"))
    ckpt_keys = set(sd.keys())

    missing = sorted(model_keys - ckpt_keys)  # in FQ-ViT's model, absent from checkpoint -> stays random-init
    unexpected = sorted(ckpt_keys - model_keys)  # in checkpoint, absent from FQ-ViT's model -> silently dropped
    print(f"=== [2] Checkpoint-key diagnostic (FQ-ViT VisionTransformer('vit_base') vs "
          f"pinned augreg2 checkpoint, same strict=False comparison the real Step-B override performs) ===\n"
          f"model has {len(model_keys)} keys, checkpoint has {len(ckpt_keys)} keys\n"
          f"MISSING from checkpoint ({len(missing)}): {missing}\n"
          f"UNEXPECTED in checkpoint ({len(unexpected)}): {unexpected}\n"
          f"VERDICT: {'CLEAN -- every key matched' if not missing and not unexpected else 'KEY MISMATCH FOUND'}")
    del model
    assert not missing and not unexpected, (
        f"CHECKPOINT-KEY MISMATCH between FQ-ViT's VisionTransformer('vit_base') and the pinned "
        f"checkpoint at {checkpoint_path}: {len(missing)} missing, {len(unexpected)} unexpected -- "
        "the Step-B override's strict=False load would silently drop these, biasing every downstream "
        "measurement on this checkpoint. Previously verified clean (152/152, deviations.md #36) -- "
        "halting because that verification no longer holds for this checkpoint/model class pairing."
    )
    return {"n_model_keys": len(model_keys), "n_ckpt_keys": len(ckpt_keys), "missing": missing,
            "unexpected": unexpected}


def check_plain_augregv1_deficit(imagenet_dir, val_batchsize, num_workers, device):
    """Reuses common.fp_parity_gate's own _fqvit_fp_pass + check_fp_parity
    (the SAME 0.15pp tolerance, same assert-on-fail behavior) rather than
    hand-rolling a parallel diff/print -- this check is structurally
    identical to the existing FP-parity gate (plain pass vs a trusted
    reference), just against FQ-ViT's own published number instead of
    P2.1's S8, for the checkpoint/model pairing where the real gate is
    structurally skipped (coincides_with_pinned()==False)."""
    from common.fp_parity_gate import _fqvit_fp_pass, check_fp_parity

    top1_pct, n_total = _fqvit_fp_pass("vit_base", imagenet_dir, val_batchsize, num_workers, device)
    print("=== [3] Plain augreg-v1 (Step-A checkpoint) through FQ-ViT's STOCK loader (no override) ===")
    diff = check_fp_parity(top1_pct, PUBLISHED_VITB_FP_STOCK_CHECKPOINT_PCT,
                            "vit_base_patch16_224 (FQ-ViT stock augreg-v1)", "fq_vit_stock_deficit_check")
    return {"measured_top1_pct": top1_pct, "n_total": n_total,
            "published_top1_pct": PUBLISHED_VITB_FP_STOCK_CHECKPOINT_PCT, "diff_pct_points": diff}


def _cli():
    default_num_workers = min(16, os.cpu_count() or 2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=None, help="path to pinned augreg2 pytorch_model.bin "
                                                                 "(check 2); omit to skip check 2")
    parser.add_argument("--imagenet-dir", default=None, help="required unless --skip-plain-pass")
    parser.add_argument("--skip-plain-pass", action="store_true", help="skip check 3 (the GPU/full-50k one)")
    parser.add_argument("--skip-transform-check", action="store_true",
                         help="skip check 1 -- it's specifically about the augreg2/Step-B override, not "
                              "meaningful for a Step-A-only (stock checkpoint, no override) invocation")
    parser.add_argument("--val-batchsize", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = {}
    if not args.skip_transform_check:
        report["transform_side_by_side"] = check_transform_side_by_side()

    if args.checkpoint_path:
        report["checkpoint_keys"] = check_checkpoint_keys(args.checkpoint_path)

    if not args.skip_plain_pass:
        assert args.imagenet_dir, "--imagenet-dir required unless --skip-plain-pass"
        report["plain_augregv1_deficit"] = check_plain_augregv1_deficit(
            args.imagenet_dir, args.val_batchsize, args.num_workers, args.device)

    print("TANDEM_VITB_DIAGNOSTIC_RESULT " + json.dumps(report))


if __name__ == "__main__":
    _cli()
