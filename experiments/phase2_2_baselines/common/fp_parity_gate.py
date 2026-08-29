"""FP-parity gate (spec item #2): before any method runs on a COMPARISON
checkpoint (TANDEM's pinned tag -- see common/checkpoint_pin.py), evaluate
the UNMODIFIED (unquantized, unmerged) model through that baseline's OWN
harness -- its own model construction, its own preprocessing, its own
val-set loader -- on the full 50k set, and assert the result is within
0.15pp of P2.1's own S8 measurement for that exact checkpoint (72.166 /
79.848 / 81.97 / 85.10 for deit_tiny/small/base/vit_base respectively, see
common.checkpoint_pin.PINNED_TAGS). A miss means the baseline's eval loop
or transform differs from TANDEM's own in some way that would silently bias
every downstream method number -- halt before trusting any of them.

This is a CHEAP check relative to a full quantization sweep (one plain
forward pass over 50k images, no calibration, no fake-quant), but it is
still a full-50k pass -- not a smoke subset -- because the whole point is
to catch a harness-level discrepancy that a 256-image smoke check (see
common/equivalence_gate.py, which checks CROSS-HOST reproducibility of a
baseline's OWN numbers, a different question) would not reliably surface.

B1 (ToMe)/B4 (composites) run in-process and already have a real model
object in hand -- they call `fp_parity_check_inprocess()` directly, no
subprocess needed. B2 (RepQ-ViT)/B3 (FQ-ViT) run everything as a subprocess
of an UNMODIFIED upstream script and have no in-process model object to
reuse -- this module's `_cli()` entry point (`python -m
common.fp_parity_gate ...`) is a SEPARATE, cheap side-pass (same pattern as
common/calib_capture.py: replicates the target repo's own model-construction
+ data-loading call sequence, verified against real source, never modifies
their file on disk) that builds the plain model, evaluates it, and prints
the result for the parent run.py to parse.
"""
import json
import time

FP_PARITY_TOLERANCE_PCT = 0.15


def check_fp_parity(measured_top1_pct, reference_top1_pct, model_name, harness_desc):
    diff = abs(measured_top1_pct - reference_top1_pct)
    passed = diff <= FP_PARITY_TOLERANCE_PCT
    line = (f"[FP-parity / {model_name} / {harness_desc}] harness={measured_top1_pct:.4f}%  "
            f"P2.1_S8={reference_top1_pct:.4f}%  diff={diff:.4f}pp (tol={FP_PARITY_TOLERANCE_PCT}pp) "
            f"-> {'PASS' if passed else 'FAIL'}")
    print(line)
    assert passed, (
        f"FP-PARITY GATE FAILED for {model_name} ({harness_desc}): harness measured "
        f"{measured_top1_pct:.4f}%, P2.1's own S8 measured {reference_top1_pct:.4f}% on the SAME "
        f"checkpoint -- diff={diff:.4f}pp exceeds {FP_PARITY_TOLERANCE_PCT}pp. This means the "
        "baseline harness's eval loop or transform differs from TANDEM's own in some way that would "
        "silently bias every downstream method number on this model. Halting -- do not proceed to "
        "any method-specific run until this is diagnosed."
    )
    return diff


def fp_parity_check_inprocess(model, val_loader, device, model_name, reference_top1_pct, harness_desc="in-process"):
    """B1/B4: `model` must be the PLAIN, unmodified model (no ToMe patch,
    no quantization) -- callers are responsible for that; this function
    only evaluates and gates, it does not know how to strip a patch back
    off. Uses the plain (non-resumable) eval loop -- deliberately simpler
    than the real method run's checkpointed pass, since this is a cheap
    pre-flight check, not the expensive part.

    Returns (measured_top1_pct, diff) -- the raw measured number, not just
    the diff, so a caller with an ADDITIONAL use for the same single
    forward pass (e.g. b1_tome/run.py's r0_baseline row + its own second
    gate against ToMe's stated off-the-shelf baseline) never needs a
    second 50k pass just to get the number back. B4 (the only other
    caller) discards the return value entirely, so this is not a breaking
    change for it."""
    from common.eval_utils import evaluate_top1
    top1, n, _ = evaluate_top1(model, val_loader, device=device, desc=f"FP-parity {model_name}")
    assert n == 50000, f"FP-parity gate expected the full 50k val set, got n={n}"
    diff = check_fp_parity(top1 * 100, reference_top1_pct, model_name, harness_desc)
    return top1 * 100, diff


def _repq_fp_pass(model_arg, imagenet_dir, val_batchsize, num_workers, device):
    """Replicates RepQ-ViT's own test_quant.py main() UP TO (never
    including) `quant_model(...)` -- i.e. `build_model()` + their own
    `build_dataset()`, evaluated plain. Verified against their real
    source: `model = build_model(model_zoo[args.model]); model.eval()`
    happens BEFORE any `quant_model`/`set_quant_state` call, so this is a
    faithful prefix of their real call sequence, not a reimplementation."""
    import torch
    from types import SimpleNamespace
    from utils.build_model import build_model
    from utils.build_dataset import build_dataset

    # RepQ-ViT's own test_quant.py resolves --model through this exact
    # dict INSIDE their main() (a local variable, not an importable
    # module-level name) before ever calling build_model() -- build_model()
    # itself does NOT do this translation, it expects the resolved timm
    # name directly. Verified against their real source at the pinned
    # commit; duplicated here (not imported) because it's function-local.
    model_zoo = {
        "vit_small": "vit_small_patch16_224", "vit_base": "vit_base_patch16_224",
        "deit_tiny": "deit_tiny_patch16_224", "deit_small": "deit_small_patch16_224",
        "deit_base": "deit_base_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224", "swin_small": "swin_small_patch4_window7_224",
    }
    model = build_model(model_zoo[model_arg]).to(device)
    model.eval()

    args = SimpleNamespace(model=model_arg, dataset=imagenet_dir, calib_batchsize=32,
                            val_batchsize=val_batchsize, num_workers=num_workers)
    _, val_loader = build_dataset(args)
    return _plain_eval_loop(model, val_loader, device)


def _fqvit_fp_pass(model_arg, imagenet_dir, val_batchsize, num_workers, device):
    """Replicates FQ-ViT's own test_quant.py main() UP TO (never
    including) the `if args.quant:` calibration/model_quant() block.
    Verified against their real source: `model =
    str2model(args.model)(pretrained=True, cfg=cfg)` defaults to
    `quant=False` (the function signature's own default) and is NEVER
    quantized unless `model_open_calibrate()`/`model_quant()` are
    explicitly called later -- skipping straight to `validate()` after
    construction is a faithful prefix of their real call sequence."""
    import os
    import torchvision.datasets as datasets
    import torchvision.transforms as tv_transforms
    import torch
    from torch.utils.data import DataLoader
    from config import Config
    from test_quant import str2model, build_transform  # noqa -- their own module-level helper, unmodified

    cfg = Config(False, False, "minmax")  # ptf/lis/quant_method -- irrelevant, never quantizing
    model = str2model(model_arg)(pretrained=True, cfg=cfg).to(device)
    model.eval()

    model_type = model_arg.split("_")[0]
    if model_type == "deit":
        mean, std, crop_pct = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.875
    elif model_type == "vit":
        mean, std, crop_pct = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 0.9
    else:
        raise NotImplementedError(f"FP-parity gate: unhandled model_type {model_type!r}")
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_dataset = datasets.ImageFolder(os.path.join(imagenet_dir, "val"), val_transform)
    val_loader = DataLoader(val_dataset, batch_size=val_batchsize, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return _plain_eval_loop(model, val_loader, device)


EARLY_ABORT_CHECK_BATCHES = 3
EARLY_ABORT_MAX_TOP1 = 0.03  # 3% -- generous headroom above 1000-class random chance (~0.1%)


def _plain_eval_loop(model, val_loader, device):
    """Early-abort sanity check, first: accuracy over the first
    `EARLY_ABORT_CHECK_BATCHES` batches is checked against
    `EARLY_ABORT_MAX_TOP1` before continuing to the full pass. This is
    NOT the real gate (that's `check_fp_parity`'s tight 0.15pp tolerance
    against P2.1's own S8 reference, after the FULL 50k pass) -- it exists
    only to fail in seconds rather than minutes on a CATASTROPHICALLY
    broken harness (wrong labels, wrong model, corrupt data). This exact
    failure mode -- a scrambled class-directory-to-label mapping in
    common/colab_imagenet.py -- cost a real ~6.5-minute full pass before
    this check existed; any legitimate model+data pipeline scores far
    above this threshold within the first few hundred images."""
    import torch
    n_correct = n_total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            n_correct += (preds == target).sum().item()
            n_total += target.shape[0]
            if batch_idx + 1 == EARLY_ABORT_CHECK_BATCHES:
                early_top1 = n_correct / n_total
                if early_top1 < EARLY_ABORT_MAX_TOP1:
                    raise RuntimeError(
                        f"Early sanity abort: top1={early_top1 * 100:.2f}% over the first {n_total} "
                        "images is consistent with near-random guessing (1000-class chance level "
                        "~0.1%), not a working model+data pipeline. Aborting before the full 50k "
                        "pass rather than spending minutes confirming what's already clear -- check "
                        "the ImageFolder's class-directory naming/label mapping (see "
                        "common/colab_imagenet.py's module docstring for exactly this failure mode: "
                        "class dirs must be zero-padded integer labels, never a human-readable "
                        "ClassLabel.int2str() name) before re-running."
                    )
    elapsed_s = time.perf_counter() - t0
    top1_pct = 100.0 * n_correct / n_total
    print(f"FP-parity plain pass: top1={top1_pct:.4f}%  n={n_total}  ({elapsed_s / 60:.1f} min)")
    return top1_pct, n_total


def _cli():
    import argparse
    import os
    # RepQ-ViT's/FQ-ViT's own --num-workers default is 16 -- fine on the
    # university GPU box, but a real OOM-kill (SIGKILL, exit -9) risk on a
    # 2-CPU Colab runtime once combined with a larger model's memory
    # footprint (DataLoader worker PROCESSES, not just threads, each carry
    # real overhead). This is a throughput/memory knob, not a scientific
    # setting -- shuffle=False, same images, same order, same everything
    # that actually affects the measured number -- so capping it to the
    # host's real core count is not a fairness-rule violation, only a
    # hardware-adaptive safety margin. Still overridable via --num-workers.
    default_num_workers = min(16, os.cpu_count() or 2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["repq_vit", "fq_vit"])
    parser.add_argument("--model-arg", required=True, help="the METHOD's own --model string")
    parser.add_argument("--imagenet-dir", required=True)
    parser.add_argument("--val-batchsize", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fn = {"repq_vit": _repq_fp_pass, "fq_vit": _fqvit_fp_pass}[args.method]
    top1_pct, n_total = fn(args.model_arg, args.imagenet_dir, args.val_batchsize, args.num_workers, args.device)
    # Machine-readable line for the parent run.py to parse (last line of
    # stdout, same convention as common.determinism's host-info CLI).
    print("TANDEM_FP_PARITY_RESULT " + json.dumps({"top1_pct": top1_pct, "n_total": n_total}))


if __name__ == "__main__":
    _cli()
