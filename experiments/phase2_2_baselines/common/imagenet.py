"""Local ImageNet-1k val loader for OUR OWN eval loop (B1, B4; B2/B3 use
each upstream repo's own data loader instead, invoked as a subprocess --
see their run.py). torch-dependent (unlike bops.py/results_io.py/
repro_gate.py) -- only imported inside b1_tome/ and
b4_sequential_composites/'s pinned old-timm venvs (deviations.md #1).

Preprocessing is DERIVED from the loaded model's own resolved timm data
config (timm.data.resolve_data_config + create_transform), exactly like
experiments/phase2_1_fleet/P2_tandem_pipeline.py's Section 5a -- NOT a hardcoded
resize-256/bilinear/224-crop recipe. Bug found via a real Colab run
(2026-07): that hardcoded recipe assumes bilinear interpolation and
crop_pct~0.875 (resize 256 -> crop 224), but `deit_small_patch16_224.fb_in1k`
actually resolves to bicubic interpolation and crop_pct=0.9 (resize ~249,
confirmed directly from P2's own real run on this exact checkpoint tag) --
the same class of defect the ViT-B preprocessing incident (P2.1) already
found and fixed for P2 itself, just not yet ported here. Required for
top1_50k to be comparable to TANDEM's own P2 measurements on the same
cost axis, not just internally consistent within this directory.
"""
import os

import timm.data
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# A DataLoader worker COUNT (real OS processes, real overhead each), not a
# scientific setting -- shuffle=False, same images/order either way.
# Capping to the real host core count is a hardware-adaptive safety
# margin against an OOM-kill (SIGKILL) on a low-core-count host (e.g.
# Colab), same reasoning as B2/B3's NUM_WORKERS -- see their run.py.
DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 2)


def build_val_loader(imagenet_dir, model, batch_size=128, num_workers=DEFAULT_NUM_WORKERS):
    """`model`: the loaded timm model whose OWN resolved data config
    (input_size/interpolation/crop_pct/mean/std) determines preprocessing
    -- does not need real pretrained weights, only `pretrained_cfg`
    metadata (registered at the architecture level), so callers that
    don't otherwise need a model instance yet can pass a cheap
    `timm.create_model(MODEL_NAME, pretrained=False)` just for this."""
    val_dir = f"{imagenet_dir}/val"
    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    print(f"[imagenet] resolved transform for {val_dir}: interpolation={data_cfg.get('interpolation')}, "
          f"crop_pct={data_cfg.get('crop_pct')}, input_size={data_cfg.get('input_size')}, "
          f"mean={data_cfg.get('mean')}, std={data_cfg.get('std')}")
    dataset = ImageFolder(val_dir, transform=transform)
    if os.environ.get("TANDEM_SMOKE_RUN") == "1":
        # common/equivalence_gate.py's build_smoke_subset deliberately
        # builds a SMALL (N_VAL_SMOKE=256, not 50,000), class-skewed (first
        # ~5-6 of 1000 classes in deterministic sorted order, not all 1000)
        # val subset for the cross-host smoke check -- the strict counts
        # below exist to catch an incomplete/corrupted REAL copy, which
        # this deliberately is not. Skipped here, matching every other
        # smoke-mode relaxation in this project (FP-parity/reproduction
        # gates) -- never applies to a real (non-smoke) run.
        print(f"TANDEM_SMOKE_RUN=1 -- skipping the 50,000-image/1000-class strict count checks "
              f"(found {len(dataset)} images, {len(dataset.classes)} classes at {val_dir}).")
    else:
        assert len(dataset) == 50000, (
            f"expected the standard ImageNet-1k val split (50,000 images) at {val_dir}, found "
            f"{len(dataset)} -- check IMAGENET_DIR points at a complete, correctly-laid-out copy "
            "(<root>/val/<wnid>/*.JPEG)."
        )
        # ImageFolder assigns class index = position in the alphabetically-sorted
        # list of subdirectory names -- correct ONLY if there are exactly the
        # standard 1000 classes present (a partial/mis-structured local copy
        # would silently produce a different, wrong index assignment without
        # this check; this is the local-path analogue of the class-directory
        # sanity check in common/colab_imagenet.py, which exists because
        # exactly this class of bug -- a scrambled class-to-index mapping --
        # was found there and silently produced near-random accuracy).
        assert len(dataset.classes) == 1000, (
            f"expected exactly 1000 class directories under {val_dir}, found {len(dataset.classes)} -- "
            "a wrong count means ImageFolder's alphabetically-derived class indices will NOT match the "
            "standard ImageNet-1k convention every pretrained model expects, silently corrupting every "
            "top-1 number. Check the local copy is complete and correctly laid out."
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def get_sample_batch(imagenet_dir, model, batch_size=128):
    """A single batch, for dry-run timing or token-trajectory measurement
    (common.bops.measure_token_trajectory) -- not a full pass."""
    loader = build_val_loader(imagenet_dir, model, batch_size=batch_size, num_workers=0)
    images, _ = next(iter(loader))
    return images
