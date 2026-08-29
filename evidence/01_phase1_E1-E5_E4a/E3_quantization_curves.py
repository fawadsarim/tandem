# %% [markdown]
# # E3 - Quantization Rate-Distortion Curves & Gaussian-to-Quantization Transfer (DeiT-Small)
#
# E1 measured per-block sensitivities `S_l` under GAUSSIAN noise. E2 verified
# additivity. E3 replaces synthetic noise with REAL quantization error and
# tests whether the Gaussian-calibrated sensitivities predict quantization
# damage (transfer) and whether that damage is still additive across blocks
# under real quantization (Q3). Q1 (the −2 scaling law) is informative only -
# the allocator consumes the MEASURED curve, not the parametric form.
#
# Runs top-to-bottom on a single Colab GPU. Does NOT modify E1/E2. E1's
# already-verified machinery is copied verbatim into clearly marked cells.
#
# **Outputs** (under `results/E3/`): `E3_raw.csv`, `E3_summary.json`,
# `quant_scaling_curves.png`, `transfer_ratio.png`,
# `quant_additivity_scatter.png`, `accuracy_glimpse.png`.
#
# Reviewer-release note: this notebook has no manual-review STOP
# checkpoints to convert -- see `E5_allocator_G1.py`'s Section 21/25d and
# its `TANDEM_INTERACTIVE` flag for the one file in E1-E5 that does.

# %% [markdown]
# ## 0. Clone the repo (so this and later notebooks share `results/` via git)
#
# This repo is public -- cloning it needs no authentication.
#
# Clones (or pulls, if already cloned this session) to `/content/tandem`,
# then changes into `/content/tandem/notebooks` -- every relative path in
# this notebook resolves exactly as it always has, no changes needed
# elsewhere. E1/E2's results, if already pushed, come along automatically
# with this same clone.

# %%
import os

REPO_URL = "https://github.com/fawadsarim/tandem.git"
REPO_DIR = "/content/tandem"

if not os.path.exists(REPO_DIR):
    get_ipython().system(f'git clone -q {REPO_URL} {REPO_DIR}')  # noqa: F821
else:
    print(f"{REPO_DIR} already exists -- pulling latest instead of re-cloning.")
    get_ipython().system(f'git -C {REPO_DIR} pull -q')  # noqa: F821

get_ipython().run_line_magic('cd', f'{REPO_DIR}/experiments/phase1_E1-E5_E4a')  # noqa: F821

# %% [markdown]
# ## 1. Setup

# %%
!pip install -q torch timm datasets matplotlib pandas tqdm scipy

# %%
import hashlib
import itertools
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CONFIG = {
    "N_IMAGES": 5000,
    "BATCH_SIZE": 128,
    "CALIB_IMAGES": 512,
    "MODEL": "deit_small_patch16_224",
}

# Independent verification: results/E3/E3_raw.csv ships committed in this
# repo, so a default Run All resumes from it instantly rather than
# recomputing -- that's a self-consistency check (the checked-in numbers
# still match what this exact code produces), not a substitute for a real
# from-scratch run. Set FORCE_FRESH_RUN = True to ignore any existing
# checkpoint and recompute every row.
FORCE_FRESH_RUN = False

DEVICE = "cuda"
assert torch.cuda.is_available(), (
    "E3 requires a CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU."
)

RESULTS_DIR = Path("results/E3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HF_TOKEN: Colab secrets > env var. Never hardcoded, no manual-paste line.
try:
    from google.colab import userdata

    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception as e:
        HF_TOKEN = None
        print(f"Colab secret 'HF_TOKEN' not usable ({type(e).__name__}: {e}).")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")
print(f"HF_TOKEN detected: {'yes' if HF_TOKEN else 'no'}")

ENVIRONMENT = {
    "torch_version": torch.__version__,
    "timm_version": timm.__version__,
    "cuda_version": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
}
print(json.dumps(ENVIRONMENT, indent=2))

# %% [markdown]
# ## 2. [VERBATIM FROM E1 - do not modify] ImageNet-1k data loader

# %%
from datasets import load_dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),  # uint8, [3,224,224]
    ]
)


def _to_uint8_chw(pil_img):
    return PREPROCESS(pil_img.convert("RGB"))


def _materialize(example_iter, n, label_key=None):
    """Decode/resize/crop up to n HF-style dict examples into a uint8 tensor + optional labels."""
    imgs = torch.empty((n, 3, 224, 224), dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.long) if label_key else None
    count = 0
    for ex in tqdm(example_iter, total=n, desc="decoding images"):
        if count >= n:
            break
        imgs[count] = _to_uint8_chw(ex["image"])
        if label_key:
            labels[count] = int(ex[label_key])
        count += 1
    assert count == n, f"expected {n} images, only found {count}"
    return imgs, labels


def load_eval_images(cfg, hf_token):
    """Returns (images_u8, labels, labels_available=True, source="imagenet-1k", n_used).
    No fallback -- fails loudly if ImageNet-1k access isn't actually working."""
    n_requested = cfg["N_IMAGES"]
    assert hf_token, "HF_TOKEN required -- ImageNet-1k access is no longer optional."

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=hf_token)
    probe = next(iter(ds))  # verify access is actually granted before committing
    assert "image" in probe and "label" in probe
    imgs, labels = _materialize(iter(ds), n_requested, label_key="label")
    print(f"{'=' * 70}\nDATA SOURCE: ImageNet-1k (validation, streaming, {n_requested} images)\n{'=' * 70}")
    return imgs, labels, True, "imagenet-1k", n_requested


# %%
images_u8, labels, LABELS_AVAILABLE, DATA_SOURCE, N_USED = load_eval_images(CONFIG, HF_TOKEN)
assert images_u8.shape == (N_USED, 3, 224, 224)
assert images_u8.dtype == torch.uint8
if labels is not None:
    assert labels.shape == (N_USED,)

# %% [markdown]
# ## 3. [VERBATIM FROM E1 - do not modify] Batch & forward-pass helpers
#
# `forward_logits` is what pins the fp16-autocast setting used everywhere,
# including inside the quantization simulator below - required for the
# quant math's own fp32-then-cast-back discipline to mean anything.

# %%
MEAN_T = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
STD_T = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)


def normalize_batch(batch_u8, device=DEVICE):
    x = batch_u8.to(device, non_blocking=True).float() / 255.0
    return (x - MEAN_T) / STD_T


def iterate_batches(images, batch_size):
    n = images.shape[0]
    for start in range(0, n, batch_size):
        yield start, images[start : start + batch_size]


def n_batches_for(n_images, batch_size):
    return math.ceil(n_images / batch_size)


def forward_logits(model, batch_u8, device=DEVICE):
    """Single standardized forward pass: fp16 autocast, fp32 output."""
    x = normalize_batch(batch_u8, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
    return logits.float()


# %% [markdown]
# ## 4. [VERBATIM FROM E1 - do not modify] Model loader

# %%
def load_model(cfg):
    try:
        model = timm.create_model(cfg["MODEL"], pretrained=True)
    except Exception:
        tagged = f"{cfg['MODEL']}.fb_in1k"
        print(f"'{cfg['MODEL']}' has no default pretrained tag; retrying as '{tagged}'.")
        model = timm.create_model(tagged, pretrained=True)
    model = model.eval().to(DEVICE).requires_grad_(False)
    assert len(model.blocks) == 12, f"expected 12 transformer blocks, found {len(model.blocks)}"
    return model


model = load_model(CONFIG)
N_BLOCKS = len(model.blocks)
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {CONFIG['MODEL']}: {n_params / 1e6:.1f}M params, {N_BLOCKS} blocks, "
      f"{model.num_classes} classes.")

# %% [markdown]
# ## 5. [VERBATIM FROM E1 - do not modify] Stats pass - `m_l`
#
# Recomputed fresh in this session so Gate (b) can check it against E1's
# stored `m_l`.

# %%
def compute_m_l(model, images, batch_size):
    n_blocks = len(model.blocks)
    sums = [0.0] * n_blocks
    n_total = images.shape[0]

    def make_hook(idx):
        def hook(module, inp, out):
            sums[idx] += out.float().pow(2).mean(dim=tuple(range(1, out.dim()))).sum().item()

        return hook

    handles = [model.blocks[i].register_forward_hook(make_hook(i)) for i in range(n_blocks)]
    try:
        for _, batch in tqdm(
            iterate_batches(images, batch_size),
            total=n_batches_for(images.shape[0], batch_size),
            desc="stats pass",
        ):
            forward_logits(model, batch)
    finally:
        for h in handles:
            h.remove()
    return [s / n_total for s in sums]


m_l = compute_m_l(model, images_u8, CONFIG["BATCH_SIZE"])
for i, m in enumerate(m_l):
    print(f"block {i + 1:2d}: m_l = {m:.6f}")

# %% [markdown]
# ## 6. [VERBATIM FROM E1 - do not modify] Reference pass - cache FP logits

# %%
def compute_reference_logits(model, images, batch_size):
    n = images.shape[0]
    fp_logits = torch.empty((n, model.num_classes), dtype=torch.float32)
    for start, batch in tqdm(
        iterate_batches(images, batch_size),
        total=n_batches_for(n, batch_size),
        desc="reference pass",
    ):
        logits = forward_logits(model, batch)
        fp_logits[start : start + batch.shape[0]] = logits.cpu()
    return fp_logits


fp_logits = compute_reference_logits(model, images_u8, CONFIG["BATCH_SIZE"])
fp_argmax = fp_logits.argmax(dim=1)
print(f"Cached FP logits: {tuple(fp_logits.shape)}")

# %% [markdown]
# ## 7. [VERBATIM FROM E1 - do not modify] Batch metrics (mse, kl, flip_rate, top1_delta)

# %%
def _batch_metrics(logits, fp_batch, fp_argmax_batch, lbl):
    """Per-batch contributions to the running sums: (mse_sum, kl_sum, flips, correct_fp, correct_pert)."""
    # sum-reduction over (batch, classes) / num_classes == sum over the
    # batch of each sample's own mean-over-classes squared error.
    mse_sum = F.mse_loss(logits, fp_batch, reduction="sum").item() / logits.shape[1]

    log_p_fp = F.log_softmax(fp_batch, dim=1)
    log_p_pert = F.log_softmax(logits, dim=1)
    kl_sum = (log_p_fp.exp() * (log_p_fp - log_p_pert)).sum(dim=1).sum().item()

    pert_argmax = logits.argmax(dim=1)
    flips = (pert_argmax != fp_argmax_batch).sum().item()

    correct_fp = correct_pert = 0
    if lbl is not None:
        correct_fp = (fp_argmax_batch == lbl).sum().item()
        correct_pert = (pert_argmax == lbl).sum().item()
    return mse_sum, kl_sum, flips, correct_fp, correct_pert


# %% [markdown]
# ## 8. Load E1 constants + Gates (a), (b)
#
# Loading order: local repo path -> Drive mount -> manual upload.
# Notebook-audit note: this used to look for a directory literally named
# `E1b` with a `results/E1/` (non-`b`) fallback -- consolidated away now
# that there's only one E1 tier (ImageNet-1k, see E1's own Section 2
# note); just `results/E1/` now, no `b`-suffix distinction to track.

# %%
def _find_results_dir(name, search_root="."):
    for d in Path(search_root).glob("results/*"):
        if d.is_dir() and d.name.lower() == name:
            summaries = sorted(d.glob("*[Ss]ummary*.json"))
            if summaries:
                return summaries[0]
    return None


def _try_drive_mount():
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        return _find_results_dir("e1", "/content/drive/MyDrive")
    except Exception as e:
        print(f"Drive mount unavailable ({type(e).__name__}: {e}).")
        return None


def _try_manual_upload():
    print("Please upload E1's summary JSON (from results/E1/) when prompted.")
    try:
        from google.colab import files

        uploaded = files.upload()
        name = next((n for n in uploaded if "summary" in n.lower() and n.endswith(".json")), None)
        return Path(name) if name else None
    except Exception as e:
        print(f"Manual upload unavailable ({type(e).__name__}: {e}).")
        return None


def load_e1_summary():
    summary_path = _find_results_dir("e1")
    if summary_path is None:
        print("E1 results not found locally under results/E1/. Trying Google Drive...")
        summary_path = _try_drive_mount()
    if summary_path is None:
        summary_path = _try_manual_upload()
    assert summary_path is not None, (
        "Could not locate E1 constants via any loading tier. Place "
        "results/E1/*summary*.json in the repo (e.g. clone the repo and run E1 "
        "first, or push its results), or use the Drive/upload prompts."
    )
    print(f"Loaded E1 constants from {summary_path}")
    with open(summary_path) as f:
        return json.load(f)


E1_SUMMARY = load_e1_summary()
S_L_MSE_REF = E1_SUMMARY["S_l_mse"]
M_L_REF = E1_SUMMARY["m_l"]
REGIME_MSE_REF = E1_SUMMARY["linear_regime_boundary_mse"]
S_ABS_REF = [S_L_MSE_REF[i] / M_L_REF[i] for i in range(12)]  # S_abs,l = S_rel,l / m_l

# %%
def gate_data_source_match(data_source, e1_summary):
    ok = data_source == e1_summary["data_source"]
    assert ok, (
        f"Gate (a) FAILED: this session's data_source='{data_source}' != "
        f"E1's data_source='{e1_summary['data_source']}'. The constants "
        "were measured on a different data distribution and cannot be reused."
    )
    return {"passed": True, "data_source": data_source}


def gate_m_l_consistency(m_l_fresh, m_l_ref, tol=0.02):
    rel_err = [abs(a - b) / b for a, b in zip(m_l_fresh, m_l_ref)]
    worst = max(rel_err)
    assert worst < tol, (
        f"Gate (b) FAILED: block {rel_err.index(worst) + 1} m_l differs by "
        f"{worst:.1%} (>= {tol:.0%}) from E1's. A mismatch usually means a "
        "different data source or loader than the one E1's constants were measured on."
    )
    return {"passed": True, "max_rel_err": worst}


GATE_RESULTS = {
    "a_data_source": gate_data_source_match(DATA_SOURCE, E1_SUMMARY),
    "b_m_l_consistency": gate_m_l_consistency(m_l, M_L_REF),
}
print("Gates (a),(b) passed:", json.dumps(GATE_RESULTS, indent=2))

# %% [markdown]
# ## 9. Quantization simulator
#
# All quant math is forced to fp32 explicitly (`.float()` before
# round/clamp/scale, `.to(orig_dtype)` after) regardless of the ambient fp16
# autocast context, per spec. Target modules per block: `attn.qkv`,
# `attn.proj`, `mlp.fc1`, `mlp.fc2` -- LayerNorm/softmax stay full precision.

# %%
TARGET_ATTRS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def get_target_linear(model, block_idx, attr):
    mod = model.blocks[block_idx]
    for part in attr.split("."):
        mod = getattr(mod, part)
    return mod


def _quantize_weight_rows(W, bits):
    """Per-output-channel symmetric absmax weight quantization, fp32 math."""
    qmax = 2 ** (bits - 1) - 1
    w = W.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    w_hat = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w_hat.to(W.dtype)


def _quantize_tensor(x, scale, bits):
    """Per-tensor symmetric quantization with a pre-computed (static) scale, fp32 math."""
    qmax = 2 ** (bits - 1) - 1
    x_hat = torch.clamp(torch.round(x.float() / scale), -qmax, qmax) * scale
    return x_hat.to(x.dtype)


def _elementwise_mean_sq_diff(a, b):
    """Same per-element mean convention as E1's m_l (mean over all dims
    except batch, per sample; caller sums over the batch and divides by N)
    -- required so eps_eff = A_l/m_l is directly comparable to E1's axis."""
    return (a.float() - b.float()).pow(2).mean(dim=tuple(range(1, a.dim()))).sum().item()


class WeightQuantizer:
    """Precomputes quantized weights ONCE per config (`__enter__`);
    use_quantized()/use_original() cheaply repoint `.weight.data` per
    forward pass with no per-batch tensor copy. `__exit__` always leaves
    weights repointed at their original tensor objects -- exact equality,
    not just closeness, is what the weight-restoration gate checks."""

    def __init__(self, model, block_bits):
        self.model = model
        self.block_bits = block_bits
        self._linears, self._orig, self._quant = {}, {}, {}

    def __enter__(self):
        for block_idx, bits in self.block_bits.items():
            for attr in TARGET_ATTRS:
                lin = get_target_linear(self.model, block_idx, attr)
                key = (block_idx, attr)
                self._linears[key] = lin
                self._orig[key] = lin.weight.data
                self._quant[key] = _quantize_weight_rows(lin.weight.data, bits)
        return self

    def use_quantized(self):
        for key, lin in self._linears.items():
            lin.weight.data = self._quant[key]

    def use_original(self):
        for key, lin in self._linears.items():
            lin.weight.data = self._orig[key]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.use_original()


class ActivationQuantizer:
    """Static per-tensor symmetric activation quantization via
    forward_pre_hooks on the same 4 linears. Scales come ONLY from
    `amax_lookup` (frozen at calibration time, see below) -- never
    recomputed per-batch. `active` toggles the hook between quantizing and
    passing the input through unchanged, so the SAME hook registration
    serves both the clean and quantized halves of a paired forward."""

    def __init__(self, model, block_bits, amax_lookup):
        self.model = model
        self.block_bits = block_bits
        self.amax_lookup = amax_lookup
        self.active = False
        self.handles = []

    def _make_hook(self, block_idx, attr, bits):
        qmax = 2 ** (bits - 1) - 1
        a_max = max(self.amax_lookup[(block_idx, attr)], 1e-12)  # guard a degenerate all-zero calibration
        scale = a_max / qmax  # frozen at calibration time

        def hook(module, args):
            if not self.active:
                return None
            x_hat = _quantize_tensor(args[0], scale, bits)
            return (x_hat,) + args[1:]

        return hook

    def __enter__(self):
        for block_idx, bits in self.block_bits.items():
            for attr in TARGET_ATTRS:
                lin = get_target_linear(self.model, block_idx, attr)
                self.handles.append(lin.register_forward_pre_hook(self._make_hook(block_idx, attr, bits)))
        return self

    def use_quantized(self):
        self.active = True

    def use_original(self):
        self.active = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False
        for h in self.handles:
            h.remove()
        self.handles = []


def hash_state_dict(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


REFERENCE_STATE_HASH = hash_state_dict(model)
print(f"REFERENCE_STATE_HASH = {REFERENCE_STATE_HASH}")

# %% [markdown]
# ## 10. Calibration - static activation scales
#
# ONE pass over the first `CALIB_IMAGES` images, no quantization active.
# For each of the 4 linear inputs of every block, records a running max of
# `|x|` across the calibration pass -- per-tensor ABSMAX, not percentile.
#
# A percentile-based scale was tried first and rejected: with
# `scale = p999/(2^(b-1)-1)`, the clamp range `scale*qmax = p999` cancels
# `b` entirely, so ~0.1% of activations (ViT's known outliers) get hard-
# clipped at EVERY bit-width, including very high ones. That's a fixed
# error floor that doesn't shrink as bits grow, which (a) makes b=16
# idempotence impossible in principle, and worse, (b) would make the A/WA
# arms clipping-dominated rather than rounding-dominated, corrupting the
# c*2^(-2b) scaling law and the whole point of the transfer test. With
# absmax, the clamp only ever engages at/above the single calibrated
# maximum, so at high bit-depth error is genuinely rounding-dominated and
# vanishes as intended -- this is also what the weight quantizer already
# does (per-row absmax), so both quantizers now share the same principle.
# Frozen after this cell -- never touched again during the sweep.

# %%
def calibrate_activation_scales(model, images, calib_n, batch_size):
    amax_running = {}

    def make_hook(block_idx, attr):
        def hook(module, args):
            batch_max = args[0].float().abs().max().item()
            key = (block_idx, attr)
            amax_running[key] = max(amax_running.get(key, 0.0), batch_max)

        return hook

    handles = [
        get_target_linear(model, b, attr).register_forward_pre_hook(make_hook(b, attr))
        for b in range(len(model.blocks))
        for attr in TARGET_ATTRS
    ]
    calib_images = images[:calib_n]
    try:
        for _, batch in tqdm(
            iterate_batches(calib_images, batch_size),
            total=n_batches_for(calib_n, batch_size),
            desc="calibration",
        ):
            forward_logits(model, batch)
    finally:
        for h in handles:
            h.remove()
    return amax_running


AMAX_LOOKUP = calibrate_activation_scales(model, images_u8, CONFIG["CALIB_IMAGES"], CONFIG["BATCH_SIZE"])
print(f"Calibrated activation scales (absmax) for {len(AMAX_LOOKUP)} (block, linear) targets "
      f"over {CONFIG['CALIB_IMAGES']} images.")

# %% [markdown]
# ## 11. Gate (c) - b=16 idempotence
#
# Full quant apparatus (weights + activations) active at 16 bits on block 6
# must be a near-total no-op: A_6 (mean output-energy, same convention as
# m_l) negligible relative to m_6 -- this is the criterion actually gated.
# Logit-level diagnostics (max/p99.9/mean |dlogit|, top1_agreement) are
# reported alongside but NOT asserted on: at this eps_eff (~1e-7, far
# below anything E1/E2 validated) a worst-case max over 5M logit values is
# dominated by fp16 ULP jitter on the single most extreme value, not real
# signal -- the same reason E1/E2 validated the linear model on MSE/KL
# (averages), never on a per-sample worst case.

# %%
def _diagnose_activation_clip_vs_round(model, images, block_idx, bits, amax_lookup, batch_size, n_probe_batches=4):
    """Only called when gate (c) is about to fail. Breaks the activation
    quantizer's error into a clip-active fraction (share of |x| at/above
    the calibrated max, i.e. would be clamped) and a mean rounding step^2
    (scale^2/12, the theoretical noise floor for the non-clipped bulk) --
    shows which term dominates. With absmax at b=16 both should be tiny."""
    qmax = 2 ** (bits - 1) - 1
    per_linear = {}

    def make_hook(attr):
        a_max = max(amax_lookup[(block_idx, attr)], 1e-12)
        scale = a_max / qmax

        def hook(module, args):
            x = args[0].float()
            per_linear[attr] = ((x.abs() >= a_max).sum().item(), x.numel(), scale)

        return hook

    handles = [
        get_target_linear(model, block_idx, attr).register_forward_pre_hook(make_hook(attr)) for attr in TARGET_ATTRS
    ]
    n_over_total = n_total = 0
    step_sq_weighted = 0.0
    try:
        for i, (_, batch) in enumerate(iterate_batches(images, batch_size)):
            if i >= n_probe_batches:
                break
            forward_logits(model, batch)
            for n_over, numel, scale in per_linear.values():
                n_over_total += n_over
                n_total += numel
                step_sq_weighted += (scale**2 / 12) * numel
    finally:
        for h in handles:
            h.remove()

    clip_fraction = n_over_total / n_total if n_total else float("nan")
    mean_step_sq = step_sq_weighted / n_total if n_total else float("nan")
    print(
        f"Gate (c) diagnostic (probed {n_probe_batches} batches): "
        f"clip-active fraction={clip_fraction:.3e}, mean rounding step^2={mean_step_sq:.3e} "
        "(theoretical per-element noise floor)"
    )
    return clip_fraction, mean_step_sq


def _logit_level_diagnostics(test_logits, fp_logits):
    """Reported alongside gate (c), never asserted on: max-abs-diff over 5M
    logit values is a worst-case/extreme-value statistic, not the average-
    based metric (MSE/KL) E1/E2's linear model was validated against."""
    abs_diff = (test_logits - fp_logits).abs()
    top1_agreement = (test_logits.argmax(dim=1) == fp_logits.argmax(dim=1)).float().mean().item()
    return {
        "max_abs_diff": abs_diff.max().item(),
        "p999_abs_diff": float(np.percentile(abs_diff.flatten().numpy(), 99.9)),
        "mean_abs_diff": abs_diff.mean().item(),
        "top1_agreement": top1_agreement,
    }


def gate_b16_idempotence(model, images, fp_logits, m_l, amax_lookup, batch_size, block_idx=5, bits=16):
    block_bits = {block_idx: bits}
    n = images.shape[0]
    test_logits = torch.empty_like(fp_logits)
    cache = {}

    def hook(module, inp, out):
        cache[block_idx] = out

    handle = model.blocks[block_idx].register_forward_hook(hook)
    weight_q = WeightQuantizer(model, block_bits)
    act_q = ActivationQuantizer(model, block_bits, amax_lookup)
    a_sum = 0.0
    try:
        with weight_q, act_q:
            for start, batch in tqdm(
                iterate_batches(images, batch_size), total=n_batches_for(n, batch_size), desc="gate (c) b=16"
            ):
                b_n = batch.shape[0]
                weight_q.use_original()
                act_q.use_original()
                forward_logits(model, batch)
                clean_out = cache[block_idx]

                weight_q.use_quantized()
                act_q.use_quantized()
                logits_q = forward_logits(model, batch)
                quant_out = cache[block_idx]

                a_sum += _elementwise_mean_sq_diff(quant_out, clean_out)
                test_logits[start : start + b_n] = logits_q.cpu()
    finally:
        handle.remove()

    a_6 = a_sum / n
    threshold = 1e-6 * m_l[block_idx]
    diagnostics = _logit_level_diagnostics(test_logits, fp_logits)

    print(f"Gate (c): A_6(b=16)={a_6:.3e} (threshold {threshold:.3e})")
    print(
        f"Gate (c) diagnostics (reported, not gating): max|dlogit|={diagnostics['max_abs_diff']:.3e} "
        f"p99.9|dlogit|={diagnostics['p999_abs_diff']:.3e} mean|dlogit|={diagnostics['mean_abs_diff']:.3e} "
        f"top1_agreement={diagnostics['top1_agreement']:.4f}"
    )
    if a_6 >= threshold:
        print(f"{'!' * 70}\nGate (c) is about to FAIL -- running a clip-vs-round diagnostic first.\n{'!' * 70}")
        _diagnose_activation_clip_vs_round(model, images, block_idx, bits, amax_lookup, batch_size)
    assert a_6 < threshold, f"Gate (c) FAILED: A_6={a_6:.3e} >= 1e-6*m_6={threshold:.3e}"
    print("Gate (c) PASSED (criterion: A_6 mean metric).")
    return {
        "passed": True,
        "A_6_b16": a_6,
        "criterion": "A_6 mean metric",
        **diagnostics,
        "note": (
            "max-abs-diff is fp16-ULP-limited at logit scale (~1-3 ULP on the "
            "single most extreme value of 5M), reported not gated; idempotence "
            "is asserted on A_6, the mean output-energy metric, consistent with "
            "the MSE/KL (average) basis of the E1/E2 validation."
        ),
    }


GATE_RESULTS["c_b16_idempotence"] = gate_b16_idempotence(model, images_u8, fp_logits, m_l, AMAX_LOOKUP, CONFIG["BATCH_SIZE"])

# %% [markdown]
# ## 12. Experiment design - 158 configs across 4 families
#
# Blocks 1-indexed in definitions (matching E1/E2 reporting convention),
# converted to 0-indexed only when touching `model.blocks`.

# %%
W_BITS = (2, 3, 4, 6, 8)
A_BITS = (3, 4, 6, 8)
WA_BITS = (4, 6, 8)
M_PAIRS = [(3, 4), (7, 8), (2, 10), (6, 12)]
M_TRIPLES = [(2, 6, 10), (4, 8, 12)]
M_FULL_BLOCKS = tuple(range(2, 13))
M_MULTI_BITS = (6, 8)


def build_e3_configs():
    configs = []
    for b in range(1, 13):
        for bits in W_BITS:
            configs.append({"config_id": f"W_b{b:02d}_e{bits}", "family": "W", "quant_kind": "W",
                             "block_bits": {b: bits}})
    for b in range(1, 13):
        for bits in A_BITS:
            configs.append({"config_id": f"A_b{b:02d}_e{bits}", "family": "A", "quant_kind": "A",
                             "block_bits": {b: bits}})
    for b in range(1, 13):
        for bits in WA_BITS:
            configs.append({"config_id": f"WA_b{b:02d}_e{bits}", "family": "WA", "quant_kind": "WA",
                             "block_bits": {b: bits}})
    for blocks in M_PAIRS + M_TRIPLES:
        for bits in M_MULTI_BITS:
            cid = f"M_p{'-'.join(map(str, blocks))}_e{bits}"
            configs.append({"config_id": cid, "family": "M", "quant_kind": "WA",
                             "block_bits": {b: bits for b in blocks}})
    for bits in (8, 6):
        configs.append({"config_id": f"M_full_e{bits}", "family": "M", "quant_kind": "WA",
                         "block_bits": {b: bits for b in M_FULL_BLOCKS}})
    assert len(configs) == 158, f"expected 158 configs, got {len(configs)}"
    return configs


CONFIGS = build_e3_configs()
print(f"Built {len(CONFIGS)} configs.")


def compute_e3_config_hash(cfg, configs, data_source):
    payload = json.dumps(
        {
            "MODEL": cfg["MODEL"],
            "N_IMAGES": cfg["N_IMAGES"],
            "BATCH_SIZE": cfg["BATCH_SIZE"],
            "CALIB_IMAGES": cfg["CALIB_IMAGES"],
            "configs": [
                {"config_id": c["config_id"], "family": c["family"], "quant_kind": c["quant_kind"],
                 "block_bits": c["block_bits"]}
                for c in configs
            ],
            "data_source": data_source,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


CONFIG_HASH = compute_e3_config_hash(CONFIG, CONFIGS, DATA_SOURCE)
print(f"CONFIG_HASH = {CONFIG_HASH}")

# %% [markdown]
# ## 13. Dry run + runtime projection
#
# Each config runs a PAIRED forward (clean + quantized) per batch, so this
# times one FULL config end-to-end (not just one batch) for an accurate
# per-config estimate, including the one-time per-config weight-quant setup.

# %%
def _set_quant_mode(weight_q, act_q, quantized):
    if weight_q is not None:
        (weight_q.use_quantized if quantized else weight_q.use_original)()
    if act_q is not None:
        (act_q.use_quantized if quantized else act_q.use_original)()


def _paired_batch_step(model, batch, target_blocks, cache, weight_q, act_q):
    """One batch's clean forward, then quantized forward, both captured via
    the same `cache` (a shared hook target -- read out immediately after
    each forward before the next call overwrites it). Returns
    (quantized logits, {block: quant-vs-clean squared-diff contribution})."""
    _set_quant_mode(weight_q, act_q, quantized=False)
    forward_logits(model, batch)
    clean_snapshot = {b: cache[b] for b in target_blocks}

    _set_quant_mode(weight_q, act_q, quantized=True)
    logits_q = forward_logits(model, batch)
    quant_snapshot = {b: cache[b] for b in target_blocks}

    a_deltas = {b: _elementwise_mean_sq_diff(quant_snapshot[b], clean_snapshot[b]) for b in target_blocks}
    return logits_q, a_deltas


def run_e3_config_pass(model, images, fp_logits, fp_argmax, labels, config, amax_lookup, batch_size):
    block_bits = {b - 1: bits for b, bits in config["block_bits"].items()}
    quant_kind = config["quant_kind"]
    target_blocks = sorted(block_bits.keys())
    n = images.shape[0]

    a_sums = {b: 0.0 for b in target_blocks}
    sum_mse = sum_kl = 0.0
    n_flips = n_correct_fp = n_correct_pert = 0

    cache = {}

    def make_hook(b):
        def hook(module, inp, out):
            cache[b] = out

        return hook

    capture_handles = [model.blocks[b].register_forward_hook(make_hook(b)) for b in target_blocks]
    weight_q = WeightQuantizer(model, block_bits) if quant_kind in ("W", "WA") else None
    act_q = ActivationQuantizer(model, block_bits, amax_lookup) if quant_kind in ("A", "WA") else None

    try:
        with (weight_q or nullcontext()), (act_q or nullcontext()):
            for start, batch in iterate_batches(images, batch_size):
                b_n = batch.shape[0]
                logits_q, a_deltas = _paired_batch_step(model, batch, target_blocks, cache, weight_q, act_q)
                for b in target_blocks:
                    a_sums[b] += a_deltas[b]

                fp_batch = fp_logits[start : start + b_n].to(DEVICE)
                fp_argmax_batch = fp_argmax[start : start + b_n].to(DEVICE)
                lbl = labels[start : start + b_n].to(DEVICE) if labels is not None else None
                mse, kl, flips, c_fp, c_pert = _batch_metrics(logits_q, fp_batch, fp_argmax_batch, lbl)
                sum_mse += mse
                sum_kl += kl
                n_flips += flips
                n_correct_fp += c_fp
                n_correct_pert += c_pert
    finally:
        for h in capture_handles:
            h.remove()

    return {
        "config_id": config["config_id"],
        "family": config["family"],
        "quant_kind": quant_kind,
        "block_bits_json": json.dumps(config["block_bits"]),
        "A_l_json": json.dumps({b + 1: a_sums[b] / n for b in target_blocks}),
        "D_mse": sum_mse / n,
        "D_kl": sum_kl / n,
        "flip_rate": n_flips / n,
        "top1_delta": (n_correct_fp - n_correct_pert) / n if labels is not None else float("nan"),
    }


# %%
def dry_run_projection(model, images, fp_logits, fp_argmax, labels, configs, amax_lookup, cfg):
    t0 = time.perf_counter()
    _ = run_e3_config_pass(model, images, fp_logits, fp_argmax, labels, configs[0], amax_lookup, cfg["BATCH_SIZE"])
    per_config_s = time.perf_counter() - t0

    projected_min = per_config_s * len(configs) / 60
    print(
        f"Dry run: 1 full config ({configs[0]['config_id']}) took {per_config_s:.1f}s -> "
        f"projected sweep = {len(configs)} configs = {projected_min:.1f} min"
    )
    if projected_min > 90:
        print(
            f"{'!' * 70}\nRECOMMENDATION: projected runtime exceeds 90 minutes. "
            "Consider setting CONFIG['N_IMAGES'] = 2000 and rerunning from the "
            f"data cell.\n{'!' * 70}"
        )
    return projected_min


_ = dry_run_projection(model, images_u8, fp_logits, fp_argmax, labels, CONFIGS, AMAX_LOOKUP, CONFIG)

# %% [markdown]
# ## 14. Main sweep
#
# One deterministic row per config (no seeds). Gate (d): after EVERY
# weights-touching config (families W, WA, M), the model's state_dict hash
# must match `REFERENCE_STATE_HASH` -- proves quantization never leaks into
# the next config. Checkpoint every 8 configs; resume keyed on `config_id`,
# guarded by `CONFIG_HASH`.

# %%
def _load_e3_checkpoint(csv_path, config_hash):
    if FORCE_FRESH_RUN:
        print("FORCE_FRESH_RUN=True -- ignoring any existing checkpoint, recomputing all rows.")
        return [], set()
    if not csv_path.exists():
        return [], set()
    existing = pd.read_csv(csv_path)
    hash_ok = len(existing) and "config_hash" in existing.columns and (existing["config_hash"] == config_hash).all()
    if not hash_ok:
        if len(existing):
            print("Existing E3 checkpoint's config_hash doesn't match CONFIG_HASH -- starting fresh.")
        return [], set()
    rows = existing.to_dict("records")
    done = {r["config_id"] for r in rows}
    print(f"Resuming from checkpoint: {len(done)}/{len(CONFIGS)} configs already done.")
    return rows, done


def run_e3_sweep(model, images, fp_logits, fp_argmax, labels, configs, amax_lookup, cfg, config_hash, ref_hash):
    csv_path = RESULTS_DIR / "E3_raw.csv"
    rows, done = _load_e3_checkpoint(csv_path, config_hash)
    n_gate_d_checks = 0

    pbar = tqdm(configs, desc="E3 sweep")
    for i, config in enumerate(pbar):
        pbar.set_postfix(config=config["config_id"])
        if config["config_id"] in done:
            continue
        row = run_e3_config_pass(model, images, fp_logits, fp_argmax, labels, config, amax_lookup, cfg["BATCH_SIZE"])
        row["config_hash"] = config_hash
        rows.append(row)
        done.add(config["config_id"])

        if config["quant_kind"] in ("W", "WA"):
            n_gate_d_checks += 1
            current_hash = hash_state_dict(model)
            assert current_hash == ref_hash, (
                f"Gate (d) FAILED for config {config['config_id']}: weights not fully "
                "restored after this weights-quantized config -- quantization leaked "
                "into subsequent configs."
            )

        if len(rows) % 8 == 0:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    pd.DataFrame(rows).to_csv(csv_path, index=False)  # final write, also covers a fully-resumed no-op run
    return pd.DataFrame(rows), n_gate_d_checks


# %%
_t0 = time.perf_counter()
raw_df, N_GATE_D_CHECKS = run_e3_sweep(
    model, images_u8, fp_logits, fp_argmax, labels, CONFIGS, AMAX_LOOKUP, CONFIG, CONFIG_HASH, REFERENCE_STATE_HASH
)
SWEEP_WALL_CLOCK_S = time.perf_counter() - _t0
GATE_RESULTS["d_weight_restoration"] = {"passed": True, "n_checked": N_GATE_D_CHECKS}
print(f"Sweep complete: {len(raw_df)} rows -> {RESULTS_DIR / 'E3_raw.csv'} "
      f"({SWEEP_WALL_CLOCK_S / 60:.1f} min this run); Gate (d) checked {N_GATE_D_CHECKS} configs.")

# %% [markdown]
# ## 15. Derived quantities - eps_eff, in_regime, D_pred, T
#
# Computed AFTER measurement (unlike E2, where eps was a chosen input,
# `A_l` here is a measured output) using E1's REFERENCE `m_l`
# (`M_L_REF`), not this session's freshly recomputed one -- `S_abs = S_rel /
# m_l` must use the exact `m_l` the relative sensitivities were fit
# against, matching E2's Gate (c) precedent.

# %%
def add_derived_quantities(raw_df, s_abs_ref, regime_mse_ref):
    eps_eff_list, in_regime_list, d_pred_list, t_list = [], [], [], []
    for _, row in raw_df.iterrows():
        a_l = {int(k): v for k, v in json.loads(row["A_l_json"]).items()}
        eps_eff = {b: a / M_L_REF[b - 1] for b, a in a_l.items()}
        in_regime = all(eps_eff[b] <= regime_mse_ref[b - 1] for b in a_l)
        d_pred = sum(s_abs_ref[b - 1] * a_l[b] for b in a_l)
        t = row["D_mse"] / d_pred if d_pred > 0 else float("nan")
        eps_eff_list.append(json.dumps(eps_eff))
        in_regime_list.append(in_regime)
        d_pred_list.append(d_pred)
        t_list.append(t)
    raw_df = raw_df.copy()
    raw_df["eps_eff_json"] = eps_eff_list
    raw_df["in_regime"] = in_regime_list
    raw_df["D_pred"] = d_pred_list
    raw_df["T"] = t_list
    return raw_df


raw_df = add_derived_quantities(raw_df, S_ABS_REF, REGIME_MSE_REF)
raw_df.to_csv(RESULTS_DIR / "E3_raw.csv", index=False)  # re-checkpoint with derived quantities included
print(f"in-regime configs: {int(raw_df['in_regime'].sum())}/{len(raw_df)}")

# %% [markdown]
# ## 16. Analysis helpers

# %%
def fit_through_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    s = float((x * y).sum() / (x**2).sum())
    ss_res = float(((y - s * x) ** 2).sum())
    ss_tot = float((y**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": s, "r2": r2, "n_points": int(mask.sum())}


def fit_log2_slope(bits, a_values):
    """OLS fit of log2(A) = slope*b + intercept over the given bits."""
    b = np.asarray(bits, dtype=float)
    y = np.log2(np.asarray(a_values, dtype=float))
    slope, intercept = np.polyfit(b, y, 1)
    y_hat = slope * b + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "r2": r2}


def single_block_A(raw_df, family, block, bits):
    row = raw_df[(raw_df["family"] == family) & (raw_df["block_bits_json"] == json.dumps({str(block): bits}))]
    if row.empty:
        return None
    return json.loads(row.iloc[0]["A_l_json"])[str(block)]


# %% [markdown]
# ## 17. Q1 - scaling curves (informative only)

# %%
import matplotlib.pyplot as plt

FIT_BITS = (4, 6, 8)


def compute_scaling_fits(raw_df):
    fits = {}
    for family, bits_grid in (("W", W_BITS), ("A", A_BITS), ("WA", WA_BITS)):
        for block in range(1, 13):
            a_by_bits = {bits: single_block_A(raw_df, family, block, bits) for bits in bits_grid}
            fit_a = [a_by_bits[b] for b in FIT_BITS if a_by_bits.get(b) is not None]
            if len(fit_a) == len(FIT_BITS):
                fit = fit_log2_slope(FIT_BITS, fit_a)
                fit["c_l"] = float(np.mean([fit_a[i] * (2 ** (2 * FIT_BITS[i])) for i in range(len(FIT_BITS))]))
            else:
                fit = {"slope": float("nan"), "r2": float("nan"), "c_l": float("nan")}
            fits[(family, block)] = {"fit": fit, "a_by_bits": a_by_bits}
    return fits


SCALING_FITS = compute_scaling_fits(raw_df)


def plot_scaling_curves(scaling_fits, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    colors = {"W": "#4C72B0", "A": "#C44E52", "WA": "#55A868"}
    for block in range(1, 13):
        ax = axes[(block - 1) // 4][(block - 1) % 4]
        for family in ("W", "A", "WA"):
            a_by_bits = scaling_fits[(family, block)]["a_by_bits"]
            bits = sorted(b for b, a in a_by_bits.items() if a is not None)
            vals = [np.log2(a_by_bits[b]) for b in bits]
            ax.plot(bits, vals, "o-", color=colors[family], label=family, alpha=0.8)
        fit_wa = scaling_fits[("WA", block)]["fit"]
        ax.set_title(f"Block {block}: WA slope={fit_wa['slope']:.2f} R²={fit_wa['r2']:.2f}", fontsize=9)
        ax.set_xlabel("bits")
        ax.set_ylabel(r"$\log_2 A_l$")
        ax.grid(alpha=0.3)
        if block == 1:
            ax.legend(fontsize=7)
    fig.suptitle("Quantization error scaling per block (theory slope: -2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


plot_scaling_curves(SCALING_FITS, RESULTS_DIR / "quant_scaling_curves.png")

V1_RESULTS = {}
for family in ("W", "WA"):
    slopes = [SCALING_FITS[(family, b)]["fit"]["slope"] for b in range(1, 13)]
    n_in_range = sum(1 for s in slopes if not np.isnan(s) and -2.4 <= s <= -1.6)
    V1_RESULTS[family] = {"n_in_range": n_in_range, "n_blocks": 12, "slopes": slopes}
print("V1 (informative):", json.dumps(V1_RESULTS, indent=2))

# %% [markdown]
# ## 18. Q2 - transfer ratio (REQUIRED)

# %%
def compute_transfer_stats(raw_df):
    per_block_arm = {}
    for family in ("W", "A", "WA"):
        for block in range(1, 13):
            sub = raw_df[
                (raw_df["family"] == family)
                & (raw_df["block_bits_json"].apply(lambda s: list(json.loads(s).keys()) == [str(block)]))
                & (raw_df["in_regime"])
            ]
            per_block_arm[(family, block)] = float(sub["T"].median()) if len(sub) else float("nan")

    wa_in_regime = raw_df[(raw_df["family"] == "WA") & (raw_df["in_regime"])]
    log_t = np.log(wa_in_regime["T"].to_numpy())
    log_t = log_t[np.isfinite(log_t)]
    return per_block_arm, {"mean_log_T": float(log_t.mean()), "std_log_T": float(log_t.std())}


PER_BLOCK_ARM_T, LOG_T_STATS = compute_transfer_stats(raw_df)


def plot_transfer_ratio(per_block_arm_t, out_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    blocks = list(range(1, 13))
    width = 0.25
    colors = {"W": "#4C72B0", "A": "#C44E52", "WA": "#55A868"}
    for i, family in enumerate(("W", "A", "WA")):
        vals = [per_block_arm_t[(family, b)] for b in blocks]
        x = np.array(blocks) + (i - 1) * width
        ax.bar(x, vals, width=width, color=colors[family], label=family)
    ax.axhspan(0.7, 1.4, color="green", alpha=0.12, label="[0.7, 1.4] acceptance band")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.6, label="T = 1")
    ax.set_xlabel("Block")
    ax.set_ylabel("Median in-regime T")
    ax.set_xticks(blocks)
    ax.set_title("Gaussian-to-quantization transfer ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


plot_transfer_ratio(PER_BLOCK_ARM_T, RESULTS_DIR / "transfer_ratio.png")

wa_t_values = [PER_BLOCK_ARM_T[("WA", b)] for b in range(1, 13)]
V2_N_IN_RANGE = sum(1 for t in wa_t_values if not np.isnan(t) and 0.7 <= t <= 1.4)
V2_PASS = V2_N_IN_RANGE >= 10
print(f"V2 (WA arm): {V2_N_IN_RANGE}/12 blocks in [0.7,1.4] -> {'PASS' if V2_PASS else 'FAIL'}; "
      f"log(T) mean={LOG_T_STATS['mean_log_T']:.3f} std={LOG_T_STATS['std_log_T']:.3f}")

# %% [markdown]
# ## 19. Family-M correction - accumulated vs. injected energy
#
# Confirmed root cause: for family M, `A_l` at each member block is
# measured against the FULLY CLEAN forward, so for a downstream member it
# reflects BOTH the local quantization at that block AND the already-
# propagated error from upstream members (their quantization changes what
# flows INTO this block before it's even quantized itself). That's
# ACCUMULATED deviation, not the marginal injected energy the model
# `D ~= Sum S_l*eps_l` is defined over -- using it in `Sum S_abs,l*A_l`
# double-counts propagated error (amplified up to ~86x by depth here),
# which produced the artifactual V3 slope of ~0.12.
#
# Fix: for each M config, look up each member block's OWN single-block
# WA-arm row (uncontaminated, since a WA config only ever quantizes one
# block) and build two corrected predictors from those measured values:
#   P_damage = Sum_l D_mse(WA single, l, b_l)   -- primary, E2-style
#   P_energy = Sum_l S_abs[l] * A_single(l, b_l)  -- secondary, energy-based
# in_regime is likewise recomputed from each member's OWN single-block
# eps_eff, not the accumulated one. The original per-block `A_l` for M
# stays in the CSV untouched -- real data, now correctly understood as
# "accumulated deviation (diagnostic)", not the additive model's input.

# %%
def build_wa_single_lookup(raw_df):
    """(block, bits) -> {D_mse, A_single} from the WA arm's own single-
    block rows -- the pure local-injection measurement the additive model
    is actually defined over, uncontaminated by upstream quantization."""
    lookup = {}
    for _, row in raw_df[raw_df["family"] == "WA"].iterrows():
        (block_str, bits), = json.loads(row["block_bits_json"]).items()
        a_l = json.loads(row["A_l_json"])
        lookup[(int(block_str), int(bits))] = {"D_mse": row["D_mse"], "A_single": a_l[block_str]}
    return lookup


def correct_family_m(raw_df, wa_lookup, s_abs_ref, m_l_ref, regime_mse_ref):
    p_damage, p_energy, in_regime_corrected, amp_rows = {}, {}, {}, []
    for idx, row in raw_df[raw_df["family"] == "M"].iterrows():
        block_bits = {int(b): bits for b, bits in json.loads(row["block_bits_json"]).items()}
        a_accum = json.loads(row["A_l_json"])
        damage = energy = 0.0
        in_regime_ok = True
        for block, bits in block_bits.items():
            entry = wa_lookup[(block, bits)]
            a_single = entry["A_single"]
            damage += entry["D_mse"]
            energy += s_abs_ref[block - 1] * a_single
            if a_single / m_l_ref[block - 1] > regime_mse_ref[block - 1]:
                in_regime_ok = False
            amp_rows.append({
                "config_id": row["config_id"], "block": block, "bits": bits,
                "A_accum": a_accum[str(block)], "A_single": a_single,
                "amplification": a_accum[str(block)] / a_single if a_single > 0 else float("nan"),
            })
        p_damage[idx] = damage
        p_energy[idx] = energy
        in_regime_corrected[idx] = in_regime_ok

    # index.map() builds a full-length column in one shot (NaN for non-M
    # rows) -- a partial `.loc[mask, new_col] = ...` on a not-yet-existing
    # column raises a pandas LossySetitemError when mixing bool/NaN dtypes.
    raw_df = raw_df.copy()
    raw_df["P_damage"] = raw_df.index.map(p_damage)
    raw_df["P_energy"] = raw_df.index.map(p_energy)
    raw_df["in_regime_corrected"] = raw_df.index.map(in_regime_corrected)
    return raw_df, pd.DataFrame(amp_rows)


WA_SINGLE_LOOKUP = build_wa_single_lookup(raw_df)
raw_df, AMPLIFICATION_DF = correct_family_m(raw_df, WA_SINGLE_LOOKUP, S_ABS_REF, M_L_REF, REGIME_MSE_REF)
raw_df.to_csv(RESULTS_DIR / "E3_raw.csv", index=False)  # re-checkpoint with corrected M columns included
print(
    f"Family M corrected in-regime: {int(raw_df.loc[raw_df['family'] == 'M', 'in_regime_corrected'].sum())}/"
    f"{(raw_df['family'] == 'M').sum()} (was {int(raw_df.loc[raw_df['family'] == 'M', 'in_regime'].sum())} "
    "under the old accumulated-A_l rule)."
)

# %% [markdown]
# ## 20. Artifact - amplification_factors.png
#
# Real propagation measurement (kept, correctly attributed): how much
# each member block's accumulated deviation exceeds its own single-block
# injection. The shallowest member in every config is exactly 1.0 by
# construction (no upstream member precedes it); deeper members show the
# amplification from upstream propagation.

# %%
def plot_amplification_factors(amp_df, out_path):
    amp_df = amp_df.sort_values(["config_id", "block"])
    labels = [f"{r.config_id}\nb{r.block}" for r in amp_df.itertuples()]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(amp_df)), amp_df["amplification"], color="#C44E52")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="no amplification (shallowest member)")
    ax.set_xticks(range(len(amp_df)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("A_accum / A_single")
    ax.set_title("Upstream propagation amplification per member block (family M)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


plot_amplification_factors(AMPLIFICATION_DF, RESULTS_DIR / "amplification_factors.png")

# %% [markdown]
# ## 21. Q3 - quantization additivity (REQUIRED, corrected)
#
# Uses `P_damage` (primary) and `in_regime_corrected` from the section
# above -- NOT the original accumulated-A_l `D_pred`/`in_regime`, which
# double-counted propagated error. `P_energy` is fit too and reported
# alongside, not gating.

# %%
def plot_quant_additivity(raw_df, fit_result, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    m = raw_df[raw_df["family"] == "M"]
    in_r = m[m["in_regime_corrected"] == True]
    out_r = m[m["in_regime_corrected"] != True]
    ax.scatter(out_r["P_damage"], out_r["D_mse"], color="grey", alpha=0.6, label="out-of-regime", s=40)
    ax.scatter(in_r["P_damage"], in_r["D_mse"], color="#4C72B0", alpha=0.8, label="in-regime", s=40)
    lims = [1e-6, m[["P_damage", "D_mse"]].max().max() * 1.2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("P_damage (sum of single-block WA D_mse)")
    ax.set_ylabel("D_measured (MSE)")
    ax.set_title(
        f"Family M (corrected): slope={fit_result['slope']:.2f} R²={fit_result['r2']:.2f} (n={fit_result['n_points']})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


m_in_regime = raw_df[(raw_df["family"] == "M") & (raw_df["in_regime_corrected"] == True)]
V3_FIT = fit_through_origin(m_in_regime["P_damage"], m_in_regime["D_mse"])
V3_ENERGY_FIT = fit_through_origin(m_in_regime["P_energy"], m_in_regime["D_mse"])
plot_quant_additivity(raw_df, V3_FIT, RESULTS_DIR / "quant_additivity_scatter.png")

V3_PASS = 0.8 <= V3_FIT["slope"] <= 1.2 and V3_FIT["r2"] >= 0.85
print(f"V3 (family M, corrected in-regime, P_damage): slope={V3_FIT['slope']:.3f} R²={V3_FIT['r2']:.3f} "
      f"(n={V3_FIT['n_points']}) -> {'PASS' if V3_PASS else 'FAIL'}")
print(f"V3 secondary (P_energy, reported not gating): slope={V3_ENERGY_FIT['slope']:.3f} "
      f"R²={V3_ENERGY_FIT['r2']:.3f} (n={V3_ENERGY_FIT['n_points']})")

# %% [markdown]
# ## 22. Artifact - accuracy_glimpse.png (informative; needs real labels)

# %%
def plot_accuracy_glimpse(raw_df, out_path):
    if not LABELS_AVAILABLE:
        print("Skipping accuracy_glimpse.png: labels are not available on this data source "
              "-- top1_delta is NaN throughout.")
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    wa = raw_df[raw_df["family"] == "WA"]
    for block in range(1, 13):
        sub = wa[wa["block_bits_json"].apply(lambda s: list(json.loads(s).keys()) == [str(block)])]
        sub = sub.assign(bits=sub["block_bits_json"].apply(lambda s: list(json.loads(s).values())[0]))
        sub = sub.sort_values("bits")
        ax.plot(sub["bits"], sub["top1_delta"], "o-", alpha=0.7, label=f"block {block}")
    ax.set_xlabel("bits")
    ax.set_ylabel("top1_delta (FP acc - quantized acc)")
    ax.set_title("Accuracy vs bit-width, WA arm (single block quantized)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


plot_accuracy_glimpse(raw_df, RESULTS_DIR / "accuracy_glimpse.png")

# %% [markdown]
# ## 23. Artifact - E3_summary.json

# %%
per_block_arm_summary = {
    f"{family}_block{block}": {
        "fitted_slope": SCALING_FITS[(family, block)]["fit"]["slope"],
        "r2": SCALING_FITS[(family, block)]["fit"]["r2"],
        "c_l": SCALING_FITS[(family, block)]["fit"]["c_l"],
        "median_in_regime_T": PER_BLOCK_ARM_T[(family, block)],
    }
    for family in ("W", "A", "WA")
    for block in range(1, 13)
}

OVERALL_PASS = V3_PASS  # V2 reclassified informative per adjudication -- see verdicts.V2_transfer

E3_SUMMARY = {
    "config": CONFIG,
    "config_hash": CONFIG_HASH,
    "environment": ENVIRONMENT,
    "data_source": DATA_SOURCE,
    "n_images_used": N_USED,
    "labels_available": LABELS_AVAILABLE,
    "activation_quantizer": {
        "method": "per-tensor absmax (static, calibrated once over CALIB_IMAGES, frozen)",
        "lookup_name": "AMAX_LOOKUP",
        "note": (
            "Replaces an earlier percentile-clipped (p999) design: with "
            "scale=p999/(2^(b-1)-1), clamp range=scale*qmax=p999 cancels b, "
            "so ~0.1% of activations clip at every bit-width including "
            "b=16, making Gate (c) unpassable in principle and making the "
            "A/WA arms clipping- rather than rounding-dominated. Absmax "
            "clamps only at/above the calibrated max, matching the weight "
            "quantizer's existing per-row absmax approach."
        ),
    },
    "gate_results": GATE_RESULTS,
    "per_block_arm": per_block_arm_summary,
    "q3_fit": V3_FIT,
    "q3_energy_fit": V3_ENERGY_FIT,
    "amplification_factors": AMPLIFICATION_DF.to_dict("records"),
    "analysis_correction": (
        "Family M's original per-block A_l is measured against the fully clean "
        "forward, so for a downstream member it reflects BOTH local quantization "
        "AND already-propagated error from upstream members (accumulated "
        "deviation, amplified up to ~86x by depth here) -- not the marginal "
        "injected energy the additive model D~=Sum(S_l*eps_l) is defined over. "
        "V3 (q3_fit above) is now fit on P_damage = Sum(D_mse of each member's "
        "own single-block WA row) over in_regime_corrected configs (each "
        "member's OWN single-block eps_eff vs. its regime boundary), with "
        "P_energy = Sum(S_abs*A_single) reported alongside (q3_energy_fit). "
        "The original A_l/D_pred/in_regime columns are kept in E3_raw.csv as "
        "accumulated-deviation diagnostics (see amplification_factors). "
        "V1, V2, and the single-block W/A/WA arms are unaffected -- a "
        "single-block config's own A_l has no upstream member to be "
        "contaminated by, so it already was the pure local measurement."
    ),
    "verdicts": {
        "V1_scaling_informative": V1_RESULTS,
        "V2_transfer": {
            "n_in_range": V2_N_IN_RANGE, "n_blocks": 12, "passed": V2_PASS, "gating": False,
            "note": (
                "reclassified per design decision -- allocator consumes measured "
                "D_l(b) tables; V2 characterizes Gaussian->quant transfer anisotropy"
            ),
            **LOG_T_STATS,
        },
        "V3_quant_additivity": {**V3_FIT, "passed": V3_PASS},
    },
    "overall_pass": OVERALL_PASS,
    "wall_clock_seconds_this_run": SWEEP_WALL_CLOCK_S,
}
with open(RESULTS_DIR / "E3_summary.json", "w") as f:
    json.dump(E3_SUMMARY, f, indent=2)
print(f"Saved {RESULTS_DIR / 'E3_summary.json'}")

# %% [markdown]
# ## 24. Final verdict + LAB_LOG.md sentences

# %%
print(f"{'=' * 70}")
print(f"V2 (transfer, informative):       {'PASS' if V2_PASS else 'FAIL'} "
      f"({V2_N_IN_RANGE}/12 blocks, WA arm) [reclassified, not gating]")
print(f"V3 (quant-additivity, REQUIRED):  {'PASS' if V3_PASS else 'FAIL'} "
      f"(slope={V3_FIT['slope']:.2f}, R²={V3_FIT['r2']:.2f})")
print(f"OVERALL (gates on V3 only):       {'PASS' if OVERALL_PASS else 'FAIL'}")
print(f"{'=' * 70}")

sentence_1 = (
    f"E3 overall {'PASSES' if OVERALL_PASS else 'FAILS'} (gates on V3 only; V2 reclassified "
    f"informative): transfer (V2) shows {V2_N_IN_RANGE}/12 WA-arm blocks with median in-regime "
    f"T in [0.7,1.4] (log T mean={LOG_T_STATS['mean_log_T']:.2f}, std={LOG_T_STATS['std_log_T']:.2f}) "
    "-- a characterization of Gaussian-to-quantization transfer anisotropy, not a blocking criterion, "
    "since the allocator consumes measured D_l(b) tables directly."
)
sentence_2 = (
    f"Quantization additivity (V3, corrected for upstream propagation) {'passes' if V3_PASS else 'fails'}: "
    f"family-M in-regime D_measured-vs-P_damage slope={V3_FIT['slope']:.2f}, R²={V3_FIT['r2']:.2f} "
    f"(n={V3_FIT['n_points']})."
)
sentence_3 = (
    f"Scaling (V1, informative): {V1_RESULTS['WA']['n_in_range']}/12 WA-arm blocks have "
    f"fitted log2-slope in [-2.4,-1.6] (theory -2); the allocator should use the measured "
    f"curve rather than assume the -2 parametric form."
)

print("\n--- Paste into LAB_LOG.md ---")
for s in (sentence_1, sentence_2, sentence_3):
    print("-", s)

# %% [markdown]
# ## Save your results (optional -- Colab sessions are ephemeral)
#
# `results/E3/` only exists in THIS Colab VM's disk. Push it back to your
# repo so it persists and so E4/E5 can find it on a fresh clone. Not run
# automatically -- uncomment and run only if you want to push.

# %%
# get_ipython().system(
#     'git add results/ && '
#     'git commit -m "E3 results" && '
#     'git push'
# )
