# %% [markdown]
# # E5 - Joint Token+Bit-Width Allocator & the Joint-vs-Sequential Kill Test (G1)
#
# The experiment the whole Phase-1 program feeds. E1 measured per-block
# Gaussian sensitivities; E3 measured per-(block,bits) quantization damage
# and proved additivity to 11 simultaneous blocks; E4/E4b measured
# per-(block,r) token-merge damage and proved schedule additivity (kappa=1.15
# for 2-point schedules). E5 composes both MEASURED damage tables into one
# objective, builds a joint (bits+tokens) allocator via a lambda-sweep
# solver, and tests the ONE claim that actually matters for TANDEM's thesis:
# that allocating jointly beats the best STAGED/sequential baseline (Kim et
# al., ICLR 2026 compression-order sense) at matched compute budget.
#
# Runs top-to-bottom on a single Colab GPU. Does NOT modify E1/E2/E3/E4.
# Their already-verified machinery (data loader, model loader, batch
# metrics, E3's quantizer classes + calibration, E4's Ward MergeScheduler)
# is copied verbatim into clearly marked cells.
#
# Three review checkpoints: STOP-1 (Section 21) and STOP-2
# RE-VERDICT (Section 25d) were unconditional live pauses during
# development; STOP-2 (Section 23) halts only if V1_E5 (cross-axis
# additivity) fails and remains a plain scientific assert, untouched by
# the note below. Reviewer-release note: STOP-1 and STOP-2 RE-VERDICT are
# now preserved as printed banners, not hard halts -- set
# TANDEM_INTERACTIVE=1 to restore the original halting behavior.
#
# **Outputs** (under `results/E5/`): `E5_familyX.csv`, `E5_plans.csv`,
# `E5_summary.json`, `pareto_top1_vs_bops.png`, `cross_axis_additivity.png`,
# `pred_vs_measured_plans.png`, `plan_composition_table.txt`.

# %% [markdown]
# ## 0. Clone the repo (so this and later notebooks share `results/` via git)
#
# This repo is public -- cloning it needs no authentication.
#
# Clones (or pulls, if already cloned this session) to `/content/tandem`,
# then changes into `/content/tandem/notebooks` -- every relative path in
# this notebook resolves exactly as it always has, no changes needed
# elsewhere. E1/E2/E3/E4's results, if already pushed, come along
# automatically with this same clone.

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
from contextlib import ExitStack, nullcontext
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
    "N_IMAGES_G1": 20000,
    "BATCH_SIZE": 128,
    "CALIB_IMAGES": 512,
    "MODEL": "deit_small_patch16_224",
}

# Independent verification: results/E5/E5_plans.csv and
# E5_g1_correctness_checkpoint.json ship committed in this repo, so a
# default Run All resumes from them instantly rather than recomputing --
# that's a self-consistency check (the checked-in numbers still match what
# this exact code produces), not a substitute for a real from-scratch run.
# Set FORCE_FRESH_RUN = True to ignore both existing checkpoints and
# recompute everything. (Unlike E1-E4, these two checkpoints are keyed by
# row identity only, with no config_hash gate -- FORCE_FRESH_RUN is the
# only way to force a full recompute if BUDGETS or CONFIG change between
# runs, since a stale checkpoint here won't self-detect the mismatch.)
FORCE_FRESH_RUN = False

# Reviewer-release mode: STOP-1 (Section 21) and STOP-2 RE-VERDICT (Section
# 25d) were live review pauses during development -- unconditional
# `raise RuntimeError` after printing a review block, requiring the reviewer
# to read it and manually re-run from the next cell. Default False here
# prints the identical review block and a "[checkpoint: ...]" line instead
# of raising, so a reviewer's Run All completes end-to-end. Set
# TANDEM_INTERACTIVE=1 (env var) to restore the original halting behavior.
# Every SCIENTIFIC gate (V1_E5/Family-X, gates a-f, sanity/theorem/
# data-source asserts) is a plain `assert` elsewhere in this notebook, NOT
# gated by this flag -- those still fail loudly in every mode.
INTERACTIVE = os.environ.get("TANDEM_INTERACTIVE", "0") == "1"

DEVICE = "cuda"
assert torch.cuda.is_available(), (
    "E5 requires a CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU."
)

RESULTS_DIR = Path("results/E5")
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


def load_eval_images(cfg, hf_token, n_requested=None):
    """Returns (images_u8, labels, labels_available=True, source="imagenet-1k", n_used).
    `n_requested` overrides cfg["N_IMAGES"] (used for E5's separate 20k G1
    pull). No fallback -- fails loudly if ImageNet-1k access isn't
    actually working."""
    n_requested = cfg["N_IMAGES"] if n_requested is None else n_requested
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
# Recomputed fresh so Gate (b) can check it against E1's stored `m_l`.

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
# ## 8. Load upstream artifacts + Gates (a), (b)
#
# Same loading tiers as E2-E4: local repo path -> Drive mount -> manual
# upload. Generalized here to fetch E1, E3, and E4's summary JSONs plus
# E4b's reanalysis JSON (which does not match the `*summary*.json` glob,
# so it gets its own explicit filename search).

# %%
def _find_file(dir_name, search_root=".", pattern="*[Ss]ummary*.json"):
    for d in Path(search_root).glob("results/*"):
        if d.is_dir() and d.name.lower() == dir_name:
            matches = sorted(d.glob(pattern))
            if matches:
                return matches[0]
    return None


def _try_drive_mount(dir_name, pattern="*[Ss]ummary*.json"):
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        return _find_file(dir_name, "/content/drive/MyDrive", pattern)
    except Exception as e:
        print(f"Drive mount unavailable ({type(e).__name__}: {e}).")
        return None


def _try_manual_upload(hint):
    print(f"Please upload {hint} when prompted.")
    try:
        from google.colab import files

        uploaded = files.upload()
        name = next(iter(uploaded), None)
        return Path(name) if name else None
    except Exception as e:
        print(f"Manual upload unavailable ({type(e).__name__}: {e}).")
        return None


def load_artifact_json(dir_name, hint, pattern="*[Ss]ummary*.json", required=True):
    """Search tiers: local repo -> Drive mount -> manual upload. `hint` is
    shown in the upload prompt and the failure message."""
    path = _find_file(dir_name, pattern=pattern)
    if path is None:
        print(f"{hint} not found locally under results/{dir_name}/. Trying Google Drive...")
        path = _try_drive_mount(dir_name, pattern)
    if path is None:
        path = _try_manual_upload(hint)
    if path is None:
        msg = (
            f"Could not locate {hint} via any loading tier. Place it under "
            f"results/{dir_name}/, or rerun and use the Drive/upload prompts."
        )
        if required:
            raise AssertionError(msg)
        print(f"WARNING: {msg} Proceeding without it.")
        return None
    print(f"Loaded {hint} from {path}")
    with open(path) as f:
        return json.load(f)


E1_SUMMARY = load_artifact_json("e1", "E1's summary JSON (results/E1/*summary*.json)")
S_L_MSE_REF = E1_SUMMARY["S_l_mse"]
M_L_REF = E1_SUMMARY["m_l"]
REGIME_MSE_REF = E1_SUMMARY["linear_regime_boundary_mse"]
S_ABS_REF = [S_L_MSE_REF[i] / M_L_REF[i] for i in range(12)]  # S_abs,l = S_rel,l / m_l

E3_SUMMARY = load_artifact_json("e3", "E3's summary JSON (results/E3/E3_summary.json)")
E4_SUMMARY = load_artifact_json("e4", "E4's summary JSON (results/E4/E4_summary.json)")
E4B_SUMMARY = load_artifact_json(
    "e4", "E4b's additivity-reanalysis JSON (results/E4/E4b_additivity_reanalysis.json)",
    pattern="E4b_additivity_reanalysis.json",
)


# %%
def gate_artifact_loading(e1_summary, e3_summary, e4_summary, e4b_summary, data_source):
    """Gate (a): every upstream artifact this allocator depends on must
    already be internally self-consistent (its OWN overall_pass true) and
    measured on the SAME data distribution as this session -- an allocator
    built from a still-failing or cross-distribution table would be
    invalid by construction, not just imprecise."""
    problems = []

    if e3_summary.get("overall_pass") is not True:
        problems.append(f"E3_summary.json overall_pass={e3_summary.get('overall_pass')!r}, expected True")
    if not e3_summary.get("analysis_correction"):
        problems.append("E3_summary.json is missing a non-empty 'analysis_correction' note")

    if e4_summary.get("overall_pass") is not True:
        problems.append(f"E4_summary.json overall_pass={e4_summary.get('overall_pass')!r}, expected True")
    e4_notes = e4_summary.get("notes", "") or ""
    e4_provenance = e4_summary.get("eps_abs_rescale_provenance")
    if "196" not in e4_notes and e4_provenance is None:
        problems.append(
            "E4_summary.json has no eps_abs 197->196 normalization note (neither 'notes' "
            "mentions 196 nor is 'eps_abs_rescale_provenance' present) -- this session's E4 "
            "results predate the CLS-inclusion off-by-one fix; rerun E4's Corrections 1 & 2 first."
        )

    if e4b_summary is None:
        problems.append("E4b_additivity_reanalysis.json could not be loaded")
    else:
        slope = e4b_summary.get("predictor_all_6_nan_dropped", {}).get("slope")
        if slope is None or not (0.7 <= slope <= 1.3):
            problems.append(f"E4b measured-damage slope={slope!r}, expected in [0.7, 1.3]")

    for name, summ in (("E1", e1_summary), ("E3", e3_summary), ("E4", e4_summary)):
        if summ.get("data_source") != data_source:
            problems.append(
                f"{name}_summary.json data_source={summ.get('data_source')!r} != "
                f"this session's data_source={data_source!r}"
            )

    assert not problems, "Gate (a) FAILED:\n  - " + "\n  - ".join(problems)
    return {"passed": True, "checked": ["E3.overall_pass", "E4.overall_pass", "E4.eps_norm_note",
                                         "E4b.slope_in_range", "data_source_match_all"]}


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
    "a_artifact_loading": gate_artifact_loading(E1_SUMMARY, E3_SUMMARY, E4_SUMMARY, E4B_SUMMARY, DATA_SOURCE),
    "b_m_l_consistency": gate_m_l_consistency(m_l, M_L_REF),
}
print("Gates (a), (b) passed:", json.dumps(GATE_RESULTS, indent=2))

# %% [markdown]
# ## 9. [VERBATIM FROM E3 - do not modify] Quantization simulator
#
# All quant math is forced to fp32 explicitly (`.float()` before
# round/clamp/scale, `.to(orig_dtype)` after) regardless of the ambient
# fp16 autocast context. Target modules per block: `attn.qkv`, `attn.proj`,
# `mlp.fc1`, `mlp.fc2` -- LayerNorm/softmax stay full precision, and
# attention matmuls (QK^T, AV) are never quantized, matching E3 exactly
# (the cost model's `16*16` attention term below encodes this).

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
# ## 10. [VERBATIM FROM E3 - do not modify] Calibration - static activation scales
#
# ONE pass over the first `CALIB_IMAGES` images, no quantization active,
# on the CLEAN full-token model -- the realistic deploy-once condition
# (activation scales are not re-calibrated per merge schedule). Per-tensor
# ABSMAX, not percentile (percentile clipping was tried and rejected in
# E3: it makes the clamp range bit-width-invariant, corrupting the
# rounding-dominated scaling law).

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
# ## 11. [VERBATIM FROM E4 - do not modify] Ward merge machinery
#
# E5's plan space only ever uses `method="ward"` (size-weighted Ward at
# block output, CLS never a merge candidate) -- `bipartite`/`random`
# methods and `ProportionalAttentionPatch` are NOT copied here, since
# `MergeScheduler`'s dispatch branches for them are only ever reached if a
# point's `method` is actually `"bipartite"`/`"random"`, which E5 never
# constructs. This is the exact machinery that found and fixed a real
# fp16-autocast contamination bug (gate (d) failure at max_rel_err=1.2e-3)
# and a per-batch state-leak bug (crash on the final, size-8 batch) during
# E4 -- both fixes are load-bearing here too, since E5 runs the identical
# forward-pass/hook pattern.

# %%
def ward_greedy_merge(features, sizes_init, r, track_membership=False):
    """Batched greedy Ward agglomeration via Lance-Williams cost updates --
    no python loop over images, only over the r merge steps (each step is
    one vectorized argmin + update across the whole batch). `sizes_init`
    need not be all-ones: a later point in a multi-point schedule starts
    from whatever sizes the earlier point left behind.

    This function is called from inside the model's forward pass, which
    runs under an ambient `torch.autocast(dtype=torch.float16)` context.
    Autocast silently re-downcasts the OUTPUT of autocast-eligible ops
    (torch.bmm included) to fp16 regardless of the input dtype, so an
    explicit `.float()` cast on the inputs alone is not enough -- the
    entire computation is wrapped in a nested `torch.autocast(enabled=
    False)`, which overrides the outer context for this scope only.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = features.float()
        assert features.dtype == torch.float32, (
            f"ward_greedy_merge requires float32 features, got {features.dtype}"
        )
        sizes_init = sizes_init.float()
        b, k, d = features.shape
        device = features.device
        centroids = features.clone()
        sizes = sizes_init.clone()
        active = torch.ones(b, k, dtype=torch.bool, device=device)
        assign = torch.arange(k, device=device).unsqueeze(0).repeat(b, 1) if track_membership else None

        # memory-efficient pairwise squared distance: ||a-b||^2 = |a|^2+|b|^2-2a.b
        sq_norms = (centroids**2).sum(-1)
        dot = torch.bmm(centroids, centroids.transpose(1, 2))
        sq_dist = (sq_norms.unsqueeze(2) + sq_norms.unsqueeze(1) - 2 * dot).clamp_min(0.0)
        # Permanent guards: sq_dist's batch dim must come from `features`
        # and sizes_init must agree with it -- if a caller ever passes a
        # stale/mismatched sizes_init, fail here with an explicit message.
        assert sq_dist.shape == (b, k, k), (
            f"sq_dist shape {tuple(sq_dist.shape)} != expected {(b, k, k)}; "
            f"features.shape={tuple(features.shape)}"
        )
        size_prod = sizes.unsqueeze(2) * sizes.unsqueeze(1)
        size_sum = sizes.unsqueeze(2) + sizes.unsqueeze(1)
        assert size_prod.shape == sq_dist.shape, (
            f"size_prod shape {tuple(size_prod.shape)} != sq_dist shape {tuple(sq_dist.shape)}; "
            f"sizes_init.shape={tuple(sizes_init.shape)} vs features.shape={tuple(features.shape)} "
            f"-- sizes_init's batch dim does not match features' batch dim"
        )
        cost = (size_prod / size_sum) * sq_dist
        diag = torch.eye(k, dtype=torch.bool, device=device).unsqueeze(0).expand(b, -1, -1)
        cost = cost.masked_fill(diag, float("inf"))

        cumulative_cost = torch.zeros(b, device=device)
        batch_idx = torch.arange(b, device=device)

        for _ in range(r):
            flat = cost.view(b, -1)
            min_idx = flat.argmin(dim=1)
            i, j = min_idx // k, min_idx % k

            s_i, s_j = sizes[batch_idx, i], sizes[batch_idx, j]
            s_new = s_i + s_j
            c_i, c_j = centroids[batch_idx, i], centroids[batch_idx, j]
            c_new = (s_i.unsqueeze(1) * c_i + s_j.unsqueeze(1) * c_j) / s_new.unsqueeze(1)
            merge_cost = cost[batch_idx, i, j]

            s_k = sizes
            old_ik, old_jk = cost[batch_idx, i, :], cost[batch_idx, j, :]
            denom = s_i.unsqueeze(1) + s_j.unsqueeze(1) + s_k
            new_row = (
                (s_i.unsqueeze(1) + s_k) * old_ik + (s_j.unsqueeze(1) + s_k) * old_jk - s_k * merge_cost.unsqueeze(1)
            ) / denom

            cumulative_cost = cumulative_cost + merge_cost
            centroids[batch_idx, i] = c_new
            sizes[batch_idx, i] = s_new
            sizes[batch_idx, j] = 0.0
            active[batch_idx, j] = False
            if track_membership:
                match = assign == j.unsqueeze(1)
                assign = torch.where(match, i.unsqueeze(1).expand(-1, k), assign)

            cost[batch_idx, i, :] = new_row
            cost[batch_idx, :, i] = new_row
            cost[batch_idx, i, i] = float("inf")
            cost[batch_idx, j, :] = float("inf")
            cost[batch_idx, :, j] = float("inf")

        order = torch.argsort((~active).float(), dim=1, stable=True)
        keep = order[:, : k - r]
        new_patches = torch.gather(centroids, 1, keep.unsqueeze(-1).expand(-1, -1, d))
        new_sizes = torch.gather(sizes, 1, keep)

        return {
            "patches": new_patches, "sizes": new_sizes, "cost": cumulative_cost,
            "centroids": centroids, "assign": assign, "active": active,
        }


def _assert_ward_theorem_gate(pre_merge_features, pre_merge_sizes, ward_result, rel_tol=1e-4):
    """Gate (d)-style runtime check reused from E4: independently
    recompute the final partition's WEIGHTED SSE from the PRE-MERGE
    features and check it matches the Lance-Williams-tracked cumulative
    cost -- on every batch of every Ward call, not just a one-time check."""
    assign = ward_result["assign"]
    centroids = ward_result["centroids"]
    b, k, d = pre_merge_features.shape
    gathered = torch.gather(centroids, 1, assign.unsqueeze(-1).expand(-1, -1, d))
    true_sse = (pre_merge_sizes.unsqueeze(-1) * (pre_merge_features - gathered) ** 2).sum(dim=(1, 2))
    cum_cost = ward_result["cost"]
    rel_err = (true_sse - cum_cost).abs() / cum_cost.clamp_min(1e-12)
    max_rel_err = rel_err.max().item()
    assert max_rel_err < rel_tol, (
        f"Gate (d) FAILED: theorem mismatch, max_rel_err={max_rel_err:.3e} >= {rel_tol:.0e} -- "
        "the cumulative Lance-Williams cost should equal the independently recomputed final SSE."
    )
    return max_rel_err


N_PATCHES = 196  # 14x14 grid, deit_small_patch16_224 at 224px
EPS_ABS_DIVISOR = N_PATCHES * 384  # per-element normalization: 196 patch tokens x 384 dims


class MergeScheduler:
    """Attaches one forward hook per merge point; hooks fire in block order
    during a single forward pass and share size state on the instance.
    `self.sizes` is reset EVERY batch (not just once ever) via the
    `is_first_point` flag -- the previous batch's size tensor (and its
    batch size, which differs for the final, short batch) must never leak
    into a new forward pass."""

    def __init__(self, model, points, prop_attn_patch=None, device=DEVICE):
        self.model = model
        self.points = points  # list of dicts: block_idx, r, method
        self.prop_attn_patch = prop_attn_patch  # unused in E5 (ward-only, no F_P)
        self.device = device
        self.handles = []
        self.sizes = None
        self.point_results = []
        self.gate_d_rel_errs = []

    def _make_hook(self, point, is_first_point):
        def hook(module, inp, out):
            cls_tok, patches = out[:, :1, :], out[:, 1:, :]
            b, n_cur, d = patches.shape
            if is_first_point:
                self.sizes = torch.ones(b, n_cur, device=self.device)
            assert self.sizes.shape[1] == n_cur, (
                f"size-vector/token-count mismatch at block {point['block_idx'] + 1}: "
                f"sizes has {self.sizes.shape[1]} entries, patches has {n_cur}"
            )
            r, method = point["r"], point["method"]
            feats_fp32 = patches.float()
            sizes_before = self.sizes

            if method == "ward":
                result = ward_greedy_merge(feats_fp32, sizes_before, r, track_membership=True)
                rel_err = _assert_ward_theorem_gate(feats_fp32, sizes_before, result)
                self.gate_d_rel_errs.append(rel_err)
            elif method == "bipartite":
                result = bipartite_merge(feats_fp32, sizes_before, r)
            elif method == "random":
                result = random_merge(feats_fp32, sizes_before, r, point["seed"])
            else:
                raise ValueError(f"unknown merge method {method!r}")

            new_patches, new_sizes, cost = result["patches"], result["sizes"], result["cost"]
            assert new_sizes.shape[1] == n_cur - r, (
                f"post-merge token count wrong at block {point['block_idx'] + 1}: "
                f"{new_sizes.shape[1]} != {n_cur - r}"
            )
            assert torch.equal(new_sizes.sum(dim=1), torch.full_like(new_sizes.sum(dim=1), float(N_PATCHES))), (
                f"size mass not conserved at block {point['block_idx'] + 1}: sizes must always sum to "
                f"{N_PATCHES} (every original patch token is accounted for exactly once, merged or not)"
            )
            self.sizes = new_sizes
            if self.prop_attn_patch is not None:
                cls_size = torch.ones(b, 1, device=self.device)
                self.prop_attn_patch.set_log_sizes(torch.cat([cls_size, new_sizes], dim=1).log())

            new_out = torch.cat([cls_tok, new_patches.to(out.dtype)], dim=1)
            assert torch.equal(new_out[:, 0, :], out[:, 0, :]), (
                f"CLS token (index 0) was disturbed by the merge hook at block {point['block_idx'] + 1} "
                "-- it must never be a merge candidate."
            )

            eps_abs = (cost.double() / EPS_ABS_DIVISOR).cpu()
            self.point_results.append({
                "block_idx": point["block_idx"], "r": r, "method": method,
                "eps_abs_mean": eps_abs.mean().item(), "eps_abs_per_image": eps_abs,
            })
            return new_out

        return hook

    def __enter__(self):
        for idx, point in enumerate(self.points):
            h = self.model.blocks[point["block_idx"]].register_forward_hook(self._make_hook(point, idx == 0))
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.sizes = None
        self.point_results = []
        self.gate_d_rel_errs = []

# %% [markdown]
# ## 12. Measured damage tables - Dq[l,bits] (E3, family WA) and Dm[l,r] (E4, family F_S)
#
# Both archived raw CSVs use 1-INDEXED block numbers in `config_id` and in
# the JSON columns (`block_bits_json`, `points_json`) -- confirmed by
# direct inspection (e.g. `WA_b01_e4` -> `block_bits_json={"1": 4}`;
# `FS_b03_r64` -> `points_json=[{"block": 3, "r": 64}]`). The 0-indexed
# `block_idx` convention only exists INTERNALLY, at the point each
# notebook calls `model.blocks[block_idx]` / `WeightQuantizer`/
# `MergeScheduler` -- both `run_e3_config_pass` and `build_e4_configs`
# convert 1-indexed table block numbers to 0-indexed `block_idx` right
# before that call (`{b - 1: bits ...}` / `l - 1`). E5's `PlanExecutor`
# below does the identical conversion; asserted here so a future change
# to either archived table's convention fails loudly instead of silently
# mis-indexing every plan built from it.

# %%
def _find_raw_csv(dir_name, filename):
    path = Path("results") / dir_name / filename
    if path.exists():
        return path
    for d in Path(".").glob("results/*"):
        if d.is_dir() and d.name.lower() == dir_name:
            cand = d / filename
            if cand.exists():
                return cand
    return None


E3_RAW_PATH = _find_raw_csv("e3", "E3_raw.csv")
E4_RAW_PATH = _find_raw_csv("e4", "E4_raw.csv")
assert E3_RAW_PATH is not None, "Could not find results/E3/E3_raw.csv (needed for the WA damage table)"
assert E4_RAW_PATH is not None, "Could not find results/E4/E4_raw.csv (needed for the F_S damage table)"
E3_RAW_DF = pd.read_csv(E3_RAW_PATH)
E4_RAW_DF = pd.read_csv(E4_RAW_PATH)

QUANT_BITS = (4, 6, 8)
MERGE_BLOCKS = (1, 3, 5, 7, 9, 11)
MERGE_RS = (32, 64, 98, 128)


def build_dq_table(e3_raw_df):
    """Dq[(l, bits)] = measured D_mse of quantizing block l ALONE (weights
    + activations) at `bits`, l in 1..12 (1-indexed, matching the table),
    bits in {4,6,8}. FP16 (bits=16) is not a table entry -- it is defined
    as exactly 0.0 damage (identity vs itself) wherever looked up."""
    wa = e3_raw_df[e3_raw_df["family"] == "WA"]
    table = {}
    for _, row in wa.iterrows():
        block_bits = json.loads(row["block_bits_json"])
        assert len(block_bits) == 1, f"expected a single-block WA row, got {block_bits}"
        (block_str, bits), = block_bits.items()
        table[(int(block_str), int(bits))] = float(row["D_mse"])
    expected = {(l, b) for l in range(1, 13) for b in QUANT_BITS}
    missing = expected - table.keys()
    assert not missing, f"Dq table missing entries: {sorted(missing)}"
    return table


def build_dm_table(e4_raw_df):
    """Dm[(l, r)] = measured D_mse of Ward-merging r tokens at block l
    ALONE, l in {1,3,5,7,9,11} (1-indexed, matching the table), r in
    {32,64,98,128}."""
    fs = e4_raw_df[e4_raw_df["family"] == "F_S"]
    table = {}
    for _, row in fs.iterrows():
        (pt,) = json.loads(row["points_json"])
        table[(int(pt["block"]), int(pt["r"]))] = float(row["D_mse"])
    expected = {(l, r) for l in MERGE_BLOCKS for r in MERGE_RS}
    missing = expected - table.keys()
    assert not missing, f"Dm table missing entries: {sorted(missing)}"
    return table


DQ_TABLE = build_dq_table(E3_RAW_DF)
DM_TABLE = build_dm_table(E4_RAW_DF)
print(f"Dq table: {len(DQ_TABLE)} (block,bits) entries. Dm table: {len(DM_TABLE)} (block,r) entries.")

# %% [markdown]
# ## 13. Plan representation, cost model, and objective
#
# A plan is `{"bits": {l: bits_or_16 for l in 1..12}, "merges": [{"block":
# l, "r": r}, ...]}` (at most 2 merge points, blocks/r on the measured
# grids, all 1-indexed to match the tables). `plan["bits"]` need not list
# every block explicitly -- `plan_bits(plan, l)` defaults to FP16 (16) for
# any block not present.
#
# Cost model (BOPs, formula-exact per spec): token count ENTERING block l
# is `N_l = 197 - sum(r for merge points at block p < l)` (merges apply at
# block p's OUTPUT, so block p itself still computes at its incoming
# count; only LATER blocks see the reduction). Attention matmuls are
# ALWAYS 16x16 (never quantized, matching E3). Patch-embed/head/LayerNorm/
# softmax are plan-invariant constants, excluded. Merge-operator compute
# overhead is excluded (Phase 3 measures real latency).

# %%
D_MODEL = 384
N_TOKENS_TOTAL = 197  # 196 patches + CLS

# STOP-2 RULING: WA4 removed from the precision menu everywhere
# (plan space, both solvers, both sequential stages). The original E5
# spec's menu {FP16,WA8,WA6,WA4} was itself a spec error -- WA4 is
# out-of-regime at all 12 blocks and was measured to compose
# superlinearly (Family X ratios 1.35/2.07 on the two WA4 configs),
# violating Design Decision D4. LEGACY_BIT_MENU exists ONLY so the two
# retained WA4 evidence configs (X8/X9, kept for the regime-enforcement
# finding, never discarded) can still be looked up/validated -- it is
# never used by any solver, enumeration, or sequential stage.
BIT_MENU = (16, 8, 6)
LEGACY_BIT_MENU = (16, 8, 6, 4)
KAPPA = {0: 1.0, 1: 1.0, 2: 1.15}  # E4b-measured multiplicative offset for 2-point schedules

WA4_EXCLUSION_JUSTIFICATION = (
    "WA4 excluded: out-of-regime at 12/12 blocks (eps_eff up to 0.69 vs boundaries 0.03/0.003) "
    "and empirically superlinear in composition (Family X ratios 1.35/2.07). WA6 retained under "
    "a documented exemption: violates the strict boundary at 6/12 blocks but composition is "
    "validated by E3 family-M (b=6, 0.93-1.27) and Family X (WA6-on-all-12 + merge, ratio 0.963)."
)


def plan_bits(plan, l):
    return plan["bits"].get(l, 16)


def n_tokens_entering(plan, l):
    """N_l: patch+CLS tokens entering block l (1-indexed). Merges at block
    p reduce what blocks AFTER p see, never block p itself."""
    return N_TOKENS_TOTAL - sum(m["r"] for m in plan["merges"] if m["block"] < l)


def linear_macs(n_l):
    return 12 * n_l * D_MODEL**2  # qkv 3Nd^2 + proj Nd^2 + fc1 4Nd^2 + fc2 4Nd^2


def attention_macs(n_l):
    return 2 * n_l**2 * D_MODEL  # QK^T and AV


def block_bops(n_l, bits):
    bw = ba = bits  # WA_b: both weights and activations at b bits; FP16: bw=ba=16
    return linear_macs(n_l) * (bw * ba) + attention_macs(n_l) * (16 * 16)


def plan_cost(plan):
    """C(plan) = sum over blocks of block_BOPs(N_l, bits_l)."""
    total = 0.0
    for l in range(1, 13):
        total += block_bops(n_tokens_entering(plan, l), plan_bits(plan, l))
    return total


def validate_plan(plan, bit_menu=None):
    """`bit_menu` defaults to the CURRENT (restricted) BIT_MENU -- pass
    LEGACY_BIT_MENU explicitly for the two retained WA4 evidence configs
    (X8/X9) only; every solver-constructed plan uses the strict default."""
    if bit_menu is None:
        bit_menu = BIT_MENU
    assert set(plan.keys()) == {"bits", "merges"}, f"unexpected plan keys: {plan.keys()}"
    for l, bits in plan["bits"].items():
        assert 1 <= l <= 12 and bits in bit_menu, f"bad bits entry ({l}, {bits})"
    assert len(plan["merges"]) <= 2, f"plan has {len(plan['merges'])} merge points, max is 2"
    seen_blocks = set()
    for m in plan["merges"]:
        assert m["block"] in MERGE_BLOCKS and m["r"] in MERGE_RS, f"bad merge point {m}"
        assert m["block"] not in seen_blocks, f"duplicate merge point at block {m['block']}"
        seen_blocks.add(m["block"])
    return True


def objective_d_pred(plan):
    """D_pred(plan) = Sum_l Dq[l,bits_l] + kappa(n_pts) * Sum_points Dm[l,r]
    (DESIGN DECISION 7: measured tables on both axes, summed additively)."""
    d_quant = sum(
        DQ_TABLE[(l, plan_bits(plan, l))] for l in range(1, 13) if plan_bits(plan, l) != 16
    )
    d_merge = sum(DM_TABLE[(m["block"], m["r"])] for m in plan["merges"])
    kappa = KAPPA.get(len(plan["merges"]), KAPPA[2])
    return d_quant + kappa * d_merge


IDENTITY_PLAN = {"bits": {}, "merges": []}
C_ID = plan_cost(IDENTITY_PLAN)
print(f"C_id (identity plan BOPs) = {C_ID:.4e}")

# %% [markdown]
# ## 14. PlanExecutor - composes the E3 quantizer with the E4 MergeScheduler
#
# ONE forward pass per batch (both mechanisms active simultaneously), not
# E3's own paired clean+quantized-per-batch pattern -- that pattern exists
# in E3 only to measure per-block A_l (marginal deviation) diagnostics,
# which E5 does not need: the objective/gates only ever compare a plan's
# output against the globally-cached `fp_logits`/`fp_argmax` (exactly
# E4's `run_e4_config_pass` pattern), so a single forward per batch is
# sufficient and roughly halves compute versus E3's own measurement loop.
#
# Weight quantization is entered ONCE per plan (`WeightQuantizer.__enter__`
# precomputes quantized weights ONCE), activation scales come from the
# ONE calibration pass above (never recalibrated per plan or per merge
# schedule -- the realistic deploy-once condition), and the merge
# scheduler's per-batch state is reset every batch regardless of batch
# size (defensive battery #2). A `hash_state_dict` check before/after
# confirms weights are restored EXACTLY, not just closely, after every
# plan (defensive insurance, cheap relative to a full image-set forward).

# %%
def plan_to_block_bits(plan):
    """1-indexed table block -> 0-indexed model.blocks[] key, FP16 blocks omitted
    (no quantizer entry needed for them)."""
    return {l - 1: bits for l, bits in plan["bits"].items() if bits != 16}


def plan_to_merge_points(plan):
    return [
        {"block_idx": m["block"] - 1, "r": m["r"], "method": "ward"}
        for m in sorted(plan["merges"], key=lambda m: m["block"])
    ]


def run_plan(model, images, fp_logits, fp_argmax, labels, plan, amax_lookup, batch_size,
             track_per_image_correct=False, collect_logits=False, bit_menu=None):
    """`bit_menu` defaults to the current (restricted) BIT_MENU -- pass
    LEGACY_BIT_MENU explicitly only for the retained WA4 evidence configs
    (X8/X9), which must still be EXECUTED (to get their real measured
    D_mse) even though the solvers can never construct them."""
    validate_plan(plan, bit_menu=bit_menu)
    block_bits = plan_to_block_bits(plan)
    merge_points = plan_to_merge_points(plan)
    n = images.shape[0]

    sum_mse = sum_kl = 0.0
    n_flips = n_correct_fp = n_correct_pert = 0
    per_image_correct = torch.zeros(n, dtype=torch.bool) if track_per_image_correct else None
    all_logits = torch.empty((n, fp_logits.shape[1]), dtype=torch.float32) if collect_logits else None

    weight_q = WeightQuantizer(model, block_bits) if block_bits else None
    act_q = ActivationQuantizer(model, block_bits, amax_lookup) if block_bits else None
    sched = MergeScheduler(model, merge_points) if merge_points else None

    pre_hash = hash_state_dict(model)
    gate_d_max = None
    with ExitStack() as stack:
        if weight_q is not None:
            stack.enter_context(weight_q)
            weight_q.use_quantized()
        if act_q is not None:
            stack.enter_context(act_q)
            act_q.use_quantized()
        if sched is not None:
            stack.enter_context(sched)

        for start, batch in iterate_batches(images, batch_size):
            b_n = batch.shape[0]
            logits = forward_logits(model, batch)
            fp_batch = fp_logits[start : start + b_n].to(DEVICE)
            fp_argmax_batch = fp_argmax[start : start + b_n].to(DEVICE)
            lbl = labels[start : start + b_n].to(DEVICE) if labels is not None else None
            mse, kl, flips, c_fp, c_pert = _batch_metrics(logits, fp_batch, fp_argmax_batch, lbl)
            sum_mse += mse
            sum_kl += kl
            n_flips += flips
            n_correct_fp += c_fp
            n_correct_pert += c_pert
            if track_per_image_correct and lbl is not None:
                per_image_correct[start : start + b_n] = (logits.argmax(dim=1) == lbl).cpu()
            if collect_logits:
                all_logits[start : start + b_n] = logits.cpu()

        # sched's accumulated state is cleared on __exit__, so gate (d)'s
        # worst-case must be read here, before the context closes.
        if sched is not None and sched.gate_d_rel_errs:
            gate_d_max = max(sched.gate_d_rel_errs)

    post_hash = hash_state_dict(model)
    assert post_hash == pre_hash, (
        "PlanExecutor FAILED to restore model weights exactly after running this plan -- "
        "a WeightQuantizer context did not clean up correctly."
    )
    if gate_d_max is not None:
        assert gate_d_max < 1e-4, f"Gate (d) FAILED inside PlanExecutor: max_rel_err={gate_d_max:.3e}"

    result = {
        "D_mse": sum_mse / n, "D_kl": sum_kl / n, "flip_rate": n_flips / n,
        "top1_delta": (n_correct_fp - n_correct_pert) / n if labels is not None else float("nan"),
        "top1_pert": n_correct_pert / n if labels is not None else float("nan"),
        "gate_d_max_rel_err": gate_d_max,
    }
    if track_per_image_correct:
        result["per_image_correct"] = per_image_correct
    if collect_logits:
        result["logits"] = all_logits
    return result


# %% [markdown]
# ## 15. Gate (c) - identity-plan reproduction
#
# The IDENTITY plan (`bits={}`, `merges=[]`) resolves to NO quantizer/
# scheduler being attached at all, so this is a plain forward pass through
# the composed executor's call path. Gated on MEAN metrics only
# (defensive battery #3): max|dlogit| is reported as a diagnostic, never
# asserted on, since a worst-case over 5M logit values is dominated by
# fp16 ULP jitter on the single most extreme value, not real signal.

# %%
def gate_identity_plan(model, images, fp_logits, fp_argmax, labels, amax_lookup, batch_size):
    result = run_plan(model, images, fp_logits, fp_argmax, labels, IDENTITY_PLAN, amax_lookup,
                       batch_size, collect_logits=True)
    test_logits = result["logits"]
    abs_diff = (test_logits - fp_logits).abs()
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()
    top1_agreement = (test_logits.argmax(dim=1) == fp_argmax).float().mean().item()

    assert mean_abs_diff < 1e-3, (
        f"Gate (c) FAILED: identity plan mean|dlogit|={mean_abs_diff:.3e} >= 1e-3 -- the composed "
        "executor is not reproducing the cached FP reference on a no-op plan."
    )
    assert top1_agreement > 0.999, (
        f"Gate (c) FAILED: identity plan top1_agreement={top1_agreement:.4f} <= 0.999"
    )
    return {
        "passed": True, "mean_abs_diff": mean_abs_diff, "top1_agreement": top1_agreement,
        "max_abs_diff_diagnostic_only": max_abs_diff,
    }


GATE_RESULTS["c_identity_plan"] = gate_identity_plan(
    model, images_u8, fp_logits, fp_argmax, labels, AMAX_LOOKUP, CONFIG["BATCH_SIZE"]
)
print("Gate (c):", json.dumps(GATE_RESULTS["c_identity_plan"], indent=2))

# %% [markdown]
# ## 16. Gate (d) - table fidelity + executor reproduction
#
# (i) The objective evaluated on a single-action plan must equal the raw
# table entry EXACTLY (no interpolation, no drift) -- this is checked for
# every one of the 36 Dq entries and 24 Dm entries, not a sample.
# (ii) Re-runs FOUR archived configs (two quant-only, two merge-only)
# through the NEW composed PlanExecutor and checks the freshly measured
# D_mse is within 2% of the value archived in E3/E4's own CSVs -- this is
# the proof that the composed executor actually reproduces the same
# instruments that built the tables, not just that the tables were copied
# correctly.

# %%
def gate_table_fidelity():
    for (l, bits), d_val in DQ_TABLE.items():
        plan = {"bits": {l: bits}, "merges": []}
        d_pred = objective_d_pred(plan)
        assert d_pred == d_val, f"Dq table drift at (block={l}, bits={bits}): objective={d_pred} != table={d_val}"
    for (l, r), d_val in DM_TABLE.items():
        plan = {"bits": {}, "merges": [{"block": l, "r": r}]}
        d_pred = objective_d_pred(plan)
        assert d_pred == d_val, f"Dm table drift at (block={l}, r={r}): objective={d_pred} != table={d_val}"
    return {"passed": True, "n_dq_checked": len(DQ_TABLE), "n_dm_checked": len(DM_TABLE)}


ARCHIVED_REPRO_CONFIGS = [
    {"kind": "quant", "block": 3, "bits": 8, "archived_config_id": "WA_b03_e8"},
    {"kind": "quant", "block": 9, "bits": 6, "archived_config_id": "WA_b09_e6"},
    {"kind": "merge", "block": 3, "r": 64, "archived_config_id": "FS_b03_r64"},
    {"kind": "merge", "block": 9, "r": 32, "archived_config_id": "FS_b09_r32"},
]


def gate_executor_reproduction(model, images, fp_logits, fp_argmax, labels, amax_lookup, batch_size, tol=0.02):
    rows = []
    for spec in ARCHIVED_REPRO_CONFIGS:
        if spec["kind"] == "quant":
            plan = {"bits": {spec["block"]: spec["bits"]}, "merges": []}
            archived = DQ_TABLE[(spec["block"], spec["bits"])]
        else:
            plan = {"bits": {}, "merges": [{"block": spec["block"], "r": spec["r"]}]}
            archived = DM_TABLE[(spec["block"], spec["r"])]
        measured = run_plan(model, images, fp_logits, fp_argmax, labels, plan, amax_lookup, batch_size)["D_mse"]
        rel_err = abs(measured - archived) / archived
        rows.append({
            "archived_config_id": spec["archived_config_id"], "archived_D_mse": archived,
            "measured_D_mse": measured, "rel_err": rel_err,
        })
        assert rel_err < tol, (
            f"Gate (d) FAILED: PlanExecutor reproduction of {spec['archived_config_id']} "
            f"measured D_mse={measured:.6f} vs archived={archived:.6f} (rel_err={rel_err:.3%} >= {tol:.0%})"
        )
    return {"passed": True, "configs": rows}


GATE_RESULTS["d_table_fidelity"] = gate_table_fidelity()
GATE_RESULTS["d_executor_reproduction"] = gate_executor_reproduction(
    model, images_u8, fp_logits, fp_argmax, labels, AMAX_LOOKUP, CONFIG["BATCH_SIZE"]
)
print("Gate (d):", json.dumps(GATE_RESULTS["d_table_fidelity"], indent=2))
print(json.dumps(GATE_RESULTS["d_executor_reproduction"], indent=2))

# %% [markdown]
# ## 17. Solver core - schedule enumeration + per-lambda independent bit argmin
#
# This is the SAME machinery used by gate (e)'s reduced-space exactness
# check and by the full production solver (Section 23) -- written once,
# parameterized by `blocks_in_scope`/`bit_menu`/`merge_blocks`/`merge_rs`/
# `max_merge_points`, so gate (e) is a genuine test of the actual solver
# code path, not a separate reimplementation that could silently diverge.
#
# Per DESIGN: conditioned on a fixed merge schedule, each block's bits
# choice affects ONLY that block's own (cost, damage) contribution --
# the attention term is bit-independent (always 16x16) and merges only
# change N_l, which is schedule-dependent but bits-independent. So for a
# FIXED schedule, minimizing Sum_l [Dq[l,bits_l] + lambda*cost_l(bits_l)]
# decomposes into 12 (or fewer, in a restricted scope) independent
# per-block minimizations -- no cross-block interaction to search over.

# %%
def enumerate_schedules(merge_blocks, merge_rs, max_points):
    """All merge schedules with 0..max_points points, at most one r per
    block, distinct blocks only.

    Two-point schedules are additionally constrained (STOP-1
    ruling -- an omission in the original spec, patched before Family X):
      (i) HARD FEASIBILITY: for points at blocks p1<p2 removing r1, r2
          tokens respectively, a Ward merge of r2 pairs at p2 needs at
          least r2+1 tokens actually present there once p1's merge has
          already run, i.e. r2 <= N_PATCHES - r1 - 1 (N_PATCHES=196 is
          the fixed count of mergeable patch tokens; CLS is never a merge
          candidate and is excluded from this count).
      (ii) VALIDATED-REGIME CAP: r1 + r2 <= max(merge_rs) -- a two-point
          schedule may not remove more tokens IN TOTAL than the largest
          SINGLE-point measurement in the archived damage table, keeping
          the kappa=1.15 two-point correction near its E4b-validated
          territory rather than extrapolating it to combinations E4b
          never measured.
    Single-point schedules are unconstrained by (i)/(ii) (there is no
    second point). On the production grid (6 blocks x {32,64,98,128}),
    this yields 25 single/empty (1 + 6*4, unchanged) + 60 two-point (of
    the naive C(6,2)*4^2=240 -- only (r1,r2) in
    {(32,32),(32,64),(64,32),(64,64)} survive both constraints for every
    block pair) = 85 total, down from the unconstrained 265."""
    schedules = [[]]
    single_points = [{"block": b, "r": r} for b in merge_blocks for r in merge_rs]
    schedules += [[p] for p in single_points]
    if max_points >= 2:
        for b1, b2 in itertools.combinations(merge_blocks, 2):
            for r1 in merge_rs:
                for r2 in merge_rs:
                    feasible = r2 <= N_PATCHES - r1 - 1
                    in_regime = r1 + r2 <= max(merge_rs)
                    if feasible and in_regime:
                        schedules.append([{"block": b1, "r": r1}, {"block": b2, "r": r2}])
    return schedules


def best_bits_for_schedule(schedule, lam, blocks_in_scope, bit_menu):
    """Fixed schedule, fixed lambda -> per-block independent bit argmin of
    Dq[l,bits] + lambda * linear_macs(N_l)*(bits^2). Returns {block: bits}
    covering exactly `blocks_in_scope` (blocks outside scope stay
    implicitly FP16 via `plan_bits`'s default)."""
    plan_stub = {"bits": {}, "merges": schedule}
    bits_choice = {}
    for l in blocks_in_scope:
        n_l = n_tokens_entering(plan_stub, l)
        best_bits, best_val = 16, float("inf")
        for bits in bit_menu:
            d = DQ_TABLE[(l, bits)] if bits != 16 else 0.0
            c = linear_macs(n_l) * (bits * bits)
            val = d + lam * c
            if val < best_val:
                best_val, best_bits = val, bits
        bits_choice[l] = best_bits
    return bits_choice


def joint_lambda_sweep(schedules, lambda_grid, blocks_in_scope, bit_menu):
    """Returns a list of (cost, D_pred, plan) tuples, one per (schedule,
    lambda) pair -- the raw candidate set the lower convex envelope is
    extracted from."""
    points = []
    for schedule in schedules:
        for lam in lambda_grid:
            bits_choice = best_bits_for_schedule(schedule, lam, blocks_in_scope, bit_menu)
            plan = {"bits": bits_choice, "merges": schedule}
            points.append((plan_cost(plan), objective_d_pred(plan), plan))
    return points


def lower_convex_hull(points, tol=1e-9):
    """Standard monotone-chain lower hull on (cost, D_pred), sorted by
    cost ascending; ties broken by lower D_pred first. Returns the hull
    subset, sorted by cost."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            c1, d1, _ = hull[-2]
            c2, d2, _ = hull[-1]
            c3, d3 = p[0], p[1]
            cross = (c2 - c1) * (d3 - d1) - (d2 - d1) * (c3 - c1)
            if cross <= tol * max(1.0, abs((c2 - c1) * (d3 - d1))):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def pareto_frontier(points, tol=1e-12):
    """TRUE discrete Pareto frontier (non-dominated set): p1 dominates p2
    if cost(p1) <= cost(p2) and D(p1) <= D(p2) with at least one strict.
    This is NOT restricted to the convex hull -- used only by gate (e) to
    check whether the (mathematically expected, but data-dependent)
    convexity of this additive/separable problem actually holds."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier = []
    best_d_so_far = float("inf")
    for p in pts:
        if p[1] < best_d_so_far - tol:
            frontier.append(p)
            best_d_so_far = p[1]
    return frontier


# %% [markdown]
# ### 17b. Exact per-schedule Pareto DP (pulled forward from the
# pre-committed refinement trigger -- see Section 27b)
#
# For a FIXED schedule, `exact_pareto_bits` folds blocks 1..12 sequentially
# (multi-choice knapsack: 3 menu options per block), maintaining only the
# non-dominated (cumulative_cost, cumulative_quant_damage) pairs after
# EVERY block fold via `pareto_frontier` (already defined, reused
# verbatim -- no separate pruning logic to maintain). This is EXACT for
# the additive objective: pruning a dominated pair can never discard the
# optimum for any budget, since anything reachable from a dominated pair
# is also reachable (at equal-or-lower cost, equal-or-lower damage) from
# whatever dominates it. Unlike the lambda-sweep, this reaches EVERY
# discrete combination on THIS schedule's own Pareto frontier, including
# duality-gap points -- verified against brute force in gate (e)'s third
# assertion (Section 18).
#
# Returns bits-only (cost, D_pred, bits_dict) triples -- the caller adds
# the schedule's own kappa-weighted merge damage on top (a constant shift
# that doesn't depend on the bits choice, so it is correct to add AFTER
# pruning, not before).

# %%
def exact_pareto_bits(schedule, blocks_in_scope, bit_menu):
    plan_stub = {"bits": {}, "merges": schedule}
    # Blocks NOT in blocks_in_scope are always FP16 (matching plan_bits'
    # default for any block absent from a plan's bits dict), but they are
    # NOT cost-free: plan_cost sums ALL 12 blocks, and a block's cost
    # still depends on its OWN N_l (which the schedule can still reduce,
    # e.g. blocks 5-12 sit downstream of a block-3 merge even when only
    # blocks 1-4 are in scope). Added as a fixed baseline so this
    # function's cumulative_cost matches plan_cost's convention exactly
    # for ANY blocks_in_scope, not just the full 12.
    fixed_outside_scope_cost = sum(
        block_bops(n_tokens_entering(plan_stub, l), 16) for l in range(1, 13) if l not in blocks_in_scope
    )
    pareto_set = [(fixed_outside_scope_cost, 0.0, {})]
    for l in blocks_in_scope:
        n_l = n_tokens_entering(plan_stub, l)
        options = [(block_bops(n_l, bits), (DQ_TABLE[(l, bits)] if bits != 16 else 0.0), bits) for bits in bit_menu]
        expanded = []
        for cum_c, cum_d, choices in pareto_set:
            for c, d, bits in options:
                new_choices = dict(choices)
                new_choices[l] = bits
                expanded.append((cum_c + c, cum_d + d, new_choices))
        pareto_set = pareto_frontier(expanded)
    return pareto_set


def exact_select_for_budget(schedule_pareto_sets, budget):
    """schedule_pareto_sets: list of (schedule, merge_damage, bits_pareto_set).
    Returns (cost, D_pred, plan) -- the global exact optimum over ALL
    schedules' Pareto sets, feasible at this budget."""
    best = None
    for schedule, merge_damage, pareto in schedule_pareto_sets:
        for c, bits_d, bits in pareto:
            if c > budget + 1e-6:
                continue
            total_d = bits_d + merge_damage
            if best is None or total_d < best[1] - 1e-15:
                best = (c, total_d, {"bits": bits, "merges": schedule})
    assert best is not None, f"no feasible exact-DP plan for budget {budget:.4e}"
    return best


def build_schedule_pareto_sets(schedules, blocks_in_scope, bit_menu):
    """One-time per-schedule computation, reused across all budget queries
    (bits Pareto sets don't depend on budget)."""
    result = []
    for schedule in schedules:
        kappa = KAPPA.get(len(schedule), KAPPA[2])
        merge_damage = kappa * sum(DM_TABLE[(m["block"], m["r"])] for m in schedule) if schedule else 0.0
        pareto = exact_pareto_bits(schedule, blocks_in_scope, bit_menu)
        result.append((schedule, merge_damage, pareto))
    return result


# %% [markdown]
# ## 18. Gate (e) - solver soundness and hull completeness
#
# STOP-1 RULING (second pass): the original "recover every true
# discrete Pareto-frontier point" spec was itself an error, not a
# bug to fix here -- a Lagrangian sweep can only ever reach lower-convex-
# hull points; interior discrete Pareto points are, in general, supported
# by no lambda at all (Everett / Shoham-Gersho). Gate (e) is formally
# re-scoped to the two claims that ARE achievable and must both hold:
#
# (i) SOUNDNESS: every point on the swept pool's OWN frontier (i.e. what
# `pareto_frontier(swept_points)` actually hands to plan selection) must
# be genuinely non-dominated within the EXHAUSTIVELY enumerated reduced
# space -- not merely non-dominated within the swept pool itself. This
# was checked carefully against the real archived table: raw, UNFILTERED
# per-schedule sweep points routinely dominate each other ACROSS
# different schedules (e.g. an aggressively-quantized no-merge point can
# be both costlier AND more damaging than a merge-only, no-quantization
# point from a different schedule) -- but `pareto_frontier()` already
# discards exactly these before anything is used for selection, and once
# restricted to that actual selection-frontier, verified ZERO domination
# violations against the full 162-plan brute force.
#
# (ii) HULL-COMPLETENESS: every lower-convex-hull vertex of the TRUE
# reduced-space frontier is recovered by the sweep -- mathematically
# guaranteed for a Lagrangian sweep with a wide-enough lambda grid, and
# confirmed exact here (10/10).
#
# Reduced space: blocks 1-4 only, bit menu {FP16,WA8,WA6} (STOP-2: swapped
# from WA4 to WA6, matching the production menu restriction), one
# optional merge point at block 3 with r in {0(=no merge),64}. 3^4 x 2 =
# 162 plans, small enough to enumerate exactly.
#
# The k (of 19, under the WA4 version; recomputed under WA6 here) true
# discrete Pareto-frontier points that are NEITHER on the hull NOR
# recovered by the swept frontier are recorded as a `duality_gap_points`
# diagnostic field, with the note text explaining why -- reported, never
# gated, per defensive-battery item 6. This is not closable at production
# scale either (3^12 x 85 combinations, brute force is infeasible there
# -- exactly why a lambda-sweep was chosen for production).
#
# (iii) DP-EXACTNESS (STOP-2 ruling, B5): the exact per-schedule
# Pareto DP (Section 17b) is a THIRD, separate claim -- checked directly
# against brute force over 20 test budgets spanning the reduced space's
# full cost range, requiring EXACT (cost, D_pred) match at every one
# (plan ties allowed), including duality-gap points, which the DP (unlike
# the lambda-sweep) MUST reach. This is what B1-B3's production upgrade
# (JOINT and both sequential stages' bit allocation) depends on -- with
# this holding, BEST-SEQUENTIAL's bit-allocation stage now uses the
# IDENTICAL exact primitive as JOINT, so the joint-vs-sequential
# COMPARISON in G1 isolates staging as the only remaining difference (a
# permanent consistency guard in Section 25 separately asserts JOINT
# weakly dominates BEST-SEQUENTIAL's predicted D_pred at every production
# budget).

# %%
REDUCED_BLOCKS = (1, 2, 3, 4)
REDUCED_BIT_MENU = (16, 8, 6)
REDUCED_MERGE_BLOCKS = (3,)
REDUCED_MERGE_RS = (64,)


def brute_force_reduced_space():
    schedules = enumerate_schedules(REDUCED_MERGE_BLOCKS, REDUCED_MERGE_RS, max_points=1)
    assert len(schedules) == 2, f"expected 2 schedules (no-merge, merge), got {len(schedules)}"
    all_plans = []
    for schedule in schedules:
        for bits_combo in itertools.product(REDUCED_BIT_MENU, repeat=len(REDUCED_BLOCKS)):
            bits = dict(zip(REDUCED_BLOCKS, bits_combo))
            plan = {"bits": bits, "merges": schedule}
            all_plans.append((plan_cost(plan), objective_d_pred(plan), plan))
    assert len(all_plans) == 162, f"expected 162 reduced-space plans, got {len(all_plans)}"
    return all_plans


def _dominates(a, b, tol_cost=1e-3, tol_d=1e-9):
    """a=(cost,D,plan) dominates b=(cost,D,plan): cost(a)<=cost(b) and
    D(a)<=D(b), with at least one strict (scale-appropriate tolerances --
    costs are exact-integer-valued BOPs at ~1e11 scale, D values are table
    sums at ~0.01-2 scale with negligible floating-point summation noise)."""
    ca, da, _ = a
    cb, db, _ = b
    at_least_as_good = ca <= cb + tol_cost and da <= db + tol_d
    strictly_better = ca < cb - tol_cost or da < db - tol_d
    return at_least_as_good and strictly_better


def gate_solver_soundness_and_hull_completeness():
    all_plans = brute_force_reduced_space()
    true_frontier = pareto_frontier(all_plans)
    true_hull = lower_convex_hull(all_plans)

    lambda_grid = np.logspace(-15, -2, 300)
    swept_points = joint_lambda_sweep(
        enumerate_schedules(REDUCED_MERGE_BLOCKS, REDUCED_MERGE_RS, max_points=1),
        lambda_grid, REDUCED_BLOCKS, REDUCED_BIT_MENU,
    )
    swept_frontier = pareto_frontier(swept_points)
    swept_hull = lower_convex_hull(swept_points)

    # sanity: the sweep's cost range must actually span the reduced
    # space's true min/max achievable cost (confirms the lambda grid is
    # wide enough, not just internally self-consistent).
    true_costs = [c for c, _, _ in all_plans]
    swept_costs = [c for c, _, _ in swept_points]
    assert min(swept_costs) <= min(true_costs) + 1e-6, "lambda grid never reached the min-cost plan"
    assert max(swept_costs) >= max(true_costs) - 1e-6, "lambda grid never reached the max-cost plan"

    # (i) SOUNDNESS: every point on the swept pool's OWN frontier (what
    # plan selection actually uses) must be genuinely non-dominated
    # within the full exhaustive enumeration -- checked against ALL 162
    # brute-force plans, not just the swept pool.
    unsound_points = []
    for sp in swept_frontier:
        for ap in all_plans:
            if _dominates(ap, sp):
                unsound_points.append({
                    "swept": {"cost": sp[0], "D_pred": sp[1], "plan": sp[2]},
                    "dominated_by": {"cost": ap[0], "D_pred": ap[1], "plan": ap[2]},
                })
                break
    assert not unsound_points, (
        f"Gate (e) SOUNDNESS FAILED: {len(unsound_points)} point(s) on the swept pool's own "
        f"selection frontier are dominated by a plan from the exhaustive enumeration -- this WOULD "
        f"be a real solver bug (unlike the expected, separately-diagnosed duality gap below). "
        f"{json.dumps(unsound_points[:5], indent=2)}"
    )

    # (ii) HULL-COMPLETENESS: every lower-convex-hull vertex of the TRUE
    # reduced-space frontier is recovered by the sweep.
    true_hull_keys = {(round(c, 6), round(d, 12)) for c, d, _ in true_hull}
    swept_hull_keys = {(round(c, 6), round(d, 12)) for c, d, _ in swept_hull}
    missed_hull = true_hull_keys - swept_hull_keys
    extra_hull = swept_hull_keys - true_hull_keys
    assert not missed_hull and not extra_hull, (
        f"Gate (e) HULL-COMPLETENESS FAILED: the lambda-sweep's convex hull does not exactly match "
        f"the brute-forced convex hull (missed={len(missed_hull)}, spurious={len(extra_hull)}) -- "
        "this WOULD be a real solver bug."
    )

    # DIAGNOSTIC (never gated): duality-gap points -- true non-dominated
    # frontier points the swept pool's own frontier never reaches, because
    # no lambda for their schedule selects that exact per-block bits
    # combination (Everett / Shoham-Gersho: Lagrangian relaxation reaches
    # only the lower convex hull; interior discrete Pareto points are, in
    # general, unsupported by any lambda -- the documented resolution).
    swept_frontier_keys = {(round(c, 6), round(d, 12)) for c, d, _ in swept_frontier}
    duality_gap_points = [
        {"cost": c, "D_pred": d, "plan": p} for c, d, p in true_frontier
        if (round(c, 6), round(d, 12)) not in swept_frontier_keys
    ]
    note = (
        f"{len(duality_gap_points)}/{len(true_frontier)} true discrete Pareto-frontier points are "
        "in the Lagrangian duality gap: individually non-dominated (no other single plan beats "
        "them on both axes) yet unreachable by ANY lambda for their own schedule, because "
        "different blocks have different sensitivities and their local rate-distortion curves "
        "don't share a common marginal-rate breakpoint. Textbook property of discrete/integer rate "
        "allocation (Everett / Shoham-Gersho), not a bug. Reported here, never gated."
    )
    print(f"Gate (e) diagnostic (NOT gated): {note}")

    # (iii) DP-EXACTNESS (STOP-2 ruling, B5): the exact per-schedule
    # Pareto DP must recover the TRUE brute-force optimum EXACTLY at every
    # one of 20 test budgets spanning the reduced space's full cost range
    # -- including duality-gap points, which (unlike the lambda-sweep) it
    # MUST reach, since it is not restricted to what a single shared
    # lambda can select.
    reduced_schedules = enumerate_schedules(REDUCED_MERGE_BLOCKS, REDUCED_MERGE_RS, max_points=1)
    schedule_pareto_sets = build_schedule_pareto_sets(reduced_schedules, REDUCED_BLOCKS, REDUCED_BIT_MENU)
    test_budgets = np.linspace(min(true_costs), max(true_costs), 20)
    dp_exactness_failures = []
    for budget in test_budgets:
        feasible_bf = [(c, d, p) for c, d, p in all_plans if c <= budget + 1e-6]
        bf_best = min(feasible_bf, key=lambda t: t[1])
        dp_best = exact_select_for_budget(schedule_pareto_sets, budget)
        if abs(dp_best[0] - bf_best[0]) > 1e-3 or abs(dp_best[1] - bf_best[1]) > 1e-9:
            dp_exactness_failures.append({
                "budget": float(budget),
                "brute_force": {"cost": bf_best[0], "D_pred": bf_best[1]},
                "exact_dp": {"cost": dp_best[0], "D_pred": dp_best[1]},
            })
    assert not dp_exactness_failures, (
        f"Gate (e) DP-EXACTNESS FAILED at {len(dp_exactness_failures)}/{len(test_budgets)} test "
        f"budgets: {json.dumps(dp_exactness_failures[:5], indent=2)}"
    )
    print(f"Gate (e) DP-exactness: exact-DP matches brute-force optimum at all {len(test_budgets)} "
          "test budgets (including duality-gap points).")

    return {
        "passed": True,
        "n_plans": len(all_plans), "n_true_frontier": len(true_frontier),
        "n_true_hull": len(true_hull), "n_swept_hull": len(swept_hull),
        "soundness_passed": True, "soundness_n_checked": len(swept_frontier),
        "hull_completeness_passed": True,
        "duality_gap_points": duality_gap_points,
        "n_duality_gap_points": len(duality_gap_points),
        "dp_exactness_passed": True, "n_test_budgets_checked": len(test_budgets),
        "note": note,
    }


GATE_RESULTS["e_solver_soundness_and_hull_completeness"] = gate_solver_soundness_and_hull_completeness()
print("Gate (e):", json.dumps(
    {k: v for k, v in GATE_RESULTS["e_solver_soundness_and_hull_completeness"].items()
     if k != "duality_gap_points"},
    indent=2,
))

# %% [markdown]
# ## 19. Gate (f) - per-mechanism toggle test
#
# One mixed config (merge at block 5, r=64; WA6 at blocks 6,7), run three
# ways: merge-only, quant-only, both. Sanity that the composed executor
# actually has BOTH mechanisms active when a mixed plan says so -- not a
# scientific claim about additivity (that is V1_E5/Family X's job).

# %%
GATE_F_MERGE_ONLY = {"bits": {}, "merges": [{"block": 5, "r": 64}]}
GATE_F_QUANT_ONLY = {"bits": {6: 6, 7: 6}, "merges": []}
GATE_F_BOTH = {"bits": {6: 6, 7: 6}, "merges": [{"block": 5, "r": 64}]}


def gate_mechanism_toggle(model, images, fp_logits, fp_argmax, labels, amax_lookup, batch_size):
    d_merge = run_plan(model, images, fp_logits, fp_argmax, labels, GATE_F_MERGE_ONLY, amax_lookup, batch_size)["D_mse"]
    d_quant = run_plan(model, images, fp_logits, fp_argmax, labels, GATE_F_QUANT_ONLY, amax_lookup, batch_size)["D_mse"]
    d_both = run_plan(model, images, fp_logits, fp_argmax, labels, GATE_F_BOTH, amax_lookup, batch_size)["D_mse"]
    assert d_both > d_merge and d_both > d_quant, (
        f"Gate (f) FAILED: both-mechanisms D_mse={d_both:.6f} does not exceed BOTH single-mechanism "
        f"values (merge-only={d_merge:.6f}, quant-only={d_quant:.6f}) -- one mechanism may not be active."
    )
    return {"passed": True, "D_mse_merge_only": d_merge, "D_mse_quant_only": d_quant, "D_mse_both": d_both}


GATE_RESULTS["f_mechanism_toggle"] = gate_mechanism_toggle(
    model, images_u8, fp_logits, fp_argmax, labels, AMAX_LOOKUP, CONFIG["BATCH_SIZE"]
)
print("Gate (f):", json.dumps(GATE_RESULTS["f_mechanism_toggle"], indent=2))
print("\nAll gates (a)-(f) passed.")

# %% [markdown]
# ## 20. Budget selection
#
# C_min: the true minimum-cost plan over the FULL space is WA4 on every
# block (cost is strictly monotonic in bits: 4^2 < 6^2 < 8^2 < 16^2, so
# WA4 is cost-optimal at every block regardless of schedule) combined with
# whichever 2-point merge schedule minimizes cost -- found by brute force
# over all schedules at bits=4 (cheap, no GPU). 6 budgets, log-spaced in
# [max(1.05*C_min, 0.15*C_id), 0.80*C_id].
#
# STOP-1 PATCH: the schedule enumeration (Section 17) now enforces
# hard feasibility and a validated-regime cap on two-point schedules. This
# was not cosmetic -- checking it directly against the archived table
# found that the UNPATCHED enumeration's own C_min-minimizing schedule was
# `[(block 1, r=128), (block 3, r=128)]`, which leaves block 4 onward with
# N_l = 197 - 128 - 128 = -59 tokens: a physically impossible plan that
# was silently setting the cost floor. The patched enumeration is asserted
# to have exactly 85 schedules (25 single/empty + 60 two-point, down from
# the unconstrained 265); C_min is recomputed from this corrected space
# and compared against what the unpatched space would have produced.

# %%
FULL_SCHEDULES = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
_n_empty = sum(1 for s in FULL_SCHEDULES if len(s) == 0)
_n_single = sum(1 for s in FULL_SCHEDULES if len(s) == 1)
_n_double = sum(1 for s in FULL_SCHEDULES if len(s) == 2)
assert len(FULL_SCHEDULES) == 85, (
    f"expected 85 schedules (25 single/empty + 60 two-point) after the STOP-1 feasibility/regime "
    f"patch, got {len(FULL_SCHEDULES)} (empty={_n_empty}, single={_n_single}, double={_n_double}). "
    f"Full schedule list: {json.dumps(FULL_SCHEDULES, indent=2)}"
)


def compute_c_min(schedules, bit_menu=None):
    """The cheapest available bit-width (min(bit_menu), not hardcoded) is
    always cost-optimal at every block regardless of schedule -- reads
    the CURRENT global BIT_MENU at call time if bit_menu is not given
    explicitly, so this stays correct across the STOP-2 menu restriction
    without needing a separate copy."""
    if bit_menu is None:
        bit_menu = BIT_MENU
    cheapest_bits = min(bit_menu)
    all_cheapest_bits = {l: cheapest_bits for l in range(1, 13)}
    costs = [(plan_cost({"bits": all_cheapest_bits, "merges": s}), s) for s in schedules]
    return min(costs, key=lambda cs: cs[0])


# unpatched-space comparison (diagnostic only, not used for anything downstream)
_unpatched_schedules = [[]] + [[{"block": b, "r": r}] for b in MERGE_BLOCKS for r in MERGE_RS]
for _b1, _b2 in itertools.combinations(MERGE_BLOCKS, 2):
    for _r1 in MERGE_RS:
        for _r2 in MERGE_RS:
            _unpatched_schedules.append([{"block": _b1, "r": _r1}, {"block": _b2, "r": _r2}])
C_MIN_UNPATCHED, C_MIN_SCHEDULE_UNPATCHED = compute_c_min(_unpatched_schedules)

C_MIN, C_MIN_SCHEDULE = compute_c_min(FULL_SCHEDULES)
BUDGET_LOW = max(1.05 * C_MIN, 0.15 * C_ID)
BUDGET_HIGH = 0.80 * C_ID
assert BUDGET_LOW < BUDGET_HIGH, (
    f"budget range is empty/inverted: low={BUDGET_LOW:.4e} >= high={BUDGET_HIGH:.4e} -- "
    "C_min is not small enough relative to C_id for this budget window."
)
BUDGETS = np.logspace(np.log10(BUDGET_LOW), np.log10(BUDGET_HIGH), 6).tolist()

print(f"C_id = {C_ID:.4e}")
print(f"Schedule count: {len(FULL_SCHEDULES)} (empty={_n_empty}, single={_n_single}, double={_n_double})")
print(f"C_min UNPATCHED (diagnostic, invalid schedule) = {C_MIN_UNPATCHED:.4e}, "
      f"schedule={C_MIN_SCHEDULE_UNPATCHED}")
print(f"C_min PATCHED (used from here on)              = {C_MIN:.4e}, schedule={C_MIN_SCHEDULE}")
print(f"Budget range: [{BUDGET_LOW:.4e}, {BUDGET_HIGH:.4e}] "
      f"(floor binds at 15% of C_id: {abs(BUDGET_LOW - 0.15 * C_ID) < 1e-6})")
print("Chosen budgets:", [f"{b:.4e}" for b in BUDGETS])

# %% [markdown]
# ### 20b. Gate (e) regression check + STOP-1 patch summary
#
# Re-runs gate (e) unchanged: its reduced-space schedule enumeration uses
# `max_points=1`, so the new two-point feasibility/regime constraints
# (Section 17) are never exercised there -- this re-run confirms the patch
# left it untouched, not just asserts it by inspection.

# %%
GATE_RESULTS["e_solver_soundness_and_hull_completeness_regression"] = gate_solver_soundness_and_hull_completeness()
assert (GATE_RESULTS["e_solver_soundness_and_hull_completeness_regression"]
        == GATE_RESULTS["e_solver_soundness_and_hull_completeness"]), (
    "Gate (e) regression check FAILED: result changed after the STOP-1 schedule-enumeration patch, "
    "but gate (e)'s reduced space (max_points=1) should be completely unaffected by it."
)

print(f"{'=' * 70}")
print("STOP-1 PATCH APPLIED")
print(f"{'=' * 70}")
print(f"Schedule count: {len(FULL_SCHEDULES)} (25 single/empty + 60 two-point, was 265)")
print(f"C_min: {C_MIN_UNPATCHED:.4e} (unpatched, invalid schedule {C_MIN_SCHEDULE_UNPATCHED}) -> "
      f"{C_MIN:.4e} (patched, valid schedule {C_MIN_SCHEDULE})")
print(f"Budgets: {[f'{b:.4e}' for b in BUDGETS]} (unchanged from unpatched -- 15% of C_id floor "
      "was already binding in both cases)")
_gate_e_regression = GATE_RESULTS["e_solver_soundness_and_hull_completeness_regression"]
print(f"Gate (e): {'PASS' if _gate_e_regression['passed'] else 'FAIL'} (soundness="
      f"{_gate_e_regression['soundness_passed']}, hull_completeness={_gate_e_regression['hull_completeness_passed']}, "
      f"regression-checked, reduced space unaffected by the patch, as expected)")
print(f"{'=' * 70}")

# %% [markdown]
# ## 21. STOP-1 - mandatory review checkpoint
#
# Interactive mode (`TANDEM_INTERACTIVE=1`): this cell ALWAYS halts
# (raises), regardless of gate outcomes, once the review block below has
# been printed. Re-run only the cells AFTER this one, and only once the
# reviewer has reviewed the block. Non-interactive (default, reviewer
# release): the identical review block still prints, but execution
# continues automatically past a "[checkpoint: ...]" line instead of
# raising -- see the `INTERACTIVE` flag defined in Section 2.

# %%
_DRY_RUN_CONFIG = GATE_F_BOTH  # arbitrary representative mixed plan; timing ~ image/batch count, not plan content
_t0_dry = time.perf_counter()
_ = run_plan(model, images_u8, fp_logits, fp_argmax, labels, _DRY_RUN_CONFIG, AMAX_LOOKUP, CONFIG["BATCH_SIZE"])
_PER_CONFIG_5K_SECONDS = time.perf_counter() - _t0_dry
_N_FAMILY_X_CONFIGS = 10
_N_PLAN_AUDIT_CONFIGS = 2 * len(BUDGETS)  # joint + best-sequential, per budget
_PROJECTED_STOP1_SECONDS = _PER_CONFIG_5K_SECONDS * (_N_FAMILY_X_CONFIGS + _N_PLAN_AUDIT_CONFIGS)

print(f"{'=' * 70}")
print("STOP-1 REVIEW BLOCK -- paste this for review before launching Family X")
print(f"{'=' * 70}")
print("Gate outcomes:")
for name, result in GATE_RESULTS.items():
    print(f"  {name}: {'PASS' if result.get('passed') else 'FAIL'}")
print(f"\nReachable cost range: C_min={C_MIN:.4e}  C_id={C_ID:.4e}  "
      f"(min reaches {C_MIN / C_ID:.1%} of identity cost)")
print(f"Chosen budgets (fraction of C_id): {[f'{b / C_ID:.1%}' for b in BUDGETS]}")
print(f"Per-5000-image-config wall-clock (dry run): {_PER_CONFIG_5K_SECONDS:.1f}s")
print(f"Projected Family X ({_N_FAMILY_X_CONFIGS} configs) + plan-audit "
      f"({_N_PLAN_AUDIT_CONFIGS} configs) wall-clock: "
      f"{_PROJECTED_STOP1_SECONDS / 60:.1f} min")
print(f"{'=' * 70}")

if INTERACTIVE:
    raise RuntimeError(
        "STOP-1: mandatory pause. Paste the review block above for review before "
        "launching Family X. Re-run only the cells AFTER this one once cleared -- "
        "this halt is unconditional and does not indicate a failure."
    )
else:
    print("[checkpoint: non-interactive mode -- continuing]")

# %% [markdown]
# ## 22. Family X - cross-axis additivity (REQUIRED verdict V1_E5)
#
# Quant-only additivity is proven (E3, family M); merge-only is proven
# (E4b); quant+merge TOGETHER has never been measured. 10 configs mixing
# both axes, run through the real composed PlanExecutor; predictor =
# `objective_d_pred(plan)` (the SAME objective the solver optimizes).

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


FAMILY_X_CONFIGS = {
    "X1": {"bits": {5: 8, 6: 8, 7: 8, 8: 8}, "merges": [{"block": 3, "r": 64}]},
    "X2": {"bits": {l: 8 for l in range(1, 13)}, "merges": [{"block": 3, "r": 32}, {"block": 9, "r": 32}]},
    "X3": {"bits": {7: 6, 8: 6}, "merges": [{"block": 5, "r": 98}]},
    "X4": {"bits": {2: 8, 3: 8}, "merges": [{"block": 1, "r": 32}]},
    "X5": {"bits": {l: 6 for l in range(1, 13)}, "merges": [{"block": 7, "r": 64}]},
    "X6": {"bits": {10: 8, 11: 8, 12: 8}, "merges": [{"block": 9, "r": 98}]},
    "X7": {"bits": {4: 6, 5: 6, 6: 6}, "merges": [{"block": 3, "r": 64}, {"block": 9, "r": 64}]},
    "X8": {"bits": {2: 4, 3: 4, 4: 4}, "merges": [{"block": 11, "r": 32}]},  # WA4 -- excluded_out_of_menu
    "X9": {"bits": {6: 4, 7: 4}, "merges": [{"block": 5, "r": 64}]},  # WA4 -- excluded_out_of_menu
    "X10": {"bits": {l: 8 for l in range(1, 13)}, "merges": [{"block": 1, "r": 64}, {"block": 7, "r": 32}]},
    # STOP-2 ruling: WA6 twins of X8/X9, same merge points, replacing WA4 with WA6.
    "X8b": {"bits": {2: 6, 3: 6, 4: 6}, "merges": [{"block": 11, "r": 32}]},
    "X9b": {"bits": {6: 6, 7: 6}, "merges": [{"block": 5, "r": 64}]},
}
EXCLUDED_OUT_OF_MENU_CONFIGS = {"X8", "X9"}  # WA4 evidence -- retained, never fit, never discarded

for _cid, _plan in FAMILY_X_CONFIGS.items():
    validate_plan(_plan, bit_menu=LEGACY_BIT_MENU if _cid in EXCLUDED_OUT_OF_MENU_CONFIGS else BIT_MENU)


def n_quantized_downstream_of_merge(plan):
    """Informative: how many quantized blocks sit strictly after the
    EARLIEST merge point -- the QUOTA-section calibration-shift mechanism
    candidate (activation scales calibrated on the clean full-token
    model; a downstream quantized block's activations have already been
    reshaped by an upstream merge when the plan actually runs)."""
    if not plan["merges"]:
        return 0
    earliest_merge_block = min(m["block"] for m in plan["merges"])
    return sum(1 for l, bits in plan["bits"].items() if bits != 16 and l > earliest_merge_block)


def run_family_x(model, images, fp_logits, fp_argmax, labels, amax_lookup, batch_size):
    rows = []
    for config_id, plan in tqdm(FAMILY_X_CONFIGS.items(), desc="Family X"):
        bit_menu = LEGACY_BIT_MENU if config_id in EXCLUDED_OUT_OF_MENU_CONFIGS else BIT_MENU
        measured = run_plan(model, images, fp_logits, fp_argmax, labels, plan, amax_lookup, batch_size,
                             bit_menu=bit_menu)
        d_pred = objective_d_pred(plan)
        rows.append({
            "config_id": config_id, "plan_json": json.dumps(plan), "D_pred": d_pred,
            "D_mse": measured["D_mse"], "top1_delta": measured["top1_delta"],
            "ratio_measured_over_pred": measured["D_mse"] / d_pred if d_pred > 0 else float("nan"),
            "n_quant_downstream_of_merge": n_quantized_downstream_of_merge(plan),
            "excluded_out_of_menu": config_id in EXCLUDED_OUT_OF_MENU_CONFIGS,
        })
    return pd.DataFrame(rows)


FAMILY_X_DF = run_family_x(model, images_u8, fp_logits, fp_argmax, labels, AMAX_LOOKUP, CONFIG["BATCH_SIZE"])
FAMILY_X_DF.to_csv(RESULTS_DIR / "E5_familyX.csv", index=False)

# STOP-2 ruling: V1_E5 is fit ONLY on the 10 menu-consistent
# configs (8 original survivors + X8b + X9b); X8/X9 (WA4) stay in the CSV
# flagged excluded_out_of_menu=true -- evidence for the regime-enforcement
# finding (F-X1), never silently dropped, never fit.
FAMILY_X_MENU_CONSISTENT = FAMILY_X_DF[~FAMILY_X_DF["excluded_out_of_menu"]]
assert len(FAMILY_X_MENU_CONSISTENT) == 10, (
    f"expected 10 menu-consistent Family X configs, got {len(FAMILY_X_MENU_CONSISTENT)}"
)

V1_FIT = fit_through_origin(FAMILY_X_MENU_CONSISTENT["D_pred"], FAMILY_X_MENU_CONSISTENT["D_mse"])
V1_E5 = {**V1_FIT, "passed": bool(0.8 <= V1_FIT["slope"] <= 1.2 and V1_FIT["r2"] >= 0.85)}

QUOTA_CORRELATION = float(np.corrcoef(
    FAMILY_X_MENU_CONSISTENT["n_quant_downstream_of_merge"],
    FAMILY_X_MENU_CONSISTENT["ratio_measured_over_pred"],
)[0, 1])

print("Full ratio table (all 12 configs, including the 2 WA4 evidence rows):")
print(FAMILY_X_DF[["config_id", "D_pred", "D_mse", "ratio_measured_over_pred",
                    "n_quant_downstream_of_merge", "excluded_out_of_menu"]].to_string(index=False))
print(f"\nV1_E5 (cross-axis additivity, 10 menu-consistent configs): slope={V1_E5['slope']:.3f} "
      f"R²={V1_E5['r2']:.3f} (n={V1_E5['n_points']}) -> {'PASS' if V1_E5['passed'] else 'FAIL'}")
print(f"QUOTA correlation (menu-consistent subset, informative): {QUOTA_CORRELATION:.3f}")

# %% [markdown]
# ### 22b. Findings F-X1, F-X2 and D4 amendment (STOP-2 ruling)

# %%
FINDING_F_X1 = {
    "id": "F-X1",
    "title": "WA4 superlinear composition -> menu enforcement",
    "statement": (
        "The two WA4 Family-X configs (X8: WA4 on blocks {2,3,4} + merge(11,32); X9: WA4 on "
        "blocks {6,7} + merge(5,64)) measured ratio_measured_over_pred of "
        f"{FAMILY_X_DF.loc[FAMILY_X_DF['config_id'] == 'X8', 'ratio_measured_over_pred'].iloc[0]:.3f} "
        f"and "
        f"{FAMILY_X_DF.loc[FAMILY_X_DF['config_id'] == 'X9', 'ratio_measured_over_pred'].iloc[0]:.3f} "
        "respectively -- composition is superlinear (measured damage exceeds the summed-table "
        "prediction by up to ~2x), not merely offset. WA4 is out-of-regime at 12/12 blocks "
        "(eps_eff up to 0.69 vs E1 boundaries 0.03/0.003), so this is consistent with compounding "
        "a Gaussian-regime-violating quantizer with a second distortion mechanism (merging). "
        "Resolution: WA4 removed from the precision menu everywhere (plan space, both solvers, "
        "both sequential stages); X8/X9 retained in E5_familyX.csv as evidence, never fit."
    ),
    "wa4_exclusion_justification": WA4_EXCLUSION_JUSTIFICATION,
    "x8_ratio": float(FAMILY_X_DF.loc[FAMILY_X_DF["config_id"] == "X8", "ratio_measured_over_pred"].iloc[0]),
    "x9_ratio": float(FAMILY_X_DF.loc[FAMILY_X_DF["config_id"] == "X9", "ratio_measured_over_pred"].iloc[0]),
}

FINDING_F_X2 = {
    "id": "F-X2",
    "title": "Structured mild sub-additivity on the menu-consistent subset",
    "statement": (
        f"On the 10 menu-consistent configs, ratio_measured_over_pred correlates with "
        f"n_quant_downstream_of_merge at r={QUOTA_CORRELATION:.3f} -- configs with MORE quantized "
        "blocks downstream of a merge point tend to measure BELOW their predicted damage (mild "
        "sub-additivity), not above. Mechanism: merging DELETES tokens whose quantization noise "
        "would otherwise have injected into the residual stream -- fewer surviving tokens carry "
        "quantization error forward, so downstream damage is structurally reduced. This sign is "
        "OPPOSITE to the QUOTA calibration-mismatch concern (which predicts downstream quantized "
        "blocks should measure ABOVE prediction, since their activation statistics have already "
        "been reshaped by an upstream merge relative to the clean-model calibration). The two "
        "mechanisms partially offset; the net effect measured here is mild sub-additivity, "
        "correctly captured within V1_E5's acceptance band."
    ),
    "correlation": QUOTA_CORRELATION,
}

D4_AMENDMENT = {
    "id": "D4-amendment",
    "statement": (
        "Design Decision D4 is amended to an operational 'validated composition territory' "
        "criterion: a precision menu entry is admissible only if its composition with the token "
        "axis has been empirically validated (E3 family-M and/or Family X measurements), not "
        "merely if it individually sits near the E1 linear regime boundary. WA8 and WA6 both "
        "satisfy this (WA6 under a documented exemption: violates the strict per-block regime "
        "boundary at 6/12 blocks, but E3 family-M (b=6, ratios 0.93-1.27) and Family X "
        "(WA6-on-all-12 + merge, ratio 0.963) empirically validate its composition behavior). WA4 "
        "does not satisfy this criterion (F-X1) and is excluded."
    ),
}

SPEC_ERROR_NOTE = (
    "The original E5 spec's precision menu {FP16,WA8,WA6,WA4} was itself a spec error: including "
    "WA4 violated Design Decision D4's (now-amended) validated-composition-territory requirement, "
    "since WA4 is out-of-regime at all 12 blocks and was never validated for quant+merge "
    "composition before being placed in the plan space. This is the second on-the-record "
    "spec correction this experiment (the first being gate (e)'s original 'recover every discrete "
    "Pareto-frontier point' criterion at STOP-1, corrected to the achievable convex-hull claim)."
)

print(f"\n{FINDING_F_X1['id']}: {FINDING_F_X1['title']}")
print(f"  X8 ratio={FINDING_F_X1['x8_ratio']:.3f}  X9 ratio={FINDING_F_X1['x9_ratio']:.3f}")
print(f"\n{FINDING_F_X2['id']}: {FINDING_F_X2['title']}")
print(f"  correlation={FINDING_F_X2['correlation']:.3f}")

# %% [markdown]
# ## 23. STOP-2 - conditional halt on V1_E5
#
# Halts ONLY if V1_E5 fails (the allocator's objective would be invalid
# as-is and this must be ruled on before any plan sweep). If V1_E5
# passes, execution proceeds automatically -- no separate pause.

# %%
print(f"{'=' * 70}")
print(f"V1_E5 (cross-axis additivity, REQUIRED, 10 menu-consistent configs): "
      f"{'PASS' if V1_E5['passed'] else 'FAIL'} "
      f"(slope={V1_E5['slope']:.3f}, R²={V1_E5['r2']:.3f}, n={V1_E5['n_points']})")
print(f"{'=' * 70}")
assert V1_E5["passed"], (
    "STOP-2: V1_E5 FAILED even on the menu-consistent subset. The allocator's summed "
    "measured-damage objective is not validated under joint quant+merge composition -- halt here; "
    "this must be ruled on before any plan sweep (joint solver, sequential baselines, or "
    "G1) is run."
)
print("V1_E5 PASSED (menu-consistent subset) -- proceeding to the joint solver and sequential baselines.")

# %% [markdown]
# ### 23b. Budget re-check under the restricted menu (STOP-2 ruling, A4)
#
# C_min under {FP16,WA8,WA6} is necessarily HIGHER than the old WA4-based
# C_min (WA6 costs more than WA4 at every block, 6²=36 vs 4²=16), so the
# reachable cost range shrinks from below. Recomputed here explicitly
# (re-asserting BIT_MENU is the restricted menu first, so this is correct
# regardless of whether the kernel was restarted or is resuming from
# STOP-2) and compared against the SIX budgets already chosen at STOP-1 --
# those must NOT move, since the floor is `max(1.05*C_min, 0.15*C_id)` and
# 0.15*C_id was already confirmed binding for the old (smaller) C_min; a
# larger C_min can only keep that floor binding or, if it overtakes 15%
# of C_id, change the low end -- checked explicitly, not assumed.

# %%
BIT_MENU = (16, 8, 6)  # re-asserted explicitly; see Section 13's definition for the ruling/justification
C_MIN_OLD_MENU, C_MIN_SCHEDULE_OLD_MENU = C_MIN, C_MIN_SCHEDULE  # captured before recompute, for comparison
BUDGETS_BEFORE_MENU_RESTRICTION = list(BUDGETS)

C_MIN, C_MIN_SCHEDULE = compute_c_min(FULL_SCHEDULES, BIT_MENU)
BUDGET_LOW = max(1.05 * C_MIN, 0.15 * C_ID)
BUDGET_HIGH = 0.80 * C_ID
assert BUDGET_LOW < BUDGET_HIGH, (
    f"budget range is empty/inverted under the restricted menu: low={BUDGET_LOW:.4e} >= "
    f"high={BUDGET_HIGH:.4e}."
)
_floor_binds_at_15pct = abs(BUDGET_LOW - 0.15 * C_ID) < 1e-6
assert _floor_binds_at_15pct, (
    f"Gate: the 15%-of-C_id floor no longer binds under the restricted menu (C_min={C_MIN:.4e} "
    f"now exceeds 15% of C_id={0.15 * C_ID:.4e}) -- the budget range must be reconsidered "
    "before proceeding; not silently recomputed."
)
BUDGETS = np.logspace(np.log10(BUDGET_LOW), np.log10(BUDGET_HIGH), 6).tolist()

_budgets_unchanged = all(abs(a - b) < 1e-6 * b for a, b in zip(BUDGETS, BUDGETS_BEFORE_MENU_RESTRICTION))
print(f"C_min: {C_MIN_OLD_MENU:.4e} (old menu incl. WA4, schedule={C_MIN_SCHEDULE_OLD_MENU}) -> "
      f"{C_MIN:.4e} (restricted menu, schedule={C_MIN_SCHEDULE}, {C_MIN / C_ID:.2%} of C_id)")
print(f"Floor binds at 15% of C_id: {_floor_binds_at_15pct}")
print(f"Budgets unchanged: {_budgets_unchanged}")
if not _budgets_unchanged:
    print("!" * 70)
    print("BUDGETS MOVED under the restricted menu -- flagging loudly, not silently proceeding:")
    print(f"  before: {[f'{b:.4e}' for b in BUDGETS_BEFORE_MENU_RESTRICTION]}")
    print(f"  after:  {[f'{b:.4e}' for b in BUDGETS]}")
    print("!" * 70)
else:
    print("Budgets confirmed unchanged from the STOP-1 patch:", [f"{b:.4e}" for b in BUDGETS])

# %% [markdown]
# ## 24. JOINT solver - production lambda sweep over all 85 (feasibility/
# regime-constrained) schedules
#
# Reuses the exact solver core (Section 17). Enumerates all 85 schedules
# x 400 log-spaced lambdas; per (schedule, lambda), bits are chosen
# independently per block.
#
# IMPORTANT: per-budget selection uses the PARETO FRONTIER of the swept
# points (`pareto_frontier`), NOT their convex hull. This was found to
# matter by testing against the real archived tables: for a HARD budget
# constraint ("cheapest D among all points with cost<=B"), restricting to
# the convex hull is the wrong data structure -- it silently discards
# non-hull points that can still be the best FEASIBLE choice under a hard
# constraint (the hull only characterizes what a single shared LAMBDA can
# reach, which is a different query). Concretely: at the lowest production
# budget, hull-restricted selection picked D_pred=0.229, while the actual
# best point already present in the very same swept pool (just not on the
# hull) achieved D_pred=0.150 -- a real, fixable bug caught before it
# could quietly make JOINT look worse than BEST-SEQUENTIAL in the kill
# test for no genuine reason. Gate (e)'s own hull-exactness claim (a
# distinct, still-true property of the swept pool) is unaffected by this
# fix -- see Section 18.

# %%
PRODUCTION_LAMBDA_GRID = np.logspace(-15, -2, 400)

_t0_joint = time.perf_counter()
JOINT_SWEEP_POINTS = joint_lambda_sweep(FULL_SCHEDULES, PRODUCTION_LAMBDA_GRID, list(range(1, 13)), BIT_MENU)
JOINT_FRONTIER = pareto_frontier(JOINT_SWEEP_POINTS)
JOINT_SOLVER_SECONDS = time.perf_counter() - _t0_joint
print(f"Joint sweep: {len(JOINT_SWEEP_POINTS)} raw points -> {len(JOINT_FRONTIER)} frontier points "
      f"({JOINT_SOLVER_SECONDS:.1f}s, no GPU).")


def select_frontier_plan_for_budget(frontier, budget):
    feasible = [(c, d, p) for c, d, p in frontier if c <= budget]
    assert feasible, f"no frontier point fits budget {budget:.4e} (C_min={C_MIN:.4e})"
    return min(feasible, key=lambda t: t[1])


JOINT_PLANS_HULL_REFERENCE = {budget: select_frontier_plan_for_budget(JOINT_FRONTIER, budget) for budget in BUDGETS}
for budget, (c, d, plan) in JOINT_PLANS_HULL_REFERENCE.items():
    print(f"  budget={budget:.4e} ({budget / C_ID:.1%} of C_id): JOINT (hull) cost={c:.4e} D_pred={d:.6f}")

# %% [markdown]
# ### 24b. JOINT selection via the exact per-schedule Pareto DP (STOP-2
# ruling, B2 -- pulled-forward refinement)
#
# Replaces the hull-based selection above as the plan actually used for
# the kill test (plan audit + G1): over all 85 schedules, over each
# schedule's EXACT bits-Pareto set (Section 17b), pick the minimum total
# D_pred (bits damage + that schedule's own kappa-weighted merge damage)
# with cost <= B. The hull-based computation above is RETAINED, unchanged,
# as the reference the cross-check below compares against -- not deleted.

# %%
_t0_exact_joint = time.perf_counter()
JOINT_SCHEDULE_PARETO_SETS = build_schedule_pareto_sets(FULL_SCHEDULES, list(range(1, 13)), BIT_MENU)
JOINT_PLANS = {budget: exact_select_for_budget(JOINT_SCHEDULE_PARETO_SETS, budget) for budget in BUDGETS}
JOINT_EXACT_DP_SECONDS = time.perf_counter() - _t0_exact_joint
_n_pareto_points_total = sum(len(p) for _, _, p in JOINT_SCHEDULE_PARETO_SETS)
print(f"Exact DP: {len(JOINT_SCHEDULE_PARETO_SETS)} schedules, {_n_pareto_points_total} total Pareto "
      f"points ({JOINT_EXACT_DP_SECONDS:.2f}s, no GPU).")

print("\nHull-vs-exact cross-check (JOINT):")
print(f"{'budget':>12}  {'hull D_pred':>12}  {'exact D_pred':>12}  {'improvement':>12}")
JOINT_HULL_VS_EXACT_TABLE = []
for budget in BUDGETS:
    d_hull = JOINT_PLANS_HULL_REFERENCE[budget][1]
    c_exact, d_exact, _ = JOINT_PLANS[budget]
    improvement = d_hull - d_exact
    assert d_exact <= d_hull + 1e-9, (
        f"Exact-DP selection is WORSE than the hull selection at budget={budget:.4e}: "
        f"D_pred_exact={d_exact:.6f} > D_pred_hull={d_hull:.6f} -- the exact DP explores a superset "
        "of what the hull-restricted lambda-sweep reaches, so this should be structurally impossible."
    )
    JOINT_HULL_VS_EXACT_TABLE.append({
        "budget": budget, "D_pred_hull": d_hull, "D_pred_exact": d_exact, "improvement": improvement,
    })
    print(f"{budget:>12.4e}  {d_hull:>12.6f}  {d_exact:>12.6f}  {improvement:>+12.6f}")
    print(f"  budget={budget:.4e} ({budget / C_ID:.1%} of C_id): JOINT (exact) cost={c_exact:.4e} D_pred={d_exact:.6f}")

# %% [markdown]
# ## 25. SEQUENTIAL baselines - both orders x alpha grid (Kim et al. compliance)
#
# S1 (tokens->bits): stage 1 picks the token schedule minimizing predicted
# merge damage subject to token-ONLY cost reduction (at FP16 bits)
# >= alpha*S; stage 2 allocates bits on that FIXED schedule. S2
# (bits->tokens): mirrors this with the roles of the two axes swapped.
# BEST-SEQUENTIAL(B) is the minimum-D_pred plan over BOTH orders x the
# full alpha grid -- this is what "compliance = max over both orderings"
# means operationally: a single-order sequential baseline is a strawman,
# so the STRONGEST sequential candidate (whichever order+alpha performs
# best) is what JOINT must be tested against, not an arbitrarily chosen
# order.
#
# STOP-2 RULING (B3): S1's stage-2 bit allocation (fixed schedule)
# and S2's stage-1 bit allocation (full tokens, schedule=[]) BOTH now use
# `exact_pareto_bits` instead of the lambda-sweep -- fairness requires
# BOTH arms to use the same exactness level, since the whole point of G1
# is to isolate STAGING as the only difference between JOINT and
# SEQUENTIAL. S1's stage-1 (schedule selection via a damage-minimizing
# threshold) and S2's stage-2 (schedule selection via direct enumeration
# over a FIXED bits dict) were already exact -- direct enumeration, no
# lambda sweep involved -- and are unchanged.

# %%
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]


def sequential_s1_tokens_then_bits(budget, alpha, schedules):
    S = C_ID - budget
    target_reduction = alpha * S
    candidates = []
    for s in schedules:
        token_only_cost = plan_cost({"bits": {}, "merges": s})
        reduction = C_ID - token_only_cost
        if reduction >= target_reduction - 1e-9:
            kappa = KAPPA.get(len(s), KAPPA[2])
            merge_damage = kappa * sum(DM_TABLE[(m["block"], m["r"])] for m in s)
            candidates.append((merge_damage, s))
    if not candidates:
        return None
    _, chosen_schedule = min(candidates, key=lambda t: t[0])

    kappa = KAPPA.get(len(chosen_schedule), KAPPA[2])
    merge_damage = kappa * sum(DM_TABLE[(m["block"], m["r"])] for m in chosen_schedule) if chosen_schedule else 0.0
    pareto = exact_pareto_bits(chosen_schedule, list(range(1, 13)), BIT_MENU)
    feasible = [(c, bd + merge_damage, bits) for c, bd, bits in pareto if c <= budget + 1e-6]
    if not feasible:
        return None
    c_best, d_best, bits_best = min(feasible, key=lambda t: t[1])
    plan_best = {"bits": bits_best, "merges": chosen_schedule}
    return {"plan": plan_best, "cost": c_best, "D_pred": d_best, "order": "S1_tokens_then_bits", "alpha": alpha}


def sequential_s2_bits_then_tokens(budget, alpha, schedules):
    S = C_ID - budget
    target_reduction = alpha * S
    pareto_empty = exact_pareto_bits([], list(range(1, 13)), BIT_MENU)  # schedule=[] (no merges), fixed
    candidates = [(c, bd, bits) for c, bd, bits in pareto_empty if (C_ID - c) >= target_reduction - 1e-9]
    if not candidates:
        return None
    _, _, fixed_bits = min(candidates, key=lambda t: t[1])

    stage2_candidates = []
    for s in schedules:
        plan = {"bits": fixed_bits, "merges": s}
        c = plan_cost(plan)
        if c <= budget + 1e-6:
            stage2_candidates.append((objective_d_pred(plan), plan, c))
    if not stage2_candidates:
        return None
    d_best, plan_best, c_best = min(stage2_candidates, key=lambda t: t[0])
    return {"plan": plan_best, "cost": c_best, "D_pred": d_best, "order": "S2_bits_then_tokens", "alpha": alpha}


def best_sequential_for_budget(budget, schedules):
    candidates = []
    for alpha in ALPHA_GRID:
        for fn in (sequential_s1_tokens_then_bits, sequential_s2_bits_then_tokens):
            r = fn(budget, alpha, schedules)
            if r is not None:
                candidates.append(r)
    assert candidates, f"no feasible sequential plan found for budget {budget:.4e} at any order/alpha"
    return min(candidates, key=lambda c: c["D_pred"])


_t0_seq = time.perf_counter()
BEST_SEQUENTIAL_PLANS = {budget: best_sequential_for_budget(budget, FULL_SCHEDULES) for budget in BUDGETS}
SEQUENTIAL_SOLVER_SECONDS = time.perf_counter() - _t0_seq
for budget, result in BEST_SEQUENTIAL_PLANS.items():
    print(f"  budget={budget:.4e}: BEST-SEQUENTIAL cost={result['cost']:.4e} D_pred={result['D_pred']:.6f} "
          f"winner={result['order']} alpha={result['alpha']}")

# %%
# Permanent guard: JOINT (Section 24b) explores the EXACT per-schedule
# Pareto set of all 85 schedules -- a strictly larger candidate set than
# any staged sequential procedure, which is constrained to whichever ONE
# schedule/bits-fixing its own (exact, since B3) stage picked first -- so
# JOINT's predicted D_pred must weakly dominate BEST-SEQUENTIAL's at
# every budget. This is exactly the property the original hull-vs-
# frontier bug broke silently; asserting it here means that class of bug
# (or any future regression in the exact-DP plumbing) can never quietly
# reappear and contaminate the kill test.
for budget in BUDGETS:
    d_joint = JOINT_PLANS[budget][1]
    d_seq = BEST_SEQUENTIAL_PLANS[budget]["D_pred"]
    assert d_joint <= d_seq + 1e-9, (
        f"JOINT does not weakly dominate BEST-SEQUENTIAL at budget={budget:.4e}: "
        f"D_pred_joint={d_joint:.6f} > D_pred_seq={d_seq:.6f}. JOINT searches a strictly larger "
        "space than any staged procedure, so this should be structurally impossible -- likely a "
        "regression in the frontier/budget-selection logic (see Section 24's note)."
    )
print("Consistency guard: JOINT weakly dominates BEST-SEQUENTIAL's predicted D_pred at every budget. OK.")

# %% [markdown]
# ### 25b. Budget-utilization report (STOP-1 ruling, item 2;
# reinterpreted at STOP-2 per B6)
#
# utilization = C(chosen plan) / B, for JOINT and BEST-SEQUENTIAL at each
# of the 6 budgets. Originally intended to measure whether the hull/
# frontier restriction was leaving material budget slack on the table
# (a possible sign of a missed-optimality problem). Now that JOINT and
# BEST-SEQUENTIAL's bit allocation both use the EXACT per-schedule Pareto
# DP (Section 17b/24b/25, gate (e)'s DP-exactness assertion, Section 18)
# rather than the lambda-sweep, utilization below 100% is PROVABLY not a
# missed-optimality artifact -- it is a granularity diagnostic: the
# discrete (schedule, bits) menu simply may not have a combination that
# lands exactly on a given budget. Reported here, at G1 (Section 27), and
# stored per-row in `E5_plans.csv`.

# %%
BUDGET_UTILIZATION = {
    "joint": {budget: JOINT_PLANS[budget][0] / budget for budget in BUDGETS},
    "best_sequential": {budget: BEST_SEQUENTIAL_PLANS[budget]["cost"] / budget for budget in BUDGETS},
}
print(f"{'budget':>12}  {'joint util.':>12}  {'best-seq util.':>15}")
for budget in BUDGETS:
    print(f"{budget:>12.4e}  {BUDGET_UTILIZATION['joint'][budget]:>11.1%}  "
          f"{BUDGET_UTILIZATION['best_sequential'][budget]:>14.1%}")
_min_utilization = min(min(BUDGET_UTILIZATION["joint"].values()), min(BUDGET_UTILIZATION["best_sequential"].values()))
print(f"Minimum utilization across both arms and all budgets: {_min_utilization:.1%}")

# %% [markdown]
# ### 25c. Exactness-refinement trigger - fired pre-emptively (STOP-2
# ruling, B6)
#
# The pre-committed rule (from the STOP-1 ruling) was never allowed to
# wait for G1's own result: the utilization evidence already on hand at
# STOP-1 (89.0-95.2% at 4/6 budgets, under the OLD hull-based solver and
# the OLD WA4-inclusive menu) made firing near-certain, and the STOP-2
# menu restriction (removing WA4) was expected to widen the cost-axis
# gaps further. Rather than spend the expensive 20k-image G1 pass on a
# solver already known to likely need upgrading, this ruling pulled the
# refinement forward: B1-B5 (Sections 17b, 18, 24b, 25) already implement
# and verify the exact per-schedule Pareto DP for BOTH arms before this
# cell runs, so `implemented_now` is `True` here, not a future
# conditional. The G1-dependent fields (`measured_margin_pp` etc.) are
# filled in retrospectively once G1 completes (Section 27b) -- they were
# never the deciding factor.

# %%
UTILIZATION_TRIGGER_THRESHOLD = 0.97
MARGIN_TRIGGER_TOLERANCE_PP = 0.15
G1_ACCEPTANCE_BAR_PP = 0.4
PRE_EMPTIVE_UTILIZATION_EVIDENCE = "89.0-95.2% at 4/6 budgets (old hull-based solver, old WA4-inclusive menu)"

EXACTNESS_REFINEMENT_TRIGGER = {
    "rule": (
        "IF any G1 budget shows utilization < 97% (joint OR best-sequential), OR the final G1 "
        "margin lands within 0.15 points of the 0.4-point bar, THEN before finalizing G1 implement "
        "the exact per-schedule Pareto-set DP (multi-choice knapsack over the 12 blocks, "
        "dominance-pruned) for the joint arm AND for the sequential stages' bit allocation "
        "(fairness requires upgrading both arms together), and re-select plans."
    ),
    "utilization_threshold": UTILIZATION_TRIGGER_THRESHOLD,
    "margin_tolerance_pp": MARGIN_TRIGGER_TOLERANCE_PP,
    "acceptance_bar_pp": G1_ACCEPTANCE_BAR_PP,
    "triggered": True,
    "fired": f"pre-emptively ({PRE_EMPTIVE_UTILIZATION_EVIDENCE}; menu restriction expected to widen gaps)",
    "implemented_now": True,
    "min_utilization_observed_post_upgrade": _min_utilization,
    "note": (
        "Post-upgrade utilization (Section 25b) is retained as a granularity diagnostic, not an "
        "optimality concern -- gate (e)'s DP-exactness assertion (Section 18) proves the exact DP "
        "reaches the true per-schedule optimum, including duality-gap points the lambda-sweep "
        "could not reach. G1-dependent fields added retrospectively in Section 27b."
    ),
}
print(f"Exactness-refinement trigger: FIRED PRE-EMPTIVELY -- {EXACTNESS_REFINEMENT_TRIGGER['fired']}")
print(f"implemented_now={EXACTNESS_REFINEMENT_TRIGGER['implemented_now']}  "
      f"post-upgrade min_utilization={_min_utilization:.1%}")

# %% [markdown]
# ## 25d. STOP-2 RE-VERDICT - mandatory review checkpoint before
# the kill test
#
# Interactive mode (`TANDEM_INTERACTIVE=1`): this cell ALWAYS halts
# (raises), regardless of outcomes, once the re-verdict block below has
# been printed -- Family-X's re-verdict, the budget confirmation, gate
# (e)'s triple status, and the hull-vs-exact improvement table are ALL now
# settled; only the kill test (plan audit + G1, the expensive 20k-image
# pass) remains. Re-run only the cells AFTER this one, and only once the
# reviewer has given GO. Non-interactive (default, reviewer release): the
# identical re-verdict block still prints, but execution continues
# automatically past a "[checkpoint: ...]" line instead of raising.

# %%
_gate_e = GATE_RESULTS["e_solver_soundness_and_hull_completeness"]
print(f"{'=' * 70}")
print("STOP-2 RE-VERDICT -- paste this for review before launching the kill test")
print(f"{'=' * 70}")
print(f"V1_E5 (10 menu-consistent configs): slope={V1_E5['slope']:.4f} R²={V1_E5['r2']:.4f} "
      f"(n={V1_E5['n_points']}) -> {'PASS' if V1_E5['passed'] else 'FAIL'}")
print(f"QUOTA correlation (menu-consistent subset): {QUOTA_CORRELATION:.3f}")
print(f"\nBudgets: C_min {C_MIN_OLD_MENU:.4e} -> {C_MIN:.4e} ({C_MIN / C_ID:.2%} of C_id); "
      f"floor binds at 15%: {_floor_binds_at_15pct}; budgets unchanged: {_budgets_unchanged}")
print(f"\nGate (e) triple status: soundness={_gate_e['soundness_passed']}  "
      f"hull_completeness={_gate_e['hull_completeness_passed']}  "
      f"dp_exactness={_gate_e['dp_exactness_passed']} ({_gate_e['n_test_budgets_checked']} test budgets, "
      f"including {_gate_e['n_duality_gap_points']} duality-gap points)")
print(f"\nHull-vs-exact improvement (JOINT), per budget:")
print(f"{'budget':>12}  {'hull D_pred':>12}  {'exact D_pred':>12}  {'improvement':>12}")
for row in JOINT_HULL_VS_EXACT_TABLE:
    print(f"{row['budget']:>12.4e}  {row['D_pred_hull']:>12.6f}  {row['D_pred_exact']:>12.6f}  "
          f"{row['improvement']:>+12.6f}")
print(f"\nExactness-refinement trigger: FIRED PRE-EMPTIVELY, implemented_now="
      f"{EXACTNESS_REFINEMENT_TRIGGER['implemented_now']}")
print(f"{'=' * 70}")

if INTERACTIVE:
    raise RuntimeError(
        "STOP-2 RE-VERDICT: mandatory pause. Paste the block above for review and GO before "
        "launching the kill test (plan audit + 20k-image G1). Re-run only the cells AFTER this one "
        "once cleared -- this halt is unconditional and does not indicate a failure."
    )
else:
    print("[checkpoint: non-interactive mode -- continuing]")

# %% [markdown]
# ## 26. Plan audit - measure JOINT and BEST-SEQUENTIAL on the 5,000-image set
#
# Checkpointed like E3/E4's sweeps: `E5_plans.csv` is keyed by
# (method, budget) and resumable. Set `FORCE_FRESH_RUN = True` (Section 2)
# to ignore it and recompute from scratch.

# %%
def plan_config_hash(method, budget, plan):
    payload = json.dumps({"method": method, "budget": budget, "plan": plan}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_e5_plans_checkpoint(csv_path):
    if FORCE_FRESH_RUN:
        print("FORCE_FRESH_RUN=True -- ignoring any existing checkpoint, recomputing all rows.")
        return [], set()
    if not csv_path.exists():
        return [], set()
    existing = pd.read_csv(csv_path)
    return existing.to_dict("records"), set(existing["row_key"])


def run_plan_audit(model, images, fp_logits, fp_argmax, labels, amax_lookup, batch_size,
                    joint_plans, best_sequential_plans, budgets):
    csv_path = RESULTS_DIR / "E5_plans.csv"
    rows, done = _load_e5_plans_checkpoint(csv_path)
    candidates = []
    for budget in budgets:
        c, d_pred, plan = joint_plans[budget]
        candidates.append(("joint", budget, plan, d_pred, c))
        seq = best_sequential_plans[budget]
        candidates.append(("best_sequential", budget, seq["plan"], seq["D_pred"], seq["cost"]))

    for method, budget, plan, d_pred, c in tqdm(candidates, desc="plan audit (5k)"):
        row_key = f"{method}_{budget:.6e}"
        if row_key in done:
            continue
        measured = run_plan(model, images, fp_logits, fp_argmax, labels, plan, amax_lookup, batch_size)
        rows.append({
            "row_key": row_key, "method": method, "budget": budget, "plan_json": json.dumps(plan),
            "D_pred": d_pred, "C": c, "utilization": c / budget,
            "D_mse_5k": measured["D_mse"], "top1_5k": measured["top1_pert"],
            "config_hash": plan_config_hash(method, budget, plan),
        })
        done.add(row_key)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pd.DataFrame(rows)


PLAN_AUDIT_DF = run_plan_audit(
    model, images_u8, fp_logits, fp_argmax, labels, AMAX_LOOKUP, CONFIG["BATCH_SIZE"],
    JOINT_PLANS, BEST_SEQUENTIAL_PLANS, BUDGETS,
)
print(PLAN_AUDIT_DF[["method", "budget", "D_pred", "D_mse_5k", "top1_5k"]].to_string(index=False))

PLAN_AUDIT_FIT = fit_through_origin(PLAN_AUDIT_DF["D_pred"], PLAN_AUDIT_DF["D_mse_5k"])
print(f"\nPlan-audit predicted-vs-measured fidelity: slope={PLAN_AUDIT_FIT['slope']:.3f} "
      f"R²={PLAN_AUDIT_FIT['r2']:.3f} (n={PLAN_AUDIT_FIT['n_points']})")

# %% [markdown]
# ### 26a. Plan-collapse & token-frontier check (STOP-3 ruling,
# artifact patch)
#
# This is pure schedule/table arithmetic on JOINT_PLANS/BEST_SEQUENTIAL_PLANS
# -- no G1 data needed, so it is computed and printed BEFORE G1 runs.
# `PLANS_IDENTICAL`: whether JOINT and BEST-SEQUENTIAL picked the same
# plan at each budget. `TOKEN_FRONTIER_CHECK`: the pre-registered necessary
# CONDITION for that collapse to be possible at all -- D_pred is additively
# separable in (schedule, bits) by construction (DQ_TABLE doesn't depend
# on token count, DM_TABLE doesn't depend on bits), but the BUDGET
# constraint is coupled through N_l (a merge changes the token count, and
# hence the BOPs cost, of every block after it). A staged search (S1's
# stage 1) can only ever select a schedule that is Pareto-efficient on the
# token axis ALONE (FP16-costed savings vs kappa-weighted merge damage,
# ignoring bits) -- if the true joint-optimal schedule is NOT on that
# frontier, no staged threshold search could reach it even in principle.
# Computed for all 85 schedules and checked against each budget's actual
# joint-selected schedule.

# %%
PLANS_IDENTICAL = {budget: JOINT_PLANS[budget][2] == BEST_SEQUENTIAL_PLANS[budget]["plan"] for budget in BUDGETS}
for budget in BUDGETS:
    print(f"  budget={budget:.4e}: plans_identical={PLANS_IDENTICAL[budget]}")
print(f"Plans identical at {sum(PLANS_IDENTICAL.values())}/{len(BUDGETS)} budgets.")


def compute_token_frontier(schedules, c_id):
    points = []
    for s in schedules:
        kappa = KAPPA.get(len(s), KAPPA[2])
        damage = kappa * sum(DM_TABLE[(m["block"], m["r"])] for m in s) if s else 0.0
        token_only_cost = plan_cost({"bits": {}, "merges": s})
        savings = c_id - token_only_cost
        points.append((token_only_cost, damage, {"schedule": s, "savings": savings, "damage": damage}))
    frontier_points = pareto_frontier(points)
    frontier_schedules = [p[2]["schedule"] for p in frontier_points]
    return points, frontier_schedules


TOKEN_FRONTIER_POINTS, TOKEN_FRONTIER_SCHEDULES = compute_token_frontier(FULL_SCHEDULES, C_ID)

TOKEN_FRONTIER_TABLE = pd.DataFrame([
    {"schedule": json.dumps(payload["schedule"]), "fp16_costed_savings": payload["savings"],
     "merge_damage_kappa": payload["damage"], "on_token_frontier": payload["schedule"] in TOKEN_FRONTIER_SCHEDULES}
    for _, _, payload in TOKEN_FRONTIER_POINTS
])
TOKEN_FRONTIER_TABLE.to_csv(RESULTS_DIR / "E5_schedule_token_frontier.csv", index=False)
print(f"\nToken frontier: {len(TOKEN_FRONTIER_SCHEDULES)}/{len(FULL_SCHEDULES)} of the 85 schedules are "
      f"Pareto-efficient on the token axis alone (saved to E5_schedule_token_frontier.csv).")

TOKEN_FRONTIER_CHECK = {}
for budget in BUDGETS:
    joint_schedule = JOINT_PLANS[budget][2]["merges"]
    on_frontier = joint_schedule in TOKEN_FRONTIER_SCHEDULES
    TOKEN_FRONTIER_CHECK[budget] = on_frontier
    print(f"  budget={budget:.4e}: joint schedule={joint_schedule} on_token_frontier={on_frontier}")
assert all(TOKEN_FRONTIER_CHECK.values()), (
    "Staging-collapse condition FAILS at at least one budget -- the joint-optimal schedule is not "
    "token-frontier-efficient there, so BEST-SEQUENTIAL's exact tie with JOINT cannot be explained "
    "by the staging-collapse mechanism at that budget; re-examine before repeating the F-collapse claim."
)
print(f"Token-frontier condition holds at all {len(BUDGETS)}/{len(BUDGETS)} budgets: matches the "
      "independent verification.")

STAGING_COLLAPSE_FINDING = {
    "id": "F-collapse",
    "title": "Staging collapse: JOINT ties (does not beat) BEST-SEQUENTIAL because the joint schedule "
             "is token-frontier-efficient",
    "statement": (
        "D_pred is separable in (schedule, bits) by construction (no cross term: DQ_TABLE entries "
        "don't depend on token count, DM_TABLE entries don't depend on bits), but the budget "
        "CONSTRAINT is coupled via N_l (a merge changes the token count, hence the BOPs cost, of "
        "every later block's bit choice). Conditional result: whenever the joint-optimal schedule "
        "is Pareto-efficient on the token axis ALONE (ignoring bits), a sufficiently strong staged "
        "search (both compression orders, a dense alpha grid, exact per-stage sub-solves) recovers "
        "it, because its stage-1 schedule selection is itself a token-axis Pareto search. Verified "
        "to hold at all 6 production budgets -- JOINT and BEST-SEQUENTIAL select bit-identical "
        "plans at every budget, which is why G1's measured delta_top1 is exactly 0.000pp with a "
        "zero-width CI: the necessary consequence of plan collapse, not a statistical artifact. "
        "Corollary: any joint>staged margin under the REAL (measured-cost) objective, rather than "
        "the predicted D_pred proxy, would have to come from cost-structure NON-additivity this "
        "predicted-cost model does not capture -- i.e. a case where the truly optimal schedule is "
        "NOT token-frontier-efficient, so no staged threshold search could reach it even in "
        "principle."
    ),
    "plans_identical": {f"{b:.4e}": bool(v) for b, v in PLANS_IDENTICAL.items()},
    "token_frontier_check": {f"{b:.4e}": bool(v) for b, v in TOKEN_FRONTIER_CHECK.items()},
}
print(f"\n{STAGING_COLLAPSE_FINDING['id']}: {STAGING_COLLAPSE_FINDING['title']}")

# %% [markdown]
# ## 27. G1 FINAL - 20,000-image paired evaluation + bootstrap (V2_E5)
#
# Requires real labels (ImageNet-1k via HF_TOKEN) -- G1 is defined on
# paired TOP-1 ACCURACY, which `load_eval_images` above already requires
# (no fallback exists to silently produce a differently-labeled result).

# %%
assert LABELS_AVAILABLE, (
    "G1 requires labeled ImageNet-1k data (HF_TOKEN + access) -- halting rather than silently "
    "producing a meaningless G1 result."
)

images_g1, labels_g1, labels_available_g1, data_source_g1, n_used_g1 = load_eval_images(
    CONFIG, HF_TOKEN, n_requested=CONFIG["N_IMAGES_G1"]
)
assert labels_available_g1 and data_source_g1 == "imagenet-1k", (
    f"G1's 20k pull did not return labeled ImageNet-1k data (got data_source={data_source_g1!r}, "
    f"labels_available={labels_available_g1!r})."
)
assert images_g1.shape[0] == CONFIG["N_IMAGES_G1"]
print(f"G1 image set: {images_g1.shape[0]} images, deterministic streaming order, labels available.")

# %%
def _load_g1_checkpoint(path):
    if FORCE_FRESH_RUN:
        print("FORCE_FRESH_RUN=True -- ignoring any existing checkpoint, recomputing all rows.")
        return {}
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def run_g1_correctness(model, images_g1, labels_g1, amax_lookup, batch_size, joint_plans,
                        best_sequential_plans, budgets, checkpoint_path):
    """Returns {budget: {"joint": bool_tensor[N], "best_sequential": bool_tensor[N]}}.
    No fp_logits/fp_argmax needed here -- G1 only needs per-image top-1
    correctness against the TRUE labels, not logit-MSE vs the FP reference."""
    ckpt = _load_g1_checkpoint(checkpoint_path)
    # Shape must match model.num_classes, not just be "some placeholder" --
    # _batch_metrics (inside run_plan) still unconditionally computes
    # MSE/KL against this dummy even though G1 only reads per_image_correct
    # (see docstring above) and discards the rest; a [N,1] placeholder
    # broadcasts against real [N,1000] logits without erroring, which only
    # produces a UserWarning and a meaningless-but-discarded MSE, not wrong
    # G1 results -- but it's needless noise and needless broadcast compute.
    fp_logits_dummy = torch.zeros(images_g1.shape[0], model.num_classes)  # unused (top1-only)
    fp_argmax_dummy = torch.zeros(images_g1.shape[0], dtype=torch.long)
    results = {}
    for budget in tqdm(budgets, desc="G1 (20k, paired)"):
        results[budget] = {}
        for method, (plan_source, key) in (
            ("joint", (joint_plans[budget][2], "joint")),
            ("best_sequential", (best_sequential_plans[budget]["plan"], "best_sequential")),
        ):
            ckpt_key = f"{budget:.6e}_{method}"
            if ckpt_key in ckpt:
                correct = torch.tensor(ckpt[ckpt_key], dtype=torch.bool)
            else:
                out = run_plan(model, images_g1, fp_logits_dummy, fp_argmax_dummy, labels_g1, plan_source,
                                amax_lookup, batch_size, track_per_image_correct=True)
                correct = out["per_image_correct"]
                ckpt[ckpt_key] = correct.tolist()
                with open(checkpoint_path, "w") as f:
                    json.dump(ckpt, f)
            results[budget][method] = correct
    return results


G1_CHECKPOINT_PATH = RESULTS_DIR / "E5_g1_correctness_checkpoint.json"
G1_CORRECTNESS = run_g1_correctness(
    model, images_g1, labels_g1, AMAX_LOOKUP, CONFIG["BATCH_SIZE"],
    JOINT_PLANS, BEST_SEQUENTIAL_PLANS, BUDGETS, G1_CHECKPOINT_PATH,
)

# %%
def bootstrap_g1(correctness_by_budget, budgets, n_resamples=10000, seed=0, chunk=500):
    """Paired bootstrap over IMAGES (same resample indices shared across
    budgets within each iteration, since all budgets use the same 20k
    image set): per-budget CIs, plus the CI of the MEAN delta across
    budgets (acceptance is on the mean, not any single budget)."""
    n = next(iter(correctness_by_budget.values()))["joint"].shape[0]
    rng = np.random.RandomState(seed)
    per_budget_resampled = {budget: np.empty(n_resamples) for budget in budgets}

    arrays = {
        budget: (
            correctness_by_budget[budget]["joint"].numpy().astype(np.float64),
            correctness_by_budget[budget]["best_sequential"].numpy().astype(np.float64),
        )
        for budget in budgets
    }
    for start in range(0, n_resamples, chunk):
        this_chunk = min(chunk, n_resamples - start)
        idx = rng.randint(0, n, size=(this_chunk, n))
        for budget in budgets:
            a, b = arrays[budget]
            per_budget_resampled[budget][start : start + this_chunk] = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    per_budget_result = {}
    for budget in budgets:
        a, b = arrays[budget]
        point = float(a.mean() - b.mean())
        ci_low, ci_high = np.percentile(per_budget_resampled[budget], [2.5, 97.5])
        per_budget_result[budget] = {
            "delta_top1_pp": point * 100, "ci_low_pp": float(ci_low) * 100, "ci_high_pp": float(ci_high) * 100,
        }

    mean_resampled = np.mean([per_budget_resampled[b] for b in budgets], axis=0)
    mean_point_pp = float(np.mean([per_budget_result[b]["delta_top1_pp"] for b in budgets]))
    mean_ci_low_pp, mean_ci_high_pp = (np.percentile(mean_resampled, [2.5, 97.5]) * 100).tolist()
    return per_budget_result, mean_point_pp, mean_ci_low_pp, mean_ci_high_pp


G1_PER_BUDGET, G1_MEAN_DELTA_PP, G1_MEAN_CI_LOW_PP, G1_MEAN_CI_HIGH_PP = bootstrap_g1(G1_CORRECTNESS, BUDGETS)

V2_E5 = {
    "mean_delta_top1_pp": G1_MEAN_DELTA_PP,
    "mean_ci_low_pp": G1_MEAN_CI_LOW_PP,
    "mean_ci_high_pp": G1_MEAN_CI_HIGH_PP,
    "per_budget": {f"{b:.4e}": v for b, v in G1_PER_BUDGET.items()},
    "passed": bool(G1_MEAN_DELTA_PP >= 0.4 and G1_MEAN_CI_LOW_PP > 0.0),
}
# STOP-3 ruling (artifact patch): label the failure mechanism.
# `PLANS_IDENTICAL` (Section 26a) was computed BEFORE G1 ran, from the
# solvers' own selected plans alone -- an all-True result there predicts,
# and here is confirmed by, an exact G1 tie (not a loss).
_all_plans_identical = all(PLANS_IDENTICAL.values())
V2_E5["mechanism"] = (
    "exact_tie_via_plan_collapse" if (not V2_E5["passed"] and _all_plans_identical and G1_MEAN_DELTA_PP == 0.0)
    else ("loss" if not V2_E5["passed"] else "win")
)
_v2_verdict_label = {
    "win": "PASS",
    "loss": "FAIL",
    "exact_tie_via_plan_collapse": "FAIL-with-mechanism (exact tie via plan collapse, not a loss -- see F-collapse)",
}[V2_E5["mechanism"]]
print(f"\nV2_E5 / G1: mean delta_top1={G1_MEAN_DELTA_PP:+.3f}pp "
      f"(95% CI [{G1_MEAN_CI_LOW_PP:+.3f}, {G1_MEAN_CI_HIGH_PP:+.3f}]pp) "
      f"-> {_v2_verdict_label}")
for budget, v in G1_PER_BUDGET.items():
    print(f"  budget={budget:.4e}: delta_top1={v['delta_top1_pp']:+.3f}pp "
          f"(95% CI [{v['ci_low_pp']:+.3f}, {v['ci_high_pp']:+.3f}]pp) "
          f"[utilization: joint={BUDGET_UTILIZATION['joint'][budget]:.1%}, "
          f"best-seq={BUDGET_UTILIZATION['best_sequential'][budget]:.1%}]")

# %% [markdown]
# ### 27b. Exactness-refinement trigger - G1-dependent fields (retrospective)
#
# The trigger already FIRED PRE-EMPTIVELY and was already IMPLEMENTED
# before the kill test ran (Section 25c, before STOP-2's re-verdict halt)
# -- that decision did not wait for G1. These fields are added
# retrospectively, purely for the record: did the ACTUAL G1 margin also,
# independently, satisfy the original margin-proximity criterion.

# %%
_margin_close_to_bar = abs(G1_MEAN_DELTA_PP - G1_ACCEPTANCE_BAR_PP) <= MARGIN_TRIGGER_TOLERANCE_PP
EXACTNESS_REFINEMENT_TRIGGER["measured_margin_pp"] = G1_MEAN_DELTA_PP
EXACTNESS_REFINEMENT_TRIGGER["margin_within_tolerance_of_bar"] = _margin_close_to_bar

print(f"Exactness-refinement trigger (recap): fired={EXACTNESS_REFINEMENT_TRIGGER['fired']}, "
      f"implemented_now={EXACTNESS_REFINEMENT_TRIGGER['implemented_now']}")
print(f"Retrospective: measured G1 margin={G1_MEAN_DELTA_PP:+.3f}pp vs bar={G1_ACCEPTANCE_BAR_PP}pp "
      f"(within {MARGIN_TRIGGER_TOLERANCE_PP}pp tolerance: {_margin_close_to_bar})")

# %% [markdown]
# ## 28. Naive references - uniform-bits-only and merge-only frontiers
#
# Context lines for the Pareto plot only, NOT part of G1's acceptance.
# Measured (not just predicted) on the 5,000-image set for a fair visual
# comparison against JOINT/BEST-SEQUENTIAL's own measured points.

# %%
def uniform_bits_plans():
    return [{"bits": {l: b for l in range(1, 13)}, "merges": []} for b in BIT_MENU]


def merge_only_frontier_plans():
    points = [
        (plan_cost({"bits": {}, "merges": s}), objective_d_pred({"bits": {}, "merges": s}), {"bits": {}, "merges": s})
        for s in FULL_SCHEDULES
    ]
    frontier = pareto_frontier(points)
    return [p for _, _, p in frontier]


NAIVE_UNIFORM_BITS_PLANS = uniform_bits_plans()
NAIVE_MERGE_ONLY_PLANS = merge_only_frontier_plans()


def measure_naive_references(model, images, fp_logits, fp_argmax, labels, amax_lookup, batch_size):
    rows = []
    for kind, plans in (("uniform_bits", NAIVE_UNIFORM_BITS_PLANS), ("merge_only", NAIVE_MERGE_ONLY_PLANS)):
        for plan in tqdm(plans, desc=f"naive reference ({kind})"):
            measured = run_plan(model, images, fp_logits, fp_argmax, labels, plan, amax_lookup, batch_size)
            rows.append({
                "kind": kind, "plan_json": json.dumps(plan), "C": plan_cost(plan),
                "D_pred": objective_d_pred(plan), "D_mse_5k": measured["D_mse"], "top1_5k": measured["top1_pert"],
            })
    return pd.DataFrame(rows)


NAIVE_REFS_DF = measure_naive_references(model, images_u8, fp_logits, fp_argmax, labels, AMAX_LOOKUP, CONFIG["BATCH_SIZE"])

# %% [markdown]
# ## 29. Artifacts - plots, plan-composition table, E5_summary.json

# %%
import matplotlib.pyplot as plt


def plot_pareto_top1_vs_bops(plan_audit_df, naive_refs_df, budgets, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, label, color in (("joint", "JOINT", "#4C72B0"), ("best_sequential", "BEST-SEQUENTIAL", "#DD8452")):
        sub = plan_audit_df[plan_audit_df["method"] == method].sort_values("C")
        ax.plot(sub["C"], sub["top1_5k"], "o-", label=label, color=color)
    for kind, label, color in (("uniform_bits", "uniform-bits-only (context)", "grey"),
                                ("merge_only", "merge-only (context)", "lightgrey")):
        sub = naive_refs_df[naive_refs_df["kind"] == kind].sort_values("C")
        ax.plot(sub["C"], sub["top1_5k"], "x--", label=label, color=color, alpha=0.7)
    for budget in budgets:
        ax.axvline(budget, color="k", linestyle=":", alpha=0.2)
    ax.set_xlabel("Cost C(plan) [simulated BOPs]")
    ax.set_ylabel("Measured top-1 accuracy (5,000-image set)")
    ax.set_title("Pareto: measured top-1 vs simulated BOPs cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_cross_axis_additivity(family_x_df, fit_result, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(family_x_df["D_pred"], family_x_df["D_mse"], color="#4C72B0", s=60)
    for _, row in family_x_df.iterrows():
        ax.annotate(row["config_id"], (row["D_pred"], row["D_mse"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    lims = [0, family_x_df[["D_pred", "D_mse"]].max().max() * 1.15]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    ax.set_xlabel("D_pred (objective: Sum Dq + kappa*Sum Dm)")
    ax.set_ylabel("D_mse (measured, composed PlanExecutor)")
    ax.set_title(f"Family X cross-axis additivity: slope={fit_result['slope']:.3f} R²={fit_result['r2']:.3f} "
                 f"(n={fit_result['n_points']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


def plot_pred_vs_measured_plans(family_x_df, plan_audit_df, fit_result, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(family_x_df["D_pred"], family_x_df["D_mse"], color="#4C72B0", alpha=0.7, label="Family X", s=50)
    ax.scatter(plan_audit_df["D_pred"], plan_audit_df["D_mse_5k"], color="#DD8452", alpha=0.7,
               label="plan audit (joint/best-seq)", s=50)
    lims = [1e-6, max(family_x_df["D_mse"].max(), plan_audit_df["D_mse_5k"].max()) * 1.2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D_pred")
    ax.set_ylabel("D_measured (MSE)")
    ax.set_title(f"All executed plans: predicted vs measured (slope={fit_result['slope']:.3f} "
                 f"R²={fit_result['r2']:.3f}, n={fit_result['n_points']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()


plot_pareto_top1_vs_bops(PLAN_AUDIT_DF, NAIVE_REFS_DF, BUDGETS, RESULTS_DIR / "pareto_top1_vs_bops.png")
plot_cross_axis_additivity(FAMILY_X_DF, V1_E5, RESULTS_DIR / "cross_axis_additivity.png")

# STOP-3 ruling (artifact patch): the pooled fit was polluted by
# the 2 retained WA4 evidence rows (X8/X9, excluded_out_of_menu=True) --
# those are known out-of-regime/superlinear (F-X1) and were kept in
# E5_familyX.csv only as evidence, never meant to be fit against. Both
# fits are recorded; the polluted one is retained for the record but the
# corrected (menu-consistent) fit is what's reported/plotted.
_combined_pred_measured_polluted = pd.concat([
    FAMILY_X_DF[["D_pred", "D_mse"]],
    PLAN_AUDIT_DF[["D_pred"]].assign(D_mse=PLAN_AUDIT_DF["D_mse_5k"]),
], ignore_index=True)
PRED_VS_MEASURED_FIT_POLLUTED = fit_through_origin(
    _combined_pred_measured_polluted["D_pred"], _combined_pred_measured_polluted["D_mse"]
)

_combined_pred_measured = pd.concat([
    FAMILY_X_DF[~FAMILY_X_DF["excluded_out_of_menu"]][["D_pred", "D_mse"]],
    PLAN_AUDIT_DF[["D_pred"]].assign(D_mse=PLAN_AUDIT_DF["D_mse_5k"]),
], ignore_index=True)
PRED_VS_MEASURED_FIT = fit_through_origin(_combined_pred_measured["D_pred"], _combined_pred_measured["D_mse"])
assert PRED_VS_MEASURED_FIT["n_points"] == 22, (
    f"expected 22 menu-consistent pooled points (12 Family X - 2 WA4 + 12 plan-audit), got "
    f"{PRED_VS_MEASURED_FIT['n_points']}"
)
print(f"Pooled predicted-vs-measured fidelity: polluted (n={PRED_VS_MEASURED_FIT_POLLUTED['n_points']}, "
      f"includes WA4 evidence) slope={PRED_VS_MEASURED_FIT_POLLUTED['slope']:.3f} "
      f"R²={PRED_VS_MEASURED_FIT_POLLUTED['r2']:.3f}  |  corrected (n={PRED_VS_MEASURED_FIT['n_points']}, "
      f"menu-consistent) slope={PRED_VS_MEASURED_FIT['slope']:.3f} R²={PRED_VS_MEASURED_FIT['r2']:.3f}")
plot_pred_vs_measured_plans(FAMILY_X_DF, PLAN_AUDIT_DF, PRED_VS_MEASURED_FIT, RESULTS_DIR / "pred_vs_measured_plans.png")

# %%
def render_plan_composition_table(joint_plans, best_sequential_plans, budgets):
    lines = ["Plan composition per budget -- JOINT vs BEST-SEQUENTIAL", "=" * 100]
    for budget in budgets:
        c_j, d_j, plan_j = joint_plans[budget]
        seq = best_sequential_plans[budget]
        lines.append(f"\nBudget = {budget:.4e} ({budget / C_ID:.1%} of C_id)")
        lines.append(f"  JOINT           : cost={c_j:.4e}  D_pred={d_j:.6f}")
        lines.append(f"    bits   : {json.dumps(plan_j['bits'])}")
        lines.append(f"    merges : {json.dumps(plan_j['merges'])}")
        lines.append(f"  BEST-SEQUENTIAL : cost={seq['cost']:.4e}  D_pred={seq['D_pred']:.6f}  "
                     f"(winner: {seq['order']}, alpha={seq['alpha']})")
        lines.append(f"    bits   : {json.dumps(seq['plan']['bits'])}")
        lines.append(f"    merges : {json.dumps(seq['plan']['merges'])}")
    return "\n".join(lines)


PLAN_COMPOSITION_TABLE = render_plan_composition_table(JOINT_PLANS, BEST_SEQUENTIAL_PLANS, BUDGETS)
print(PLAN_COMPOSITION_TABLE)
with open(RESULTS_DIR / "plan_composition_table.txt", "w") as f:
    f.write(PLAN_COMPOSITION_TABLE)

# %%
def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


OVERALL_PASS_E5 = bool(V1_E5["passed"] and V2_E5["passed"])

E5_SUMMARY = {
    "config": CONFIG,
    "environment": ENVIRONMENT,
    "data_source": DATA_SOURCE,
    "data_source_g1": data_source_g1,
    "n_images_used": N_USED,
    "n_images_g1": n_used_g1,
    "gate_results": GATE_RESULTS,
    "c_id": C_ID,
    "c_min": C_MIN,
    "budgets": BUDGETS,
    "verdicts": {
        "V1_E5_cross_axis_additivity": V1_E5,
        "V2_E5_G1_kill_test": V2_E5,
    },
    "sequential_winners": {
        f"{b:.4e}": {"order": r["order"], "alpha": r["alpha"]} for b, r in BEST_SEQUENTIAL_PLANS.items()
    },
    "budget_utilization": {
        "joint": {f"{b:.4e}": u for b, u in BUDGET_UTILIZATION["joint"].items()},
        "best_sequential": {f"{b:.4e}": u for b, u in BUDGET_UTILIZATION["best_sequential"].items()},
    },
    "exactness_refinement_trigger": EXACTNESS_REFINEMENT_TRIGGER,
    "plan_audit_fidelity": PLAN_AUDIT_FIT,
    "pred_vs_measured_fidelity_all_plans": PRED_VS_MEASURED_FIT,
    "pred_vs_measured_fidelity_all_plans_note": (
        "Corrected: excludes the 2 retained WA4 evidence rows (X8/X9, excluded_out_of_menu=True; "
        "see F-X1) from the pooled fit. The uncorrected/polluted fit (includes those rows) is "
        "recorded separately below for the record and is never the reported fidelity number."
    ),
    "pred_vs_measured_fidelity_all_plans_polluted": PRED_VS_MEASURED_FIT_POLLUTED,
    "quota_correlation_informative": QUOTA_CORRELATION,
    "plans_identical": {f"{b:.4e}": bool(v) for b, v in PLANS_IDENTICAL.items()},
    "token_frontier_check": {f"{b:.4e}": bool(v) for b, v in TOKEN_FRONTIER_CHECK.items()},
    "findings": {
        "F_X1_wa4_menu_enforcement": FINDING_F_X1,
        "F_X2_structured_subadditivity": FINDING_F_X2,
        "F_collapse_staging_collapse": STAGING_COLLAPSE_FINDING,
    },
    "design_decision_amendments": {"D4_amendment": D4_AMENDMENT},
    "spec_error_note": SPEC_ERROR_NOTE,
    "wa4_exclusion_justification": WA4_EXCLUSION_JUSTIFICATION,
    "overall_pass": OVERALL_PASS_E5,
    "wall_clock_seconds": {
        "joint_solver": JOINT_SOLVER_SECONDS, "sequential_solver": SEQUENTIAL_SOLVER_SECONDS,
    },
}
with open(RESULTS_DIR / "E5_summary.json", "w") as f:
    json.dump(E5_SUMMARY, f, indent=2, default=_json_default)
print(f"Saved {RESULTS_DIR / 'E5_summary.json'}")

# %% [markdown]
# ## 30. Final verdict + LAB_LOG.md sentences

# %%
print(f"{'=' * 70}")
print(f"V1_E5 (cross-axis additivity, REQUIRED): {'PASS' if V1_E5['passed'] else 'FAIL'} "
      f"(slope={V1_E5['slope']:.3f}, R²={V1_E5['r2']:.3f})")
print(f"V2_E5 / G1 (joint-vs-sequential kill test, REQUIRED): {_v2_verdict_label} "
      f"(mean delta_top1={G1_MEAN_DELTA_PP:+.3f}pp, CI [{G1_MEAN_CI_LOW_PP:+.3f}, {G1_MEAN_CI_HIGH_PP:+.3f}]pp)")
print(f"OVERALL: {'PASS' if OVERALL_PASS_E5 else 'FAIL'}")
print(f"{'=' * 70}")

_v2_verdict_phrase = (
    "passes" if V2_E5["passed"]
    else "fails with an exact tie via plan collapse (not a loss)" if V2_E5["mechanism"] == "exact_tie_via_plan_collapse"
    else "fails"
)
e5_sentence_1 = (
    f"E5 overall {'PASSES' if OVERALL_PASS_E5 else 'FAILS'}: V1_E5 (cross-axis quant+merge additivity, "
    f"Family X) {'passes' if V1_E5['passed'] else 'fails'} with slope={V1_E5['slope']:.2f}, "
    f"R²={V1_E5['r2']:.3f} (n={V1_E5['n_points']}); V2_E5/G1 (joint-vs-sequential kill test on 20k "
    f"paired images) {_v2_verdict_phrase} with mean delta_top1="
    f"{G1_MEAN_DELTA_PP:+.3f}pp (95% CI [{G1_MEAN_CI_LOW_PP:+.3f}, {G1_MEAN_CI_HIGH_PP:+.3f}]pp) "
    "across 6 budgets."
)
_seq_winner_summary = {f"{b:.2e}": f"{r['order']}@{r['alpha']}" for b, r in BEST_SEQUENTIAL_PLANS.items()}
e5_sentence_2 = (
    f"BEST-SEQUENTIAL's winning order/alpha varied by budget ({_seq_winner_summary}), "
    "answering Kim et al.'s compression-order question on TANDEM's own axes rather than assuming one."
)
e5_sentence_3 = (
    f"Plan-audit predicted-vs-measured fidelity across all executed plans (menu-consistent, WA4 "
    f"evidence excluded): slope={PRED_VS_MEASURED_FIT['slope']:.2f}, R²={PRED_VS_MEASURED_FIT['r2']:.3f} "
    f"(n={PRED_VS_MEASURED_FIT['n_points']}; polluted fit incl. WA4 evidence: "
    f"slope={PRED_VS_MEASURED_FIT_POLLUTED['slope']:.2f}, R²={PRED_VS_MEASURED_FIT_POLLUTED['r2']:.3f}, "
    f"n={PRED_VS_MEASURED_FIT_POLLUTED['n_points']}); QUOTA calibration-shift correlation (informative): "
    f"{QUOTA_CORRELATION:.2f}."
)
e5_sentence_4 = (
    f"G1's exact tie is a plan-collapse artifact, not a search failure: JOINT and BEST-SEQUENTIAL "
    f"selected bit-identical plans at all {sum(PLANS_IDENTICAL.values())}/{len(BUDGETS)} budgets "
    f"(F-collapse), provably possible because each budget's joint-optimal schedule is "
    "token-frontier-efficient (verified all 6/6) -- the zero-width G1 CI follows necessarily, not "
    "statistically."
)
print("\n--- Paste into LAB_LOG.md ---")
for s in (e5_sentence_1, e5_sentence_2, e5_sentence_3, e5_sentence_4):
    print("-", s)

# %% [markdown]
# ## Save your results (optional -- Colab sessions are ephemeral)
#
# `results/E5/` only exists in THIS Colab VM's disk. Push it back to your
# repo so it persists. Not run automatically -- uncomment and run only if
# you want to push.

# %%
# get_ipython().system(
#     'git add results/ && '
#     'git commit -m "E5 results" && '
#     'git push'
# )
