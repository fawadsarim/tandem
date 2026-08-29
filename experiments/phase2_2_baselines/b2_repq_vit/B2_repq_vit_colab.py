# %% [markdown]
# # B2 (RepQ-ViT) -- standalone Colab fallback
#
# **This notebook is a FALLBACK, not the primary path.** The primary path
# for B2 is `experiments/phase2_2_baselines/b2_repq_vit/run.py` on the university 24GB GPU,
# against a local ImageNet-1k copy (see `experiments/phase2_2_baselines/README.md`). Use this
# notebook only if university GPU access is unavailable -- it invokes the
# EXACT SAME `run.py`, unmodified, which itself subprocesses RepQ-ViT's own
# `classification/test_quant.py` at the pinned commit with the
# verified-from-source `--w_bits`/`--a_bits` flags. Method, models
# (DeiT-Tiny/DeiT-Small/DeiT-Base/ViT-B, W8A8 and W6A6), and the
# reproduction gate are identical either way. For ViT-B specifically,
# `run.py` also runs a SECOND pass on TANDEM's own pinned checkpoint
# (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) -- RepQ-ViT's own
# `--model vit_base` resolves to a DIFFERENT, lower-accuracy checkpoint
# (`augreg_in21k_ft_in1k`); see `common/checkpoint_pin.py`'s module
# docstring for the source-verified evidence. That second pass downloads
# ~350MB of pinned weights from huggingface.co the first time it runs
# (cached under `experiments/phase2_2_baselines/_pinned_checkpoints/` after that). The only
# difference from the university-GPU path is where ImageNet-1k comes
# from:
#
# - University GPU: a real local `IMAGENET_DIR` ImageFolder copy (`train/`
#   for the 32-image calibration draw, `val/` for the full 50k eval).
# - This notebook: the FULL 50k validation split is streamed and
#   materialized on Colab's local disk, plus a class-diverse TRAIN
#   CALIBRATION SUBSET (not the full 1.28M-image train set -- see
#   `common/colab_imagenet.py`'s docstring for exactly what this
#   approximates and how to widen it if needed).
#
# **Prerequisite**: push `experiments/phase2_2_baselines/` (including this notebook and
# `common/colab_imagenet.py`) to your GitHub remote before opening this in
# Colab.
#
# This repo is public -- cloning it needs no authentication.
#
# Runtime: Colab GPU runtime (Runtime -> Change runtime type -> GPU).

# %%
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
get_ipython().system('nvidia-smi')  # noqa: F821

# %%
import os

REPO_URL = "https://github.com/fawadsarim/tandem.git"
REPO_DIR = "/content/tandem"

if not os.path.exists(REPO_DIR):
    get_ipython().system(f'git clone -q {REPO_URL} {REPO_DIR}')  # noqa: F821
else:
    print(f"{REPO_DIR} already exists -- pulling latest instead of re-cloning.")
    get_ipython().system(f'git -C {REPO_DIR} pull -q')  # noqa: F821

get_ipython().run_line_magic('cd', REPO_DIR)  # noqa: F821

# %% [markdown]
# ## Install B2's own pinned environment
#
# `run.py` itself has no torch/timm dependency (it subprocesses
# `test_quant.py`), but `b2_repq_vit/requirements.txt` pins what
# `test_quant.py` needs (`torch==1.12.1`, `timm==0.4.12`, per RepQ-ViT's
# README) -- and current Colab runtimes ship a Python too new for PyPI's
# `torch==1.12.1` wheels (no compatible distribution -- this is a real
# upstream constraint, not something a pip flag fixes). Rather than bump
# the pin (which WOULD change the standard baseline), this builds a
# dedicated Python 3.10 venv inside the Colab VM and installs the exact
# pinned requirements into THAT -- the same "one pinned venv per baseline"
# design the university-GPU path already uses (`experiments/phase2_2_baselines/README.md`),
# just built on the fly here instead of pre-existing. `datasets`/
# `huggingface_hub` stay in the DEFAULT Colab kernel (below) since
# `colab_imagenet.py` never imports torch.

# %%
import subprocess

get_ipython().system('apt-get -qq update && apt-get -qq install -y python3.10 python3.10-venv > /dev/null')  # noqa: F821, E501
get_ipython().system('python3.10 -m venv /content/venv_b2')  # noqa: F821
get_ipython().system('/content/venv_b2/bin/pip install -q --upgrade pip')  # noqa: F821
get_ipython().system('/content/venv_b2/bin/pip install -q -r experiments/phase2_2_baselines/b2_repq_vit/requirements.txt')  # noqa: F821
VENV_B2_PYTHON = "/content/venv_b2/bin/python"
_check = subprocess.run(
    [VENV_B2_PYTHON, "-c", "import torch; print('TANDEM_CUDA_CHECK', torch.__version__, torch.cuda.is_available())"],
    capture_output=True, text=True,
)
print(_check.stdout)
# Filtered to a single marker-prefixed line (not a fixed line-count unpack)
# so a stray warning printed to stdout by torch/numpy/etc. during import
# can never masquerade as a parse error -- if the marker line is simply
# missing, the assert below reports the real stderr instead.
_result_lines = [ln for ln in _check.stdout.splitlines() if ln.startswith("TANDEM_CUDA_CHECK")]
assert _result_lines, f"venv torch import failed or produced no output. stderr:\n{_check.stderr[-2000:]}"
_, _torch_version, _cuda_available = _result_lines[-1].split()
assert _cuda_available == "True", (
    f"venv torch {_torch_version} cannot see the GPU (torch.cuda.is_available()=False) -- do not "
    "proceed past this check. Check Runtime -> Change runtime type -> GPU is actually selected, and "
    "that this cell's apt/venv setup completed without errors above."
)
get_ipython().system('pip install -q datasets huggingface_hub')  # noqa: F821

# %% [markdown]
# ## HF_TOKEN (Colab Secrets only -- never paste a token into a cell)
#
# Add a Colab Secret named `HF_TOKEN` (key icon in the left sidebar) with a
# read-token that has accepted the `ILSVRC/imagenet-1k` gated-dataset terms
# on huggingface.co, and enable notebook access for it.

# %%
from google.colab import userdata  # noqa: E402

HF_TOKEN = userdata.get("HF_TOKEN")
assert HF_TOKEN, "Add an HF_TOKEN Colab Secret (see markdown above) before continuing."

# %% [markdown]
# ## Materialize ImageNet-1k (val + train calibration subset) as a local ImageFolder
#
# One-time per Colab runtime (idempotent -- safe to re-run after a
# disconnect). If you mount Google Drive, point `IMAGENET_ROOT` at a Drive
# path to cache this across sessions and across this project's other
# baseline notebooks.

# %%
import sys

sys.path.insert(0, "experiments/phase2_2_baselines")
from common.colab_imagenet import materialize_train_calib_subset, materialize_val_split  # noqa: E402

IMAGENET_ROOT = "/content/imagenet"  # or e.g. "/content/drive/MyDrive/tandem_imagenet_cache" if Drive-mounted
materialize_val_split(IMAGENET_ROOT, HF_TOKEN, n_images=50000)
# RepQ-ViT's own calibration draws exactly ONE batch of 32 images
# (`--calib-batchsize 32`, their default) from a shuffled DataLoader over
# train/ -- a few thousand class-diverse images is comfortably enough pool
# for that draw.
materialize_train_calib_subset(IMAGENET_ROOT, HF_TOKEN, images_per_class=10, max_images=5000)

# %% [markdown]
# ## P2.2 post-conversion self-test -- run BEFORE any real run cell
#
# Ten hard-assert checks (wrapper purity + repo HEAD, upstream commit
# pinning, TF32/numeric environment, secrets scan, data-manifest
# determinism, cost-ruler unit checks, results.csv schema, a resume-
# mechanism self-test, a 256-image host-equivalence smoke check, and a
# gate-ordering guard), printed as one PASS/FAIL banner. A FAILURE HALTS
# here (raises) -- "Run All" cannot reach any `--only-model` cell below
# until every check passes. Do not skip or comment out this cell. `run.py`
# itself is torch-free, so this runs under the DEFAULT Colab kernel;
# `--python` points the numeric-environment and equivalence-fingerprint
# checks at the pinned Python 3.10 venv built above, same as every other
# torch-dependent step in this notebook.

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/common/notebook_self_test.py --baseline repq_vit --imagenet-dir {IMAGENET_ROOT} --python {VENV_B2_PYTHON} --host-tag colab')  # noqa: F821, E501

# %% [markdown]
# ## Run B2 -- the real, unmodified `run.py`, one model per cell
#
# Everything below is IDENTICAL to the university-GPU path: same pinned
# RepQ-ViT commit, same verified `--w_bits`/`--a_bits` flags, same
# reproduction gate, same `experiments/phase2_2_baselines/results.csv` schema. `run.py` itself
# is torch-free and runs under the DEFAULT Colab kernel; `--python` points
# it at the dedicated Python 3.10 venv built above for every torch-
# dependent subprocess call (host-info, calibration capture, FP-parity
# gate, `test_quant.py` itself).
#
# Split into **four separate cells, one per model**, instead of one long
# cell covering all four -- this is exactly where a real run previously
# hit a session-ending OOM crash on DeiT-Base's w8a8 config, AFTER
# DeiT-Tiny and DeiT-Small had already succeeded (see `deviations.md`
# #24). Each cell is independently safe to re-run: `run.py` checks
# `experiments/phase2_2_baselines/results.csv` before doing any (model, config, row_kind) work
# and skips whatever's already recorded, so re-running a cell (or "Run
# All" after a partial failure) never redoes completed work or duplicates
# a row.
#
# Per model, `run.py` runs an FP-parity full-50k gate pass (plain model
# vs. P2.1's own S8 reference) before the checkpoint that gate validates
# is swept, then the {w8a8, w6a6} sweep. **ViT-B's cell is the heaviest**:
# its own `--model vit_base` resolves to a DIFFERENT, lower-accuracy
# checkpoint than TANDEM's pinned one (see the note above), so `run.py`
# additionally runs a real SECOND {w8a8, w6a6} sweep on TANDEM's pinned
# checkpoint (Step B) -- downloading ~350MB of weights the first time.
# DeiT-Tiny/Small/Base's Step B rows instead reuse Step A's measurement
# (checkpoints coincide, no second sweep). Each cell appends 4 rows (1
# model x 2 configs x 2 row_kinds); all four together append 16.

# %% [markdown]
# ### DeiT-Tiny

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b2_repq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B2_PYTHON} --host-tag colab --only-model deit_tiny_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ### DeiT-Small

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b2_repq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B2_PYTHON} --host-tag colab --only-model deit_small_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ### DeiT-Base

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b2_repq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B2_PYTHON} --host-tag colab --only-model deit_base_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ### ViT-B (heaviest cell -- real Step A + Step B sweeps, ~350MB checkpoint download)

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b2_repq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B2_PYTHON} --host-tag colab --only-model vit_base_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ## Results + completeness check
#
# Confirms all four models actually landed in `results.csv` (16 rows: 4
# models x 2 configs x 2 row_kinds) before you treat this baseline as
# done -- if any of the four cells above was skipped or failed, this will
# say so explicitly rather than silently showing a partial table.

# %%
import pandas as pd

results_df = pd.read_csv("experiments/phase2_2_experiments/phase2_2_baselines/results.csv")
repq_rows = results_df[results_df["method"] == "repq_vit"]
print(repq_rows.to_string(index=False))
EXPECTED_ROWS = 16  # 4 models x 2 configs x 2 row_kinds
assert len(repq_rows) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} repq_vit rows (4 models x 2 configs x 2 row_kinds), found {len(repq_rows)} "
    "-- re-run whichever --only-model cell above is missing from the table before treating B2 as complete."
)
print(f"\nAll {EXPECTED_ROWS} expected rows present -- B2 complete.")

# %% [markdown]
# ## Save your results (optional -- Colab sessions are ephemeral)
#
# `experiments/phase2_2_baselines/results.csv` only exists in THIS Colab VM's disk. Either
# download it (Files sidebar -> right-click -> Download) or push it back to
# your repo so it persists and B5 (`compare_pareto.py`) can read it later.
# Not run automatically -- uncomment and run only if you want to push.

# %%
# get_ipython().system(
#     'git add experiments/phase2_2_baselines/results.csv && '
#     'git commit -m "B2 (RepQ-ViT) Colab results" && '
#     'git push'
# )
