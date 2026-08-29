# %% [markdown]
# # B3 (FQ-ViT) -- standalone Colab fallback
#
# **This notebook is a FALLBACK, not the primary path.** The primary path
# for B3 is `experiments/phase2_2_baselines/b3_fq_vit/run.py` on the university 24GB GPU,
# against a local ImageNet-1k copy (see `experiments/phase2_2_baselines/README.md`). Use this
# notebook only if university GPU access is unavailable -- it invokes the
# EXACT SAME `run.py`, unmodified, which itself subprocesses FQ-ViT's own
# `test_quant.py` at the pinned commit with their "Ours" flags (`--quant
# --ptf --lis --quant-method minmax`). Method, models (DeiT-Tiny/Small/Base
# + ViT-B), and the reproduction gate are identical either way. For ViT-B
# specifically, `run.py` also runs a SECOND pass on TANDEM's own pinned
# checkpoint (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) -- FQ-ViT's own
# `vit_base` loader resolves to a DIFFERENT, lower-accuracy checkpoint
# (`augreg_in21k_ft_in1k`); see `common/checkpoint_pin.py`'s module
# docstring for the source-verified evidence. That second pass downloads
# ~350MB of pinned weights from huggingface.co the first time it runs
# (cached under `experiments/phase2_2_baselines/_pinned_checkpoints/` after that). The only
# other difference is where ImageNet-1k comes from (see
# `common/colab_imagenet.py`).
#
# **CUDA-compatibility risk, carried over from `deviations.md` #1 (this is
# not a new risk introduced by this notebook, just surfaced here too):**
# FQ-ViT's README pins `torch==1.7.1` with `cudatoolkit=10.1`, which
# predates Colab's current GPU driver stack. `pip install torch==1.7.1`
# will pull a CUDA build that may not initialize against Colab's driver.
# The cell right after install checks `torch.cuda.is_available()`
# explicitly and tells you what to do if it fails -- do not proceed past
# that check silently.
#
# **Prerequisite**: push `experiments/phase2_2_baselines/` (including this notebook and
# `common/colab_imagenet.py`) to your GitHub remote before opening this in
# Colab.
#
# This repo is public -- cloning it needs no authentication.
#
# Runtime: Colab GPU runtime (Runtime -> Change runtime type -> GPU).

# %%
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
# ## Install B3's own pinned environment
#
# `b3_fq_vit/requirements.txt` pins FQ-ViT's own documented `torch==1.7.1`,
# which needs Python <=3.8 -- older than `python3.10` (B1/B2/B4's need) and
# no longer available at all as a plain `apt-get install` on Colab's
# current Ubuntu base, so this pulls Python 3.8 from the `deadsnakes` PPA
# specifically. This is B3's highest-risk step in this whole set of
# baselines (see `deviations.md` #1) for TWO independent reasons: (1) the
# Python-3.8-via-PPA install itself is slower and more fragile than B1/B2/
# B4's plain `apt-get install python3.10`, and (2) even once torch==1.7.1
# is installed, `cudatoolkit=10.1` predates Colab's current GPU driver
# stack and may simply refuse to see the GPU -- the STOP-AND-CHECK below
# catches that explicitly rather than letting it fail silently deep into a
# slow CPU run. `datasets`/`huggingface_hub` stay in the DEFAULT Colab
# kernel, not the venv.

# %%
import subprocess

get_ipython().system('apt-get -qq update && apt-get -qq install -y software-properties-common > /dev/null')  # noqa: F821, E501
get_ipython().system('add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1')  # noqa: F821
get_ipython().system('apt-get -qq update && apt-get -qq install -y python3.8 python3.8-venv python3.8-distutils > /dev/null')  # noqa: F821, E501
get_ipython().system('python3.8 -m venv /content/venv_b3')  # noqa: F821
get_ipython().system('/content/venv_b3/bin/python -m pip install -q --upgrade pip')  # noqa: F821
get_ipython().system('/content/venv_b3/bin/pip install -q -r experiments/phase2_2_baselines/b3_fq_vit/requirements.txt')  # noqa: F821, E501
get_ipython().system('pip install -q datasets huggingface_hub')  # noqa: F821

# %% [markdown]
# ## STOP AND CHECK: does torch==1.7.1 actually see Colab's GPU?
#
# If this prints `False`, the pinned torch build is not compatible with
# this Colab runtime's driver -- do not proceed. Options: (a) try a
# CUDA-11-built `torch==1.7.1+cu110` wheel instead (edit
# `b3_fq_vit/requirements.txt`'s pin and re-run the cell above), or (b)
# skip B3 in this session and run it on the university GPU instead, where
# the environment is fully under your control. This is a real,
# hardware-dependent risk already documented in `deviations.md` #1 for the
# university-GPU path -- unresolved here for the same reason.

# %%
VENV_B3_PYTHON = "/content/venv_b3/bin/python"
_check = subprocess.run(
    [VENV_B3_PYTHON, "-c", "import torch; print('TANDEM_CUDA_CHECK', torch.__version__, torch.cuda.is_available())"],
    capture_output=True, text=True,
)
print(_check.stdout)
if _check.returncode != 0:
    print(_check.stderr[-3000:])
# Filtered to a single marker-prefixed line (not a fixed line-count unpack)
# -- if torch failed to import at all (a real, live possibility for B3
# specifically: python3.8-via-PPA install issues, or torch==1.7.1 itself
# refusing to load against this driver), stdout has ZERO lines, not just
# an unexpected one, and unpacking a fixed count would raise a confusing
# "not enough values to unpack" instead of pointing at the real cause.
_result_lines = [ln for ln in _check.stdout.splitlines() if ln.startswith("TANDEM_CUDA_CHECK")]
assert _result_lines, f"venv torch import failed or produced no output. stderr:\n{_check.stderr[-3000:]}"
_, _torch_version, _cuda_available = _result_lines[-1].split()
print("torch:", _torch_version, "CUDA available:", _cuda_available)
assert _cuda_available == "True", (
    "torch==1.7.1 cannot see this Colab GPU -- see the markdown above for options. "
    "Do not continue past this point with CPU-only torch 1.7.1 (B3's full 50k eval "
    "would be impractically slow)."
)

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
# One-time per Colab runtime (idempotent). If you mount Google Drive, point
# `IMAGENET_ROOT` at a Drive path to cache this across sessions and across
# this project's other baseline notebooks.

# %%
import sys

sys.path.insert(0, "experiments/phase2_2_baselines")
from common.colab_imagenet import materialize_train_calib_subset, materialize_val_split  # noqa: E402

IMAGENET_ROOT = "/content/imagenet"  # or e.g. "/content/drive/MyDrive/tandem_imagenet_cache" if Drive-mounted
materialize_val_split(IMAGENET_ROOT, HF_TOKEN, n_images=50000)
# FQ-ViT's own calibration is 100 x 10 = 1,000 images (their default,
# --calib-batchsize/--calib-iter, unchanged) drawn from train/ -- a larger
# class-diverse pool than B2's since FQ-ViT's own draw is bigger.
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
# checks at the pinned Python 3.8 venv built above, same as every other
# torch-dependent step in this notebook.

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/common/notebook_self_test.py --baseline fq_vit --imagenet-dir {IMAGENET_ROOT} --python {VENV_B3_PYTHON} --host-tag colab')  # noqa: F821, E501

# %% [markdown]
# ## Run B3 -- the real, unmodified `run.py`, one model per cell
#
# Everything below is IDENTICAL to the university-GPU path: same pinned
# FQ-ViT commit, same "Ours" flags, same reproduction gate, same
# `experiments/phase2_2_baselines/results.csv` schema. `run.py` itself is torch-free and runs
# under the DEFAULT Colab kernel; `--python` points it at the dedicated
# Python 3.8 venv built above for every torch-dependent subprocess call.
#
# Split into **four separate cells, one per model**, instead of one long
# cell covering all four -- so a session limit or a transient failure on
# one model never costs the models that already finished. Each cell is
# independently safe to re-run: `run.py` checks `experiments/phase2_2_baselines/results.csv`
# before doing any (model, config, row_kind) work and skips whatever's
# already recorded, so re-running a cell (or "Run All" after a partial
# failure) never redoes completed work or duplicates a row.
#
# Per model, `run.py` runs an FP-parity full-50k gate pass before the
# checkpoint that gate validates is swept, then the W8A8 config. **ViT-B's
# cell is the heaviest**: its own loader resolves to a DIFFERENT, lower-
# accuracy checkpoint than TANDEM's pinned one (see the note above), so
# `run.py` additionally runs a real SECOND W8A8 pass on TANDEM's pinned
# checkpoint (Step B) -- downloading ~350MB of weights the first time.
# DeiT-Tiny/Small/Base's Step B rows instead reuse Step A's measurement
# (checkpoints coincide, no second pass). Each DeiT cell appends 2 rows (1
# model x 1 config x 2 row_kinds); ViT-B's cell also appends 2 (Step A +
# Step B, both real passes); all four together append 8.

# %% [markdown]
# ### DeiT-Tiny

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b3_fq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B3_PYTHON} --host-tag colab --only-model deit_tiny_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ### DeiT-Small

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b3_fq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B3_PYTHON} --host-tag colab --only-model deit_small_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ### DeiT-Base

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b3_fq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B3_PYTHON} --host-tag colab --only-model deit_base_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ### ViT-B (heaviest cell -- real Step A + Step B passes, ~350MB checkpoint download)

# %%
get_ipython().system(f'python experiments/phase2_2_baselines/b3_fq_vit/run.py --imagenet-dir {IMAGENET_ROOT} --python {VENV_B3_PYTHON} --host-tag colab --only-model vit_base_patch16_224')  # noqa: F821, E501

# %% [markdown]
# ## Results + completeness check
#
# Confirms all four models actually landed in `results.csv` (8 rows: 4
# models x 2 row_kinds) before you treat this baseline as done -- if any
# of the four cells above was skipped or failed, this will say so
# explicitly rather than silently showing a partial table.

# %%
import pandas as pd

results_df = pd.read_csv("experiments/phase2_2_experiments/phase2_2_baselines/results.csv")
fqvit_rows = results_df[results_df["method"] == "fq_vit"]
print(fqvit_rows.to_string(index=False))
EXPECTED_ROWS = 8  # 4 models x 2 row_kinds
assert len(fqvit_rows) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} fq_vit rows (4 models x 2 row_kinds), found {len(fqvit_rows)} -- "
    "re-run whichever --only-model cell above is missing from the table before treating B3 as complete."
)
print(f"\nAll {EXPECTED_ROWS} expected rows present -- B3 complete.")

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
#     'git commit -m "B3 (FQ-ViT) Colab results" && '
#     'git push'
# )
