# %% [markdown]
# # B1 (ToMe) -- standalone Colab fallback
#
# **This notebook is a FALLBACK, not the primary path.** The primary path
# for B1 is `experiments/phase2_2_baselines/b1_tome/run.py` on the university 24GB GPU, inside
# its own pinned venv, against a local ImageNet-1k copy (see
# `experiments/phase2_2_baselines/README.md`). Use this notebook only if university GPU access
# is unavailable -- it invokes the EXACT SAME `run.py`, unmodified, so the
# method (ToMe r in {8,13,16} on DeiT-S, `prop_attn=True`, the pinned
# commit, the reproduction gate) is identical either way. The only
# difference is where ImageNet-1k comes from:
#
# - University GPU: a real local `IMAGENET_DIR` ImageFolder copy.
# - This notebook: the FULL 50k validation split is streamed from the HF
#   Hub and materialized on Colab's local disk as an ImageFolder (one-time,
#   ~10-20 min) via `experiments/phase2_2_baselines/common/colab_imagenet.py` -- a new,
#   Colab-only helper that does not touch any existing file. B1 needs no
#   training-split calibration data, so only `val/` is materialized here.
#
# **Prerequisite**: push `experiments/phase2_2_baselines/` (including this notebook and
# `common/colab_imagenet.py`) to your GitHub remote before opening this in
# Colab -- the clone cell below pulls from there.
#
# This repo is public -- cloning it needs no authentication.
#
# Runtime: Colab GPU runtime (Runtime -> Change runtime type -> GPU).

# %%
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
get_ipython().system('nvidia-smi')  # noqa: F821 -- Colab/IPython builtin

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
# ## Install B1's own pinned environment
#
# `b1_tome/requirements.txt` pins ToMe's own documented deps
# (`torch==1.12.1`, `timm==0.4.12`). Current Colab runtimes ship a Python
# too new for PyPI's `torch==1.12.1` wheels (no compatible distribution --
# a real upstream constraint, not a pip-flag fix). Rather than bump the
# pin (which WOULD change the standard baseline), this builds a dedicated
# Python 3.10 venv inside the Colab VM and installs the exact pinned
# requirements into THAT -- the same "one pinned venv per baseline" design
# the university-GPU path already uses (`experiments/phase2_2_baselines/README.md`), just built
# on the fly here. B1 runs IN-PROCESS (not a subprocess like B2/B3), so
# `run.py` itself must be launched BY this venv's python -- see the run
# cell below. `datasets`/`huggingface_hub` (for the Colab-only ImageNet
# materialization step) stay in the DEFAULT Colab kernel, not the venv.

# %%
import subprocess

get_ipython().system('apt-get -qq update && apt-get -qq install -y python3.10 python3.10-venv > /dev/null')  # noqa: F821, E501
get_ipython().system('python3.10 -m venv /content/venv_b1')  # noqa: F821
get_ipython().system('/content/venv_b1/bin/pip install -q --upgrade pip')  # noqa: F821
get_ipython().system('/content/venv_b1/bin/pip install -q -r experiments/phase2_2_baselines/b1_tome/requirements.txt')  # noqa: F821
VENV_B1_PYTHON = "/content/venv_b1/bin/python"
_check = subprocess.run(
    [VENV_B1_PYTHON, "-c", "import torch; print('TANDEM_CUDA_CHECK', torch.__version__, torch.cuda.is_available())"],
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
# ## Materialize ImageNet-1k val split as a local ImageFolder
#
# One-time per Colab runtime (idempotent -- safe to re-run this cell after
# a disconnect). If you mount Google Drive, you can point `IMAGENET_ROOT`
# at a Drive path instead of `/content` to cache this across sessions and
# across this project's other baseline notebooks (they all use the same
# materialized layout).

# %%
import sys

sys.path.insert(0, "experiments/phase2_2_baselines")
from common.colab_imagenet import materialize_val_split  # noqa: E402

IMAGENET_ROOT = "/content/imagenet"  # or e.g. "/content/drive/MyDrive/tandem_imagenet_cache" if Drive-mounted
materialize_val_split(IMAGENET_ROOT, HF_TOKEN, n_images=50000)

# %% [markdown]
# ## P2.2 post-conversion self-test -- run BEFORE any real run cell
#
# Ten hard-assert checks (wrapper purity + repo HEAD, upstream commit
# pinning, TF32/numeric environment, secrets scan, data-manifest
# determinism, cost-ruler unit checks, results.csv schema, a REAL
# interrupt-then-resume simulation on a 512-image slice, a 256-image
# host-equivalence smoke check, and a gate-ordering guard), printed as one
# PASS/FAIL banner. A FAILURE HALTS here (raises) -- "Run All" cannot reach
# any `--only-r` cell below until every check passes. Do not skip or
# comment out this cell. B1 runs IN-PROCESS, so this is launched by the
# pinned venv's python directly, same as `run.py` itself.

# %%
get_ipython().system(f'{VENV_B1_PYTHON} experiments/phase2_2_baselines/common/notebook_self_test.py --baseline tome --imagenet-dir {IMAGENET_ROOT} --device cuda --host-tag colab')  # noqa: F821, E501

# %% [markdown]
# ## Two settings, never comparable (setting-mismatch correction, 2026-07)
#
# ToMe's README/paper publish DeiT-S numbers for TWO different things that
# happen to share the r=13 label:
# - **`off_the_shelf`** (what every cell below measures): an off-the-shelf
#   pretrained DeiT-S checkpoint (`fb_in1k`, no ToMe involvement in
#   training), ToMe applied POST-HOC at eval time only. No published
#   off-the-shelf DeiT-S reference exists anywhere in ToMe's README or
#   paper (checked, including the paper's appendix Table 11 -- see
#   `b1_tome/published.json`'s `"note"` field) -- so NONE of r=8/13/16
#   below are gated; all three are reported with
#   `gate_status=no_published_reference`.
# - **`trained_with_tome`**: a DIFFERENT DeiT-S checkpoint, fine-tuned FROM
#   SCRATCH with ToMe merging active throughout training. Only located for
#   r=13, as a downloadable checkpoint (`deit_S_r13.pth`) with a published
#   number (79.36%). A previous version of this notebook incorrectly gated
#   our off-the-shelf r=13 run against this trained number -- a setting
#   mismatch, not a reproduction failure. Corrected: the trained checkpoint
#   now gets its OWN dedicated cell/gate below, and is never compared
#   against the off-the-shelf sweep.
#
# **Two BINDING gates** (failure halts `run.py`, same as before):
# - **r0_baseline**: the plain/unpatched off-the-shelf checkpoint, gated
#   against ToMe's own stated 79.8% baseline (in addition to P2.1's own S8
#   reference, already checked by the FP-parity gate) -- certifies this
#   harness's eval path. Free (reuses the FP-parity pass, no extra GPU
#   time) -- runs automatically as part of the first cell below.
# - **r=13 trained_with_tome**: downloads and evaluates ToMe's own
#   checkpoint, gated against 79.36% -- certifies the ToMe-patched eval
#   path itself. Its own dedicated cell (a real, separate ~7-10 min GPU
#   pass + one-time checkpoint download).

# %% [markdown]
# ## Run B1 -- the real, unmodified `run.py`, one r-value per cell
#
# Everything below is IDENTICAL to the university-GPU path: same pinned
# ToMe commit, same `prop_attn=True` convention, same reproduction gates,
# same `experiments/phase2_2_baselines/results.csv` schema. B1 runs IN-PROCESS, so it must be
# LAUNCHED by the pinned venv's python (not the default Colab kernel) --
# `{VENV_B1_PYTHON}`, built above.
#
# Split into **separate cells, one per r-value (plus one for the trained
# checkpoint)**, instead of one long cell -- so a Colab/university-GPU
# session limit or a transient failure on one never costs the ones that
# already finished. Each cell is independently safe to re-run: `run.py`
# checks `experiments/phase2_2_baselines/results.csv` before doing any work and skips
# whatever's already recorded there, so re-running a cell that already
# succeeded (or clicking "Run All" again after a partial failure) is a
# fast no-op, not a duplicate row or a wasted GPU pass. The first cell you
# run also pays a ONE-TIME full-50k FP-parity pass (plain, unpatched model
# vs. P2.1's own S8 reference AND ToMe's own stated 79.8% baseline --
# `r0_baseline`, both gated) -- cached, so later cells skip it.
#
# r=8/r=13(off_the_shelf)/r=16 are all reported without a gate (no located
# off-the-shelf reference for any of them -- see `deviations.md` #3 and
# `published.json`'s `"note"`). Each writes 2 rows to `results.csv`
# (`row_kind=reproduction` + `row_kind=comparison`, same measurement --
# DeiT-S's checkpoint coincides with TANDEM's pinned tag, see
# `common/checkpoint_pin.py`); all three together append 6 rows.
# `r0_baseline` and the trained-checkpoint gate each write exactly 1 row
# (no `comparison` duplicate -- see module docstring in `run.py`), for 8
# `tome` rows total.

# %% [markdown]
# ### r=8 (no published off-the-shelf reference -- reported, not gated)
#
# Also pays the `r0_baseline` gate (binding, vs ToMe's 79.8%) as part of
# the same FP-parity pass -- first cell run only.

# %%
get_ipython().system(f'{VENV_B1_PYTHON} experiments/phase2_2_baselines/b1_tome/run.py --imagenet-dir {IMAGENET_ROOT} --device cuda --host-tag colab --only-r 8')  # noqa: F821, E501

# %% [markdown]
# ### r=13 off-the-shelf (no published off-the-shelf reference -- reported, not gated)
#
# NOT gated against 79.36% -- that number is for the `trained_with_tome`
# checkpoint (see the setting-mismatch note above), evaluated separately
# in the next cell.

# %%
get_ipython().system(f'{VENV_B1_PYTHON} experiments/phase2_2_baselines/b1_tome/run.py --imagenet-dir {IMAGENET_ROOT} --device cuda --host-tag colab --only-r 13')  # noqa: F821, E501

# %% [markdown]
# ### r=16 (no published off-the-shelf reference -- reported, not gated)

# %%
get_ipython().system(f'{VENV_B1_PYTHON} experiments/phase2_2_baselines/b1_tome/run.py --imagenet-dir {IMAGENET_ROOT} --device cuda --host-tag colab --only-r 16')  # noqa: F821, E501

# %% [markdown]
# ### r=13 trained_with_tome (reproduction-gated against ToMe's published 79.36%)
#
# Downloads ToMe's own `deit_S_r13.pth` (~90MB, one-time, cached) and
# evaluates it on the full 50k val set -- a genuinely separate GPU pass
# from the off-the-shelf r=13 cell above (different checkpoint). A FAILURE
# halts `run.py`, same as any other binding gate in this project. Also
# (re-)pays `r0_baseline` if it hasn't run yet in this session (harmless,
# cached either way).

# %%
get_ipython().system(f'{VENV_B1_PYTHON} experiments/phase2_2_baselines/b1_tome/run.py --imagenet-dir {IMAGENET_ROOT} --device cuda --host-tag colab --trained-checkpoint-r13')  # noqa: F821, E501

# %% [markdown]
# ## Results + completeness check
#
# Confirms all off-the-shelf r-values PLUS both binding-gate rows actually
# landed in `results.csv` (8 rows: 3 r-values x 2 row_kinds, + r0_baseline,
# + r13 trained_with_tome) before you treat this baseline as done -- if
# any cell above was skipped or failed, this will say so explicitly rather
# than silently showing a partial table.

# %%
import pandas as pd

results_df = pd.read_csv("experiments/phase2_2_experiments/phase2_2_baselines/results.csv")
tome_rows = results_df[results_df["method"] == "tome"]
print(tome_rows.to_string(index=False))
EXPECTED_ROWS = 8  # 3 off-the-shelf r-values x 2 row_kinds + r0_baseline (1) + r13 trained_with_tome (1)
assert len(tome_rows) == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} tome rows (r in {{8,13,16}} off_the_shelf x 2 row_kinds, + r0_baseline, "
    f"+ r13 trained_with_tome), found {len(tome_rows)} -- re-run whichever cell above is missing from "
    "the table before treating B1 as complete."
)
print(f"\nAll {EXPECTED_ROWS} expected rows present -- B1 complete.")

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
#     'git commit -m "B1 (ToMe) Colab results" && '
#     'git push'
# )
