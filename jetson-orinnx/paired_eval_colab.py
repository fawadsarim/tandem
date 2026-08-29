# %% [markdown]
# # jetson-orinnx/paired_eval.py -- Colab runner
#
# Runs `paired_eval.py` on Colab's GPU (deliberately NOT the Jetson -- see
# `jetson-orinnx/README.md`'s "Harness usage" section: this step is pure PyTorch,
# no TensorRT/`trtexec` involved at all, so it runs wherever is easiest).
# This is the step that actually answers G1-HW's real question -- does the
# plan TANDEM's simulated BOPs cost picks still win once cost is measured
# real Jetson latency? -- so getting the REGISTERED PROTOCOL right here
# (see `paired_eval.py`'s own module docstring, reproduced in spirit
# below) is what makes this notebook's output usable as the actual
# thesis result, not just a run log.
#
# **Prerequisite**: `replan.py` must have already run ON THE JETSON,
# producing `jetson-orinnx/out/replan/plans_hw.csv`. This notebook cannot reach
# the Jetson directly (different networks) -- you upload that one file
# below.
#
# This repo is public -- cloning it needs no authentication.
# **HF auth** -- add a Colab Secret named `HF_TOKEN`, needed
# to stream the paired-eval image set below. That token must have
# accepted the `ILSVRC/imagenet-1k` GATED dataset's terms on
# huggingface.co (visit that dataset's page while logged in, accept, then
# make sure the token has notebook/read access) -- without that, the
# materialization cell below fails with an access-denied error, not a
# missing-token one. Same requirement every experiments/phase2_2_baselines/*_colab.ipynb
# notebook already documents. Never a literal in any cell.
#
# Runtime: Runtime -> Change runtime type -> a GPU (A100/L4 will finish
# faster than T4 here -- this step evaluates 6 budgets x 3 plans (BOPs,
# JOINT, STAGED -- third arm added on review, 2026-07, see
# replan.py's own docstring) x 20,000 images, the heaviest compute in
# this whole notebook.

# %%
# Cheap, early GPU check using Colab's own pre-installed torch -- before cloning anything or
# building a venv, so a forgotten GPU runtime fails in seconds, not after several minutes of
# setup. Same pattern as experiments/phase2_2_baselines/b1_tome/B1_tome_colab.ipynb's own first cell.
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
# ## Isolated venv for `requirements-host.txt`
#
# Real bug found and fixed, 2026-07: an earlier version of this cell ran
# `pip install -r jetson-orinnx/requirements-host.txt` directly into Colab's
# DEFAULT kernel environment. That pins `numpy==1.26.4`, which pip
# happily installs -- but Colab's default kernel comes with a long list
# of *other* pre-installed packages (`shap`, `opencv-python`, `jax`,
# `cupy-cuda12x`, `tifffile`, `rasterio`, ...) that require `numpy>=2`,
# none of which this notebook uses but all of which pip's resolver now
# reports as broken. Worse, that same downgrade left whatever `pandas`
# Colab already had in place compiled against a *newer* numpy ABI than
# the one now installed, which crashed with a `numpy.dtype size
# changed... Expected 96 from C header, got 88 from PyObject` error the
# first time `datasets` (below) touched `pandas`.
#
# This is the exact same class of problem P2.2's `experiments/phase2_2_baselines/` notebooks
# already solved (see e.g. `b1_tome/B1_tome_colab.ipynb`'s own venv
# cell): give the pinned, framework-heavy requirements their OWN
# isolated venv instead of touching the shared default kernel at all.
# Unlike the baselines (which pin an old torch to match an upstream
# repo's documented floor, hence their `python3.10`), nothing here has
# an old-version floor to match -- `torch==2.11.0` is a current, modern
# pin -- so this venv just reuses the Colab VM's own current `python3`.
#
# `datasets`/`huggingface_hub` (needed below only to materialize the
# paired-eval image set, not a `jetson-orinnx/` dependency) deliberately stay
# OUT of this venv and get installed into the DEFAULT Colab kernel
# instead (next cell) -- so they keep whatever numpy Colab's base image
# already ships, and never have to coexist with this venv's
# `numpy==1.26.4` pin at all.

# %%
# Uses `virtualenv` (a pip package, bundles its own pip bootstrap), not stdlib `venv` +
# the `python3-venv` apt package -- real bug found and fixed, 2026-07: `apt-get install
# python3-venv` depends on Ubuntu's package mirrors having that exact .deb build
# available, and on a real Colab run this 404'd (`Failed to fetch .../python3.10-venv_...
# 404 Not Found`), which then made stdlib `venv`'s `ensurepip` step fail with no pip
# installed into the new venv at all. `virtualenv` never touches apt or ensurepip -- it's
# a plain PyPI wheel already carrying what it needs -- so this whole failure mode can't
# recur regardless of Ubuntu mirror state.
get_ipython().system('pip install -q virtualenv')  # noqa: F821
# --clear: real bug found and fixed, 2026-07. Colab's "Runtime -> Restart session" restarts
# only the KERNEL, not the underlying VM disk -- /content/ (including this venv) can persist
# across a restart. `virtualenv` reuses an existing directory by default rather than rebuilding
# clean, so a venv left in a bad state by an earlier failed run (e.g. a stray pip install that
# bumped numpy past this file's pin) could still be sitting there even after "restart and
# re-run from top." --clear forces a genuinely fresh venv every time this cell runs, regardless
# of what's already on disk -- costs a little install time, removes an entire class of
# "worked differently depending on session history" bugs.
get_ipython().system('python3 -m virtualenv -q --clear /content/venv_jetson-orinnx_host')  # noqa: F821
get_ipython().system('/content/venv_jetson-orinnx_host/bin/pip install -q -r jetson-orinnx/requirements-host.txt')  # noqa: F821
VENV_PYTHON = "/content/venv_jetson-orinnx_host/bin/python"

import subprocess  # noqa: E402

_check = subprocess.run(
    [VENV_PYTHON, "-c",
     "import torch, numpy, pandas; print('TANDEM_ENV_CHECK', torch.__version__, torch.cuda.is_available(), numpy.__version__, pandas.__version__)"],
    capture_output=True, text=True,
)
print(_check.stdout)
if _check.returncode != 0:
    print(_check.stderr[-2000:])
_result_lines = [l for l in _check.stdout.splitlines() if l.startswith("TANDEM_ENV_CHECK")]
assert _result_lines, f"venv import check failed or produced no output. stderr:\n{_check.stderr[-2000:]}"
_, _torch_version, _cuda_ok, _numpy_version, _pandas_version = _result_lines[-1].split()
assert _cuda_ok == "True", (
    f"venv torch {_torch_version} cannot see the GPU (torch.cuda.is_available()=False) -- do not "
    "continue until this is fixed (Runtime -> Change runtime type -> a GPU runtime), and confirm "
    "that this cell's virtualenv setup completed without errors above."
)
print(f"torch {_torch_version}, numpy {_numpy_version}, pandas {_pandas_version}, CUDA available: True (isolated venv at {VENV_PYTHON!r})")

# %% [markdown]
# ## `datasets`/`huggingface_hub` -- DEFAULT Colab kernel, not the venv
#
# Only used below to materialize the paired-eval image set, in the
# notebook's own (default) kernel -- never imported by `paired_eval.py`
# itself, which runs through the isolated venv.
#
# **`HF_TOKEN` secret**: needed below to stream the `ILSVRC/imagenet-1k`
# GATED dataset. Add it via Colab's Secrets panel (key icon on the left).
# Never a literal in any cell.

# %%
get_ipython().system('pip install -q datasets huggingface_hub')  # noqa: F821

# %% [markdown]
# ## HF auth (env var only, never a literal)

# %%
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
assert HF_TOKEN, "Add an HF_TOKEN Colab Secret -- see markdown above."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# %% [markdown]
# ## Prerequisite: upload `plans_hw.csv` from the Jetson
#
# `replan.py` must have run ON the Jetson first, producing
# `jetson-orinnx/out/replan/plans_hw.csv`. Run the cell below to upload it here.
# It places it at the exact path `paired_eval.py` expects.

# %%
import pathlib  # noqa: E402
from google.colab import files  # noqa: E402

REPLAN_DIR = pathlib.Path("jetson-orinnx/out/replan")
REPLAN_DIR.mkdir(parents=True, exist_ok=True)
HW_PATH = REPLAN_DIR / "plans_hw.csv"

if not HW_PATH.exists():
    print(f"Upload plans_hw.csv (from the Jetson's jetson-orinnx/out/replan/plans_hw.csv):")
    uploaded = files.upload()
    assert "plans_hw.csv" in uploaded, "Expected a file named 'plans_hw.csv' -- aborting."
    HW_PATH.write_bytes(uploaded["plans_hw.csv"])
    print(f"Wrote {HW_PATH} ({len(uploaded['plans_hw.csv'])} bytes)")
else:
    print(f"{HW_PATH} already present -- reusing.")

# %% [markdown]
# ## Materialize the paired-eval image set
#
# `paired_eval.py --imagenet-dir` needs a LOCAL ImageFolder-layout
# directory (not HF streaming directly -- see that script's own
# docstring on why: it ensures exact, reproducible, sorted-path
# ordering).
#
# Downloads `CALIB_N + PAIRED_EVAL_N = 512 + 20,000 = 20,512` images off
# the Hugging Face Hub (the validation split, streaming mode -- no full
# dataset download). This is the heaviest download cell in this notebook
# (~2-3 GB on disk); budget ~5-10 minutes depending on network.

# %%
import pathlib  # noqa: E402
from datasets import load_dataset  # noqa: E402

EVAL_DIR = pathlib.Path("/content/paired_eval_imagenet")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
N_NEEDED = 20512  # 512 calib + 20000 paired-eval

existing = sum(1 for _ in EVAL_DIR.rglob("*.jpg")) if EVAL_DIR.exists() else 0
if existing >= N_NEEDED:
    print(f"{existing} images already present under {EVAL_DIR} -- skipping re-download.")
else:
    # "ILSVRC/imagenet-1k", not the bare "imagenet-1k" -- real bug found and fixed, 2026-07: HF
    # moved this dataset under the ILSVRC namespace, and current huggingface_hub/datasets no longer
    # resolve the old bare repo id (fails with HfUriError: "Repository id must be 'namespace/name'").
    # Same repo id already used and proven working in experiments/phase2_2_baselines/common/colab_imagenet.py.
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=HF_TOKEN)
    n_written = existing
    for i, example in enumerate(ds):
        if n_written >= N_NEEDED:
            break
        if i < existing:  # crude resume: skip past however many we already wrote
            continue
        label = example["label"]
        class_dir = EVAL_DIR / f"class_{label:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        img = example["image"].convert("RGB")
        img.save(class_dir / f"img_{i:06d}.jpg", quality=95)
        n_written += 1
        if n_written % 2000 == 0:
            print(f"  {n_written}/{N_NEEDED} images written...")
    print(f"Done: {n_written} images under {EVAL_DIR}")

# %% [markdown]
# ## Run paired_eval.py
#
# This is a real run of the REGISTERED protocol (paired_eval.py's own
# module docstring) -- fixed disjoint calibration/eval split, both plans
# evaluated on identical images, the pooled gate computed exactly once
# after the full sweep, written unconditionally regardless of outcome.
# No peeking at partial results happens here or inside the script.
#
# Launched BY the isolated venv's python (`VENV_PYTHON`, built above),
# not the default Colab kernel -- see that cell for why.
#
# Real bug found and fixed, 2026-07 (two rounds): `get_ipython().system(...)` used to silently
# swallow a nonzero exit code -- a crash here used to print its error and then let the notebook
# continue into the Results cell below, which then failed with a confusing `FileNotFoundError`
# on `paired_eval_result.json` instead of showing the real error where it happened. The first
# fix (capturing `get_ipython().system(...)`'s return value and asserting it's 0) was ITSELF
# wrong: IPython's `system()` deliberately always returns `None` (it stores the real exit code
# in `_exit_code` on the user namespace instead, specifically so a non-None return doesn't
# trigger cell auto-printing) -- so that assert failed on every run, including a fully
# successful one. Fixed for real by using `subprocess.run(...)` directly instead of the
# `!`/`.system()` shell magic -- a real, documented `.returncode`, no IPython-internal behavior
# to get wrong.
#
# Real bug found and fixed, 2026-07 (third round): the plain `subprocess.run(...)` above
# inherits this process's stdout/stderr fds, but `paired_eval.py`'s own interpreter still
# block-buffers its writes because that inherited fd isn't a real tty -- so output arrived in
# occasional bursts (sometimes only at exit) instead of live, same buffering class already fixed
# with `-u` in `int8_qdq_experiment.py`'s subprocess call. Fixed by (a) passing `-u` so the
# CHILD interpreter itself writes unbuffered, and (b) reading its combined stdout+stderr via a
# `Popen` pipe and printing line-by-line as it arrives, rather than trusting fd inheritance alone.

# %%
import subprocess  # noqa: E402

_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson-orinnx/paired_eval.py", "--imagenet-dir", str(EVAL_DIR)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"paired_eval.py failed (exit code {_proc.returncode}) -- see the real error printed above."

# %% [markdown]
# ## Results -- this is the thesis paste-back
#
# Prints the full registered-protocol output verbatim: the pooled mean
# paired delta-top1 + 95% CI, the gate pass/fail, the per-budget
# breakdown (including `plans_identical`/`n_blocks_differing` -- see
# `replan.py`'s divergence log for why that distinction matters for how
# a null result should be interpreted), and the manifest/checkpoint
# hashes for provenance.

# %%
import json  # noqa: E402

result = json.loads(pathlib.Path("jetson-orinnx/out/paired_eval/paired_eval_result.json").read_text())
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Get the result file onto your laptop

# %%
files.download("jetson-orinnx/out/paired_eval/paired_eval_result.json")

# %% [markdown]
# ## Alternative: push through the repo instead (optional, off by default)
#
# Commented out deliberately -- commits/pushes are the operator's own
# action in this project, never automatic.

# %%
# get_ipython().system(
#     'git add jetson-orinnx/out/paired_eval/paired_eval_result.json && '
#     'git commit -m "paired_eval.py result (Colab)" && '
#     'git push'
# )
