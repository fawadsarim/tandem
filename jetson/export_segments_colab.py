# %% [markdown]
# # jetson/export_segments.py -- Colab runner
#
# Runs `export_segments.py` on Colab's free GPU (this step is deliberately
# NOT run on the Jetson -- see `jetson/README.md`'s "Harness usage"
# section: it's plain PyTorch/ONNX export with no TensorRT/`trtexec`
# dependency, so it runs wherever is easiest, same as the rest of this
# project). Produces `jetson/out/segments/` (5 ONNX block segments +
# calibration `.npy` arrays + `manifest.json`) -- download that at the
# end of this notebook and copy it onto the Jetson before running
# `build_engines.sh` there.
#
# This repo is public -- cloning it needs no authentication.
#
# **HF auth**: add a Colab Secret named `HF_TOKEN` (a Hugging Face
# access token) -- needed to download the `deit_small_patch16_224.fb_in1k`
# checkpoint AND to stream calibration images below. For the latter: this
# token must have accepted the `ILSVRC/imagenet-1k` GATED dataset's terms
# on huggingface.co (visit that dataset's page while logged in, accept,
# then make sure the token has notebook/read access) -- without that,
# the materialization cell below fails with an access-denied error, not a
# missing-token one. Same requirement every experiments/phase2_2_baselines/*_colab.ipynb
# notebook already documents. Never a literal in any cell.
#
# Runtime: Runtime -> Change runtime type -> any GPU (T4 is enough; this
# step is a handful of tiny ONNX exports, not a training job).

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
# `pip install -r jetson/requirements-host.txt` directly into Colab's
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
# `datasets`/`huggingface_hub` (needed below only for calibration image
# streaming, not a `jetson/` dependency) deliberately stay OUT of this
# venv and get installed into the DEFAULT Colab kernel instead (next
# cell) -- so they keep whatever numpy Colab's base image already ships,
# and never have to coexist with this venv's `numpy==1.26.4` pin at all.

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
get_ipython().system('python3 -m virtualenv -q --clear /content/venv_jetson_host')  # noqa: F821
get_ipython().system('/content/venv_jetson_host/bin/pip install -q -r jetson/requirements-host.txt')  # noqa: F821
VENV_PYTHON = "/content/venv_jetson_host/bin/python"

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
# Only used below to stream calibration images off the Hugging Face
# Hub, in the notebook's own (default) kernel -- never imported by
# `export_segments.py` itself, which runs through the isolated venv.

# %%
get_ipython().system('pip install -q datasets huggingface_hub')  # noqa: F821

# %% [markdown]
# ## HF auth (env var only, never a literal)

# %%
HF_TOKEN = userdata.get("HF_TOKEN")
assert HF_TOKEN, "Add an HF_TOKEN Colab Secret -- see markdown above."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# %% [markdown]
# ## Materialize a small local calibration image set
#
# `export_segments.py --imagenet-dir` needs a LOCAL ImageFolder-layout
# directory (not HF streaming directly -- see that script's own
# docstring on why: it mirrors the same local-directory design used
# elsewhere in this folder). Only `N_CALIB_IMAGES=128` are actually used
# (see `export_segments.py`'s own constant) -- this is deliberately a
# small, fast materialization, not the full 50k-image pipeline P2.1
# uses. All 128 are written under one synthetic class folder -- the
# class LABEL is irrelevant here, only the pixels are used, for INT8
# calibration range-setting, not accuracy evaluation.

# %%
import pathlib  # noqa: E402
from datasets import load_dataset  # noqa: E402

CALIB_DIR = pathlib.Path("/content/calib_imagenet/class_0")
CALIB_DIR.mkdir(parents=True, exist_ok=True)
N_NEEDED = 128

existing = len(list(CALIB_DIR.glob("*.jpg")))
if existing >= N_NEEDED:
    print(f"{existing} calibration images already present -- skipping re-download.")
else:
    # "ILSVRC/imagenet-1k", not the bare "imagenet-1k" -- real bug found and fixed, 2026-07: HF
    # moved this dataset under the ILSVRC namespace, and current huggingface_hub/datasets no longer
    # resolve the old bare repo id (fails with HfUriError: "Repository id must be 'namespace/name'").
    # Same repo id already used and proven working in experiments/phase2_2_baselines/common/colab_imagenet.py.
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=HF_TOKEN)
    n_written = 0
    for i, example in enumerate(ds):
        if n_written >= N_NEEDED:
            break
        img = example["image"].convert("RGB")
        img.save(CALIB_DIR / f"img_{n_written:04d}.jpg", quality=95)
        n_written += 1
    print(f"Wrote {n_written} calibration images to {CALIB_DIR.parent}")

# %% [markdown]
# ## Run export_segments.py
#
# Launched BY the isolated venv's python (`VENV_PYTHON`, built above),
# not the default Colab kernel -- so it sees the pinned
# torch/numpy/pandas/timm/onnx stack, untouched by anything installed
# into the default kernel above or below this cell.
#
# Real bug found and fixed, 2026-07 (two rounds): `get_ipython().system(...)` used to silently
# swallow a nonzero exit code -- a crash here used to print its error and then let the notebook
# continue into the Results cell below, which then failed with a confusing `FileNotFoundError`
# on `manifest.json` instead of showing the real error where it happened. The first fix
# (capturing `get_ipython().system(...)`'s return value and asserting it's 0) was ITSELF wrong:
# IPython's `system()` deliberately always returns `None` (it stores the real exit code in
# `_exit_code` on the user namespace instead, specifically so a non-None return doesn't trigger
# cell auto-printing) -- so that assert failed on every run, including a fully successful one.
# Fixed for real by using `subprocess.run(...)` directly instead of the `!`/`.system()` shell
# magic -- a real, documented `.returncode`, no IPython-internal behavior to get wrong. Output
# still streams live to the cell (stdout/stderr aren't captured, they inherit this process's).

# %%
import subprocess  # noqa: E402

_result = subprocess.run([VENV_PYTHON, "jetson/export_segments.py", "--imagenet-dir", str(CALIB_DIR.parent)])
assert _result.returncode == 0, f"export_segments.py failed (exit code {_result.returncode}) -- see the real error printed above."

# %% [markdown]
# ## Results

# %%
import json  # noqa: E402

manifest = json.loads(pathlib.Path("jetson/out/segments/manifest.json").read_text())
print(json.dumps(manifest, indent=2)[:2000])

# %% [markdown]
# ## Get the files onto your laptop (then onto the Jetson)
#
# Downloads `jetson/out/segments/` as a zip. Unzip it into this repo's
# `jetson/out/segments/` on your laptop, then copy it to the Jetson the
# same way `pull_results.sh` pulls things the other direction, e.g.:
# `scp -r jetson/out/segments jetson:~/tandem/jetson/out/` (adjust the
# host alias to whatever your SSH config uses).

# %%
get_ipython().system('cd jetson/out && zip -qr /content/segments.zip segments')  # noqa: F821
from google.colab import files  # noqa: E402
files.download("/content/segments.zip")

# %% [markdown]
# ## REQUIRED: explicit Q/DQ INT8 export, all 5 buckets (production path, reviewed)
#
# **Not optional -- this is now how every INT8 LUT entry is produced.**
# On-device measurement found TensorRT's IMPLICIT INT8 calibration path
# (`int8_calibrate.py`) consistently ~2-2.5x SLOWER than FP16 (see
# `jetson/README.md`'s INT8 findings section for the full ladder: step
# 1's per-layer diagnosis, step 2a's `trtexec --best` result, step 2b's
# explicit-Q/DQ result at N=197 which narrowed the gap to ~1.5x without
# closing it). Confirmed on review: INT8 remains slower even at its best
# -- that finding stands -- but the LUT must price INT8 at its BEST
# achievable implementation, not the naive one, since pricing it
# artificially high inflates the measured cost of every plan that uses
# it (including several of P2.1's own BOPs-selected plans). This cell
# now runs that best-achievable export for ALL 5 token buckets, not
# just N=197 -- needs the calibration data the run cell above already
# produced (`jetson/out/segments/calib/calib_n{197,165,133,101,69}.npy`).
#
# This is genuinely new, unverified-beyond-N=197 code (N=197 alone was
# execution-tested for real; the other 4 buckets haven't been yet) --
# see `int8_qdq_experiment.py`'s own docstring for exactly which
# assumptions it's least sure about.
#
# Real bug found and fixed, 2026-07: `nvidia-modelopt` doesn't pin numpy,
# so installing it as a SEPARATE `pip install` call (as this cell
# originally did) let pip's resolver bump numpy to 2.x inside this venv
# -- the exact same ABI-conflict class the venv-isolation cell already
# fixed once (see its own comment), just triggered again by a later,
# separate install rather than the original shared-kernel problem.
# `numpy==1.26.4`/`pandas==2.2.2` are re-asserted in the SAME pip
# invocation as `nvidia-modelopt`, forcing pip to keep them pinned (or
# fail loudly with a clear resolver conflict, not a silent bump) rather
# than resolve modelopt's own dependencies in isolation. The version
# print right after catches it immediately if this ever needs revisiting.
#
# Real bug found and fixed, 2026-07: `import modelopt.torch.quantization` (below) triggers a
# transitive import chain through `modelopt.torch.utils.dataset_utils`, which unconditionally
# does `import requests` at module level -- but `nvidia-modelopt`'s own package metadata
# doesn't declare `requests` as a hard dependency, so pip never installed it. Adding it
# explicitly here. (If a similar undeclared-dependency error shows up again after this,
# it's the same class of problem -- add whatever's missing to this same line.)

# %%
get_ipython().system(f'{VENV_PYTHON} -m pip install -q nvidia-modelopt requests "numpy==1.26.4" "pandas==2.2.2"')  # noqa: F821

_check2 = subprocess.run(
    [VENV_PYTHON, "-c", "import torch, numpy, pandas; print('TANDEM_QDQ_ENV_CHECK', torch.__version__, numpy.__version__, pandas.__version__)"],
    capture_output=True, text=True,
)
print(_check2.stdout)
if _check2.returncode != 0:
    print(_check2.stderr[-2000:])
assert "TANDEM_QDQ_ENV_CHECK" in _check2.stdout, (
    f"venv import check failed after installing nvidia-modelopt -- see stderr above.\n{_check2.stderr[-2000:]}"
)

# `-u` (unbuffered) + explicit capture-and-print, unlike the export_segments.py/paired_eval.py
# run cells above: those exit cleanly on success, which flushes Python's stdout buffer as part
# of normal interpreter shutdown, so live streaming was never a problem for them. A crash via
# an uncaught exception is not guaranteed to flush a fully-buffered stdout the same way -- real
# symptom hit here: this cell's first real run produced NO visible output at all before the
# AssertionError, even though the script prints its own header line before anything else could
# fail. Capturing explicitly and printing both streams afterward guarantees the real error is
# visible here regardless of buffering or how the subprocess actually exits.
_qdq_result = subprocess.run([VENV_PYTHON, "-u", "jetson/int8_qdq_experiment.py"], capture_output=True, text=True)
print(_qdq_result.stdout)
if _qdq_result.returncode != 0:
    print(_qdq_result.stderr)
assert _qdq_result.returncode == 0, f"int8_qdq_experiment.py failed (exit code {_qdq_result.returncode}) -- see the real error printed above."

# %%
# All 5 buckets' QDQ ONNX files, zipped together -- same pattern as the segments.zip cell
# above. Unzip on your laptop into jetson/out/segments/, then copy all 5 to the Jetson:
# `scp jetson/out/segments/block_seg_n*_qdq.onnx jetson:~/tandem/jetson/out/segments/`
get_ipython().system('cd jetson/out/segments && zip -q /content/qdq_segments.zip block_seg_n*_qdq.onnx')  # noqa: F821
files.download("/content/qdq_segments.zip")

# %% [markdown]
# ## Alternative: push through the repo instead (optional, off by default)
#
# Commented out deliberately -- this project's standing rule is that
# commits/pushes are the operator's own action, never automatic. Uncomment
# only if you specifically want a committed record of these artifacts.

# %%
# get_ipython().system(
#     'git add jetson/out/segments && '
#     'git commit -m "export_segments.py output (Colab)" && '
#     'git push'
# )
