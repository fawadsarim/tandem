# %% [markdown]
# # jetson/frontier_paired_eval.py -- Colab runner
#
# G1-HW final phase, Task 3 -- see `jetson/README.md`'s "Re-scoped sweep"
# section and `jetson/CHARACTERIZATION.md`. Evaluates `frontier.py`'s two
# arms (ARM A -- TANDEM's own joint DP; ARM B -- uniform-r ToMe) on the
# SAME 20k paired-eval image manifest `paired_eval.py` already uses.
#
# **This is NOT a joint-vs-staged test** -- that comparison is VOID on
# this device/LUT (see `CHARACTERIZATION.md`). No pass/fail gate was
# registered for this comparison; it is purely descriptive.
#
# **REDUCED SCOPE (directed, 2026-07)**: `frontier.py` found ARM
# B (uniform-r ToMe) LATENCY-INFEASIBLE at 4 of the 6 budgets (77/79/82/
# 85% of identity latency -- a real, measured finding about merge
# PLACEMENT granularity, not a bug; see `frontier_paired_eval.py`'s own
# `LATENCY_INFEASIBLE_NOTE`). This run therefore: (1) evaluates ARM A
# (TANDEM's own joint DP) at ALL SIX budgets against the FP-identity
# plan -- this is the frontier curve the thesis needs; (2) evaluates ARM
# B, and the ARM-A-vs-ARM-B paired comparison, ONLY at the 89% and 94%
# budgets, the two where ARM B actually exists.
#
# **Prerequisite**: none beyond what this notebook clones itself --
# unlike `paired_eval_colab.ipynb`, there is nothing to upload here.
# `frontier.py` needs only `out/lut/latency_lut.json` (already committed
# from the completed G1-HW run) and P2.1's own committed
# `experiments/phase2_1_fleet/results/P2/...` tables, both already in the cloned repo, so
# this notebook runs `frontier.py` directly, in-session -- no device, no
# TensorRT.
#
# This repo is public -- cloning it needs no authentication.
# **HF auth** -- add a Colab Secret named `HF_TOKEN`, needed
# to stream the paired-eval image set below (same requirement as
# `paired_eval_colab.ipynb` -- see that notebook for the gated-dataset
# acceptance step). Never a literal in any cell.
#
# Runtime: Runtime -> Change runtime type -> a GPU. This step evaluates
# ARM A at 6 budgets + ARM B at 2 budgets (8 plans total) x 20,000
# images -- lighter than the original all-6-if-feasible design.

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
# ## Isolated venv for `requirements-host.txt`
#
# Same isolated-venv pattern as `paired_eval_colab.py`/`export_segments_colab.py` -- see either
# file's own markdown for the full numpy/pandas ABI-break history this avoids.

# %%
get_ipython().system('pip install -q virtualenv')  # noqa: F821
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
    "continue until this is fixed (Runtime -> Change runtime type -> a GPU runtime)."
)
print(f"torch {_torch_version}, numpy {_numpy_version}, pandas {_pandas_version}, CUDA available: True")

# %% [markdown]
# ## Run frontier.py (no device, no upload -- everything it needs is already in the clone)

# %%
_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson/frontier.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"frontier.py failed (exit code {_proc.returncode}) -- see the real error printed above."

# %% [markdown]
# ## `datasets`/`huggingface_hub` -- DEFAULT Colab kernel, not the venv

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
# ## Materialize the SAME paired-eval image set `paired_eval.py` uses
#
# Identical manifest/split logic as `paired_eval_colab.py`'s own materialization cell --
# reused verbatim (not re-derived) so both scripts partition the SAME 20,512 images the SAME
# deterministic way. If this Colab session already ran `paired_eval_colab.ipynb`, this cell
# just confirms the images are already there and skips re-downloading.

# %%
import pathlib  # noqa: E402
from datasets import load_dataset  # noqa: E402

EVAL_DIR = pathlib.Path("/content/paired_eval_imagenet")
N_NEEDED = 20512  # CALIB_N + PAIRED_EVAL_N, see paired_eval.py's own constants

existing = sum(1 for _ in EVAL_DIR.rglob("*.jpg")) if EVAL_DIR.exists() else 0
if existing >= N_NEEDED:
    print(f"{existing} images already present under {EVAL_DIR} -- skipping re-download.")
else:
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=HF_TOKEN)
    n_written = existing
    for i, example in enumerate(ds):
        if n_written >= N_NEEDED:
            break
        if i < existing:
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
# ## Run frontier_paired_eval.py
#
# Descriptive only -- no pass/fail gate was registered for ARM A vs ARM B. Streams output live
# (`-u` + `Popen` pipe, same fix already applied to `paired_eval_colab.py` -- see that file's
# own comment for why plain `subprocess.run(...)` alone does not actually stream).

# %%
_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson/frontier_paired_eval.py", "--imagenet-dir", str(EVAL_DIR)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"frontier_paired_eval.py failed (exit code {_proc.returncode}) -- see the real error printed above."

# %% [markdown]
# ## Results

# %%
import json  # noqa: E402

result = json.loads(pathlib.Path("jetson/out/frontier/frontier_result.json").read_text())
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Get the result files onto your laptop

# %%
from google.colab import files  # noqa: E402

files.download("jetson/out/frontier/frontier_plans.csv")
files.download("jetson/out/frontier/frontier_manifest.json")
files.download("jetson/out/frontier/frontier_result.json")
