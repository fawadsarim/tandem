# %% [markdown]
# # jetson/frontier_v2_paired_eval.py -- Colab runner
#
# LUT EXTENSION, step 6 -- see `jetson/README.md`'s "LUT extension" section
# and `jetson/CHARACTERIZATION.md`. Evaluates ARM A vs ARM B, paired, at
# EVERY budget where `frontier_v2.py` found both now feasible against the
# extended LUT (real result: 82%, 85%, 89%, 94% -- 77%/79% remain
# infeasible for ARM B even at the extended range's maximum r).
#
# **This is NOT a joint-vs-staged test** -- that comparison is VOID on
# this device/LUT (see `CHARACTERIZATION.md`). No pass/fail gate was
# registered for this comparison; it is purely descriptive.
#
# **Prerequisite**: none beyond what this notebook clones itself --
# `frontier_v2.py` needs only `out/lut_ext/latency_lut_extended.json` and
# `out/frontier/frontier_plans.csv` (both already committed), so this
# notebook runs it directly, in-session -- no device, no upload.
#
# This repo is public -- cloning it needs no authentication.
# **HF auth** -- add a Colab Secret named `HF_TOKEN`, needed
# to stream/reuse the paired-eval image set below (same requirement as
# `paired_eval_colab.ipynb`). Never a literal in any cell.
#
# Runtime: Runtime -> Change runtime type -> a GPU. This evaluates ARM A
# vs ARM B at up to 4 budgets (8 plan evaluations) x 20,000 images.

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
    f"venv torch {_torch_version} cannot see the GPU -- Runtime -> Change runtime type -> a GPU runtime."
)
print(f"torch {_torch_version}, numpy {_numpy_version}, pandas {_pandas_version}, CUDA available: True")

# %% [markdown]
# ## Run frontier_v2.py (no device, no upload -- everything it needs is already in the clone)

# %%
_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson/frontier_v2.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"frontier_v2.py failed (exit code {_proc.returncode}) -- see the real error printed above."

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

# %%
import pathlib  # noqa: E402
from datasets import load_dataset  # noqa: E402

EVAL_DIR = pathlib.Path("/content/paired_eval_imagenet")
N_NEEDED = 20512

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
# ## Run frontier_v2_paired_eval.py
#
# Descriptive only -- no pass/fail gate. Streams output live (`-u` + `Popen` pipe).

# %%
_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson/frontier_v2_paired_eval.py", "--imagenet-dir", str(EVAL_DIR)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"frontier_v2_paired_eval.py failed (exit code {_proc.returncode}) -- see the real error printed above."

# %% [markdown]
# ## Results

# %%
import json  # noqa: E402

result = json.loads(pathlib.Path("jetson/out/frontier_v2/frontier_result_v2.json").read_text())
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Get the result files onto your laptop

# %%
from google.colab import files  # noqa: E402

files.download("jetson/out/frontier_v2/frontier_plans_v2.csv")
files.download("jetson/out/frontier_v2/arm_b_r_sweep_v2.csv")
files.download("jetson/out/frontier_v2/frontier_manifest_v2.json")
files.download("jetson/out/frontier_v2/frontier_result_v2.json")
