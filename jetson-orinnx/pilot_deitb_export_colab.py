# %% [markdown]
# # jetson-orinnx/pilot_deitb_export.py -- Colab runner
#
# DeiT-B INT8 PILOT, Part 1 -- see `jetson-orinnx/README.md`'s "DeiT-B INT8
# pilot" section and `jetson-orinnx/out/pilot_deitb/PREREGISTRATION.md` for the
# hypothesis and decision rule. Runs on Colab's GPU -- deliberately NOT
# the Jetson (`nvidia-modelopt` cannot import under this device's
# JetPack-specific torch, confirmed during the batch-8 pilot -- see that
# section of `jetson-orinnx/README.md` for the full investigation).
#
# This repo is public -- cloning it needs no authentication.
# **HF auth** -- add a Colab Secret named `HF_TOKEN`, needed
# to stream 128 calibration images (DeiT-B needs its OWN calibration set
# -- cannot reuse DeiT-S's). Never a literal in any cell.
#
# Runtime: Runtime -> Change runtime type -> any GPU (T4 is enough -- two
# tiny block-segment exports of a single DeiT-B block, not a training job).

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
# ## Isolated venv for `requirements-host.txt` + `nvidia-modelopt`
#
# Same isolated-venv + combined-pin-install pattern as `pilot_b8_export_colab.py`/
# `export_segments_colab.py`'s own QDQ cell -- see either for the numpy/pandas ABI-break
# history this avoids.

# %%
get_ipython().system('pip install -q virtualenv')  # noqa: F821
get_ipython().system('python3 -m virtualenv -q --clear /content/venv_jetson-orinnx_host')  # noqa: F821
get_ipython().system('/content/venv_jetson-orinnx_host/bin/pip install -q -r jetson-orinnx/requirements-host.txt')  # noqa: F821
get_ipython().system('/content/venv_jetson-orinnx_host/bin/pip install -q nvidia-modelopt requests "numpy==1.26.4" "pandas==2.2.2"')  # noqa: F821
VENV_PYTHON = "/content/venv_jetson-orinnx_host/bin/python"

import subprocess  # noqa: E402

_check = subprocess.run(
    [VENV_PYTHON, "-c",
     "import torch, numpy, pandas, modelopt.torch.quantization; "
     "print('TANDEM_ENV_CHECK', torch.__version__, torch.cuda.is_available(), numpy.__version__, pandas.__version__)"],
    capture_output=True, text=True,
)
print(_check.stdout)
if _check.returncode != 0:
    print(_check.stderr[-3000:])
_result_lines = [l for l in _check.stdout.splitlines() if l.startswith("TANDEM_ENV_CHECK")]
assert _result_lines, f"venv import check failed (incl. modelopt.torch.quantization) -- see stderr above.\n{_check.stderr[-3000:]}"
_, _torch_version, _cuda_ok, _numpy_version, _pandas_version = _result_lines[-1].split()
assert _cuda_ok == "True", (
    f"venv torch {_torch_version} cannot see the GPU -- Runtime -> Change runtime type -> a GPU runtime."
)
print(f"torch {_torch_version}, numpy {_numpy_version}, pandas {_pandas_version}, "
      f"CUDA available: True, modelopt.torch.quantization importable: True")

# %% [markdown]
# ## `datasets`/`huggingface_hub` -- DEFAULT Colab kernel, not the venv

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
# ## Materialize DeiT-B's OWN 128-image calibration set
#
# Same mechanism as `export_segments_colab.py`'s calibration cell -- a DIFFERENT model
# (DeiT-B, not DeiT-S) needs its own calibration activations, cannot reuse the existing
# `calib_n197.npy`/`calib_n69.npy` (those were captured from DeiT-S's weights).

# %%
import pathlib  # noqa: E402
from datasets import load_dataset  # noqa: E402

CALIB_DIR = pathlib.Path("/content/calib_imagenet_deitb/class_0")
CALIB_DIR.mkdir(parents=True, exist_ok=True)
N_NEEDED = 128

existing = len(list(CALIB_DIR.glob("*.jpg")))
if existing >= N_NEEDED:
    print(f"{existing} calibration images already present -- skipping re-download.")
else:
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
# ## Run pilot_deitb_export.py

# %%
_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson-orinnx/pilot_deitb_export.py", "--imagenet-dir", str(CALIB_DIR.parent)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"pilot_deitb_export.py failed (exit code {_proc.returncode}) -- see the real error printed above."

# %% [markdown]
# ## Results

# %%
import json  # noqa: E402

manifest = json.loads(pathlib.Path("jetson-orinnx/out/pilot_deitb/pilot_deitb_export_manifest.json").read_text())
print(json.dumps(manifest, indent=2))

# %% [markdown]
# ## Get the files onto your laptop (then onto the Jetson)
#
# `scp -r jetson-orinnx/out/pilot_deitb jetson-orinnx:~/tandem/jetson-orinnx/out/` (adjust the host alias to
# whatever your SSH config uses), then run `pilot_deitb_build.sh` on the Jetson.

# %%
get_ipython().system('cd jetson-orinnx/out && zip -qr /content/pilot_deitb_onnx.zip pilot_deitb --exclude "*/calib/*"')  # noqa: F821
from google.colab import files  # noqa: E402
files.download("/content/pilot_deitb_onnx.zip")
