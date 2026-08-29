# %% [markdown]
# # jetson-orinnx/pilot_b8_export.py -- Colab runner
#
# G1-HW final phase, Task 2 (batch-8 INT8 pilot) -- see `jetson-orinnx/README.md`'s
# "Batch-8 INT8 pilot" section and `jetson-orinnx/CHARACTERIZATION.md`. Runs
# `pilot_b8_export.py` on Colab's GPU -- deliberately NOT the Jetson.
# **Confirmed for real, 2026-07**: `nvidia-modelopt` (needed for the QDQ
# half) cannot run on the Jetson's own JetPack-specific `torch==2.12.0`
# build at all -- `modelopt.torch`'s own `__init__.py` unconditionally
# imports `torch.distributed.tensor`, which needs a compiled symbol
# (`torch._C._distributed_c10d`) this device's torch build does not have
# (JetPack PyTorch wheels are commonly built with `USE_DISTRIBUTED=0`).
# See `jetson-orinnx/README.md`'s "Batch-8 INT8 pilot" section for the full
# finding -- this is why this step has to run here, not on-device.
#
# This repo is public -- cloning it needs no authentication.
#
# **Prerequisite**: `calib_n197.npy` and `calib_n69.npy`, already captured
# on the Jetson by the completed G1-HW run's `export_segments.py` step
# (`out/segments/calib/` -- gitignored, not in the repo). On your laptop:
# `scp jetson-orinnx:~/tandem/jetson-orinnx/out/segments/calib/calib_n197.npy
# jetson-orinnx:~/tandem/jetson-orinnx/out/segments/calib/calib_n69.npy .` then upload
# both below -- same pattern as `paired_eval_colab.ipynb`'s `plans_hw.csv`
# upload cell.
#
# Runtime: Runtime -> Change runtime type -> any GPU (T4 is enough -- two
# tiny block-segment exports, not a training job).

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
# Same isolated-venv pattern as `export_segments_colab.py`/`paired_eval_colab.py` -- see either
# file's own markdown for the full numpy/pandas ABI-break history this avoids. `nvidia-modelopt`
# and `numpy==1.26.4`/`pandas==2.2.2` are pinned together in ONE pip invocation -- same fix as
# `export_segments_colab.py`'s own QDQ cell, for the same reason (modelopt doesn't pin numpy,
# so a separate install call would let pip's resolver silently bump it past this venv's pin).

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
# ## Upload calib_n197.npy / calib_n69.npy (already captured on the Jetson)

# %%
import pathlib  # noqa: E402
from google.colab import files  # noqa: E402

calib_dir = pathlib.Path("jetson-orinnx/out/segments/calib")
calib_dir.mkdir(parents=True, exist_ok=True)
for fname in ("calib_n197.npy", "calib_n69.npy"):
    target = calib_dir / fname
    if target.exists():
        print(f"{target} already present -- skipping upload.")
        continue
    print(f"Select {fname} from your laptop:")
    uploaded = files.upload()
    assert fname in uploaded, f"Expected a file named exactly {fname!r}, got: {list(uploaded.keys())}"
    target.write_bytes(uploaded[fname])
    print(f"Wrote {target}")

# %% [markdown]
# ## Run pilot_b8_export.py

# %%
_proc = subprocess.Popen(
    [VENV_PYTHON, "-u", "jetson-orinnx/pilot_b8_export.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for _line in _proc.stdout:
    print(_line, end="")
_proc.wait()
assert _proc.returncode == 0, f"pilot_b8_export.py failed (exit code {_proc.returncode}) -- see the real error printed above."

# %% [markdown]
# ## Results

# %%
import json  # noqa: E402

manifest = json.loads(pathlib.Path("jetson-orinnx/out/pilot_b8/pilot_b8_export_manifest.json").read_text())
print(json.dumps(manifest, indent=2))

# %% [markdown]
# ## Get the files onto your laptop (then onto the Jetson)
#
# `scp -r jetson-orinnx/out/pilot_b8 jetson-orinnx:~/tandem/jetson-orinnx/out/` (adjust the host alias to whatever
# your SSH config uses), then run `pilot_b8_build.sh` on the Jetson.

# %%
get_ipython().system('cd jetson-orinnx/out && zip -qr /content/pilot_b8_onnx.zip pilot_b8')  # noqa: F821
files.download("/content/pilot_b8_onnx.zip")
