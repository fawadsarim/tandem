#!/usr/bin/env bash
# jetson/setup.sh -- idempotent environment setup for G1-HW on the Jetson
# Orin Nano Super. Safe to re-run: every step checks current state before
# acting. Run from inside jetson/ on the device itself, AFTER Day-0
# bring-up (README.md section 1) is complete -- this script REFUSES to
# continue if clocks are unlocked (see step 5) rather than silently
# running sudo commands on the user's behalf.
#
# Usage: bash setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# CUDA's bin dir is not reliably on PATH here: this script is normally invoked
# non-interactively (a plain `ssh host "bash setup.sh"`, or similar), and bash
# only sources .bashrc for interactive shells -- relying on the operator's own
# shell rc file being loaded is fragile across invocation styles, so add it
# explicitly and unconditionally instead (found for real on a JetPack 7.2
# device, 2026-07: nvcc exists at /usr/local/cuda/bin but wasn't on PATH under
# a plain SSH command invocation even after adding it to .bashrc).
export PATH="/usr/local/cuda/bin:${PATH}"

VENV_DIR="./venv"
OUT_DIR="./out"

echo "=== [1/6] Platform check ==="
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: /etc/nv_tegra_release not found -- this does not look like an L4T/Jetson device."
    echo "setup.sh will continue (useful for a dry run on a dev host), but versions.json will record"
    echo "this as a non-Jetson environment."
fi
L4T_VERSION=$(cat /etc/nv_tegra_release 2>/dev/null || echo "unknown")
echo "L4T: ${L4T_VERSION}"

echo "=== [2/6] apt dependencies (python3-venv, jetson-stats) ==="
# jetson-stats (jtop) needs apt, not pip, on L4T -- installed system-wide,
# never inside the venv (it talks to a system service, tegrastats).
if ! command -v jtop >/dev/null 2>&1; then
    echo "jtop not found -- installing jetson-stats. This requires a reboot before jtop works (see README"
    echo "Day-0 item 2); setup.sh does NOT reboot for you."
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3-pip
    sudo pip3 install -U jetson-stats
else
    echo "jtop already installed -- skipping."
    sudo apt-get install -y python3-venv python3-pip >/dev/null
fi

echo "=== [3/6] Python venv ==="
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "Created ${VENV_DIR} (--system-site-packages: TensorRT's python bindings and any JetPack-"
    echo "provided torch wheel live outside pip and are only importable this way)."
else
    echo "${VENV_DIR} already exists -- reusing."
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "=== [4/6] pip dependencies ==="
pip install --upgrade pip -q
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt
else
    echo "requirements.txt not found next to setup.sh -- skipping pip install (expected on first clone"
    echo "only if requirements.txt was not committed; this should not happen)."
fi
if "${VENV_DIR}/bin/python3" -c "import torch" >/dev/null 2>&1; then
    echo "torch is importable in this venv (version: $("${VENV_DIR}/bin/python3" -c 'import torch; print(torch.__version__)'))."
else
    echo "NOTE: torch is NOT in requirements.txt for the Jetson venv (by design -- it's"
    echo "aarch64+CUDA-specific per JetPack release and not on PyPI), and isn't importable yet either."
    echo "export_segments.py and paired_eval.py both run on a HOST/Colab, not on this device (see"
    echo "README.md) -- so this is only relevant if you have your own on-device reason to need torch"
    echo "here. If so, install the wheel matching this exact JetPack/Python/CUDA combination (check"
    echo "your JetPack release notes or a community wheel index), then re-run this script -- it will"
    echo "detect an importable torch via --system-site-packages and record its version in versions.json."
fi

echo "=== [5/6] trtexec presence + clock-lock guard ==="
TRTEXEC=/usr/src/tensorrt/bin/trtexec
if [ ! -x "${TRTEXEC}" ]; then
    echo "FATAL: ${TRTEXEC} not found or not executable."
    echo "TensorRT should ship as part of the JetPack SDK component stack (the nvidia-jetpack apt"
    echo "meta-package, not the base L4T/BSP image). Verify with: apt list --installed 2>/dev/null | grep nvidia-jetpack"
    exit 1
fi
echo "Found trtexec: ${TRTEXEC}"

NVPMODEL_OUT=$(nvpmodel -q 2>&1 || true)
JETSON_CLOCKS_OUT=$(sudo jetson_clocks --show 2>&1 || true)
echo "--- nvpmodel -q ---"
echo "${NVPMODEL_OUT}"
echo "--- jetson_clocks --show ---"
echo "${JETSON_CLOCKS_OUT}"

if ! echo "${NVPMODEL_OUT}" | grep -qi "SUPER"; then
    echo ""
    echo "REFUSING TO CONTINUE: nvpmodel does not report a MAXN SUPER-named power mode as active."
    echo "This script will NOT silently run sudo for you (irreversible-ish power-mode/clock changes"
    echo "belong to the operator, not a setup script). Run these yourself, then re-run setup.sh:"
    echo ""
    echo "    sudo nvpmodel -q --verbose        # find the MAXN SUPER mode's index for THIS device"
    echo "    sudo nvpmodel -m <index>          # activate it"
    echo "    sudo jetson_clocks                # lock clocks to that mode's max"
    echo ""
    exit 1
fi
echo "nvpmodel reports a SUPER power mode -- clocks assumed locked. If build_engines.sh or"
echo "measure_lut.py later assert otherwise, re-check with 'jetson_clocks --show'."

echo "=== [6/6] versions.json + output dirs ==="
mkdir -p "${OUT_DIR}/segments" "${OUT_DIR}/engines" "${OUT_DIR}/logs" "${OUT_DIR}/lut" \
         "${OUT_DIR}/replan" "${OUT_DIR}/paired_eval" "${OUT_DIR}/self_test"

TRT_VERSION=$(${TRTEXEC} 2>&1 | grep -m1 -oE "TensorRT v[0-9.]+" || echo "unknown")
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oE "release [0-9.]+" || echo "unknown (nvcc not on PATH)")
PY_VERSION=$(python3 --version 2>&1)
GPU_NAME=$(python3 - <<'PYEOF'
try:
    import torch
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda unavailable to torch")
except Exception as e:
    print(f"torch not importable: {e!r}")
PYEOF
)
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
PIP_FREEZE_HASH=$(pip freeze 2>/dev/null | python3 -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:16])")

python3 - "${OUT_DIR}/versions.json" <<PYEOF
import json, sys, subprocess
out_path = sys.argv[1]
versions = {
    "l4t": """${L4T_VERSION}""".strip(),
    "tensorrt": "${TRT_VERSION}",
    "cuda": "${CUDA_VERSION}",
    "python": "${PY_VERSION}",
    "torch": "${TORCH_VERSION}",
    "gpu_name": "${GPU_NAME}",
    "pip_freeze_hash": "${PIP_FREEZE_HASH}",
    "nvpmodel_q": """${NVPMODEL_OUT}""",
    "jetson_clocks_show": """${JETSON_CLOCKS_OUT}""",
}
with open(out_path, "w") as f:
    json.dump(versions, f, indent=2)
print(f"Wrote {out_path}")
print(json.dumps(versions, indent=2))
PYEOF

echo ""
echo "setup.sh complete. Activate the venv in new shells with: source ${VENV_DIR}/bin/activate"
