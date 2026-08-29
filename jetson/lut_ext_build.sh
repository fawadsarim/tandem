#!/usr/bin/env bash
# jetson/lut_ext_build.sh -- ON-DEVICE. LUT EXTENSION, step 2a -- see jetson/README.md's
# "LUT extension" section. FP16-only trtexec build for the 3 new buckets (N=53/37/21) --
# INT8 is CLOSED per CHARACTERIZATION.md, not rebuilt here. Does NOT touch build_engines.sh or
# any registered out/engines/*.engine file -- everything here writes to out/lut_ext/ only.
#
# Same clock-lock refusal-to-run guard as build_engines.sh/pilot_b8_build.sh -- a benchmark on
# unlocked clocks is not trustworthy.
#
# Usage: bash lut_ext_build.sh   (from inside jetson/, venv activated)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TRTEXEC=/usr/src/tensorrt/bin/trtexec
LUT_EXT_DIR="./out/lut_ext"
ENGINES_DIR="${LUT_EXT_DIR}/engines"
LOGS_DIR="${LUT_EXT_DIR}/logs"
WARMUP_MS=500
ITERATIONS=200
BUCKETS="53 37 21"

mkdir -p "${ENGINES_DIR}" "${LOGS_DIR}"

echo "=== Clock-lock guard (same as build_engines.sh) ==="
CLOCKS_SHOW=$(sudo -n jetson_clocks --show 2>&1 || true)
echo "${CLOCKS_SHOW}"
python3 - <<'PYEOF' || { echo "REFUSING TO BUILD: clocks are not locked. Run 'sudo jetson_clocks' first."; exit 1; }
import subprocess, sys, re
out = subprocess.run(["sudo", "-n", "jetson_clocks", "--show"], capture_output=True, text=True).stdout
lines = [l for l in out.splitlines() if "MinFreq" in l and "MaxFreq" in l and "CurrentFreq" in l]
if not lines:
    print("Could not parse jetson_clocks --show output -- treating as unlocked.")
    sys.exit(1)
for l in lines:
    freqs = re.findall(r"(?:Min|Max|Current)Freq=(\d+)", l)
    if len(set(freqs)) > 1:
        print(f"Unlocked rail detected: {l.strip()}")
        sys.exit(1)
sys.exit(0)
PYEOF
echo "Clocks locked -- proceeding."

for N in ${BUCKETS}; do
    ONNX_PATH="${LUT_EXT_DIR}/block_seg_n${N}.onnx"
    if [ ! -f "${ONNX_PATH}" ]; then
        echo "FATAL: expected ONNX segment not found: ${ONNX_PATH}"
        echo "Run lut_ext_export.py first (on-device -- see its own module docstring)."
        exit 1
    fi

    echo ""
    echo "=== bucket n_tokens=${N} (FP16 only -- INT8 CLOSED per CHARACTERIZATION.md) ==="

    FP16_ENGINE="${ENGINES_DIR}/n${N}_fp16.engine"
    FP16_BUILD_LOG="${LOGS_DIR}/n${N}_fp16_build.log"
    FP16_BENCH_LOG="${LOGS_DIR}/n${N}_fp16_bench.log"
    echo "-- FP16 build --"
    "${TRTEXEC}" --onnx="${ONNX_PATH}" --saveEngine="${FP16_ENGINE}" --fp16 \
        > "${FP16_BUILD_LOG}" 2>&1 || { echo "FP16 build FAILED for n=${N} -- see ${FP16_BUILD_LOG}"; exit 1; }
    echo "-- FP16 benchmark (warmUp=${WARMUP_MS}ms, iterations=${ITERATIONS}) --"
    "${TRTEXEC}" --loadEngine="${FP16_ENGINE}" --fp16 --warmUp="${WARMUP_MS}" --iterations="${ITERATIONS}" \
        > "${FP16_BENCH_LOG}" 2>&1 || { echo "FP16 benchmark FAILED for n=${N} -- see ${FP16_BENCH_LOG}"; exit 1; }
    grep -E "Throughput|Latency|median" "${FP16_BENCH_LOG}" | head -5 || true
done

echo ""
echo "lut_ext_build.sh complete. Engines/logs in ${ENGINES_DIR} / ${LOGS_DIR}."
echo "Next: ./venv/bin/python3 lut_ext_measure.py"
