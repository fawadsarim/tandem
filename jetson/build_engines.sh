#!/usr/bin/env bash
# jetson/build_engines.sh -- ON-DEVICE. Builds FP16 and INT8 TensorRT
# engines for every ONNX segment export_segments.py produced (one per
# token-count bucket), then benchmarks each via trtexec (--warmUp,
# --iterations>=200), saving a log per engine. measure_lut.py parses these
# logs afterward -- this script's job is build+benchmark, not LUT assembly.
#
# REFUSES to build anything if clocks are not locked (jetson_clocks),
# matching setup.sh's own refusal -- a benchmark run on unlocked clocks is
# not trustworthy and this project does not silently produce numbers it
# doesn't trust.
#
# Usage: bash build_engines.sh   (from inside jetson/, venv activated)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="./venv"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
SEGMENTS_DIR="./out/segments"
ENGINES_DIR="./out/engines"
LOGS_DIR="./out/logs"
WARMUP_MS=500
ITERATIONS=200

mkdir -p "${ENGINES_DIR}" "${LOGS_DIR}"

echo "=== Clock-lock guard ==="
# jetson_clocks --show needs root even in --show (read-only) mode on this JetPack release --
# found for real, 2026-07 (see jetson/setup.sh's matching fix and self_test.py item 3).
CLOCKS_SHOW=$(sudo -n jetson_clocks --show 2>&1 || true)
echo "${CLOCKS_SHOW}"
# jetson_clocks --show prints per-rail "MinFreq == MaxFreq == CurrentFreq" once locked; a
# mismatch anywhere means clocks are not pinned to max. NOTE: the real field name is
# "CurrentFreq", not "CurFreq" -- an earlier version of this check looked for "CurFreq", which
# is not a substring of "CurrentFreq" and so never matched any real line, meaning this guard
# would refuse to build EVERY time regardless of whether clocks were actually locked. Confirmed
# and fixed against real output on a JetPack 7.2 device.
python3 - <<'PYEOF' || { echo "REFUSING TO BUILD: clocks are not locked. Run 'sudo jetson_clocks' first (see README Day-0 item 3)."; exit 1; }
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

if [ ! -f "${SEGMENTS_DIR}/manifest.json" ]; then
    echo "FATAL: ${SEGMENTS_DIR}/manifest.json not found. Run export_segments.py on a HOST first, "
    echo "then copy jetson/out/segments/ to this device."
    exit 1
fi

BUCKETS=$(python3 -c "
import json
m = json.load(open('${SEGMENTS_DIR}/manifest.json'))
print(' '.join(str(s['n_tokens']) for s in m['segments']))
")
echo "Buckets from manifest: ${BUCKETS}"

for N in ${BUCKETS}; do
    ONNX_PATH="${SEGMENTS_DIR}/block_seg_n${N}.onnx"
    if [ ! -f "${ONNX_PATH}" ]; then
        echo "FATAL: expected ONNX segment not found: ${ONNX_PATH}"
        exit 1
    fi

    echo ""
    echo "=== bucket n_tokens=${N} ==="

    # --- FP16 build + benchmark (pure trtexec, no calibration needed) ---
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

    # --- INT8 build (via int8_calibrate.py -- see that file's docstring for why trtexec alone
    #     cannot consume our real, pinned calibration data) + benchmark (via trtexec) ---
    CALIB_NPY="${SEGMENTS_DIR}/calib/calib_n${N}.npy"
    INT8_ENGINE="${ENGINES_DIR}/n${N}_int8.engine"
    INT8_CACHE="${ENGINES_DIR}/n${N}_int8.cache"
    INT8_BUILD_LOG="${LOGS_DIR}/n${N}_int8_build.log"
    INT8_BENCH_LOG="${LOGS_DIR}/n${N}_int8_bench.log"
    if [ ! -f "${CALIB_NPY}" ]; then
        echo "FATAL: ${CALIB_NPY} not found -- re-run export_segments.py on the HOST with "
        echo "--imagenet-dir to produce calibration data, then re-copy jetson/out/segments/ here."
        exit 1
    fi
    echo "-- INT8 build (via int8_calibrate.py, pinned calibration data) --"
    # Explicit venv python, not bare `python3` -- real bug found and fixed, 2026-07:
    # int8_calibrate.py needs torch (see that file's own docstring for why), which is only
    # installed INTO jetson/venv, not system-wide. This script's own usage comment says "venv
    # activated" is a precondition, but relying on the operator having actually done that in
    # their current shell is exactly the kind of thing that's silently bitten this project
    # before -- calling the venv's interpreter by its full path removes the dependency on shell
    # activation state entirely, matching setup.sh's own "${VENV_DIR}/bin/python3" pattern.
    "${VENV_DIR}/bin/python3" int8_calibrate.py --onnx "${ONNX_PATH}" --calib-npy "${CALIB_NPY}" \
        --cache-out "${INT8_CACHE}" --engine-out "${INT8_ENGINE}" \
        > "${INT8_BUILD_LOG}" 2>&1 || { echo "INT8 build FAILED for n=${N} -- see ${INT8_BUILD_LOG}"; exit 1; }
    echo "-- INT8 benchmark (warmUp=${WARMUP_MS}ms, iterations=${ITERATIONS}) --"
    "${TRTEXEC}" --loadEngine="${INT8_ENGINE}" --int8 --warmUp="${WARMUP_MS}" --iterations="${ITERATIONS}" \
        > "${INT8_BENCH_LOG}" 2>&1 || { echo "INT8 benchmark FAILED for n=${N} -- see ${INT8_BENCH_LOG}"; exit 1; }
    grep -E "Throughput|Latency|median" "${INT8_BENCH_LOG}" | head -5 || true
done

echo ""
echo "build_engines.sh complete. Engines in ${ENGINES_DIR}, logs in ${LOGS_DIR}."
echo "Next: python measure_lut.py"
