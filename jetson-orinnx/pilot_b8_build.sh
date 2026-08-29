#!/usr/bin/env bash
# jetson-orinnx/pilot_b8_build.sh -- ON-DEVICE. Batch-8 INT8 pilot, step 2 of 3 -- see
# jetson-orinnx/README.md's "Batch-8 INT8 pilot" section. Builds FP16 and explicit-QDQ INT8 engines
# (via trtexec directly -- QDQ mode carries its own baked-in scales, no --calib flag needed,
# same as the original int8_qdq_experiment.py path) for the two pilot buckets only
# (N=197, N=69), at BATCH=8. Does NOT touch build_engines.sh, does NOT touch any registered
# out/engines/*.engine file -- everything here writes to out/pilot_b8/ only.
#
# Same clock-lock refusal-to-run guard as build_engines.sh -- a benchmark on unlocked clocks
# is not trustworthy.
#
# Usage: bash pilot_b8_build.sh   (from inside jetson-orinnx/, venv activated)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TRTEXEC=/usr/src/tensorrt/bin/trtexec
PILOT_DIR="./out/pilot_b8"
WARMUP_MS=500
ITERATIONS=200
BUCKETS="197 69"

mkdir -p "${PILOT_DIR}"

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
    FP_ONNX="${PILOT_DIR}/block_seg_n${N}_b8_fp.onnx"
    QDQ_ONNX="${PILOT_DIR}/block_seg_n${N}_b8_qdq.onnx"
    for f in "${FP_ONNX}" "${QDQ_ONNX}"; do
        if [ ! -f "${f}" ]; then
            echo "FATAL: expected ONNX not found: ${f}"
            echo "Run pilot_b8_export.py on a HOST/Colab GPU first, then copy jetson-orinnx/out/pilot_b8/*.onnx here."
            exit 1
        fi
    done

    echo ""
    echo "=== bucket n_tokens=${N}, batch=8 ==="

    # --- FP16 build + benchmark ---
    FP16_ENGINE="${PILOT_DIR}/n${N}_b8_fp16.engine"
    FP16_BUILD_LOG="${PILOT_DIR}/n${N}_b8_fp16_build.log"
    FP16_BENCH_LOG="${PILOT_DIR}/n${N}_b8_fp16_bench.log"
    echo "-- FP16 build --"
    "${TRTEXEC}" --onnx="${FP_ONNX}" --saveEngine="${FP16_ENGINE}" --fp16 \
        > "${FP16_BUILD_LOG}" 2>&1 || { echo "FP16 build FAILED for n=${N} -- see ${FP16_BUILD_LOG}"; exit 1; }
    echo "-- FP16 benchmark (warmUp=${WARMUP_MS}ms, iterations=${ITERATIONS}) --"
    "${TRTEXEC}" --loadEngine="${FP16_ENGINE}" --fp16 --warmUp="${WARMUP_MS}" --iterations="${ITERATIONS}" \
        > "${FP16_BENCH_LOG}" 2>&1 || { echo "FP16 benchmark FAILED for n=${N} -- see ${FP16_BENCH_LOG}"; exit 1; }
    grep -E "Throughput|Latency|median" "${FP16_BENCH_LOG}" | head -5 || true

    # --- Explicit-QDQ INT8 build (no --calib -- graph carries its own baked-in scales, same
    #     as int8_qdq_experiment.py's production path) + per-layer graph dump (confirms fused
    #     INT8 kernels, no Reformat bracketing -- same check as the original N=197 diagnosis)
    #     + benchmark ---
    INT8_ENGINE="${PILOT_DIR}/n${N}_b8_int8_qdq.engine"
    INT8_BUILD_LOG="${PILOT_DIR}/n${N}_b8_int8_qdq_build.log"
    INT8_BENCH_LOG="${PILOT_DIR}/n${N}_b8_int8_qdq_bench.log"
    LAYERINFO="${PILOT_DIR}/n${N}_b8_qdq_layerinfo.json"
    echo "-- explicit-QDQ INT8 build (no --calib, per-layer graph dump) --"
    "${TRTEXEC}" --onnx="${QDQ_ONNX}" --int8 --saveEngine="${INT8_ENGINE}" \
        --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo="${LAYERINFO}" \
        > "${INT8_BUILD_LOG}" 2>&1 || { echo "QDQ INT8 build FAILED for n=${N} -- see ${INT8_BUILD_LOG}"; exit 1; }
    echo "-- explicit-QDQ INT8 benchmark (warmUp=${WARMUP_MS}ms, iterations=${ITERATIONS}) --"
    "${TRTEXEC}" --loadEngine="${INT8_ENGINE}" --int8 --warmUp="${WARMUP_MS}" --iterations="${ITERATIONS}" \
        > "${INT8_BENCH_LOG}" 2>&1 || { echo "QDQ INT8 benchmark FAILED for n=${N} -- see ${INT8_BENCH_LOG}"; exit 1; }
    grep -E "Throughput|Latency|median" "${INT8_BENCH_LOG}" | head -5 || true
done

echo ""
echo "pilot_b8_build.sh complete. Engines/logs in ${PILOT_DIR}."
echo "Next: ./venv/bin/python3 pilot_b8_measure.py"
