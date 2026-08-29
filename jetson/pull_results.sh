#!/usr/bin/env bash
# jetson/pull_results.sh -- runs on YOUR OWN machine (the laptop), NOT on
# the Jetson. Pulls jetson/out/ (everything build_engines.sh/measure_lut.py/
# replan.py/self_test.py produce -- engines, logs, the latency LUT,
# plans_hw.csv, versions.json) down from the Jetson into this local
# checkout, so you can inspect/analyze results here without needing to be
# SSH'd into the device or use a monitor.
#
# Uses scp (not rsync -- not installed on all operator machines by
# default, e.g. Windows Git Bash) so a re-run re-copies everything rather
# than doing an incremental sync; fine for this folder's actual size
# (engine files, logs, a handful of JSON/CSV -- not a large dataset).
#
# Usage:
#   bash pull_results.sh                # pulls from the 'jetson' SSH alias
#   bash pull_results.sh user@host      # or an explicit target
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REMOTE="${1:-jetson}"
REMOTE_OUT="~/tandem/jetson/out"

echo "Pulling ${REMOTE}:${REMOTE_OUT} -> $(pwd)/out ..."

if ! ssh -o BatchMode=yes -o ConnectTimeout=6 "${REMOTE}" "test -d ${REMOTE_OUT}" 2>/dev/null; then
    echo "Nothing to pull yet -- ${REMOTE_OUT} doesn't exist on ${REMOTE}."
    echo "(Expected before setup.sh/build_engines.sh/measure_lut.py/replan.py have run.)"
    exit 0
fi

mkdir -p out
scp -r -q "${REMOTE}:${REMOTE_OUT}/*" out/ 2>/dev/null || {
    # scp with a glob fails if the remote dir has exactly one entry in some
    # shells -- fall back to copying the directory itself in that case.
    scp -r -q "${REMOTE}:${REMOTE_OUT}" .tmp_out_pull && \
        cp -r .tmp_out_pull/* out/ && rm -rf .tmp_out_pull
}

echo "Done. Local contents:"
find out -type f | sort
