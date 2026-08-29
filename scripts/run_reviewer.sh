#!/usr/bin/env bash
# run_reviewer.sh -- orchestrates docs/REVIEWER_RUNBOOK.md.
#
# --tier1 (default, no flags needed): runs every Tier-1 read-only
#   verification step in order, stops at the first failure. No GPU, no
#   network, no secrets, no venv beyond `pip install pandas numpy`. This
#   is the mode a fresh clone with no GPU should always be able to run
#   end-to-end.
#
# --tier2 --stage <name>: Tier 2 is real GPU re-measurement across
#   several separate pinned venvs, real ImageNet/HF access, and (for
#   E1-E3/E5/P2.1) Colab as the primary execution environment -- none of
#   which this script can safely auto-provision or blindly execute
#   without real hardware and credentials actually present. For stages
#   with a plain, local, GPU-optional-or-required Python entry point
#   (B1-B5, E4a), this script attempts to run them IF the obvious
#   prerequisites look present, and prints exactly what's missing and
#   exactly what command to run otherwise. For Colab-primary stages
#   (E1, E2, E3, E5, P2.1), it prints the exact instructions from
#   docs/REVIEWER_RUNBOOK.md rather than pretending to automate a Colab
#   session from a shell script.
#
# Usage:
#   bash run_reviewer.sh                       # Tier 1, default
#   bash run_reviewer.sh --tier1               # same, explicit
#   bash run_reviewer.sh --tier2 --stage B1     # Tier 2, one stage
#   bash run_reviewer.sh --tier2 --stage list   # list valid stage names
set -uo pipefail
# This script moved from repo root to scripts/ during the public-release
# reorg -- one more ".." than before to land back at repo root, where
# every path below (experiments/, jetson/) is still relative to.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="tier1"
STAGE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --tier1) MODE="tier1"; shift ;;
        --tier2) MODE="tier2"; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

run_tier1() {
    local failures=0
    echo "======================================================================"
    echo "TIER 1 -- verify from artifacts (no GPU)"
    echo "======================================================================"

    echo; echo "--- [1/8] P2.1 headline claims ---"
    python scripts/verify_p2_claims.py || failures=$((failures + 1))

    echo; echo "--- [2/8] E1-E5/E4a (Phase 1, G1 kill test) ---"
    python scripts/verify_g1_claims.py || failures=$((failures + 1))

    echo; echo "--- [3/8] P2.2 (baselines B1-B5) headline claims ---"
    python scripts/verify_baselines_claims.py || failures=$((failures + 1))

    echo; echo "--- [4/8] Cost-ruler unit checks ---"
    (cd experiments/phase2_2_baselines && python -m common.bops_identity_check) || failures=$((failures + 1))

    echo; echo "--- [5/8] B4 closure re-derivation ---"
    (cd experiments/phase2_2_baselines/b4_sequential_composites && python analyze_deit_small.py) || failures=$((failures + 1))

    echo; echo "--- [6/8] B5 final closure re-derivation ---"
    (cd experiments/phase2_2_baselines && python b5_final_closure.py) || failures=$((failures + 1))

    echo; echo "--- [7/8] G1-HW (jetson/) static self-test ---"
    (cd jetson && python self_test.py) || failures=$((failures + 1))

    echo; echo "--- [8/8] G1-HW-2 (jetson-orinnx/) headline re-derivation + static self-test ---"
    (cd jetson-orinnx && python verify_orinnx_claims.py && python self_test.py) || failures=$((failures + 1))

    echo
    echo "======================================================================"
    if [ "$failures" -eq 0 ]; then
        echo "TIER 1: ALL 8 STEPS PASSED."
        echo "======================================================================"
        return 0
    else
        echo "TIER 1: $failures STEP(S) FAILED -- see [FAIL]/traceback output above."
        echo "Do not cite any number from this repo until diagnosed."
        echo "======================================================================"
        return 1
    fi
}


print_stage_list() {
    cat <<'EOF'
Valid --stage names:
  E1, E2, E3, E4, E4a, E5      (Phase 1 -- Colab-primary, this script prints instructions)
  P2.1                          (4-model fleet -- Colab-primary, this script prints instructions)
  B1, B2, B3, B4                (baselines -- this script attempts to run them if venvs/env look ready)
  B5                             (compare_pareto + b5_final_closure -- no GPU, this script runs them)
  jetson                          (G1-HW -- see jetson/README.md directly, real hardware required)
  jetson-orinnx                   (G1-HW-2 -- see jetson-orinnx/README.md directly, real hardware required)
See docs/REVIEWER_RUNBOOK.md's Tier 2 section for full per-stage detail (secrets/env,
expected numbers + tolerance class, resume behavior, legitimate-deviation guidance).
EOF
}

run_tier2() {
    if [ -z "$STAGE" ]; then
        echo "ERROR: --tier2 requires --stage <name>." >&2
        print_stage_list
        exit 2
    fi
    if [ "$STAGE" = "list" ]; then
        print_stage_list
        exit 0
    fi

    case "$STAGE" in
        E1|E2|E3|E5)
            echo "$STAGE is Colab-primary. This script does not drive a Colab session."
            echo "Open experiments/phase1_E1-E5_E4a/${STAGE}_*.ipynb in Colab, Runtime -> Run all."
            echo "Requires: HF_TOKEN (Colab Secret or env)."
            echo "See docs/REVIEWER_RUNBOOK.md's '$STAGE' section for expected output + tolerance class."
            ;;
        E4a)
            echo "E4a needs no GPU (pure numerical proof) -- attempting a local run."
            if command -v jupytext >/dev/null 2>&1; then
                python experiments/phase1_E1-E5_E4a/E4a_ward_verification.py
            else
                echo "jupytext not found in this environment -- run via Colab instead:"
                echo "  experiments/phase1_E1-E5_E4a/E4a_ward_verification.ipynb, Runtime -> Run all."
            fi
            ;;
        P2.1)
            echo "P2.1 is Colab-primary, run once per model (edit CONFIG[\"MODEL_NAME\"])."
            echo "Open experiments/phase2_1_fleet/P2_tandem_pipeline.ipynb in Colab, Runtime -> Run all."
            echo "Requires: HF_TOKEN. Recommended order: deit_small_patch16_224 first (ARCHIVED"
            echo "mode, fastest), then deit_tiny_patch16_224, deit_base_patch16_224, vit_base_patch16_224."
            echo "See docs/REVIEWER_RUNBOOK.md's 'P2.1' section for expected output + tolerance class."
            ;;
        B1)
            run_baseline_stage "b1_tome" "python -m venv .venv && .venv/bin/pip install -r requirements.txt" \
                ".venv/bin/python run.py --imagenet-dir \$IMAGENET_DIR --host-tag manual_tier2"
            ;;
        B2)
            run_baseline_stage_orch "b2_repq_vit"
            ;;
        B3)
            run_baseline_stage_orch "b3_fq_vit"
            ;;
        B4)
            run_baseline_stage "b4_sequential_composites" \
                "python -m venv .venv && .venv/bin/pip install -r requirements.txt" \
                ".venv/bin/python run.py --imagenet-dir \$IMAGENET_DIR --host-tag manual_tier2"
            ;;
        B5)
            echo "B5 needs no GPU -- running directly."
            (cd experiments/phase2_2_baselines && python compare_pareto.py && python b5_final_closure.py)
            ;;
        jetson)
            echo "G1-HW requires real Jetson Orin Nano Super hardware -- see jetson/README.md in full"
            echo "(Day-0 bring-up, harness usage, per-stage runtimes, thermal guard, resume behavior)."
            echo "Static, no-hardware check: cd jetson && python self_test.py"
            ;;
        jetson-orinnx)
            echo "G1-HW-2 requires real Jetson Orin NX 8GB hardware -- see jetson-orinnx/README.md in full"
            echo "(Day-0 bring-up, power-mode audit, sysfs gate, canonical Run 3 LUT, energy instrument)."
            echo "Static, no-hardware check: cd jetson-orinnx && python verify_orinnx_claims.py && python self_test.py"
            ;;
        *)
            echo "Unknown stage: $STAGE" >&2
            print_stage_list
            exit 2
            ;;
    esac
}

run_baseline_stage() {
    local dir="$1" setup_cmd="$2" run_cmd="$3"
    echo "Stage: $dir"
    if [ -z "${IMAGENET_DIR:-}" ]; then
        echo "IMAGENET_DIR is not set. Required: a local ImageFolder-layout ImageNet-1k copy."
        echo "  export IMAGENET_DIR=/path/to/imagenet"
        echo "Then, from experiments/phase2_2_baselines/:"
        echo "  cd $dir && $setup_cmd"
        echo "  $run_cmd"
        return 1
    fi
    if [ ! -d "experiments/phase2_2_baselines/$dir/.venv" ]; then
        echo "experiments/phase2_2_baselines/$dir/.venv not found. Set it up first:"
        echo "  cd experiments/phase2_2_baselines/$dir && $setup_cmd"
        return 1
    fi
    echo "IMAGENET_DIR and .venv found -- running."
    (cd "experiments/phase2_2_baselines/$dir" && eval "$run_cmd")
}

run_baseline_stage_orch() {
    local dir="$1"
    echo "Stage: $dir (subprocess-orchestrated -- needs both its own venv AND experiments/phase2_2_baselines/.venv_orch)"
    if [ -z "${IMAGENET_DIR:-}" ]; then
        echo "IMAGENET_DIR is not set. Required: a local ImageFolder-layout ImageNet-1k copy."
        echo "  export IMAGENET_DIR=/path/to/imagenet"
    fi
    if [ ! -d "experiments/phase2_2_baselines/$dir/.venv" ] || [ ! -d "experiments/phase2_2_baselines/.venv_orch" ]; then
        echo "Missing venv(s). Set up both:"
        echo "  cd experiments/phase2_2_baselines/$dir && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
        echo "  cd experiments/phase2_2_baselines && python -m venv .venv_orch && .venv_orch/bin/pip install -r requirements.txt"
        echo "Then:"
        echo "  experiments/phase2_2_baselines/.venv_orch/bin/python experiments/phase2_2_baselines/$dir/run.py --imagenet-dir \$IMAGENET_DIR \\"
        echo "    --python experiments/phase2_2_baselines/$dir/.venv/bin/python --host-tag manual_tier2"
        return 1
    fi
    echo "Venvs and IMAGENET_DIR found -- running."
    (cd experiments/phase2_2_baselines && .venv_orch/bin/python "$dir/run.py" --imagenet-dir "$IMAGENET_DIR" \
        --python "$dir/.venv/bin/python" --host-tag manual_tier2)
}

if [ "$MODE" = "tier1" ]; then
    run_tier1
    exit $?
else
    run_tier2
    exit $?
fi
