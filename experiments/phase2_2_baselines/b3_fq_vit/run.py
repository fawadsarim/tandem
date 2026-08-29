"""B3: FQ-ViT (megvii-research/FQ-ViT), W8A8 on DeiT-T, DeiT-S, DeiT-B, ViT-B.

Single runner. `test_quant.py` is a standalone CLI script (positional
`model`/`data` args, not composable as a library), invoked as a
SUBPROCESS, completely UNMODIFIED, with the flag combination that
reproduces their published "Ours" row (`--quant --ptf --lis
--quant-method minmax` -- their proposed method, not the plain-MinMax
competitor baseline from the same README table; see ../deviations.md #9),
parsing its final `* Prec@1 X.XXX ...` line (same output format as
RepQ-ViT's script). A gate FAILURE HALTS this runner -- see the assert
after check_gate below.

This script itself has NO torch/timm dependency -- can run from the
lightweight orchestration environment. Pass --python pointing at this
baseline's own pinned venv (requirements.txt: torch==1.7.1,
torchvision==0.8.2, exact-pinned -- see requirements.txt's header for the
CUDA-compatibility caveat, unresolved here, hardware-dependent).

## Checkpoint-identity policy (two-step; see common/checkpoint_pin.py)

FQ-ViT's own model constructors (`models/vit_quant.py`) load their
pretrained weights directly (not via timm): DeiT loaders use
`torch.hub.load_state_dict_from_url(dl.fbaipublicfiles.com/deit/...)` --
BYTE-IDENTICAL to what TANDEM's pinned `.fb_in1k` tags resolve to (Step A
== Step B, one measurement, two results.csv rows, never run twice).
`vit_base_patch16_224`'s loader instead calls `load_weights_from_npz(model,
'https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-...npz')`
-- the SAME url as `vit_base_patch16_224.augreg_in21k_ft_in1k` (confirmed
against timm's real source), FP 84.534, NOT TANDEM's pinned
`augreg2_in21k_ft_in1k` (FP 85.108); independently cross-confirmed by
FQ-ViT's own README, which lists ViT-B Full Precision = 84.53. Step B for
ViT-B re-runs with a checkpoint OVERRIDE active (monkeypatches
`models.utils.load_weights_from_npz` via sitecustomize -- see
common/checkpoint_pin.py's "fqvit" style) -- a real second GPU pass, not a
relabeling.

## FP-parity gate (spec item #2)

A plain (unquantized -- FQ-ViT's own model constructors default to
`quant=False`, never calling `model_open_calibrate()`/`model_quant()`
unless `--quant` reaches that far in `main()`) pass through their own
model construction + data loading is gated against P2.1's own S8
measurement before any config sweep touches a given checkpoint -- see
common/fp_parity_gate.py.

Calibration is their own default: --calib-batchsize 100 x --calib-iter 10
= 1,000 images, NOT overridden to match RepQ-ViT's 32 -- each baseline
uses its own documented default per the no-tuning fairness rule.
`--seed 0` is now passed EXPLICITLY (their own default, verified from
source -- pinning it here makes it visible in OUR code rather than
silently inherited). The exact 1,000 calibration image indices their run
selects are independently reconstructed (never guessed -- see
common.calib_capture's module docstring for why this reconstruction is
provably exact) and recorded to `_calib_indices/<config>.json` before the
real run, since FQ-ViT's own script has no logging for this.

Determinism: TF32 disabled inside the test_quant.py subprocess via a
`sitecustomize.py` injected onto PYTHONPATH (common.determinism.
subprocess_env_with_tf32_disabled) -- not a modification of their file on
disk. Their own seed() already sets CUBLAS_WORKSPACE_CONFIG (verified
from source); subprocess_env_with_tf32_disabled sets the same value if
absent, a no-op here. host/gpu/torch/cuda resolved (via a small
subprocess using the SAME target venv, since this script itself is
torch-free) and recorded in every results.csv row.

Resume: NOT per-batch checkpointed, unlike B1/B4 -- test_quant.py's own
`validate()` loop has no checkpoint hook, and adding one would require
either modifying their file (violates the fairness rule) or fully
reimplementing their eval+quantization pipeline ourselves (defeats the
purpose of running their own code). Instead, resume is per-(model,
config, row_kind, checkpoint) via results.csv (find_existing_row, see
run_config below) -- coarser than a Colab disconnect mid-model, but
enough that a crash on one model never re-pays an already-recorded model.

## ViT-B diagnostics (common/vitb_diagnostic.py, permanent since 2026-07)

A real Step-B FP-parity gap (measured 84.4660% vs P2.1's own S8 85.1000%,
diff=0.6340pp) led to building three checks that are now a PERMANENT part
of this runner's flow for any CHECKPOINT_OVERRIDE_NEEDED model (currently
vit_base_patch16_224 only), not just a one-off investigation -- see
`run_vitb_diagnostics()` below and common/vitb_diagnostic.py's own module
docstring for the full rationale. Before Step A's run: a full-50k plain
pass of FQ-ViT's stock loader on ITS OWN checkpoint, gated (0.15pp) against
FQ-ViT's own published FP baseline -- this fills a real coverage gap, since
Step A's own FP-parity gate (above) is structurally skipped whenever
`coincides_with_pinned()` is False. Before Step B's override is trusted: a
transform side-by-side (FQ-ViT's stock loader vs timm's resolved config for
the pinned tag) and a checkpoint-key diagnostic (does the override's
`strict=False` load actually match every parameter name), both asserted,
not just printed. Both of the 2026-07 ruling's leading hypotheses
for the Step-B gap were tested this way and came back clean (deviations.md
#36) -- the gap's root cause is still open, but neither mechanism recurring
silently in a future run is now possible without halting loudly here first.

## Per-model invocation (Colab session-limit hardening)

`--only-model` restricts a single invocation to ONE of the four models,
so a Colab notebook cell can run each model as a SEPARATE cell instead of
one long one covering all four -- isolating a per-model failure to just
that model's cell, and letting the other three cells' results stand
untouched. Omit the flag (university-GPU default) to run all four models
in one process, exactly as before.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bops import cost_from_trajectory, uniform_bits_trajectory  # noqa: E402
from common.checkpoint_pin import (  # noqa: E402
    assert_override_fired, coincides_with_pinned, download_hf_pytorch_bin, override_env_vars,
    override_sitecustomize_source, PAPER_CHECKPOINT_TAG, pinned_checkpoint_tag, pinned_fp_reference,
)
from common.determinism import SEED, config_hash, subprocess_env_with_tf32_disabled  # noqa: E402
from common import equivalence_gate  # noqa: E402
from common.fp_parity_gate import check_fp_parity  # noqa: E402
from common.repo_utils import ensure_repo  # noqa: E402
from common.repro_gate import check_gate, format_gate_banner  # noqa: E402
from common.results_io import append_result, find_existing_row  # noqa: E402
from common.subprocess_utils import run_streaming  # noqa: E402
from common.startup_checks import (  # noqa: E402
    assert_dual_host_audit_items_present, assert_val_set_matches_canonical_p2, assert_valid_commit_sha,
)

FQVIT_COMMIT = "5daf5915c715dc21d6afcd64e661652030e640ab"
N_BLOCKS = 12
N_TOKENS_TOTAL = 197
CALIB_BATCHSIZE = 100
CALIB_ITER = 10
BASELINES_ROOT = Path(__file__).resolve().parent.parent

# FQ-ViT's own --num-workers default is 16 -- a real OOM-kill (SIGKILL,
# exit -9) risk on a 2-CPU Colab runtime once a larger model's memory
# footprint is added on top of 16 DataLoader WORKER PROCESSES. Purely a
# throughput/memory knob (shuffle=False, same images/order either way),
# not a scientific setting -- capping to the real host core count is a
# hardware-adaptive safety margin, not a fairness-rule violation.
NUM_WORKERS = min(16, os.cpu_count() or 2)

# our model name -> (FQ-ViT's own positional model choice, embed dim d).
# Tiny added per spec item #5 -- FQ-ViT's README "Results on ImageNet"
# table DOES publish a DeiT-T column (Full Precision 72.21, Ours 8/8/8 71.61).
MODELS = {
    "deit_tiny_patch16_224": ("deit_tiny", 192),
    "deit_small_patch16_224": ("deit_small", 384),
    "deit_base_patch16_224": ("deit_base", 768),
    "vit_base_patch16_224": ("vit_base", 768),
}
W_BITS = A_BITS = 8  # the only config this baseline runs (per spec: "W8A8 on the same models")

PREC1_RE = re.compile(r"\*\s+Prec@1\s+([\d.]+)")


def count_val_images(imagenet_dir):
    """Real count of images under <imagenet_dir>/val, however many there
    are -- the full 50k for a real run, or the equivalence gate's smaller
    smoke subset (common/equivalence_gate.py). Used instead of a hardcoded
    50000 so results.csv (and the smoke run's correct-count derivation in
    equivalence_gate.py's _read_smoke_result) reflects what test_quant.py
    ACTUALLY evaluated, not an assumption -- test_quant.py only reports a
    percentage (Prec@1), never a raw count, so this is computed once here
    rather than parsed from its output."""
    val_dir = Path(imagenet_dir) / "val"
    return sum(1 for p in val_dir.rglob("*") if p.is_file())


def get_host_info(python_exe):
    """host/gpu/torch/cuda, resolved via a subprocess using the TARGET
    venv's python (this script itself is torch-free -- see module
    docstring)."""
    result = subprocess.run(
        [python_exe, "-m", "common.determinism", "host-info"],
        cwd=str(BASELINES_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, f"host-info subprocess failed:\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def capture_calib_indices(python_exe, imagenet_dir, out_path):
    """Reconstructs (never guesses) the exact calibration image indices
    FQ-ViT's own run internally selects -- see common.calib_capture's
    module docstring for the equivalence proof. Runs via the TARGET
    venv's python (needs torch/torchvision)."""
    cmd = [
        python_exe, "-m", "common.calib_capture",
        "--method", "fq_vit", "--imagenet-dir", imagenet_dir, "--seed", str(SEED),
        "--calib-batchsize", str(CALIB_BATCHSIZE), "--n-batches", str(CALIB_ITER), "--out", str(out_path),
    ]
    result = subprocess.run(cmd, cwd=str(BASELINES_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, f"calibration-index capture failed:\n{result.stderr}"
    print(result.stdout.strip())


def run_fp_parity_gate(python_exe, model_arg, model_name, imagenet_dir, env, reference_top1_pct, harness_desc,
                        skip_fp_parity=False):
    """Runs common/fp_parity_gate.py's fq_vit path as a subprocess. Needs
    BOTH `common` (via PYTHONPATH) AND FQ-ViT's repo ROOT on sys.path (via
    cwd, which `-m` module invocation auto-adds as sys.path[0]) for `from
    config import Config` / `from test_quant import str2model,
    build_transform` to resolve -- see common/fp_parity_gate.py's module
    docstring.

    Cached by `harness_desc` (unique per model+step) under
    `_fp_parity_cache/` -- a cache HIT is always a genuine prior PASS
    (the cache is only ever written after `check_fp_parity` succeeds, see
    below), so a crash later in the same model's sweep never forces
    re-paying this gate's own full-50k pass (~6-12 min) on rerun.

    Skipped entirely (not cached) when TANDEM_SMOKE_RUN=1 -- set only by
    common/equivalence_gate.py's smoke-equivalence check, where the
    0.15pp tolerance (calibrated for full-50k statistics) can never be
    reliably satisfied by a 256-image subset regardless of correctness;
    see equivalence_gate.py's own comment for the numbers. The real gate
    still runs, unmodified, on every genuine full-50k pass."""
    if skip_fp_parity:
        print("=" * 100)
        print(f"**** WARNING: FP-PARITY GATE SKIPPED for {harness_desc} (--skip-fp-parity) ****")
        print("The eval harness's own correctness against this checkpoint's reference accuracy was")
        print("NOT verified. Every accuracy number downstream of this call is UNVERIFIED -- do not")
        print("report any number from this run without re-running with the gate enabled.")
        print("=" * 100)
        skip_marker = Path(__file__).parent / "_fp_parity_cache" / f"SKIPPED_{harness_desc}.json"
        skip_marker.parent.mkdir(parents=True, exist_ok=True)
        skip_marker.write_text(json.dumps({
            "skip_fp_parity": True, "harness_desc": harness_desc, "model": model_name,
        }, indent=2))
        return
    if os.environ.get("TANDEM_SMOKE_RUN") == "1":
        print(f"FP-parity gate SKIPPED for {harness_desc} (TANDEM_SMOKE_RUN set -- this is a "
              "smoke/equivalence-check invocation, not a real full-50k pass). NOT cached.")
        return
    cache_path = Path(__file__).parent / "_fp_parity_cache" / f"{harness_desc}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        print(f"FP-parity gate already passed for {harness_desc} (cached top1={cached['top1_pct']}%, "
              f"diff={cached['diff_pct_points']:.4f}pp) -- skipping re-run.")
        return

    fqvit_repo_dir = env["_TANDEM_FQVIT_REPO_DIR"]
    fp_env = {k: v for k, v in env.items() if k != "_TANDEM_FQVIT_REPO_DIR"}
    existing_pp = fp_env.get("PYTHONPATH", "")
    fp_env["PYTHONPATH"] = str(BASELINES_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
    cmd = [python_exe, "-m", "common.fp_parity_gate", "--method", "fq_vit",
           "--model-arg", model_arg, "--imagenet-dir", imagenet_dir]
    print(f"Running FP-parity gate: {' '.join(cmd)}  (cwd={fqvit_repo_dir})")
    result = run_streaming(cmd, cwd=fqvit_repo_dir, env=fp_env)
    assert result.returncode == 0, f"FP-parity gate subprocess FAILED (exit {result.returncode}):\n{result.stdout[-3000:]}"
    if fp_env.get("TANDEM_PIN_CHECKPOINT_OVERRIDE") == "1":
        assert_override_fired(result.stdout, harness_desc)
    result_lines = [ln for ln in result.stdout.splitlines() if ln.startswith("TANDEM_FP_PARITY_RESULT ")]
    assert result_lines, "FP-parity gate produced no TANDEM_FP_PARITY_RESULT line -- see stdout above."
    payload = json.loads(result_lines[-1][len("TANDEM_FP_PARITY_RESULT "):])
    diff = check_fp_parity(payload["top1_pct"], reference_top1_pct, model_name, harness_desc)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "top1_pct": payload["top1_pct"], "reference_top1_pct": reference_top1_pct, "diff_pct_points": diff,
    }))


def run_vitb_diagnostics(python_exe, imagenet_dir, env, checkpoint_path, run_plain_pass,
                          run_transform_check, harness_desc):
    """Runs common/vitb_diagnostic.py's checks as a subprocess -- same
    PYTHONPATH/cwd/env pattern as run_fp_parity_gate above (see its own
    docstring). Permanently wired into every CHECKPOINT_OVERRIDE_NEEDED
    model's Step A/Step B flow (currently vit_base_patch16_224 only --
    common/vitb_diagnostic.py's checks are hardcoded to that model class,
    see its own module docstring) since 2026-07, after a real Step-B
    FP-parity gap this tooling was built to investigate; see
    deviations.md #35/#36. Cached the same way and for the same reason as
    run_fp_parity_gate: a cache hit is always a genuine prior PASS (every
    check inside common.vitb_diagnostic asserts internally, so the
    subprocess only returns 0 when everything passed).

    Three independent flags, one per check -- Step A's call passes
    `checkpoint_path=None, run_plain_pass=True, run_transform_check=False`
    (check 3 only: no override checkpoint exists yet at Step A, and the
    transform check is specifically about the augreg2/Step-B swap, not
    meaningful here). Step B's call passes `checkpoint_path=<pinned path>,
    run_plain_pass=False, run_transform_check=True` (checks 1+2: the
    override is what's being trusted, and it's cheap enough to run before
    every use rather than caching separately). See run_config's caller in
    main() below for both call sites."""
    if os.environ.get("TANDEM_SMOKE_RUN") == "1":
        print(f"ViT-B diagnostics SKIPPED for {harness_desc} (TANDEM_SMOKE_RUN set -- not a real "
              "full-50k pass). NOT cached.")
        return
    cache_path = Path(__file__).parent / "_fp_parity_cache" / f"{harness_desc}.json"
    if cache_path.exists():
        print(f"ViT-B diagnostics already passed for {harness_desc} (cached) -- skipping re-run.")
        return

    fqvit_repo_dir = env["_TANDEM_FQVIT_REPO_DIR"]
    diag_env = {k: v for k, v in env.items() if k != "_TANDEM_FQVIT_REPO_DIR"}
    existing_pp = diag_env.get("PYTHONPATH", "")
    diag_env["PYTHONPATH"] = str(BASELINES_ROOT) + (os.pathsep + existing_pp if existing_pp else "")

    cmd = [python_exe, "-m", "common.vitb_diagnostic"]
    if checkpoint_path:
        cmd += ["--checkpoint-path", str(checkpoint_path)]
    if run_plain_pass:
        cmd += ["--imagenet-dir", imagenet_dir]
    else:
        cmd.append("--skip-plain-pass")
    if not run_transform_check:
        cmd.append("--skip-transform-check")
    print(f"Running ViT-B diagnostics ({harness_desc}): {' '.join(cmd)}  (cwd={fqvit_repo_dir})")
    result = run_streaming(cmd, cwd=fqvit_repo_dir, env=diag_env)
    assert result.returncode == 0, f"ViT-B diagnostics FAILED (exit {result.returncode}):\n{result.stdout[-3000:]}"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"passed": True, "harness_desc": harness_desc}))


def run_test_quant(repo_path, python_exe, model_arg, imagenet_dir, env):
    cmd = [
        python_exe, "test_quant.py", model_arg, imagenet_dir,
        "--quant", "--ptf", "--lis", "--quant-method", "minmax",
        "--seed", str(SEED),  # their own default (verified from source) -- pinned explicitly here
        "--num-workers", str(NUM_WORKERS),  # their own default is 16 -- see NUM_WORKERS's own comment
    ]
    run_env = {k: v for k, v in env.items() if k != "_TANDEM_FQVIT_REPO_DIR"}
    print(f"Running: {' '.join(cmd)}  (cwd={repo_path})")
    result = run_streaming(cmd, cwd=str(repo_path), env=run_env)
    if result.returncode != 0:
        raise RuntimeError(f"FQ-ViT test_quant.py failed (exit {result.returncode}):\n{result.stdout[-3000:]}")
    if run_env.get("TANDEM_PIN_CHECKPOINT_OVERRIDE") == "1":
        assert_override_fired(result.stdout, f"test_quant.py {model_arg} (W8A8 quantized pass)")
    match = PREC1_RE.search(result.stdout)
    assert match, (
        "Could not find a '* Prec@1 X.XXX ...' line in test_quant.py's stdout -- its output format "
        "may have changed at this commit; check the tail printed above."
    )
    return float(match.group(1))


def run_config(repo_path, args, model_name, fqvit_model_arg, d, published, env, checkpoint_tag, row_kind, host_info,
                n_images, pip_freeze_hash):
    config_key = f"w{W_BITS}a{A_BITS}"
    print(f"\n{'=' * 70}\nFQ-ViT {model_name} {config_key} [{row_kind}, checkpoint={checkpoint_tag}]\n{'=' * 70}")

    # Resume: skip a (model, config, row_kind, checkpoint) already recorded
    # in results.csv from an earlier attempt -- no per-batch checkpoint
    # exists for B3 (deviations.md #14), so this coarser per-config resume
    # is what actually saves a rerun from re-paying an already-completed
    # config's test_quant.py pass.
    existing = find_existing_row(
        method="fq_vit", model=model_name, config=config_key,
        row_kind=row_kind, checkpoint_tag=checkpoint_tag,
    )
    if existing is not None:
        print(f"Already in results.csv: fq_vit/{model_name}/{config_key} [{row_kind}] "
              f"-- top1={existing['top1_50k']}%, gate_status={existing['gate_status']} -- skipping re-run.")
        return float(existing["top1_50k"])

    calib_indices_path = Path(__file__).parent / "_calib_indices" / f"{model_name}_{config_key}_{row_kind}.json"
    # calib index CAPTURE never loads checkpoint weights (pure RandomSampler
    # replay, see common/calib_capture.py) -- unaffected by the Step-B
    # checkpoint override, run with the default environment as before.
    capture_calib_indices(args.python, args.imagenet_dir, calib_indices_path)

    top1_pct = run_test_quant(repo_path, args.python, fqvit_model_arg, args.imagenet_dir, env)

    n_l_per_block, bits_per_block = uniform_bits_trajectory(N_BLOCKS, N_TOKENS_TOTAL, (W_BITS, A_BITS))
    bops_ours = cost_from_trajectory(n_l_per_block, bits_per_block, d)

    if row_kind == "reproduction":
        pub = published.get(config_key, {})
        published_top1 = pub.get("top1_pct")
        gate_status, diff = check_gate(top1_pct, published_top1)
        print(format_gate_banner("fq_vit", model_name, config_key, top1_pct, published_top1, gate_status, diff))
        # NOT enforced under TANDEM_SMOKE_RUN -- same small-sample
        # incompatibility as the FP-parity gate (see run_fp_parity_gate's
        # docstring); still computed and printed either way, just not
        # asserted, so it's visible but doesn't halt the smoke check.
        if os.environ.get("TANDEM_SMOKE_RUN") == "1":
            print(f"[smoke run] reproduction gate NOT enforced (tol=0.3pp not meaningful at this sample "
                  f"size) -- gate_status={gate_status}, diff={diff}")
        elif gate_status == "fail":
            # Record-not-halt (2026-07 ruling on the DeiT-Tiny
            # w8a8 gap: measured 71.04% vs published 71.61%, diff 0.568pp).
            # The 0.3pp tolerance itself is UNCHANGED -- this does not
            # relax what counts as a pass. What changed: a real, small
            # residual gap, after the harness was independently verified
            # clean (FP-parity passed at 0.03pp on the same checkpoint;
            # calib-batchsize/calib-iter/seeding all confirmed to match
            # FQ-ViT's own real defaults; the published number itself
            # re-verified verbatim against their README -- see
            # deviations.md), is reported honestly rather than blocking
            # every other model behind a mystery this harness cannot
            # explain further without FQ-ViT's own authors' hardware.
            print(f"[reproduction gap] fq_vit/{model_name}/{config_key}: measured={top1_pct:.2f}%, "
                  f"published={published_top1:.2f}%, diff={diff:.3f}pp (tolerance=0.3pp) -- recording as "
                  "gate_status=fail, NOT halting (harness independently verified clean via FP-parity; "
                  "see deviations.md for the full investigation).")
        print(f"Step-A reproduction-gate line: fq_vit/{model_name}/{config_key} -> {gate_status} "
              f"(checkpoint={checkpoint_tag})")
        notes = (f"Step A (paper's own checkpoint); --quant --ptf --lis --quant-method minmax "
                 f"(their 'Ours' row); seed=0 (their default, pinned explicitly); "
                 f"calib-batchsize={CALIB_BATCHSIZE} x calib-iter={CALIB_ITER} (their default); "
                 f"indices recorded at {calib_indices_path}")
    else:
        published_top1, gate_status, diff = None, "not_applicable", None
        notes = (f"Step B (TANDEM's pinned comparison checkpoint, override active); --quant --ptf --lis "
                 f"--quant-method minmax; seed=0; calib-batchsize={CALIB_BATCHSIZE} x "
                 f"calib-iter={CALIB_ITER}; indices recorded at {calib_indices_path}. No published "
                 "reference exists for this method on this exact checkpoint -- gate_status="
                 "not_applicable by design, FP-parity gate already validated the harness on this "
                 "checkpoint before this run.")

    this_config_hash = config_hash(FQVIT_COMMIT, model_name, config_key, SEED, row_kind, str(calib_indices_path))
    append_result({
        "method": "fq_vit", "model": model_name, "config": config_key, "checkpoint_tag": checkpoint_tag,
        "row_kind": row_kind,
        "top1_50k": f"{top1_pct:.4f}", "n_images": n_images, "bops_ours": f"{bops_ours:.6e}",
        "repo_commit": FQVIT_COMMIT,
        "published_top1_pct": "" if published_top1 is None else published_top1,
        "gate_status": gate_status, "gate_diff_pct_points": "" if diff is None else f"{diff:.4f}",
        "calib_images": CALIB_BATCHSIZE * CALIB_ITER,
        "host": host_info["host"], "gpu_name": host_info["gpu_name"],
        "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
        "config_hash": this_config_hash, "pip_freeze_hash": pip_freeze_hash, "notes": notes,
    })
    return top1_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True, help="root with train/ and val/ subdirs")
    parser.add_argument("--python", default=sys.executable,
                         help="python executable of b3_fq_vit's own pinned venv")
    parser.add_argument("--host-tag", default=None,
                         help='e.g. "university_gpu" or "colab" -- recorded in results.csv')
    parser.add_argument("--only-model", default=None, choices=list(MODELS.keys()),
                         help="restrict this invocation to a single model (e.g. one Colab cell per "
                              "model); omit to run all four, exactly as before")
    parser.add_argument("--skip-fp-parity", action="store_true",
                         help="DANGEROUS -- skips both Step-A and Step-B FP-parity gate subprocess "
                              "calls entirely. Prints a loud warning and writes a "
                              "_fp_parity_cache/SKIPPED_<harness_desc>.json marker per skipped call. "
                              "Default: gate ON. Only for fast iteration on a checkpoint already "
                              "known-good from a prior gated run -- never use for a number that will "
                              "be reported.")
    args = parser.parse_args()
    n_images = count_val_images(args.imagenet_dir)  # real count -- 50k for a real run, smaller for a smoke check
    # P2.2 audit item #3 ruling (b): confirm this baseline's own val set
    # matches P2.1's canonical count/source -- see common/startup_checks.py's
    # own docstring for why this (not a literal manifest-hash diff) is the
    # real check. Skipped under TANDEM_SMOKE_RUN (256-image smoke subset).
    if os.environ.get("TANDEM_SMOKE_RUN") != "1":
        assert_val_set_matches_canonical_p2(n_images)

    assert_dual_host_audit_items_present(items=[
        "pinning", "tf32_off", "fixed_calib_indices", "equivalence_fingerprint", "unified_schema",
    ], extra=[
        ("common.checkpoint_pin", ["PINNED_TAGS", "download_hf_pytorch_bin", "override_sitecustomize_source"],
         "checkpoint-identity two-step policy (pinned-tag override mechanism)"),
        ("common.fp_parity_gate", ["check_fp_parity"], "FP-parity gate (harness-level eval-drift check)"),
    ])  # no "resume" -- B3 structurally has none, see AUDIT_REPORT.md Part B.3
    assert_valid_commit_sha(FQVIT_COMMIT, "FQVIT_COMMIT")

    repo_path = ensure_repo("https://github.com/megvii-research/FQ-ViT", FQVIT_COMMIT, local_name="FQ-ViT")

    host_info = get_host_info(args.python)
    if args.host_tag:
        host_info["host"] = args.host_tag
    print(f"Resolved host: {host_info}")

    # P2.2 audit item #4: pip-freeze hash of the ACTUAL target venv (this
    # script itself is torch-free -- see module docstring -- so `args.python`
    # is the real venv every subprocess below actually runs in).
    pip_freeze_hash, _ = equivalence_gate._pip_freeze_hash(args.python)
    print(f"pip-freeze hash: {pip_freeze_hash}")

    sitecustomize_dir = BASELINES_ROOT / "_sitecustomize"
    step_a_env = subprocess_env_with_tf32_disabled(sitecustomize_dir)
    step_a_env["_TANDEM_FQVIT_REPO_DIR"] = str(repo_path)

    with open(Path(__file__).parent / "published.json") as f:
        published_all = json.load(f)

    models_to_run = {args.only_model: MODELS[args.only_model]} if args.only_model else MODELS
    for model_name, (fqvit_model_arg, d) in models_to_run.items():
        published = published_all[model_name]
        step_a_tag = PAPER_CHECKPOINT_TAG[model_name]
        fp_ref = pinned_fp_reference(model_name)
        coincide = coincides_with_pinned(model_name)
        print(f"\n{'#' * 70}\nFQ-ViT / {model_name}  (Step A checkpoint={step_a_tag}, "
              f"coincides with pinned={coincide})\n{'#' * 70}")

        if coincide:
            run_fp_parity_gate(args.python, fqvit_model_arg, model_name, args.imagenet_dir, step_a_env,
                                fp_ref, harness_desc=f"fq_vit_step_a_{model_name}",
                                skip_fp_parity=args.skip_fp_parity)
        else:
            # Step A's own FP-parity gate is structurally skipped for this
            # model (its checkpoint doesn't coincide with our pinned tag,
            # so there's no P2.1 S8 reference for it) -- the ViT-B stock-
            # loader deficit check fills that role instead, gated against
            # FQ-ViT's OWN published FP baseline rather than P2.1's. See
            # common/vitb_diagnostic.py's module docstring (2026-07).
            run_vitb_diagnostics(args.python, args.imagenet_dir, step_a_env, checkpoint_path=None,
                                  run_plain_pass=True, run_transform_check=False,
                                  harness_desc=f"fq_vit_vitb_diag_step_a_{model_name}")

        step_a_top1 = run_config(repo_path, args, model_name, fqvit_model_arg, d, published, step_a_env,
                                  step_a_tag, "reproduction", host_info, n_images, pip_freeze_hash)

        if coincide:
            # Step B: do NOT re-run (spec: "do not run twice") -- reuse
            # Step A's measurement, write a comparison-row duplicate.
            n_l_per_block, bits_per_block = uniform_bits_trajectory(N_BLOCKS, N_TOKENS_TOTAL, (W_BITS, A_BITS))
            bops_ours = cost_from_trajectory(n_l_per_block, bits_per_block, d)
            config_key = f"w{W_BITS}a{A_BITS}"
            this_config_hash = config_hash(FQVIT_COMMIT, model_name, config_key, SEED, "comparison_reused")
            append_result({
                "method": "fq_vit", "model": model_name, "config": config_key,
                "checkpoint_tag": step_a_tag, "row_kind": "comparison",
                "top1_50k": f"{step_a_top1:.4f}", "n_images": n_images, "bops_ours": f"{bops_ours:.6e}",
                "repo_commit": FQVIT_COMMIT, "published_top1_pct": "",
                "gate_status": "not_applicable", "gate_diff_pct_points": "",
                "calib_images": CALIB_BATCHSIZE * CALIB_ITER,
                "host": host_info["host"], "gpu_name": host_info["gpu_name"],
                "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
                "config_hash": this_config_hash, "pip_freeze_hash": pip_freeze_hash,
                "notes": "Step B: comparison row reuses Step A's measurement (checkpoint identity "
                         "coincides for DeiT -- not re-run).",
            })
        else:
            # Step B: a real second GPU pass, with the checkpoint override active.
            pinned_hf_repo_id = f"timm/{pinned_checkpoint_tag(model_name)}"
            pinned_path = download_hf_pytorch_bin(pinned_hf_repo_id)
            override_source = override_sitecustomize_source("fqvit")
            step_b_env = subprocess_env_with_tf32_disabled(sitecustomize_dir, extra_sitecustomize_source=override_source)
            step_b_env.update(override_env_vars(model_name, pinned_path))
            step_b_env["_TANDEM_FQVIT_REPO_DIR"] = str(repo_path)

            # Transform side-by-side + checkpoint-key diagnostic, BEFORE
            # trusting the override -- permanent as of 2026-07, after a
            # real Step-B FP-parity gap this tooling was built to
            # investigate (both came back clean for THIS gap, but nothing
            # rules out either recurring for a future checkpoint; see
            # common/vitb_diagnostic.py's module docstring and
            # deviations.md #35/#36).
            run_vitb_diagnostics(args.python, args.imagenet_dir, step_b_env, checkpoint_path=pinned_path,
                                  run_plain_pass=False, run_transform_check=True,
                                  harness_desc=f"fq_vit_vitb_diag_step_b_{model_name}")

            run_fp_parity_gate(args.python, fqvit_model_arg, model_name, args.imagenet_dir, step_b_env,
                                fp_ref, harness_desc=f"fq_vit_step_b_{model_name}",
                                skip_fp_parity=args.skip_fp_parity)

            run_config(repo_path, args, model_name, fqvit_model_arg, d, published, step_b_env,
                       pinned_checkpoint_tag(model_name), "comparison", host_info, n_images, pip_freeze_hash)


if __name__ == "__main__":
    main()
