"""B2: RepQ-ViT (zkkli/RepQ-ViT), W8A8 and W6A6 PTQ on DeiT-T, DeiT-S,
DeiT-B, ViT-B.

Single runner. RepQ-ViT's `classification/test_quant.py` is a standalone
CLI script (not a composable library -- it does `from utils import *` /
`from quant import *` at module scope and drives everything from
`if __name__ == "__main__"`), so this runner invokes it as a SUBPROCESS,
completely UNMODIFIED, with its exact documented flags (verified against
the pinned commit's actual source, not just README prose -- the README's
usage snippet shows `--w_bit`/`--a_bit`, singular, which does NOT match
the real `--w_bits`/`--a_bits` argparse names; see ../deviations.md #6)
and parses its final `* Prec@1 X.XXX ...` line. A gate FAILURE (not
"no_published_reference") HALTS this runner -- see the assert after
check_gate below.

This script itself has NO torch/timm dependency (subprocess + stdlib
only) -- it can run from the lightweight orchestration environment. Pass
--python pointing at THIS baseline's own pinned venv (requirements.txt:
torch==1.12.1, timm==0.4.12, exact-pinned) for the actual test_quant.py
subprocess call.

## Checkpoint-identity policy (two-step; see common/checkpoint_pin.py)

RepQ-ViT's own `build_model(name)` calls `timm.create_model(name,
pretrained=True)` unmodified -- for DeiT this resolves to the SAME
fb_in1k weights TANDEM pins (Step A == Step B, verified in
common/checkpoint_pin.py's module docstring): one measurement, two
results.csv rows (row_kind="reproduction" then "comparison"), never run
twice. For ViT-B it does NOT: their `--model vit_base` resolves to
`vit_base_patch16_224.augreg_in21k_ft_in1k` (FP 84.534, confirmed both
from timm's real source AND from RepQ-ViT's own README, which lists
"ViT-B (84.54)") -- NOT TANDEM's pinned `augreg2_in21k_ft_in1k` (FP
85.108). Step A runs unmodified (their real checkpoint, gated against
their real published numbers, UNCHANGED from this runner's prior
behavior). Step B re-runs the SAME method settings with a checkpoint
OVERRIDE active (see common/checkpoint_pin.py) so the resulting numbers
are actually comparable to TANDEM's own Pareto curve -- a real second GPU
pass, not a relabeling.

## FP-parity gate (spec item #2)

Before Step B's config sweep touches a checkpoint that differs from
Step A's (currently: ViT-B only), and before Step A's sweep for any model
where Step A IS the pinned checkpoint (currently: every DeiT), a plain
(unquantized) pass through RepQ-ViT's own `build_model()` + `build_dataset()`
on the full 50k set is gated against P2.1's own S8 measurement for that
exact checkpoint (see common/fp_parity_gate.py) -- catches a harness-level
eval-loop/transform drift before any method number is trusted.

Calibration is exactly one batch of 32 images from the training split
(`--calib-batchsize 32`, their default, verified from source) -- not
overridden, per the "32-image calibration per their protocol" spec and
the no-tuning fairness rule. `--seed 0` is now passed EXPLICITLY (their
own default, verified from source -- pinning it here makes it visible in
OUR code rather than silently inherited). The exact 32 calibration image
indices their run selects are independently reconstructed (never guessed
-- see common.calib_capture's docstring for why this reconstruction is
provably exact, not just plausible) and recorded to
`_calib_indices/<config>.json` before the real run, since RepQ-ViT's own
script has no logging for this.

Determinism: TF32 disabled inside the test_quant.py subprocess via a
`sitecustomize.py` injected onto PYTHONPATH (common.determinism.
subprocess_env_with_tf32_disabled) -- not a modification of their file on
disk. host/gpu/torch/cuda resolved (via a small subprocess using the
SAME target venv, since this script itself is torch-free) and recorded
in every results.csv row.

Resume: NOT per-batch checkpointed, unlike B1/B4 -- test_quant.py's own
`validate()` loop has no checkpoint hook, and adding one would require
either modifying their file (violates the fairness rule) or fully
reimplementing their eval+quantization pipeline ourselves (defeats the
purpose of running their own code). Instead, resume is per-(model,
config, row_kind, checkpoint) via results.csv (find_existing_row, see
run_bit_sweep below) -- coarser than a Colab disconnect mid-model, but
enough that a crash on one model (e.g. an OOM on DeiT-Base -- see
deviations.md #24) never re-pays an already-recorded model or config.

## Per-model invocation (Colab session-limit hardening)

`--only-model` restricts a single invocation to ONE of the four models,
so a Colab notebook cell can run each model as a SEPARATE cell instead of
one long one covering all four -- isolating a per-model failure (the
DeiT-Base OOM in deviations.md #24 happened after DeiT-Tiny/Small had
already succeeded) to just that model's cell, and letting the other three
cells' results stand untouched. Omit the flag (university-GPU default) to
run all four models in one process, exactly as before.
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
    override_sitecustomize_source,
    PAPER_CHECKPOINT_TAG, pinned_checkpoint_tag, pinned_fp_reference,
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

REPQ_COMMIT = "a160561b9233945e6680132f129d13041b787318"
N_BLOCKS = 12
N_TOKENS_TOTAL = 197
CALIB_BATCHSIZE = 32  # their default; see module docstring
BASELINES_ROOT = Path(__file__).resolve().parent.parent

# RepQ-ViT's own --num-workers default is 16 -- a real OOM-kill (SIGKILL,
# exit -9) risk on a 2-CPU Colab runtime once a larger model's memory
# footprint (quantizer wrapper state, not just the plain weights) is
# added on top of 16 DataLoader WORKER PROCESSES. Purely a throughput/
# memory knob (shuffle=False, same images/order either way), not a
# scientific setting -- capping to the real host core count is a
# hardware-adaptive safety margin, not a fairness-rule violation.
NUM_WORKERS = min(16, os.cpu_count() or 2)

# our model name -> (RepQ-ViT's own --model choice, embed dim d). Tiny added
# per spec item #5 -- RepQ-ViT's README DOES publish a DeiT-T row (W6/A6
# only, same "no W8/A8 located" gap as the other three models).
MODELS = {
    "deit_tiny_patch16_224": ("deit_tiny", 192),
    "deit_small_patch16_224": ("deit_small", 384),
    "deit_base_patch16_224": ("deit_base", 768),
    "vit_base_patch16_224": ("vit_base", 768),
}
BIT_CONFIGS = [(8, 8), (6, 6)]

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
    RepQ-ViT's own run internally selects -- see common.calib_capture's
    module docstring for the equivalence proof. Runs via the TARGET
    venv's python (needs torch/torchvision)."""
    cmd = [
        python_exe, "-m", "common.calib_capture",
        "--method", "repq_vit", "--imagenet-dir", imagenet_dir, "--seed", str(SEED),
        "--calib-batchsize", str(CALIB_BATCHSIZE), "--n-batches", "1", "--out", str(out_path),
    ]
    result = subprocess.run(cmd, cwd=str(BASELINES_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, f"calibration-index capture failed:\n{result.stderr}"
    print(result.stdout.strip())


def run_fp_parity_gate(python_exe, model_arg, model_name, imagenet_dir, env, reference_top1_pct, harness_desc,
                        skip_fp_parity=False):
    """Runs common/fp_parity_gate.py's repq_vit path as a subprocess.
    Needs BOTH `common` (via PYTHONPATH -- this repo's own package) AND
    RepQ-ViT's `classification/` dir on sys.path (via cwd, which `-m`
    module invocation auto-adds as sys.path[0]) for `from utils.build_model
    import build_model` etc. to resolve -- see common/fp_parity_gate.py's
    module docstring.

    Cached by `harness_desc` (unique per model+step) under
    `_fp_parity_cache/` -- a cache HIT is always a genuine prior PASS
    (the cache is only ever written after `check_fp_parity` succeeds, see
    below), so a crash later in the same model's bit sweep never forces
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

    repq_classification_dir = env["_TANDEM_REPQ_CLASSIFICATION_DIR"]
    fp_env = {k: v for k, v in env.items() if k != "_TANDEM_REPQ_CLASSIFICATION_DIR"}
    existing_pp = fp_env.get("PYTHONPATH", "")
    fp_env["PYTHONPATH"] = str(BASELINES_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
    cmd = [python_exe, "-m", "common.fp_parity_gate", "--method", "repq_vit",
           "--model-arg", model_arg, "--imagenet-dir", imagenet_dir]
    print(f"Running FP-parity gate: {' '.join(cmd)}  (cwd={repq_classification_dir})")
    result = run_streaming(cmd, cwd=repq_classification_dir, env=fp_env)
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


def run_test_quant(repo_path, python_exe, model_arg, imagenet_dir, w_bits, a_bits, env):
    cmd = [
        python_exe, "test_quant.py",
        "--model", model_arg, "--dataset", imagenet_dir,
        "--w_bits", str(w_bits), "--a_bits", str(a_bits),
        "--seed", str(SEED),  # their own default (verified from source) -- pinned explicitly here
        "--num-workers", str(NUM_WORKERS),  # their own default is 16 -- see NUM_WORKERS's own comment
    ]
    run_env = {k: v for k, v in env.items() if k != "_TANDEM_REPQ_CLASSIFICATION_DIR"}
    print(f"Running: {' '.join(cmd)}  (cwd={repo_path / 'classification'})")
    result = run_streaming(cmd, cwd=str(repo_path / "classification"), env=run_env)
    if result.returncode != 0:
        raise RuntimeError(f"RepQ-ViT test_quant.py failed (exit {result.returncode}):\n{result.stdout[-3000:]}")
    if run_env.get("TANDEM_PIN_CHECKPOINT_OVERRIDE") == "1":
        assert_override_fired(result.stdout, f"test_quant.py {model_arg} (W{w_bits}A{a_bits} quantized pass)")
    match = PREC1_RE.search(result.stdout)
    assert match, (
        "Could not find a '* Prec@1 X.XXX ...' line in test_quant.py's stdout -- its output format "
        "may have changed at this commit; check the tail printed above."
    )
    return float(match.group(1))


def run_bit_sweep(repo_path, args, model_name, repq_model_arg, d, published, env, checkpoint_tag, row_kind,
                   host_info, n_images, pip_freeze_hash):
    """Runs both BIT_CONFIGS on `model_name` under `env` (Step A: their
    own checkpoint, gated; Step B: pinned-checkpoint override, not gated
    -- caller decides which by which `env`/`published`/`row_kind` it
    passes in), appending one results.csv row per config."""
    rows = {}
    for w_bits, a_bits in BIT_CONFIGS:
        config_key = f"w{w_bits}a{a_bits}"
        print(f"\n{'=' * 70}\nRepQ-ViT {model_name} {config_key} [{row_kind}, checkpoint={checkpoint_tag}]\n{'=' * 70}")

        # Resume: skip a (model, config, row_kind, checkpoint) already
        # recorded in results.csv from an earlier attempt (e.g. this exact
        # process crashed partway through a full sweep) -- no per-batch
        # checkpoint exists for B2 (deviations.md #14), so this coarser
        # per-config resume is what actually saves a rerun from re-paying
        # every already-completed config's ~6-7 minute test_quant.py pass.
        existing = find_existing_row(
            method="repq_vit", model=model_name, config=config_key,
            row_kind=row_kind, checkpoint_tag=checkpoint_tag,
        )
        if existing is not None:
            print(f"Already in results.csv: repq_vit/{model_name}/{config_key} [{row_kind}] "
                  f"-- top1={existing['top1_50k']}%, gate_status={existing['gate_status']} -- skipping re-run.")
            rows[config_key] = float(existing["top1_50k"])
            continue

        calib_indices_path = (Path(__file__).parent / "_calib_indices" /
                               f"{model_name}_{config_key}_{row_kind}.json")
        # calib index CAPTURE never loads checkpoint weights (pure RandomSampler
        # replay, see common/calib_capture.py) -- unaffected by the Step-B
        # checkpoint override, run with the default environment as before.
        capture_calib_indices(args.python, args.imagenet_dir, calib_indices_path)

        top1_pct = run_test_quant(repo_path, args.python, repq_model_arg, args.imagenet_dir,
                                   w_bits, a_bits, env)

        n_l_per_block, bits_per_block = uniform_bits_trajectory(N_BLOCKS, N_TOKENS_TOTAL, (w_bits, a_bits))
        bops_ours = cost_from_trajectory(n_l_per_block, bits_per_block, d)

        if row_kind == "reproduction":
            pub = published.get(config_key, {})
            published_top1 = pub.get("top1_pct")
            gate_status, diff = check_gate(top1_pct, published_top1)
            print(format_gate_banner("repq_vit", model_name, config_key, top1_pct, published_top1, gate_status, diff))
            # NOT enforced under TANDEM_SMOKE_RUN -- same small-sample
            # incompatibility as the FP-parity gate (see run_fp_parity_gate's
            # docstring); still computed and printed either way, just not
            # asserted, so it's visible but doesn't halt the smoke check.
            if os.environ.get("TANDEM_SMOKE_RUN") == "1":
                print(f"[smoke run] reproduction gate NOT enforced (tol=0.3pp not meaningful at this "
                      f"sample size) -- gate_status={gate_status}, diff={diff}")
            else:
                assert gate_status != "fail", (
                    f"REPRODUCTION GATE FAILED for repq_vit/{model_name}/{config_key}: measured="
                    f"{top1_pct:.2f}%, published={published_top1:.2f}% (source: published.json, RepQ-ViT "
                    f"README W6A6 table), diff={diff:.3f}pp (tolerance=0.3pp). Halting -- this must be "
                    "diagnosed before any downstream comparison (compare_pareto.py) treats this number as "
                    "trustworthy."
                )
            print(f"Step-A reproduction-gate line: repq_vit/{model_name}/{config_key} -> {gate_status} "
                  f"(checkpoint={checkpoint_tag})")
            notes = (f"Step A (paper's own checkpoint); calib-batchsize=32 (their default), seed=0 "
                     f"(their default, pinned explicitly), single batch from train split, indices "
                     f"recorded at {calib_indices_path}")
        else:
            published_top1, gate_status, diff = None, "not_applicable", None
            notes = (f"Step B (TANDEM's pinned comparison checkpoint, override active); calib-batchsize=32, "
                     f"seed=0, single batch from train split, indices recorded at {calib_indices_path}. "
                     "No published reference exists for this method on this exact checkpoint -- "
                     "gate_status=not_applicable by design, FP-parity gate already validated the harness "
                     "on this checkpoint before this sweep ran.")

        this_config_hash = config_hash(REPQ_COMMIT, model_name, config_key, SEED, row_kind, str(calib_indices_path))
        append_result({
            "method": "repq_vit", "model": model_name, "config": config_key, "checkpoint_tag": checkpoint_tag,
            "row_kind": row_kind,
            "top1_50k": f"{top1_pct:.4f}", "n_images": n_images, "bops_ours": f"{bops_ours:.6e}",
            "repo_commit": REPQ_COMMIT,
            "published_top1_pct": "" if published_top1 is None else published_top1,
            "gate_status": gate_status, "gate_diff_pct_points": "" if diff is None else f"{diff:.4f}",
            "calib_images": CALIB_BATCHSIZE,
            "host": host_info["host"], "gpu_name": host_info["gpu_name"],
            "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
            "config_hash": this_config_hash, "pip_freeze_hash": pip_freeze_hash, "notes": notes,
        })
        rows[config_key] = top1_pct
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True, help="root with train/ and val/ subdirs")
    parser.add_argument("--python", default=sys.executable,
                         help="python executable of b2_repq_vit's own pinned venv")
    parser.add_argument("--host-tag", default=None,
                         help='e.g. "university_gpu" or "colab" -- recorded in results.csv')
    parser.add_argument("--only-model", default=None, choices=list(MODELS.keys()),
                         help="restrict this invocation to a single model (e.g. one Colab cell per "
                              "model, isolating a per-model crash like an OOM on a larger model from "
                              "the models already completed); omit to run all four, exactly as before")
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
    ])  # no "resume" -- B2 structurally has none, see AUDIT_REPORT.md Part B.3
    assert_valid_commit_sha(REPQ_COMMIT, "REPQ_COMMIT")

    repo_path = ensure_repo("https://github.com/zkkli/RepQ-ViT", REPQ_COMMIT, local_name="RepQ-ViT")

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
    step_a_env["_TANDEM_REPQ_CLASSIFICATION_DIR"] = str(repo_path / "classification")

    with open(Path(__file__).parent / "published.json") as f:
        published_all = json.load(f)

    models_to_run = {args.only_model: MODELS[args.only_model]} if args.only_model else MODELS
    for model_name, (repq_model_arg, d) in models_to_run.items():
        published = published_all[model_name]
        step_a_tag = PAPER_CHECKPOINT_TAG[model_name]
        fp_ref = pinned_fp_reference(model_name)
        coincide = coincides_with_pinned(model_name)
        print(f"\n{'#' * 70}\nRepQ-ViT / {model_name}  (Step A checkpoint={step_a_tag}, "
              f"coincides with pinned={coincide})\n{'#' * 70}")

        if coincide:
            # FP-parity gate on the SAME checkpoint Step A is about to use.
            run_fp_parity_gate(args.python, repq_model_arg, model_name, args.imagenet_dir, step_a_env,
                                fp_ref, harness_desc=f"repq_vit_step_a_{model_name}",
                                skip_fp_parity=args.skip_fp_parity)

        step_a_rows = run_bit_sweep(repo_path, args, model_name, repq_model_arg, d, published, step_a_env,
                                     step_a_tag, "reproduction", host_info, n_images, pip_freeze_hash)

        if coincide:
            # Step B: do NOT re-run (spec: "do not run twice") -- reuse
            # Step A's measurement, write comparison-row duplicates.
            for config_key, top1_pct in step_a_rows.items():
                w_bits, a_bits = (int(x) for x in re.findall(r"\d+", config_key))
                n_l_per_block, bits_per_block = uniform_bits_trajectory(N_BLOCKS, N_TOKENS_TOTAL, (w_bits, a_bits))
                bops_ours = cost_from_trajectory(n_l_per_block, bits_per_block, d)
                this_config_hash = config_hash(REPQ_COMMIT, model_name, config_key, SEED, "comparison_reused")
                append_result({
                    "method": "repq_vit", "model": model_name, "config": config_key,
                    "checkpoint_tag": step_a_tag, "row_kind": "comparison",
                    "top1_50k": f"{top1_pct:.4f}", "n_images": n_images, "bops_ours": f"{bops_ours:.6e}",
                    "repo_commit": REPQ_COMMIT, "published_top1_pct": "",
                    "gate_status": "not_applicable", "gate_diff_pct_points": "",
                    "calib_images": CALIB_BATCHSIZE,
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
            override_source = override_sitecustomize_source("timm")
            step_b_env = subprocess_env_with_tf32_disabled(sitecustomize_dir, extra_sitecustomize_source=override_source)
            step_b_env.update(override_env_vars(model_name, pinned_path))
            step_b_env["_TANDEM_REPQ_CLASSIFICATION_DIR"] = str(repo_path / "classification")

            run_fp_parity_gate(args.python, repq_model_arg, model_name, args.imagenet_dir, step_b_env,
                                fp_ref, harness_desc=f"repq_vit_step_b_{model_name}",
                                skip_fp_parity=args.skip_fp_parity)

            run_bit_sweep(repo_path, args, model_name, repq_model_arg, d, published, step_b_env,
                          pinned_checkpoint_tag(model_name), "comparison", host_info, n_images, pip_freeze_hash)


if __name__ == "__main__":
    main()
