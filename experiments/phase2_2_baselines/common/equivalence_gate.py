"""Host-equivalence gate (cheap, runs on BOTH the university GPU and
Colab BEFORE any full run): a smoke evaluation on a FIXED 256-image val
subset (plus a fixed train subset for calibration, ONLY for baselines that
actually calibrate -- see BASELINES_NEEDING_CALIBRATION; ToMe has no PTQ
step and its Colab notebook never materializes a train/ split at all)
using each baseline's own run.py, completely UNMODIFIED -- the exact same
mechanism used for a real run, just pointed at a small symlinked
ImageFolder subset instead of the full 50k, via a separate results CSV
(`TANDEM_RESULTS_CSV_OVERRIDE`, see common/results_io.py) so a smoke run
never pollutes the real experiments/phase2_2_baselines/results.csv.

Canonical fingerprint (per-config correct-count on the subset + repo
commit(s) + requirements-hash) is stored in
experiments/phase2_2_baselines/equivalence_fingerprint.json the FIRST time this runs (on
whichever host runs it first); every subsequent run on ANY host
recomputes and COMPARES:
  - correct-counts must match EXACTLY;
  - a +-1-image discrepancy is tolerated ONLY if BOTH hosts' GPU/arch are
    logged (a known, narrow failure mode: a borderline logit can flip
    sign across GPU architectures even with TF32 disabled and identical
    seeds -- floating-point summation order in cuDNN/cuBLAS kernels is
    NOT guaranteed bit-identical across GPU generations, only typically
    within one);
  - a larger delta, or a requirements-hash mismatch, HALTS -- the
    requirements-hash mismatch prints a full pip-freeze diff so the
    drift is immediately visible, not just asserted.

Framework-free (subprocess + stdlib only) -- runs from the orchestration
environment, same as common/repo_utils.py. Each baseline's own run.py is
still invoked with THAT baseline's own pinned-venv python (passed in
`python_exes`), exactly as a real run would be.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASELINES_ROOT))
from common.subprocess_utils import run_streaming  # noqa: E402

FINGERPRINT_PATH = BASELINES_ROOT / "equivalence_fingerprint.json"
N_VAL_SMOKE = 256
N_TRAIN_SMOKE = 2000  # covers B3's largest single calibration draw (1000 images) with headroom

# One representative config per baseline -- `_run_smoke` scopes each
# subprocess invocation down to ONLY this (model, config) via the same
# --only-* flag each run.py already exposes for real-run session-limit
# hardening (see _smoke_scope_args). PRIOR design (until 2026-07) let the
# subprocess run its WHOLE config matrix regardless, on the assumption
# this was "cheap at 256 images: seconds" -- WRONG in practice (confirmed
# via a real Colab run going silent for many minutes): that assumption
# only accounts for per-image compute shrinking with fewer images, not
# FIXED per-invocation costs that don't -- RepQ-ViT's/FQ-ViT's own
# test_quant.py setup overhead (model construction, quantizer wrapping,
# scale reparameterization), and critically, B2's/B3's ViT-B Step B pass,
# which downloads a real ~350MB checkpoint from the HF Hub independent of
# how many images it's about to evaluate. Since NO baseline's
# REPRESENTATIVE_CONFIG below uses vit_base_patch16_224, that download was
# ALWAYS pure waste from the smoke check's own perspective -- it was
# never on the path to the one row _read_smoke_result actually consults.
REPRESENTATIVE_CONFIG = {
    "tome": ("deit_small_patch16_224", "r13"),
    "repq_vit": ("deit_small_patch16_224", "w6a6"),
    "fq_vit": ("deit_small_patch16_224", "w8a8"),
    # 2026-07 B4 rewrite: config values are now "r{r}_w{w}a{a}_{merge_first,
    # quant_first,best}" (see b4_sequential_composites/run.py's
    # config_key_for/config_key_best) -- "best" is the row_kind=comparison
    # row for budget_fraction=0.15 (r=17, w8a8), the SAME operating point
    # SEQUENTIAL_COMPOSITE_SMOKE_BUDGET_FRACTION below scopes the smoke
    # subprocess to. Kept in sync BY HAND with
    # b4_sequential_composites.run.OPERATING_POINTS (that module imports
    # torch/timm at load time, so it cannot be imported here -- this
    # module stays framework-free) -- if that grid changes, this must be
    # updated to match, or the smoke check will not find the row it's
    # looking for.
    "sequential_composite": ("deit_small_patch16_224", "r17_w8a8_best"),
}
SEQUENTIAL_COMPOSITE_SMOKE_BUDGET_FRACTION = 0.15

# ToMe (B1) has no PTQ calibration step at all (see b1_tome/run.py's own
# module docstring) -- its Colab notebook correspondingly never
# materializes a train/ split (B1_tome_colab.py: "only val/ is
# materialized here"). build_smoke_subset must not unconditionally demand
# a train/ split just because SOME baselines need one.
BASELINES_NEEDING_CALIBRATION = {"repq_vit", "fq_vit", "sequential_composite"}

RUNNERS = {
    "tome": BASELINES_ROOT / "b1_tome" / "run.py",
    "repq_vit": BASELINES_ROOT / "b2_repq_vit" / "run.py",
    "fq_vit": BASELINES_ROOT / "b3_fq_vit" / "run.py",
    "sequential_composite": BASELINES_ROOT / "b4_sequential_composites" / "run.py",
}


def _sorted_samples(imagefolder_root):
    """Deterministic (class dir, file) sorted list -- matches
    torchvision.datasets.ImageFolder's own internal ordering exactly
    (it sorts both classes and filenames), so index N here IS index N
    in any real ImageFolder built from the same directory."""
    root = Path(imagefolder_root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    samples = []
    for cls in classes:
        files = sorted((root / cls).iterdir())
        samples.extend((cls, f) for f in files)
    return samples


def build_smoke_subset(real_imagenet_dir, smoke_dir, n_val=N_VAL_SMOKE, n_train=N_TRAIN_SMOKE, include_train=True):
    """Symlinks (never copies -- fast, no extra disk) the first n_val val
    images and (if `include_train`) first n_train train images, in
    DETERMINISTIC sorted order, into `smoke_dir/{val,train}/<wnid>/<file>`.
    Idempotent: skips a split entirely if it already has exactly the right
    count. `include_train=False` for a gate invocation scoped to baselines
    that don't calibrate (e.g. tome alone) -- their Colab notebooks never
    materialize a train/ split, so demanding one here would fail on a
    directory that isn't a bug, it was just never supposed to exist."""
    smoke_dir = Path(smoke_dir)
    real_imagenet_dir = Path(real_imagenet_dir)

    def _link_subset(split, n):
        dst_split = smoke_dir / split
        existing = [p for p in dst_split.rglob("*")] if dst_split.exists() else []
        existing_files = [p for p in existing if p.is_file() or p.is_symlink()]
        if len(existing_files) == n:
            print(f"{dst_split} already has {n} images -- skipping.")
            return
        assert not existing_files, (
            f"{dst_split} has {len(existing_files)} images, expected 0 or {n} -- looks like a "
            "partial/stale smoke subset. Delete it and re-run: `import shutil; "
            f"shutil.rmtree('{dst_split}')`."
        )
        samples = _sorted_samples(real_imagenet_dir / split)
        assert len(samples) >= n, f"{real_imagenet_dir / split} has only {len(samples)} images, need {n}"
        for cls, src_file in samples[:n]:
            dst_cls_dir = dst_split / cls
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_cls_dir / src_file.name
            if not dst_file.exists():
                try:
                    dst_file.symlink_to(src_file.resolve())
                except OSError:
                    # symlinks unavailable (e.g. no privilege on some Windows setups) -- fall back to a real copy
                    import shutil
                    shutil.copy2(src_file, dst_file)
        print(f"Built smoke subset: {n} images at {dst_split}")

    _link_subset("val", n_val)
    if include_train:
        _link_subset("train", n_train)
    return smoke_dir


def _pip_freeze_hash(python_exe):
    result = subprocess.run([python_exe, "-m", "pip", "freeze"], capture_output=True, text=True)
    assert result.returncode == 0, f"pip freeze failed for {python_exe}:\n{result.stderr}"
    frozen = result.stdout
    return hashlib.sha256(frozen.encode()).hexdigest()[:16], frozen


def _smoke_scope_args(baseline, model, config):
    """Extra CLI args scoping a smoke-check subprocess down to ONLY the
    representative (model, config) REPRESENTATIVE_CONFIG/_read_smoke_result
    actually reads -- see REPRESENTATIVE_CONFIG's own comment for why this
    matters. Each baseline's own run.py flag semantics differ slightly:
    tome/sequential_composite select by r-value/composite-name directly;
    repq_vit/fq_vit only support narrowing by MODEL (no per-config flag
    exists), which still eliminates 3 of 4 models -- including, for every
    current REPRESENTATIVE_CONFIG entry, ViT-B and its checkpoint download."""
    if baseline == "tome":
        r = config[1:] if config.startswith("r") else config  # "r13" -> "13"
        return ["--only-r", r]
    if baseline in ("repq_vit", "fq_vit"):
        return ["--only-model", model]
    if baseline == "sequential_composite":
        return ["--only-budget-fraction", str(SEQUENTIAL_COMPOSITE_SMOKE_BUDGET_FRACTION)]
    raise ValueError(f"no smoke-scoping rule defined for baseline {baseline!r}")


def _run_smoke(baseline, python_exe, smoke_dir, host_tag, smoke_csv_path, model, config):
    env = dict(os.environ)
    env["TANDEM_RESULTS_CSV_OVERRIDE"] = str(smoke_csv_path)
    # B1/B4's per-batch eval checkpoint (_checkpoints/<config>.json) is
    # keyed by a config_hash that does NOT include dataset identity/size
    # (see their own config_hash(...) calls) -- without this override, a
    # smoke run's 256-image checkpoint and a real run's 50k-image
    # checkpoint for the SAME config would collide under the SAME
    # config_hash, and a real run could silently "resume" from a smoke
    # run's partial/complete state. Same isolation pattern as
    # TANDEM_RESULTS_CSV_OVERRIDE above, just for checkpoints.
    env["TANDEM_CHECKPOINT_DIR_OVERRIDE"] = str(BASELINES_ROOT / "_smoke_checkpoints" / baseline)
    # Both the FP-parity gate (0.15pp tolerance, common/fp_parity_gate.py)
    # and the reproduction gate (0.3pp tolerance, common/repro_gate.py) are
    # calibrated for full-50k-scale statistics -- at N_VAL_SMOKE=256 images,
    # sampling noise alone (~2.8pp standard error) makes either impossible
    # to reliably pass regardless of correctness, and the smoke subset is
    # also class-skewed (first ~5 of 1000 classes in sorted order, not a
    # representative sample -- see build_smoke_subset). The smoke run's own
    # job (cross-host correct-count equivalence on this fixed subset) is
    # unrelated to and unaffected by this -- both real gates still run,
    # unmodified, on every genuine full-50k pass. Every FP-parity-gate call
    # site and every reproduction-gate assertion (B1/B2/B3 -- B4 has no
    # reproduction gate, see its own module docstring) checks this flag.
    env["TANDEM_SMOKE_RUN"] = "1"
    cmd = ([python_exe, str(RUNNERS[baseline]), "--imagenet-dir", str(smoke_dir), "--host-tag", host_tag]
           + _smoke_scope_args(baseline, model, config))
    print(f"Running smoke check: {' '.join(cmd)}")
    result = run_streaming(cmd, env=env)
    assert result.returncode == 0, (
        f"Smoke run FAILED for {baseline} (exit {result.returncode}) -- see output above.\n"
        f"{result.stdout[-2000:]}"
    )


def _read_smoke_result(smoke_csv_path, method, model, config):
    with open(smoke_csv_path) as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if r["method"] == method and r["model"] == model and r["config"] == config]
    assert matches, f"No results row for {method}/{model}/{config} in {smoke_csv_path} after the smoke run"
    row = matches[-1]  # last write wins, in case this smoke check itself was resumed/re-run
    n_images = int(row["n_images"])
    top1_pct = float(row["top1_50k"])
    correct_count = round(top1_pct / 100 * n_images)
    return {
        "correct_count": correct_count, "n_images": n_images, "repo_commit": row["repo_commit"],
        "gpu_name": row["gpu_name"], "torch_version": row["torch_version"], "host": row["host"],
    }


def run_gate(imagenet_dir, host_tag, python_exes, baselines=None):
    """python_exes: dict mapping baseline name -> that baseline's own
    pinned-venv python executable. `baselines` (subset of
    REPRESENTATIVE_CONFIG's keys) lets a host skip a baseline whose venv
    isn't set up yet -- defaults to everything in `python_exes`."""
    baselines = baselines or list(python_exes)
    smoke_dir = BASELINES_ROOT / "_smoke_imagenet"
    needs_train = any(b in BASELINES_NEEDING_CALIBRATION for b in baselines)
    build_smoke_subset(imagenet_dir, smoke_dir, include_train=needs_train)

    fingerprint = {}
    if FINGERPRINT_PATH.exists():
        with open(FINGERPRINT_PATH) as f:
            fingerprint = json.load(f)

    results = {}
    halted = []
    for baseline in baselines:
        python_exe = python_exes[baseline]
        model, config = REPRESENTATIVE_CONFIG[baseline]
        smoke_csv_path = BASELINES_ROOT / f"_smoke_results_{baseline}.csv"
        if smoke_csv_path.exists():
            smoke_csv_path.unlink()  # fresh per gate invocation -- never read a stale row from a prior attempt

        req_hash, req_freeze = _pip_freeze_hash(python_exe)
        _run_smoke(baseline, python_exe, smoke_dir, host_tag, smoke_csv_path, model, config)
        measured = _read_smoke_result(smoke_csv_path, baseline, model, config)
        measured["requirements_hash"] = req_hash
        key = f"{baseline}/{model}/{config}"
        results[key] = measured

        if key not in fingerprint:
            print(f"{'=' * 70}\nNo canonical fingerprint yet for {key} -- recording THIS run as "
                  f"canonical: {measured}\n{'=' * 70}")
            fingerprint[key] = measured
            continue

        canonical = fingerprint[key]
        if canonical["requirements_hash"] != req_hash:
            print(f"{'!' * 70}\nREQUIREMENTS-HASH MISMATCH for {key}: canonical="
                  f"{canonical['requirements_hash']} this host={req_hash}\n"
                  f"--- pip freeze (this host) ---\n{req_freeze}\n{'!' * 70}")
            halted.append((key, "requirements_hash_mismatch"))
            continue

        delta = abs(canonical["correct_count"] - measured["correct_count"])
        if delta == 0:
            print(f"{key}: EXACT match ({measured['correct_count']}/{measured['n_images']}) -- "
                  "host-equivalence OK.")
        elif delta == 1:
            print(f"{'!' * 70}\n{key}: +-1 image discrepancy ({canonical['correct_count']} vs "
                  f"{measured['correct_count']}, out of {measured['n_images']}) -- TOLERATED, logged: "
                  f"canonical host={canonical['host']}/{canonical['gpu_name']}, this host="
                  f"{measured['host']}/{measured['gpu_name']} (a borderline-logit flip across GPU "
                  "architectures is a known, narrow failure mode even with TF32 disabled and "
                  f"identical seeds).\n{'!' * 70}")
        else:
            print(f"{'!' * 70}\nHOST-EQUIVALENCE GATE FAILED for {key}: canonical="
                  f"{canonical['correct_count']}/{canonical['n_images']} ({canonical['host']}/"
                  f"{canonical['gpu_name']}), this host={measured['correct_count']}/"
                  f"{measured['n_images']} ({measured['host']}/{measured['gpu_name']}) -- "
                  f"delta={delta} exceeds the +-1 tolerance.\n{'!' * 70}")
            halted.append((key, f"correct_count_delta_{delta}"))

    with open(FINGERPRINT_PATH, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"\nSaved {FINGERPRINT_PATH}")

    assert not halted, (
        f"Host-equivalence gate FAILED for: {halted}. Do NOT proceed to a full run on this host until "
        "this is diagnosed -- see the per-baseline messages above."
    )
    print("\nHOST-EQUIVALENCE GATE: ALL CHECKED BASELINES PASSED.")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True)
    parser.add_argument("--host-tag", required=True, help='e.g. "university_gpu" or "colab"')
    parser.add_argument("--tome-python", default=None)
    parser.add_argument("--repq-vit-python", default=None)
    parser.add_argument("--fq-vit-python", default=None)
    parser.add_argument("--sequential-composite-python", default=None)
    args = parser.parse_args()

    python_exes = {
        "tome": args.tome_python, "repq_vit": args.repq_vit_python,
        "fq_vit": args.fq_vit_python, "sequential_composite": args.sequential_composite_python,
    }
    baselines = [b for b, p in python_exes.items() if p]
    assert baselines, (
        "pass at least one of --tome-python / --repq-vit-python / --fq-vit-python / "
        "--sequential-composite-python (each baseline's own pinned-venv python)"
    )
    run_gate(args.imagenet_dir, args.host_tag, python_exes, baselines=baselines)


if __name__ == "__main__":
    main()
