"""P2.2 post-conversion self-test: the FIRST executable cell after
bootstrap (clone/venv/secrets/materialize) in every `*_colab.ipynb`, run
BEFORE any real run cell. Ten hard-assert checks, printed as one PASS/FAIL
banner; any failure raises, halting a "Run All" right here rather than
partway into a real 50k pass.

Most of what's checked here already exists elsewhere in common/ (this
project's own prior audit trail, see ../deviations.md) -- this module
mostly SURFACES those existing guarantees into one banner per notebook,
reusing the real functions rather than re-implementing them:
  - common.verify_colab_thin_wrappers.check_file -- item 1 (wrapper purity)
  - common.repo_utils.ensure_repo -- item 2 (upstream pinning)
  - common.determinism.disable_tf32 -- item 3 (numeric environment)
  - common.bops_identity_check -- item 6 (cost ruler unit checks)
  - common.results_io.RESULTS_COLUMNS / find_existing_row -- item 7
    (schema) and item 8 for B2/B3 (results.csv-level resume)
  - common.eval_utils.evaluate_top1_resumable / common.checkpoint -- item 8
    for B1/B4 (real interrupt/resume simulation)
  - common.equivalence_gate.run_gate / FINGERPRINT_PATH -- item 9
    (256-image smoke equivalence check, already first-run-canonical)

Three places where this project's actual (already-audited) design differs
from a literal reading of the spec this module implements -- documented in
../deviations.md, not silently reconciled:
  - Item 1's "PINNED_REF": the TANDEM repo itself is OUR OWN actively-
    developed code (unlike the frozen third-party ToMe/RepQ-ViT/FQ-ViT
    pins) -- `git rev-parse HEAD` is RECORDED (asserted well-formed,
    printed, saved) rather than compared against one hard-coded SHA that
    would need manual upkeep on every push and would falsely fail the
    moment any future improvement is pushed.
  - Item 7's schema check asserts our real (20-column) RESULTS_COLUMNS is
    a SUPERSET of the 15 named columns (including "setting", added 2026-07
    for the B1 off-the-shelf/trained-with-method reference-setting
    correction), not exact equality -- the 5 extra columns (gate_status,
    gate_diff_pct_points, calib_images, published_top1_pct, cuda_version)
    are load-bearing elsewhere in this project's own gates; deleting them
    to match a shorter list would be a regression, not a fix.
  - Item 5's manifest-hash check does not literally diff against
    experiments/phase2_1_fleet/P2_tandem_pipeline.py (already-validated, out of scope to
    touch) -- it hashes baselines' OWN materialized val manifest and
    treats it as baselines' own canonical fingerprint (same first-run-
    defines-canonical pattern as common/equivalence_gate.py). This module
    is ONLY ever invoked from the Colab notebooks (never from the
    university-GPU run.py path), so in practice this catches drift
    ACROSS COLAB SESSIONS ONLY (re-materializing on a fresh Colab VM,
    across baselines that share IMAGENET_ROOT). It is NOT a Colab-vs-
    university-GPU check: common/imagenet.py (university GPU) points at
    whatever real WNID-named ImageFolder the user already has locally,
    while common/colab_imagenet.py (Colab) names class directories by
    zero-padded integer label (see its own docstring, ../deviations.md
    #23) -- these two naming conventions would produce DIFFERENT hashes
    for the exact same underlying image set, so comparing across hosts
    here would be a false mismatch, not a real one. Literal identity to
    P2.1's own S8 measurement is asserted INDIRECTLY, via the FP-parity
    gate's existing 0.15pp accuracy check against P2.1's own S8 number (a
    scrambled/wrong manifest would fail that, regardless of host).

Dual-mode, mirroring common/fp_parity_gate.py's own pattern: mostly
torch-free (checks 1, 2, 4, 5, 6, 7, and B2/B3's item 8 need no torch at
all), except item 3 (numeric environment) and B1/B4's item 8 (a real
interrupt/resume simulation), which need an actual torch/model. For B1/B4,
this WHOLE script is launched by the pinned venv's python directly (same
convention their own run.py uses), so those checks run in-process. For
B2/B3 (torch-free orchestrator, same as their own run.py), item 3 dispatches
a small subprocess into the pinned venv (`--python`); item 9 already
subprocesses internally via common.equivalence_gate.run_gate, regardless of
caller context.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import equivalence_gate, repo_utils  # noqa: E402
from common import bops_identity_check as bic  # noqa: E402
from common.results_io import append_result, find_existing_row, RESULTS_COLUMNS  # noqa: E402
from common.verify_colab_thin_wrappers import check_file as _scan_wrapper_purity  # noqa: E402

BASELINES_ROOT = Path(__file__).resolve().parent.parent

# Duplicated as plain DATA (not logic) from each baseline's own run.py
# module-level constants -- same precedent as common/equivalence_gate.py's
# own RUNNERS/REPRESENTATIVE_CONFIG dicts, which likewise reference each
# baseline by file path/short config rather than importing run.py as a
# module (b1_tome/ etc. aren't proper packages, and B1/B4's run.py pull in
# torch/timm at import time regardless of whether this check needs them).
# MUST match TOME_COMMIT/REPQ_COMMIT/FQVIT_COMMIT in the respective run.py.
UPSTREAM_COMMITS = {
    "tome": [("https://github.com/facebookresearch/ToMe",
              "af95e4b1befa172dadccd8c81e223b10090f9579", "ToMe")],
    "repq_vit": [("https://github.com/zkkli/RepQ-ViT",
                  "a160561b9233945e6680132f129d13041b787318", "RepQ-ViT")],
    "fq_vit": [("https://github.com/megvii-research/FQ-ViT",
                "5daf5915c715dc21d6afcd64e661652030e640ab", "FQ-ViT")],
    "sequential_composite": [
        ("https://github.com/facebookresearch/ToMe",
         "af95e4b1befa172dadccd8c81e223b10090f9579", "ToMe"),
        ("https://github.com/zkkli/RepQ-ViT",
         "a160561b9233945e6680132f129d13041b787318", "RepQ-ViT"),
    ],
}
REQUIREMENTS_PATHS = {
    "tome": "b1_tome/requirements.txt",
    "repq_vit": "b2_repq_vit/requirements.txt",
    "fq_vit": "b3_fq_vit/requirements.txt",
    "sequential_composite": "b4_sequential_composites/requirements.txt",
}
COLAB_PY_PATHS = {
    "tome": "b1_tome/B1_tome_colab.py",
    "repq_vit": "b2_repq_vit/B2_repq_vit_colab.py",
    "fq_vit": "b3_fq_vit/B3_fq_vit_colab.py",
    "sequential_composite": "b4_sequential_composites/B4_sequential_composites_colab.py",
    "compare_pareto": "B5_compare_pareto_colab.py",
}
RESUME_SIM_MODEL = {"tome": "deit_small_patch16_224", "sequential_composite": "deit_small_patch16_224"}
IN_PROCESS_BASELINES = {"tome", "sequential_composite"}  # launched directly by the venv python, like their run.py

CHECK_NAMES = {
    1: "WRAPPER PURITY + REPO HEAD", 2: "UPSTREAM PINNING", 3: "NUMERIC ENVIRONMENT",
    4: "SECRETS", 5: "DATA DETERMINISM", 6: "COST RULER UNIT CHECKS", 7: "SCHEMA",
    8: "RESUME", 9: "EQUIVALENCE FINGERPRINT", 10: "GATE ORDERING GUARD",
}

REQUIRED_RESULT_COLUMNS = [
    "method", "model", "config", "row_kind", "setting", "checkpoint_tag", "top1_50k", "n_images",
    "BOPs_ours", "repo_commit", "host", "gpu_name", "torch_version", "config_hash", "notes",
]

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]


# --- item 1: wrapper purity + TANDEM repo HEAD -------------------------

def check_wrapper_purity_and_repo_head(colab_py_path, repo_dir):
    violations = _scan_wrapper_purity(Path(colab_py_path))
    assert not violations, (
        f"{len(violations)} forbidden pattern(s) in {colab_py_path}: " +
        "; ".join(f"line {ln}: {txt!r}" for ln, txt, _ in violations)
    )
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True)
    assert result.returncode == 0, f"git rev-parse HEAD failed in {repo_dir}: {result.stderr}"
    head = result.stdout.strip()
    assert _SHA_RE.match(head), f"unexpected HEAD format: {head!r}"
    return True, (f"0 forbidden patterns in {Path(colab_py_path).name}; TANDEM repo HEAD={head[:12]}... "
                   "(recorded for provenance, not hard-pinned -- see module docstring)")


# --- item 2: upstream pinning -------------------------------------------

def _assert_requirements_pinned(requirements_path):
    lines = [l.strip() for l in Path(requirements_path).read_text().splitlines()]
    real_lines = [l for l in lines if l and not l.startswith("#")]
    unpinned = [l for l in real_lines if "==" not in l]
    assert not unpinned, f"{requirements_path} has non-==-pinned line(s): {unpinned}"
    return real_lines


def check_upstream_pinning(baseline, python_exe):
    if baseline == "compare_pareto":
        return True, "skipped -- B5 has no upstream repo"
    pinned = []
    for url, commit, local_name in UPSTREAM_COMMITS[baseline]:
        repo_utils.ensure_repo(url, commit, local_name=local_name)
        pinned.append(f"{local_name}@{commit[:12]}")
    req_path = BASELINES_ROOT / REQUIREMENTS_PATHS[baseline]
    pinned_lines = _assert_requirements_pinned(req_path)
    target_python = python_exe or sys.executable
    req_hash, _ = equivalence_gate._pip_freeze_hash(target_python)
    return True, f"{', '.join(pinned)}; {len(pinned_lines)} ==-pinned requirement(s); pip-freeze sha={req_hash}"


# --- item 3: numeric environment ----------------------------------------

def _numeric_environment_inprocess():
    import torch
    from common.determinism import disable_tf32
    disable_tf32()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (no CUDA)"
    detail = f"TF32 matmul=False cudnn=False; GPU={gpu_name} torch={torch.__version__} cuda={torch.version.cuda}"
    return True, detail


def check_numeric_environment(baseline, python_exe):
    if baseline == "compare_pareto":
        return True, "skipped -- B5 has no GPU work"
    if python_exe is None:
        return _numeric_environment_inprocess()
    result = subprocess.run(
        [python_exe, "-m", "common.notebook_self_test", "_worker_numeric_env"],
        cwd=str(BASELINES_ROOT), capture_output=True, text=True,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("TANDEM_SELF_TEST_NUMERIC_ENV ")]
    assert lines, f"numeric-environment worker produced no output. stderr:\n{result.stderr[-2000:]}"
    payload = json.loads(lines[-1][len("TANDEM_SELF_TEST_NUMERIC_ENV "):])
    assert payload["passed"], payload["detail"]
    return True, payload["detail"]


# --- item 4: secrets ------------------------------------------------------

def check_secrets(colab_py_path):
    source = Path(colab_py_path).read_text(encoding="utf-8")
    matches = [m for pat in _SECRET_PATTERNS for m in pat.findall(source)]
    assert not matches, (
        f"{len(matches)} token-literal-shaped string(s) found in {colab_py_path} -- never commit a "
        "real token, only userdata.get() calls"
    )
    return True, (f"0 token-literal matches ({len(_SECRET_PATTERNS)} patterns: hf_/github_pat_/ghp_) "
                   f"in {Path(colab_py_path).name}")


# --- item 5: data determinism ---------------------------------------------

def _hash_val_manifest(imagenet_dir):
    val_dir = Path(imagenet_dir) / "val"
    classes = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
    h = hashlib.sha256()
    n = 0
    for cls in classes:
        for f in sorted((val_dir / cls).iterdir()):
            h.update(f"{cls}/{f.name}".encode())
            n += 1
    return h.hexdigest()[:16], n


VAL_MANIFEST_CANONICAL_PATH = BASELINES_ROOT / "val_manifest_canonical.json"


def check_data_determinism(imagenet_dir, baseline):
    if baseline == "compare_pareto" or imagenet_dir is None:
        return True, "skipped -- no --imagenet-dir (B5 has no eval data)"
    manifest_hash, n = _hash_val_manifest(imagenet_dir)
    assert n == 50000, f"expected the full 50k ImageNet-1k val split, found {n} images -- incomplete/wrong materialization"

    train_dir = (Path(imagenet_dir) / "train").resolve()
    val_dir = (Path(imagenet_dir) / "val").resolve()
    assert train_dir != val_dir, "calibration source (train/) and eval source (val/) must be different directories"

    # P2.2 audit item #3 ruling (part 2), 2026-07: val_manifest_canonical.json
    # is committed and PERMANENT -- the old "no canonical yet, recording this
    # run as canonical" first-run path is gone. A missing file here is no
    # longer a legitimate first run, it's a broken/partial checkout.
    assert VAL_MANIFEST_CANONICAL_PATH.exists(), (
        f"{VAL_MANIFEST_CANONICAL_PATH} not found -- this file is committed to the repo and must always "
        "be present (frozen 2026-07: B1's val-manifest hash is the permanent cross-baseline canonical). "
        "A missing file here means a broken/partial checkout -- there is no longer a legitimate "
        "'first run defines canonical' path."
    )
    canonical = json.loads(VAL_MANIFEST_CANONICAL_PATH.read_text())
    assert manifest_hash == canonical["manifest_hash"], (
        f"Val manifest hash MISMATCH: this materialization={manifest_hash}, frozen cross-baseline "
        f"canonical={canonical['manifest_hash']} (recorded by {canonical['recorded_by']} on "
        f"{canonical['date']}, see {VAL_MANIFEST_CANONICAL_PATH.name}) -- the materialized ImageNet-1k "
        "val set differs from the one every other baseline is required to match (different image "
        "set/order, or a corrupted/partial materialization). Do not proceed."
    )
    detail = (f"Matches the FROZEN cross-baseline canonical val-manifest hash {manifest_hash} (n={n}, "
              f"frozen by {canonical['recorded_by']} on {canonical['date']}); calib/eval sources disjoint")
    return True, detail


# --- item 6: cost ruler unit checks ---------------------------------------

def check_cost_ruler():
    from common import checkpoint_pin
    details = []
    for name, d in bic.MODEL_DIMS.items():
        c_id = bic.check_identity_equivalence(bic.N_BLOCKS, bic.N_TOKENS_TOTAL, d)
        details.append(f"{name}: BOPs(identity)={c_id:.4e}")
        _, c_id_detail = bic.check_c_id_matches_canonical_summary(name, bic.N_BLOCKS, bic.N_TOKENS_TOTAL, d)
        details.append(f"{name}: {c_id_detail}")
    tome_c_id = bic.check_tome_schedule_r0_equivalence(
        bic.N_BLOCKS, bic.N_TOKENS_TOTAL, bic.MODEL_DIMS["deit_small_patch16_224"])
    details.append(f"ToMe r=0 schedule matches C_id ({tome_c_id:.4e})")
    checkpoint_pin.assert_fp_references_match_canonical_summaries()
    details.append("PINNED_TAGS FP references match canonical P2.1 summaries")
    return True, "; ".join(details)


# --- item 7: schema --------------------------------------------------------

def check_schema():
    actual_lower = {c.lower() for c in RESULTS_COLUMNS}
    missing = [c for c in REQUIRED_RESULT_COLUMNS if c.lower() not in actual_lower]
    assert not missing, f"results.csv schema missing required column(s): {missing} (actual: {RESULTS_COLUMNS})"
    return True, (f"{len(RESULTS_COLUMNS)} actual columns, superset of all {len(REQUIRED_RESULT_COLUMNS)} "
                   f"required: {RESULTS_COLUMNS}")


# --- item 8: resume --------------------------------------------------------

def check_resume_csv_mechanism(method_name):
    tmp = tempfile.mktemp(suffix=".csv")
    row = {c: "" for c in RESULTS_COLUMNS}
    row.update({
        "method": method_name, "model": "deit_tiny_patch16_224", "config": "w8a8",
        "row_kind": "reproduction", "checkpoint_tag": "deit_tiny_patch16_224.fb_in1k",
        "top1_50k": "71.6100", "n_images": 50000, "gate_status": "pass",
    })
    append_result(row, csv_path=tmp)
    found = find_existing_row(tmp, method=method_name, model="deit_tiny_patch16_224", config="w8a8",
                               row_kind="reproduction", checkpoint_tag="deit_tiny_patch16_224.fb_in1k")
    assert found is not None, "exact-match resume lookup failed to find a row that was just written"
    not_found = find_existing_row(tmp, method=method_name, model="deit_tiny_patch16_224", config="w6a6",
                                   row_kind="reproduction", checkpoint_tag="deit_tiny_patch16_224.fb_in1k")
    assert not_found is None, "resume lookup incorrectly matched a DIFFERENT config (w6a6 vs w8a8)"
    os.unlink(tmp)
    return True, "find_existing_row: exact-match found; differing-config correctly NOT matched"


def _resume_simulation_inprocess(baseline, imagenet_dir, device):
    import timm
    import timm.data
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    from common.checkpoint import load_checkpoint
    from common.determinism import config_hash as det_config_hash
    from common.eval_utils import evaluate_top1_resumable

    model_name = RESUME_SIM_MODEL[baseline]
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()

    # Preprocessing DERIVED from the model's own resolved timm data config
    # (common/imagenet.py's build_val_loader does the same, for the same
    # reason -- a hardcoded resize/interpolation recipe silently diverged
    # from deit_small_patch16_224.fb_in1k's real requirements, found via a
    # real Colab run, 2026-07).
    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    full_dataset = ImageFolder(f"{imagenet_dir}/val", transform=transform)
    assert len(full_dataset) >= 512, f"need >=512 val images for the resume simulation, found {len(full_dataset)}"
    subset = Subset(full_dataset, range(512))
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    meta_dir = BASELINES_ROOT / "_self_test_metadata" / "resume_sim"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ckpt_a = meta_dir / f"{baseline}_A.json"
    ckpt_b = meta_dir / f"{baseline}_B.json"
    for p in (ckpt_a, ckpt_b):
        if p.exists():
            p.unlink()

    shared_hash = det_config_hash("resume_sim", baseline, model_name)

    top1_a, n_a, _ = evaluate_top1_resumable(model, loader, ckpt_a, shared_hash, device=device,
                                              desc="resume-sim (uninterrupted)")
    correct_a = round(top1_a * n_a)

    def _interrupt_after(it, n):
        for i, batch in enumerate(it):
            if i >= n:
                raise RuntimeError("TANDEM_SELF_TEST_SIMULATED_INTERRUPT")
            yield batch

    try:
        evaluate_top1_resumable(model, _interrupt_after(loader, 5), ckpt_b, shared_hash, device=device,
                                 desc="resume-sim (pre-interrupt)", save_every_n_batches=3)
        raise AssertionError("expected a simulated interrupt but the pre-interrupt pass completed cleanly")
    except RuntimeError as e:
        assert "TANDEM_SELF_TEST_SIMULATED_INTERRUPT" in str(e), f"unexpected exception during interrupt: {e}"

    top1_b, n_b, _ = evaluate_top1_resumable(model, loader, ckpt_b, shared_hash, device=device,
                                              desc="resume-sim (resumed)")
    correct_b = round(top1_b * n_b)

    assert n_a == 512 and n_b == 512, f"expected 512 images both passes, got n_a={n_a} n_b={n_b}"
    assert correct_a == correct_b, (
        f"resume mismatch: uninterrupted correct={correct_a}, interrupted-then-resumed correct={correct_b} "
        "-- resuming from a mid-pass checkpoint must give EXACTLY the same result as an uninterrupted pass."
    )

    wrong_hash = det_config_hash("resume_sim", baseline, model_name, "WRONG")
    guard_state = load_checkpoint(ckpt_b, wrong_hash)
    assert guard_state["n_total"] == 0 and guard_state["n_batches_done"] == 0, (
        "checkpoint config-hash guard failed: loading an existing checkpoint against the WRONG expected "
        "hash must start fresh (n_total=0), not silently reuse the prior state"
    )
    return True, (f"512/512 images both passes; uninterrupted correct={correct_a} == "
                   f"interrupted-then-resumed correct={correct_b}; config-hash guard verified")


def check_resume(baseline, imagenet_dir, device):
    if baseline == "compare_pareto":
        return True, "skipped -- B5 has no eval loop"
    if baseline in IN_PROCESS_BASELINES:
        assert imagenet_dir, "--imagenet-dir required for the B1/B4 resume simulation"
        return _resume_simulation_inprocess(baseline, imagenet_dir, device)
    return check_resume_csv_mechanism(baseline)


# --- item 9: equivalence fingerprint ---------------------------------------

def check_equivalence_fingerprint(baseline, imagenet_dir, host_tag, python_exe):
    if baseline == "compare_pareto":
        return True, "skipped -- B5 has no eval loop"
    assert imagenet_dir, "--imagenet-dir required for the equivalence-fingerprint check"
    target_python = python_exe or sys.executable
    results = equivalence_gate.run_gate(imagenet_dir, host_tag or "colab", {baseline: target_python},
                                         baselines=[baseline])
    key = next(iter(results))
    m = results[key]
    return True, f"{key}: {m['correct_count']}/{m['n_images']} correct on the 256-image smoke set"


# --- item 10: gate ordering guard -------------------------------------------

def check_gate_ordering(results, checks_to_run):
    # Only judge checks that were actually IN SCOPE for this invocation --
    # a partial `--checks-only` selection (e.g. B5's 1,4,7) that excludes
    # both 6 and 9 must not be treated as if they silently failed.
    relevant = [n for n in (6, 9) if n in checks_to_run]
    if not relevant:
        return True, "skipped -- neither cost-ruler(6) nor equivalence-fingerprint(9) in scope for this invocation"
    missing = [n for n in relevant if n not in results or not results[n][0]]
    assert not missing, f"gate-ordering guard failed: check(s) {missing} did not pass earlier in this run"
    ran = " and ".join(f"{CHECK_NAMES[n].lower()} ({n})" for n in relevant)
    return True, (f"{ran} passed earlier in THIS self-test invocation, which runs before any --only-* "
                   "run cell in this notebook (a FAIL here halts \"Run All\" before any run cell "
                   "executes). FP-parity -> Step-A -> Step-B ordering within a real run is enforced by "
                   "run.py's own unchanged control flow.")


# --- orchestrator -----------------------------------------------------------

def run_self_test(baseline, imagenet_dir=None, python_exe=None, device="cuda", host_tag=None,
                   colab_py_path=None, repo_dir=None, checks_only=None):
    checks_to_run = checks_only or list(range(1, 11))
    colab_py_path = colab_py_path or str(BASELINES_ROOT / COLAB_PY_PATHS[baseline])
    repo_dir = repo_dir or str(BASELINES_ROOT.parent)
    results = {}

    print("=" * 70)
    print(f"TANDEM P2.2 POST-CONVERSION SELF-TEST -- {baseline}")
    print("=" * 70)

    def _run(n, fn):
        if n not in checks_to_run:
            return
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"{type(e).__name__}: {e}"
        results[n] = (passed, detail)
        print(f"[{n}/10] {CHECK_NAMES[n]:<28} {'PASS' if passed else 'FAIL'}  {detail}")

    _run(1, lambda: check_wrapper_purity_and_repo_head(colab_py_path, repo_dir))
    _run(2, lambda: check_upstream_pinning(baseline, python_exe))
    _run(3, lambda: check_numeric_environment(baseline, python_exe))
    _run(4, lambda: check_secrets(colab_py_path))
    _run(5, lambda: check_data_determinism(imagenet_dir, baseline))
    _run(6, check_cost_ruler)
    _run(7, check_schema)
    _run(8, lambda: check_resume(baseline, imagenet_dir, device))
    _run(9, lambda: check_equivalence_fingerprint(baseline, imagenet_dir, host_tag, python_exe))
    _run(10, lambda: check_gate_ordering(results, checks_to_run))

    meta_dir = BASELINES_ROOT / "_self_test_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{baseline}.json").write_text(json.dumps(
        {str(n): {"passed": p, "detail": d} for n, (p, d) in results.items()}, indent=2))

    failed = [n for n, (p, d) in results.items() if not p]
    print("=" * 70)
    if failed:
        print(f"SELF-TEST FAILED -- {len(failed)}/{len(results)} check(s) failed: {failed}")
        print("=" * 70)
        raise AssertionError(
            f"P2.2 self-test FAILED for {baseline}: check(s) {failed} did not pass -- see the banner "
            "above for details. Fix the underlying issue before proceeding to any run cell below; do "
            "not skip or comment out this cell."
        )
    print(f"ALL {len(results)} CHECKS PASSED -- safe to proceed to the run cells below.")
    print("=" * 70)
    return results


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker_numeric_env":
        passed, detail = _numeric_environment_inprocess()
        print("TANDEM_SELF_TEST_NUMERIC_ENV " + json.dumps({"passed": passed, "detail": detail}))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=list(COLAB_PY_PATHS.keys()))
    parser.add_argument("--imagenet-dir", default=None)
    parser.add_argument("--python", default=None,
                         help="pinned venv python for B2/B3/B5 (torch-free orchestrator); omit for "
                              "B1/B4, which launch this script directly with the venv's own python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-tag", default=None)
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--checks-only", default=None,
                         help="comma-separated check numbers, e.g. 1,4,7 (default: all 10)")
    args = parser.parse_args()

    checks_only = [int(x) for x in args.checks_only.split(",")] if args.checks_only else None
    run_self_test(
        baseline=args.baseline, imagenet_dir=args.imagenet_dir, python_exe=args.python,
        device=args.device, host_tag=args.host_tag, repo_dir=args.repo_dir, checks_only=checks_only,
    )


if __name__ == "__main__":
    main()
