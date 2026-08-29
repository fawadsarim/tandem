"""B1: ToMe (facebookresearch/ToMe), DeiT-S, r in {8, 13, 16}.

Single runner: clones+pins ToMe at its exact commit (checked out AND
verified at runtime -- see common.repo_utils.ensure_repo), patches a
fresh DeiT-S with ToMe's `prop_attn=True` (their documented DeiT
convention -- see ../deviations.md #4), evaluates top-1 on the full 50k
ImageNet-1k val set, measures the REAL per-block token trajectory via a
live forward hook (common.bops.measure_token_trajectory) rather than
reverse-engineering ToMe's internal `parse_r` schedule formula, and
computes cost under TANDEM's own BOPs model so it lands on TANDEM's
Pareto axes.

## Two SETTINGS, never comparable, never conflated (see ../deviations.md,
## "setting mismatch" entry, corrected 2026-07)

ToMe's README/paper publish DeiT-S numbers for TWO fundamentally
different things that happen to share the same r=13 label:
  - "off_the_shelf": OUR run -- an off-the-shelf pretrained DeiT-S
    checkpoint (fb_in1k, no ToMe involvement in training), ToMe applied
    POST-HOC at eval time only. This is what every row in the main r-sweep
    below measures, and what compare_pareto.py must read.
  - "trained_with_tome": THEIR run -- a DIFFERENT DeiT-S checkpoint,
    fine-tuned FROM SCRATCH WITH ToMe merging active throughout training
    (README: "We've trained some DeiT (v1) models using the official
    implementation"). Only located for r=13, as a downloadable checkpoint
    (deit_S_r13.pth) with a published number (79.36%, README's DeiT
    table). Evaluating that checkpoint is a real, useful reproduction
    check (proves this harness's ToMe-patched eval path against a
    published number in the MATCHING setting) but its result is NOT a
    valid comparison point for our off-the-shelf sweep -- see
    published.json's own "note" field for the full paper-appendix search
    (Table 11 is trained-only too; no off-the-shelf DeiT-S r-sweep exists
    anywhere in ToMe's own materials).

Two BINDING gates (both HALT on failure, same as every other gate in this
project -- never silently record a bad number):
  (a) r0_baseline: the plain/unpatched off-the-shelf checkpoint, already
      evaluated by the FP-parity gate below, additionally gated against
      ToMe's own stated 79.8 baseline (not just P2.1's S8 79.848) --
      certifies this harness's eval path (preprocessing, loader, labels)
      against ToMe's own reported number in the ONE setting where "off
      the shelf" and "their baseline" genuinely mean the same thing.
  (b) r13 trained_with_tome: downloads deit_S_r13.pth, loads it onto a
      ToMe-patched model exactly per the README's instructions, evaluates
      on the same 50k val set, gates against 79.36%. Certifies the
      ToMe-patched eval path (the merge machinery itself) against a
      published number in a setting where the checkpoint genuinely
      matches.
None of r=8/r=13(off_the_shelf)/r=16 (the main sweep below) are gated --
no valid off-the-shelf reference exists for any of them (see
published.json) -- and are reported with gate_status=no_published_reference,
same as before.

Determinism: TF32 disabled (common.determinism.disable_tf32) before any
CUDA op -- the #1 silent cross-host numeric difference (Ampere GPUs
default TF32 on, Colab's T4 has no TF32 path at all). Seeded
(common.determinism.seed_everything) with the same recipe RepQ-ViT/FQ-ViT
use internally (verified from their source). host/gpu/torch/cuda
resolved and recorded in every results.csv row. Full 50k eval is
per-batch checkpointed (common.eval_utils.evaluate_top1_resumable) so a
Colab disconnect resumes rather than restarts.

## Per-step invocation and resume (Colab session-limit hardening)

`--only-r` restricts a single invocation to ONE r value, so a Colab
notebook cell can run r=8/13/16 as three SEPARATE, independently-timed
cells instead of one long one -- a crash or session limit hit partway
through no longer costs the r-values that already finished. Omit the flag
(university-GPU default) to run the full sweep exactly as before, in one
process, unchanged. Independently of `--only-r`, EVERY invocation first
checks `results.csv` (common.results_io.find_existing_row) for that r's
row already recorded and skips straight past it if so -- this is what
actually makes re-running a completed cell (or a full "Run All" after a
partial failure) a safe no-op rather than a silent results.csv duplicate,
the same coarse per-config resume mechanism B2/B3 already use. The
FP-parity gate (below) is separately cached to `_fp_parity_cache/` so it
is also never re-paid across a multi-cell run.

Environment: run inside b1_tome/'s own venv (requirements.txt exact-pins
torch==1.12.1, timm==0.4.12 -- ToMe's own documented floor, NOT TANDEM's
modern timm). `import tome` after this script inserts the pinned clone
onto sys.path -- do not `pip install tome` from PyPI (not the same
package; ToMe is used directly from source, as the upstream repo
documents).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bops import cost_from_trajectory, measure_token_trajectory, tome_schedule_n_l  # noqa: E402
from common.checkpoint_pin import _download_progress_hook, coincides_with_pinned, PAPER_CHECKPOINT_TAG  # noqa: E402
from common.determinism import config_hash, disable_tf32, resolved_host_info, seed_everything, SEED  # noqa: E402
from common import equivalence_gate  # noqa: E402
from common.eval_utils import evaluate_top1_resumable  # noqa: E402
from common.fp_parity_gate import fp_parity_check_inprocess  # noqa: E402
from common.imagenet import build_val_loader, get_sample_batch  # noqa: E402
from common.repo_utils import ensure_repo  # noqa: E402
from common.repro_gate import check_gate, format_gate_banner  # noqa: E402
from common.results_io import append_result, find_existing_row  # noqa: E402
from common.startup_checks import (  # noqa: E402
    assert_dual_host_audit_items_present, assert_val_set_matches_canonical_p2, assert_valid_commit_sha,
)

TOME_COMMIT = "af95e4b1befa172dadccd8c81e223b10090f9579"
MODEL_NAME = "deit_small_patch16_224"
D_MODEL = 384
N_BLOCKS = 12
N_TOKENS_TOTAL = 197
R_VALUES = (8, 13, 16)

# DeiT-S: Step A (ToMe's own paper checkpoint) and Step B (TANDEM's pinned
# comparison checkpoint) are the SAME weights (fb_in1k) -- verified in
# common/checkpoint_pin.py's module docstring. B1 therefore never runs
# twice for a checkpoint-identity reason; every row is written with BOTH
# row_kind="reproduction" and row_kind="comparison" (identical measured
# numbers, one config_hash), not two separate GPU passes.
assert coincides_with_pinned(MODEL_NAME), (
    f"checkpoint_pin.py says {MODEL_NAME}'s Step A/B checkpoints do NOT coincide -- B1 (ToMe, "
    "DeiT-S only) was written assuming they always do. If this assertion fires, the checkpoint-"
    "identity policy changed and this runner's row-duplication logic needs revisiting, not bypassing."
)


def _off_the_shelf_ref(pub_entry):
    """published.json entries are FLAT for r8/r16 ({"top1_pct":...,
    "gate":...}) but NESTED for r13 ({"off_the_shelf": {...},
    "trained_with_tome": {...}}), since r13 is the only config with a
    located reference in BOTH settings. This normalizes both shapes to
    "the off-the-shelf reference dict" for the main sweep loop, which only
    ever cares about the off-the-shelf setting."""
    return pub_entry.get("off_the_shelf", pub_entry)


def _download_tome_checkpoint(url, cache_dir):
    """Downloads a ToMe-published trained checkpoint (e.g. deit_S_r13.pth)
    to `cache_dir`, idempotent (skips if already cached and non-trivially
    sized) -- same tmp-file-then-replace safety pattern as
    common/checkpoint_pin.py's download_hf_pytorch_bin, duplicated rather
    than imported because this downloads from dl.fbaipublicfiles.com
    directly (ToMe's own file server), not the HF Hub."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / Path(url).name
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        print(f"ToMe trained checkpoint already cached: {url} -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
    print(f"Downloading ToMe trained checkpoint: {url} -> {out_path}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path, reporthook=_download_progress_hook(f"Downloading {url}"))
    assert tmp_path.stat().st_size > 1_000_000, (
        f"Downloaded checkpoint from {url} is suspiciously small ({tmp_path.stat().st_size} bytes) -- "
        "likely an HTML error page, not real weights. Halting rather than caching a corrupt file."
    )
    tmp_path.replace(out_path)
    print(f"Cached ToMe trained checkpoint: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True, help="root with val/<wnid>/*.JPEG")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-tag", default=None,
                         help='e.g. "university_gpu" or "colab" -- recorded in results.csv; '
                              "falls back to the node hostname if omitted")
    parser.add_argument("--only-r", type=int, default=None, choices=list(R_VALUES),
                         help="restrict this invocation to a single r value from the off-the-shelf "
                              "sweep (e.g. one Colab cell per r, see module docstring); omit (with "
                              "--trained-checkpoint-r13 also omitted) to run everything -- FP-parity, "
                              "r0_baseline gate, the trained-checkpoint r13 gate, AND the full "
                              "{8,13,16} off-the-shelf sweep -- in one process, exactly as before")
    parser.add_argument("--trained-checkpoint-r13", action="store_true",
                         help="run ONLY the trained_with_tome r=13 reproduction gate (downloads "
                              "ToMe's own deit_S_r13.pth, gates against 79.36%%) -- a separate, "
                              "independently-timed Colab cell, mutually exclusive with --only-r")
    parser.add_argument("--skip-fp-parity", action="store_true",
                         help="DANGEROUS -- skips the FP-parity gate AND its dependent r0_baseline "
                              "reproduction-gate row entirely (no r0 row is written to results.csv "
                              "this run). Prints a loud warning and writes a "
                              "_fp_parity_cache/SKIPPED_*.json marker recording skip_fp_parity=true. "
                              "Default: gate ON. Only for fast iteration on a checkpoint already "
                              "known-good from a prior gated run -- never use for a number that will "
                              "be reported.")
    args = parser.parse_args()
    assert not (args.only_r is not None and args.trained_checkpoint_r13), (
        "--only-r and --trained-checkpoint-r13 are mutually exclusive -- the former scopes the "
        "off-the-shelf sweep to one r value, the latter runs the separate trained-checkpoint gate "
        "instead of the sweep entirely."
    )
    assert torch.cuda.is_available() or args.device == "cpu", "CUDA not available; pass --device cpu to override"

    assert_dual_host_audit_items_present(items=[
        "pinning", "tf32_off", "equivalence_fingerprint", "unified_schema", "resume",
    ])  # no "fixed_calib_indices" -- B1 does not calibrate anything (no PTQ)
    assert_valid_commit_sha(TOME_COMMIT, "TOME_COMMIT")

    disable_tf32()
    seed_everything(SEED)

    repo_path = ensure_repo("https://github.com/facebookresearch/ToMe", TOME_COMMIT, local_name="ToMe")
    sys.path.insert(0, str(repo_path))
    import tome  # noqa: E402  (must come after the sys.path insert above)

    with open(Path(__file__).parent / "published.json") as f:
        published = json.load(f)[MODEL_NAME]

    host_info = resolved_host_info(args.host_tag)
    print(f"Resolved host: {host_info}")

    # P2.2 audit item #4: pip-freeze hash of the ACTUAL venv producing this
    # run's rows -- B1 runs in-process, launched BY the pinned venv's own
    # python (module docstring's "Environment" section), so sys.executable
    # here already IS that venv; no separate subprocess venv to point at.
    pip_freeze_hash, _ = equivalence_gate._pip_freeze_hash(sys.executable)
    print(f"pip-freeze hash: {pip_freeze_hash}")

    print(f"Building val loader from {args.imagenet_dir} ...")
    # Architecture only (pretrained=False) -- resolving the correct
    # preprocessing (interpolation/crop_pct/mean/std) only needs
    # `pretrained_cfg` metadata, not real downloaded weights; every later
    # `timm.create_model(MODEL_NAME, pretrained=True)` call below resolves
    # to the identical config, since MODEL_NAME is fixed for this whole run.
    _cfg_model = timm.create_model(MODEL_NAME, pretrained=False)
    val_loader = build_val_loader(args.imagenet_dir, _cfg_model)
    sample_batch = get_sample_batch(args.imagenet_dir, _cfg_model)
    del _cfg_model

    # P2.2 audit item #3 ruling (b): confirm this baseline's own val set
    # matches P2.1's canonical count/source -- see common/startup_checks.py's
    # own docstring for why this (not a literal manifest-hash diff) is the
    # real check. Skipped under TANDEM_SMOKE_RUN -- the 256-image smoke
    # subset is deliberately NOT the full 50k, see equivalence_gate.py.
    if os.environ.get("TANDEM_SMOKE_RUN") != "1":
        assert_val_set_matches_canonical_p2(len(val_loader.dataset))

    checkpoint_tag = PAPER_CHECKPOINT_TAG[MODEL_NAME]  # == our pinned tag for DeiT -- see coincides_with_pinned above
    print(f"Step A/B checkpoint identity: {checkpoint_tag} (Step A == Step B for DeiT -- not run twice)")

    # FP-parity gate (spec item #2): plain, UNPATCHED model through this
    # same val_loader, once, before any r-value runs -- proves the harness
    # (preprocessing/eval loop) is sound on this checkpoint before trusting
    # any ToMe number built on top of it. Cached to _fp_parity_cache/ (a
    # cache HIT is always a genuine prior PASS -- only written after
    # fp_parity_check_inprocess succeeds below) so a --only-r-scoped Colab
    # cell for r=13/16 never re-pays this full-50k pass after r=8's cell
    # already paid it. Skipped entirely (not cached) when
    # TANDEM_SMOKE_RUN=1 -- set only by common/equivalence_gate.py's
    # smoke-equivalence check, where the 0.15pp tolerance (calibrated for
    # full-50k statistics) can never be reliably satisfied by a 256-image
    # subset regardless of correctness; see equivalence_gate.py's own
    # comment for the numbers.
    fp_cache_path = Path(__file__).parent / "_fp_parity_cache" / "tome_plain_timm.json"
    if args.skip_fp_parity:
        print("=" * 100)
        print("**** WARNING: FP-PARITY GATE SKIPPED (--skip-fp-parity) ****")
        print("The plain/unpatched model's accuracy was NOT verified against ToMe's own published")
        print("off-the-shelf baseline before any r>0 number below is computed. The r0_baseline")
        print("reproduction-gate row is NOT written this run. Every accuracy number produced by")
        print("this invocation is UNVERIFIED against the eval harness's own correctness -- do not")
        print("report any number from this run without re-running with the gate enabled.")
        print("=" * 100)
        skip_marker = (Path(__file__).parent / "_fp_parity_cache" /
                       f"SKIPPED_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        skip_marker.parent.mkdir(parents=True, exist_ok=True)
        skip_marker.write_text(json.dumps({
            "skip_fp_parity": True, "model": MODEL_NAME, "method": "tome",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
    elif os.environ.get("TANDEM_SMOKE_RUN") == "1":
        print("FP-parity gate SKIPPED (TANDEM_SMOKE_RUN set -- this is a smoke/equivalence-check "
              "invocation, not a real full-50k pass). NOT cached.")
    elif fp_cache_path.exists():
        print("FP-parity gate already passed (cached) -- skipping re-run.")
    else:
        plain_model = timm.create_model(MODEL_NAME, pretrained=True).to(args.device)
        fp_top1_pct, _ = fp_parity_check_inprocess(plain_model, val_loader, args.device, MODEL_NAME,
                                                     reference_top1_pct=79.848, harness_desc="tome_plain_timm")
        del plain_model
        fp_cache_path.parent.mkdir(parents=True, exist_ok=True)
        fp_cache_path.write_text("{}")

        # r0_baseline gate (binding, see module docstring item (a)): the
        # SAME plain-model pass, additionally gated against ToMe's own
        # stated 79.8 off-the-shelf baseline (not just P2.1's S8 79.848)
        # -- certifies this harness against ToMe's OWN reported number,
        # the one setting where "off the shelf" and "their baseline" mean
        # the same thing. No second GPU pass: reuses fp_top1_pct above.
        r0_ref = published["r0_baseline"]
        r0_existing = find_existing_row(method="tome", model=MODEL_NAME, config="r0", row_kind="reproduction")
        if r0_existing is not None:
            print(f"r0_baseline already in results.csv: top1={r0_existing['top1_50k']}%, "
                  f"gate_status={r0_existing['gate_status']} -- not re-writing.")
        else:
            r0_gate_status, r0_diff = check_gate(fp_top1_pct, r0_ref["top1_pct"])
            print(format_gate_banner("tome", MODEL_NAME, "r0", fp_top1_pct, r0_ref["top1_pct"],
                                      r0_gate_status, r0_diff))
            if os.environ.get("TANDEM_SMOKE_RUN") == "1":
                print(f"[smoke run] r0_baseline gate NOT enforced -- gate_status={r0_gate_status}, "
                      f"diff={r0_diff}")
            else:
                assert r0_gate_status != "fail", (
                    f"R0_BASELINE GATE FAILED for tome/{MODEL_NAME}/r0: measured={fp_top1_pct:.2f}%, "
                    f"published={r0_ref['top1_pct']:.2f}% (source: published.json, ToMe README 'original "
                    f"acc' column), diff={r0_diff:.3f}pp (tolerance=0.3pp). This means the off-the-shelf "
                    "eval path itself (preprocessing/loader/labels) disagrees with ToMe's own reported "
                    "baseline -- halting before trusting ANY r>0 number built on this same path."
                )
            append_result({
                "method": "tome", "model": MODEL_NAME, "config": "r0", "checkpoint_tag": checkpoint_tag,
                "row_kind": "reproduction", "setting": "off_the_shelf",
                "top1_50k": f"{fp_top1_pct:.4f}", "n_images": 50000,
                "bops_ours": f"{cost_from_trajectory(tome_schedule_n_l(0, N_BLOCKS, N_TOKENS_TOTAL), [16] * N_BLOCKS, D_MODEL):.6e}",
                "repo_commit": TOME_COMMIT, "published_top1_pct": r0_ref["top1_pct"],
                "gate_status": r0_gate_status, "gate_diff_pct_points": f"{r0_diff:.4f}",
                "calib_images": "", "host": host_info["host"], "gpu_name": host_info["gpu_name"],
                "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
                "config_hash": config_hash(TOME_COMMIT, MODEL_NAME, "r0", SEED),
                "pip_freeze_hash": pip_freeze_hash,
                "notes": "plain/unpatched off-the-shelf checkpoint, no ToMe merging (r=0) -- reuses the "
                         "FP-parity pass above, no second GPU pass; certifies the eval path against "
                         "ToMe's own stated baseline, gated (see module docstring item (a)).",
            })

    # Trained-checkpoint r=13 gate (binding, see module docstring item
    # (b)): skipped entirely for an --only-r cell (those are the
    # off-the-shelf sweep's per-r Colab cells, not this) -- runs for the
    # dedicated --trained-checkpoint-r13 cell, AND for a full default
    # sweep (both flags omitted), so a single university-GPU invocation
    # still covers everything in one process. Also skipped entirely (not
    # just un-asserted) under TANDEM_SMOKE_RUN -- same reasoning as the
    # FP-parity block above: common/equivalence_gate.py's smoke invocation
    # never passes --only-r, so without this guard a 256-image smoke
    # check would download and evaluate a full ~90MB real checkpoint for
    # no purpose (the smoke check's job is cross-host correct-count
    # equivalence on ToMe's OWN off-the-shelf path, not this gate).
    if args.only_r is None and os.environ.get("TANDEM_SMOKE_RUN") != "1":
        trained_ref = published["r13"]["trained_with_tome"]
        trained_checkpoint_tag = Path(trained_ref["checkpoint_url"]).stem
        trained_existing = find_existing_row(
            method="tome", model=MODEL_NAME, config="r13", row_kind="reproduction",
            checkpoint_tag=trained_checkpoint_tag,
        )
        if trained_existing is not None:
            print(f"trained_with_tome r13 already in results.csv: top1={trained_existing['top1_50k']}%, "
                  f"gate_status={trained_existing['gate_status']} -- skipping re-run.")
        else:
            ckpt_path = _download_tome_checkpoint(
                trained_ref["checkpoint_url"], Path(__file__).parent / "_pinned_checkpoints",
            )
            trained_model = timm.create_model(MODEL_NAME, pretrained=False).to(args.device)
            # Per README: "instantiate a DeiT timm model, patch it with the
            # timm patch (prop_attn=True), and use ImageNet mean and
            # variance for data loading" -- same prop_attn convention and
            # same val_loader (ImageNet mean/std) already built above for
            # the off-the-shelf path; only the WEIGHTS differ.
            tome.patch.timm(trained_model, prop_attn=True)
            trained_model.r = 13
            state_dict = torch.load(ckpt_path, map_location=args.device)
            if isinstance(state_dict, dict) and "model" in state_dict:
                # Standard facebookresearch/deit training-script checkpoint
                # shape ({'model':..., 'optimizer':..., 'epoch':...,
                # ...}) -- README says these were "trained ... using the
                # official [DeiT] implementation", which saves this way.
                state_dict = state_dict["model"]
            trained_model.load_state_dict(state_dict, strict=True)

            trained_config_hash = config_hash(TOME_COMMIT, MODEL_NAME, "r13_trained_with_tome", SEED,
                                               "prop_attn=True", trained_ref["checkpoint_url"])
            trained_checkpoint_dir = Path(os.environ.get(
                "TANDEM_CHECKPOINT_DIR_OVERRIDE", str(Path(__file__).parent / "_checkpoints"),
            ))
            trained_top1, trained_n, trained_elapsed_s = evaluate_top1_resumable(
                trained_model, val_loader, trained_checkpoint_dir / "r13_trained.json", trained_config_hash,
                device=args.device, desc="ToMe r=13 (trained_with_tome)",
            )
            trained_top1_pct = trained_top1 * 100
            trained_gate_status, trained_diff = check_gate(trained_top1_pct, trained_ref["top1_pct"])
            print(format_gate_banner("tome", MODEL_NAME, "r13 (trained_with_tome)", trained_top1_pct,
                                      trained_ref["top1_pct"], trained_gate_status, trained_diff))
            if os.environ.get("TANDEM_SMOKE_RUN") == "1":
                print(f"[smoke run] trained_with_tome r13 gate NOT enforced -- "
                      f"gate_status={trained_gate_status}, diff={trained_diff}")
            else:
                assert trained_gate_status != "fail", (
                    f"TRAINED-CHECKPOINT GATE FAILED for tome/{MODEL_NAME}/r13 (trained_with_tome): "
                    f"measured={trained_top1_pct:.2f}%, published={trained_ref['top1_pct']:.2f}% (source: "
                    f"published.json, ToMe README DeiT table, checkpoint={trained_ref['checkpoint_url']}), "
                    f"diff={trained_diff:.3f}pp (tolerance=0.3pp). Halting -- this must be diagnosed before "
                    "trusting the ToMe-patched eval path (the merge machinery itself) at all."
                )
            n_l_formula_trained = tome_schedule_n_l(13, N_BLOCKS, N_TOKENS_TOTAL)
            bops_trained = cost_from_trajectory(n_l_formula_trained, [16] * N_BLOCKS, D_MODEL)
            append_result({
                "method": "tome", "model": MODEL_NAME, "config": "r13", "checkpoint_tag": trained_checkpoint_tag,
                "row_kind": "reproduction", "setting": "trained_with_tome",
                "top1_50k": f"{trained_top1_pct:.4f}", "n_images": trained_n,
                "bops_ours": f"{bops_trained:.6e}", "repo_commit": TOME_COMMIT,
                "published_top1_pct": trained_ref["top1_pct"], "gate_status": trained_gate_status,
                "gate_diff_pct_points": f"{trained_diff:.4f}", "calib_images": "",
                "host": host_info["host"], "gpu_name": host_info["gpu_name"],
                "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
                "config_hash": trained_config_hash, "pip_freeze_hash": pip_freeze_hash,
                "notes": f"ToMe's OWN checkpoint ({trained_ref['checkpoint_url']}), fine-tuned WITH "
                         f"merging during training -- NOT comparable to the off-the-shelf rows below "
                         f"(see module docstring); wall_clock_s={trained_elapsed_s:.1f}",
            })
            del trained_model

    if args.trained_checkpoint_r13:
        return

    r_values_to_run = (args.only_r,) if args.only_r is not None else R_VALUES
    for r in r_values_to_run:
        config_key = f"r{r}"
        print(f"\n{'=' * 70}\nToMe {MODEL_NAME} r={r}\n{'=' * 70}")

        # Resume: skip an r already fully recorded in results.csv (e.g. a
        # prior --only-r cell, or a rerun of "Run All" after a later cell
        # failed) -- both row_kind rows are always written together right
        # after evaluate_top1_resumable finishes (see below), so a
        # reproduction-row hit means the comparison row is there too.
        existing = find_existing_row(
            method="tome", model=MODEL_NAME, config=config_key,
            row_kind="reproduction", checkpoint_tag=checkpoint_tag,
        )
        if existing is not None:
            print(f"Already in results.csv: tome/{MODEL_NAME}/{config_key} -- top1={existing['top1_50k']}%, "
                  f"gate_status={existing['gate_status']} -- skipping re-run (both row_kinds already recorded).")
            continue

        model = timm.create_model(MODEL_NAME, pretrained=True).to(args.device)
        # DeiT convention per ToMe's own README ("instantiate a DeiT timm
        # model, patch it with the timm patch (prop_attn=True)") -- the
        # generic non-DeiT ViT example in the same README omits it.
        tome.patch.timm(model, prop_attn=True)
        model.r = r

        this_config_hash = config_hash(TOME_COMMIT, MODEL_NAME, config_key, SEED, "prop_attn=True")
        # TANDEM_CHECKPOINT_DIR_OVERRIDE (set only by equivalence_gate.py's
        # smoke check) isolates a smoke run's checkpoint from a real run's --
        # config_hash alone does NOT encode dataset identity/size, so
        # without this a smoke run's 256-image checkpoint and a real run's
        # 50k-image checkpoint for the same r would otherwise collide.
        checkpoint_dir = Path(os.environ.get("TANDEM_CHECKPOINT_DIR_OVERRIDE", str(Path(__file__).parent / "_checkpoints")))
        ckpt_path = checkpoint_dir / f"{config_key}.json"
        top1, n, elapsed_s = evaluate_top1_resumable(
            model, val_loader, ckpt_path, this_config_hash, device=args.device, desc=f"ToMe r={r}",
        )
        top1_pct = top1 * 100

        n_l_per_block = measure_token_trajectory(model, sample_batch, device=args.device)
        # One cost ruler (spec item #3): the closed-form schedule is the
        # SOURCE OF TRUTH for cost (unit-tested at r=0 in
        # bops_identity_check.py); the live forward-hook measurement above
        # is still taken and cross-checked against it here -- a mismatch
        # would mean either the formula's inflect=0 assumption or ToMe's
        # real runtime behavior changed, and either way the cost number
        # must not be silently trusted.
        n_l_formula = tome_schedule_n_l(r, N_BLOCKS, N_TOKENS_TOTAL)
        assert n_l_formula == n_l_per_block, (
            f"ToMe N_l mismatch for r={r}: closed-form schedule gives {n_l_formula}, live forward-hook "
            f"measurement gives {n_l_per_block} -- these should be IDENTICAL at ToMe's default "
            "inflection=0 (verified from tome/utils.py's parse_r). Halting rather than reporting a "
            "BOPs number built on an unreconciled disagreement between the two."
        )
        # ToMe does not quantize anything -- FP16 baseline (bits=16) at
        # every block in TANDEM's bit-menu language, matching how the
        # PlanExecutor's own identity/FP16 blocks are cost-modeled.
        bits_per_block = [16] * len(n_l_formula)
        bops_ours = cost_from_trajectory(n_l_formula, bits_per_block, D_MODEL)

        # published.json is FLAT for r8/r16 but NESTED for r13 (the only
        # config with a located reference in BOTH settings) -- see
        # _off_the_shelf_ref's own docstring. Every config in THIS loop is
        # the off-the-shelf setting; none currently gate (see published.json).
        pub = _off_the_shelf_ref(published.get(config_key, {}))
        published_top1 = pub.get("top1_pct")
        gate_status, diff = check_gate(top1_pct, published_top1)
        print(format_gate_banner("tome", MODEL_NAME, config_key, top1_pct, published_top1, gate_status, diff))
        # A located published reference exists AND was missed -- halt for
        # manual review rather than silently recording a failed
        # reproduction as if it were just another data point. NOT enforced
        # under TANDEM_SMOKE_RUN (set only by equivalence_gate.py): the
        # 0.3pp tolerance is calibrated for full-50k statistics, same
        # incompatibility as the FP-parity gate's 0.15pp (see
        # equivalence_gate.py's own comment for the numbers) -- a 256-image
        # smoke subset can never reliably satisfy it regardless of
        # correctness. Still computed and printed either way (below), just
        # not asserted, so it remains visible for a real run.
        if os.environ.get("TANDEM_SMOKE_RUN") == "1":
            print(f"[smoke run] reproduction gate NOT enforced (tol=0.3pp not meaningful at this sample "
                  f"size) -- gate_status={gate_status}, diff={diff}")
        else:
            assert gate_status != "fail", (
                f"REPRODUCTION GATE FAILED for tome/{MODEL_NAME}/{config_key}: measured={top1_pct:.2f}%, "
                f"published={published_top1:.2f}% (source: published.json, ToMe README DeiT table), diff="
                f"{diff:.3f}pp (tolerance=0.3pp). Halting -- this must be diagnosed before any downstream "
                "comparison (compare_pareto.py) treats this number as trustworthy."
            )
        print(f"Step-A reproduction-gate line: tome/{MODEL_NAME}/{config_key} -> {gate_status} "
              f"(checkpoint={checkpoint_tag})")

        base_row = {
            "method": "tome", "model": MODEL_NAME, "config": config_key, "checkpoint_tag": checkpoint_tag,
            "setting": "off_the_shelf",
            "top1_50k": f"{top1_pct:.4f}", "n_images": n, "bops_ours": f"{bops_ours:.6e}",
            "repo_commit": TOME_COMMIT,
            "published_top1_pct": "" if published_top1 is None else published_top1,
            "gate_status": gate_status, "gate_diff_pct_points": "" if diff is None else f"{diff:.4f}",
            "calib_images": "",
            "host": host_info["host"], "gpu_name": host_info["gpu_name"],
            "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
            "config_hash": this_config_hash, "pip_freeze_hash": pip_freeze_hash,
        }
        # row_kind duplication (spec item #1): Step A/B coincide for DeiT
        # (see coincides_with_pinned assert above) -- one measurement,
        # two results.csv rows, never a second GPU pass.
        append_result({**base_row, "row_kind": "reproduction",
                        "notes": f"Step A (== Step B, checkpoints coincide for DeiT); prop_attn=True; "
                                 f"n_l_per_block={n_l_formula} (closed-form, cross-checked against live "
                                 f"hook); wall_clock_s={elapsed_s:.1f}"})
        append_result({**base_row, "row_kind": "comparison",
                        "notes": "Step B: comparison row reuses Step A's measurement (checkpoint "
                                 "identity coincides for DeiT -- not re-run); see row_kind=reproduction "
                                 "row (same config_hash) for the full notes."})


if __name__ == "__main__":
    main()
