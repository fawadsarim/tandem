"""Batched top-1 eval loop for OUR OWN eval loop (B1, B4). torch-dependent
-- see common/imagenet.py's module docstring for which venvs import this.
"""
import time

import torch
from tqdm.auto import tqdm

from common.checkpoint import load_checkpoint, save_checkpoint


EARLY_ABORT_CHECK_BATCHES = 3
EARLY_ABORT_MAX_TOP1 = 0.03  # 3% -- generous headroom above 1000-class random chance (~0.1%)


def evaluate_top1(model, val_loader, device="cuda", desc="eval"):
    """Non-resumable single pass -- kept for callers that don't need
    checkpointing (e.g. a cheap smoke eval). Full 50k runs should use
    `evaluate_top1_resumable` instead (see its docstring for why).

    Early-abort sanity check, first: this is the SAME check
    `common.fp_parity_gate._plain_eval_loop` applies for B2/B3 -- accuracy
    over the first few batches is checked against a generous floor before
    continuing, so a catastrophically broken harness (wrong labels, wrong
    model, corrupt data -- e.g. the scrambled class-directory bug once
    found in common/colab_imagenet.py) fails in seconds, not minutes, on
    B1/B4 too. NOT the real gate -- that's `check_fp_parity`'s tight
    0.15pp tolerance after the full pass."""
    model.eval()
    n_correct = n_total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=desc)):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.shape[0]
            if batch_idx + 1 == EARLY_ABORT_CHECK_BATCHES:
                early_top1 = n_correct / n_total
                if early_top1 < EARLY_ABORT_MAX_TOP1:
                    raise RuntimeError(
                        f"Early sanity abort: top1={early_top1 * 100:.2f}% over the first {n_total} "
                        "images is consistent with near-random guessing (1000-class chance level "
                        "~0.1%), not a working model+data pipeline. Aborting before the full pass -- "
                        "check the ImageFolder's class-directory naming/label mapping (see "
                        "common/colab_imagenet.py's and common/imagenet.py's module docstrings) "
                        "before re-running."
                    )
    elapsed_s = time.perf_counter() - t0
    top1 = n_correct / n_total
    print(f"[{desc}] top1={top1*100:.3f}%  n={n_total}  ({elapsed_s/60:.1f} min)")
    return top1, n_total, elapsed_s


def evaluate_top1_resumable(model, val_loader, ckpt_path, config_hash, device="cuda", desc="eval",
                             save_every_n_batches=10):
    """Per-batch-checkpointed version, for full 50k runs (B1, B4 -- our
    own in-process loops, which we fully control; B2/B3 run an
    UNMODIFIED upstream subprocess and cannot get the same treatment
    without violating the fairness rule -- see AUDIT_REPORT.md).

    Mirrors P2's own checkpoint pattern (experiments/phase2_1_fleet/P2_tandem_pipeline.py):
    config-hash-guarded (a checkpoint from a DIFFERENT config -- different
    model, r-value, bit-width, ... -- is detected via the hash mismatch
    and discarded, never silently reused for the wrong config), resumed
    by re-iterating `val_loader` from the start and SKIPPING (not
    re-scoring) batches already reflected in the checkpoint -- `val_loader`
    has `shuffle=False` (common/imagenet.py), so this skip lands on
    EXACTLY the same images every time, deterministically. Skipping still
    pays the data-loading cost for already-done batches but not the GPU
    forward pass, the actually expensive part."""
    model.eval()
    state = load_checkpoint(ckpt_path, config_hash)
    n_correct, n_total, resume_from_batch = state["n_correct"], state["n_total"], state.get("n_batches_done", 0)

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=desc)):
            if batch_idx < resume_from_batch:
                continue
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.shape[0]

            if (batch_idx + 1) % save_every_n_batches == 0:
                save_checkpoint(ckpt_path, {
                    "n_correct": n_correct, "n_total": n_total, "n_batches_done": batch_idx + 1,
                    "config_hash": config_hash,
                })
    save_checkpoint(ckpt_path, {
        "n_correct": n_correct, "n_total": n_total, "n_batches_done": batch_idx + 1,
        "config_hash": config_hash,
    })
    elapsed_s = time.perf_counter() - t0
    top1 = n_correct / n_total
    print(f"[{desc}] top1={top1*100:.3f}%  n={n_total}  ({elapsed_s/60:.1f} min, resumed from "
          f"batch {resume_from_batch})")
    return top1, n_total, elapsed_s
