"""Per-batch checkpoint/resume, for full 50k eval loops (B1, B4 -- our own
in-process loops, which we fully control; B2/B3 run an UNMODIFIED
upstream subprocess and cannot get the same treatment without violating
the fairness rule -- see AUDIT_REPORT.md for the full reasoning). Mirrors
P2's own checkpoint pattern (experiments/phase2_1_fleet/P2_tandem_pipeline.py's
`_load_checkpoint`, config-hash-guarded): a checkpoint whose hash doesn't
match this run's config is treated as stale and discarded, never
silently reused for the wrong config. Framework-free (no torch import).
"""
import json
from pathlib import Path


def load_checkpoint(ckpt_path, expected_hash):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return {"n_correct": 0, "n_total": 0, "n_batches_done": 0, "config_hash": expected_hash}
    with open(ckpt_path) as f:
        state = json.load(f)
    if state.get("config_hash") != expected_hash:
        print(f"Checkpoint at {ckpt_path} doesn't match this run's config_hash -- starting fresh "
              f"(found {state.get('config_hash')!r}, expected {expected_hash!r}).")
        return {"n_correct": 0, "n_total": 0, "n_batches_done": 0, "config_hash": expected_hash}
    print(f"Resuming from checkpoint at {ckpt_path}: {state['n_total']} images "
          f"({state.get('n_batches_done', 0)} batches) already scored.")
    return state


def save_checkpoint(ckpt_path, state):
    """Atomic write (write to a .tmp file, then os.replace) so a crash or
    Colab disconnect mid-write never leaves a corrupt/partial checkpoint
    file that `load_checkpoint` would fail to parse."""
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    tmp_path.replace(ckpt_path)
