"""Reconstructs the EXACT calibration image indices RepQ-ViT's and
FQ-ViT's own test_quant.py select internally, WITHOUT modifying either
upstream script -- a pure side computation that replicates their exact
seed() -> ImageFolder(train) -> DataLoader(shuffle=True) call sequence
(verified against their real source at the pinned commits -- see
../deviations.md) and reads the RandomSampler's OWN index sequence
directly, rather than guessing at it.

Why this is provably equivalent, not just plausible: `DataLoader(...,
shuffle=True)` is implemented as an alias for `sampler=
RandomSampler(dataset)` (both call sites use `generator=None`) -- so
constructing our OWN `RandomSampler(dataset)` after replicating their
exact seed() call produces the IDENTICAL index sequence their real
subprocess run uses internally, PROVIDED nothing else consumes the global
torch RNG between seed() and the sampler's first iteration in EITHER
path. Checked directly against both scripts' real source: neither does
any RNG-consuming work between `seed(args.seed)` and `train_loader`'s
construction/first iteration. num_workers doesn't affect this: index
SELECTION happens in the main process (the sampler); only the
__getitem__ FETCH is parallelized across workers -- so this capture step
safely uses num_workers=0 for simplicity while still recovering the
identical indices a real num_workers=16 run uses.

This is READ-ONLY instrumentation: it runs as a separate, cheap pass
BEFORE the real test_quant.py subprocess, never touches its stdout/exit
code, and its output is purely an audit artifact (referenced from
results.csv's notes / config_hash).
"""
import json
from pathlib import Path


def _seed_repq_style(seed):
    """Verbatim re-implementation of RepQ-ViT's own `seed()` (classification/
    test_quant.py, pinned commit) -- must match EXACTLY or the recovered
    indices would not correspond to their real run."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_fqvit_style(seed):
    """Verbatim re-implementation of FQ-ViT's own `seed()` (test_quant.py,
    pinned commit) -- must match EXACTLY, same reason as above."""
    import os
    import random
    import sys
    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


_SEED_FNS = {"repq_vit": _seed_repq_style, "fq_vit": _seed_fqvit_style}


def capture_calib_indices(method, imagenet_dir, seed, calib_batchsize, n_batches, out_path):
    """method: "repq_vit" | "fq_vit". Writes a JSON record of
    {"index": int, "path": str, "class_idx": int} for the exact
    calib_batchsize*n_batches images their real run calibrates on, to
    `out_path`. Returns the records list. Requires torch/torchvision --
    call this from inside the baseline's own pinned venv, same as the
    real test_quant.py subprocess."""
    import torch
    from torchvision.datasets import ImageFolder

    assert method in _SEED_FNS, f"unknown method {method!r}, expected one of {list(_SEED_FNS)}"
    _SEED_FNS[method](seed)

    train_dir = Path(imagenet_dir) / "train"
    # No transform needed -- only the SAMPLING ORDER matters here, and
    # ImageFolder's .samples list (used below) is built before any
    # transform is ever applied, so omitting one doesn't affect indices.
    dataset = ImageFolder(str(train_dir))
    sampler = torch.utils.data.RandomSampler(dataset)  # generator=None, matches shuffle=True internals exactly

    n_needed = calib_batchsize * n_batches
    idx_iter = iter(sampler)
    indices = [next(idx_iter) for _ in range(n_needed)]
    records = [
        {"index": i, "path": dataset.samples[i][0], "class_idx": dataset.samples[i][1]}
        for i in indices
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "method": method, "seed": seed, "calib_batchsize": calib_batchsize,
            "n_batches": n_batches, "n_images": n_needed, "records": records,
        }, f, indent=2)
    print(f"Captured {len(records)} calibration image indices ({method}, seed={seed}) -> {out_path}")
    return records


def _cli():
    """Entry point for `python -m common.calib_capture ...`, run via
    subprocess using the TARGET baseline's own pinned venv python -- B2's
    and B3's run.py are themselves torch-free (see their module
    docstrings) and cannot import torch/torchvision directly, so this
    capture step is invoked the same way they invoke test_quant.py
    itself: as a subprocess of the target interpreter."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(_SEED_FNS))
    parser.add_argument("--imagenet-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calib-batchsize", type=int, required=True)
    parser.add_argument("--n-batches", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    capture_calib_indices(args.method, args.imagenet_dir, args.seed, args.calib_batchsize,
                           args.n_batches, args.out)


if __name__ == "__main__":
    _cli()
