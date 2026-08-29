"""Cross-host determinism utilities -- the single source of truth for
everything that could silently differ between the university GPU and
Colab (or between any two hosts). torch-dependent (only imported inside
a baseline's own pinned venv, matching common/imagenet.py's pattern) --
except `config_hash`, which stays framework-free so the orchestration
environment (common/equivalence_gate.py) can call it too.

The #1 silent cross-host numeric difference for this project: university
Ampere-class GPUs (A100/A6000/etc.) enable TF32 matmul/cudnn by DEFAULT;
Colab's T4 (Turing) has no TF32 hardware path at all, so TF32 there is
already a no-op. Left at PyTorch's own default (TF32 ON on Ampere+), the
SAME code produces measurably different logits on the two hosts (TF32
truncates the matmul mantissa). Every runner in this project calls
`disable_tf32()` (in-process runners) or uses
`subprocess_env_with_tf32_disabled` (subprocess runners) before any
GPU work.
"""
import hashlib
import json
import os
import platform
from pathlib import Path

# Matches RepQ-ViT's and FQ-ViT's own test_quant.py default (verified from
# source at the pinned commits: both default `--seed` to 0 and call their
# own seed() first thing in main()) -- pinned here explicitly, in OUR
# code, rather than silently inherited from an upstream default that
# happens to also be 0.
SEED = 0


def disable_tf32():
    """Call before any CUDA op, for IN-PROCESS runners (B1, B4). Safe to
    call even without CUDA (the backend attributes exist regardless)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_everything(seed=SEED):
    """Same seeding recipe RepQ-ViT's and FQ-ViT's own test_quant.py use
    (verified from source at the pinned commits) -- applied here for
    B1/B4, which run in-process rather than as a subprocess of a script
    that already seeds itself."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolved_host_info(host_tag=None):
    """torch/cuda/GPU actually resolved on THIS host -- recorded in every
    results.csv row (audit item A.3) so a future numeric discrepancy has
    an immediate first suspect. `host_tag` should be passed explicitly by
    the caller ("university_gpu" / "colab") -- falls back to the node
    hostname, which is a weaker but still-informative signal."""
    import torch
    return {
        "host": host_tag or os.environ.get("TANDEM_HOST_TAG") or platform.node() or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }


def config_hash(*parts):
    """sha256 (first 16 hex chars) over a JSON-canonicalized tuple of
    hashable parts (commit SHAs, config dict, seed, calibration source
    descriptor, ...) -- one hash identifies exactly what produced a given
    results.csv row. Framework-free (no torch import) so
    common/equivalence_gate.py can call it from the orchestration
    environment too. `default=str` handles any non-JSON-native values
    (e.g. Path objects) by falling back to their string form."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


SITECUSTOMIZE_SOURCE = '''\
"""Auto-imported by Python's `site` module at interpreter startup (see
PYTHONPATH injection in common/determinism.subprocess_env_with_tf32_disabled)
-- disables TF32 for every subsequent torch import in THIS process, before
the target script (test_quant.py) ever runs a single line. This is NOT a
modification of the upstream repo's own files -- their test_quant.py is
invoked completely unmodified; this only changes what PYTHONPATH the
subprocess sees, a standard, non-invasive Python mechanism."""
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except ImportError:
    pass
'''


def write_sitecustomize(dest_dir, extra_source=""):
    """Writes sitecustomize.py into `dest_dir` (idempotent -- overwrites
    with the same content every call) and returns the directory. Callers
    prepend this directory to PYTHONPATH for a subprocess call so TF32
    gets disabled inside that subprocess without touching the target
    script's own source on disk.

    `extra_source`: additional Python source appended after the TF32-disable
    block -- used by common/checkpoint_pin.py to ALSO inject the pinned-
    checkpoint override into the SAME sitecustomize.py, since Python's
    `site` module only auto-imports one module actually named
    `sitecustomize` (the first found on PYTHONPATH) -- two separate
    sitecustomize.py files on the same PYTHONPATH would silently shadow
    each other, so every subprocess-level injection in this project shares
    this one file/mechanism rather than each writing its own."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_SOURCE + extra_source)
    return dest_dir


def _cli_host_info():
    """Entry point for `python -m common.determinism host-info`, run via
    subprocess using the TARGET baseline's own pinned venv python -- B2's
    and B3's run.py are themselves torch-free and cannot call
    `resolved_host_info()` directly (see their module docstrings), so
    this is invoked the same way they invoke test_quant.py itself: as a
    subprocess of the target interpreter. Prints JSON to stdout."""
    print(json.dumps(resolved_host_info()))


def subprocess_env_with_tf32_disabled(sitecustomize_dir, base_env=None, extra_sitecustomize_source=""):
    """Returns an env dict for subprocess.run(..., env=...) with
    `sitecustomize_dir` prepended to PYTHONPATH (see write_sitecustomize),
    plus CUBLAS_WORKSPACE_CONFIG set. FQ-ViT's own seed() already sets
    CUBLAS_WORKSPACE_CONFIG=':4096:8' (verified from source) for cuBLAS
    determinism; RepQ-ViT's own seed() does NOT -- setting it here for
    BOTH is a determinism-only addition (which cuBLAS algorithm variant
    gets selected), not a protocol/transform change, so it doesn't
    conflict with the fairness rule (../deviations.md notes this
    explicitly as an added-by-us determinism knob, not an upstream
    setting).

    `extra_sitecustomize_source`: passed through to write_sitecustomize --
    lets a caller (common/checkpoint_pin.py, for the Step-B pinned-checkpoint
    override) fold additional subprocess-startup patching into this SAME
    sitecustomize.py rather than writing a competing one."""
    env = dict(base_env if base_env is not None else os.environ)
    write_sitecustomize(sitecustomize_dir, extra_source=extra_sitecustomize_source)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(sitecustomize_dir) + (os.pathsep + existing if existing else "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "host-info":
        _cli_host_info()
    else:
        print("usage: python -m common.determinism host-info", file=sys.stderr)
        sys.exit(1)
