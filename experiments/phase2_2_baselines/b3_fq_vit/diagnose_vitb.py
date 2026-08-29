"""Orchestrator for common/vitb_diagnostic.py -- one-off investigation of
the real ViT-B Step-B FP-parity failure (measured=84.4660%, P2.1's own
S8=85.1000%, diff=0.6340pp exceeds the 0.15pp gate; see run.py's real
output and deviations.md). NOT part of the normal B3 pipeline (run.py is
unchanged by this file) -- run this once, by hand, per the 2026-07 ruling's
item 1, before any fix is attempted.

Torch-free (same reasoning as run.py's own module docstring): this script
only resolves the FQ-ViT repo path, downloads the pinned checkpoint (pure
stdlib urllib, see common/checkpoint_pin.download_hf_pytorch_bin), and
subprocess-invokes common/vitb_diagnostic.py (the torch-dependent worker)
with the SAME PYTHONPATH / cwd / TF32-disabled environment run.py's own
run_fp_parity_gate() already uses for the fq_vit method -- no new
subprocess plumbing invented here.

Usage (from the repo root, same invocation convention as run.py itself --
see B3_fq_vit_colab.py):

    python experiments/phase2_2_baselines/b3_fq_vit/diagnose_vitb.py --imagenet-dir {IMAGENET_ROOT} \\
        --python {VENV_B3_PYTHON}

Add --skip-plain-pass to run only checks 1+2 (transform side-by-side +
checkpoint-key diagnostic -- seconds, no GPU, no full-50k pass).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.checkpoint_pin import download_hf_pytorch_bin, pinned_checkpoint_tag  # noqa: E402
from common.determinism import subprocess_env_with_tf32_disabled  # noqa: E402
from common.repo_utils import ensure_repo  # noqa: E402
from common.subprocess_utils import run_streaming  # noqa: E402

FQVIT_COMMIT = "5daf5915c715dc21d6afcd64e661652030e640ab"  # same pin as run.py -- see its FQVIT_COMMIT
BASELINES_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", default=None, help="required unless --skip-plain-pass")
    parser.add_argument("--python", default=sys.executable, help="python executable of b3_fq_vit's own pinned venv")
    parser.add_argument("--skip-plain-pass", action="store_true",
                         help="run only checks 1+2 (transform + checkpoint keys) -- seconds, no GPU")
    args = parser.parse_args()
    if not args.skip_plain_pass:
        assert args.imagenet_dir, "--imagenet-dir required unless --skip-plain-pass"

    repo_path = ensure_repo("https://github.com/megvii-research/FQ-ViT", FQVIT_COMMIT, local_name="FQ-ViT")

    pinned_hf_repo_id = f"timm/{pinned_checkpoint_tag('vit_base_patch16_224')}"
    checkpoint_path = download_hf_pytorch_bin(pinned_hf_repo_id)

    sitecustomize_dir = BASELINES_ROOT / "_sitecustomize"
    env = subprocess_env_with_tf32_disabled(sitecustomize_dir)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BASELINES_ROOT) + (os.pathsep + existing_pp if existing_pp else "")

    cmd = [args.python, "-m", "common.vitb_diagnostic", "--checkpoint-path", str(checkpoint_path)]
    if args.skip_plain_pass:
        cmd.append("--skip-plain-pass")
    else:
        cmd += ["--imagenet-dir", args.imagenet_dir]
    print(f"Running: {' '.join(cmd)}  (cwd={repo_path})")
    result = run_streaming(cmd, cwd=str(repo_path), env=env)
    assert result.returncode == 0, f"Diagnostic subprocess FAILED (exit {result.returncode}):\n{result.stdout[-3000:]}"


if __name__ == "__main__":
    main()
