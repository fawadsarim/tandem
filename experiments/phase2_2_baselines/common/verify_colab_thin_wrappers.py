"""Grep-assert that every *_colab.py notebook source contains NO
eval/preprocessing/calibration logic of its own -- only bootstrap
(secrets, mounts, clone, pip install, invoke). Run standalone via
`python -m common.verify_colab_thin_wrappers` (from experiments/phase2_2_baselines/); also
callable from a CI-style check. Framework-free.

This is the single-source-of-truth requirement (item B.1): all baseline
logic lives ONLY in experiments/phase2_2_baselines/ as importable modules/scripts, and every
Colab notebook is a thin wrapper that clones the repo and calls the SAME
runner entry points the university GPU uses -- never a second
implementation that could silently drift from the first.
"""
import re
import sys
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parent.parent

# Substrings that would only plausibly appear in REAL eval/model/
# preprocessing/calibration logic -- none of these belong in a notebook
# whose only job is to clone, pip install, and shell out to run.py.
FORBIDDEN_PATTERNS = [
    r"^\s*def ",            # function definitions (bootstrap needs none)
    r"^\s*class ",           # class definitions
    r"torch\.no_grad",        # a real forward pass
    r"\.argmax\(",              # decoding a prediction
    r"nn\.Module",                # a model definition
    r"CenterCrop|RandomCrop",      # a transform pipeline
    r"transforms\.Normalize",       # a transform pipeline
    r"ImageFolder\(",                 # a dataset construction (materialize_* calls are fine, they're imported)
    r"quant_model|set_quant_state",     # RepQ-ViT quantization calls
    r"\.forward\(",                       # a manual forward call
    r"DataLoader\(",                        # a data loader construction
]
FORBIDDEN_RE = [re.compile(p) for p in FORBIDDEN_PATTERNS]

# These calls are legitimate bootstrap (materializing ImageNet from HF
# streaming so run.py has something to point --imagenet-dir at) -- not
# duplicated EVAL logic, so they're allowed even though they touch
# ImageNet data.
ALLOWED_CALL_PREFIXES = ("materialize_val_split(", "materialize_train_calib_subset(")


def check_file(path):
    violations = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in ALLOWED_CALL_PREFIXES):
            continue
        for pattern in FORBIDDEN_RE:
            if pattern.search(line):
                violations.append((lineno, line.strip(), pattern.pattern))
    return violations


def main():
    colab_files = sorted(BASELINES_ROOT.glob("**/*_colab.py"))
    assert colab_files, f"no *_colab.py files found under {BASELINES_ROOT} -- check the glob/path"

    all_violations = {}
    for f in colab_files:
        violations = check_file(f)
        if violations:
            all_violations[f] = violations

    if all_violations:
        for f, violations in all_violations.items():
            print(f"\n{f}:")
            for lineno, line, pattern in violations:
                print(f"  line {lineno} (matched /{pattern}/): {line}")
        raise AssertionError(
            f"{sum(len(v) for v in all_violations.values())} forbidden pattern(s) found in "
            f"{len(all_violations)} Colab notebook source(s) -- see above. Eval/preprocessing/"
            "calibration logic belongs ONLY in experiments/phase2_2_baselines/ (run.py + common/), never duplicated "
            "into a notebook cell."
        )

    print(f"Checked {len(colab_files)} Colab notebook source(s): {[f.name for f in colab_files]}")
    print("All are thin wrappers -- no eval/preprocessing/calibration logic found.")


if __name__ == "__main__":
    main()
