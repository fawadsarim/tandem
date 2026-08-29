"""Colab-only helper: materializes a torchvision-ImageFolder-compatible
local ImageNet-1k copy (`<root>/{train,val}/<class dir>/*.JPEG`) by
streaming `ILSVRC/imagenet-1k` from the HF Hub, so the EXISTING baseline
runners (`b1_tome/run.py`, `b2_repq_vit/run.py`, `b3_fq_vit/run.py`,
`b4_sequential_composites/run.py` -- all of which expect a local
ImageFolder via `--imagenet-dir`) can run completely UNMODIFIED inside a
standalone Colab notebook. The university-GPU path's `IMAGENET_DIR`
contract (see ../deviations.md #2) is preserved exactly; only HOW that
directory gets populated differs when there is no local ImageNet-1k copy
available (e.g. university GPU access is unavailable).

Class-directory naming -- CRITICAL for label correctness, not cosmetic:
directories are named by the raw zero-padded integer label
(`f"{label:04d}"`), NEVER by `ClassLabel.int2str(label)`. HF's own
documentation states the raw int label is "indexed based on a sorted list
of synset ids" -- i.e. label 0..999 is ALREADY assigned by alphabetically
sorting the 1000 WNIDs, exactly the same procedure
`torchvision.datasets.ImageFolder` itself uses when it alphabetically
sorts a directory's subfolder names to assign its OWN 0..999 class
indices. Zero-padded integer folder names sort identically to numeric
order, so ImageFolder's re-derived indices exactly reproduce HF's
original (and therefore the standard, pretrained-model-matching) label
assignment. `int2str()` instead returns a HUMAN-READABLE class
description (e.g. "tench, Tinca tinca"), not the WNID its name would
suggest -- alphabetically sorting THOSE strings produces a completely
different, scrambled ordering that has no relationship to the standard
class-index convention, silently corrupting every label. This was found
via a real run: a plain pretrained model scored 0.47% top-1 (near
1000-way random chance) through a materialized copy that used
`int2str()`-named directories.

NOT used by the university-GPU path (which already has a real local
ImageNet-1k copy) -- Colab-only, called from the *_colab.ipynb notebooks.
"""
import re
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

_VALID_CLASS_DIR_RE = re.compile(r"^\d{4}$")


def _assert_valid_class_dirs(split_dir, split_name):
    """Guards against exactly the bug this module once had (and against
    any FUTURE regression back to it): every class directory must be a
    zero-padded 4-digit integer label, nothing else. Runs unconditionally
    -- including on the idempotent "already materialized, skipping" path
    -- so a directory tree materialized by an OLDER, buggy version of this
    module gets caught immediately on the next run, not silently reused."""
    dirs = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    bad = [d for d in dirs if not _VALID_CLASS_DIR_RE.match(d)]
    assert not bad, (
        f"{split_dir} has {len(bad)} class director{'y' if len(bad) == 1 else 'ies'} that are NOT "
        f"zero-padded integer labels (e.g. {bad[:5]!r}) -- this is exactly the class-directory-"
        "naming bug documented in this module's docstring (a directory tree materialized by an "
        "older version of this code, or corrupted some other way). Delete this split and "
        f"re-materialize: `import shutil; shutil.rmtree('{split_dir}')`."
    )
    out_of_range = [d for d in dirs if not (0 <= int(d) <= 999)]
    assert not out_of_range, f"{split_dir} has class directories outside the valid 0-999 range: {out_of_range}"
    print(f"{split_name}: {len(dirs)} class directories, all valid zero-padded labels in [0000, 0999].")


def materialize_val_split(root, hf_token, n_images=50000):
    """Streams the FULL validation split (matching every baseline's own
    published protocol, which evaluates on the complete 50k val set) and
    writes each image as <root>/val/<wnid>/<i>.JPEG. Idempotent: skips
    entirely if <root>/val already has exactly n_images files (so re-running
    this cell after a Colab disconnect doesn't re-download from scratch)."""
    val_dir = Path(root) / "val"
    existing = list(val_dir.rglob("*.JPEG")) if val_dir.exists() else []
    if len(existing) == n_images:
        print(f"{val_dir} already has {n_images} images -- skipping re-materialization.")
        _assert_valid_class_dirs(val_dir, "val")
        return val_dir
    assert not existing, (
        f"{val_dir} has {len(existing)} images, expected 0 or {n_images} -- looks like a partial/stale "
        "materialization (e.g. an interrupted previous run). Delete it and re-run rather than silently "
        "continuing from an unknown state: `import shutil; shutil.rmtree(val_dir)`."
    )
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, token=hf_token)
    stream = iter(ds)
    counters = {}
    for _ in tqdm(range(n_images), desc="materializing val (one-time; ~10-20 min for 50k images)"):
        ex = next(stream)
        # Zero-padded RAW integer label, never int2str() -- see module docstring.
        class_dir_name = f"{ex['label']:04d}"
        cls_dir = val_dir / class_dir_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        k = counters.get(class_dir_name, 0)
        counters[class_dir_name] = k + 1
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(cls_dir / f"{k:04d}.JPEG", quality=95)
    n_written = sum(counters.values())
    assert n_written == n_images, f"materialized {n_written} images, expected {n_images}"
    print(f"Materialized {n_written} val images across {len(counters)} classes at {val_dir}")
    _assert_valid_class_dirs(val_dir, "val")
    return val_dir


def materialize_train_calib_subset(root, hf_token, images_per_class=10, max_images=5000):
    """Streams the train split and keeps up to `images_per_class` images
    PER CLASS ENCOUNTERED (a class-diversity cap, not a naive first-N pull
    -- protects against a class-ordered stream producing a single-class
    calibration pool), stopping once `max_images` total have been written
    or the stream is exhausted. This is a Colab-only APPROXIMATION of "the
    training split" that B2/B3/B4's upstream calibration code (a shuffled
    DataLoader over an ImageFolder) draws its 32-1000 image calibration
    batch from -- NOT the full 1.28M-image train set. Prints the number of
    distinct classes actually covered so this deviation from the
    university-GPU path (which uses the real, full train set) is visible,
    not hidden; if that count comes back low, the underlying HF stream is
    likely class-ordered and `max_images` should be raised substantially.
    Idempotent the same way as `materialize_val_split`."""
    train_dir = Path(root) / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        n_existing = len(list(train_dir.rglob("*.JPEG")))
        n_classes_existing = len(list(train_dir.iterdir()))
        print(f"{train_dir} already has {n_existing} images across {n_classes_existing} classes -- "
              "skipping re-materialization.")
        _assert_valid_class_dirs(train_dir, "train")
        return train_dir

    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=hf_token)
    stream = iter(ds)
    counters = {}
    n_written = 0
    pbar = tqdm(total=max_images, desc="materializing train calibration subset", unit="img")
    for ex in stream:
        if n_written >= max_images:
            break
        # Zero-padded RAW integer label, never int2str() -- see module docstring.
        class_dir_name = f"{ex['label']:04d}"
        k = counters.get(class_dir_name, 0)
        if k >= images_per_class:
            continue
        cls_dir = train_dir / class_dir_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(cls_dir / f"{k:04d}.JPEG", quality=95)
        counters[class_dir_name] = k + 1
        n_written += 1
        pbar.update(1)
    pbar.close()
    print(f"Materialized {n_written} train calibration images across {len(counters)} classes at {train_dir} "
          f"(if the class count looks low relative to n_written, the HF stream is likely class-ordered -- "
          "raise max_images to improve class diversity).")
    _assert_valid_class_dirs(train_dir, "train")
    return train_dir
