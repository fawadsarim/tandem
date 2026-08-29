"""Checkpoint-identity policy (two-step), critical for ViT-B.

TANDEM's P2.1 comparison targets are PINNED tags:
  deit_tiny_patch16_224.fb_in1k, deit_small_patch16_224.fb_in1k,
  deit_base_patch16_224.fb_in1k, vit_base_patch16_224.augreg2_in21k_ft_in1k
(FP top-1 references, from P2.1's own S8 measurement: 72.166 / 79.848 /
81.97 / 85.10 -- see PINNED_TAGS below).

For DeiT, every baseline's own unmodified pretrained-loading resolves to
the SAME fb_in1k weights TANDEM pins (verified against real source at each
pinned commit -- RepQ-ViT's classification/utils/build_model.py and FQ-ViT's
models/vit_quant.py both load from `dl.fbaipublicfiles.com/deit/deit_*`,
byte-identical URLs to what timm's own `.fb_in1k` tag resolves to). Step A
(reproduction, their checkpoint) and Step B (comparison, our pinned
checkpoint) are therefore THE SAME RUN for every DeiT model -- see
`coincides_with_pinned()`.

For ViT-B, they do NOT coincide -- verified from real source, not assumed:
  - RepQ-ViT's build_model('vit_base_patch16_224') and FQ-ViT's
    vit_base_patch16_224(pretrained=True) both resolve to the checkpoint at
    https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-
    aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
    -- this is BYTE-IDENTICAL to the `url=` field of modern timm's
    `vit_base_patch16_224.augreg_in21k_ft_in1k` entry (checked against
    timm's real vision_transformer.py) and also matches timm==0.4.12's own
    (single, pre-tag-system) default_cfg url for `vit_base_patch16_224`
    (both pinned commits/timm versions predate timm's per-model tag
    system, so there was only ever ONE default checkpoint to resolve to).
    Independently cross-confirmed by each repo's OWN published FP
    baseline: RepQ-ViT's README lists "ViT-B (84.54)"; FQ-ViT's README
    lists ViT-B Full Precision = 84.53 -- both match
    `augreg_in21k_ft_in1k`'s real accuracy (84.534), NOT our pinned
    `augreg2_in21k_ft_in1k` (85.108). A 0.57pp gap -- not noise.
  - TANDEM pins `augreg2_in21k_ft_in1k` (FP 85.108, P2.1 S8 measured
    85.10) instead, per P2's own PUBLISHED_TOP1_LOOKUP.
  - Old (pre-tag-system) timm has no way to request a specific modern tag
    by name -- `install_pinned_checkpoint()` below downloads the exact
    weights from HF Hub directly (a plain `pytorch_model.bin` state dict,
    confirmed via the real safetensors header on the same repo: standard
    `blocks.N.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.{weight,bias}`
    / `cls_token` / `pos_embed` / `patch_embed.proj.*` / `norm.*` /
    `head.*` keys -- IDENTICAL to old timm's own `VisionTransformer`
    attribute names, confirmed against timm==0.4.12's real source) and
    loads it directly, bypassing whichever checkpoint the method's own
    pretrained=True path would otherwise fetch.

This module never modifies an upstream repo's file on disk -- the override
is injected via the SAME sitecustomize.py PYTHONPATH mechanism
common/determinism.py already uses for TF32, gated by an explicit env var
so it is only ever active for a deliberate Step-B run, never silently.
The FIRST real correctness proof that an override actually took effect is
common/fp_parity_gate.py, run immediately after -- if the wrong weights
loaded (or loaded with the wrong keys), that gate fails loudly before any
method-specific number is trusted.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parent.parent


def _download_progress_hook(desc):
    """`urllib.request.urlretrieve` is completely silent until the ENTIRE
    file arrives -- same "looks stuck, might just be slow" symptom
    common/subprocess_utils.py fixes for subprocesses, via a different
    mechanism (there's no subprocess here to stream; this process is
    downloading directly). Reports roughly every 10% so a multi-hundred-MB
    download (the ~350MB ViT-B checkpoint) doesn't flood output with a
    line per 8KB block, while still giving visible, periodic proof of
    progress instead of dead silence for however long the download takes."""
    last_reported = [-1]

    def hook(block_num, block_size, total_size):
        if total_size <= 0:
            return  # server didn't report Content-Length -- nothing to compute a percentage against
        downloaded = min(block_num * block_size, total_size)
        pct = (int(downloaded * 100 / total_size) // 10) * 10
        if pct != last_reported[0]:
            last_reported[0] = pct
            print(f"{desc}: {pct}% ({downloaded / 1e6:.0f}/{total_size / 1e6:.0f} MB)")

    return hook

# model_name -> (our pinned HF hub repo id, P2.1 S8 measured FP top-1 pct).
# FP references are P2.1's OWN measured S8 numbers (not the published-table
# lookup) per this spec's explicit instruction -- these are the numbers the
# FP-parity gate checks against.
PINNED_TAGS = {
    "deit_tiny_patch16_224": ("timm/deit_tiny_patch16_224.fb_in1k", 72.166),
    "deit_small_patch16_224": ("timm/deit_small_patch16_224.fb_in1k", 79.848),
    "deit_base_patch16_224": ("timm/deit_base_patch16_224.fb_in1k", 81.97),
    "vit_base_patch16_224": ("timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", 85.10),
}

# The only model where a method's own unmodified pretrained-loading resolves
# to a DIFFERENT checkpoint than our pinned tag -- see module docstring for
# the source-verified evidence. Every other model coincides by construction
# (fb_in1k is what every DeiT baseline, and TANDEM, has always used).
CHECKPOINT_OVERRIDE_NEEDED = {"vit_base_patch16_224"}

# What a method's own unmodified pretrained=True call actually resolves to,
# for citation in results.csv/notes and in the Step-A reproduction-gate
# report line -- verified from real source (see module docstring), not
# assumed identical to our pinned tag.
PAPER_CHECKPOINT_TAG = {
    "deit_tiny_patch16_224": "deit_tiny_patch16_224.fb_in1k",
    "deit_small_patch16_224": "deit_small_patch16_224.fb_in1k",
    "deit_base_patch16_224": "deit_base_patch16_224.fb_in1k",
    "vit_base_patch16_224": "vit_base_patch16_224.augreg_in21k_ft_in1k",  # NOT augreg2 -- see docstring
}


def coincides_with_pinned(model_name):
    """True if Step A (paper checkpoint) and Step B (our pinned tag) are
    the SAME checkpoint for this model -- if so, callers must not re-run
    the method a second time (per spec: "do not run twice"), only emit a
    second results.csv row (row_kind="comparison") reusing Step A's
    already-measured numbers."""
    return model_name not in CHECKPOINT_OVERRIDE_NEEDED


def pinned_checkpoint_tag(model_name):
    hf_repo_id, _ = PINNED_TAGS[model_name]
    return hf_repo_id.split("/", 1)[1]  # "timm/X.Y" -> "X.Y"


def pinned_fp_reference(model_name):
    return PINNED_TAGS[model_name][1]


def assert_fp_references_match_canonical_summaries():
    """P2.2 audit item #3: PINNED_TAGS' FP references (72.166/79.848/
    81.97/85.10) are mirrored CONSTANTS, not read live from
    experiments/phase2_1_fleet/results/P2/<model>/P2_<model>_summary.json -- per the audit
    spec ("if mirrored as constants, asserted equal to the summary values
    at startup"), this reads each real summary and asserts bit-for-bit
    equality against `summary["fp_top1_50k"] * 100`. Skips (prints, does
    not fail) a model whose summary file doesn't exist -- same reasoning
    as bops_identity_check.check_c_id_matches_canonical_summary, this
    module must stay usable from a checkout without experiments/phase2_1_fleet/results/P2/
    materialized."""
    import json
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    mismatches = []
    for model_name, (_, pinned_ref) in PINNED_TAGS.items():
        summary_path = (repo_root / "experiments" / "phase2_1_fleet" / "results" / "P2" / model_name /
                         f"P2_{model_name}_summary.json")
        if not summary_path.exists():
            print(f"assert_fp_references_match_canonical_summaries: SKIPPED {model_name} "
                  f"(canonical summary not found at {summary_path})")
            continue
        canonical_pct = json.loads(summary_path.read_text())["fp_top1_50k"] * 100
        if canonical_pct != pinned_ref:
            mismatches.append(f"{model_name}: PINNED_TAGS has {pinned_ref}, canonical summary has {canonical_pct}")
        else:
            print(f"assert_fp_references_match_canonical_summaries: {model_name} matches "
                  f"canonical fp_top1_50k={canonical_pct} ({summary_path.name})")
    assert not mismatches, (
        "PINNED_TAGS FP reference(s) do NOT match the canonical P2.1 summary: " + "; ".join(mismatches) +
        " -- these constants are supposed to mirror P2.1's own S8 measurement exactly; a real P2.1 rerun "
        "changed the canonical value without this file being updated to match."
    )


def download_hf_pytorch_bin(hf_repo_id, cache_dir=None):
    """Downloads `pytorch_model.bin` (a plain, wrapper-free torch state
    dict -- confirmed via the real HF repo's file listing and safetensors
    header, see module docstring) for `hf_repo_id` from the HF Hub, caches
    it under `cache_dir` (default: experiments/phase2_2_baselines/_pinned_checkpoints/), and
    returns the local path. Pure stdlib (urllib) -- no `torch`/`huggingface_hub`
    dependency, so this can be called from B2/B3's own torch-free run.py
    before ever touching the target venv's subprocess. Idempotent: skips
    the download if the cached file already exists and is non-empty."""
    cache_dir = Path(cache_dir) if cache_dir else BASELINES_ROOT / "_pinned_checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.sha256(hf_repo_id.encode()).hexdigest()[:16]
    out_path = cache_dir / f"{safe_name}.bin"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Pinned checkpoint already cached: {hf_repo_id} -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path

    url = f"https://huggingface.co/{hf_repo_id}/resolve/main/pytorch_model.bin"
    print(f"Downloading pinned checkpoint: {url} -> {out_path}")
    tmp_path = out_path.with_suffix(".bin.tmp")
    urllib.request.urlretrieve(url, tmp_path, reporthook=_download_progress_hook(f"Downloading {hf_repo_id}"))
    assert tmp_path.stat().st_size > 1_000_000, (
        f"Downloaded pinned checkpoint from {url} is suspiciously small "
        f"({tmp_path.stat().st_size} bytes) -- likely an HTML error page, not real weights. "
        "Halting rather than caching a corrupt file."
    )
    tmp_path.replace(out_path)
    print(f"Cached pinned checkpoint: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# --- sitecustomize injection snippets (appended to determinism.py's
# TF32-disable sitecustomize.py, see common.determinism.write_sitecustomize's
# extra_source param) -- gated by TANDEM_PIN_CHECKPOINT_OVERRIDE=1 so the
# monkeypatch is a total no-op for every call that doesn't explicitly ask
# for it (Step A, every DeiT config, every non-comparison run).

_TIMM_OVERRIDE_SOURCE = '''

# --- pinned-checkpoint override (common/checkpoint_pin.py), Step B only ---
import os as _tandem_os
if _tandem_os.environ.get("TANDEM_PIN_CHECKPOINT_OVERRIDE") == "1":
    try:
        import torch as _tandem_torch
        import timm as _tandem_timm
        _tandem_pin_model = _tandem_os.environ["TANDEM_PIN_MODEL_NAME"]
        _tandem_pin_path = _tandem_os.environ["TANDEM_PIN_CHECKPOINT_PATH"]
        _tandem_orig_create_model = _tandem_timm.create_model

        def _tandem_patched_create_model(name, pretrained=False, **kwargs):
            if pretrained and name == _tandem_pin_model:
                _m = _tandem_orig_create_model(name, pretrained=False, **kwargs)
                _sd = _tandem_torch.load(_tandem_pin_path, map_location="cpu")
                if isinstance(_sd, dict) and set(_sd.keys()) <= {"model", "state_dict"}:
                    _sd = _sd.get("model", _sd.get("state_dict"))
                _m.load_state_dict(_sd, strict=True)
                print(f"[checkpoint_pin] Loaded PINNED weights for {name} from {_tandem_pin_path} "
                      "(overriding timm's own pretrained=True resolution).")
                return _m
            return _tandem_orig_create_model(name, pretrained=pretrained, **kwargs)

        _tandem_timm.create_model = _tandem_patched_create_model
    except ImportError:
        pass
'''

_FQVIT_OVERRIDE_SOURCE = '''

# --- pinned-checkpoint override (common/checkpoint_pin.py), Step B only ---
import os as _tandem_os
if _tandem_os.environ.get("TANDEM_PIN_CHECKPOINT_OVERRIDE") == "1":
    try:
        # sitecustomize.py runs at interpreter startup, before the target
        # script's own sys.path setup completes -- for a `python -m X`
        # invocation specifically, the current directory is not
        # guaranteed to be on sys.path yet at this point (its insertion
        # happens later, in runpy's own -m bootstrap), even though the
        # SUBPROCESS's real OS-level cwd (set by subprocess.Popen(cwd=...))
        # already IS the FQ-ViT repo root. Without this, `import
        # models.utils` below raised ModuleNotFoundError (an ImportError
        # subclass) for `-m`-invoked subprocesses (run_fp_parity_gate's
        # `-m common.fp_parity_gate`), silently swallowed by the `except
        # ImportError` below -- the ENTIRE override silently no-opped for
        # every `-m` invocation, a second real bug found only after the
        # first fix (patching both names) still didn't make the override
        # fire. Explicitly ensuring the real cwd is on sys.path removes
        # the dependency on invocation style entirely.
        import sys as _tandem_sys
        _tandem_cwd = _tandem_os.getcwd()
        if _tandem_cwd not in _tandem_sys.path:
            _tandem_sys.path.insert(0, _tandem_cwd)

        import torch as _tandem_torch
        import models.utils as _tandem_fqvit_utils
        import models.vit_quant as _tandem_fqvit_vit_quant
        _tandem_pin_path = _tandem_os.environ["TANDEM_PIN_CHECKPOINT_PATH"]

        def _tandem_patched_load_weights_from_npz(model, url, check_hash=False):
            _sd = _tandem_torch.load(_tandem_pin_path, map_location="cpu")
            if isinstance(_sd, dict) and set(_sd.keys()) <= {"model", "state_dict"}:
                _sd = _sd.get("model", _sd.get("state_dict"))
            missing, unexpected = model.load_state_dict(_sd, strict=False)
            print(f"[checkpoint_pin] Loaded PINNED weights from {_tandem_pin_path} (overriding "
                  f"FQ-ViT's own npz loader for url={url}); missing={len(missing)} "
                  f"unexpected={len(unexpected)} keys (same strict=False tolerance FQ-ViT's own "
                  "DeiT loader uses, for the quantization-specific extras that aren't in any "
                  "pretrained checkpoint).")

        # BOTH names must be patched, not just models.utils's own. FQ-ViT's
        # models/__init__.py does `from .vit_quant import *`, and
        # models/vit_quant.py itself does `from .utils import
        # load_weights_from_npz` at module level -- a `from X import Y`
        # binds a LOCAL name in vit_quant.py's own namespace, decoupled
        # from models.utils's namespace from that point on. vit_base_
        # patch16_224()/vit_large_patch16_224() call THEIR OWN local
        # `load_weights_from_npz` (bound at vit_quant.py's import time,
        # before this patch ever runs) -- patching only
        # models.utils.load_weights_from_npz never reaches that call site.
        # This was a REAL bug, not a hypothetical: a real run showed
        # Step B's "override" measurement (84.4660%) exactly matching a
        # deliberately-unoverridden stock run to 4 decimal places over the
        # full 50k set -- statistically impossible unless the override was
        # silently never taking effect (confirmed further by the override's
        # own print line above never appearing in that run's real output).
        # See deviations.md.
        _tandem_fqvit_utils.load_weights_from_npz = _tandem_patched_load_weights_from_npz
        _tandem_fqvit_vit_quant.load_weights_from_npz = _tandem_patched_load_weights_from_npz
        print("[checkpoint_pin] Override installed: models.utils/models.vit_quant."
              "load_weights_from_npz both patched.")
    except ImportError as _tandem_e:
        # NOT silently swallowed: TANDEM_PIN_CHECKPOINT_OVERRIDE=1 means a
        # Step-B run explicitly REQUIRES this override -- a failure to
        # install it must be as visible as possible, as early as possible,
        # not hidden behind a bare `pass` (that exact silence is what let
        # the sys.path bug above go undetected until assert_override_fired
        # caught its downstream symptom). Still only a print here (not a
        # raise) because sitecustomize.py errors during site initialization
        # can produce confusing, hard-to-read tracebacks in some Python
        # versions -- assert_override_fired (common/checkpoint_pin.py) is
        # the actual hard gate, called by the real runner right after this
        # subprocess returns, and it WILL halt the pipeline if this print
        # ever fires.
        print(f"[checkpoint_pin] WARNING: override installation FAILED ({_tandem_e!r}) -- "
              "the pinned checkpoint will NOT be loaded; the caller's assert_override_fired "
              "check is expected to catch this and halt.")
'''

_OVERRIDE_SOURCE_BY_STYLE = {"timm": _TIMM_OVERRIDE_SOURCE, "fqvit": _FQVIT_OVERRIDE_SOURCE}


def override_sitecustomize_source(style):
    """style: "timm" (RepQ-ViT, build_model -> timm.create_model) or
    "fqvit" (FQ-ViT's vendored vit_base_patch16_224 -> load_weights_from_npz).
    Returns the extra source to pass as
    common.determinism.subprocess_env_with_tf32_disabled(...,
    extra_sitecustomize_source=...)."""
    assert style in _OVERRIDE_SOURCE_BY_STYLE, f"unknown style {style!r}, expected one of {list(_OVERRIDE_SOURCE_BY_STYLE)}"
    return _OVERRIDE_SOURCE_BY_STYLE[style]


OVERRIDE_FIRED_MARKER = "[checkpoint_pin] Loaded PINNED weights"


def assert_override_fired(subprocess_stdout, context_desc):
    """Call after ANY subprocess run with TANDEM_PIN_CHECKPOINT_OVERRIDE=1
    active in its env (`run_streaming` merges stderr into stdout, so a
    caller's `result.stdout` is the complete output to pass here) --
    asserts the override's own confirmation line actually appeared, i.e.
    the patched load function was genuinely CALLED, not just installed.

    This is the exact check that would have caught a real 2026-07 bug
    immediately instead of requiring an extended, multi-step
    investigation to find: common/checkpoint_pin.py's
    _FQVIT_OVERRIDE_SOURCE patched models.utils.load_weights_from_npz,
    but models/vit_quant.py's own factory functions call a DIFFERENT
    local binding of the same name (bound via `from .utils import
    load_weights_from_npz` at vit_quant.py's own import time, decoupled
    from models.utils's namespace from that point on) -- the override
    was installed successfully but silently never invoked, and Step B's
    ViT-B measurement was on the STOCK checkpoint the entire time. See
    deviations.md for the full trail (a numerical coincidence -- Step B's
    measured 84.4660% exactly matching a deliberately-unoverridden run --
    was what actually surfaced it; this assertion means that kind of
    detective work is never required again for either override style,
    "timm" or "fqvit")."""
    assert OVERRIDE_FIRED_MARKER in subprocess_stdout, (
        f"CHECKPOINT OVERRIDE NEVER FIRED for {context_desc}: TANDEM_PIN_CHECKPOINT_OVERRIDE=1 was set "
        f"but the override's own confirmation line ({OVERRIDE_FIRED_MARKER!r}) never appeared in the "
        "subprocess output -- the patched load function was installed but never actually CALLED, "
        "meaning the STOCK checkpoint was silently used instead of the pinned one. This is exactly the "
        "2026-07 bug documented in deviations.md -- halting rather than trusting a measurement that may "
        "be on the wrong weights."
    )


def override_env_vars(model_name, checkpoint_path):
    """Env vars the injected sitecustomize snippet reads -- set these on
    the subprocess env (in ADDITION to whatever
    subprocess_env_with_tf32_disabled already set) for a Step-B pinned-
    checkpoint run; omit entirely (or leave TANDEM_PIN_CHECKPOINT_OVERRIDE
    unset) for a normal Step-A run."""
    return {
        "TANDEM_PIN_CHECKPOINT_OVERRIDE": "1",
        "TANDEM_PIN_MODEL_NAME": model_name,
        "TANDEM_PIN_CHECKPOINT_PATH": str(checkpoint_path),
    }
