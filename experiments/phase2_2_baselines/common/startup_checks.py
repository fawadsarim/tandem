"""Self-defense check, run at the top of every run.py's main(): asserts the
six "dual-host audit" items (AUDIT_REPORT.md Parts A/B) are actually
present in this checkout before proceeding, rather than silently assuming
a fresh clone/branch has them. Framework-free (no torch import) so it can
run before any baseline-specific import, catching a missing/reverted
common/ module as the very first failure, not a confusing downstream
NameError or a silently-uncovered gap.

Each item is checked by import + attribute presence (and, where cheap, a
basic shape/format assertion), never by re-deriving the actual audit
finding -- this is a presence check, not a re-run of the whole P2.2 audit.
"""
import importlib
import json
import re
from pathlib import Path

# item -> (module path, required attribute names, one-line description)
DUAL_HOST_AUDIT_ITEMS = {
    "pinning": ("common.repo_utils", ["ensure_repo"],
                "commit-pin runtime assert (checkout == pinned SHA)"),
    "tf32_off": ("common.determinism", ["disable_tf32", "subprocess_env_with_tf32_disabled", "SEED"],
                 "TF32 disabled + fixed seed, both hosts"),
    "fixed_calib_indices": ("common.calib_capture", ["capture_calib_indices"],
                             "RepQ-ViT/FQ-ViT calibration indices reconstructed and recorded"),
    "equivalence_fingerprint": ("common.equivalence_gate", ["run_gate", "FINGERPRINT_PATH"],
                                 "cross-host smoke-eval equivalence gate"),
    "unified_schema": ("common.results_io", ["RESULTS_COLUMNS", "append_result"],
                        "results.csv single unified schema across runners/hosts"),
    "resume": ("common.checkpoint", ["load_checkpoint", "save_checkpoint"],
               "per-batch checkpoint/resume (B1/B4 in-process loops)"),
}

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def assert_dual_host_audit_items_present(items=None, extra=None):
    """`items`: subset of DUAL_HOST_AUDIT_ITEMS keys this baseline actually
    depends on (default: all except "resume", since B2/B3 structurally
    don't have it -- see AUDIT_REPORT.md Part B.3 -- and asserting its
    presence there would fail on a correct checkout, not a broken one).
    `extra`: additional (module_path, [attrs], desc) tuples to check
    (e.g. a baseline-specific check for common.checkpoint_pin on B2/B3).

    Aborts with a clear, item-named message on the FIRST missing piece --
    never silently continues with a partial audit surface."""
    keys = items if items is not None else [k for k in DUAL_HOST_AUDIT_ITEMS if k != "resume"]
    checks = [(k, *DUAL_HOST_AUDIT_ITEMS[k]) for k in keys]
    if extra:
        checks += [(f"extra:{mod}", mod, attrs, desc) for mod, attrs, desc in extra]

    missing = []
    for key, module_path, attrs, desc in checks:
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            missing.append(f"  - [{key}] {desc}: could not import {module_path} ({e})")
            continue
        for attr in attrs:
            if not hasattr(mod, attr):
                missing.append(f"  - [{key}] {desc}: {module_path}.{attr} not found")

    assert not missing, (
        "STARTUP SELF-DEFENSE CHECK FAILED -- one or more dual-host audit items "
        "(AUDIT_REPORT.md Parts A/B) assumed already implemented are missing from this "
        "checkout:\n" + "\n".join(missing) +
        "\n\nThis check exists so a stale/reverted/partial checkout fails LOUDLY at startup, "
        "before any GPU time is spent, rather than silently running without pinning/determinism/"
        "resume/schema guarantees the rest of this project's numbers depend on."
    )
    print(f"Startup self-defense check: {len(checks)} dual-host audit item(s) present -- OK "
          f"({', '.join(k for k, *_ in checks)}).")


def assert_valid_commit_sha(commit, label):
    """Cheap format check for a pinned-commit constant (40 lowercase hex
    chars) -- catches a truncated/placeholder SHA (e.g. copy-paste error)
    before common.repo_utils.ensure_repo ever reaches the network."""
    assert isinstance(commit, str) and _SHA_RE.match(commit), (
        f"{label}={commit!r} does not look like a full 40-hex-char git commit SHA -- "
        "check commits.md and this runner's pinned-commit constant."
    )


def assert_val_set_matches_canonical_p2(n_images, canonical_model="deit_small_patch16_224"):
    """P2.2 audit item #3 ruling (b), 2026-07: P2.1 exports no literal
    50k-val-MANIFEST hash anywhere (verified via exhaustive grep of
    experiments/phase2_1_fleet/P2_tandem_pipeline.py and every file under
    experiments/phase2_1_fleet/results/P2/ -- only state-dict hashes, `config_hash`, and
    `calib_hash` for CALIBRATION indices exist) -- see
    experiments/phase2_2_baselines/REPRODUCIBILITY.md for the full reasoning, including why
    P2.1 itself is not being modified to add one. This asserts the two
    real fields P2.1 DOES store instead -- `config.N_IMAGES_FINAL` and
    `data_source_50k` -- read LIVE from a canonical P2.1 summary, never
    hardcoded as literals here, so a genuine future P2.1 re-run with a
    different count/source propagates automatically instead of this
    check silently going stale.

    `canonical_model`: any ONE canonical summary suffices -- P2.1's own
    convention is the SAME val split for every model (verified: all four
    summaries already agree on both fields). Defaults to
    deit_small_patch16_224 since every one of B1-B4 evaluates it."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    summary_path = (repo_root / "experiments" / "phase2_1_fleet" / "results" / "P2" / canonical_model /
                     f"P2_{canonical_model}_summary.json")
    assert summary_path.exists(), (
        f"Canonical P2.1 summary not found at {summary_path} -- cannot verify this baseline's val-set "
        "count/source against P2.1's own recorded expectation without it (expected on a baselines-only "
        "checkout missing experiments/phase2_1_fleet/results/P2/; not expected on a full checkout)."
    )
    summary = json.loads(summary_path.read_text())
    canonical_n = summary["config"]["N_IMAGES_FINAL"]
    canonical_source = summary["data_source_50k"]
    assert n_images == canonical_n, (
        f"This baseline's own materialized val set has {n_images} images, but P2.1's canonical summary "
        f"({summary_path.name}) records N_IMAGES_FINAL={canonical_n} -- a different image COUNT means "
        "this baseline is not evaluating the same 50k split TANDEM's own P2 Pareto curve was measured "
        "on. Halting -- do not trust a cross-method comparison built on a different eval set."
    )
    assert canonical_source == "imagenet-1k", (
        f"P2.1's own canonical summary ({summary_path.name}) records data_source_50k={canonical_source!r}, "
        "not the expected 'imagenet-1k' -- this project's documented dataset convention has drifted. "
        "Halting rather than silently trusting an unverified assumption."
    )
    print(f"Val-set count/source match P2.1's canonical summary ({summary_path.name}): "
          f"n_images={n_images} == N_IMAGES_FINAL={canonical_n}, data_source_50k={canonical_source!r}.")
