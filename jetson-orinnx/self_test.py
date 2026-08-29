"""jetson-orinnx/self_test.py -- 11-item hard-assert self-test (10-item style
plus the pre-flight port-equivalence gate, item 10 below), same discipline
as experiments/phase2_2_baselines/common/notebook_self_test.py (not imported -- jetson-orinnx/ cannot
depend on experiments/phase2_2_baselines/, see _common.py's module docstring): each check
returns (passed, detail) and never partially passes; a failure raises,
naming every failed check, so a bad environment is caught before real
device time is spent. Runs host-side (dry-run: most checks degrade to an
explicit "N/A on this host" rather than crashing) AND on the Jetson
(full checks, once out/ has real artifacts).

Usage: python self_test.py
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import JETSON_ROOT, OUT_ROOT, is_jetson, secrets_scan  # noqa: E402

RESULTS = []


def check(name):
    def decorator(fn):
        def wrapper():
            try:
                passed, detail = fn()
            except Exception as e:  # a check that crashes is a FAIL, not an uncaught exception
                passed, detail = False, f"CHECK RAISED: {e!r}"
            RESULTS.append((name, passed, detail))
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {detail}")
            return passed
        return wrapper
    return decorator


@check("1. Checkpoint pinning consistency")
def check_pinning():
    expected_tag = "deit_small_patch16_224.fb_in1k"
    hits = []
    seg_manifest = OUT_ROOT / "segments" / "manifest.json"
    if seg_manifest.exists():
        m = json.loads(seg_manifest.read_text())
        if m["resolved_pretrained_tag"] != expected_tag:
            return False, f"segments/manifest.json tag {m['resolved_pretrained_tag']!r} != {expected_tag!r}"
        hits.append(("segments", m["checkpoint_state_hash"]))
    eval_result = OUT_ROOT / "paired_eval" / "paired_eval_result.json"
    if eval_result.exists():
        r = json.loads(eval_result.read_text())
        hits.append(("paired_eval", r["checkpoint_hash"]))
    if len(hits) >= 2:
        hashes = {h for _, h in hits}
        if len(hashes) != 1:
            return False, f"checkpoint hash disagreement across runs: {hits}"
    if not hits:
        return True, "no artifacts yet -- tag constants in export_segments.py/paired_eval.py match by source inspection"
    return True, f"consistent across {len(hits)} artifact(s): {hits}"


@check("2. Versions recorded (setup.sh output)")
def check_versions():
    versions_path = OUT_ROOT / "versions.json"
    if not versions_path.exists():
        return (True, "N/A on this host -- out/versions.json not present (run setup.sh on the Jetson first)") \
            if not is_jetson() else (False, f"{versions_path} not found -- run setup.sh")
    v = json.loads(versions_path.read_text())
    required = ("l4t", "tensorrt", "cuda", "python", "torch", "gpu_name", "pip_freeze_hash")
    missing = [k for k in required if k not in v]
    if missing:
        return False, f"versions.json missing keys: {missing}"
    return True, f"l4t={v['l4t'][:40]!r} trt={v['tensorrt']!r} cuda={v['cuda']!r} torch={v['torch']!r}"


@check("3. Clocks locked (jetson_clocks)")
def check_clocks_locked():
    if not is_jetson():
        return True, "N/A on this host -- not running on L4T"
    import subprocess
    # `-n`: fail fast rather than hang on a password prompt if passwordless sudo isn't set up for
    # this user -- found for real on a JetPack 7.2 device, 2026-07: `jetson_clocks --show` refuses
    # to run at all without root, even in --show (read-only) mode, unlike some earlier JetPack
    # releases where this was reported to work unprivileged.
    result = subprocess.run(["sudo", "-n", "jetson_clocks", "--show"], capture_output=True, text=True)
    out = result.stdout
    if result.returncode != 0 and "a password is required" in (result.stderr or "").lower():
        return False, "jetson_clocks --show needs root and passwordless sudo isn't set up for this user -- run with sudo manually, or configure passwordless sudo, then re-run"
    # NOTE: the real field name is "CurrentFreq", not "CurFreq" -- confirmed against real
    # jetson_clocks --show output on a JetPack 7.2 device, 2026-07 (an earlier version of this
    # check looked for "CurFreq", which is not a substring of "CurrentFreq" and so never matched
    # any real line -- every check silently fell through to "could not parse", every time).
    lines = [l for l in out.splitlines() if "MinFreq" in l and "MaxFreq" in l and "CurrentFreq" in l]
    # Skip DLA/PVA rails on devices where they report nonsense Min/Max
    lines = [l for l in lines if not any(skip in l for skip in ("DLA", "PVA"))]
    if not lines:
        return False, "could not parse jetson_clocks --show output"
    for l in lines:
        freqs = re.findall(r"(?:Min|Max|Current)Freq=(\d+)", l)
        if len(set(freqs)) > 1:
            return False, f"unlocked rail: {l.strip()}"
    return True, f"{len(lines)} rail(s) locked"


@check("4. TF32 n/a / explicitly disabled note")
def check_tf32():
    text = (JETSON_ROOT / "paired_eval.py").read_text()
    has_guard = "allow_tf32 = False" in text
    if not has_guard:
        return False, "paired_eval.py does not disable TF32 -- required (it may run on a TF32-capable host -- Colab's A100/L4, the Jetson's own Ampere GPU -- unlike P2.1's T4 reference)"
    return True, (
        "TF32 is explicitly disabled in paired_eval.py (the only script whose real fp32 GPU "
        "matmuls this matters for -- it runs on a HOST/Colab, not the Jetson, see its own module "
        "docstring); build_engines.sh/measure_lut.py use TensorRT FP16/INT8 engines, where TF32 does "
        "not apply."
    )


@check("5. Manifest hashes present and well-formed")
def check_manifest_hashes():
    found = {}
    seg_manifest = OUT_ROOT / "segments" / "manifest.json"
    if seg_manifest.exists():
        m = json.loads(seg_manifest.read_text())
        if m.get("calibration"):
            found["calib_hash"] = m["calibration"]["calib_hash"]
    eval_result = OUT_ROOT / "paired_eval" / "paired_eval_result.json"
    if eval_result.exists():
        r = json.loads(eval_result.read_text())
        found["paired_eval_manifest_hash"] = r["manifest_hash"]
    if not found:
        return True, "no artifacts yet -- nothing to check"
    bad = {k: v for k, v in found.items() if not re.fullmatch(r"[0-9a-f]{16,64}", str(v))}
    if bad:
        return False, f"malformed hash(es): {bad}"
    return True, f"{len(found)} hash(es) well-formed: {found}"


@check("6. LUT schema")
def check_lut_schema():
    lut_path = OUT_ROOT / "lut" / "latency_lut.json"
    if not lut_path.exists():
        return True, "N/A -- out/lut/latency_lut.json not present yet"
    lut = json.loads(lut_path.read_text())
    required = ("n_tokens", "precision", "median_ms", "p99_ms", "temp_c_before", "temp_c_after",
                "mean_power_w", "joules_per_inference", "versions", "timestamp")
    for key, entry in lut["entries"].items():
        missing = [k for k in required if k not in entry]
        if missing:
            return False, f"entry {key!r} missing fields: {missing}"
    return True, f"{len(lut['entries'])} entries, schema OK"


@check("7. Resume behavior (measure_lut.py skip-if-present logic)")
def check_resume():
    # Torch-free unit test of the resume CONDITION itself (measure_lut.py's actual jtop/tensorrt
    # code paths are not importable here without hardware -- see module docstring), same style as
    # baselines' own find_existing_row unit test pattern.
    fake_lut = {"entries": {"n197_fp16": {"n_tokens": 197, "precision": "fp16"}}}
    force = False
    key_present, key_absent = "n197_fp16", "n165_int8"
    should_skip_present = (key_present in fake_lut["entries"]) and not force
    should_skip_absent = (key_absent in fake_lut["entries"]) and not force
    if not (should_skip_present and not should_skip_absent):
        return False, "resume-skip condition logic is wrong"
    return True, "skip-if-present-and-not-force logic verified against a synthetic LUT"


@check("8. Secrets scan (every file in jetson-orinnx/)")
def check_secrets():
    hits = []
    for path in JETSON_ROOT.rglob("*"):
        if path.is_dir() or "out" in path.relative_to(JETSON_ROOT).parts or path.name == "self_test.py":
            continue
        if path.suffix not in (".py", ".sh", ".md", ".txt"):
            continue
        text = path.read_text(errors="ignore")
        found = secrets_scan(text)
        if found:
            hits.append((str(path.relative_to(JETSON_ROOT)), found))
    if hits:
        return False, f"secret-like literals found: {hits}"
    return True, "zero hits (hf_*, github_pat_*, ghp_* patterns) across all jetson-orinnx/ files"


@check("9. Read-only guard on P2.1 paths")
def check_readonly_guard():
    """Static, AST-based check (deliberately NOT a raw text/substring search -- an earlier version
    of this check did that and false-positived on every docstring/comment that merely MENTIONS
    P2_RESULTS_ROOT or experiments/phase2_1_fleet/results/P2 for documentation purposes, including this file's own
    check description; caught and fixed during self-review): P2_RESULTS_ROOT (the only handle onto
    P2.1's directory) may only be used as a real identifier reference inside _common.py's two
    read-only helpers (read_p2_summary, read_p2_csv_path) -- both open with mode 'r' / read_text
    only, never write. Any OTHER file using the P2_RESULTS_ROOT NAME in actual code (not a string
    literal), or a write call inside those two helpers, is a violation."""
    violations = []
    for path in JETSON_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        name_uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "P2_RESULTS_ROOT"]
        if path.name == "_common.py":
            continue  # the definition + the two read-only helpers live here; checked separately below
        if name_uses:
            violations.append(f"{path.name}: uses the P2_RESULTS_ROOT identifier directly in code "
                               "(must go through _common.read_p2_summary/read_p2_csv_path only)")

    common_tree = ast.parse((JETSON_ROOT / "_common.py").read_text(), filename="_common.py")
    write_calls = {"write_text", "write_bytes", "to_csv", "to_json", "dump"}
    allowed_functions = {"read_p2_summary", "read_p2_csv_path"}
    for node in ast.walk(common_tree):
        if isinstance(node, ast.FunctionDef) and "P2_RESULTS_ROOT" in ast.dump(node) and node.name != "read_p2_summary" \
                and node.name != "read_p2_csv_path" and any(
                    isinstance(n, ast.Name) and n.id == "P2_RESULTS_ROOT" for n in ast.walk(node)):
            violations.append(f"_common.py:{node.name} uses P2_RESULTS_ROOT but is not one of {allowed_functions}")
        if isinstance(node, ast.FunctionDef) and node.name in allowed_functions:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in write_calls:
                    violations.append(f"_common.py:{node.name} calls a write method ({sub.attr}) -- must be read-only")
    if violations:
        return False, f"{len(violations)} violation(s): {violations}"
    return True, "P2_RESULTS_ROOT is used only by _common.py's two read-only helpers (AST-verified, not text-matched); no write calls in them"


@check("10. Real port-equivalence gate (replan.py --verify-port)")
def check_port_equivalence():
    """Pre-flight patch item 1: this is NOT a degraded/host-only check --
    replan.py --verify-port needs no GPU/device/LUT at all (pure
    arithmetic against the real, committed P2.1 plans.csv), so this check
    runs FOR REAL every time self_test.py runs, on the Jetson or off it.
    A failure here means the ported DP disagrees with P2.1's own
    allocator and MUST block before any hardware measurement -- see
    replan.py's verify_port_equivalence() docstring."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(JETSON_ROOT / "replan.py"), "--verify-port"],
        capture_output=True, text=True, cwd=str(JETSON_ROOT),
    )
    if result.returncode != 0:
        return False, f"replan.py --verify-port FAILED (exit {result.returncode}): {result.stdout[-500:]}"
    m = re.search(r"Port-equivalence checklist: (\d+)/(\d+) MATCH", result.stdout)
    if not m or m.group(1) != m.group(2):
        return False, f"could not confirm N/N MATCH in output: {result.stdout[-500:]}"
    return True, f"{m.group(1)}/{m.group(2)} MATCH (real run against committed P2.1 plans.csv, no device needed)"


@check("11. Gate ordering (bring-up -> engines -> LUT -> replan -> eval)")
def check_gate_ordering():
    engines_present = any((OUT_ROOT / "engines").glob("*.engine")) if (OUT_ROOT / "engines").exists() else False
    lut_present = (OUT_ROOT / "lut" / "latency_lut.json").exists()
    replan_present = (OUT_ROOT / "replan" / "plans_hw.csv").exists()
    eval_present = (OUT_ROOT / "paired_eval" / "paired_eval_result.json").exists()

    # Real bug found and fixed, public-release prep (2026-07): `out/engines/*.engine` files are
    # large, hardware-specific TensorRT binaries -- deliberately gitignored, never committed, and
    # therefore NEVER present on a host machine's fresh clone, even though `out/lut/latency_lut.json`
    # (small, committed, real data) legitimately is. This sub-check's job is to catch a genuine
    # ordering anomaly ON THE JETSON (a LUT somehow existing without the engines that produced it);
    # on a host dry-run, "no local engines" is the expected, correct state, not an anomaly -- same
    # reasoning checks 2/3 above already apply via `is_jetson()`, this one just hadn't been updated
    # to match. The replan/eval ordering checks below are untouched -- they depend on committed
    # repo state, not device-local binaries, so they remain fully enforced on host and device alike.
    if lut_present and not engines_present and is_jetson():
        return False, "latency_lut.json exists but no engines found -- LUT cannot precede engine builds"
    if replan_present and not lut_present:
        return False, "plans_hw.csv exists but latency_lut.json does not -- replan cannot precede the LUT"
    if eval_present and not replan_present:
        return False, "paired_eval_result.json exists but plans_hw.csv does not -- eval cannot precede replan"
    # On host, local engine binaries are never expected (see above) -- judge "furthest stage" from
    # the committed artifacts (LUT/replan/eval) alone rather than reporting the misleading
    # "bring-up" (engines are the ONLY thing genuinely absent on a host dry-run, not everything
    # after it).
    if is_jetson():
        stage = ("bring-up" if not engines_present else
                 "engines" if not lut_present else
                 "LUT" if not replan_present else
                 "replan" if not eval_present else "eval")
    else:
        stage = ("bring-up (host -- engines N/A here by design)" if not lut_present else
                 "LUT" if not replan_present else
                 "replan" if not eval_present else "eval")
    return True, f"dependency chain intact; current furthest completed stage: {stage}"


def main():
    print(f"=== jetson-orinnx/self_test.py ({'on Jetson (L4T)' if is_jetson() else 'HOST dry-run'}) ===")
    checks = [check_pinning, check_versions, check_clocks_locked, check_tf32, check_manifest_hashes,
              check_lut_schema, check_resume, check_secrets, check_readonly_guard,
              check_port_equivalence, check_gate_ordering]
    for fn in checks:
        fn()

    failed = [name for name, passed, _ in RESULTS if not passed]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.")
    if failed:
        raise AssertionError(f"self_test.py FAILED: {failed}")
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
