"""Internal shared helpers for the jetson/ folder ONLY -- every other
script in this folder imports from here, never from experiments/phase2_2_baselines/common/ or
experiments/phase1_E1-E5_E4a/ or experiments/phase2_1_fleet/ (jetson/ must be clonable and
runnable standalone, and may not
import from or write to P2.1/P2.2 code paths -- see README.md's top
banner). This is the one deliberate exception to "no cross-file imports
within jetson/": duplicating this boilerplate seven times would be worse
than one small internal-only module.

Canonical P2.1 artifacts (Dq_table.csv, Dm_table.csv, plans.csv, the
per-model summary.json) are READ via `read_p2_summary`/`read_p2_csv` below
and NEVER written to -- both helpers open with mode="r" only, and no
function in this module ever accepts a P2.1 path as a write target.
"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

JETSON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = JETSON_ROOT.parent
P2_RESULTS_ROOT = REPO_ROOT / "experiments" / "phase2_1_fleet" / "results" / "P2"
OUT_ROOT = JETSON_ROOT / "out"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path) -> str:
    return sha256_hex(Path(path).read_bytes())


def sha256_of_json_obj(obj) -> str:
    """Stable hash of a JSON-serializable object (sorted keys, no
    whitespace) -- used for manifest/config fingerprints throughout this
    folder, same principle as experiments/phase2_2_baselines/common/results_io.py's
    config_hash (not imported, reimplemented here -- see module docstring)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return sha256_hex(blob)


def read_p2_summary(model_name: str) -> dict:
    """READ-ONLY. Loads experiments/phase2_1_fleet/results/P2/<model>/P2_<model>_summary.json."""
    path = P2_RESULTS_ROOT / model_name / f"P2_{model_name}_summary.json"
    assert path.exists(), f"P2.1 canonical summary not found (read-only lookup): {path}"
    with open(path, "r") as f:
        return json.load(f)


def read_p2_csv_path(model_name: str, filename: str) -> Path:
    """READ-ONLY. Returns the path to a P2.1 CSV (plans.csv, Dq_table.csv,
    Dm_table.csv, ...) after asserting it exists -- callers open it
    themselves (pandas/csv), never through a write-capable helper."""
    path = P2_RESULTS_ROOT / model_name / filename
    assert path.exists(), f"P2.1 canonical file not found (read-only lookup): {path}"
    return path


def ensure_out_dir(*parts) -> Path:
    d = OUT_ROOT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def run_cmd(cmd, **kwargs):
    """subprocess.run wrapper: text mode, captures output, raises with the
    captured stderr/stdout visible in the exception (bare `check=True`
    swallows this on failure, which is exactly the "silent failure" class
    of bug this project has hit before -- see deviations.md's checkpoint-
    override entries)."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {cmd}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def pip_freeze_hash(python_exe=sys.executable) -> str:
    result = subprocess.run([python_exe, "-m", "pip", "freeze"], capture_output=True, text=True)
    return sha256_hex(result.stdout.encode())[:16]


def is_jetson() -> bool:
    """Best-effort device detection: L4T ships /etc/nv_tegra_release on
    every Jetson image; aarch64 alone is not sufficient (some CI/dev ARM
    hosts are not Jetsons)."""
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


def read_l4t_version() -> str:
    path = Path("/etc/nv_tegra_release")
    if not path.exists():
        return "unknown (no /etc/nv_tegra_release -- not running on L4T)"
    return path.read_text().strip()


def plan_bits_at(plan: dict, l: int) -> int:
    """Per-block bit-width for a plan dict {"bits": {...}, "merges": [...]}.
    Handles both string and int dict keys -- a plan freshly built in memory
    uses int keys (range(1,13)); a plan round-tripped through
    json.dumps/json.loads (e.g. read back from a CSV's plan_json column)
    uses string keys. Shared by replan.py and paired_eval.py so both
    compare plans the same way."""
    return plan["bits"].get(str(l), plan["bits"].get(l, 16))


def plan_n_tokens_entering(plan: dict, l: int, n_tokens_total: int) -> int:
    return n_tokens_total - sum(m["r"] for m in plan["merges"] if m["block"] < l)


def compute_plan_divergence(plan_a: dict, plan_b: dict, n_tokens_total: int, n_blocks: int = 12):
    """Per-block (n_l, bits) diff between two plans -- the ONLY correct way
    to compare two plan dicts for equality (see plan_bits_at's docstring on
    the string/int key-type pitfall; a raw `plan_a == plan_b` can silently
    say "different" for two plans that are logically identical, or vice
    versa is never actually wrong but the per-block form is needed anyway
    for the required diff detail).

    Returns (plans_identical: bool, n_blocks_differing: int,
    per_block_diff: list[dict]), each per_block_diff entry:
    {"block": l, "a": {"n_l":, "bits":}, "b": {"n_l":, "bits":}, "differs": bool}.
    """
    per_block_diff = []
    n_differing = 0
    for l in range(1, n_blocks + 1):
        bits_a, bits_b = plan_bits_at(plan_a, l), plan_bits_at(plan_b, l)
        n_a = plan_n_tokens_entering(plan_a, l, n_tokens_total)
        n_b = plan_n_tokens_entering(plan_b, l, n_tokens_total)
        differs = (bits_a != bits_b) or (n_a != n_b)
        if differs:
            n_differing += 1
        per_block_diff.append({
            "block": l, "a": {"n_l": n_a, "bits": bits_a}, "b": {"n_l": n_b, "bits": bits_b},
            "differs": differs,
        })
    return n_differing == 0, n_differing, per_block_diff


def secrets_scan(text: str):
    """Regex scan for literal HF/GitHub tokens -- same two patterns
    common/notebook_self_test.py's check_secrets uses in experiments/phase2_2_baselines/,
    reimplemented here (jetson/ cannot import experiments/phase2_2_baselines/common/).
    Returns a list of (pattern_name, match) tuples; empty list == clean."""
    import re

    patterns = {
        "hf_token": r"hf_[A-Za-z0-9]{20,}",
        "github_pat": r"github_pat_[A-Za-z0-9_]{20,}",
        "github_pat_classic": r"ghp_[A-Za-z0-9]{20,}",
    }
    hits = []
    for name, pat in patterns.items():
        for m in re.finditer(pat, text):
            hits.append((name, m.group(0)))
    return hits
