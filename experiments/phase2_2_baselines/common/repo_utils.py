"""Clone-and-pin-at-exact-commit helper, shared by every baseline runner.
Framework-free (subprocess + git only).

Never trusts a branch tip: after cloning (or fetching, if the clone already
exists), hard-checks-out the pinned SHA from commits.md and ASSERTS the
resulting HEAD matches before returning -- a baseline runner that calls
this and gets a return value back is guaranteed to be running the exact
pinned commit, not "whatever main happened to be today".
"""
import subprocess
from pathlib import Path

EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "external"


def _run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)} (cwd={cwd})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def ensure_repo(github_url, pinned_commit, local_name=None):
    """Clones `github_url` into experiments/phase2_2_baselines/external/<local_name>/ if not
    already present, then hard-checks-out `pinned_commit` and asserts the
    resulting HEAD sha matches exactly. Returns the local repo path."""
    local_name = local_name or github_url.rstrip("/").split("/")[-1]
    repo_path = EXTERNAL_DIR / local_name
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    if not repo_path.exists():
        print(f"Cloning {github_url} -> {repo_path}")
        _run(["git", "clone", github_url, str(repo_path)])
    else:
        print(f"{repo_path} already exists -- fetching (not re-cloning).")
        _run(["git", "fetch", "--all"], cwd=repo_path)

    _run(["git", "checkout", pinned_commit], cwd=repo_path)
    head_sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_path)
    assert head_sha == pinned_commit, (
        f"Pin mismatch for {github_url}: expected HEAD={pinned_commit}, got {head_sha} after checkout. "
        "Something is wrong with the local clone -- do not proceed with a run whose commit isn't verified."
    )
    print(f"Verified {repo_path} is pinned at {head_sha}.")
    return repo_path
