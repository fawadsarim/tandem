# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # E4a - Ward-Linkage Theorem Package: Numerical Verification
#
# Three claims, checked against random data (numpy/scipy only, CPU, seed 0):
#
#   1. Two-point merge identity (Ward's minimum-variance formula).
#   2. Telescoping: cumulative pairwise merge cost equals the final
#      partition's weighted within-cluster SSE, regardless of merge order.
#   3. Greedy minimum-cost agglomeration produces the exact same merge
#      sequence as scipy's `linkage(..., method="ward")`, and our raw
#      delta-SSE is a fixed multiple of scipy's reported height (fitted
#      empirically, not assumed).
#
# CPU-only -- no GPU, model, or dataset needed; runs in ~1-2 seconds. This is
# the original, standalone, algorithm-only proof, cited directly by
# `notes/TANDEM_notes.tex` ("Numerical verification"). It is distinct from
# `E4_token_merge_curves.py`'s own `_self_test_ward_greedy_merge`, which
# re-derives Claim 2 (telescoping) independently on the GPU feature path,
# under real fp16 autocast, as a runtime regression guard for that specific
# code path -- that one is not a substitute for this one, and vice versa.
#
# **Outputs** (under `results/E4a/`): `E4a_verification.json`.
#
# Reviewer-release note: this notebook has no manual-review STOP
# checkpoints to convert -- see `E5_allocator_G1.py`'s Section 21/25d and
# its `TANDEM_INTERACTIVE` flag for the one file in E1-E5 that does.

# %% [markdown]
# ## 0. Clone the repo (so this and other notebooks share `results/` via git)
#
# This repo is public -- cloning it needs no authentication.
#
# Clones (or pulls, if already cloned this session) to `/content/tandem`,
# then changes into `/content/tandem/notebooks`, so `RESULTS_DIR =
# Path("results/E4a")` below resolves the same way every other notebook's
# `RESULTS_DIR` does.

# %%
import os

REPO_URL = "https://github.com/fawadsarim/tandem.git"
REPO_DIR = "/content/tandem"

if not os.path.exists(REPO_DIR):
    get_ipython().system(f'git clone -q {REPO_URL} {REPO_DIR}')  # noqa: F821
else:
    print(f"{REPO_DIR} already exists -- pulling latest instead of re-cloning.")
    get_ipython().system(f'git -C {REPO_DIR} pull -q')  # noqa: F821

get_ipython().run_line_magic('cd', f'{REPO_DIR}/experiments/phase1_E1-E5_E4a')  # noqa: F821

# %% [markdown]
# ## 1. Setup

# %%
!pip install -q numpy scipy

# %%
import json
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage

SEED = 0
RESULTS_DIR = Path("results/E4a")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All three claims below share ONE rng stream, called in this exact order
# (Claim 1, then 2, then 3) -- matching the original script's `main()` loop
# so results are bit-identical to a non-notebook run of the same seed.
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 2. Claim 1 - two-point merge identity
#
# `s_i||x_i-mu||^2 + s_j||x_j-mu||^2 == (s_i*s_j/(s_i+s_j))||x_i-x_j||^2`,
# Ward's minimum-variance formula for merging two weighted points.

# %%
def claim1_two_point_identity(rng, n_trials=10000, tol=1e-9):
    max_rel_err = 0.0
    for _ in range(n_trials):
        d = int(rng.integers(4, 513))
        s_i, s_j = int(rng.integers(1, 65)), int(rng.integers(1, 65))
        x_i, x_j = rng.standard_normal(d), rng.standard_normal(d)

        mu = (s_i * x_i + s_j * x_j) / (s_i + s_j)
        lhs = s_i * np.sum((x_i - mu) ** 2) + s_j * np.sum((x_j - mu) ** 2)
        rhs = (s_i * s_j / (s_i + s_j)) * np.sum((x_i - x_j) ** 2)

        rel_err = abs(lhs - rhs) / max(abs(rhs), 1e-12)
        max_rel_err = max(max_rel_err, rel_err)

    assert max_rel_err < tol, f"Claim 1 FAILED: max_rel_err={max_rel_err:.3e} >= {tol:.0e}"
    return {"n_trials": n_trials, "max_rel_err": max_rel_err, "tol": tol, "passed": True}


# %%
CLAIM1 = claim1_two_point_identity(rng)
print(f"Claim 1 PASSED: {CLAIM1}")

# %% [markdown]
# ## 3. Claim 2 - telescoping (order-independence)
#
# Cumulative sum of Claim-1 merge costs, under RANDOM merge order, equals
# the final partition's weighted within-cluster SSE.

# %%
def claim2_telescoping(rng, n_trials=1000, tol=1e-8):
    max_rel_err = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(4, 33))
        d = int(rng.integers(2, 17))
        x = rng.standard_normal((n, d))

        clusters = {i: {"members": (i,), "size": 1, "centroid": x[i].copy()} for i in range(n)}
        active = list(range(n))
        next_id = n
        k_target = int(rng.integers(1, n))
        cumulative_cost = 0.0

        while len(active) > k_target:
            i, j = rng.choice(active, size=2, replace=False)
            ci, cj = clusters[int(i)], clusters[int(j)]
            si, sj = ci["size"], cj["size"]
            cumulative_cost += (si * sj / (si + sj)) * np.sum((ci["centroid"] - cj["centroid"]) ** 2)
            clusters[next_id] = {
                "members": ci["members"] + cj["members"],
                "size": si + sj,
                "centroid": (si * ci["centroid"] + sj * cj["centroid"]) / (si + sj),
            }
            active = [c for c in active if c not in (int(i), int(j))] + [next_id]
            del clusters[int(i)], clusters[int(j)]
            next_id += 1

        final_sse = 0.0
        for cid in active:
            pts = x[list(clusters[cid]["members"])]
            final_sse += np.sum((pts - pts.mean(axis=0)) ** 2)

        rel_err = abs(cumulative_cost - final_sse) / max(abs(final_sse), 1e-12)
        max_rel_err = max(max_rel_err, rel_err)

    assert max_rel_err < tol, f"Claim 2 FAILED: max_rel_err={max_rel_err:.3e} >= {tol:.0e}"
    return {"n_trials": n_trials, "max_rel_err": max_rel_err, "tol": tol, "passed": True}


# %%
CLAIM2 = claim2_telescoping(rng)
print(f"Claim 2 PASSED: {CLAIM2}")

# %% [markdown]
# ## 4. Claim 3 - greedy minimum-cost merging equals scipy's Ward linkage
#
# Greedy agglomeration (merge the pair minimizing the Claim-1 cost at every
# step) must match scipy's `linkage(..., method="ward")` step-for-step, and
# `delta_sse / height^2` must be one fixed constant across all steps/trials
# (fitted empirically, not assumed).

# %%
def _greedy_ward_merge(x):
    """Greedy agglomeration: merge the pair minimizing the Claim-1 cost at
    every step. Returns a list of (frozenset-pair-of-members, delta_sse)."""
    n = x.shape[0]
    clusters = {i: {"members": frozenset({i}), "size": 1, "centroid": x[i].copy()} for i in range(n)}
    active, next_id, steps = list(range(n)), n, []

    while len(active) > 1:
        best_cost, best_pair = None, None
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                i, j = active[a], active[b]
                si, sj = clusters[i]["size"], clusters[j]["size"]
                diff = clusters[i]["centroid"] - clusters[j]["centroid"]
                cost = (si * sj / (si + sj)) * np.sum(diff**2)
                if best_cost is None or cost < best_cost:
                    best_cost, best_pair = cost, (i, j)
        i, j = best_pair
        ci, cj = clusters[i], clusters[j]
        si, sj = ci["size"], cj["size"]
        steps.append((frozenset({ci["members"], cj["members"]}), best_cost))
        clusters[next_id] = {
            "members": ci["members"] | cj["members"],
            "size": si + sj,
            "centroid": (si * ci["centroid"] + sj * cj["centroid"]) / (si + sj),
        }
        active = [c for c in active if c not in (i, j)] + [next_id]
        del clusters[i], clusters[j]
        next_id += 1
    return steps


def _scipy_ward_steps(x):
    """Same (frozenset-pair, height) structure, from scipy's own linkage."""
    n = x.shape[0]
    z = linkage(x, method="ward")
    members = {i: frozenset({i}) for i in range(n)}
    steps = []
    for row, (idx1, idx2, height, _count) in enumerate(z):
        m1, m2 = members[int(idx1)], members[int(idx2)]
        members[n + row] = m1 | m2
        steps.append((frozenset({m1, m2}), height))
    return steps


def claim3_greedy_equals_ward(rng, n_trials=200, const_rel_std_tol=1e-6):
    ratios = []
    for _ in range(n_trials):
        n = int(rng.integers(6, 21))
        d = int(rng.integers(2, 9))
        x = rng.standard_normal((n, d))

        greedy_steps = _greedy_ward_merge(x)
        scipy_steps = _scipy_ward_steps(x)
        assert len(greedy_steps) == len(scipy_steps) == n - 1, "Claim 3 FAILED: step count mismatch"

        for (g_pair, g_cost), (s_pair, s_height) in zip(greedy_steps, scipy_steps):
            assert g_pair == s_pair, f"Claim 3 FAILED: merge sequence diverged ({g_pair} != {s_pair})"
            if s_height > 0:
                ratios.append(g_cost / (s_height**2))

    ratios = np.array(ratios)
    fitted_constant = float(ratios.mean())
    rel_std = float(ratios.std() / fitted_constant)
    assert rel_std < const_rel_std_tol, f"Claim 3 FAILED: constant rel_std={rel_std:.3e} >= {const_rel_std_tol:.0e}"
    return {
        "n_trials": n_trials,
        "n_steps_compared": int(ratios.size),
        "sequence_match": True,
        "fitted_constant_deltaSSE_over_height_sq": fitted_constant,
        "constant_rel_std": rel_std,
        "const_rel_std_tol": const_rel_std_tol,
        "passed": True,
    }


# %%
CLAIM3 = claim3_greedy_equals_ward(rng)
print(f"Claim 3 PASSED: {CLAIM3}")

# %% [markdown]
# ## 5. Artifact - E4a_verification.json
#
# Deviation from the original standalone script this notebook replaces
# (`scripts/E4a_ward_verification.py`, now retired): that script caught
# each claim's `AssertionError` individually so it could print a full
# pass/fail breakdown for all three claims before exiting nonzero. Here,
# a failing claim halts the cell immediately with a raw traceback instead
# -- matching every other hard gate/assert in this project's notebooks
# (E1-E5, baselines), where a failure is meant to stop execution loudly,
# not be silently recorded and continued past.

# %%
_t0 = time.perf_counter()
results = {
    "seed": SEED,
    "claim1_two_point_identity": CLAIM1,
    "claim2_telescoping": CLAIM2,
    "claim3_greedy_equals_ward": CLAIM3,
    "all_passed": True,
}
results["wall_clock_seconds"] = time.perf_counter() - _t0

with open(RESULTS_DIR / "E4a_verification.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 70}\nE4a WARD VERIFICATION: ALL CLAIMS PASSED\n{'=' * 70}")
print(f"Saved {RESULTS_DIR / 'E4a_verification.json'}")

# %% [markdown]
# ## Save your results (optional -- Colab sessions are ephemeral)
#
# `results/E4a/` only exists in THIS Colab VM's disk. Push it back to your
# repo so it persists. Not run automatically -- uncomment and run only if
# you want to push.

# %%
# get_ipython().system(
#     'git add results/ && '
#     'git commit -m "E4a results" && '
#     'git push'
# )
