# %% [markdown]
# # B5 (compare_pareto) -- standalone Colab notebook
#
# Lightweight -- no GPU, no torch/timm, no ImageNet needed. Runs the exact
# same `experiments/phase2_2_baselines/compare_pareto.py`, unmodified: reads `experiments/phase2_2_baselines/
# results.csv` (from B1-B4, whichever environment produced them -- Colab
# fallback notebooks or the university GPU) and TANDEM's own
# `results/P2/<model>/plans.csv` files (from `P2_tandem_pipeline.ipynb`),
# linearly interpolates TANDEM's Pareto curve at each baseline's measured
# cost, and writes `experiments/phase2_2_baselines/pareto_comparison.csv`. No conclusions drawn
# here -- numbers only, per spec.
#
# **Prerequisite**: `experiments/phase2_2_baselines/results.csv` (with at least one B1-B4 row)
# and at least one `results/P2/<model>/plans.csv` must already exist in
# your repo -- either push them from wherever they were produced (a B1-B4
# Colab notebook, the university GPU, or `P2_tandem_pipeline.ipynb`) before
# running this, or upload them directly into this Colab session's file
# browser under the matching paths.
#
# This repo is public -- cloning it needs no authentication.
#
# Runtime: CPU is fine (Runtime -> Change runtime type -> CPU is OK, no
# need for GPU here).

# %%
import os

REPO_URL = "https://github.com/fawadsarim/tandem.git"
REPO_DIR = "/content/tandem"

if not os.path.exists(REPO_DIR):
    get_ipython().system(f'git clone -q {REPO_URL} {REPO_DIR}')  # noqa: F821
else:
    print(f"{REPO_DIR} already exists -- pulling latest instead of re-cloning.")
    get_ipython().system(f'git -C {REPO_DIR} pull -q')  # noqa: F821

get_ipython().run_line_magic('cd', REPO_DIR)  # noqa: F821

# %%
get_ipython().system('pip install -q -r experiments/phase2_2_baselines/requirements.txt')  # noqa: F821

# %% [markdown]
# ## Check inputs are actually present before running

# %%
from pathlib import Path

results_csv = Path("experiments/phase2_2_experiments/phase2_2_baselines/results.csv")
assert results_csv.exists(), (
    f"{results_csv} not found -- push it from wherever B1-B4 were run (a Colab fallback notebook or "
    "the university GPU), or upload it into this Colab session under experiments/phase2_2_baselines/results.csv."
)

p2_dirs = sorted(p.name for p in Path("results/P2").iterdir()) if Path("results/P2").exists() else []
print(f"Found P2 results for: {p2_dirs or '(none yet)'}")
print(f"Found {results_csv}: {sum(1 for _ in open(results_csv)) - 1} result row(s).")

# %% [markdown]
# ## P2.2 post-conversion self-test (lightweight -- B5 has no GPU/upstream-repo/eval work)
#
# B5 only reads `results.csv` and interpolates -- of the 10 checks every
# other baseline notebook runs, only three apply here: wrapper purity +
# repo HEAD, secrets scan, and the results.csv schema check. A FAILURE
# HALTS here (raises) -- "Run All" cannot reach the run cell below until
# these pass.

# %%
get_ipython().system('python experiments/phase2_2_baselines/common/notebook_self_test.py --baseline compare_pareto --checks-only 1,4,7')  # noqa: F821

# %% [markdown]
# ## Run B5 -- the real, unmodified `compare_pareto.py`

# %%
get_ipython().system('python experiments/phase2_2_baselines/compare_pareto.py')  # noqa: F821

# %% [markdown]
# ## Results

# %%
import pandas as pd

print(pd.read_csv("experiments/phase2_2_experiments/phase2_2_baselines/pareto_comparison.csv").to_string(index=False))

# %% [markdown]
# ## Final P2.2 closure pass -- pooled cross-method fronts,
# ## flagged comparison table, 4-panel figure, reviewer-release package
#
# Reads ONLY `row_kind=comparison` rows (asserted, not assumed), computes
# each model's pooled-baseline Pareto front across all 4 methods, asserts
# DeiT-S's front and saturation ceiling against the pre-registered claim,
# writes `experiments/phase2_2_baselines/compare_summary.json`, and assembles
# `experiments/phase2_2_baselines/reviewer_release/`. Torch-free -- same no-GPU runtime as the
# rest of this notebook.

# %%
get_ipython().system('python experiments/phase2_2_baselines/b5_final_closure.py')  # noqa: F821

# %% [markdown]
# ## Save your results (optional -- Colab sessions are ephemeral)

# %%
# get_ipython().system(
#     'git add experiments/phase2_2_baselines/pareto_comparison.csv && '
#     'git commit -m "B5 pareto_comparison.csv" && '
#     'git push'
# )
