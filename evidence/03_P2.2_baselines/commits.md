# Pinned external repo commits

Every baseline clones its upstream repo and checks out an EXACT commit --
never a branch tip, which drifts. `common/repo_utils.ensure_repo(...)` enforces
this: it clones (or fetches, if already present) and hard-checks-out the SHA
below, and asserts the checked-out `HEAD` matches before any run proceeds.

Confirmed live against the GitHub API (`default_branch` +
`/commits/{branch}` for each repo) -- these are real commits, not guessed.

| # | Repo | URL | Pinned commit (full SHA) | Commit date |
|---|------|-----|---------------------------|-------------|
| B1 | facebookresearch/ToMe | https://github.com/facebookresearch/ToMe | `af95e4b1befa172dadccd8c81e223b10090f9579` | 2023-03-31 |
| B2 | zkkli/RepQ-ViT | https://github.com/zkkli/RepQ-ViT | `a160561b9233945e6680132f129d13041b787318` | 2024-01-10 |
| B3 | megvii-research/FQ-ViT | https://github.com/megvii-research/FQ-ViT | `5daf5915c715dc21d6afcd64e661652030e640ab` | 2023-04-11 |
| B4 | (uses B1 + B2's pinned commits together; no separate repo) | -- | -- | -- |

At the time these were pinned, each SHA was the tip of the repo's default
branch (`main` in all three cases) -- i.e., these are simply "latest at
pin time", not a specific tagged release (none of the three repos publish
version tags). If either upstream repo is ever re-pinned to a newer commit,
update this table AND re-run that baseline's reproduction gate before
trusting any new numbers -- a commit bump is exactly the kind of change the
gate exists to catch.

`common/repo_utils.py` clones each repo into `external/<name>/` (relative to this file's own directory)
(git-ignored; not committed -- these are large third-party trees with their
own licenses, not TANDEM code) and refuses to proceed if the checked-out SHA
doesn't match the table above.
