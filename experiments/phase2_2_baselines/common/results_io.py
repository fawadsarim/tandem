"""results.csv schema + append helper. Framework-free.

One row per (method, model, config). Every runner in this project appends
through `append_result` -- never writes results.csv directly -- so the
schema stays identical no matter which baseline (and which of the four
separate venvs, see deviations.md #1), and no matter which HOST (university
GPU or Colab, see equivalence_gate.py), produced the row.

Self-healing schema (2026-07, see `ensure_header`): if RESULTS_COLUMNS gains
a column while a results.csv from an OLDER schema already exists on disk (a
persisted file across several `git pull`s in one long Colab runtime is the
common way this happens), `ensure_header` detects and auto-migrates a pure
column-addition in place -- loudly, not silently -- before either
`append_result` or `find_existing_row` touches the file. This exists because
that exact drift once caused every row to look invisible on read (`pandas`
silently shifts an unexpected extra field into an implicit row index) after
a real ~1.2 hour debugging session.
"""
import csv
import os
from pathlib import Path

RESULTS_COLUMNS = [
    "method",           # "tome" | "repq_vit" | "fq_vit" | "sequential_composite"
    "model",             # "deit_small_patch16_224" etc.
    "config",             # human-readable config id, e.g. "r13", "w8a8", "r17_w8a8_merge_first"
    "checkpoint_tag",      # resolved checkpoint identity actually loaded for THIS row, e.g.
                           # "vit_base_patch16_224.augreg_in21k_ft_in1k" (Step A, paper's own) or
                           # "vit_base_patch16_224.augreg2_in21k_ft_in1k" (Step B, TANDEM's pinned
                           # tag) -- see common/checkpoint_pin.py. Always populated (never "").
    "row_kind",             # "reproduction" (Step A: method run on the checkpoint ITS OWN paper
                            # used, gated against a published number where one exists) |
                            # "comparison" (Step B: method run on TANDEM's PINNED checkpoint --
                            # what compare_pareto.py should read). See common/checkpoint_pin.py's
                            # module docstring for the two-step policy and why they sometimes
                            # coincide (same measurement, two rows) and sometimes don't (two runs).
    "setting",               # B1 (ToMe) only: "off_the_shelf" (our own pretrained checkpoint,
                             # ToMe applied post-hoc with NO training) | "trained_with_tome"
                             # (ToMe's own shipped checkpoint, fine-tuned WITH merging active
                             # during training -- a DIFFERENT, non-comparable set of weights;
                             # see b1_tome/published.json's "note" and ../deviations.md). "" for
                             # every other baseline/row (no training-setting ambiguity elsewhere).
                             # compare_pareto.py and any TANDEM-vs-baseline plot must read ONLY
                             # setting="off_the_shelf" (or "") rows -- a "trained_with_tome" row
                             # is a reference point, never a fair training-free comparison.
    "top1_50k",            # measured top-1 on the full 50k ImageNet-1k val ImageFolder
    "n_images",            # image count actually evaluated (always 50000 for a full run; smaller
                           # for a host-equivalence smoke run -- distinguishes the two explicitly)
    "bops_ours",             # cost under TANDEM's own BOPs model (common/bops.py)
    "repo_commit",            # exact upstream commit SHA(s) used (see commits.md)
    "published_top1_pct",      # reference used for the gate, or "" if none located
    "gate_status",               # "pass" | "fail" | "no_published_reference" | "not_applicable"
    "gate_diff_pct_points",       # abs(measured - published), or "" if not applicable
    "calib_images",                 # how many images were used for calibration, or "" if n/a
    "host",                           # "university_gpu" | "colab" | other free-text host tag
    "gpu_name",                        # torch.cuda.get_device_name(0) -- e.g. "NVIDIA A100", "Tesla T4"
    "torch_version",                    # torch.__version__ actually resolved (may differ from the pin
                                         # if the environment drifted -- recorded, not assumed)
    "cuda_version",                      # torch.version.cuda
    "config_hash",                        # sha256(sorted(pinned commit + config + seed + calib
                                           # indices/source)) -- see common/determinism.py
    "pip_freeze_hash",                      # sha256(pip freeze)[:16] of the ACTUAL venv that produced
                                             # this row (common/equivalence_gate._pip_freeze_hash) --
                                             # P2.2 audit item #4 ("pip-freeze hash recorded into
                                             # results"). A drifted dependency in a venv that still
                                             # satisfies every ==-pin (e.g. a transitive dep with no
                                             # pin of its own) changes this hash without changing any
                                             # requirements.txt line -- first place to check if a
                                             # number stops reproducing on a re-run of an "unchanged" venv.
    "notes",                                # free text: deviations, environment specifics, etc.
]


def results_csv_path(baselines_root=None):
    """`TANDEM_RESULTS_CSV_OVERRIDE` lets the host-equivalence gate
    (common/equivalence_gate.py) point a baseline's UNMODIFIED run.py at
    a separate smoke-run CSV, so a 256-image equivalence check never
    pollutes the real experiments/phase2_2_baselines/results.csv with a non-50k row."""
    override = os.environ.get("TANDEM_RESULTS_CSV_OVERRIDE")
    if override:
        return Path(override)
    root = Path(baselines_root) if baselines_root else Path(__file__).resolve().parent.parent
    return root / "results.csv"


def _is_pure_addition(old_header, new_header):
    """True iff `old_header` is a SUBSEQUENCE of `new_header` -- every name
    in `old_header` appears in `new_header`, unrenamed, in the same
    relative order (columns may have been inserted/appended around them,
    never reordered/renamed/removed). `name in it` on a shared iterator
    correctly implements the subsequence check: each lookup consumes the
    iterator up to (and including) a match, so a later name can only match
    LATER in `new_header`, never re-match something already passed."""
    it = iter(new_header)
    return all(name in it for name in old_header)


def _migrate_header_addition(csv_path, old_header, new_header):
    """Rewrites `csv_path` in place: new header, every existing row
    reconciled to it. Atomic (tmp-file-then-replace, same pattern as
    common/checkpoint.py's save_checkpoint) so a crash mid-write can never
    leave a half-migrated file. Never called except after `_is_pure_addition`
    has already confirmed no name was renamed/removed -- a REMOVED column
    would silently discard real recorded data, which this never does.

    Each row can legitimately be EITHER length: `append_result`'s
    `csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)` always writes using
    whatever RESULTS_COLUMNS is current AT WRITE TIME, completely
    independent of what the on-disk header (stale or not) actually says --
    so a row written after the schema grew already has `len(new_header)`
    fields (already in new_header's order, needs no remapping), while a
    row written before still has `len(old_header)` fields (in old_header's
    order, remapped by NAME below). A single file can contain a mix of
    both if the code changed mid-session -- both are handled per-row."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    old_rows = rows[1:]
    added = [c for c in new_header if c not in old_header]
    print(f"{csv_path}: on-disk header is missing {added} (present in the current RESULTS_COLUMNS) -- "
          f"auto-migrating: rewriting the header in place, reconciling {len(old_rows)} existing row(s). "
          "No row deleted or reordered.")
    old_index = {name: i for i, name in enumerate(old_header)}
    migrated_rows = []
    for line_num, row in enumerate(old_rows, start=2):
        if len(row) == len(new_header):
            migrated_rows.append(row)  # already written under the current (longer) schema, untouched
        elif len(row) == len(old_header):
            migrated_rows.append([row[old_index[name]] if name in old_index else "" for name in new_header])
        else:
            raise AssertionError(
                f"{csv_path} line {line_num} has {len(row)} field(s), matching neither the old header's "
                f"{len(old_header)} nor the current schema's {len(new_header)} -- this row predates a "
                "DIFFERENT, unrecognized schema version (more than one change happened across this "
                "file's life). Stop and inspect this file by hand rather than guessing further."
            )
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(migrated_rows)
    tmp_path.replace(csv_path)
    print(f"{csv_path}: migration complete, {len(migrated_rows)} row(s) preserved.")


def ensure_header(csv_path):
    """Writes RESULTS_COLUMNS as the header for a brand-new file -- OR, for
    a file that already exists, RECONCILES a stale on-disk header against
    the current RESULTS_COLUMNS before anything reads or writes it.

    Why this exists (found via a real ~1.2 hour debugging session, 2026-07):
    this function used to be a no-op whenever the file already existed. If
    results.csv persisted (e.g. across several `git pull`s within one long
    Colab runtime) while RESULTS_COLUMNS gained a column in between, every
    `append_result` after that point wrote MORE fields per row than the
    stale on-disk header named -- and `pandas.read_csv` does not error on
    that, it silently shifts the first (extra) value into an implicit row
    index, so a filter like `results_df["method"] == "tome"` matches ZERO
    rows even though the data is sitting right there, correct. The bug was
    invisible until someone actually read the file back.

    A pure column ADDITION (existing names unrenamed, same relative order --
    see `_is_pure_addition`) is auto-migrated in place, loudly (printed, not
    silent) -- this is the common case (a schema grows over a project's
    life) and safe to automate. Anything else (a rename/removal/reorder)
    halts with a clear message instead of guessing -- silently discarding
    or misattributing a real recorded number is worse than a loud stop."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(RESULTS_COLUMNS)
        return

    with open(csv_path, newline="") as f:
        try:
            on_disk_header = next(csv.reader(f))
        except StopIteration:
            on_disk_header = None  # 0-byte file -- treat like nonexistent
    if on_disk_header is None:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(RESULTS_COLUMNS)
        return
    if on_disk_header == RESULTS_COLUMNS:
        return  # already in sync -- the common case, cheap to confirm

    assert _is_pure_addition(on_disk_header, RESULTS_COLUMNS), (
        f"{csv_path}'s on-disk header does not match the current RESULTS_COLUMNS, and the difference is "
        f"NOT a pure column-addition this can safely auto-migrate (a name was renamed, removed, or "
        f"reordered):\n  on disk: {on_disk_header}\n  current: {RESULTS_COLUMNS}\n"
        "Appending now would silently misalign every row. Resolve by hand: delete the stale file if its "
        "rows are disposable, or manually reconcile the rename/removal/reorder before continuing."
    )
    _migrate_header_addition(csv_path, on_disk_header, RESULTS_COLUMNS)


def find_existing_row(csv_path=None, **filters):
    """Returns the LAST row already in results.csv matching every
    `filters` key=value pair (as strings), or None. Lets a runner make a
    crash-and-rerun idempotent: skip re-running a (method, model, config,
    row_kind, checkpoint_tag) combination already recorded, instead of
    burning GPU time re-deriving (or worse, silently duplicating) a number
    already on disk. B2/B3 have no per-batch checkpoint (deviations.md
    #14) -- this is the coarser, per-CONFIG-level resume that exists
    instead, and it costs nothing on a fresh/empty results.csv.

    Calls `ensure_header` first (not just append_result does) -- this is
    typically the FIRST thing a runner calls per config, often before any
    append_result in that same process, so the schema-reconciliation
    `ensure_header` performs must not wait for a write to happen first."""
    path = Path(csv_path) if csv_path else results_csv_path()
    if not path.exists():
        return None
    ensure_header(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in filters.items())]
    return matches[-1] if matches else None


def _assert_round_trips_clean(csv_path):
    """Post-write invariant (2026-07): after every append, the file must
    round-trip through pandas with a clean RangeIndex (never a
    MultiIndex/shifted index -- the exact corruption a header/row
    field-count mismatch causes, see `ensure_header`'s module-level note)
    and numeric `top1_50k`/`bops_ours` columns. Catches a BROADER class of
    corruption than the field-count checks in `ensure_header`/
    `_migrate_header_addition` alone would -- e.g. an unescaped comma
    inside a free-text `notes` value shifting everything after it, which
    still has the "right" field count per row but parses wrong.

    Best-effort: pandas is not a dependency of B1's/B4's own pinned venvs
    (torch==1.12.1/timm==0.4.12/etc., no pandas -- see their
    requirements.txt) -- this is skipped there, not required, since the
    framework-free checks already ran during THIS SAME write. Runs for
    real wherever pandas IS available: the orchestration environment,
    Colab's default kernel, B2's/B3's torch-free process."""
    try:
        import pandas as pd
    except ImportError:
        return
    df = pd.read_csv(csv_path)
    assert isinstance(df.index, pd.RangeIndex), (
        f"{csv_path} did NOT round-trip through pandas with a clean RangeIndex (got "
        f"{type(df.index).__name__}) immediately after a write -- this is the exact corruption a "
        "header/row field-count mismatch causes (pandas shifts an unexpected extra field into an "
        "implicit index). Halting -- a stale header must never be committable again."
    )
    for col in ("top1_50k", "bops_ours"):
        series = df[col]
        # "Blank" means either a real NaN (pandas inferred this column as
        # numeric, and this particular row's cell was genuinely empty --
        # e.g. this self-test's own resume-mechanism check writes a
        # synthetic row that never sets bops_ours) OR an empty string
        # (pandas inferred the column as object/string dtype instead,
        # because some OTHER row's value made it non-numeric -- still a
        # legitimately blank cell, not the corruption itself). Both must
        # be excluded BEFORE checking numeric-ness of what's left -- doing
        # `.astype(str)` first (the previous, buggy version of this check)
        # turns a real NaN into the 4-character string "nan", which the
        # blank-check then misses, so a legitimately empty cell gets
        # flagged as "corrupted" -- a false alarm, not a real one.
        is_blank = series.isna() | (series.astype(str).str.strip() == "")
        present = series[~is_blank]
        if not len(present):
            continue
        numeric = pd.to_numeric(present, errors="coerce")
        bad = present[numeric.isna()]
        assert bad.empty, (
            f"{csv_path}'s {col!r} column has non-numeric, non-blank value(s) after a write: "
            f"{bad.tolist()} -- the file is misaligned or corrupted. Halting rather than leaving a bad "
            "file for the next reader."
        )


def append_result(row: dict, csv_path=None):
    """`row` must contain exactly the keys in RESULTS_COLUMNS (missing keys
    are written as empty strings; unexpected keys raise -- a typo in a
    runner's row-building code should fail loudly, not silently drop a
    column)."""
    unexpected = set(row.keys()) - set(RESULTS_COLUMNS)
    assert not unexpected, f"unexpected column(s) in row: {unexpected} -- check RESULTS_COLUMNS"
    path = Path(csv_path) if csv_path else results_csv_path()
    ensure_header(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writerow({col: row.get(col, "") for col in RESULTS_COLUMNS})
    print(f"Appended to {path}: {row.get('method')}/{row.get('model')}/{row.get('config')}")
    _assert_round_trips_clean(path)
