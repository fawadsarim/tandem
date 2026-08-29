"""jetson-orinnx/powersweep_repro_gate.py -- device-independent (pure JSON comparison, no GPU/device
needed -- can run on the Jetson or after pulling results locally). POWER-MODE SWEEP, Part 1 --
the BLOCKING reproduction gate. Compares out/powersweep/latency_lut_MAXN_SUPER.json (produced by
powersweep_measure.py --mode-label MAXN_SUPER) against the committed out/lut/latency_lut.json,
row by row, for all 5 buckets x 2 precisions (10 rows). Per the pre-registered rule in
out/powersweep/PREREGISTRATION.md: any row deviating by more than 3% HALTS the sweep -- no
lower-mode numbers are trusted until the deviation is explained.

If all 10 rows pass, this is ALSO a reproducibility result for the paper in its own right: same
board, same engines, weeks apart -- recorded as such, not just as a gate.

Usage: python powersweep_repro_gate.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, JETSON_ROOT, read_json  # noqa: E402

BUCKETS = (197, 165, 133, 101, 69)
PRECISIONS = ("fp16", "int8")
TOLERANCE_PCT = 3.0


def main():
    print("=== powersweep_repro_gate.py: Part 1 -- BLOCKING reproduction gate at MAXN_SUPER ===")

    remeasured_path = JETSON_ROOT / "out" / "powersweep" / "latency_lut_MAXN.json"
    if not remeasured_path.exists():
        remeasured_path = JETSON_ROOT / "out" / "powersweep" / "latency_lut_MAXN_SUPER.json"
    assert remeasured_path.exists(), (
        f"{remeasured_path} not found -- run "
        "`./venv/bin/python3 powersweep_measure.py --mode-label MAXN` on the Jetson first."
    )
    remeasured = read_json(remeasured_path)["entries"]

    committed_path = JETSON_ROOT / "out" / "lut" / "latency_lut.json"
    assert committed_path.exists(), f"{committed_path} not found -- this is the committed, registered LUT."
    committed = read_json(committed_path)["entries"]

    rows = []
    for n in BUCKETS:
        for prec in PRECISIONS:
            key = f"n{n}_{prec}"
            assert key in committed, f"{key} missing from committed latency_lut.json -- cannot gate."
            assert key in remeasured, f"{key} missing from the re-measured MAXN_SUPER run -- incomplete sweep."
            re_entry = remeasured[key]
            assert not re_entry.get("rebuild_needed"), (
                f"{key}: re-measurement recorded rebuild_needed=True ({re_entry.get('error')}) -- "
                "cannot gate on a failed measurement. HALTING."
            )
            committed_median = committed[key]["median_ms"]
            remeasured_median = re_entry["median_ms"]
            pct_delta = (remeasured_median - committed_median) / committed_median * 100.0
            within_tol = abs(pct_delta) <= TOLERANCE_PCT
            rows.append({
                "bucket_n_tokens": n, "precision": prec,
                "committed_median_ms": committed_median, "remeasured_median_ms": remeasured_median,
                "pct_delta": pct_delta, f"within_{TOLERANCE_PCT:.0f}pct": within_tol,
            })

    df = pd.DataFrame(rows)
    csv_path = ensure_out_dir("powersweep") / "reproduction_gate.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'bucket':>6}  {'precision':>9}  {'committed(ms)':>14}  {'remeasured(ms)':>15}  "
          f"{'%delta':>8}  {'within 3%':>10}")
    for r in rows:
        print(f"{r['bucket_n_tokens']:>6}  {r['precision']:>9}  {r['committed_median_ms']:>14.4f}  "
              f"{r['remeasured_median_ms']:>15.4f}  {r['pct_delta']:>+7.2f}%  "
              f"{'YES' if r[f'within_{TOLERANCE_PCT:.0f}pct'] else 'NO':>10}")

    n_pass = sum(1 for r in rows if r[f"within_{TOLERANCE_PCT:.0f}pct"])
    print(f"\n{n_pass}/{len(rows)} rows within {TOLERANCE_PCT:.0f}%.")
    print(f"Wrote {csv_path}")

    failures = [r for r in rows if not r[f"within_{TOLERANCE_PCT:.0f}pct"]]
    if failures:
        raise AssertionError(
            f"REPRODUCTION GATE FAILED: {len(failures)}/{len(rows)} row(s) exceed {TOLERANCE_PCT:.0f}% "
            f"deviation: {failures}. HALTING per the pre-registered hard-stop rule -- no lower-mode "
            "numbers from this sweep should be trusted until this is explained."
        )

    print(f"\nREPRODUCTION GATE PASSED -- all {len(rows)} rows within {TOLERANCE_PCT:.0f}%. This is "
          "also a reproducibility result in its own right: the same board and engines reproduce "
          "the committed latency_lut.json medians weeks later, within measurement noise.")


if __name__ == "__main__":
    main()
