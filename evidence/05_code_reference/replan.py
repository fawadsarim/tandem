"""jetson/replan.py -- re-runs P2.1's EXACT Pareto DP allocator for
deit_small_patch16_224, with MEASURED on-device latency substituted for
simulated BOPs as the cost axis. The accuracy objective (D_pred, from
Dq_table.csv/Dm_table.csv) is UNCHANGED -- only the cost function
changes, per spec. Emits plans_hw.csv (latency-optimal plans) alongside
P2.1's own plans.csv (BOPs-optimal plans, read-only) for the SAME 6
budget fractions, and prints both side by side.

THREE ARMS (third one added on review, 2026-07 -- see the
"STAGED (BEST-SEQUENTIAL) allocator" section below and
jetson/README.md's "Joint vs staged under measured latency" section):
  1. BOPs-selected -- P2.1's own committed plan (read-only).
  2. JOINT (latency-selected) -- this file's own ported DP, cost axis
     substituted to measured latency.
  3. STAGED (latency-selected, "BEST-SEQUENTIAL") -- TANDEM's staged
     baseline (ported verbatim from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py),
     re-run under the SAME measured-latency cost axis and the SAME
     per-budget cost ceiling (tau_b) as arm 2. Arms 1 and 2 test "does
     the cost proxy matter" (the originally registered question). Arm 3
     tests the thesis's actual headline claim, "does joint beat staged",
     under real hardware cost -- which only TIED under the BOPs proxy
     (experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's "staging collapse" finding).

READ-ONLY access to P2.1: this script reads
experiments/phase2_1_fleet/results/P2/deit_small_patch16_224/{Dq_table.csv, Dm_table.csv,
plans.csv, P2_deit_small_patch16_224_summary.json} via jetson/_common.py's
read-only helpers, and never writes to that directory. All jetson/ output
goes to jetson/out/replan/.

THE DP, PORTED (not imported -- jetson/ cannot import experiments/ code, see
_common.py's docstring): `enumerate_schedules`, `objective_d_pred`,
`pareto_frontier`, `exact_pareto_bits`, `exact_select_for_budget`,
`build_schedule_pareto_sets` below are a faithful port of
experiments/phase2_1_fleet/P2_tandem_pipeline.py's sections 10, 18-19 (S6 allocator), with
one substitution: every call to `block_bops(n_l, bits)` there becomes a
call to `block_latency(n_l, bits)` here. BIT_MENU, MERGE_BLOCKS, MERGE_RS,
KAPPA, BUDGET_FRACTIONS are the same model-independent plan-space
constants P2.1 uses (Phase-1 validated, cited verbatim in
experiments/phase2_1_fleet/P2_tandem_pipeline.py section 10) -- reproduced here as
constants, not re-derived, since they are plan-SPACE definition, not
measured data.

BITS -> HARDWARE PRECISION MAPPING (a real, documented assumption, not a
detail to gloss over): TensorRT/this Jetson has no native INT6 kernel.
BIT_MENU = (16, 8, 6) is a SIMULATED weight-quantization scheme in P2.1's
own accuracy model (`_quantize_weight_rows` fake-quantizes to N bits in
plain PyTorch, it never touches a distinct hardware datapath per bit
count) -- for LATENCY purposes, bits=16 executes as the FP16 engine and
bits in {8, 6} BOTH execute as the INT8 engine (there is no separate
"6-bit engine" on this hardware; a plan calling for 6-bit blocks is
assumed to run at measured INT8 speed). This is stated explicitly here
and in the printed output/JSON so it cannot be missed or silently assumed
away.

TOKEN-COUNT INTERPOLATION: the plan space can require an entering token
count that isn't one of the 5 measured buckets (e.g. n_l=99, from a single
r=98 merge point -- see export_segments.py's TOKEN_BUCKETS). Latency at an
unmeasured n_l is LINEARLY interpolated between the two nearest measured
buckets (latency scales continuously and close to linearly in token count
for a fixed-architecture transformer block, same reasoning the BOPs model
itself uses -- block_bops(n_l, bits) is a smooth function of n_l).
Every plan's n_l values fall inside [69, 197] (the buckets' own min/max,
by construction of the merge grid -- see MERGE_RS), so this is always
interpolation, never extrapolation; asserted below.

BUDGET PAIRING (corrected on review, 2026-07 -- a real methodology fix, not a bug
worked around): tau_b (the per-budget latency ceiling the DP searches under) is the MEASURED
LATENCY of P2.1's own BOPs-selected plan at budget b -- NOT `fraction x Latency_id` (the
original pairing, which this file used until this fix). The original pairing was a real
confound: BOPs cost scales with a much wider dynamic range than real measured latency does on
this hardware (n_l^2 for attention, bits^2 for quantization, vs. real latency's near-fixed
per-block overhead where bits<16 measures SLOWER not faster -- see the INT8 findings in
README.md), so most of BUDGET_FRACTIONS' nominal targets fell below the plan space's true
achievable floor (~76% of Latency_id using this hardware's own measured numbers -- see
`find_min_achievable_cost()`) -- 5 of 6 budgets were mathematically infeasible under that
pairing, not because the task was infeasible, but because of how the budget itself was defined.
The corrected pairing (tau_b = the BOPs plan's own measured cost) guarantees feasibility BY
CONSTRUCTION: the BOPs-selected plan is itself always a valid candidate at cost tau_b, in the
identical 85-schedule x BIT_MENU plan space `--verify-port` already proved this port searches
faithfully -- so the latency-selected search can only match or beat it. This also reframes the
comparison to what it was meant to test: at EQUAL real milliseconds, which selection rule
(BOPs-guided or latency-guided) gives better predicted accuracy -- not "which budget survives
being redefined in a proxy that doesn't transfer." Any INFEASIBLE result under this pairing
now indicates a real bug (in the DP or the LUT), not a hardware-reality finding, and is
asserted as such in main().

Usage: python replan.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    compute_plan_divergence, ensure_out_dir, JETSON_ROOT, read_json, read_p2_csv_path, read_p2_summary, write_json,
)

MODEL_NAME = "deit_small_patch16_224"
N_TOKENS_TOTAL = 197  # cross-checked against P2.1's summary below, not hardcoded blindly
BIT_MENU = (16, 8, 6)
MERGE_BLOCKS = (1, 3, 5, 7, 9, 11)
MERGE_RS = (32, 64, 98, 128)
KAPPA = {0: 1.0, 1: 1.0, 2: 1.15}
BUDGET_FRACTIONS = (0.15, 0.21, 0.293, 0.41, 0.572, 0.80)
TOKEN_BUCKETS = (197, 165, 133, 101, 69)


def bits_to_precision(bits: int) -> str:
    return "fp16" if bits == 16 else "int8"


def load_dq_dm_tables():
    """READ-ONLY from P2.1's own Dq_table.csv/Dm_table.csv (block,bits/r,D_mse columns --
    see this repo's experiments/phase2_2_baselines/b5_final_closure.py-style provenance discipline: read the
    already-computed table, never re-derive the regression that produced it)."""
    dq_path = read_p2_csv_path(MODEL_NAME, "Dq_table.csv")
    dm_path = read_p2_csv_path(MODEL_NAME, "Dm_table.csv")
    dq_df, dm_df = pd.read_csv(dq_path), pd.read_csv(dm_path)
    dq_table = {(int(r.block), int(r.bits)): float(r.D_mse) for r in dq_df.itertuples()}
    dm_table = {(int(r.block), int(r.r)): float(r.D_mse) for r in dm_df.itertuples()}
    return dq_table, dm_table


def load_latency_lut():
    # JETSON_ROOT-relative, not CWD-relative -- real bug found and fixed elsewhere in this
    # folder, 2026-07 (see measure_lut.py's own comment and jetson/README.md's INT8 findings
    # section) -- fixed here too before it caused the same failure.
    lut_path = JETSON_ROOT / "out" / "lut" / "latency_lut.json"
    assert lut_path.exists(), f"{lut_path} not found -- run measure_lut.py on the Jetson first."
    lut = read_json(lut_path)["entries"]
    by_precision = {"fp16": {}, "int8": {}}
    for entry in lut.values():
        # Real bug found and fixed, 2026-07: measure_lut.py also records "int8_naive_implicit"
        # entries now (provenance/comparison only -- see that file's own module docstring's
        # "WHICH ENGINE 'int8' MEANS" section) -- this DP only ever reads "fp16"/"int8" (best
        # achievable), so anything else is skipped here rather than indexed into a dict that
        # was never meant to hold it.
        if entry["precision"] not in by_precision:
            continue
        by_precision[entry["precision"]][entry["n_tokens"]] = entry["median_ms"]
    for prec in ("fp16", "int8"):
        missing = set(TOKEN_BUCKETS) - set(by_precision[prec].keys())
        assert not missing, f"latency_lut.json missing {prec} entries for buckets {missing}"
    return by_precision


def make_block_latency_fn(lut_by_precision):
    buckets_sorted = sorted(TOKEN_BUCKETS)

    def block_latency(n_l: int, bits: int) -> float:
        precision = bits_to_precision(bits)
        table = lut_by_precision[precision]
        assert buckets_sorted[0] <= n_l <= buckets_sorted[-1], (
            f"n_l={n_l} outside measured bucket range [{buckets_sorted[0]}, {buckets_sorted[-1]}] "
            "-- this should be structurally impossible given MERGE_RS; halting rather than extrapolating."
        )
        if n_l in table:
            return table[n_l]
        lo = max(b for b in buckets_sorted if b <= n_l)
        hi = min(b for b in buckets_sorted if b >= n_l)
        if lo == hi:
            return table[lo]
        frac = (n_l - lo) / (hi - lo)
        return table[lo] * (1 - frac) + table[hi] * frac

    return block_latency


# --- plan representation (identical to P2.1's, ported) ---
def plan_bits(plan, l):
    return plan["bits"].get(str(l), plan["bits"].get(l, 16))


def n_tokens_entering(plan, l):
    return N_TOKENS_TOTAL - sum(m["r"] for m in plan["merges"] if m["block"] < l)


def plan_cost(plan, block_cost_fn):
    return sum(block_cost_fn(n_tokens_entering(plan, l), plan_bits(plan, l)) for l in range(1, 13))


def objective_d_pred(plan, dq_table, dm_table):
    d_quant = sum(dq_table[(l, plan_bits(plan, l))] for l in range(1, 13) if plan_bits(plan, l) != 16)
    d_merge = sum(dm_table[(m["block"], m["r"])] for m in plan["merges"])
    kappa = KAPPA.get(len(plan["merges"]), KAPPA[2])
    return d_quant + kappa * d_merge


def enumerate_schedules(merge_blocks, merge_rs, max_points):
    """Ported verbatim from experiments/phase2_1_fleet/P2_tandem_pipeline.py's enumerate_schedules."""
    schedules = [[]]
    single_points = [{"block": b, "r": r} for b in merge_blocks for r in merge_rs]
    schedules += [[p] for p in single_points]
    if max_points >= 2:
        for b1, b2 in itertools.combinations(merge_blocks, 2):
            for r1 in merge_rs:
                for r2 in merge_rs:
                    feasible = r2 <= (N_TOKENS_TOTAL - 1) - r1 - 1
                    in_regime = r1 + r2 <= max(merge_rs)
                    if feasible and in_regime:
                        schedules.append([{"block": b1, "r": r1}, {"block": b2, "r": r2}])
    return schedules


def pareto_frontier(points, tol=1e-12):
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier, best_d_so_far = [], float("inf")
    for p in pts:
        if p[1] < best_d_so_far - tol:
            frontier.append(p)
            best_d_so_far = p[1]
    return frontier


def exact_pareto_bits(schedule, blocks_in_scope, bit_menu, dq_table, block_cost_fn):
    """`block_cost_fn(n_l, bits) -> float` is generic over the cost axis --
    called with `block_latency` (measured latency, the normal on-device
    mode) or `block_bops` (P2.1's own BOPs formula, the --verify-port
    mode below) -- the DP itself is IDENTICAL either way, which is exactly
    what makes --verify-port a genuine test of the port's faithfulness
    rather than a separate reimplementation that could hide its own bugs."""
    plan_stub = {"bits": {}, "merges": schedule}
    fixed_outside_scope_cost = sum(
        block_cost_fn(n_tokens_entering(plan_stub, l), 16) for l in range(1, 13) if l not in blocks_in_scope
    )
    pareto_set = [(fixed_outside_scope_cost, 0.0, {})]
    for l in blocks_in_scope:
        n_l = n_tokens_entering(plan_stub, l)
        options = [(block_cost_fn(n_l, bits), (dq_table[(l, bits)] if bits != 16 else 0.0), bits)
                   for bits in bit_menu]
        expanded = []
        for cum_c, cum_d, choices in pareto_set:
            for c, d, bits in options:
                new_choices = dict(choices)
                new_choices[l] = bits
                expanded.append((cum_c + c, cum_d + d, new_choices))
        pareto_set = pareto_frontier(expanded)
    return pareto_set


def build_schedule_pareto_sets(schedules, blocks_in_scope, bit_menu, dq_table, dm_table, block_cost_fn):
    result = []
    for schedule in schedules:
        kappa = KAPPA.get(len(schedule), KAPPA[2])
        merge_damage = kappa * sum(dm_table[(m["block"], m["r"])] for m in schedule) if schedule else 0.0
        pareto = exact_pareto_bits(schedule, blocks_in_scope, bit_menu, dq_table, block_cost_fn)
        result.append((schedule, merge_damage, pareto))
    return result


# --- STAGED (BEST-SEQUENTIAL) allocator, ported verbatim from experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's
# sequential_s1_tokens_then_bits / sequential_s2_bits_then_tokens / best_sequential_for_budget
# (STOP-2 ruled exact version, "Kim et al. compliance" -- see that file's own section
# 25 markdown for the full reasoning). Added here on review, 2026-07 (see
# jetson/README.md's "Joint vs staged under measured latency" section): the original G1 (E5)
# finding was that JOINT and BEST-SEQUENTIAL tie exactly under BOPs cost ("staging collapse"),
# a consequence of D_pred being additively separable in (schedule, bits) whenever the
# joint-optimal schedule is also token-frontier-efficient. Real measured latency is NOT
# separable this way (a block's cost is a genuinely joint, non-linear function of token count
# AND precision -- see the INT8 findings above), so re-running this exact same staged
# procedure under the SAME LUT and SAME tau_b as the joint comparison directly tests whether
# TANDEM's central "joint beats staged" claim, which only tied under the BOPs proxy, actually
# separates under real hardware cost. Every primitive below (plan_cost, objective_d_pred,
# exact_pareto_bits, KAPPA) is the SAME already-ported function the joint DP above uses, with
# the SAME block_cost_fn substitution principle (BOPs -> latency) -- this is not a separate
# reimplementation that could hide its own bugs, by the same logic --verify-port relies on.
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]


def sequential_s1_tokens_then_bits(budget, alpha, schedules, cost_id, dq_table, dm_table, block_cost_fn):
    """Stage 1 picks the token schedule minimizing kappa-weighted merge damage subject to
    token-ONLY cost reduction (at FP16 bits, i.e. bits={}) >= alpha * (cost_id - budget). Stage
    2 fixes that schedule and runs exact_pareto_bits to pick bits. Returns None if no schedule
    meets the stage-1 threshold, or no bit choice on the chosen schedule fits the budget."""
    S = cost_id - budget
    target_reduction = alpha * S
    candidates = []
    for s in schedules:
        token_only_cost = plan_cost({"bits": {}, "merges": s}, block_cost_fn)
        reduction = cost_id - token_only_cost
        if reduction >= target_reduction - 1e-9:
            kappa = KAPPA.get(len(s), KAPPA[2])
            merge_damage = kappa * sum(dm_table[(m["block"], m["r"])] for m in s)
            candidates.append((merge_damage, s))
    if not candidates:
        return None
    _, chosen_schedule = min(candidates, key=lambda t: t[0])

    kappa = KAPPA.get(len(chosen_schedule), KAPPA[2])
    merge_damage = kappa * sum(dm_table[(m["block"], m["r"])] for m in chosen_schedule) if chosen_schedule else 0.0
    pareto = exact_pareto_bits(chosen_schedule, list(range(1, 13)), BIT_MENU, dq_table, block_cost_fn)
    feasible = [(c, bd + merge_damage, bits) for c, bd, bits in pareto if c <= budget + 1e-6]
    if not feasible:
        return None
    c_best, d_best, bits_best = min(feasible, key=lambda t: t[1])
    plan_best = {"bits": bits_best, "merges": chosen_schedule}
    return {"plan": plan_best, "cost": c_best, "D_pred": d_best, "order": "S1_tokens_then_bits", "alpha": alpha}


def sequential_s2_bits_then_tokens(budget, alpha, schedules, cost_id, dq_table, dm_table, block_cost_fn):
    """Mirrors S1 with the two axes swapped: Stage 1 fixes bits via exact_pareto_bits on the
    EMPTY schedule (schedule=[], i.e. no merges) subject to the same alpha-savings threshold;
    Stage 2 fixes those bits and scans all schedules directly for min objective_d_pred."""
    S = cost_id - budget
    target_reduction = alpha * S
    pareto_empty = exact_pareto_bits([], list(range(1, 13)), BIT_MENU, dq_table, block_cost_fn)
    candidates = [(c, bd, bits) for c, bd, bits in pareto_empty if (cost_id - c) >= target_reduction - 1e-9]
    if not candidates:
        return None
    _, _, fixed_bits = min(candidates, key=lambda t: t[1])

    stage2_candidates = []
    for s in schedules:
        plan = {"bits": fixed_bits, "merges": s}
        c = plan_cost(plan, block_cost_fn)
        if c <= budget + 1e-6:
            stage2_candidates.append((objective_d_pred(plan, dq_table, dm_table), plan, c))
    if not stage2_candidates:
        return None
    d_best, plan_best, c_best = min(stage2_candidates, key=lambda t: t[0])
    return {"plan": plan_best, "cost": c_best, "D_pred": d_best, "order": "S2_bits_then_tokens", "alpha": alpha}


def best_sequential_for_budget(budget, schedules, cost_id, dq_table, dm_table, block_cost_fn):
    """BEST-SEQUENTIAL(B) is the min-D_pred plan over BOTH orders x the full ALPHA_GRID (11
    thresholds) -- a single-order sequential baseline is a strawman; the STRONGEST sequential
    candidate (whichever order+alpha performs best) is what JOINT must be tested against."""
    candidates = []
    for alpha in ALPHA_GRID:
        for fn in (sequential_s1_tokens_then_bits, sequential_s2_bits_then_tokens):
            r = fn(budget, alpha, schedules, cost_id, dq_table, dm_table, block_cost_fn)
            if r is not None:
                candidates.append(r)
    assert candidates, f"no feasible sequential (staged) plan found for budget={budget:.4f} at any order/alpha"
    return min(candidates, key=lambda c: c["D_pred"])


def block_bops(n_l: int, bits: int, d_model: int = 384) -> float:
    """Verbatim port of P2.1's own block_bops formula
    (experiments/phase2_1_fleet/P2_tandem_pipeline.py section 10) -- used ONLY by
    --verify-port below, never by the normal on-device latency path.
    d_model=384 is deit_small_patch16_224's own dim, cross-checked against
    P2.1's canonical summary in verify_port_equivalence()."""
    linear_macs = 12 * n_l * d_model ** 2
    attention_macs = 2 * n_l ** 2 * d_model
    return linear_macs * (bits * bits) + attention_macs * (16 * 16)


def exact_select_for_budget(schedule_pareto_sets, budget, raise_on_infeasible=True):
    """raise_on_infeasible=True (default) is verify_port_equivalence()'s own behavior --
    unchanged: infeasibility there means the ported DP disagrees with P2.1's own allocator
    under IDENTICAL BOPs cost semantics, which really would be a port bug, not a finding.

    raise_on_infeasible=False is main()'s real-hardware-latency path -- real finding, found
    and handled (not fixed away), 2026-07: with measured latency as the cost axis, several
    BUDGET_FRACTIONS (inherited unchanged from P2.1's BOPs-calibrated plan space) turn out to
    be below the minimum latency ANY plan in the 85-schedule plan space can achieve on this
    hardware -- BOPs cost has a much wider achievable dynamic range (scales with n_l^2 for
    attention, bits^2 for quantization) than real measured latency does here (dominated by
    near-fixed per-block overhead, and bits<16 measured SLOWER not faster -- see the INT8
    findings section of jetson/README.md). This is a real, structurally-forced result, not a
    bug -- the previous behavior (hard `assert`) crashed the whole script on the FIRST
    infeasible budget, silently preventing output for every OTHER budget including ones that
    ARE feasible. Returns None instead so main() can record "infeasible" as an explicit,
    reported finding per budget, matching this project's standing "always report, never
    silently hide" principle (same one behind the plan-divergence log, the INT8 gate writing
    unconditionally, etc.)."""
    best = None
    for schedule, merge_damage, pareto in schedule_pareto_sets:
        for c, bits_d, bits in pareto:
            if c > budget + 1e-9:
                continue
            total_d = bits_d + merge_damage
            if best is None or total_d < best[1] - 1e-15:
                best = (c, total_d, {"bits": bits, "merges": schedule})
    if best is None:
        if raise_on_infeasible:
            raise AssertionError(f"no feasible latency-DP plan for budget={budget:.4f}ms")
        return None
    return best


def verify_port_equivalence():
    """PORT-EQUIVALENCE GATE (pre-flight patch, item 1) -- blocking, no
    device needed. Runs the exact same PORTED DP as the normal on-device
    path, but with `block_bops` (P2.1's own BOPs formula) substituted for
    `block_latency` as the cost function -- i.e. the identical code this
    file uses for the real latency-based replan, just fed a different
    cost function. If the resulting 6 plans (per-block bits, per-block
    token counts, cost, D_pred) don't exactly match P2.1's own real
    plans.csv, the PORT itself (schedule enumeration, per-schedule bit
    DP, cross-schedule budget selection) has a bug -- independent of
    whether latency or BOPs is the cost axis. This is the only available
    proof the port is faithful: without it, a measured on-device delta
    could not be attributed to latency specifically, since it might
    instead be an artifact of the port disagreeing with P2.1's own
    allocator even under IDENTICAL cost semantics.

    Prints a 6/6 MATCH checklist. Raises (halts) on the FIRST budget with
    any mismatch, naming the first differing block -- per spec, this must
    stop before any hardware measurement is attempted.
    """
    print("=== replan.py --verify-port: PORT-EQUIVALENCE GATE (no device needed) ===")
    p2_summary = read_p2_summary(MODEL_NAME)
    d_model = p2_summary["model_dims"]["d"]
    assert p2_summary["model_dims"]["n_tokens_total"] == N_TOKENS_TOTAL, "N_TOKENS_TOTAL constant drifted from P2.1"
    print(f"d_model={d_model} (cross-checked against P2.1's canonical summary)")

    dq_table, dm_table = load_dq_dm_tables()

    def block_cost_fn(n_l, bits):
        return block_bops(n_l, bits, d_model=d_model)

    identity_plan = {"bits": {}, "merges": []}
    c_id_ported = plan_cost(identity_plan, block_cost_fn)
    c_id_p2 = p2_summary["c_id"]
    assert abs(c_id_ported - c_id_p2) / c_id_p2 < 1e-9, (
        f"C_id mismatch before any DP even runs: ported={c_id_ported:.6e} P2.1={c_id_p2:.6e} -- "
        "the block_bops port itself is wrong; halting."
    )
    print(f"C_id: ported={c_id_ported:.6e}  P2.1={c_id_p2:.6e}  MATCH")

    budgets_bops = [f * c_id_ported for f in BUDGET_FRACTIONS]

    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules, got {len(schedules)}"
    schedule_pareto_sets = build_schedule_pareto_sets(
        schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_cost_fn
    )

    bops_df = pd.read_csv(read_p2_csv_path(MODEL_NAME, "plans.csv"))
    assert len(bops_df) == len(BUDGET_FRACTIONS), (
        f"P2.1 plans.csv has {len(bops_df)} rows, expected {len(BUDGET_FRACTIONS)} (one per budget fraction)"
    )

    results = []
    for i, (budget_frac, budget) in enumerate(zip(BUDGET_FRACTIONS, budgets_bops)):
        c, d_pred, ported_plan = exact_select_for_budget(schedule_pareto_sets, budget)
        p2_row = bops_df.iloc[i]
        p2_plan = json.loads(p2_row["plan_json"])
        p2_c, p2_d_pred = float(p2_row["C"]), float(p2_row["D_pred"])

        plans_identical, n_blocks_differing, per_block_diff = compute_plan_divergence(
            ported_plan, p2_plan, N_TOKENS_TOTAL
        )
        c_match = abs(c - p2_c) / p2_c < 1e-6
        d_pred_match = abs(d_pred - p2_d_pred) < 1e-9
        budget_match = plans_identical and c_match and d_pred_match

        results.append({
            "budget_fraction": budget_frac, "match": budget_match,
            "plans_identical": plans_identical, "n_blocks_differing": n_blocks_differing,
            "c_ported": c, "c_p2": p2_c, "c_match": c_match,
            "d_pred_ported": d_pred, "d_pred_p2": p2_d_pred, "d_pred_match": d_pred_match,
            "per_block_diff": per_block_diff,
        })
        status = "MATCH" if budget_match else "FAIL"
        print(f"  [{status}] budget_fraction={budget_frac:.1%}: plans_identical={plans_identical} "
              f"n_blocks_differing={n_blocks_differing} C_match={c_match} D_pred_match={d_pred_match}")

    n_match = sum(1 for r in results if r["match"])
    print(f"\nPort-equivalence checklist: {n_match}/{len(results)} MATCH")

    failures = [r for r in results if not r["match"]]
    if failures:
        first = failures[0]
        first_diff_block = next(d for d in first["per_block_diff"] if d["differs"])
        print(f"\nFIRST FAILURE: budget_fraction={first['budget_fraction']:.1%}")
        print(f"  first differing block: {first_diff_block}")
        print(f"  C: ported={first['c_ported']:.6e} P2.1={first['c_p2']:.6e}")
        print(f"  D_pred: ported={first['d_pred_ported']:.6f} P2.1={first['d_pred_p2']:.6f}")
        raise AssertionError(
            f"PORT-EQUIVALENCE GATE FAILED at budget_fraction={first['budget_fraction']:.1%} "
            f"(block {first_diff_block['block']} differs: ported={first_diff_block['a']}, "
            f"P2.1={first_diff_block['b']}) -- the port disagrees with P2.1's own allocator under "
            "IDENTICAL cost semantics. HALTING -- do not proceed to any hardware measurement."
        )

    print(f"\nPORT-EQUIVALENCE GATE PASSED: {n_match}/{len(results)} MATCH. The port is faithful; a "
          "measured on-device delta can be attributed to latency, not a DP-port bug.")
    return results


def find_min_achievable_cost(schedule_pareto_sets):
    """The absolute cheapest latency ANY plan in the 85-schedule x BIT_MENU plan space can
    achieve, ignoring D_pred entirely. STANDING FINDING (requested, 2026-07): reports
    the plan space's true achievable latency floor on real hardware -- well above what
    BUDGET_FRACTIONS' naive fraction-of-Latency_id framing implies, since merging buys
    sublinear latency savings and INT8 buys none here (see README.md's INT8 findings)."""
    return min(c for schedule, merge_damage, pareto in schedule_pareto_sets for c, bits_d, bits in pareto)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-port", action="store_true",
                         help="PORT-EQUIVALENCE GATE: run the ported DP with P2.1's own BOPs cost "
                              "function instead of the latency LUT, and assert exact agreement with "
                              "P2.1's real plans.csv at all 6 budgets. No device/LUT needed. Blocking "
                              "pre-flight check before any hardware measurement -- see self_test.py.")
    args = parser.parse_args()

    if args.verify_port:
        verify_port_equivalence()
        return

    print("=== replan.py: deit_small_patch16_224, latency-based Pareto DP ===")
    p2_summary = read_p2_summary(MODEL_NAME)
    assert p2_summary["model_dims"]["n_tokens_total"] == N_TOKENS_TOTAL, "N_TOKENS_TOTAL constant drifted from P2.1"

    dq_table, dm_table = load_dq_dm_tables()
    lut_by_precision = load_latency_lut()
    block_latency_fn = make_block_latency_fn(lut_by_precision)

    identity_plan = {"bits": {}, "merges": []}
    latency_id = plan_cost(identity_plan, block_latency_fn)
    print(f"Latency_id (identity plan, all-FP16, r=0) = {latency_id:.4f} ms")

    schedules = enumerate_schedules(MERGE_BLOCKS, MERGE_RS, max_points=2)
    assert len(schedules) == 85, f"expected 85 schedules (same plan space as P2.1), got {len(schedules)}"

    schedule_pareto_sets = build_schedule_pareto_sets(
        schedules, list(range(1, 13)), BIT_MENU, dq_table, dm_table, block_latency_fn
    )

    # --- STANDING FINDING: the plan space's true achievable latency floor, independent of any
    # budget (requested, 2026-07) -- see module docstring's "BUDGET PAIRING" section.
    min_achievable_ms = find_min_achievable_cost(schedule_pareto_sets)
    min_achievable_frac = min_achievable_ms / latency_id
    print(f"\nPlan-space achievable latency floor: {min_achievable_ms:.4f}ms "
          f"({min_achievable_frac:.1%} of Latency_id) -- the cheapest ANY plan in this "
          f"85-schedule x {len(BIT_MENU)}-bit-menu space can be, regardless of accuracy.")
    print(f"Real achievable range on this hardware: [{min_achievable_frac:.1%}, 100%]  vs.  "
          f"BUDGET_FRACTIONS' naive range: [{min(BUDGET_FRACTIONS):.1%}, {max(BUDGET_FRACTIONS):.1%}] "
          "(calibrated against BOPs cost's much wider dynamic range -- see README.md's INT8 findings).")

    # --- BUDGET PAIRING (registered protocol, corrected 2026-07): tau_b = measured
    # latency of P2.1's own BOPs-selected plan at budget b, NOT fraction x Latency_id. See
    # module docstring's "BUDGET PAIRING" section for the full reasoning. Guarantees
    # feasibility by construction -- any INFEASIBLE result below is a real bug, not a finding.
    bops_df = pd.read_csv(read_p2_csv_path(MODEL_NAME, "plans.csv"))
    assert len(bops_df) == len(BUDGET_FRACTIONS), (
        f"P2.1 plans.csv has {len(bops_df)} rows, expected {len(BUDGET_FRACTIONS)} (one per budget fraction)"
    )
    bops_plans = [json.loads(row["plan_json"]) for _, row in bops_df.iterrows()]
    taus_ms = [plan_cost(p, block_latency_fn) for p in bops_plans]
    bops_d_preds = [objective_d_pred(p, dq_table, dm_table) for p in bops_plans]

    print("\nBudget pairing: tau_b = measured latency of P2.1's own BOPs-selected plan at each budget")
    print(f"{'budget (nominal)':>17}  {'tau_b (ms)':>11}  {'tau_b / Latency_id':>19}")
    for budget_frac, tau_ms in zip(BUDGET_FRACTIONS, taus_ms):
        print(f"{budget_frac:>16.1%}  {tau_ms:>11.4f}  {tau_ms / latency_id:>18.1%}")

    rows_hw = []
    for i, budget_frac in enumerate(BUDGET_FRACTIONS):
        tau_ms = taus_ms[i]
        result = exact_select_for_budget(schedule_pareto_sets, tau_ms, raise_on_infeasible=False)
        assert result is not None, (
            f"INFEASIBLE at budget_fraction={budget_frac:.1%} (tau_b={tau_ms:.4f}ms) -- this "
            "should be IMPOSSIBLE by construction: P2.1's own BOPs-selected plan at this budget "
            "is itself always a feasible candidate at its own measured cost, in the SAME "
            "85-schedule x BIT_MENU plan space --verify-port already proved this port searches "
            "faithfully. This indicates a real bug in the DP or the LUT, not a hardware-reality "
            "finding -- halting rather than silently reporting a wrong result."
        )
        c, d_pred, plan = result
        plans_identical, n_blocks_differing, per_block_diff = compute_plan_divergence(
            bops_plans[i], plan, N_TOKENS_TOTAL
        )

        # --- THIRD ARM: TANDEM-staged (BEST-SEQUENTIAL), same tau_b, same LUT (added,
        # 2026-07 -- see module docstring). G1 (E5) found JOINT and staged tie EXACTLY under
        # BOPs cost ("staging collapse"); real measured latency is not separable the same way,
        # so this directly tests whether the thesis's central "joint beats staged" claim
        # separates under real hardware cost, where it only tied under the BOPs proxy.
        staged_result = best_sequential_for_budget(tau_ms, schedules, latency_id, dq_table, dm_table, block_latency_fn)
        staged_plan = staged_result["plan"]
        joint_staged_identical, n_blocks_differing_joint_staged, _ = compute_plan_divergence(
            plan, staged_plan, N_TOKENS_TOTAL
        )

        rows_hw.append({
            "budget_fraction": budget_frac, "tau_ms": tau_ms,
            "bops_plan_latency_ms": tau_ms,  # == tau_ms by construction; kept explicit for clarity
            "bops_plan_d_pred": bops_d_preds[i], "bops_plan_json": json.dumps(bops_plans[i]),
            "latency_plan_latency_ms": c, "latency_plan_d_pred": d_pred, "plan_json": json.dumps(plan),
            "staged_plan_latency_ms": staged_result["cost"], "staged_plan_d_pred": staged_result["D_pred"],
            "staged_plan_json": json.dumps(staged_plan),
            "staged_order": staged_result["order"], "staged_alpha": staged_result["alpha"],
            "utilization": c / tau_ms,
            "plans_identical": plans_identical, "n_blocks_differing": n_blocks_differing,
            "joint_staged_identical": joint_staged_identical,
            "n_blocks_differing_joint_staged": n_blocks_differing_joint_staged,
        })
        print(f"\n  budget_fraction={budget_frac:.1%}  tau_b={tau_ms:.4f}ms")
        print(f"    BOPs-selected:    latency={tau_ms:.4f}ms (== tau_b, by definition)  "
              f"D_pred={bops_d_preds[i]:.6f}  bits={bops_plans[i]['bits']} merges={bops_plans[i]['merges']}")
        print(f"    latency-selected (JOINT):  latency={c:.4f}ms (util={c / tau_ms:.1%})  D_pred={d_pred:.6f}  "
              f"bits={plan['bits']} merges={plan['merges']}")
        print(f"    latency-selected (STAGED): latency={staged_result['cost']:.4f}ms  "
              f"D_pred={staged_result['D_pred']:.6f}  winner={staged_result['order']} alpha={staged_result['alpha']}  "
              f"bits={staged_plan['bits']} merges={staged_plan['merges']}")
        print(f"    BOPs vs JOINT:   plans_identical={plans_identical}  n_blocks_differing={n_blocks_differing}")
        print(f"    JOINT vs STAGED: plans_identical={joint_staged_identical}  "
              f"n_blocks_differing={n_blocks_differing_joint_staged}  "
              f"D_pred(joint) {'<=' if d_pred <= staged_result['D_pred'] + 1e-12 else '>'} D_pred(staged) "
              "(JOINT must weakly dominate STAGED in predicted D_pred by construction -- see "
              "module docstring's permanent-guard note)")
        assert d_pred <= staged_result["D_pred"] + 1e-9, (
            f"budget_fraction={budget_frac:.1%}: JOINT's D_pred ({d_pred:.6f}) is WORSE than "
            f"STAGED's ({staged_result['D_pred']:.6f}) -- JOINT explores the exact per-schedule "
            "Pareto set of all 85 schedules, a strictly larger candidate set than any staged "
            "procedure's single fixed schedule/bits choice, so JOINT must weakly dominate STAGED "
            "in PREDICTED D_pred. This should be structurally impossible -- real bug, halting "
            "rather than silently reporting a wrong result (same guard experiments/phase1_E1-E5_E4a/"
            "E5_allocator_G1.py's own joint-vs-sequential comparison already asserts, ported here)."
        )

    hw_df = pd.DataFrame(rows_hw)
    out_dir = ensure_out_dir("replan")
    hw_path = out_dir / "plans_hw.csv"
    hw_df.to_csv(hw_path, index=False)
    print(f"\nWrote {hw_path}")

    # PLAN-DIVERGENCE LOG (pre-flight patch, item 2): per-block (n_l, bits) diff between the
    # BOPs-selected and latency-selected plans -- see module docstring. If these coincide at
    # every budget, a null accuracy delta downstream means "hardware cost was effectively
    # additive on this device", a DIFFERENT finding from "joint allocation does not help";
    # only this distinguishes the two.
    divergence_path = out_dir / "plans_hw_divergence.json"
    write_json(divergence_path, {
        "model_name": MODEL_NAME,
        "budgets": [
            {"budget_fraction": r["budget_fraction"],
             "bops_vs_joint_identical": r["plans_identical"],
             "bops_vs_joint_n_blocks_differing": r["n_blocks_differing"],
             "joint_vs_staged_identical": r["joint_staged_identical"],
             "joint_vs_staged_n_blocks_differing": r["n_blocks_differing_joint_staged"]}
            for r in rows_hw
        ],
    })
    print(f"Wrote {divergence_path}")
    print("\n=== Plan-divergence summary ===")
    print(f"{'budget':>8}  {'BOPs vs JOINT':>15}  {'n_diff':>7}  {'JOINT vs STAGED':>17}  {'n_diff':>7}")
    for r in rows_hw:
        print(f"{r['budget_fraction']:>7.1%}  {str(r['plans_identical']):>15}  {r['n_blocks_differing']:>7}  "
              f"{str(r['joint_staged_identical']):>17}  {r['n_blocks_differing_joint_staged']:>7}")
    n_joint_staged_collapsed = sum(1 for r in rows_hw if r["joint_staged_identical"])
    # CORRECTED framing on review, 2026-07 (do NOT report this as "staging collapse
    # persists" -- that was this script's own first-pass mischaracterization). At any budget
    # where tau_b >= latency_id (the budget is looser than doing nothing), the identity plan
    # (D_pred=0, cost=latency_id <= tau_b) is trivially feasible and trivially optimal --
    # EVERY selection rule (BOPs-mapped, JOINT, STAGED) converges on it or something very close
    # for the same reason, not because staging genuinely ties JOINT on a non-trivial joint
    # optimization. That is a DEGENERATE coincidence, not a re-confirmation of G1/E5's real
    # finding (which was a non-trivial tie at BOPs-scale budgets where real compression was
    # actually being chosen). See "Joint vs staged" section of jetson/README.md for the planned
    # follow-up sweep (budgets set directly inside the achievable range) that actually forces
    # compression and makes this a live, non-degenerate test.
    n_degenerate = sum(1 for r in rows_hw if r["tau_ms"] >= latency_id - 1e-9)
    print(f"\nJOINT vs STAGED landed on the identical plan at {n_joint_staged_collapsed}/{len(rows_hw)} "
          f"budgets under measured latency ({n_degenerate}/{len(rows_hw)} of THESE budgets have "
          "tau_b >= Latency_id, i.e. looser than doing nothing -- the identity plan is trivially "
          "optimal there, so this is NOT a re-test of G1/E5's \"staging collapse\" finding, just a "
          "degenerate coincidence). Joint-vs-staged remains genuinely UNTESTED under real hardware "
          "cost until a budget sweep set INSIDE the achievable range forces real compression.")

    # --- STANDING FINDING: "BOPs plan believed X% cost, measured Y% latency" (requested,
    # 2026-07) -- the proxy-divergence result, stated positively: at tight budgets,
    # BOPs-guided compression believed it was buying a large cost reduction; priced in real
    # measured milliseconds, it bought far less (or nothing).
    print("\n=== Proxy divergence: BOPs plan's BELIEVED cost fraction vs its MEASURED latency fraction ===")
    print(f"{'nominal (believed)':>19}  {'measured (tau_b / Latency_id)':>30}")
    for budget_frac, tau_ms in zip(BUDGET_FRACTIONS, taus_ms):
        print(f"{budget_frac:>18.1%}  {tau_ms / latency_id:>29.1%}")

    manifest = {
        "model_name": MODEL_NAME, "latency_id_ms": latency_id,
        "min_achievable_latency_ms": min_achievable_ms,
        "min_achievable_fraction_of_identity": min_achievable_frac,
        "budget_fractions": list(BUDGET_FRACTIONS), "taus_ms": taus_ms,
        "budget_pairing_note": (
            "tau_b = measured latency of P2.1's own BOPs-selected plan at budget b (NOT "
            "fraction x Latency_id -- that pairing made 5/6 budgets mathematically infeasible, "
            "a confound from BOPs cost's much wider achievable dynamic range than real "
            "latency's, not a hardware-reality finding -- see README.md's Budget pairing "
            "section). Guarantees feasibility by construction."
        ),
        "n_schedules": len(schedules),
        "bits_to_precision_mapping": {"16": "fp16", "8": "int8", "6": "int8"},
        "bits_to_precision_note": (
            "bits=8 and bits=6 both execute at measured INT8 engine latency (BEST ACHIEVABLE, "
            "explicit Q/DQ -- see jetson/README.md's INT8 findings) -- there is no distinct "
            "hardware datapath per bit count on this device. See this file's module docstring."
        ),
        "token_buckets": list(TOKEN_BUCKETS),
        "n_joint_staged_collapsed": n_joint_staged_collapsed,
        "n_degenerate_budgets": n_degenerate,
        "joint_staged_note": (
            "CORRECTED on review, 2026-07 -- do NOT read n_joint_staged_collapsed as a "
            "re-test of experiments/phase1_E1-E5_E4a/E5_allocator_G1.py's 'staging collapse' finding. At this run's "
            "6 budgets (paired to P2.1's own BOPs-selected plans' measured latency -- see "
            "'Budget pairing'), tau_b >= Latency_id at n_degenerate_budgets of them -- looser "
            "than doing nothing, so the identity plan is trivially optimal for every selection "
            "rule there, a degenerate coincidence, not a genuine joint-vs-staged test. Joint vs "
            "staged under real hardware cost remains UNTESTED until a budget sweep set directly "
            "INSIDE the achievable range (min_achievable_fraction_of_identity above 1.0, i.e. "
            "compression genuinely forced) is run -- see jetson/README.md's 'Joint vs staged' "
            "section for the planned follow-up."
        ),
    }
    write_json(out_dir / "replan_manifest.json", manifest)
    print(f"\nWrote {out_dir / 'replan_manifest.json'}")


if __name__ == "__main__":
    main()
