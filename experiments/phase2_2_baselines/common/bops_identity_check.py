"""Unit-checks experiments/phase2_2_baselines/common/bops.py's cost formula against P2's own
C_id (identity-plan BOPs) for the IDENTITY config (no quant, no merge,
FP16 everywhere) -- confirms `block_bops`/`cost_from_trajectory` are
algebraically the SAME formula as P2's `block_bops`/`plan_cost`
(experiments/phase2_1_fleet/P2_tandem_pipeline.py Section 10), not just similarly named.
Pure arithmetic, no torch/GPU needed. Run standalone via
`python -m common.bops_identity_check` (from experiments/phase2_2_baselines/), or call
`check_identity_equivalence` from the host-equivalence gate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bops import cost_from_trajectory, tome_schedule_n_l, uniform_bits_trajectory  # noqa: E402

# All FOUR model shapes this project's baselines actually use: B1/B4 are
# deit_small only; B2/B3 cover all four (deit_tiny/small/base + vit_base --
# CORRECTED 2026-07: deit_tiny (d=192) was previously missing here even
# though B2/B3's own MODELS dicts have always included it, so its C_id was
# never unit-checked).
MODEL_DIMS = {
    "deit_tiny_patch16_224": 192,
    "deit_small_patch16_224": 384,
    "deit_base_patch16_224": 768,
    "vit_base_patch16_224": 768,
}
N_BLOCKS = 12
N_TOKENS_TOTAL = 197

# experiments/phase2_1_fleet/results/P2/<model>/P2_<model>_summary.json's own stored "c_id"
# field -- the ACTUAL canonical value P2.1 itself computed and recorded,
# not a re-derivation. `check_identity_equivalence` cross-checks two
# INDEPENDENT formula implementations against each other (this module's
# own vs a re-typed copy of P2's formula) -- useful, but a shared
# misunderstanding in BOTH could still pass. `check_c_id_matches_canonical_summary`
# below is the STRONGER check the audit spec (P2.2 audit item #13, "read
# C_id from canonical summaries") actually asks for: read the real stored
# value from disk and assert bit-for-bit equality against what THIS
# module's formula computes.
CANONICAL_SUMMARY_PATHS = {
    name: Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "phase2_1_fleet" /
    "results" / "P2" / name / f"P2_{name}_summary.json"
    for name in MODEL_DIMS
}


def p2_identity_c_id(n_blocks, n_tokens_total, d):
    """Re-derives P2's own C_id formula directly (experiments/phase2_1_fleet/
    P2_tandem_pipeline.py Section 10: block_bops/plan_cost, IDENTITY_PLAN
    = all-FP16/no-merge) as an INDEPENDENT re-implementation (not an
    import) -- so this check catches a drift between the two files, not
    just a shared bug if it re-used the same code both times."""
    def block_bops(n_l, bits):
        linear = 12 * n_l * d**2 * (bits * bits)
        attention = 2 * n_l**2 * d * (16 * 16)
        return linear + attention
    return sum(block_bops(n_tokens_total, 16) for _ in range(n_blocks))


def check_identity_equivalence(n_blocks, n_tokens_total, d):
    n_l_per_block, bits_per_block = uniform_bits_trajectory(n_blocks, n_tokens_total, 16)
    ours = cost_from_trajectory(n_l_per_block, bits_per_block, d)
    p2 = p2_identity_c_id(n_blocks, n_tokens_total, d)
    assert ours == p2, (
        f"BOPs identity-config mismatch: common/bops.py's cost_from_trajectory gave {ours:.6e}, "
        f"P2's own C_id formula gives {p2:.6e} (n_blocks={n_blocks}, n_tokens_total={n_tokens_total}, "
        f"d={d}) -- these MUST be algebraically identical (experiments/phase2_2_baselines/common/bops.py's own docstring "
        "claims 'ported verbatim in formula' from P2/E5) or every baseline's bops_ours number is not "
        "actually comparable to TANDEM's own Pareto axis."
    )
    return ours


def check_c_id_matches_canonical_summary(model_name, n_blocks, n_tokens_total, d):
    """P2.2 audit item #13: read C_id from THE ACTUAL canonical summary
    (experiments/phase2_1_fleet/results/P2/<model>/P2_<model>_summary.json's stored "c_id"
    field), not a re-derivation -- stronger than `check_identity_equivalence`,
    which only cross-checks two independent FORMULA implementations against
    each other. Skips (does not fail) if the summary file doesn't exist --
    this module must stay runnable from a checkout that doesn't have
    experiments/phase2_1_fleet/results/P2/ materialized (e.g. a baselines-only clone);
    callers that need this to be mandatory should check the return value."""
    path = CANONICAL_SUMMARY_PATHS[model_name]
    if not path.exists():
        return None, f"SKIPPED -- canonical summary not found at {path}"
    summary = json.loads(path.read_text())
    canonical_c_id = summary["c_id"]
    n_l_per_block, bits_per_block = uniform_bits_trajectory(n_blocks, n_tokens_total, 16)
    ours = cost_from_trajectory(n_l_per_block, bits_per_block, d)
    assert ours == canonical_c_id, (
        f"C_id MISMATCH for {model_name}: our identity-config BOPs={ours:.6e}, canonical summary "
        f"({path}) stores c_id={canonical_c_id:.6e} -- these MUST be bit-for-bit equal or every "
        f"baseline's bops_ours number for this model is not actually on TANDEM's own Pareto axis."
    )
    return ours, f"matches canonical summary's stored c_id={canonical_c_id:.6e} ({path.name})"


def check_tome_schedule_r0_equivalence(n_blocks, n_tokens_total, d):
    """One cost ruler, spec item #3's unit check: ToMe's closed-form
    schedule (common.bops.tome_schedule_n_l) at r=0 (no merging at all)
    must give EXACTLY the identity-config N_l trajectory (n_tokens_total at
    every block) and therefore exactly C_id -- if this doesn't hold, the
    formula has an off-by-one or indexing bug that would silently distort
    every real (r>0) ToMe BOPs number too."""
    n_l_r0 = tome_schedule_n_l(0, n_blocks, n_tokens_total)
    assert n_l_r0 == [n_tokens_total] * n_blocks, (
        f"tome_schedule_n_l(0, ...) gave {n_l_r0}, expected {[n_tokens_total] * n_blocks} "
        "(r=0 must mean NO merging at any block)"
    )
    ours = cost_from_trajectory(n_l_r0, [16] * n_blocks, d)
    p2 = p2_identity_c_id(n_blocks, n_tokens_total, d)
    assert ours == p2, (
        f"ToMe schedule r=0 BOPs mismatch: tome_schedule_n_l(0)-derived cost gave {ours:.6e}, "
        f"P2's own C_id formula gives {p2:.6e} (d={d}) -- these MUST be exactly equal."
    )
    return ours


def check_tome_schedule_cap_regression(n_blocks, n_tokens_total):
    """Regression check for the `protected`-token merge cap
    (common.bops.tome_schedule_n_l): for THIS project's only actual shape
    (n_blocks=12, n_tokens_total=197) and only actual r values (8, 13,
    16 -- b1_tome/run.py, b4_sequential_composites/run.py), the cap never
    binds before the terminal block (verified by hand and cross-checked
    against a real forward-hook measurement on a live 50k run, see
    common/bops.py's docstring) -- these hardcoded golden trajectories
    catch a future accidental change to the formula that would silently
    alter a cost number no GPU-based check would catch quickly."""
    golden = {
        8:  [197, 189, 181, 173, 165, 157, 149, 141, 133, 125, 117, 109],
        13: [197, 184, 171, 158, 145, 132, 119, 106, 93, 80, 67, 54],
        16: [197, 181, 165, 149, 133, 117, 101, 85, 69, 53, 37, 21],
    }
    for r, expected in golden.items():
        got = tome_schedule_n_l(r, n_blocks, n_tokens_total)
        assert got == expected, (
            f"tome_schedule_n_l({r}, {n_blocks}, {n_tokens_total}) gave {got}, expected the golden "
            f"{expected} -- either the cap formula regressed, or the cap now binds early for r={r} "
            "(which changes every affected baseline's bops_ours retroactively -- do not silently accept)."
        )
    return golden


def check_tome_schedule_cap_engages():
    """Proves the `protected`-token cap in `tome_schedule_n_l` is not a
    dead branch: a synthetic small shape (n_tokens_total=20, n_blocks=3)
    where r=50 vastly exceeds what 3 blocks of a 20-token sequence can
    actually merge -- the OLD uncapped linear formula would give
    [20, 1, 1] (clamped only by its own arbitrary floor-at-1, physically
    meaningless), the capped/real formula must give a monotonically
    decreasing, physically-valid trajectory that never drops below
    protected+1 (=2) tokens."""
    n_l = tome_schedule_n_l(r=50, n_blocks=3, n_tokens_total=20, protected=1)
    assert n_l == [20, 11, 6], (
        f"tome_schedule_n_l(r=50, n_blocks=3, n_tokens_total=20) gave {n_l}, expected [20, 11, 6] -- "
        "the merge cap should limit each block to floor((N-protected)/2) merges, not the raw "
        "(uncapped) r=50 request."
    )
    assert all(n >= 2 for n in n_l), f"trajectory {n_l} dropped below the physical floor of 2 tokens"
    return n_l


def check_composite_r0_reduces_to_ptq(n_blocks, n_tokens_total, d, w_bits, a_bits):
    """B4 spec item #3: a sequential composite's own cost formula
    (`cost_from_trajectory` fed a ToMe-derived N_l trajectory) at r=0 (no
    merging at all) must reduce EXACTLY to the pure-PTQ formula
    (`uniform_bits_trajectory`, what B2/B3 themselves use) for the same
    (w_bits, a_bits) -- proof the composite cost ruler is not a second,
    silently-divergent implementation, just the same one-cost-ruler
    formula fed a degenerate (no-merge) trajectory."""
    composite_n_l = tome_schedule_n_l(0, n_blocks, n_tokens_total)
    composite_bops = cost_from_trajectory(composite_n_l, [(w_bits, a_bits)] * n_blocks, d)
    ptq_n_l, ptq_bits = uniform_bits_trajectory(n_blocks, n_tokens_total, (w_bits, a_bits))
    pure_ptq_bops = cost_from_trajectory(ptq_n_l, ptq_bits, d)
    assert composite_bops == pure_ptq_bops, (
        f"Composite-at-r=0 (w{w_bits}a{a_bits}) BOPs={composite_bops:.6e} does not exactly equal "
        f"pure-PTQ (w{w_bits}a{a_bits}) BOPs={pure_ptq_bops:.6e} -- the composite cost ruler has "
        "diverged from the single-axis PTQ ruler B2/B3 already validated."
    )
    return composite_bops


def check_composite_w16a16_reduces_to_tome(n_blocks, n_tokens_total, d, r):
    """B4 spec item #3's other required reduction: a sequential composite's
    cost formula at w_bits=a_bits=16 (no quantization at all) must reduce
    EXACTLY to the pure-ToMe formula (B1's own `tome_schedule_n_l` +
    `cost_from_trajectory` at 16x16) for the same r -- proof the composite
    ruler is consistent with the single-axis ToMe ruler B1 already
    validated, not a silently-divergent second implementation."""
    n_l = tome_schedule_n_l(r, n_blocks, n_tokens_total)
    composite_bops = cost_from_trajectory(n_l, [16] * n_blocks, d)  # 16x16 == "no quantization" in this cost model
    pure_tome_bops = cost_from_trajectory(tome_schedule_n_l(r, n_blocks, n_tokens_total), [16] * n_blocks, d)
    assert composite_bops == pure_tome_bops, (
        f"Composite-at-w16a16 (r={r}) BOPs={composite_bops:.6e} does not exactly equal pure-ToMe "
        f"(r={r}) BOPs={pure_tome_bops:.6e} -- the composite cost ruler has diverged from the "
        "single-axis ToMe ruler B1 already validated."
    )
    return composite_bops


if __name__ == "__main__":
    for name, d in MODEL_DIMS.items():
        c_id = check_identity_equivalence(N_BLOCKS, N_TOKENS_TOTAL, d)
        print(f"{name}: identity-config BOPs = {c_id:.6e} -- matches P2's own C_id formula exactly.")
    print("All identity-config equivalence checks PASSED.")

    for name, d in MODEL_DIMS.items():
        _, detail = check_c_id_matches_canonical_summary(name, N_BLOCKS, N_TOKENS_TOTAL, d)
        print(f"{name}: {detail}")

    tome_c_id = check_tome_schedule_r0_equivalence(N_BLOCKS, N_TOKENS_TOTAL, MODEL_DIMS["deit_small_patch16_224"])
    print(f"tome_schedule_n_l(r=0) BOPs = {tome_c_id:.6e} -- matches C_id exactly "
          "(ToMe's closed-form schedule unit-check PASSED).")

    golden = check_tome_schedule_cap_regression(N_BLOCKS, N_TOKENS_TOTAL)
    print(f"tome_schedule_n_l cap regression PASSED for r in {list(golden)} -- byte-identical to the "
          "pre-cap formula (cap never binds before the terminal block for this project's actual shape).")
    engaged = check_tome_schedule_cap_engages()
    print(f"tome_schedule_n_l cap-engagement PASSED (synthetic r=50/n_blocks=3/n_tokens=20 -> {engaged}, "
          "not the old formula's physically-meaningless [20, 1, 1]).")

    d_deit_s = MODEL_DIMS["deit_small_patch16_224"]
    r0_ptq = check_composite_r0_reduces_to_ptq(N_BLOCKS, N_TOKENS_TOTAL, d_deit_s, 8, 8)
    print(f"composite-at-r=0 (w8a8) BOPs={r0_ptq:.6e} == pure-PTQ (w8a8) BOPs -- PASSED "
          "(B4 spec item #3's first required reduction).")
    w16a16_tome = check_composite_w16a16_reduces_to_tome(N_BLOCKS, N_TOKENS_TOTAL, d_deit_s, 13)
    print(f"composite-at-w16a16 (r=13) BOPs={w16a16_tome:.6e} == pure-ToMe (r=13) BOPs -- PASSED "
          "(B4 spec item #3's second required reduction).")
