"""TANDEM's own BOPs cost model (E5/P2 `block_bops`/`plan_cost`, ported
verbatim in formula), generalized from a discrete merge-grid PLAN to an
arbitrary MEASURED token trajectory -- because baseline methods (ToMe's
bipartite matching, sequential composites) don't necessarily land on
TANDEM's own {1,3,5,7,9,11}x{32,64,98,128} merge grid, so their true N_l
per block has to be measured empirically (see `measure_token_trajectory`
below), not looked up in a table.

Framework-free (no torch/timm import) so this module works identically
regardless of which baseline-specific venv last touched results.csv --
see ../deviations.md #1.
"""


def block_bops(n_l, bits_w, bits_a, d):
    """C(block) = 12*N_l*d^2*(bw*ba) + 2*N_l^2*d*(16*16).
    Linear term: qkv (3Nd^2) + proj (Nd^2) + fc1 (4Nd^2) + fc2 (4Nd^2) = 12*N*d^2,
    scaled by the bit-product bw*ba. Attention term: QK^T and AV, ALWAYS at
    16x16 -- attention is never quantized anywhere in TANDEM's cost model
    (matches E3/E5/P2's measurement instrument), regardless of what a given
    baseline method actually does to attention internally."""
    linear = 12 * n_l * d**2 * (bits_w * bits_a)
    attention = 2 * n_l**2 * d * (16 * 16)
    return linear + attention


def cost_from_trajectory(n_l_per_block, bits_per_block, d):
    """n_l_per_block: list of token counts ENTERING each block (len =
    n_blocks; index 0 = block 1), matching TANDEM's N_l convention
    (`n_tokens_entering`: a merge at block p reduces what blocks AFTER p
    see, never block p itself). bits_per_block: list of either a single
    int (bw=ba=that value) or an (bw, ba) tuple, one entry per block.
    This is the GENERAL form of E5/P2's `plan_cost` -- their version is the
    special case where n_l_per_block is derived from a discrete
    {block,r}-schedule plan rather than measured directly."""
    assert len(n_l_per_block) == len(bits_per_block), (
        f"n_l_per_block has {len(n_l_per_block)} entries, bits_per_block has {len(bits_per_block)} "
        "-- must be one entry per block"
    )
    total = 0.0
    for n_l, bits in zip(n_l_per_block, bits_per_block):
        bw, ba = bits if isinstance(bits, tuple) else (bits, bits)
        total += block_bops(n_l, bw, ba, d)
    return total


def uniform_bits_trajectory(n_blocks, n_tokens_total, bits):
    """The common case for pure-PTQ baselines (B2, B3): no token reduction
    at all, so N_l = n_tokens_total at every block, and the SAME (bw, ba)
    at every block. Returns (n_l_per_block, bits_per_block) ready for
    `cost_from_trajectory`."""
    n_l_per_block = [n_tokens_total] * n_blocks
    bits_per_block = [bits] * n_blocks
    return n_l_per_block, bits_per_block


def tome_schedule_n_l(r, n_blocks=12, n_tokens_total=197, protected=1):
    """Closed-form N_l for ToMe's ACTUAL per-layer schedule: `parse_r`
    requests a CONSTANT r at every block (verified against ToMe's real
    source at the pinned commit, tome/utils.py's `parse_r` at the default
    inflection=0 -- never overridden anywhere in this project, every
    runner sets `model.r = r`, a plain scalar), but the merge itself
    (tome/merge.py's `bipartite_soft_matching`) additionally clips that
    request to `min(r, (N_l - protected) // 2)` -- it can never merge more
    PAIRS than exist among the unprotected tokens (protected=1: the CLS
    token, never a merge candidate for a DeiT/ViT class-token model, which
    is the only kind this project runs). Implemented here as the real,
    capped, per-block recurrence -- not the uncapped linear special case --
    so a future r/block-count/token-count combination that WOULD hit this
    cap early doesn't silently get a wrong (understated) cost.

    For this project's ONLY actual shape (n_blocks=12, n_tokens_total=197,
    r in {8,13,16} -- b1_tome/run.py and b4_sequential_composites/run.py,
    deit_small_patch16_224 only), the cap is VERIFIED (common/
    bops_identity_check.py's regression check, and independently by
    b1_tome/run.py's own real forward-hook cross-check against a live
    50k run) to never bind before block 12 -- and block 12's own merge
    output feeds no later block, so it never changes N_l for that shape.
    The trajectory this function returns for r in {8,13,16} is therefore
    BYTE-IDENTICAL to the old uncapped linear formula, not a new number to
    recompute -- this function only differs from the old one for inputs
    this project doesn't currently use."""
    n_l = []
    n = n_tokens_total
    for _ in range(n_blocks):
        n_l.append(n)
        actual_r = min(r, max((n - protected) // 2, 0))
        n -= actual_r
    assert len(n_l) == n_blocks
    return n_l


def measure_token_trajectory(model, sample_batch, device="cuda"):
    """Runs ONE forward pass with a forward-PRE-hook on every block,
    capturing the token count ENTERING each block -- this is exactly
    TANDEM's N_l convention (`n_tokens_entering`: the count a block itself
    computes at, before its own reduction if any). Works for any model
    exposing `.blocks` (timm ViT-family: DeiT/ViT), regardless of WHAT
    mechanism (if any) is reducing tokens internally -- this is how B1/B4
    get a real, method-agnostic N_l trajectory from ToMe's bipartite
    merging without needing to reverse-engineer its internal per-layer `r`
    schedule formula (`tome.utils.parse_r`) by hand.

    Import torch lazily -- this module is imported by common/repro_gate.py
    and common/results_io.py too, which must stay torch-free (see module
    docstring)."""
    import torch

    n_l_per_block = []

    def make_hook():
        def hook(module, args):
            n_l_per_block.append(args[0].shape[1])
        return hook

    handles = [blk.register_forward_pre_hook(make_hook()) for blk in model.blocks]
    try:
        model.eval()
        with torch.no_grad():
            model(sample_batch.to(device))
    finally:
        for h in handles:
            h.remove()
    return n_l_per_block
