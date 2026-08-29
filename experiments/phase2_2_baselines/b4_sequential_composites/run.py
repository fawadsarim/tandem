"""B4: sequential composites -- ToMe THEN RepQ-ViT, and RepQ-ViT THEN ToMe,
run in BOTH orders at multiple operating points chosen to land near
TANDEM's own budget-fraction grid, on DeiT-S. This is the STAGED baseline
the thesis's joint-vs-staged claim is measured against -- its construction
must be exactly right (2026-07 ruling; see ../deviations.md #10-11
for the original integration-risk writeup this ruling builds on).

**HIGHEST-RISK RUNNER IN THIS PROJECT -- read this docstring before
trusting any number it produces.** Neither ToMe nor RepQ-ViT is designed
to compose with the other; this is new integration code, not simply
invoking either upstream tool unmodified.

## PTQ composite partner: RepQ-ViT (spec item #1)

RepQ-ViT is the composite partner throughout this baseline -- the
stronger/cleaner PTQ method of B2/B3 (per the documented ruling). If an
FQ-ViT composite family is ever added, it will be a SEPARATE, independently
labeled composite family (a distinct `method` value), never blended into
these rows.

## Single source of truth (spec item #1)

B4 imports B1's ToMe merging machinery (`tome.patch.timm`, ToMe's own
bipartite-matching internals) and B2's RepQ-ViT quantization machinery
(`quant.quant_model`, `quant.set_quant_state`) as MODULES from the pinned
upstream repos -- no forked or re-implemented copy of either algorithm.
The scale-reparameterization block below is NOT a divergent
reimplementation of RepQ-ViT's own logic: it is a necessary REPLICATION of
`test_quant.py`'s own internal orchestration sequence (verified against
their real source at the pinned commit), the same "replicate a monolithic
script's own call sequence" pattern `common/fp_parity_gate.py`'s
`_repq_fp_pass`/`_fqvit_fp_pass` already use elsewhere in this project --
required because RepQ-ViT ships this as inline script logic, not an
importable function. `assert_no_forked_algorithm_logic()` below statically
verifies this file defines no function shadowing ToMe's or RepQ-ViT's own
core algorithm entry points.

## The Attention.forward conflict, and how this runner resolves it

Both repos monkey-patch `timm.models.vision_transformer.Attention.forward`,
for INCOMPATIBLE purposes:
  - ToMe's `patch.timm.apply_patch` reassigns `module.__class__` to
    `ToMeAttention`, whose `forward(self, x, size=None)` returns a TUPLE
    `(x, k.mean(1))` -- the second element is the similarity metric
    `ToMeBlock.forward` needs to decide what to merge.
  - RepQ-ViT's `utils.build_model.build_model()` installs `matmul1`/
    `matmul2` `MatMul` submodules on every `Attention` instance and
    OVERRIDES `.forward` with an instance-level `MethodType` that routes
    Q@K/Attn@V through those submodules (so `quant.quant_model()` can find
    and fake-quantize them) -- this forward returns a single tensor `x`,
    not a tuple.
  Whichever patch is applied SECOND wins, and either ordering breaks the
  OTHER repo's mechanism.

**Resolution used here: skip RepQ-ViT's `build_model()` entirely** (use
plain `timm.create_model` + ToMe's patch instead). `quant.quant_model()`
does not require `MatMul` submodules to exist -- it wraps `nn.Conv2d`/
`nn.Linear`/`MatMul` instances wherever `named_modules()` finds them and
silently wraps nothing for a type with zero instances (verified from
`quant/quant_model.py`'s source at the pinned commit). Consequence: **this
composite quantizes qkv/proj/fc1/fc2 (the Linear layers) exactly as B2's
plain RepQ-ViT does, but leaves the Q@K/Attn@V matmuls unquantized** --
UNLIKE B2, which (via `build_model`) quantizes those too. This is a
genuine methodological difference from B2's own numbers, not an oversight:
it is also the ONLY way to keep ToMe's merge-metric contract intact, and
it happens to make B4 MORE consistent with TANDEM's own cost model
(attention is never quantized in TANDEM's BOPs formula either). Recorded
in `notes` on every B4 results.csv row. This resolution is
ORDER-INDEPENDENT: `quant_model()`'s generic Linear-wrapping works
identically whether applied to an already-ToMe-patched model or a plain
one, and `tome.patch.timm`'s class-level reassignment works identically
whether applied to an already-quant_model()-wrapped model or a plain one
(the two mechanisms operate at different levels of the module hierarchy
and do not interfere in either order) -- this is what makes the
best-over-both-orders requirement below actually implementable.

## THE DEFINING REQUIREMENT -- best over both orders (spec item #2)

A staged pipeline has two orders, and the baseline must be the STRONGER of
the two at each operating point (Kim et al. lesson -- a weak baseline
invalidates the comparison):
  - **merge_first**: ToMe patches the plain model and merges live; RepQ-ViT
    then wraps + calibrates + quantizes ON THE ALREADY-MERGED model (its
    calibration forward pass sees the reduced token count).
  - **quant_first**: RepQ-ViT wraps + calibrates + quantizes the PLAIN
    (full-token) model first, exactly as B2 does; ToMe's patch is then
    applied ON TOP of the already-quantized, already-calibrated model
    (merging is a pure inference-time reduction from that point on).
Both orders are run for EVERY operating point and BOTH are recorded as
`row_kind=reproduction` rows (config includes the order suffix, e.g.
`r17_w8a8_mergefirst` / `r17_w8a8_quantfirst`); a per-operating-point
`row_kind=comparison` row records the max-top1 of the two, the ONLY row
`compare_pareto.py`/B5 is meant to consume. Both per-order measurements
are always in `results.csv` so the max is auditable -- never silently
keep one order.

**Degenerate exception, not a shortcut**: when an operating point's
quantization stage is a no-op (`w_bits == a_bits == 16`) or its merge
stage is a no-op (`r == 0`), the two orders are MATHEMATICALLY IDENTICAL
(composing with a no-op stage in either position produces the same model)
-- `orders_coincide()` detects this and runs the underlying model ONCE
(not to cut a corner, but because a second GPU pass would produce a
bit-identical number), still writing BOTH per-order rows (each noting the
degeneracy explicitly) plus the comparison row. This mirrors this
project's existing "Step A == Step B, don't run twice" pattern
(`common/checkpoint_pin.py`'s `coincides_with_pinned`) -- same principle,
applied to order-identity instead of checkpoint-identity.

## Cost ruler (spec item #3) -- one x-axis, unchanged convention

BOPs = TANDEM's own convention (`common/bops.py`, ported verbatim from
P2/E5): `12*N_l*d^2*(bw*ba) + 2*N_l^2*d*(16*16)`, attention matmuls always
FP16 (16x16) regardless of what a method does internally. For a composite,
this is ORDER-INDEPENDENT by construction: N_l trajectory comes from the
ToMe stage's r alone (`common.bops.tome_schedule_n_l`, a pure function of
r, not of construction order), and (bw, ba) per block come from the PTQ
stage's (w_bits, a_bits) alone -- the formula only sees the FINAL token
count and FINAL bit-width per block, never the sequence that produced
them. `print_cost_ruler_reductions()` below runs and prints, BEFORE any
GPU work, the three required unit checks: identity (r=0, 16x16) == C_id
(read live from the canonical P2.1 summary, not hardcoded);
composite-at-r=0 == pure-PTQ BOPs; composite-at-w16a16 == pure-ToMe BOPs
(both via `common/bops_identity_check.py`'s new
`check_composite_r0_reduces_to_ptq`/`check_composite_w16a16_reduces_to_tome`
-- shared infrastructure, not B4-local, since the same proof applies to
any future composite baseline). A live forward-hook measurement
(`measure_token_trajectory`) cross-checks the closed-form N_l against
BOTH orders' actual resulting models for every real run, exactly as B1
already does for its own single-axis trajectory.

## Operating-point grid (spec item #4)

Computed by grid search (not guessed): for DeiT-S, over r in ToMe's
already-exercised range and (w_bits, a_bits) in B2/B3's already-validated
{(16,16)="off", (8,8), (6,6)} -- deliberately NOT widened to untested
bit-widths (e.g. asymmetric w8a16) -- find the (r, bits) combo whose BOPs
lands closest to each of TANDEM's own 6 canonical budget fractions
(`experiments/phase2_1_fleet/results/P2/deit_small_patch16_224/P2_deit_small_patch16_224_summary.json`'s
own `budget_fractions`: 15/21/29.3/41/57.2/80%, read live at startup and
asserted equal to `BUDGET_FRACTIONS` below, never hardcoded independently).
Because 8-bit quantization alone already drops BOPs to ~31% of C_id, there
is a real, structural gap between ~31% and 100% that ONLY unquantized
(16-bit / "off") merge ratios can fill within this validated bit-width
set -- consequently 3 of the 6 chosen points end up using w16a16 (pure
ToMe, no quantization at all). This is reported here explicitly, not
hidden: it is the honest consequence of using only already-validated bit
widths, not a flaw in the search.

| Budget target | Chosen (r, w, a) | Achieved BOPs frac | Residual |
|---|---|---|---|
| 15.0%  | r=17, w8a8   | 15.01% | 0.09% |
| 21.0%  | r=10, w8a8   | 20.94% | 0.31% |
| 29.3%  | r=2,  w8a8   | 28.78% | 1.79% |
| 41.0%  | r=22, w16a16 | 41.47% | 1.14% |
| 57.2%  | r=15, w16a16 | 56.75% | 0.79% |
| 80.0%  | r=7,  w16a16 | 79.34% | 0.83% |

r values above ToMe's previously-exercised {8,13,16} (17, 22) are a
further point on the SAME closed-form, already-general schedule formula
(`tome_schedule_n_l` is not fit to a specific r range) -- not a new
mechanism, just new data points. Note `tome_schedule_n_l`'s own docstring
claim that the late-layer merge cap "never binds before block 12" is
specific to r in {8,13,16}; for r=22 the cap DOES bind well before block
12 (see the live trajectory printed at runtime) -- this project's first
real exercise of that code path, which is why it is asserted against a
live forward-hook measurement here, not just trusted.

## Gates / data (spec item #5)

FP-parity (plain, unmodified DeiT-S vs P2.1's own S8 reference, 79.848%,
<=0.15pp) runs ONCE before any composite, cached, HALTS on failure --
unchanged from the original runner. Val-manifest count/source asserted
against the canonical P2.1 summary (`assert_val_set_matches_canonical_p2`).
Calibration is RepQ-ViT's own protocol (single fixed batch, their own
`build_dataset`), same for every operating point and both orders --
recorded via the same `_calib_indices`-style path this project already
uses elsewhere is NOT applicable here (see "Preprocessing note" below;
B4 calibrates via RepQ's own build_dataset train_loader directly
in-process, not a subprocess replaying a captured index sequence).
Two-step checkpoint policy: DeiT-S's Step A/B checkpoints coincide
(`coincides_with_pinned` asserted at import time) -- one measurement, two
`checkpoint_tag`-labeled semantics, never a double GPU pass on that axis
(orthogonal to the order axis above, which DOES always attempt both, per
item #2). ViT-B (whose Step A/B checkpoints do NOT coincide, requiring the
same `checkpoint_pin` override B3 validated) is explicitly OUT OF SCOPE
for this pass -- DeiT-S only, per the "paste DeiT-S first"
instruction; extending to ViT-B is a real, separate follow-up (its own
operating-point grid computed the same way, its own FP-parity/override
wiring), not attempted here.

## Preprocessing note

RepQ-ViT's own `utils.build_dataset.build_dataset(args)` uses
model-family-specific preprocessing; DeiT (`crop_pct=0.875`) resolves to
`resize(floor(224/0.875)) = resize(256)` + `center_crop(224)` + ImageNet
mean/std -- numerically IDENTICAL to TANDEM's fixed convention used
everywhere else in this project.

Environment: same venv as b1_tome (torch==1.12.1, timm==0.4.12, exact-pinned
-- see requirements.txt).

Determinism: TF32 disabled (common.determinism.disable_tf32) before any
CUDA op, seeded (common.determinism.seed_everything). host/gpu/torch/cuda
resolved and recorded in every results.csv row. The full 50k eval is
per-batch checkpointed (common.eval_utils.evaluate_top1_resumable) so a
Colab disconnect resumes rather than restarts.

## Self-test + completeness (spec item #6)

The same 10-item post-conversion self-test B1-B3 run (unchanged, already
wired in the Colab notebook). `PLAN` below is the declared per-model plan
the completeness cell derives its expected row count from -- NOT a
hardcoded literal -- see its own docstring.

## Per-operating-point invocation and resume (Colab session-limit hardening)

`--only-budget-fraction` restricts a single invocation to ONE operating
point (both orders + the best-of-both row), so a Colab cell can run one
operating point at a time -- a crash or session limit partway through
never costs an already-completed point. Omit the flag (university-GPU
default) to run the full grid in one process. Resume: a completed
operating point's `merge_first` reproduction row already in results.csv
means the whole operating point (both orders + comparison row) was
written together and is skipped -- the same coarse per-config resume
mechanism B1-B3 use.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bops import cost_from_trajectory, measure_token_trajectory, tome_schedule_n_l  # noqa: E402
from common.bops_identity_check import (  # noqa: E402
    check_c_id_matches_canonical_summary, check_composite_r0_reduces_to_ptq,
    check_composite_w16a16_reduces_to_tome,
)
from common.checkpoint_pin import coincides_with_pinned, PAPER_CHECKPOINT_TAG  # noqa: E402
from common.determinism import config_hash, disable_tf32, resolved_host_info, seed_everything, SEED  # noqa: E402
from common import equivalence_gate  # noqa: E402
from common.eval_utils import evaluate_top1_resumable  # noqa: E402
from common.fp_parity_gate import fp_parity_check_inprocess  # noqa: E402
from common.repo_utils import ensure_repo  # noqa: E402
from common.results_io import append_result, find_existing_row  # noqa: E402
from common.startup_checks import (  # noqa: E402
    assert_dual_host_audit_items_present, assert_val_set_matches_canonical_p2, assert_valid_commit_sha,
)

TOME_COMMIT = "af95e4b1befa172dadccd8c81e223b10090f9579"
REPQ_COMMIT = "a160561b9233945e6680132f129d13041b787318"
MODEL_NAME = "deit_small_patch16_224"
D_MODEL = 384
N_BLOCKS = 12
N_TOKENS_TOTAL = 197
BUDGET_FRACTIONS = [0.15, 0.21, 0.293, 0.41, 0.572, 0.8]  # TANDEM's own canonical grid -- asserted, not assumed

# DeiT-S: Step A/B coincide (fb_in1k) -- see b1_tome/run.py's identical
# assertion and common/checkpoint_pin.py's module docstring.
assert coincides_with_pinned(MODEL_NAME), (
    f"checkpoint_pin.py says {MODEL_NAME}'s Step A/B checkpoints do NOT coincide -- B4 (DeiT-S only) "
    "was written assuming they always do. If this assertion fires, the checkpoint-identity policy "
    "changed and this runner's row-duplication logic needs revisiting, not bypassing."
)

# Operating-point grid, computed against DeiT-S's real BOPs formula and
# TANDEM's own budget_fractions -- see module docstring's grid-search
# table for the full search and residuals. ONE entry per budget fraction.
OPERATING_POINTS = [
    {"budget_fraction": 0.15, "r": 17, "w_bits": 8, "a_bits": 8},
    {"budget_fraction": 0.21, "r": 10, "w_bits": 8, "a_bits": 8},
    {"budget_fraction": 0.293, "r": 2, "w_bits": 8, "a_bits": 8},
    {"budget_fraction": 0.41, "r": 22, "w_bits": 16, "a_bits": 16},
    {"budget_fraction": 0.572, "r": 15, "w_bits": 16, "a_bits": 16},
    {"budget_fraction": 0.8, "r": 7, "w_bits": 16, "a_bits": 16},
]
ORDERS = ["merge_first", "quant_first"]

# Audit points (2026-07 follow-up, item #1): NOT part of the
# primary 6-point comparison grid -- these exist to test whether a
# pathological result at one of OPERATING_POINTS's grid points (r=22 at
# the 41% budget: final-block token count collapses to 4, an 8.1pp cliff
# vs the 57.2% point) is specific to that exact r, or structural to the
# budget. Written as row_kind=reproduction ONLY (both order labels, per
# the usual degenerate-order dedup) -- deliberately NO row_kind=comparison
# row, so compare_pareto.py/B5 (which only reads "comparison" rows) never
# picks these up. Evidence, not a competing answer to any of the 6
# canonical operating points.
AUDIT_POINTS = [
    {"r": 23, "w_bits": 16, "a_bits": 16, "reason": "confirm the 41% budget's r=22 pathology "
                                                      "(final-block token count 4) is structural to "
                                                      "this budget, not specific to r=22"},
]

# Declared per-model plan (spec item #6): the completeness cell derives
# its expected row count from THIS, not a hardcoded literal. Per operating
# point: 2 reproduction rows (one per order, always written even when
# `orders_coincide()` -- see module docstring) + 1 comparison row.
# Audit points are tracked SEPARATELY (not part of expected_results_csv_rows'
# core count) since they are evidence, not part of the canonical 6-point set.
PLAN = {
    "model": MODEL_NAME,
    "n_operating_points": len(OPERATING_POINTS),
    "rows_per_operating_point": len(ORDERS) + 1,  # 2 reproduction + 1 comparison
    "expected_results_csv_rows": len(OPERATING_POINTS) * (len(ORDERS) + 1),
    "fp_parity_checks": 1,  # plain-model gate, once, cached -- not a results.csv row
    "n_audit_points": len(AUDIT_POINTS),
    "audit_rows_per_point": len(ORDERS),  # 2 reproduction rows, no comparison row
    "expected_audit_rows": len(AUDIT_POINTS) * len(ORDERS),
}


def orders_coincide(op):
    """True if merge_first and quant_first are mathematically IDENTICAL
    for this operating point: either the quantization stage is a no-op
    (w_bits == a_bits == 16) or the merge stage is a no-op (r == 0) --
    either way, composing with a no-op stage in either position produces
    the SAME model, so running both orders would be two GPU passes for one
    number, not two independent measurements. See module docstring's
    "best over both orders" section."""
    return (op["w_bits"], op["a_bits"]) == (16, 16) or op["r"] == 0


def config_key_for(op, order):
    return f"r{op['r']}_w{op['w_bits']}a{op['a_bits']}_{order}"


def config_key_best(op):
    return f"r{op['r']}_w{op['w_bits']}a{op['a_bits']}_best"


def assert_no_forked_algorithm_logic():
    """Spec item #1's grep-assert: statically verify this file defines no
    function shadowing ToMe's or RepQ-ViT's own core algorithm entry
    points -- both are imported and called, never redefined. A real
    (not just docstring-claimed) check that this runner stays a thin
    integration layer over the two upstream repos' own code."""
    forbidden_names = {"bipartite_soft_matching", "merge_wavg", "apply_patch", "quant_model",
                        "set_quant_state", "build_model"}
    import ast as _ast
    source = Path(__file__).read_text(encoding="utf-8")
    tree = _ast.parse(source)
    defined_names = {
        node.name for node in _ast.walk(tree)
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
    }
    collision = forbidden_names & defined_names
    assert not collision, (
        f"b4_sequential_composites/run.py locally DEFINES {collision} -- these must be ToMe's/"
        "RepQ-ViT's own imported functions, never a local reimplementation (spec item #1, "
        "single source of truth)."
    )


def print_cost_ruler_reductions(canonical_c_id):
    """Spec item #3/#7: run and print the required unit checks BEFORE any
    GPU work. Pure arithmetic (common.bops_identity_check), no torch/model
    needed -- proves the composite cost ruler is consistent with the two
    single-axis rulers B1-B3 already validated, not a silently-divergent
    third implementation."""
    print("\n" + "=" * 70)
    print("COST-RULER UNIT CHECKS (spec item #3) -- run before any GPU work")
    print("=" * 70)
    c_id = check_c_id_matches_canonical_summary(MODEL_NAME, N_BLOCKS, N_TOKENS_TOTAL, D_MODEL)[0]
    print(f"[1] identity (r=0, 16x16) == C_id: computed={c_id:.6e}  canonical={canonical_c_id:.6e}  MATCH")
    r0_ptq = check_composite_r0_reduces_to_ptq(N_BLOCKS, N_TOKENS_TOTAL, D_MODEL, 8, 8)
    print(f"[2] composite-at-r=0 (w8a8) == pure-PTQ (w8a8): BOPs={r0_ptq:.6e}  MATCH")
    w16_tome = check_composite_w16a16_reduces_to_tome(N_BLOCKS, N_TOKENS_TOTAL, D_MODEL, 13)
    print(f"[3] composite-at-w16a16 (r=13) == pure-ToMe (r=13): BOPs={w16_tome:.6e}  MATCH")
    print("=" * 70 + "\n")


def print_operating_point_grid(canonical_c_id):
    print("\n" + "=" * 70)
    print(f"OPERATING-POINT GRID (spec item #4) -- {MODEL_NAME}, C_id={canonical_c_id:.6e}")
    print("=" * 70)
    for op in OPERATING_POINTS:
        n_l = tome_schedule_n_l(op["r"], N_BLOCKS, N_TOKENS_TOTAL)
        bops = cost_from_trajectory(n_l, [(op["w_bits"], op["a_bits"])] * N_BLOCKS, D_MODEL)
        achieved_frac = bops / canonical_c_id
        residual_pct = abs(bops - op["budget_fraction"] * canonical_c_id) / (op["budget_fraction"] * canonical_c_id) * 100
        degenerate = " [ORDER-DEGENERATE: no-op stage, both orders identical]" if orders_coincide(op) else ""
        print(f"  target={op['budget_fraction']*100:5.1f}%  r={op['r']:2d} w={op['w_bits']:2d} a={op['a_bits']:2d}  "
              f"BOPs={bops:.4e} ({achieved_frac*100:5.2f}%)  residual={residual_pct:.2f}%{degenerate}")
    print("=" * 70 + "\n")


def build_calib_and_val_loaders(repq_build_dataset_fn, imagenet_dir, calib_batchsize=32, val_batchsize=200,
                                 num_workers=min(8, os.cpu_count() or 2)):
    """Reuses RepQ-ViT's OWN build_dataset (see module docstring for why),
    via a minimal argparse.Namespace satisfying its signature. `num_workers`
    capped to the real host core count -- a throughput/memory knob, not a
    scientific setting (shuffle=False, same images/order either way) --
    same OOM-kill safety margin as B2/B3's NUM_WORKERS, see their run.py.

    `repq_build_dataset_fn` is the FUNCTION itself (caller does `from
    utils import build_dataset as repq_build_dataset`) -- RepQ-ViT's own
    `classification/utils/__init__.py` does `from .build_dataset import
    *`, so `utils.build_dataset` (the name) is already the callable, not
    a module with a `.build_dataset` attribute. A prior version of this
    function called `repq_utils_build_dataset.build_dataset(args)`,
    treating the already-imported function as if it were a module --
    this is a real bug that surfaced on this project's first actual GPU
    run of this file (this bug pre-dated this session's rewrite; it was
    never caught because B4 had never been executed before). Fixed and
    verified against RepQ-ViT's real source at the pinned commit."""
    args = SimpleNamespace(
        model="deit_small", dataset=imagenet_dir, calib_batchsize=calib_batchsize,
        val_batchsize=val_batchsize, num_workers=num_workers,
    )
    return repq_build_dataset_fn(args)


def _quantize_and_reparam(model, w_bits, a_bits, calib_data, device, quant_model_fn, set_quant_state_fn):
    """RepQ-ViT's own calibration -> scale-reparameterization -> re-
    calibration sequence, verified against test_quant.py's actual main()
    at the pinned commit (see module docstring's "single source of truth"
    section for why this is a replication, not a reimplementation).
    Operates on whatever `model` currently is -- plain (quant_first order)
    or already-ToMe-patched (merge_first order); the reparam logic below
    navigates `attn.qkv`/`mlp.fc1` submodule paths that exist identically
    either way, since ToMe's patch only touches Attention's class/forward,
    never the Linear submodule structure quant_model() wraps."""
    wq_params = {"n_bits": w_bits, "channel_wise": True}
    aq_params = {"n_bits": a_bits, "channel_wise": False}
    q_model = quant_model_fn(model, input_quant_params=aq_params, weight_quant_params=wq_params).to(device)
    q_model.eval()

    set_quant_state_fn(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data)

    with torch.no_grad():
        module_dict = {}
        for name, module in q_model.blocks.named_modules():
            module_dict[name] = module
            idx = name.rfind(".")
            idx = 0 if idx == -1 else idx
            father_name = name[:idx]
            assert father_name in module_dict, f"father module {father_name} not found"
            father_module = module_dict[father_name]

            if "norm1" in name or "norm2" in name:
                next_module = father_module.attn.qkv if "norm1" in name else father_module.mlp.fc1
                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta
                r_scale = act_delta / target_delta
                b = act_min / r_scale - target_min

                module.weight.data = module.weight.data / r_scale
                module.bias.data = module.bias.data / r_scale - b
                next_module.weight.data = next_module.weight.data * r_scale
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(
                        next_module.weight.data, b.reshape(-1, 1)
                    ).reshape(-1)
                else:
                    next_module.bias = torch.nn.Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1, 1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False

    set_quant_state_fn(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data)
    return q_model


def build_composite(order, r, w_bits, a_bits, calib_data, device, quant_model_fn, set_quant_state_fn, tome_module):
    """Builds ONE composite model for `order` in {"merge_first",
    "quant_first"} -- see module docstring's "best over both orders"
    section for the full semantics of each order. Callers with a
    `orders_coincide()`-true operating point should only call this once
    (either order label) and reuse the result for both rows -- this
    function itself does not skip the degenerate case, that's the
    caller's job (keeps this function's contract simple: always builds
    exactly what `order` says)."""
    model = timm.create_model(MODEL_NAME, pretrained=True).to(device)
    no_quant = (w_bits, a_bits) == (16, 16)

    if order == "merge_first":
        tome_module.patch.timm(model, prop_attn=True)
        model.r = r
        if no_quant:
            return model
        return _quantize_and_reparam(model, w_bits, a_bits, calib_data, device, quant_model_fn, set_quant_state_fn)
    elif order == "quant_first":
        if no_quant:
            tome_module.patch.timm(model, prop_attn=True)
            model.r = r
            return model
        q_model = _quantize_and_reparam(model, w_bits, a_bits, calib_data, device, quant_model_fn, set_quant_state_fn)
        tome_module.patch.timm(q_model, prop_attn=True)
        q_model.r = r
        return q_model
    else:
        raise ValueError(f"unknown order {order!r}, expected 'merge_first' or 'quant_first'")


def evaluate_one_order(order, op, calib_data, val_loader, device, quant_model_fn, set_quant_state_fn, tome_module,
                        checkpoint_dir, this_config_hash, desc):
    model = build_composite(order, op["r"], op["w_bits"], op["a_bits"], calib_data, device,
                             quant_model_fn, set_quant_state_fn, tome_module)

    n_l_formula = tome_schedule_n_l(op["r"], N_BLOCKS, N_TOKENS_TOTAL)
    n_l_measured = measure_token_trajectory(model, calib_data[:2], device=device)
    assert n_l_formula == n_l_measured, (
        f"ToMe N_l mismatch for {desc} (r={op['r']}): closed-form schedule gives {n_l_formula}, live "
        f"forward-hook measurement gives {n_l_measured} -- these should be IDENTICAL regardless of "
        "order or quantization (quantization changes VALUES, never token COUNTS). Halting rather than "
        "reporting a BOPs number built on an unreconciled disagreement."
    )

    ckpt_path = checkpoint_dir / f"{desc}.json"
    top1, n_total, elapsed_s = evaluate_top1_resumable(
        model, val_loader, ckpt_path, this_config_hash, device=device, desc=desc,
    )
    if os.environ.get("TANDEM_SMOKE_RUN") == "1":
        print(f"[smoke run] full-50k check NOT enforced -- n={n_total}")
    else:
        assert n_total == 50000, f"expected the full 50k val set, got n={n_total}"
    top1_pct = top1 * 100
    print(f"{desc}: top1_50k={top1_pct:.3f}%  n_l_per_block={n_l_formula}  wall_clock_s={elapsed_s:.1f}")
    del model
    return top1_pct, n_total, n_l_formula, elapsed_s


def process_point(op, args, host_info, pip_freeze_hash, checkpoint_tag, calib_data, val_loader, checkpoint_dir,
                   quant_model, set_quant_state, tome, write_comparison_row, extra_notes_prefix=""):
    """Runs both orders for ONE (operating point OR audit point) `op`
    (a dict with r/w_bits/a_bits, and a "budget_fraction" key for real
    operating points -- audit points omit it, see AUDIT_POINTS), writes
    both row_kind=reproduction rows (always -- see module docstring's
    "best over both orders" section), and, only if `write_comparison_row`
    is True, the row_kind=comparison "best" row. Audit points (spec item
    #1 of the 2026-07 follow-up) pass write_comparison_row=False: they
    are evidence for a specific question (is a pathological result
    r-specific or budget-structural), not a competing entry in the
    canonical 6-point set compare_pareto.py/B5 reads."""
    r, w_bits, a_bits = op["r"], op["w_bits"], op["a_bits"]
    budget_frac = op.get("budget_fraction")
    label = f"budget={budget_frac*100:.1f}%" if budget_frac is not None else f"AUDIT POINT (r={r})"
    merge_first_key = config_key_for(op, "merge_first")
    print(f"\n{'=' * 70}\n{label}  r={r}  w{w_bits}a{a_bits}\n{'=' * 70}")

    existing = find_existing_row(
        method="sequential_composite", model=MODEL_NAME, config=merge_first_key,
        row_kind="reproduction", checkpoint_tag=checkpoint_tag,
    )
    if existing is not None:
        print(f"Already in results.csv: sequential_composite/{MODEL_NAME}/{merge_first_key} -- "
              f"top1={existing['top1_50k']}% -- skipping re-run (already recorded).")
        return

    n_l_formula = tome_schedule_n_l(r, N_BLOCKS, N_TOKENS_TOTAL)
    bops_ours = cost_from_trajectory(n_l_formula, [(w_bits, a_bits)] * N_BLOCKS, D_MODEL)
    degenerate = orders_coincide(op)

    results_by_order = {}
    if degenerate:
        reason = "w_bits==a_bits==16 (no quantization)" if (w_bits, a_bits) == (16, 16) else "r==0 (no merging)"
        print(f"[order-degenerate] {reason} -- merge_first and quant_first are mathematically "
              "identical; running ONE GPU pass, recording both order rows from it.")
        desc = f"{merge_first_key}"
        this_hash = config_hash(f"tome={TOME_COMMIT},repq={REPQ_COMMIT}", MODEL_NAME, merge_first_key, SEED)
        top1_pct, n_total, _, elapsed_s = evaluate_one_order(
            "merge_first", op, calib_data, val_loader, args.device, quant_model, set_quant_state, tome,
            checkpoint_dir, this_hash, desc,
        )
        results_by_order["merge_first"] = (top1_pct, n_total, elapsed_s, this_hash)
        results_by_order["quant_first"] = (top1_pct, n_total, elapsed_s, this_hash)
    else:
        for order in ORDERS:
            desc = config_key_for(op, order)
            this_hash = config_hash(f"tome={TOME_COMMIT},repq={REPQ_COMMIT}", MODEL_NAME, desc, SEED)
            top1_pct, n_total, _, elapsed_s = evaluate_one_order(
                order, op, calib_data, val_loader, args.device, quant_model, set_quant_state, tome,
                checkpoint_dir, this_hash, desc,
            )
            results_by_order[order] = (top1_pct, n_total, elapsed_s, this_hash)

    budget_note = f"budget_target={budget_frac*100:.1f}%; " if budget_frac is not None else ""
    base_notes = (
        extra_notes_prefix + budget_note +
        "RepQ-ViT's build_model() attention-MatMul patching "
        "SKIPPED (conflicts with ToMe's forward override -- see run.py docstring), so Q@K/Attn@V "
        "are UNQUANTIZED here (unlike B2's plain RepQ-ViT, which does quantize them) -- only "
        f"qkv/proj/fc1/fc2 Linear layers are quantized. n_l_per_block={n_l_formula} (closed-form, "
        "cross-checked against live hook per order). No published reference exists for this "
        "composite."
    )
    for order in ORDERS:
        top1_pct, n_total, elapsed_s, this_hash = results_by_order[order]
        degenerate_note = (
            f" ORDER-DEGENERATE: this order's model is bit-identical to the other order's "
            f"(reason: {reason}) -- ONE GPU pass, both order rows record the same measurement, "
            "per this project's coincide-don't-double-run policy." if degenerate else ""
        )
        append_result({
            "method": "sequential_composite", "model": MODEL_NAME, "config": config_key_for(op, order),
            "checkpoint_tag": checkpoint_tag, "row_kind": "reproduction",
            "top1_50k": f"{top1_pct:.4f}", "n_images": n_total, "bops_ours": f"{bops_ours:.6e}",
            "repo_commit": f"tome={TOME_COMMIT[:12]},repq={REPQ_COMMIT[:12]}",
            "published_top1_pct": "", "gate_status": "not_applicable", "gate_diff_pct_points": "",
            "calib_images": 32,
            "host": host_info["host"], "gpu_name": host_info["gpu_name"],
            "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
            "config_hash": this_hash, "pip_freeze_hash": pip_freeze_hash,
            "notes": f"order={order}; wall_clock_s={elapsed_s:.1f}. " + base_notes + degenerate_note,
        })

    if not write_comparison_row:
        top1_merge, top1_quant = results_by_order["merge_first"][0], results_by_order["quant_first"][0]
        print(f"[audit point, no comparison row written] r={r}: merge_first={top1_merge:.3f}%  "
              f"quant_first={top1_quant:.3f}%  (evidence rows only -- see notes)")
        return

    top1_merge, top1_quant = results_by_order["merge_first"][0], results_by_order["quant_first"][0]
    best_order = "merge_first" if top1_merge >= top1_quant else "quant_first"
    best_top1 = max(top1_merge, top1_quant)
    best_n_total = results_by_order[best_order][1]
    best_hash = config_hash(f"tome={TOME_COMMIT},repq={REPQ_COMMIT}", MODEL_NAME, config_key_best(op), SEED)
    print(f"[best-of-both] {label}: merge_first={top1_merge:.3f}%  "
          f"quant_first={top1_quant:.3f}%  -> best={best_order} ({best_top1:.3f}%)")
    append_result({
        "method": "sequential_composite", "model": MODEL_NAME, "config": config_key_best(op),
        "checkpoint_tag": checkpoint_tag, "row_kind": "comparison",
        "top1_50k": f"{best_top1:.4f}", "n_images": best_n_total, "bops_ours": f"{bops_ours:.6e}",
        "repo_commit": f"tome={TOME_COMMIT[:12]},repq={REPQ_COMMIT[:12]}",
        "published_top1_pct": "", "gate_status": "not_applicable", "gate_diff_pct_points": "",
        "calib_images": 32,
        "host": host_info["host"], "gpu_name": host_info["gpu_name"],
        "torch_version": host_info["torch_version"], "cuda_version": host_info["cuda_version"],
        "config_hash": best_hash, "pip_freeze_hash": pip_freeze_hash,
        "notes": (f"BEST-OF-BOTH-ORDERS (spec item #2): max(merge_first={top1_merge:.4f}%, "
                  f"quant_first={top1_quant:.4f}%) = {best_order} -- this is the ONLY row "
                  "compare_pareto.py/B5 should read for this operating point; both per-order "
                  "measurements are recorded above as row_kind=reproduction for audit. " + base_notes),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True, help="root with train/ and val/ subdirs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-tag", default=None,
                         help='e.g. "university_gpu" or "colab" -- recorded in results.csv; '
                              "falls back to the node hostname if omitted")
    parser.add_argument("--only-budget-fraction", type=float, default=None,
                         choices=[op["budget_fraction"] for op in OPERATING_POINTS],
                         help="restrict this invocation to a single operating point's budget fraction "
                              "(e.g. one Colab cell per operating point); omit to run the full grid")
    parser.add_argument("--only-audit-r", type=int, default=None, choices=[p["r"] for p in AUDIT_POINTS],
                         help="run ONLY the named audit point (spec item #1, 2026-07 "
                              "follow-up) instead of the primary 6-point grid -- evidence rows only, "
                              "no comparison row, never read by compare_pareto.py/B5")
    parser.add_argument("--skip-fp-parity", action="store_true",
                         help="DANGEROUS -- skips the FP-parity gate entirely. Prints a loud warning "
                              "and writes a _fp_parity_cache/SKIPPED_*.json marker. Default: gate ON. "
                              "Only for fast iteration on a checkpoint already known-good from a "
                              "prior gated run -- never use for a number that will be reported.")
    args = parser.parse_args()
    assert torch.cuda.is_available() or args.device == "cpu", "CUDA not available; pass --device cpu to override"

    assert_no_forked_algorithm_logic()
    assert_dual_host_audit_items_present(items=[
        "pinning", "tf32_off", "equivalence_fingerprint", "unified_schema", "resume",
    ])
    assert_valid_commit_sha(TOME_COMMIT, "TOME_COMMIT")
    assert_valid_commit_sha(REPQ_COMMIT, "REPQ_COMMIT")

    disable_tf32()
    seed_everything(SEED)

    tome_repo = ensure_repo("https://github.com/facebookresearch/ToMe", TOME_COMMIT, local_name="ToMe")
    repq_repo = ensure_repo("https://github.com/zkkli/RepQ-ViT", REPQ_COMMIT, local_name="RepQ-ViT")
    sys.path.insert(0, str(tome_repo))
    sys.path.insert(0, str(repq_repo / "classification"))
    import tome  # noqa: E402
    from quant.quant_model import quant_model, set_quant_state  # noqa: E402
    from utils import build_dataset as repq_build_dataset  # noqa: E402

    host_info = resolved_host_info(args.host_tag)
    print(f"Resolved host: {host_info}")

    pip_freeze_hash, _ = equivalence_gate._pip_freeze_hash(sys.executable)
    print(f"pip-freeze hash: {pip_freeze_hash}")

    checkpoint_tag = PAPER_CHECKPOINT_TAG[MODEL_NAME]
    print(f"Step A/B checkpoint identity: {checkpoint_tag} (Step A == Step B for DeiT -- not run twice)")

    train_loader, val_loader = build_calib_and_val_loaders(repq_build_dataset, args.imagenet_dir)
    calib_data = next(iter(train_loader))[0].to(args.device)

    if os.environ.get("TANDEM_SMOKE_RUN") != "1":
        assert_val_set_matches_canonical_p2(len(val_loader.dataset))

    # Spec item #4: read TANDEM's own budget_fractions LIVE from the
    # canonical P2.1 summary and assert equal to BUDGET_FRACTIONS -- never
    # trust the hardcoded module-level copy alone.
    summary_path = (Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "phase2_1_fleet" /
                     "results" / "P2" / MODEL_NAME / f"P2_{MODEL_NAME}_summary.json")
    canonical_summary = json.loads(summary_path.read_text())
    canonical_fractions = canonical_summary["budget_fractions"]
    assert canonical_fractions == BUDGET_FRACTIONS, (
        f"BUDGET_FRACTIONS={BUDGET_FRACTIONS} does not match the canonical P2.1 summary's own "
        f"budget_fractions={canonical_fractions} ({summary_path}) -- the operating-point grid was "
        "computed against a specific set of budget targets; if P2.1's own grid changed, this "
        "runner's OPERATING_POINTS table needs recomputing, not silently reused."
    )
    canonical_c_id = canonical_summary["c_id"]

    # Spec items #3/#7: print BEFORE any GPU work.
    print_cost_ruler_reductions(canonical_c_id)
    print_operating_point_grid(canonical_c_id)

    # FP-parity gate (unchanged from the original runner): plain,
    # UNMODIFIED model through this same val_loader, once, before any
    # composite runs. Cached; skipped under TANDEM_SMOKE_RUN.
    fp_cache_path = Path(__file__).parent / "_fp_parity_cache" / "sequential_composite_plain_timm.json"
    if args.skip_fp_parity:
        print("=" * 100)
        print("**** WARNING: FP-PARITY GATE SKIPPED (--skip-fp-parity) ****")
        print("The plain/unmodified model's accuracy was NOT verified against its published/pinned")
        print("reference before any composite number below is computed. Every accuracy number")
        print("produced by this invocation is UNVERIFIED against the eval harness's own correctness")
        print("-- do not report any number from this run without re-running with the gate enabled.")
        print("=" * 100)
        skip_marker = (Path(__file__).parent / "_fp_parity_cache" /
                       f"SKIPPED_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        skip_marker.parent.mkdir(parents=True, exist_ok=True)
        skip_marker.write_text(json.dumps({
            "skip_fp_parity": True, "model": MODEL_NAME, "method": "sequential_composite",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
    elif os.environ.get("TANDEM_SMOKE_RUN") == "1":
        print("FP-parity gate SKIPPED (TANDEM_SMOKE_RUN set). NOT cached.")
    elif fp_cache_path.exists():
        print("FP-parity gate already passed (cached) -- skipping re-run.")
    else:
        plain_model = timm.create_model(MODEL_NAME, pretrained=True).to(args.device)
        fp_parity_check_inprocess(plain_model, val_loader, args.device, MODEL_NAME,
                                   reference_top1_pct=79.848, harness_desc="sequential_composite_plain_timm")
        del plain_model
        fp_cache_path.parent.mkdir(parents=True, exist_ok=True)
        fp_cache_path.write_text("{}")

    checkpoint_dir = Path(os.environ.get("TANDEM_CHECKPOINT_DIR_OVERRIDE", str(Path(__file__).parent / "_checkpoints")))

    if args.only_audit_r is not None:
        audit_points = [p for p in AUDIT_POINTS if p["r"] == args.only_audit_r]
        for op in audit_points:
            reason = op["reason"]
            print(f"\n[AUDIT POINT] r={op['r']} w{op['w_bits']}a{op['a_bits']} -- {reason}")
            process_point(op, args, host_info, pip_freeze_hash, checkpoint_tag, calib_data, val_loader,
                          checkpoint_dir, quant_model, set_quant_state, tome, write_comparison_row=False,
                          extra_notes_prefix=f"AUDIT POINT (2026-07 follow-up, item #1): {reason}. ")
        return

    points_to_run = (
        [op for op in OPERATING_POINTS if op["budget_fraction"] == args.only_budget_fraction]
        if args.only_budget_fraction is not None else OPERATING_POINTS
    )
    for op in points_to_run:
        process_point(op, args, host_info, pip_freeze_hash, checkpoint_tag, calib_data, val_loader,
                      checkpoint_dir, quant_model, set_quant_state, tome, write_comparison_row=True)


if __name__ == "__main__":
    main()
