# G1-HW characterization: why the precision axis is dead on this device, and what that does to the joint-vs-staged question

This document records a structural characterization, not a new experiment.
It exists because the registered G1-HW result (below) and the INT8
findings (`jetson/README.md`) together explain something that would otherwise
look like a repeated null result across three different comparisons
(G1/E5's BOPs-simulated tie, `replan.py`'s degenerate-budget tie,
`paired_eval.py`'s real-hardware tie) -- they are not three independent
confirmations of the same finding. They are one mechanism, observed three
times.

## (a) The registered result: G1-HW's real hardware experiment succeeded

`paired_eval.py`'s pre-registered BOPs-selected-vs-JOINT comparison
**PASSED its gate** on real Jetson hardware:

```
Pooled mean paired delta-top1 (joint - bops): +0.8800pp
95% CI: [+0.7776, +0.9824]pp
Gate (mean >= +0.4pp AND CI excludes 0): PASS
plans_identical: False -- diverged at 6/6 budgets
n_pooled_pairs: 120,000 (6 budgets x 20,000 images)
```

This is the experiment G1-HW set out to run, and it worked: a
latency-aware allocator, selecting under measured on-device cost instead
of a simulated proxy, produced a real, statistically supported accuracy
improvement over the proxy-selected plan. Per the pre-registered
interpretation locked in before this result was known (see
`jetson/README.md`'s "Budget pairing" section and `paired_eval.py`'s own
docstring), the
*mechanism* behind that PASS is that BOPs-selected compression cost more
latency than it saved on this device at every one of the 6 registered
budgets, so the latency-aware planner won by declining to compress
-- a proxy-divergence result, not by itself a demonstration of
allocation quality under a live constraint.

## (b) Derivation: why the precision axis is dead here

The INT8 findings (`jetson/README.md`) are not a single anecdote -- they hold
across the full measured range. From `latency_lut.json` (the LUT
`replan.py`'s DP actually reads, best-achievable explicit Q/DQ INT8 vs
FP16, all 5 token buckets):

| N (tokens) | FP16 (ms) | INT8/QDQ (ms) | INT8/QDQ vs FP16 |
|---|---|---|---|
| 69  | 0.3074 | 0.3228 | 1.05x slower |
| 101 | 0.3113 | 0.3411 | 1.10x slower |
| 133 | 0.3776 | 0.4341 | 1.15x slower |
| 165 | 0.3942 | 0.4616 | 1.17x slower |
| 197 | 0.4139 | 0.5564 | 1.34x slower |

INT8 is slower than FP16 at **every single measured bucket**, even under
the best achievable implementation (explicit Q/DQ via NVIDIA ModelOpt,
all 4 GEMMs correctly fused, zero Reformat-bracketing overhead -- see
`jetson/README.md`'s INT8 findings section for the full diagnostic ladder that
established this is not a sloppy-quantization artifact). There is no
token count in this plan space's range where INT8 buys anything.

The consequence for the allocator's plan space, stated precisely:

`replan.py`'s DP selects over `(schedule, bits)` jointly, where
`bits in BIT_MENU = (16, 8, 6)` maps to hardware precision as
`{16: fp16, 8: int8, 6: int8}` (`replan.py`'s own documented mapping --
there is no distinct 6-bit hardware datapath on this device). Since
`bits<16` **always** costs more latency than `bits=16` at the same token
count (every row of the table above), and also always costs more
predicted accuracy damage (`Dq_table.csv`'s `D_mse` values are
non-negative by construction -- quantizing can only hurt, never help,
the accuracy objective), choosing `bits<16` for any block is
**strictly dominated on both objectives simultaneously** by choosing
`bits=16` for that block, regardless of budget. No Pareto-optimal plan
in `exact_pareto_bits`'s frontier can ever contain a block with
`bits<16` on this device -- not as an empirical tendency, a structural
guarantee, given the table above holds at every bucket. Every
allocator that searches this plan space honestly (the joint DP, and
the staged baseline's own `exact_pareto_bits` calls on the bits axis)
therefore converges on `bits=16` everywhere, and the entire plan space
collapses from `85 schedules x 3^12 bit-assignments` to just the `85`
merge-only schedules, all at FP16.

`JOINT` and `STAGED` searching the *same* 85-schedule, all-16-bit plan
space for the same objective (`D_pred`, minimized subject to the same
`tau_b`) is exactly `STAGED`'s own `S2 (bits-then-tokens)` order at
`alpha=0` (bits stage picks all-16 trivially, since no bit reduction
ever clears the alpha-savings threshold when nothing is saved; tokens
stage then does the identical exhaustive Pareto scan over all 85
schedules that `JOINT` itself does) -- an exact match by construction,
not a coincidence discovered after the fact. **Any budget sweep run
under this LUT, at any tau, is foreordained to produce
`Delta(joint - staged) = 0.0000` exactly.** The follow-up sweep
originally planned to force real compression and make joint-vs-staged a
live test (see `jetson/README.md`'s "Joint vs staged" section) would not do
that -- it would force real *merging*, but merging alone is not what
that test needs, since staging only collapses to joint when BOTH axes
are live. **That sweep is CANCELLED as a joint-vs-staged test** on this
device; see (d) below for what any future zero on this axis means
instead.

## (c) The characterization

Stated as a general condition, not specific to this run:

**Joint-vs-staged separation requires two things simultaneously:**
1. A cost axis that is **not additively separable** in `(schedule, bits)`
   -- i.e., a block's real cost genuinely depends on both axes jointly,
   not as a sum of independent per-axis costs. (Real measured latency
   satisfies this in general -- see `jetson/README.md`'s INT8 findings on how
   Reformat/fusion overhead interacts with token count -- which is why
   G1-HW's BOPs-vs-JOINT test, itself testing "does the cost proxy
   matter," is meaningful and did separate.)
2. **Both compression axes independently profitable** -- there must
   exist at least one block where reducing bits below 16 is worth doing
   on its own terms (better accuracy-per-latency than not), so that a
   staged procedure which commits to one axis first can be led away from
   the joint optimum by that commitment.

Under an **additive** cost proxy (BOPs, `D_pred` additively separable by
construction in `experiments/phase1_E1-E5_E4a/E5_allocator_G1.py`'s own accuracy model),
condition 1 fails, and joint and staged provably coincide whenever the
joint-optimal schedule is also token-frontier-efficient -- this is
exactly G1/E5's original "staging collapse" finding (`plans_identical`
at all 6 production budgets, confirmed there under simulated BOPs cost).

On hardware where the precision axis is **dead** -- bits<16 dominated on
both cost and damage, as derived in (b) -- condition 2 fails
structurally, regardless of whether condition 1 holds. The comparison in
this regime is not a negative result about joint allocation; it is
**structurally VOID** -- there is nothing for staging order to get wrong
when the second axis was never a live choice to begin with. Void is a
different outcome from tied: a tie is a finding about the allocator; void
is a finding about the plan space it was asked to search.

## (d) Pre-registered interpretation of any future joint-vs-staged zero on this device

Any future run on this same LUT/device that produces
`Delta(joint - staged) = 0.0000pp` at any budget must be recorded as:

> **"Void by axis domination, per CHARACTERIZATION.md (b)."**

Never as "joint fails to beat staged," and never re-stated as a
re-confirmation of G1/E5's staging-collapse finding (that finding was
about an additive *proxy*; this is about a *dead axis on this specific
device*, a different and stronger structural condition -- see (c)).
This interpretation is locked in writing here, before any further sweep
on this device's INT8-dead LUT runs, per this project's standing
discipline of registering interpretations before the number is known
(same discipline `paired_eval.py`'s own pre-registered
proxy-divergence sentence and `replan.py`'s corrected
degenerate-budget framing already follow).

The only way to make joint-vs-staged a live, non-void test on this
device is to find a regime where the precision axis actually becomes
profitable -- i.e., falsify the premise in (b) -- which is exactly
what the batch-8 pilot (see `jetson/README.md`'s "Batch-8 INT8 pilot" section)
exists to check.
