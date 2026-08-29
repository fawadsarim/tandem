# PREREGISTRATION_INT8.md

Dated 2026-08-26. Written before any FP16/INT8 precision measurement runs on this
device. Scope: MAXN only (per the campaign scope cut — see `jetson-orinnx/PORT_NOTES.md`
and `evidence/06_orinnx_hardware/DEVICE_PROFILE.md`).

## Context

The Orin Nano result was joint IDENTICAL to staged, 18/18 — an exact tie. The
characterization theorem explains it: under an additive cost model the two approaches
are mathematically forced to agree. The theorem also states its own boundary: joint
can only beat staged when the cost model is non-additive AND both compression axes
are profitable. On the Orin Nano, condition (ii) failed — INT8 was slower than FP16
in 15/15 cells, so the precision axis had nothing to offer. The allocator correctly
found nothing to allocate on that axis.

This device has different tensor-core provisioning. It is therefore a LIVE TEST of
the theorem's positive branch.

## Branch A: INT8 faster, and joint BEATS staged

The pre-registered positive confirming instance. The precision axis is profitable,
condition (ii) is satisfied, and the theorem's prediction is borne out. Report the
causal chain explicitly — PB-4 establishes axis profitability, PB-5 shows the
allocator exploiting it. Mechanism, not correlation.

## Branch B: INT8 slower again, and joint TIES staged

The Orin Nano finding generalizes across the Orin family. Condition (ii) fails the
same way for the same reason. The theorem holds and its boundary is confirmed on a
second device. A legitimate and reportable result.

## Branch C: INT8 faster, but joint STILL TIES staged

The most interesting outcome and the one that must not be softened. The theorem as
currently stated has a problem: the profitability condition is insufficient as a
binary. The axis is profitable but not profitable ENOUGH to change the allocation.
This would require characterizing HOW MUCH profitability is needed before joint
allocation diverges from staged — a threshold, not a switch. Report it prominently,
state the theorem's weakness plainly, and record the magnitude of the INT8 advantage
observed so the threshold question can be posed quantitatively.

## Branch D: anything else

If the result fits none of the above, describe it exactly as observed and do not
force it into a branch. State that the pre-registration did not anticipate it.

The interpretation is fixed now, before measurement. Whatever PB-4 and PB-5 produce,
it is reported as a prediction confirmed or refuted — never as a surprise explained
after the fact.
