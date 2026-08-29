# DeiT-B INT8 pilot -- pre-registration

Written BEFORE any measurement. See `jetson/README.md`'s "DeiT-B INT8
pilot" section and `jetson/CHARACTERIZATION.md` for full context.

## Hypothesis being tested

Utilization, not precision arithmetic, is what killed INT8 on DeiT-S -- a
4x larger model should reverse it if that diagnosis is right.

## Decision rule (verbatim, registered here before any number is measured)

INT8-QDQ median GPU-compute FASTER than FP16 median at BOTH buckets ->
**GO**: the precision axis is live on DeiT-B; flag for manual review,
to spec a full DeiT-B G1-HW run.

INT8 >= FP16 at EITHER bucket -> **NO-GO**: axis domination confirmed on
a second model scale; `CHARACTERIZATION.md` is updated with the DeiT-B
evidence and device work is CLOSED.

Within +/-5% at either bucket -> **BORDERLINE**: report both numbers
for manual review.

No other model, batch size, or bucket is measured without explicit
sign-off.
