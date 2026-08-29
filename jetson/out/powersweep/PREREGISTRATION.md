# Power-mode sweep -- pre-registration

Written BEFORE Part 1 (the MAXN_SUPER reproduction gate) or any lower-mode measurement runs. See
`jetson/README.md`'s "Power-mode sweep" section (to be added) and
`jetson/out/powersweep/modes_available.json` (Part 0, written first) for the real, queried set of
nvpmodel modes this board exposes.

## PRIMARY QUESTION

Does BOPs->latency proxy divergence persist across power modes? Recorded as the proxy error
ratio (measured latency ratio / BOPs ratio) at N=69 relative to N=197, per mode. At MAXN_SUPER
this is 1.84x.

## SECONDARY QUESTION (precision axis)

Does INT8 become faster than FP16 at reduced clocks?

DECISION RULE: INT8 faster than FP16 by MORE than 5% at >=2 of the 5 buckets, at ANY mode ->
NEEDS MANUAL REVIEW, do not proceed further; the precision axis may be live in that regime.
Otherwise -> axis domination is confirmed across operating points and this closes the question
permanently.

## REPRODUCTION GATE

Re-measuring at MAXN_SUPER must reproduce the committed `latency_lut.json` medians within 3%. A
larger deviation indicates drift and HALTS the sweep -- no lower-mode numbers are trusted until
it is explained.

## Context (not part of the pre-registered rules above, recorded for completeness)

Every hardware claim in the thesis to date rests on ONE operating point (MAXN_SUPER). The Orin
Nano Super exposes multiple nvpmodel profiles that change the GPU/EMC clock ratio -- the exact
mechanism behind proxy divergence. Re-measuring at lower power modes converts "one device" into
"one device family, N operating points spanning a real power range" for the cost of an afternoon
of device time. This is also, independently, one more free test of whether the precision axis
revives once the GPU is slower relative to fixed per-kernel overhead (enqueue, launch, memory
latency) -- the regime where INT8 could finally pay off, per the secondary question above.

This is the LAST device experiment for this thesis; device work was previously closed after the
energy replan (see `jetson/out/energy_replan/PREREGISTRATION.md`), and is reopened here only for
this bounded, single-day, cheap generality check on hardware already in hand and engines already
built (no new engines are built for this sweep).
