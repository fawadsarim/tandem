# Energy measurement (E-LUT) -- pre-registration

Written BEFORE Part 1 (the energy LUT sweep) runs. See `jetson/README.md`'s "Energy measurement
(E-LUT)" section and `jetson/out/energy/energy_instrument.json` for Part 0's instrument
validation (PASSED -- idle 5.28W < N=69 8.16W < N=197 9.72W, monotonic).

## Pre-registered interpretation (verbatim)

Energy per inference is expected to diverge from BOPs MORE than latency does, because static
power is paid per unit TIME regardless of operations performed. A configuration that reduces
operations without proportionally reducing latency saves even less energy than it saves time.
If measured, this is a second, mechanistically independent instance of the thesis's central
claim.

If instead energy tracks latency closely (high R^2), the conclusion is that a single measured
latency LUT suffices as a hardware cost proxy for both objectives. That is a useful, reportable
result and NOT a null.

Instrument caveat: power sampling resolution is far coarser than one inference. All
per-inference energy is derived from sustained-loop average power and throughput, never from
single-shot measurement. Recorded as a scope limitation.
