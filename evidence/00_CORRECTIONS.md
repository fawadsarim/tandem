# 00_CORRECTIONS.md — corrections surfaced by the power-mode sweep

These are real numeric corrections to figures that circulated in prior correspondence, found
while assembling the power-mode sweep (`04_jetson_hardware/out/powersweep/`). Recorded here with
their source artifact rather than silently fixed in place, per this package's own "every gap
named honestly" principle — a correction that isn't traceable to its cause is just a new number
to distrust.

**All latency figures in this project are TOTAL per-block medians** (the host-loop-timed,
end-to-end `median_ms` field in every `latency_lut*.json` — includes kernel launch/enqueue
overhead, not just GPU-busy time). GPU-compute-only medians appear separately in raw `trtexec`
bench logs (the `trtexec_crosscheck_median_ms` field, recorded for cross-check only — see
`measure_lut.py`'s own module docstring) and must not be mixed into any ratio, utilization, or
proxy-divergence figure. Both corrections below trace to exactly this mix-up.

---

## a. Proxy error at MAXN_SUPER: 2.235x, not 1.84x

**Committed value**: `04_jetson_hardware/out/energy/proxy_divergence.csv`, row
`n_tokens=69, precision=fp16`, field `latency_divergence_factor` = **2.235170779769783**.

**Corroboration**: the power-mode sweep's fresh MAXN_SUPER remeasurement gives **2.204x** for the
same quantity (`04_jetson_hardware/out/powersweep/proxy_divergence_by_mode.csv`, mode=MAXN_SUPER,
n_tokens=69, precision=fp16, `proxy_error` field) — 1.4% from the committed value, within
ordinary reproduction-gate-level noise, not a second disagreement.

**What went wrong**: the 1.84x figure that appeared in prior correspondence was computed from
`trtexec` GPU-COMPUTE-ONLY medians (the cross-check field), not the TOTAL per-block latency the
LUT and the DP allocator actually consume everywhere else in this project. GPU-compute-only time
excludes fixed per-kernel launch/enqueue overhead — exactly the fixed cost this project's whole
"axis domination" thesis is about — so computing a divergence ratio from it systematically
understates the real divergence. Total-latency is the project standard everywhere else
(`measure_lut.py`, `replan.py`, `proxy_divergence.py` all key off `median_ms`, never
`trtexec_crosscheck_median_ms`); this one figure was the exception.

---

## b. Achieved throughput at N=197/FP16: 1.83 TFLOPS, not ~2.5

**Committed value**: `04_jetson_hardware/out/pilot_deitb/pilot_deitb_utilization.json`,
field `deits_achieved_tflops_fp16` = **1.8284366116106714**.

**Corroboration**: the power-mode sweep's own formula (`powersweep_analysis.py`) computes real
per-block MACs as `756782592.0` — an EXACT match to that same file's own
`deits_flops_per_block` field, computed independently by a different script — then derives
1.6139 / 1.7803 / 1.8209 TFLOPS at 15W / 25W / MAXN_SUPER respectively
(`04_jetson_hardware/out/powersweep/powersweep_analysis_summary.json`,
`achieved_tflops_fp16_n197_by_mode`). The MAXN_SUPER figure (1.8209) matches the committed
1.8284 to within 0.4%.

**What went wrong**: same cause as (a) — the ~2.5 TFLOPS figure used a GPU-compute-only latency
in the denominator instead of the total per-block median, which inflates the derived TFLOPS
(smaller denominator → larger ratio).

---

## c. Board scope caveat: this unit's real clock ceilings are narrower than nvpmodel's nominal per-mode values

Direct evidence, all from `04_jetson_hardware/out/powersweep/modes_available.json` and the raw
`nvpmodel -q --verbose` output captured during the sweep:

- This board's real GPU devfreq table tops out at **624.75 MHz** (confirmed via the GPU devfreq
  `available_frequencies` sysfs table: `306000000 408000000 510000000 612000000 624750000`) —
  below 25W's nominal ceiling (918 MHz) and MAXN_SUPER's uncapped value. 25W and MAXN_SUPER
  therefore lock to the SAME real GPU clock (624.75 MHz); only 15W's own lower cap (612 MHz) is
  genuinely below that ceiling.
- EMC is hardware-capped at **2133 MHz** on all three modes (confirmed via
  `nvpmodel -q --verbose`'s own REAL_VAL vs CONF_VAL mismatch for the EMC PARAM at 25W —
  REAL_VAL stayed 2133000000 against a requested CONF_VAL of 3199000000 — and independently
  confirmed by a direct write test to the kernel cap node, `/sys/kernel/nvpmodel_clk_cap/emc`,
  which accepted the write but did not change the read-back value).
- Effective variation actually exercised across the three "modes" on this unit: CPU frequency
  (1510400 / 1344000 / 1497600 kHz for MAXN_SUPER / 25W / 15W) plus GPU frequency at 15W only
  (612 MHz vs 624.75 MHz elsewhere) — a **12.8%** end-to-end N=197/FP16 latency spread between
  the fastest (MAXN_SUPER, 0.4156ms) and slowest (15W, 0.4689ms) mode measured
  (`04_jetson_hardware/out/powersweep/latency_lut_<mode>.json`).

**Any claim of a "3.5x power range" or "wide clock-ratio sweep" is NOT supported by this data and
must not be written.** The sweep is real and its two pre-registered findings (proxy divergence
persists; precision axis stays dominated) hold across the variation that WAS actually exercised
— but that variation is narrower than nvpmodel's three mode *names* (15W/25W/MAXN_SUPER) imply,
and any write-up should say so explicitly rather than imply a wider sweep than what ran.
