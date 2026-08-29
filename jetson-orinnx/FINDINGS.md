# FINDINGS.md — Orin NX 8GB vs Orin Nano Super, comparison and results

Dated 2026-08-26. Every accuracy-dependent figure below (`D_pred` values from
`replan.py`) carries the PB-3 FP-parity limitation explicitly: latency measurements
are valid, but numerical equivalence between the deployed TensorRT engines and the
source PyTorch/ONNX model was **not established** on this device (see
`PORT_NOTES.md` §9). Full evidence and reasoning for every finding below is in
`PORT_NOTES.md`; this file states results and their interpretation.

---

## 1. Proxy divergence reproduces across devices — the campaign's strongest result

BOPs-vs-measured-latency divergence at N=69 vs N=197 (FP16):

| Device | Mode | Proxy error |
|---|---|---|
| Orin NX 8GB | MAXN | **2.2582x** |
| Orin Nano Super | MAXN_SUPER | 2.2352x (committed reference) |

Two architecturally different Jetson Orin modules, measured independently, agree to
within 1.0%. This is direct, cross-device confirmation that the BOPs-to-latency
divergence this thesis's whole "axis domination" argument rests on is a real,
device-independent property of the workload's cost structure — not an artifact of
one board.

*(Figure is post-audit: recomputed from the canonical LUT after PB-3 was superseded.
The conclusion — cross-device agreement to within 1% — is rock-solid.)*

## 2. Reproduction gate: 10/10 within 3%, verified on hardware

The official reproduction gate (`out/powersweep/reproduction_gate.csv`) re-measured the
canonical LUT under locked clocks on-device: **10 of 10 cells passed within the 3.0% tolerance**,
with a maximum absolute delta of **1.88%** (at $N=101\text{ INT8}$).

Historical note: earlier intermediate sweeps (such as the initial 9/10 attempt under unverified
clock governor settings, `reproduction_gate_orinnx.csv`) are preserved for full provenance
in `out/lut/archive/` alongside `archive/README.md`.

## 3. INT8 slower than FP16, 5/5 buckets — confirms the Nano's direction on a second architecture, with a different curve shape

*(Recomputed from the canonical LUT: `out/lut/latency_lut.json`.)*

| N | INT8/FP16 ratio | INT8 slower by |
|---|---|---|
| 69 | 1.1234 | 12.3% |
| 101 | 1.1646 | 16.5% |
| 133 | 1.2529 | 25.3% |
| 165 | 1.3204 | 32.0% |
| 197 | 1.4357 | 43.6% |

INT8 is slower than FP16 at every bucket — the same direction the Nano found across
all 15 of its own (bucket x mode) cells. The ratio here is **monotonic**, worsening
steadily as token count rises. This differs in shape from the Nano's own DeiT-B
crossover pilot, where INT8 briefly became faster at a mid-range crossover point
(~N=101) before reverting — a non-monotonic curve. This device's DeiT-S result shows
no such crossover in the range tested: INT8 is uniformly, increasingly disadvantaged
as sequence length grows. Different model (DeiT-S here vs DeiT-B there) and
different device, so this is not a direct contradiction of the crossover finding —
but it is a real difference in curve shape worth carrying forward, not smoothing
into "same result, second device."

## 4. Budgets don't transfer between devices, because the achievable cost ranges are disjoint

Two views of one result, merged here rather than reported separately:

*(Recomputed from the canonical LUT and transfer experiment: `out/transfer/transfer_experiment_result.json`.)*

**View A — the pre-registered budget set doesn't reach this device's range.**
`BUDGET_FRACTIONS = (0.15, 0.21, 0.293, 0.41, 0.572, 0.80)` were calibrated against
the Nano's own achievable range. On this device they produce `tau_b >= Latency_id`
at all 6 budgets — every pre-registered budget sits looser than doing nothing at
all. `replan.py`'s joint-vs-staged comparison degenerates to the identity plan at
6/6 budgets as a direct, mechanical consequence (see finding 6).

**View B — the two devices' achievable cost ranges don't overlap at all.**
| Device | `Latency_id` | Achievable range (ms) | Achievable range (% of own `Latency_id`) |
|---|---|---|---|
| Orin NX 8GB | 3.7202 ms | `[2.8618, 3.7202]` | 76.9%-100% |
| Orin Nano Super | 4.9668 ms | `[3.7958, 4.9668]` | 76.4%-100% |

The *relative* floor is nearly identical (76.9% vs 76.4%) — the shape of the
achievable range transfers. The *absolute* values do not: NX's ceiling (3.7202ms)
sits below Nano's floor (3.7958ms). A budget in NX-milliseconds that is easily
reachable on the NX (tested: 80/85/90/95% of NX's own `Latency_id`) is
**infeasible under the Nano's own table at every one of those four points** — no
plan in Nano's own 85-schedule x 3-bit-menu plan space costs that little, under
Nano's own cost structure. `Latency_id` ratio, NX/Nano = **0.7490** — this device
is roughly 25% faster on this workload at the identity plan, giving the scale of
the difference in one number.

**Consequence, stated precisely**: no cost-gap figure between a Nano-table-selected
plan and an NX-table-selected plan is reported, because no budget exists at which
both tables produce a non-trivial (real, compressed) plan simultaneously. Below
Nano's floor, Nano's table is infeasible; above NX's ceiling, NX's own identity plan
already wins trivially. A workaround budget was considered and deliberately
rejected — any choice would only relocate which side degenerates, producing a
number that looks like a result without being one (full reasoning: `PORT_NOTES.md`
§15).

**This is a categorical result, not a quantitative one, and that is the stronger
outcome**: "the cost table transfers poorly, by X%" invites debate about whether X
matters. "The cost tables have disjoint useful ranges, so a plan selected on one
device cannot be evaluated on the other at any budget" does not — it is direct,
unconstructed, independent evidence for this campaign's own closing claim: the
*measured table* — not any fixed budget or budget fraction derived from it — is the
transferable artifact across devices in this family.

## 5. Label-vs-reality gap in vendor power-mode tooling — a measured device property

Three independent, corroborating observations, all confirmed on a clean SDK-Manager
flash (no patches touching power modes):

1. **Modes requiring a `tpc_pg_mask` transition do not apply the mask.** Confirmed
   directly via `nvpmodel -q --verbose`'s own `REAL_VAL` vs `CONF_VAL` fields (e.g.
   MAXN: `TPC_PG_MASK REAL_VAL: 0` against `CONF_VAL: 240`) — the vendor's own tool
   reports the mismatch when asked.
2. **The mode label updates and persists while the hardware state does not.**
   `nvpmodel -q` correctly names the target mode after every switch; the underlying
   GPU power-gating registers do not reflect it.
3. **An applied state drifts within ~13-20 minutes, with no reboot and no mode
   switch.** A cleanly-verified 20W state (mask, CPU/GPU/EMC/DLA/PVA all matching
   CONF) reverted on its own; a fresh clock re-lock afterward did not restore it.
4. **The mask's actual VALUE is unstable across boots, not just mismatched
   against CONF** (found during the post-campaign verification audit,
   `PORT_NOTES.md` §16). `tpc_pg_mask` read `0` at PB-2's original characterization
   and at PB-3; later, with no code change and no mode switch, it read `15`; after
   a reboot, `0` again. This is a materially stronger version of observation 1 —
   not only does the mask fail to reach its configured target, the value it
   settles on instead is not fixed either. It carries a real, measured throughput
   effect (~10% slower at `15` than at `0`, confirmed by direct re-measurement),
   which observations 1-3 alone did not establish.

Full four-mode sweep, raw-sysfs-vs-CONF, per-mode `nvpmodel.service` outcome:
`PORT_NOTES.md` §6. An official NVIDIA-acknowledged known issue (Jetson Linux 39.2.0
GA Release Notes, issue 6236259) independently corroborates the mask-application
failure specifically for this device's 10W mode (its only non-Fmax-EMC mode) — see
`PORT_NOTES.md`'s EMC-crash citation. This is reported as a measured device
property, not an operational failure — it directly narrowed the campaign's own
comparable power-mode set (finding below).

## 6. Theorem test: Branch D, genuinely untested

PB-4 predicted, before PB-5 ran: condition (ii) (precision axis profitable) is NOT
satisfied on this device → joint should NOT beat staged.

PB-5 observed: joint == staged at 6/6 pre-registered budgets — but every one of
those budgets is degenerate (finding 3, View A): `tau_b >= Latency_id` everywhere,
so the identity plan trivially wins for both joint and staged. The allocator never
faced a real decision.

**Branch D**, not Branch B. `PREREGISTRATION_INT8.md`'s Branch B text ("the theorem
holds and its boundary is confirmed") presumes a *meaningful* tie under real
compression pressure. This tie is not that — it is the same structural degeneracy
documented in finding 3. Forcing this into Branch B would be technically-true-but-
misleading; the achievable-range table catches it immediately. The theorem's
positive/negative branch was **not tested** on this device — not confirmed, not
refuted.

## 7. Scope conditions and limitations

- **Single power mode.** Only MAXN was characterized for measurement (10W/15W/20W
  all show the mask-application gap from finding 4; 10W additionally fails via the
  NVIDIA-documented EMC/`nvpmodel.service` crash). The Nano campaign covered three
  modes (15W/25W/MAXN_SUPER). The only cross-device-comparable power mode is
  MAXN(NX)-to-MAXN_SUPER(Nano).
- **8GB memory ceiling.** Recorded as a first-class scope condition per the
  campaign's own rules; no configuration was silently reduced to fit. No
  NOT-FEASIBLE cells were hit in the LUT actually measured (5 buckets x 2
  precisions, all completed) — the ceiling is a standing condition for any future
  extension of this campaign (larger batch/token configurations), not a defect
  encountered here.
- **FP-parity gate not runnable as `jetson/` defines it** — no non-QDQ ONNX
  reference available (`.gitignore`'d HOST-export artifact). Substitute checks
  (FP16-vs-plain-build on real calibration input; FP16-vs-INT8(QDQ) consistency)
  did not pass (diffs 0.20-0.35), with the comparison harness independently
  verified clean (self-consistency=0.0, identical bindings across engine variants).
  Recorded FAILED, UNRESOLVED — cause undetermined between real engine/quantization
  behavior and a subtler harness issue not caught by the two checks run. Every
  accuracy-dependent figure in this document inherits this limitation.
- **Paired eval and frontier: NOT RUN.** Both require a local ImageNet-1k copy;
  none was available on this device, this host, or anywhere checked during the loan
  window. `jetson/`'s own design runs these as HOST/Colab stages for the same
  reason. Recorded as NOT-RUN, not failed; no substitute attempted (any accuracy
  figure it produced would inherit the FP-parity limitation above regardless). The
  Nano's own registered paired-eval gate (+0.88pp) stands, unsuperseded.
- **`int8_naive_implicit` out of scope**, on two independent grounds: PB-3 asks for
  precisions {16,8} and the "8" `replan.py` consumes is the QDQ path, not naive; and
  it is infeasible for 3 of 5 buckets regardless (calibration `.npy` exists only for
  n197/n69 in the committed tree, on both devices).
- **DLA present, deliberately unused.** Confirmed zero DLA placements across all 10
  built engines (100 layers inspected) — GPU-only, matching `jetson/`'s own scope.
- **Ambient temperature not measured** — no sensor available. SoC-zone temperatures
  (idle and 10-minute sustained-load peak) are real and reported; the load run's
  power-state was not cleanly locked for its full duration (see `DEVICE_PROFILE.md`),
  so it is recorded as "SoC under sustained load, uncontrolled power state," not as
  a clean MAXN or 20W thermal profile.
