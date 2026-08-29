# DEVICE_PROFILE.md — Jetson Orin NX 8GB, P3768 carrier

Dated 2026-08-26, Gate PB-2.

## Module identity — three sources, all agree

- `/proc/device-tree/model`: "NVIDIA Jetson Orin NX Engineering Reference Developer Kit"
- `/etc/nv_boot_control.conf`: module `3767-0001`; device-tree `compatible`:
  `nvidia,p3768-0000+p3767-0001` (carrier p3768-0000 + module p3767-0001)
- Total memory: 7.4 GiB (8GB module, matches expectation)

## Software versions (`out/versions.json`, fresh SDK-Manager flash)

L4T `R39 (release), REVISION: 2.1, GCID: 46758480`; TensorRT `v101602`; CUDA
`release 13.2`; Python `3.12.3`; torch `2.13.0+cu130`; GPU `Orin`.

## CUDA compute — smoke-tested, works despite an unresolved compute-capability warning

torch import emits: "No published PyTorch CUDA builds for release 2.13.0+cu130
support this GPU" (Orin's compute capability 8.7 is not in this torch build's own
listed CC table). Despite the warning, real CUDA compute was verified: a 512×512
FP32 matmul on-device matched a CPU reference to within `9.15e-05` max abs diff; FP16
matmul matched to within `0.051` (expected FP16 rounding). `torch.cuda.get_device_name(0)`
returns "Orin" correctly. Verdict: functional, warning likely a benign PTX-JIT
fallback path, not a blocker for PB-3 — but the warning itself remains unresolved
upstream and is worth re-checking if any torch-CUDA-dependent stage misbehaves later.

## Power-mode finding — MAXN only, four-mode sweep abandoned as infeasible

The originally-planned four-mode sweep (MAXN/10W/15W/20W) was cancelled. Three
independent, measured observations establish that reliable per-mode
characterization is not achievable on this module with this JetPack build:

1. **Modes requiring a `tpc_pg_mask` transition do not apply the mask.** Confirmed
   directly by `nvpmodel -q --verbose` on a clean SDK-Manager flash: e.g. MAXN shows
   `TPC_PG_MASK REAL_VAL: 0` against its own `CONF_VAL: 240` (`FBP_PG_MASK`:
   `REAL_VAL: 0` vs `CONF_VAL: 2`). This is the vendor's own tool reporting the
   mismatch, not an inference from raw sysfs alone.
2. **The mode label updates and persists while the hardware state does not.**
   `nvpmodel -q` correctly reports the target mode by name after every switch; the
   underlying GPU power-gating registers do not reflect it.
3. **An applied state drifts within ~13-20 minutes with no reboot and no mode
   switch.** 20W was verified cleanly applied immediately after its switch (mask,
   CPU/GPU/EMC/DLA/PVA all matching CONF). ~13 minutes later, with no intervening
   action, CPU had drifted to MAXN's ceiling and the mask had reverted to the broken
   `0` state — and a fresh `sudo jetson_clocks` re-lock did not restore it.

An official, exact-matching NVIDIA known issue (Jetson Linux 39.2.0 GA Release
Notes, issue 6236259) independently corroborates part of this: reducing EMC below
Fmax via `nvpmodel.service` during systemd init can crash the service on Orin NX
8GB specifically in 10W mode (this device's only mode with a non-Fmax EMC target,
`2133000000` vs Fmax `~3199000000`) — consistent with, though not a complete
explanation for, the full pattern above (15W's narrower mask-only failure is not
covered by this specific issue).

**This is a finding, not an operator failure.** Full sweep data (all four modes,
raw sysfs vs CONF, per-mode `nvpmodel.service` outcome) is in
`jetson-orinnx/PORT_NOTES.md` §6.

**Scope condition for the whole campaign**: the comparable power-mode set is
**MAXN (NX) ↔ MAXN_SUPER (Nano) — one mode.** The Nano's 15W and 25W have no
trustworthy NX counterpart. All cross-device claims from PB-3 onward use only this
pair.

## MAXN — true clocks and masks from sysfs (the state PB-3+ measurements run under)

| Parameter | CONF (target) | Measured (raw sysfs) | Match |
|---|---|---|---|
| CPU cores online | all 6 | all 6 | yes |
| CPU max freq (all cores) | -1 (unbounded) | 1984000 kHz | yes (hardware ceiling) |
| GPU max freq | -1 (unbounded) | 765000000 Hz | yes (hardware ceiling) |
| EMC max freq | -1 (unbounded) | 3199000000 Hz | yes (hardware ceiling) |
| `tpc_pg_mask` | 240 | 0 | **no** |
| `fbp_pg_mask` | 2 | 0 | **no** |
| `nvpmodel.service` | — | failed (`status=234`) | **no** |

Per the mandatory per-measurement verification rule now in effect: every PB-3+
measurement taken under MAXN must re-read and record this same table immediately
before the measurement runs, and discard the measurement if it disagrees.

## Thermal — 10-minute sustained load, uncontrolled power state (not a MAXN or 20W profile)

Real, measured data — but the run's frequency state drifted (see power-mode
finding above), so it does not represent a clean, locked profile of any single
named mode. Recorded as: **SoC under 10 minutes of sustained GPU load, power state
not fully controlled for.**

- Idle start: ~55-60°C across 9 thermal zones (Tj 59.9°C)
- Peak: Tj 69.9°C at t=568s; CPU peak 70.0°C; GPU peak 68.9°C
- No thermal throttling observed in the load generator's own output
- No clean plateau reached — temperatures were still rising gradually through most
  of the 610-second window; "time to steady state" is not reported since none was
  clearly observed
- Ambient temperature: **not measured** — no sensor available. Recorded as a gap in
  `evidence/06_orinnx_hardware/06_MISSING.md`, not fabricated.

## Memory

8GB module, 7.4 GiB total per `free -h`. The 8GB ceiling is a first-class scope
condition for PB-3 onward: any configuration that does not fit must be recorded
NOT-FEASIBLE with observed peak memory, never silently reduced to fit.

## DLA

Present on this module (`DLA0_CORE`, `DLA1_CORE` visible in `jetson_clocks --show`
and `/etc/nvpmodel.conf`), deliberately unused by this campaign — `jetson/`'s own
pipeline does not target DLA, and mirroring it exactly means not introducing new
engine targets.

## Power supply

Measured from the unit's own label: 19.0V, 2.37A output (45.0W) — the standard
Jetson Orin Nano Developer Kit supply, unchanged across the module swap. Checked
against the Orin NX 8GB module's own ~20-25W power-mode ceiling: leaves real
headroom. Does not fit the observed failure pattern as a primary cause (10W, the
lowest-draw target, was the worst failure; 20W, mid-draw, applied cleanly at first
verification) — recorded as a checked, ruled-out variable, not an assumed non-issue.

## Full supporting detail

`jetson-orinnx/PORT_NOTES.md` — every deviation, the full four-mode sweep table,
the drift finding, the NVIDIA known-issue citation, in full.
