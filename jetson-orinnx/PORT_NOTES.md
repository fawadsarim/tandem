# PORT_NOTES.md — every deviation from `jetson/`, with its reason

`jetson/` is the specification. Everything in `jetson-orinnx/` mirrors it
exactly except where recorded here, with the reason.

## 0. `setup.sh` line 42: `--break-system-packages` added to the `jetson-stats` install — FORCED, not chosen

**Exact location**: `jetson-orinnx/setup.sh`, line 42, inside step `[2/6]`
("apt dependencies (python3-venv, jetson-stats)").

**Original content** (unmodified, still present at `jetson/setup.sh:42`):
```
    sudo pip3 install -U jetson-stats
```

**Changed content** (`jetson-orinnx/setup.sh:42` only):
```
    sudo pip3 install --break-system-packages -U jetson-stats
```

**Full error this device produced on the unmodified line**, captured from
the first `setup.sh` run this gate (`/tmp/setup_log_orinnx.txt`, then
`set -euo pipefail` halted the entire script at this point — steps 3-6,
including `versions.json` generation, never ran):
```
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.

    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.

    If you wish to install a non-Debian packaged Python application,
    it may be easiest to use pipx install xyz, which will manage a
    virtual environment for you. Make sure you have pipx installed.

    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
```

**Reason this is FORCED, not a convenience choice**: PEP 668
("externally-managed-environment") is a deliberate, permanent property of
Ubuntu 24.04 (Noble)'s Python packaging — not a bug, and not something a
better/different flash would avoid. Every JetPack 7.2 install on a Noble
base hits this identically. `jetson/setup.sh` was written and validated
against the Orin Nano Super's OS base, which predates PEP 668 enforcement
— the original bare `pip3 install` was correct and worked there; it is
not merely suboptimal here, it is impossible to run unmodified on this
device's OS. This is different in kind from the MAXN-config bug refused
earlier in this campaign: that was a defect in a specific (SD-image)
flash, absent on this clean SDK-Manager install, so patching it would
have been an unforced, unjustified deviation. PEP 668 enforcement is a
platform property present on every clean install of this JetPack/OS
combination — the deviation is forced by the platform, not chosen for
convenience.

This one flag installs `jetson-stats` system-wide as originally intended
(see the script's own comment, lines 35-36: `jtop` talks to `tegrastats`,
a system service, and needs to be reachable outside any venv) — it does
not weaken the venv isolation used everywhere else in the script (steps
3-4 are untouched).

**Scope of the change**: this one flag, on this one line, in
`jetson-orinnx/setup.sh` only. `jetson/setup.sh` itself was not touched
(read-only, per repo rules) and still contains the original line.

## 0b. `setup.sh` line 96: guard clause widened to accept `MAXN` as well as `SUPER` — FORCED, not chosen

**Exact location**: `jetson-orinnx/setup.sh`, line 96, inside step
`[5/6]` (the pre-flight clock-lock guard).

**Original content** (unmodified, still present at `jetson/setup.sh:96`):
```
if ! echo "${NVPMODEL_OUT}" | grep -qi "SUPER"; then
```

**Changed content** (`jetson-orinnx/setup.sh:96` only):
```
if ! echo "${NVPMODEL_OUT}" | grep -qiE "SUPER|MAXN"; then
```

**Full halt message this device produced on the unmodified line**,
captured from the second `setup.sh` run this gate
(`/tmp/setup_log_orinnx2.txt`), device at `15W` (mode 2) at the time:
```
NV Power Mode: 15W

REFUSING TO CONTINUE: nvpmodel does not report a MAXN SUPER-named power mode as active.
This script will NOT silently run sudo for you (irreversible-ish power-mode/clock changes
belong to the operator, not a setup script). Run these yourself, then re-run setup.sh:

    sudo nvpmodel -q --verbose        # find the MAXN SUPER mode's index for THIS device
    sudo nvpmodel -m <index>          # activate it
    sudo jetson_clocks                # lock clocks to that mode's max
```

**Reason this is FORCED, not a convenience choice**: this guard exists to
confirm the top, unconstrained power mode is active before any latency
measurement proceeds — a real, correct safety check, not a bug in intent.
But its detection method (substring match on `"SUPER"`) encodes a
Nano-Super-specific naming assumption: that device family's top mode is
literally named `MAXN_SUPER`. The Orin NX's top mode is named `MAXN` —
no mode name on this device will ever contain "SUPER", regardless of
which of its four modes (`MAXN`/`10W`/`15W`/`20W`) is active. This is not
a defect introduced by this campaign or by a specific flash — it is the
Orin NX product family's own naming convention, structurally different
from the Nano Super's, and no choice of active mode on this device could
have satisfied the original check.

**Why this specific fix, not a different one**: `grep -qiE "SUPER|MAXN"`
is additive, not a replacement — the original `"SUPER"` branch is left
intact character-for-character (still correctly gates the Nano Super's
own `MAXN_SUPER` naming, if this exact file were ever run there), with an
explicit `"MAXN"` alternative added for this device's own top-mode name.
`"MAXN_SUPER"` itself already contains `"MAXN"` as a substring, so a
single simplified `grep -qi "MAXN"` would technically suffice for both
devices — the two-branch form was chosen instead so the original check's
own literal text stays visible and un-generalized, matching this
campaign's "mirror jetson/, deviate minimally and legibly" principle.

**Known, accepted side effect, not further patched**: the SUCCESS message
immediately after this check (`jetson/setup.sh:108`, unmodified) still
prints "nvpmodel reports a SUPER power mode -- clocks assumed locked."
verbatim, even when only the new `MAXN` branch matched (e.g. this
device's plain `MAXN` mode, no "SUPER" in its name at all). Cosmetically
inaccurate wording in that one case, not a logic error — left as-is
rather than touched, to keep this deviation to the single guard-clause
line per the "one line, change nothing else" discipline already applied
to deviation #0 above.

**Scope of the change**: this one line, in `jetson-orinnx/setup.sh` only.
`jetson/setup.sh` itself was not touched and still contains the original
`"SUPER"`-only check.

**This is the same finding as the campaign's mode-set scope condition,
not a second, unrelated one.** The advisor prompt's own scope table
states only `MAXN` and `15W` exist on both devices (Nano's top mode is
named `MAXN_SUPER`; this device's is plain `MAXN`) — this guard-clause
failure is that exact naming difference surfacing in code rather than in
a comparison table. Read both together: the scope condition explains
*why* the guard needed widening, and this deviation entry is the concrete
mechanism that difference produced.

## 0c. `nvpmodel -m 0 --force` used to reach `MAXN` — PROCEDURAL DEVIATION from the Nano campaign, log which modes need it

**What happened**: to satisfy the widened `setup.sh` guard (0b above),
the device had to actually be switched from `15W` (mode 2, active at the
time) to `MAXN` (mode 0) — a `tpc_pg_mask`-changing transition (240 vs
252 in this device's `/etc/nvpmodel.conf`, same mechanism documented for
this device pre-reflash). `sudo nvpmodel -m 0 --force` was used: it
printed `NVPM WARN: Golden image context is already created` /
`Automatically rebooting to reflect the mode change` and completed a
clean, single reboot with no interactive hang.

**Why this is worth recording explicitly**: `jetson/README.md`'s own
power-mode-sweep section documents the Nano Super campaign's mode
switching as the plain interactive form (`nvpmodel -m <id>`, answer
`YES` at the reboot prompt) — it does not use `--force` anywhere. Using
`--force` here is therefore a real procedural difference from how the
Nano campaign operated, not something to silently carry forward as
equivalent.

**Standing instruction for the PB-2 four-mode sweep**: try the standard
interactive method first, for each mode; fall back to `--force` only if
the confirmation hangs or fails. Log which of the four modes
(`MAXN`/`10W`/`15W`/`20W`) actually needed `--force` versus which
switched cleanly the standard way — this table entry will be filled in
as PB-2 runs each mode:

| Mode | Method needed |
|---|---|
| `MAXN` (0) | `--force` — switched from `15W`, mask-changing transition (240 vs 252). Only mode this campaign needed it for. |
| `20W` (3) | standard (`nvpmodel -m 3`, plain `YES`) — switched from `MAXN`, no hang |
| `10W` (1) | standard — switched from `20W`, no hang |
| `15W` (2) | standard — switched from `10W`, no hang |

**Result**: only the very first switch (to unblock `setup.sh`'s guard)
needed `--force`. All three subsequent sweep transitions completed via
the plain interactive method with no confirmation hang — a real
improvement in reliability over the pre-reflash device's repeated
`optMask`/"bad input" failures on the same standard method. Whether this
difference traces to the reflash itself, fresh boot state, or something
else is not established — recorded as observed, not explained.

## 6. Full four-mode sweep results — raw sysfs vs CONF (the real test, not `nvpmodel -q`)

| Mode | Transition (mask) | CPU on/off | CPU freq | GPU freq | EMC | TPC/FBP mask | `nvpmodel.service` |
|---|---|---|---|---|---|---|---|
| MAXN | 15W→MAXN (252→240) | matches | matches | matches | matches | **stuck at 0/0**, CONF 240/2 | **failed** |
| 20W | MAXN→20W (240→240) | matches | matches | matches | matches | matches (raw 15 = bitwise complement of CONF 240) | inactive (clean) |
| 10W | 20W→10W (240→252) | **mismatch** (all 6 on, should be 4) | **mismatch** (1984000, should be 1190400) | **mismatch** (765000000, should be 612000000) | **mismatch** (unbounded, should be 2133000000) | **stuck at 0/0**, CONF 252/2 | **failed** |
| 15W | 10W→15W (252→252) | matches (4/6 on) | matches (1420800) | matches (612000000) | matches | **stuck at 0/0**, CONF 252/2 | **failed** |

**Headline**: `tpc_pg_mask`/`fbp_pg_mask` never once correctly reflects
the target mode when the CONF-declared mask differs from what was active
before the switch — confirmed on a completely clean SDK-Manager flash, no
patches touching power modes. `nvpmodel -q --verbose`'s own `REAL_VAL`
vs `CONF_VAL` fields corroborate this directly (e.g. MAXN:
`TPC_PG_MASK REAL_VAL: 0 CONF_VAL: 240`) — the vendor's own tool reports
the mismatch when asked, even while `nvpmodel -q` alone reports the
label as correctly active.

**Real correction to an earlier (pre-reflash) interpretation**:
`jetson_clocks --show`'s own "Active GPU TPCs" line is not a reliable
indicator of mask state — it printed inconsistent values (8, then 4,
then 8 again) across modes with supposedly-matching masks, while raw
sysfs and `nvpmodel -q --verbose` agreed with each other consistently.
Raw sysfs (or `nvpmodel -q --verbose`) is the only trustworthy readback;
`jetson_clocks --show`'s summary should not be used alone to certify a
mode applied correctly.

**Genuine, unresolved nuance**: 15W and 10W both request identical
`tpc_pg_mask=252`, yet 15W's non-mask parameters (CPU/GPU/EMC/DLA/PVA)
all matched CONF cleanly while 10W's matched nothing at all — the mask
transition alone does not fully explain the difference in failure
*severity* between these two modes. A plausible sequential-apply
mechanism (mask-change detection aborting the rest of the apply sequence
before it reaches CPU/GPU/EMC) fits most of the four results but not
this specific asymmetry cleanly. Left as an open question rather than
forced into an unverified theory — four data points is not enough to
fully pin down the mechanism.

**Power supply, checked and not implicated**: measured from the unit's
own label — 19.0V, 2.37A output (45.0W), the standard Jetson Orin Nano
Developer Kit supply, unchanged across the module swap (same physical
unit used for the Orin Nano Super before). Against the Orin NX 8GB
module's own ~20-25W power-mode ceiling, this leaves real headroom. Also
does not fit the failure pattern observationally: 10W (the lowest-draw
target) was the *worst* failure, not the best, while 20W (mid-draw)
fully succeeded — a supply-current explanation predicts the opposite
ordering. Recorded as a checked, ruled-out variable, not an assumed
non-issue.

**Official NVIDIA corroboration, found and confirmed after the sweep**:
NVIDIA's own Jetson Linux 39.2.0 GA Release Notes (this device's exact
L4T release — `RN_10698-r39.2.0`), Known Issue **6236259** ("General
System Usability"), states verbatim: "On Jetson Orin platforms, reducing
the EMC to a value below Fmax (such as 3199 MHz) with the following
power modes by using `nvpmodel.service` during systemd initialization
can cause system crashes upon reboot. This issue is especially
noticeable when a display is connected..." — with a table naming
**Orin NX 8GB, mode 10W, EMC 2133 MHz** as an explicitly affected
configuration (against a MAXN EMC of ~3200 MHz). This device's own
`/etc/nvpmodel.conf` confirms 10W is the *only* one of the four modes
with a reduced (`2133000000`) `EMC MAX_FREQ` — MAXN, 15W, and 20W all
declare `-1` (Fmax). This is an exact, official match for 10W's uniquely
total failure (every parameter reverted, not just the mask) and is a
more precise explanation than the general "sequential-abort" hypothesis
above for that one mode specifically. It does not explain 15W's narrower,
mask-only failure (15W's own EMC is also Fmax) — that remains
unattributed to this specific known issue.

NVIDIA's own documented workaround: switch to MAXN (restores EMC to
Fmax) before rebooting, then apply the desired mode after the restart —
not yet tested against 10W this campaign; deferred rather than chased
further, per direction, to keep the sweep moving. Source:
`https://docs.nvidia.com/jetson/archives/r39.2/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.pdf`,
Known Issue 6236259.

A separately-surfaced lead, `overlay_pcie.tbz2` (an NVIDIA-distributed
overlay described as fixing "intermittent boot issues... during power
cycles or reboots" on Orin Nano/Orin NX) — checked, not referenced
anywhere in this release's Known Issues list, name suggests PCIe/NVMe bus
initialization rather than this EMC/`nvpmodel` issue. Not applied;
relevance unconfirmed, dropped per direction rather than pursued further.

**Consequence for cross-device comparison** (per the advisor's ruling):
the comparable power-mode set for any cross-device claim is restricted
to modes whose masks actually applied — **MAXN and 20W only** on this
device. The Nano's own comparable set was `15W`/`25W`/`MAXN_SUPER`. The
only headline-safe comparison is therefore **MAXN-to-MAXN**, a single
mode, not a four-mode or three-mode table. This is a scope condition,
recorded here and to be repeated in `FINDINGS.md` at PB-6.

## 7. Applied state drifts back to unconstrained over elapsed time, not just across reboots — a new, load-bearing finding for PB-3+

**What happened**: 20W was verified cleanly applied (raw `tpc_pg_mask`
complement-of-CONF, all CPU/GPU/EMC/DLA/PVA matching CONF exactly,
`nvpmodel -q --verbose` FBP `REAL_VAL: 2 = CONF_VAL: 2`) immediately
after the mode switch. A 10-minute sustained-load thermal test was
started roughly 13 minutes later, without an explicit re-lock
immediately beforehand (relying on the earlier lock). Post-run
inspection showed `cpu0` pinned at `1984000` (MAXN's ceiling, not 20W's
`1497600`) for the entire 610-sample log, and raw `tpc_pg_mask` back to
`0`. A fresh, deliberate `sudo jetson_clocks` re-lock, run immediately
after noticing this, **did not restore 20W's correct state** — it
re-pinned CPU to `1984000` and left the mask at `0`.

**Why this matters**: this is not the same phenomenon as the boot-time
mask-application failures documented in entries 6 and the EMC/known-issue
note above (those are boot-time, `nvpmodel.service`-mediated). This is a
**live, no-reboot, no-mode-switch drift** — a correctly-applied state,
once achieved, does not stay applied. `jetson_clocks` (bare, no
arguments) does not reliably restore a mode's own specific caps once the
underlying state has drifted; it appears to pin to whatever the current
(possibly already-drifted) ceiling is, not to `/etc/nvpmodel.conf`'s
per-mode values.

**Precedent**: an analogous clock-lock drift (CPU cores reverting to
dynamic scaling ~35+ minutes after a manual lock, cause not fully
isolated) was already observed on this same device pre-reflash. This
entry is the first time the drift has been confirmed to affect the
`tpc_pg_mask`/GPU-cap state specifically, and confirms the phenomenon
survives the reflash — a genuine, reproducible platform property, not
residual state from the old flash.

**Consequence for the measurement protocol from PB-3 onward**: a single
lock-and-verify at the start of a mode's measurement session is not
sufficient. Raw sysfs (or `nvpmodel -q --verbose`) should be re-checked
immediately before — ideally alongside — every real latency/power
measurement, not just once per mode switch, or a long-running measurement
run risks silently drifting onto a different, unintended clock/power
state partway through. This is a process requirement for PB-3, not yet
implemented there — flagged here so it isn't missed.

**Honest gap this leaves in the just-completed 10-minute thermal run**:
its temperature data is real and was genuinely measured, but its
frequency columns (`cpu0_cur`, `gpu_cur`) do not represent a stable,
locked 20W state for the full duration — the run likely transitioned
from a 20W-like state toward a drifted, MAXN-like state at some point
before or during the load period, given the post-run inspection. The
temperatures observed are therefore better characterized as "this
device's SoC under 10 minutes of sustained GPU load, power state not
fully controlled for," not "20W's specific thermal profile."

## 1. `out/` moved aside; fresh `out/` seeded with only the ONNX segments

**What changed**: the `a991935` "Tandem Jetson NX Starter" commit copied
`jetson/` in its entirety, including `out/` — meaning every measured-data
file (`latency_lut.json`, all `powersweep/latency_lut_*.json`,
`paired_eval_result.json` — the registered hardware gate itself,
`energy_lut.json`, `frontier_result*.json`, `versions.json`, pilot
results, `reviewer_release/`) was the Orin Nano Super's own data, byte-
identical, sitting under an NX-labeled path. Confirmed by a full recursive
hash sweep (`MIRROR_AUDIT.md`, Gate PB-0): 132/132 files identical.

At the advisor's explicit ruling: `jetson-orinnx/out/` was renamed to
`jetson-orinnx/out_nano_stale/` (kept as a structural reference for the
rest of this campaign, to be deleted at Gate PB-6), and a fresh, empty
`jetson-orinnx/out/` was created. Only `out/segments/` (the 5 production
ONNX files) was copied back into it — verified byte-identical to the
original via a second hash pass after the copy.

**Reason**: overwriting the stale files in place would have destroyed the
ability to tell "this stage has run on the NX" from "this stage hasn't
run yet, you're reading Nano data" at any future gate. From this point,
anything under `out/` is, by construction, a real NX measurement — the
only files that were legitimately REUSE-appropriate (device-independent
definitions, not measurements) were the 5 ONNX segments, which were
restored.

**Correction during this same step**: the first copy pass only grabbed
`out/segments/*.onnx`, missing `out/segments/calib/*.npy` (the pinned
calibration images, also REUSE-appropriate by the same policy). Caught via
`git status` still showing those two files as deleted after the first
copy; fixed by copying `segments/calib/` too and re-verifying both
`.npy` files byte-identical to `out_nano_stale/`.

**Scope of the change**: `jetson-orinnx/out/` (renamed and rebuilt from
scratch) only. `jetson/out/` itself was never touched (read-only, per
repo rules). The identical move was applied on the device's own
`/home/tandem-jetson/jetson-orinnx/` at Gate PB-1, before `setup.sh` or
`self_test.py` ran, so neither would read stale Nano data — verified via
a full recursive hash comparison (132/132 files matched `jetson/` before
the move; all 7 restored segment/calib files hash-identical to the host's
copy after).

## 2. `pilot_b8/`, `pilot_deitb/`, `pilot_deitb_ext/`, `reviewer_release/` — SCOPE DECISION, out of campaign

**What changed**: these four `out/` subdirectories (present in the
original `a991935` copy, now living in `out_nano_stale/`) are **not**
being regenerated as part of this campaign and are not restored into the
fresh `out/`.

**Reason**: none of Gates PB-0 through PB-6 mention the batch-8 INT8
pilot, the DeiT-B INT8 pilots, or a reviewer-release bundle — the gate
spec's scope is the core cost-table / power-sweep / replan / paired-eval
/ frontier / energy pipeline for the primary model across the four NX
power modes. This is a decision, not an oversight: including them would
add work outside what this campaign was asked to characterize, with no
corresponding place in the PB-0–PB-6 deliverables to report them.

**Scope of the change**: these four directories remain in
`out_nano_stale/` for reference only, never regenerated on this device
during this campaign.

## 3. `README.md` left byte-identical; `README_ORINNX.md` added as a sibling

**What changed**: `jetson-orinnx/README.md` is unmodified (still says
"G1-HW: Jetson Orin Nano Super" throughout). A new file,
`README_ORINNX.md`, was added alongside it — a short note pointing a
reader to `DEVICE_PROFILE.md` for this device's identity and to this file
for the deviation log. `README.md` itself was not edited.

**Reason**: "no patches, no improvements, mirror jetson/ exactly" argues
against editing the header text — but a document whose header names the
wrong device, sitting in this directory with no pointer, is a trap for an
independent reproducer. The sibling file resolves that without touching
the mirrored file.

**Scope of the change**: one new file, `jetson-orinnx/README_ORINNX.md`.
`jetson-orinnx/README.md` (and `jetson/README.md`) untouched.

## 4. `self_test.py` check 3 ("Clocks locked") FAIL at Gate PB-1 — expected, not a defect, must not stay ambient

**What happened**: `self_test.py` run at PB-1 (before any measurement
work, before clocks were ever locked) reported:
```
[FAIL] 3. Clocks locked (jetson_clocks): unlocked rail: cpu0: ... CurrentFreq=1728000
```

**Why this is expected, not a bug**: `setup.sh` deliberately never runs
`jetson_clocks` itself — it only checks that the *mode name* is right
(step `[5/6]`), matching `jetson/`'s own documented design where clock
locking is always a manual operator action, never scripted end-to-end.
At PB-1, clocks genuinely were not yet locked — this FAIL is a correct
report of real, expected state, not a broken check or a broken device.

**What must be true from here on**: every PB-2 measurement runs behind a
verified `sudo jetson_clocks` lock (per the sweep protocol below). If
`self_test.py` is re-run during or after a measurement phase and check 3
still fails, THAT would be a real problem, not an expected one — the
distinction is timing. A future reader hitting this FAIL entry in an old
log should check *when* the run happened before concluding anything is
broken.

**Scope**: no code change — this entry exists purely so the PB-1 FAIL is
never mistaken for an unresolved defect later in the campaign.

## 5. Frozen repo copied to the device — method, a real placement mistake caught and fixed, verification

**Method**: `pscp` was abandoned for this exact task last campaign
(two silent failures, later diagnosed as PuTTY-specific). This time:
SSH key auth was set up first (added this laptop's public key to the
device's `~/.ssh/authorized_keys`, alongside an existing
`NVIDIA-SDK-Manager` entry already there from the flash process), then
the repo (`jetson/`, `experiments/`, `evidence/`, `docs/`, `scripts/`,
`requirements/`, `notebooks/`, `tandem/`, `results/`, `notes/`, and root
files — excluding `.git/`, `.claude/`, `jetson-orinnx/` itself,
`smr_study/`) was tarred locally, hashed, transferred via native `scp`
(28 seconds for 569MB), hash-verified identical on arrival, then
extracted. Native `scp` worked cleanly on the first attempt — no repeat
of the earlier PuTTY-specific failures, consistent with that diagnosis.

**Content integrity**: a full recursive hash comparison of
`experiments/` and `evidence/` (the two directories the parity/self-test
scripts actually read) — device vs. host source — showed 1614/1614
shared files byte-identical, zero differences. The 29 host-only files
were RepQ-ViT's own nested `.git/` clone metadata, correctly excluded by
the `.git` tar pattern, not a real gap.

**A real placement mistake, caught by re-running the check it was meant
to fix, not assumed**: the first extraction placed everything under a
wrapper directory, `/home/tandem-jetson/tandem_frozen_reference/` —
`replan.py --verify-port` still failed afterward, identically to before.
Root-caused (not guessed) by reading `_common.py:22-24`:
`REPO_ROOT = JETSON_ROOT.parent` — the scripts expect `experiments/`,
`evidence/`, `jetson/`, etc. to sit as **direct siblings** of
`jetson-orinnx/` under `/home/tandem-jetson/`, matching the real repo's
own top-level layout, not nested inside an extra wrapper directory.
Fixed by moving every extracted item up one level so
`/home/tandem-jetson/experiments/`, `/home/tandem-jetson/jetson/`, etc.
sit alongside `/home/tandem-jetson/jetson-orinnx/` directly. Re-ran
`replan.py --verify-port` after the fix: 6/6 MATCH. Re-ran `self_test.py`
after that: 10/11 (only check 3, clock-lock, remains — expected, see
entry 4 above).

**Scope of the change**: device filesystem only
(`/home/tandem-jetson/`), outside the git-tracked repo entirely — no
`PORT_NOTES.md`-relevant script or config file was touched. Recorded here
because it's part of this campaign's operational record and because the
placement mistake is exactly the kind of "trusted a plausible-looking
result instead of verifying it" error this project's standing discipline
exists to catch.

## 8. `out/segments/manifest.json` reconstructed; engines built from the `_qdq` ONNX; `int8_naive_implicit` excluded

**What's missing from the committed tree, and why**: `jetson/out/segments/` (and
this device's reused copy) has no `manifest.json` and no non-suffixed ONNX
(`block_seg_n{N}.onnx`) — `build_engines.sh` expects both. Confirmed via
`.gitignore`: `out/segments/*` is blanket-ignored with specific carve-outs only for
the 5 `_qdq.onnx` files and 2 of 5 buckets' calibration `.npy`. `manifest.json` and
the plain ONNX are HOST-export artifacts (produced fresh by `export_segments.py`,
normally copied to-device alongside the ONNX), correctly excluded from git as
regenerable — this is not a gap introduced by this campaign; `jetson/`'s own
committed tree has the identical absence.

**`manifest.json` reconstructed, not re-exported**: `build_engines.sh` only reads it
for the bucket-number list. Content written directly from the 5 known ONNX
filenames — no PyTorch/ONNX export pipeline was re-run. The file itself is flagged
as reconstructed via a `"_reconstructed"` field (see the file's own content) so a
future reader diffing it against a real HOST export sees immediately that it isn't
one.

**FP16 and INT8 both built from the `_qdq`-suffixed ONNX**: standard, valid
TensorRT practice — Q/DQ nodes are simply unused/no-ops in an FP16 build. No plain
non-suffixed ONNX was available to build from instead, and none was reconstructed
(that would require re-running the export, which reuse policy forbids).

**`int8_naive_implicit` excluded from this campaign, for two independent reasons**:
(1) PB-3's scope is precisions `{16,8}`, and the "8" `replan.py` actually consumes
is the QDQ path (`measure_lut.py`'s own `PRECISIONS` comment: "int8" = best-achievable
QDQ, "int8_naive_implicit" = naive, recorded for provenance only, never read by
`replan.py`) — matching the Nano's own LUT, which this campaign must stay
comparable to. (2) It is infeasible regardless for 3 of 5 buckets: calibration data
(`out/segments/calib/calib_n{N}.npy`) exists only for `n197`/`n69` — see
`evidence/06_orinnx_hardware/06_MISSING.md`. Two independent reasons to exclude it,
not a single convenient one.

## 9. FP-parity gate — NOT RUNNABLE as `jetson/` defines it; substitute checks FAILED, unresolved

**Why the gate as specified could not run**: `jetson/`'s FP-parity semantics require
a true (non-QDQ) FP32 ONNX reference to compare a built engine against. Only the
QDQ-annotated ONNX (`block_seg_n{N}_qdq.onnx`) is present in the committed tree —
the plain ONNX is a `.gitignore`'d HOST-export artifact, absent by design (same root
cause as item 8). No plain build ever escapes the graph's own baked-in Q/DQ nodes —
`--fp16`/`--int8` select tactics, they do not add or remove quantization already
present in the graph. There is no artifact on this device from which a true
unquantized reference could be built without re-running the HOST export, which reuse
policy forbids.

**Substitute checks run instead** (`jetson-orinnx/parity_check_orinnx.py`, new
script, no `jetson/` precedent):
- **Check A** (n197, n69 only — the two buckets with calibration data): plain-build
  engine vs FP16 engine, real calibration sample as input (not random noise, which
  was tried first and produces the same-order diffs for the reason above — a
  quantized graph's Q/DQ scales are calibrated for real activation distributions,
  not Gaussian noise). Result: max_abs_diff `3.52e-01` (n197), `2.37e-01` (n69).
- **Check B** (all 5 buckets): FP16 engine vs INT8(QDQ) engine, same random input.
  Result: max_abs_diff `0.20`-`0.29` across all 5 buckets — the same order of
  magnitude as Check A, not scaling with bucket size, precision, or input type.

**Harness verified clean before accepting these numbers** (not assumed, both
results recorded here so "unresolved" is a checked conclusion, not an unverified
shrug):
1. **Self-consistency**: the same FP16 engine (n197) run twice on identical
   fixed-seed random input produced **`max_abs_diff = 0.0` exactly**, output shape
   `(1,197,384)`, identical first-5 values printed and matched byte-for-byte —
   confirms the comparison code is deterministic and reads outputs correctly.
2. **Binding structure** (n197): `tokens_in`(input, shape `(1,197,384)`, `FLOAT`)
   at index 0, `tokens_out`(output, same shape/dtype) at index 1 — **identical name,
   order, shape, and dtype across all three engine variants** (plain-build, FP16,
   INT8-QDQ). No cross-engine binding-index mismatch.

**Close-out, per explicit ruling**: the gate is recorded as **FAILED, UNRESOLVED**.
The substitute checks did not pass. With the harness eliminated by the two checks
above, the divergence is attributed to real engine/quantization behavior on this
device, not a comparison-code defect — though the checks above rule out *the specific
failure modes tested* (determinism, binding mismatch), not every conceivable harness
issue, so "unresolved" (not "confirmed real") remains the accurate label. This is a
**real limitation on what PB-3's LUT can claim**: the latency numbers themselves
remain valid measurements of built, running engines on this device — but numerical
equivalence between the deployed artifact and the source model was **not
established**. This must be stated prominently in `FINDINGS.md` at PB-6, not left as
an appendix note.

**Worth a sentence in `FINDINGS.md`, not investigated further**: FP16 and INT8
engines built from the *same* QDQ graph diverging by 0.20-0.29 suggests TensorRT's
precision-tactic selection on this device produces materially different numerics for
the same graph — adjacent to this campaign's own subject (precision behavior), not
just a side detail. Flagged here so it survives to PB-6, not pursued now.

**Downstream consequence, stated now so it isn't discovered later**: any
accuracy-dependent result built on these engines (paired eval, frontier, PB-5)
inherits this same caveat — deployed-artifact numerical equivalence to the source
model was not established. PB-5 must state this explicitly wherever it reports an
accuracy number from these engines, not just LUT/latency ones.

Per the explicit "stop after two failures" rule, no further diagnostic attempt was
made beyond the one authorized self-consistency/binding check. `06_MISSING.md`
records the full gate as not run, with the reason.

## 10. `measure_lut.py`: mandatory per-measurement sysfs verification + peak memory added; `int8_naive_implicit` removed

**What changed, and why each part is not a `jetson/` precedent**:
- `PRECISIONS` changed from `("fp16", "int8", "int8_naive_implicit")` to
  `("fp16", "int8")` — see item 8 (out of scope, and infeasible for 3/5 buckets).
- Added a pre-measurement raw-sysfs read (`tpc_pg_mask`, `fbp_pg_mask`, per-CPU
  online, CPU/EMC max frequency) before every cell — the mandatory PB-3
  requirement. **Interpretation of the discard rule, confirmed explicitly**: gated
  on CPU/EMC frequency exact-match against MAXN's fixed hardware ceilings
  (`1984000` kHz / `3199000000` Hz, confirmed repeatedly this session — exact
  match chosen over a percentage band since these are fixed values at discrete
  clock steps, no reason to expect "close but not exact") — one re-lock +
  re-read on mismatch, then DISCARD and flag if still wrong. **NOT gated on
  `tpc_pg_mask`/`fbp_pg_mask` matching CONF** — that mismatch is PB-2's own
  characterized, accepted MAXN finding (`REAL_VAL: 0` vs `CONF_VAL: 240/2`,
  confirmed via `nvpmodel -q --verbose`); gating on it would discard every cell
  and produce an empty LUT, telling us nothing new. Instead, the mask value is
  recorded on every cell and compared against the *previous* cell's own reading
  — a **change** (not a CONF mismatch) is treated as a genuinely new event and
  raises, stopping the run, since it would mean the device is less stable than
  PB-2 characterized. This bridge — "discard on CONF mismatch" in the spec,
  "mask mismatches everywhere, nothing discarded for it" in the data — is
  written down here explicitly so a future reader isn't left to infer it.
- Added peak GPU memory (`torch.cuda.max_memory_allocated()`, reset before each
  cell) and peak system RAM (`jtop`'s `memory.RAM.used`) to every LUT entry —
  the 8GB-variant first-class-result requirement. GPU-max-frequency sysfs path
  was found unreliable on this device earlier (PB-2) — recorded as `null` in
  `verified_sysfs_state` rather than a wrong or guessed value.

**No jetson/ precedent existed for any of this** — the Nano campaign measured a
single mode per run with no observed live drift, so it never needed per-cell
sysfs verification, and its 24GB-class module never needed peak-memory as a
first-class result. `jetson/measure_lut.py`'s own core measurement logic
(`WARMUP_ITERS`, `MEASURE_ITERS`, thermal guard, median/p99 computation,
resume-safe write-after-every-cell) is otherwise byte-for-byte unchanged.

**Consequence for §6's "diff only in numbers" expectation**: `latency_lut.json`
entries here carry additional keys (`verified_sysfs_state`, `peak_gpu_mem_bytes`,
`peak_ram_used_kb`) beyond `jetson/`'s own schema. This is an intentional,
advisor-mandated addition, not a structural drift to be treated as an error —
every key `jetson/`'s own schema defines is still present, unchanged, in the
same format.

## 11. PB-4: single-mode `powersweep_analysis.py` run; two Nano reference files copied in unmodified

`out/lut/latency_lut.json` copied to `out/powersweep/latency_lut_MAXN.json` (the
naming `powersweep_analysis.py` expects: `latency_lut_<mode>.json`, one mode
instead of three). `jetson-orinnx/powersweep_analysis.py` itself run unmodified.
It also reads two Nano-committed reference files for print-only comparison
context (`out/energy/proxy_divergence.csv`, `out/pilot_deitb/pilot_deitb_utilization.json`)
— copied in from the frozen `jetson/` mirror already on this device (hash-verified
identical), since their absence crashed the script on an unguarded `None` format
call. Read-only comparison references, not measurements — appropriate to reuse.

**Worth naming as its own observation**: that the script crashes at all when these
two files are absent is a portability limitation in `jetson/`'s own tooling, not a
defect introduced by this campaign. `powersweep_analysis.py` was written and only
ever run against the Nano's own committed tree, where these files always existed by
construction — nothing in the original single-device project ever exercised the
"reference files missing" path. A second-device campaign is exactly the condition
that surfaces this: the script assumes its own prior committed data is always
present, which is true for a repeat run on the same device and false the first time
a *different* device's `jetson-orinnx/` tree runs it fresh. Not fixed here (`jetson/`
is read-only) — recorded as a real gap a future third-device port would hit
identically, and copied-file provenance was traced line-by-line at Gate PB-6's
provenance audit (V0) to confirm these two files never influence a computed NX
number, only labeled comparison output — see that audit for the full trace.

## 12. PB-5 `replan.py`: pre-registered budgets don't reach this device's achievable range — genuine cross-device finding, theorem test Branch D (untested, not confirmed/refuted)

**What happened**: `BUDGET_FRACTIONS = (0.15, 0.21, 0.293, 0.41, 0.572, 0.80)` (ported
unchanged from `jetson/`, calibrated against the Nano's own achievable range) produce
`tau_b >= Latency_id` at **all 6 budgets** on this device (103.7%-140.5% of
`Latency_id`). Real achievable range here: `[76.7%, 100%]` — every pre-registered
budget sits looser than doing nothing at all. JOINT and STAGED land on the identical
(identity) plan at 6/6 budgets as a direct, mechanical consequence — not a test of
the theorem's mechanism. `replan.py`'s own printed output states this plainly:
"Joint-vs-staged remains genuinely UNTESTED under real hardware cost until a budget
sweep set INSIDE the achievable range forces real compression."

**Why this is Branch D, not Branch B**, despite "joint == staged" nominally matching
Branch B's stated outcome: `PREREGISTRATION_INT8.md`'s Branch B text ("the theorem
holds and its boundary is confirmed") presumes a *meaningful* tie — one where the
allocator actually faced a decision under real compression pressure. Here it did
not. Calling this Branch B would be technically-true-but-misleading; the achievable-
range table catches it immediately. Recorded as Branch D: described exactly as
observed, not forced into a branch the pre-registration didn't anticipate.

**Independent, emergent cross-device finding**: the pre-registered budget set is
downstream of the Nano's own measured cost table and does not transfer to this
device's own table. This is a direct, unconstructed confirmation of this campaign's
own closing claim (the measured table, not any fixed budget derived from it, is the
transferable artifact) — found as a side effect of running the real protocol, not
built to demonstrate the point. Belongs in `FINDINGS.md` as its own result, not a
footnote to the replan section.

## 13. Paired eval / frontier — NOT RUN (ImageNet unavailable); see `06_MISSING.md`

No substitute attempted. Full reasoning in `evidence/06_orinnx_hardware/06_MISSING.md`.
Every accuracy-dependent figure this campaign *does* report (replan's `D_pred` values)
already carries the PB-3 parity caveat regardless.

## 14. PB-5 reproduction gate: adapted, since this campaign has no prior committed LUT to compare against

`jetson/powersweep_repro_gate.py` compares a re-measurement against a
*previously-committed* `latency_lut.json` (real weeks-apart reproducibility). This
campaign's PB-3 LUT is this device's first-ever measurement — there is no older
committed LUT to gate against. Adapted: PB-3's own LUT backed up as the reference
(`out/lut/latency_lut_pb3_reference.json`), `measure_lut.py --force` re-run for all
10 cells same-session, compared at the same 3% tolerance.

**Correction to this section, found and fixed before the PB-6 evidence package was
assembled**: the first attempt at this gate restored `out/lut/latency_lut.json` to
its original PB-3 values *before* saving a separate copy of the fresh remeasurement
— meaning `latency_lut_remeasured_reprogate.json` (as first written) was actually a
second copy of the *original* data, not the remeasurement, even though the
comparison printed at the time (10/10 within 3%, computed live before the restore)
was numerically genuine. Caught by hashing the evidence files while assembling
`06_MANIFEST.md` and finding `latency_lut_remeasured_reprogate.json` byte-identical
to `latency_lut_pb3_reference.json` — which should never happen for an independent
remeasurement. Fixed by re-running `measure_lut.py --force` a second time and
saving the fresh output *before* restoring the canonical LUT.

**Result of the correctly-saved remeasurement: 9/10 within 3%.** `n101_int8` exceeded
tolerance: committed `0.2717ms`, remeasured `0.2806ms`, **absolute gap `0.0089ms`,
delta +3.26%** (vs the 3.0% tolerance) — full table in
`out/lut/reproduction_gate_orinnx.csv`. Per the explicit reproduction-gate semantic
(record, don't halt): this failure is recorded here, not retried again, not
silently dropped. No other cell came close to the boundary (next highest:
`n69_int8` at +2.33%). **No cause is established** — the number is stated as
observed, with both its absolute and relative size given, rather than characterized
as noise or anything else without evidence for that characterization. The first
(mis-saved) run's own 10/10 figure is superseded by this one — recorded here for
transparency about the mistake, not cited as a result anywhere else in this
package.

The corrected remeasurement (`out/lut/latency_lut_remeasured_reprogate.json`) is
kept as gate evidence, hash-verified to genuinely differ from
`latency_lut_pb3_reference.json`; `out/lut/latency_lut.json` was restored to the
original PB-3 values afterward (hash-verified identical to the backup) so PB-4/PB-5's
already-reported numbers stay consistent with what's on disk.

**Flag for commit time (not acted on — no git operations performed)**: `out/lut/*`
is blanket-`.gitignore`'d with only `latency_lut.json` itself whitelisted
(inherited unchanged from `jetson/.gitignore`). The three new evidence files this
gate added under `out/lut/` — `latency_lut_pb3_reference.json`,
`latency_lut_remeasured_reprogate.json`, `reproduction_gate_orinnx.csv` — are
therefore currently git-ignored, not just untracked. They exist on disk and are
real campaign evidence, but will need an explicit `git add -f` or a `.gitignore`
adjustment at commit time or they will be silently dropped.

## 15. PB-6 transfer experiment: achievable cost ranges are DISJOINT — no cost-gap figure reported, by design

**New script, no `jetson/` equivalent**: `jetson-orinnx/transfer_experiment_orinnx.py`
reuses `replan.py`'s own DP machinery (`build_schedule_pareto_sets`,
`exact_select_for_budget`, `plan_cost`) against two cost tables: this device's own
measured LUT, and the Orin Nano's LUT read directly from the frozen `jetson/` mirror
(`/home/tandem-jetson/jetson/out/lut/latency_lut.json`, read-only, never written to).

**Result**: the two devices' achievable cost ranges do not overlap.
- NX: `[2.8292ms, 3.6902ms]` (76.7%-100% of NX's own `Latency_id = 3.6902ms`)
- Nano: `[3.7958ms, 4.9668ms]` (76.4%-100% of Nano's own `Latency_id = 4.9668ms`)

The relative floor (~76.7% vs ~76.4%) is nearly identical — the *shape* of the
achievable range transfers. The *absolute values* do not: NX's ceiling (3.6902ms)
sits below Nano's floor (3.7958ms), with a real gap between them. Four budgets
inside NX's own achievable range (80/85/90/95% of NX's `Latency_id`, chosen because
the pre-registered `BUDGET_FRACTIONS` don't reach it either — item 12) were tested
against Nano's table: **all four infeasible** — no plan in Nano's own 85-schedule x
3-bit-menu space costs that little under Nano's own table.

**No cost-gap figure is reported, and a workaround was deliberately not
constructed.** The original ask (measure the cost gap between a Nano-table-selected
plan and an NX-table-selected plan at a shared budget) presumes a budget where both
tables produce a non-trivial selection. None exists: below Nano's floor, Nano's
table is infeasible; above NX's ceiling, NX's own identity plan already wins
trivially (the same degeneracy PB-5 found). Any budget choice would only relocate
which side degenerates, producing a number that looks like a result without being
one. Considered and rejected for this reason, not overlooked.

**`Latency_id` ratio**: NX / Nano = `3.6902 / 4.9668 = 0.743` — this device is
roughly 26% faster on this workload at the identity (no-compression) plan, giving
the scale of the device difference in one number.

**This is the same result as item 12 (budgets don't transfer), seen from the other
direction** — item 12 showed the pre-registered fractions don't reach this device's
range; this shows the two devices' ranges don't reach each other at all. One
finding, two angles: budgets don't transfer *because* the achievable ranges are
disjoint. Merged into one `FINDINGS.md` entry, not reported as two separate results.

## 16. Verification campaign (V2): the per-measurement gate checked configured ceilings, not achieved state — corrective patch, PB-3 does not reproduce under verified pinning

**What V2 found.** Testing whether item 10's modified `measure_lut.py` was safe (vs
the unmodified `jetson/measure_lut.py`) required first matching `jetson/`'s own
protocol (full 5-bucket sweep, one continuous process). Under that protocol, the
*unmodified* script — byte-identical to `jetson/`'s own — failed to reproduce the
PB-3 canonical LUT by up to +54.9%, and failed to reproduce *itself* run-to-run by
up to +22.5%. A script cannot be contaminated by a modification it does not
contain; the fault was in measurement conditions, not the modification.

**Root cause, confirmed directly.** `freqs_match_maxn()` (this file's own item 10
addition) compared `scaling_max_freq` (CPU) and `bwmgr/max_freq` (EMC) — both
static, mode-level *ceilings* that `nvpmodel` sets once and DVFS governors never
touch — against MAXN's expected ceiling values. It never read GPU frequency at all
(`gpu_max_freq_hz` was hard-coded `None`, "path unreliable" per PB-2). It never read
*achieved* (current) frequency for anything. `relock_clocks()` (`jetson_clocks`)
fired only reactively, on a ceiling mismatch — which, because ceilings track the
nvpmodel mode rather than governor state, structurally could not occur. **The
result: `jetson_clocks` was never invoked during any measurement in this entire
campaign, including PB-3 itself, and the gate could not fail — it verified
nothing about actual clock state.** Confirmed live: CPU governor `schedutil`, GPU
governor `nvhost_podgov`, GPU sitting at its 306MHz floor under a 765MHz ceiling,
both governors reactive rather than pinned.

**Thermal was checked and excluded as the explanation**, using fields this campaign
already recorded: PB-3's own LUT entries show `temp_c≈60.1–60.5°C`,
`mean_power_w≈8.2–8.6W`; the V2 re-run measured `57–58°C`, `5.4–7.7W` — cooler and
lower-power while slower. Throttling produces the opposite signature. Ruled out by
data already on hand, not by new instrumentation.

**Corrective patch to `measure_lut.py`** (this device's script only, `jetson/`
untouched): `jetson_clocks` is now invoked **unconditionally** before every cell,
not reactively. `read_sysfs_state()` now reads achieved CPU/GPU/EMC frequency
(`scaling_cur_freq` / devfreq `cur_freq`) and CPU/GPU governor name, and reads GPU
ceiling properly (`/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq`,
found via `nvpmodel -q --verbose`'s own path listing — no more `None`). A new
`achieved_matches_maxn()` gates admission on **achieved** frequency matching
`MAXN_EXPECTED_ACHIEVED` (established empirically by invoking `jetson_clocks` and
reading the resulting `cur_freq` nodes directly: CPU 1984000kHz, GPU 765000000Hz,
EMC 3199000000Hz) — this, not the old ceiling check, is now the real gate.

**Deliberately not gated on governor name.** Confirmed empirically that
`jetson_clocks` pins achieved frequency to ceiling on this device *without*
renaming either governor (`schedutil`/`nvhost_podgov` persist even at fully pinned
frequency). Gating on the label `"performance"` would reject every cell
unconditionally regardless of true state — the same label-vs-reality gap
`FINDINGS.md` finding 5 already documents for `nvpmodel` mode names, recurring here
for governor names. Governor is recorded on every cell, not gated on.

**Re-measurement, patched gate, two full 5-bucket sweeps (V2-FIX run1/run2):**
achieved frequency matched MAXN on the first attempt, every cell, both runs — no
discards. Run-to-run agreement is excellent (max 0.52% across all 10 cells) —
proof the patched gate produces genuinely stable measurements. **But neither run
reproduced PB-3**: every cell 5.3%–10.8% slower, consistently, both runs. Read in
isolation this is Outcome B of the pre-registered decision tree (agree with each
other, not with PB-3 → new LUT replaces PB-3) — **but this was provisional and was
overturned by the mask investigation immediately below; it is recorded here for
the trail, not as the final call.**

**The actual driver: `tpc_pg_mask`, not clock pinning.** PB-3's own
`verified_sysfs_state` recorded `tpc_pg_mask=0, fbp_pg_mask=0` on every checked
cell. Both V2-FIX run1/run2 (mask `15,2`) diverged from PB-3 by 5.3-10.8%. A
reboot (see the unresolved event below) changed the live mask from `15,2` back to
`0,0` with no code change and no mode switch — direct proof the mask is not a
`jetson_clocks`-controllable or otherwise stable parameter, confirming the risk
that the earlier `15` reading might never recur predictably. **A third full sweep
(V2-FIX run3) was taken immediately at this new `mask=0,0` state, same patched,
achieved-frequency-verified gate: it reproduced PB-3 within 1.55% max on all 10
cells**, with temp/power also converging on PB-3's own recorded values (60.0-60.2°C
/ 7.3-8.0W vs PB-3's 60.1-60.5°C / 8.2-8.6W, vs run1/run2's cooler 57-59°C /
7.3-8.4W). This settles it: **the mask state, not the clock/governor gate hole, was
the dominant driver of the gap.** The clock fix was still necessary and correct —
without it, achieved frequency was never verified at all — it just wasn't
sufficient on its own, and would have looked sufficient (run1/run2 agreeing with
each other to 0.52%) if the mask question hadn't been chased down.

**Leading hypothesis for PB-3 itself, stated as a hypothesis, not confirmed.** PB-3
ran in the same boot session as PB-2's manual `sudo jetson_clocks` (invoked during
PB-2's four-mode sweep); the unpatched V2 baseline test ran ~16 hours into a
*different* boot session, well past any residual pinning. This plausibly explains
part of why PB-3 ran faster than the unpatched-gate V2 baseline — but does not by
itself explain the ~5-11% gap that persists even after this patch verifies
identical achieved frequency, which is why the mask discrepancy above is recorded
as a separate, live question rather than folded into this hypothesis.

**One unresolved, unexplained event**: the device rebooted at 2026-08-27 17:34,
five minutes after the unpatched V2 baseline sweep's last write. This device
retains no persistent journal across reboots (`journalctl --list-boots` shows a
single boot ID spanning the entire wtmp history), so the cause could not be
determined. Recorded as unresolved; not guessed at. (A second, deliberate reboot
was performed the same session specifically to test mask stability across boots —
see above; that is a separate, understood event, distinct from this first
unexplained one.)

**Resolution and second corrective patch.** `run3` is adopted as the new canonical
`out/lut/latency_lut.json`. PB-3's original file is archived, not deleted, at
`out/lut/latency_lut_pb3_superseded_unverified_clocks.json`. A second addition to
`measure_lut.py`: a session-start precondition (`MASK_KNOWN_GOOD = (0, 0)`,
checked once when the measurement loop opens, distinct from the existing per-cell
"mask VALUE CHANGE between cells" stop condition, which still stands) that aborts
before any cell is measured if the mask is not in the one state confirmed, by
direct re-measurement against PB-3, to be safe. This does not contradict item 10's
"record but don't gate per-cell" rule — that rule is about *within-session*
mask mismatch against `CONF` (expected, characterized, not a defect); this is a
*between-session* precondition on the mask's own value, which item 10 never
addressed because the value's own instability wasn't known until this audit. If a
future session boots into a mask value other than `(0,0)`, the script now refuses
to measure rather than silently producing another `15`-state LUT — the only lever
observed to change the mask is a reboot, outcome unpredictable, so the operator
must reboot and re-check rather than the script attempting anything automatic.

**Downstream recomputation, all from the new canonical LUT:**
`out/powersweep/latency_lut_MAXN.json` (copy), `precision_axis_by_mode.csv`,
`proxy_divergence_by_mode.csv`, `powersweep_analysis_summary.json` (re-run
`powersweep_analysis.py`); `out/replan/{plans_hw.csv,plans_hw_divergence.json,
replan_manifest.json}` (re-run `replan.py`); `out/transfer/
transfer_experiment_result.json` (re-run `transfer_experiment_orinnx.py`, Nano's
own table read-only, untouched). New headline figures: proxy divergence 2.2592x
(Nano reference 2.2352x, 1.1% apart); INT8/FP16 ratios 1.1439/1.1569/1.2554/
1.3462/1.4566 (69→197, still monotonic, still 5/5 slower); `Latency_id`(NX)
3.6330ms (was 3.6902ms); NX achievable range `[2.8031, 3.6330]`ms, 77.2%-100% of
its own `Latency_id` (was 76.7%-100%); `Latency_id` ratio NX/Nano 0.7315 (was
0.743). Every figure moved by a few percent at most; direction, shape, and every
qualitative conclusion in `FINDINGS.md` (proxy divergence reproduces, INT8 slower
5/5 monotonically, achievable ranges disjoint, Branch D untested) are unchanged.
Not affected by this recomputation, per the same reasoning that scoped V2-FIX
itself: energy (`out/energy/energy_lut.json`, measured independently, live, not
LUT-derived), the reproduction gate (§14, self-contained, both its files are NX
runs from before this correction and remain the honest historical record of that
specific check), and everything in `jetson/`.

`verify_orinnx_claims.py`'s hard-coded expected values were updated to match (with
an inline note pointing here for why) and re-run: `ALL CLAIMS VERIFIED`, 0
mismatches, from the new canonical LUT.

**Reproduction gate re-closed under the corrected protocol.** Item 14's original
9/10 figure was measured before this fix existed — both runs it compared were
equally unverified, so it was never actually a test of "reproducible under
verified-locked clocks" the way the Nano's own 10/10 was. Two fresh full 5-bucket
sweeps were taken under the corrected protocol (mask precondition confirmed
`(0,0)` at session start both times, achieved frequency verified every cell, no
discards): **10/10 within 3%**, max deviation 2.20% (`n197_fp16`), most cells under
1%. `out/lut/reproduction_gate_orinnx_verified_clocks.csv` +
`latency_lut_reprogate_verified_{a,b}.json`. Item 14's original 9/10 result is
retained as-is — it is the honest record of what that specific, now-superseded
protocol produced, not something to be overwritten — but as of this entry, this
device's reproduction gate, run properly, matches the Nano's own 10/10 precedent.

**What this episode demonstrates about the Nano campaign, for the record, since
the question was raised directly.** The Nano campaign's own protocol was a
*manual*, disciplined habit — re-lock clocks after every reboot, warm up before
measuring, take the median over the warmed-up run — carried out by the operator
every session, not enforced by `measure_lut.py` itself (`jetson/README.md:49-50`
documents it as a manual step; `jetson/`'s own script has no per-cell or
session-start verification of it at all, and never needed one — "no jetson/
precedent existed... Nano never needed this," item 10's own words). That protocol
worked because the Nano campaign ran in effectively one continuous session with no
mid-campaign reboots. This campaign inherited the same manual habit but ran across
many sessions over several days; the item-10 addition was the right instinct
(automate what the Nano did by hand, since this campaign couldn't rely on
session-continuity) but its implementation checked the wrong sysfs fields, so it
never actually enforced anything. That implementation gap is now closed. The mask
instability is a second, independent device property neither campaign's protocol
ever had reason to check for — the Nano was never audited this deeply, not because
its numbers are provably free of the same risk, but because no comparable
verification pass has been run against it. That is a limitation of this
comparison, stated plainly, not evidence of a problem on the Nano side.
