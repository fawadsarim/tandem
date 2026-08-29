# Fresh setup: Jetson Orin Nano Super, from an unopened box to SSH access

Complete, from-zero hardware + OS bring-up guide for the Jetson Orin Nano
Super Developer Kit, written for this project's specific situation:
**operator uses a Windows laptop, no separate Ubuntu machine, no monitor
attached permanently** — and written to be followed by anyone with zero
prior Jetson experience (this doc doubles as the training material the
project needs later, per its own purpose).

This file covers **hardware + OS + network access only** (unbox ->
flashed -> booted -> reachable over SSH from your laptop). Once you can
SSH in, continue with `README.md`'s "1. Day-0 bring-up" section (jtop,
MAXN SUPER power mode, the toy `trtexec` run, thermal check, `dd`
backup) — that section assumes you already have a shell on the device,
which is exactly where this file ends.

*(Note on the filename: written as `FRESH_SETUP_JETSON.md` — "Jetson,"
correcting what looked like a typo in the request. Same file, just the
standard spelling, since this doubles as reference material for others.)*

---

## 0. Decisions made for you, and why (read this before starting)

A few open questions this project needed answered — decided here so you
don't have to re-derive them:

**NVMe as the boot/working drive.** The Jetson Orin Nano Developer Kit
has no built-in eMMC — it boots from either a microSD card or the NVMe
SSD in the carrier board's M.2 slot. An SD card is the slower option and
wears out under sustained read/write, which matters directly for this
project: `build_engines.sh` and `measure_lut.py` do a lot of repeated
file I/O (engine builds, per-bucket logs, LUT writes). The NVMe (your
Lexar NM620) is faster and far more durable — the right choice for
anything that resembles real deployment, not just a one-off lab run.
**The OS gets installed onto the NVMe SSD; the SD card is never used as
the Jetson's own storage in this guide.**

**USB flash drive as the installer medium — this is what NVIDIA's own
Quick Start guide actually specifies and tests, so it's what this guide
follows as primary.** An SD card + USB reader *can* work for this same
one-time role (SD readers generally present as ordinary bootable USB
mass storage), but it's not the tested/documented path, and the
boot-device picker (step 4.4) is the wrong place to be debugging an
unusual device on a thesis timeline. Since you have a real USB flash
drive, that's what step 3 below uses; the SD card + reader is kept as a
**documented fallback** (step 3, alternate path) in case the USB drive
ever isn't available — worth keeping in the guide since this doubles as
training material and a future reader may not have a spare USB drive on
hand the way you do now.

**SSH from your laptop, not a permanently attached monitor.** You asked
this directly, so the direct answer: **SSH.** A monitor is used exactly
**once**, for a few minutes, to complete the very first Ubuntu account
setup screen (step 4) — after that, everything in this project
(`setup.sh`, `export_segments.py`'s copy step, `build_engines.sh`,
`measure_lut.py`, `replan.py`, `paired_eval.py`, `self_test.py`) runs
over SSH from your Windows laptop, with no monitor attached. This is
also the right shape for "deploy it somewhere later" — a unit sitting in
a rack, a robot, or a kiosk isn't going to have a monitor plugged in
either; operating it headless from day one, rather than only at
deployment time, is the realistic rehearsal for that.

**What you need that isn't in your current inventory:** nothing — you
have all of it (NVMe SSD, USB flash drive, SD card + reader as backup,
Windows laptop). You do need a working internet connection on the
Windows laptop and, later, either Wi-Fi credentials or an Ethernet cable
for the Jetson itself.

**A note on precision, read once:** every step below was checked against
NVIDIA's own current Developer Kit documentation before being written
here (not guessed) — but Jetson's exact installer/JetPack version number
changes over time, and some low-level pin numbers appeared inconsistently
even across NVIDIA's own current pages during that check. Anywhere that
matters, this guide tells you explicitly to confirm against the live
page rather than silently trusting a number that might have drifted —
look for the **"VERIFY LIVE"** callouts.

---

## 1. Inventory check + unboxing

From the box: the Jetson Orin Nano Super Developer Kit module+carrier
board, a 19V power supply, and a quick-start card. You're adding: the
Lexar NM620 NVMe SSD (target storage), a USB flash drive, 16GB or larger
(installer medium), a microSD card + reader (kept as a backup installer
medium, see step 3), and your Windows laptop.

Before opening the module packaging: touch a grounded metal surface (or
use an anti-static wrist strap if you have one) — the module and NVMe
SSD are static-sensitive, same handling care as any bare PC component.

---

## 2. Physical assembly — install the NVMe SSD

1. **Power must be disconnected** for this whole step.
2. Flip the Developer Kit over. The carrier board has an M.2 Key-M slot
   on the underside.
3. Insert the Lexar NM620 into the M.2 slot at roughly a 30-degree
   angle, then press it down flat and secure it with the small screw
   the kit ships with (in the accessory bag next to the board).
4. Flip the board back over. Leave the microSD slot empty — this guide's
   primary path doesn't use the SD card at all in the Jetson itself.

---

## 3. Build the installer USB on your Windows laptop

**Primary path — USB flash drive (what NVIDIA documents and tests):**

1. Install **balenaEtcher** on the Windows laptop:
   https://etcher.balena.io/#download-etcher (Windows installer, plain
   click-through install).
2. Download the **Jetson ISO** installer image for the Orin Nano
   Developer Kit from NVIDIA's official Jetson Orin Nano Developer Kit
   "Get Started" / Quick Start page. **VERIFY LIVE**: get the current
   download link and current JetPack version number directly from
   https://developer.nvidia.com/embedded/dlc/Jetson_Nano_Developer_Kit_User_Guide
   or https://nvidia.com/JetsonNano-Start
   https://developer.nvidia.com/embedded/jetpack/downloads
   at the time you actually do this — don't reuse a version number from
   an old guide. This project's own scripts were built against and
   tested on **JetPack 7.2** specifically (several real JetPack-7.2-only
   deviations are documented throughout `README.md`, e.g. the MAXN SUPER
   power-mode bug in section 1 and the `torch` wheel note in section 2)
   — if the Orin Nano Developer Kit's current-recommended release has
   moved past 7.2 by the time you do this, land on the current release
   and treat this project's JetPack-7.2-specific notes as "what to
   double-check first" rather than an exact version requirement.
3. Plug the USB flash drive into the laptop. Confirm Windows shows it as
   a removable drive, and back up anything on it — this step erases it.
4. Open balenaEtcher: "Flash from file" -> select the downloaded Jetson
   ISO. "Select target" -> pick the USB flash drive (**not** any other
   drive on your laptop — double-check the drive letter/size shown
   matches your USB drive before clicking Flash). Click **Flash**.
5. When Etcher finishes, safely eject the USB drive from Windows but
   leave it plugged in — you'll boot from it directly in step 4 below.

At the end of this step: the USB drive holds a bootable Jetson ISO
*installer*, not the final OS — that installer will ask, during boot,
which physical device to actually install onto (you'll pick NVMe).

**Alternate path — SD card + reader** (only if the USB drive isn't
available): same steps, but insert the microSD card into the reader and
select the reader's drive as Etcher's target in step 4 instead of the
USB drive. Whichever medium you used, plug that same device into the
Jetson in step 4 of the next section (a Jetson USB-A port for the USB
drive, or the carrier board's own microSD slot for the SD card).

---

## 4. Boot the installer and install to NVMe

1. Plug the installer USB drive into one of the Jetson's USB-A ports.
   (Alternate path only: if you used the SD card instead, insert it into
   the carrier board's microSD slot on the underside of the module
   instead of using a USB port.)
2. Connect a display (any HDMI/DP monitor) and a USB keyboard to the
   Jetson for this step only — the installer's boot-device picker and
   the one-time "Y" confirmation below are not things to do blind.
3. Connect the 19V power supply. The Developer Kit powers on
   automatically (no power button to press).
4. Watch for the NVIDIA logo splash screen and **press Esc** immediately
   when it appears, to enter the boot menu.
5. Select **Boot Manager**, then select your installer USB drive (or SD
   card, if using the alternate path) from the list.
6. The installer will show a firmware-update confirmation prompt.
   **Press `Y`** — this prompt times out after about 30 seconds and is
   easy to miss if you're not watching for it.
7. The installer proceeds automatically (BSP install, boot to the
   Jetson ISO installer UI). When it asks for the **target storage
   device**, select the **NVMe SSD** — not the installer drive itself
   (USB or SD card), which is the install *source*, not the destination.
8. Let it run to completion. This step takes real time (order of
   10-20 minutes, mostly unattended) — don't unplug power during it.

**VERIFY LIVE**: if your device's shipped firmware is older than the
Jetson ISO installer expects, the installer will tell you to complete a
one-time firmware "Update Path" first before it will proceed (named
after whatever JetPack version the installer you downloaded in step 3.2
actually is — don't assume it says "6.x", that was true when this note
was first written but the current installer will name its own current
version). NVIDIA documents this bootstrap on the same Get-Started page
as step 3.2 above; it only applies if the installer explicitly stops
and asks for it, so don't pre-emptively do anything here unless you see
that prompt.

---

## 5. First boot — the one and only time you need the monitor

When installation finishes, the Jetson reboots from the NVMe SSD. Keep
the monitor+keyboard connected for this one screen:

1. Accept the NVIDIA Jetson software EULA.
2. Set language, keyboard layout, timezone.
3. Connect to your network — **Ethernet is the more reliable choice for
   this project** (build_engines.sh/measure_lut.py do enough I/O and
   run long enough that you don't want Wi-Fi flakiness in the loop);
   Wi-Fi works too if that's what you have available.
4. Create a username, password, and a **computer name (hostname)** —
   pick something short and memorable, e.g. `tandem-jetson-orinnx` — you'll
   use this name to reach it over SSH.
5. Let it finish reaching the desktop.

You can now remove the keyboard and monitor. You will not need them
again for anything in this project's normal operation — only if
something goes wrong badly enough that SSH itself is unreachable.

---

## 6. Connect over SSH from the Windows laptop

OpenSSH server ships enabled by default on Jetson's Ubuntu image — no
install step needed on the Jetson side.

1. On the Windows laptop, open PowerShell (Windows 10/11 ship an OpenSSH
   *client* built in — no install needed there either).
2. Try connecting by hostname first (mDNS/Avahi is on by default on
   Jetson's Ubuntu image, so `<hostname>.local` usually resolves on the
   same network without needing to hunt for an IP):
   ```
   ssh <your-username>@tandem-jetson-orinnx.local
   ```
   Replace `tandem-jetson-orinnx` with whatever computer name you set in step
   5.4. Accept the host-key prompt the first time (`yes`).
3. If `.local` resolution doesn't work on your network, find the IP
   address instead: briefly reconnect the monitor and run `hostname -I`
   on the Jetson, or check your router's admin page / connected-devices
   list for the hostname you chose. Then:
   ```
   ssh <your-username>@<the-ip-address>
   ```
4. You should land at a normal bash prompt on the Jetson. This is your
   working session for everything else in this project.

**Recommended, not required — SSH key auth** (saves you re-typing the
password every session, and is the right habit if this unit is ever
network-reachable beyond your own desk later):
```powershell
# on the Windows laptop, PowerShell, one-time
ssh-keygen -t ed25519          # accept the default path, set a passphrase or leave blank
ssh-copy-id <your-username>@tandem-jetson-orinnx.local
```
(If `ssh-copy-id` isn't available in your PowerShell, manually append
the contents of `~\.ssh\id_ed25519.pub` from the laptop to
`~/.ssh/authorized_keys` on the Jetson, over the password-based SSH
session you already have.)

---

## 7. Remote access when you're NOT on the same network

Everything in step 6 assumes your laptop and the Jetson are on the same
LAN — that stops working the moment you're not physically near it (a
different building, a different city, eventually "deployed somewhere").
The safe way to solve this is to **not expose SSH to the public internet
at all**, rather than the traditional port-forwarding approach.

**Recommended: Tailscale** (built on WireGuard, free for personal use).
It creates a private, encrypted network between your own devices, no
matter where each of them physically is, without opening any port on
your router.

1. Sign up at https://tailscale.com (an existing Google/Microsoft/GitHub
   account works — no separate password to manage).
2. Install it on the Windows laptop: download from tailscale.com,
   install, sign in.
3. Install it on the Jetson, over your current SSH session:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
   It prints a URL — open that once, in any browser, to authorize the
   Jetson into your account.
4. Both devices now have a stable Tailscale address and a friendly
   MagicDNS hostname that work identically whether you're on the same
   WiFi or on opposite sides of the world:
   ```
   ssh <your-username>@<tailscale-hostname>
   ```
   Same command as step 6, just a different hostname — nothing else
   about how you work changes.

**Why this is the safe choice, not just the convenient one**: the
Jetson's SSH port never becomes reachable from the public internet.
Only devices you've explicitly authorized into your own private
Tailscale network (your "tailnet") can even attempt a connection to it.
The traditional alternative — forward a port on your router to the
Jetson, pair it with a Dynamic DNS service for your home's changing
public IP — puts SSH directly on the open internet instead, where it
gets scanned and brute-forced by bots within hours of being exposed.
That route is real and some people do use it, but it needs real
hardening to be safe (key-only auth with password auth fully disabled,
fail2ban, a non-default port) just to reach the baseline Tailscale gives
you automatically — and it needs router configuration you may not have
easy access to anyway (relevant here specifically, since this router
doesn't even expose static-IP/DHCP-reservation settings cleanly).

The SSH key auth from step 6 stacks directly on top of this — Tailscale
handles *reachability*, your key still handles *authentication*, same as
before.

---

## 8. Clone the repo and confirm you're in business

Still over SSH, on the Jetson:
```bash
git clone https://github.com/fawadsarim/tandem.git    # private repo -- use a token or SSH key, same as any other clone
cd tandem/jetson-orinnx
python3 self_test.py
```
This is the same static self-test that already runs clean on a
GPU-less host — on the real Jetson it will additionally pick up
`is_jetson-orinnx()==True` and start exercising the hardware-aware checks
(item 3's `jetson_clocks` check, in particular) instead of reporting
"N/A on this host." A clean run here confirms: git access works, Python
3 is present, and the port-equivalence gate (item 10) still passes for
real on this machine — a good checkpoint before spending any more setup
time.

---

## 9. What's next

Continue with `README.md`'s **"1. Day-0 bring-up"** section from here —
items 2 through 6 (install `jetson-stats`/`jtop` + reboot, set MAXN
SUPER power mode + `jetson_clocks`, one toy `trtexec` run, confirm <80C
under load, `dd` backup of this working image) — all run over the same
SSH session you just opened. That section also specifies the exact
paste-back block (L4T/TensorRT/CUDA/torch versions + `nvpmodel -q` +
toy median latency) to record once you're through it.

After that: `bash setup.sh` (from `jetson-orinnx/`, still over SSH) is the next
real step — see `README.md`'s "2. Harness usage" for the full step
order from there.

---

## 10. If something goes wrong

- **Installer doesn't show the firmware-update prompt / hangs on boot
  select**: re-check step 4.4-4.6 — the Esc-press window and the Y-press
  window are both short; a missed one usually just means reboot and
  retry from step 4.3, nothing is damaged by a false start here.
- **`.local` hostname doesn't resolve from the Windows laptop**: fall
  back to the IP-address method in step 6.3. Some corporate/university
  Wi-Fi networks block mDNS between devices even on the same SSID.
- **SSH connects but the Jetson feels sluggish or drops connection
  under load**: worth confirming you're actually booted from the NVMe
  and not accidentally still on the SD card — `lsblk` on the Jetson,
  confirm `/` is mounted on an `nvme0n1p*` device, not `mmcblk0p*`.
- **Anything else physical/firmware-level**: this guide's steps 3-5 were
  checked against NVIDIA's own current Developer Kit docs, but hardware
  bring-up is exactly where a point-release can shift a detail after
  this was written — NVIDIA's own forums
  (https://forums.developer.nvidia.com, Jetson Orin Nano category) are
  the reliable place to search the exact error text if you hit
  something this guide doesn't cover.
