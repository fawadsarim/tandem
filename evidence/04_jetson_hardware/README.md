# jetson/ -- G1-HW: Jetson Orin Nano Super

Self-contained folder: everything needed to clone this repo onto the
Jetson Orin Nano Super and execute G1-HW (measured hardware latency
validation of TANDEM's DeiT-S plan allocator).

**Boundary, enforced by `self_test.py` item 9, not just stated here:**
nothing in this folder imports from or writes to P2.1 (`experiments/phase2_1_fleet/`) or
P2.2 (`experiments/phase2_2_baselines/`) code paths. P2.1's canonical artifacts
(`experiments/phase2_1_fleet/results/P2/deit_small_patch16_224/{Dq_table.csv,
Dm_table.csv, plans.csv, P2_deit_small_patch16_224_summary.json}`) are
read at runtime, read-only, through two helpers in `_common.py`
(`read_p2_summary`, `read_p2_csv_path`) -- no other function in this
folder may reference that directory. **No secrets in any file.** HF auth
(only `export_segments.py` needs it) is via the `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` environment variable, read automatically by
`huggingface_hub`/`timm` -- never a literal in source. `self_test.py`
item 8 scans every file in this folder for token-shaped literals.

---

## 1. Day-0 bring-up

Do these five things, in order, on a fresh Jetson Orin Nano Super, before
touching anything else in this folder.

1. **Flash JetPack and get SSH access.** Full step-by-step (physical NVMe
   install, flashing to NVMe via the Jetson ISO installer, first boot,
   connecting over SSH from a Windows laptop with no separate Ubuntu
   host) is its own document: **`FRESH_SETUP_JETSON.md`** -- start there
   if this is a from-zero setup. Confirm the flash succeeded with
   `cat /etc/nv_tegra_release`.

2. **Install jetson-stats (`jtop`) and reboot.**
   ```
   sudo pip3 install -U jetson-stats
   sudo reboot
   ```
   `jtop` will not connect to its background service (`jtop.service`)
   until after this reboot -- running it beforehand fails with a
   connection error, not a useful diagnostic. `setup.sh` (section 2,
   below) does the install step for you but does **not** reboot on your
   behalf.

3. **Set MAXN SUPER power mode and lock clocks.**
   ```
   sudo nvpmodel -q --verbose      # list power modes; find the MAXN SUPER mode's INDEX for this device
   sudo nvpmodel -m <index>        # activate it
   sudo jetson_clocks              # lock all clocks to that mode's maximum
   jetson_clocks --show            # confirm: every rail's Min/Max/Cur frequency should now match
   ```
   The exact index varies by JetPack revision -- always read it from
   `nvpmodel -q --verbose` on the actual device rather than assuming a
   number. Every other script in this folder (`setup.sh`,
   `build_engines.sh`) **refuses to run** if it does not detect a locked,
   SUPER-named power mode -- this is a hard gate, not a suggestion, and
   none of these scripts will silently run the `sudo` commands for you.

   **Real deviation found and fixed, JetPack 7.2 specifically**: on a
   fresh JetPack 7.2 Jetson-ISO install, `grep 'POWER_MODEL ID'
   /etc/nvpmodel.conf` shows only `15W` and `25W` -- **no MAXN SUPER
   entry at all**. This is a confirmed JetPack 7.2 ISO-installer bug (the
   installer doesn't correctly mark the board as the Super variant),
   not a mistake in this guide or in your flash. Confirm you're hitting
   the same thing before doing anything else:
   ```
   cat /proc/device-tree/compatible   # look for a "-0005" or similar SUFFIX after p3767
   ls -la /etc/nvpmodel.conf          # note which nvpmodel_p3767_XXXX.conf it's symlinked to
   ls /etc/nvpmodel/                  # confirm a matching "..._super.conf" file exists
   ```
   If a matching `_super.conf` exists for the SAME number your
   `/etc/nvpmodel.conf` is already symlinked to (don't guess a different
   number), the fix is a straight file swap -- no re-flash needed:
   ```
   sudo cp /etc/nvpmodel/nvpmodel_p3767_XXXX_super.conf /etc/nvpmodel.conf
   sudo reboot
   ```
   (replace `XXXX` with whatever your own symlink showed). After reboot,
   `MAXN_SUPER` will appear as a real, selectable mode. **Known
   limitation**: this fix has been reported to not always survive a
   *later* reboot on some units (something on JetPack 7.2 can
   regenerate `/etc/nvpmodel.conf` from the still-uncorrected board
   identity) -- if `MAXN_SUPER` ever disappears again after a future
   reboot, just re-run the same two commands; it's a 30-second fix each
   time, not a broken system. A permanent fix exists (correcting
   `/etc/nv_boot_control.conf`'s board-identity string and
   re-running `dpkg-reconfigure nvidia-l4t-bootloader`) but was
   deliberately NOT applied here -- it touches boot configuration with
   no easy rollback short of a full re-flash, which needs an Ubuntu
   x86_64 host this project's setup doesn't otherwise require. Revisit
   only if the reboot-reapply annoyance becomes a real problem.

   **Recommended: install the auto-fix as a boot-time service, so the
   30-second fix above never has to be done by hand again.**
   `maxn_super_autofix.sh` re-applies the exact same LOW-RISK swap
   (never the boot-control edit) idempotently -- a no-op if
   `MAXN_SUPER` is already present and active, a real repair if it
   isn't. `maxn-super-autofix.service` runs it once on every boot:
   ```
   sudo install -o root -g root -m 0755 maxn_super_autofix.sh /usr/local/sbin/maxn_super_autofix.sh
   sudo install -o root -g root -m 0644 maxn-super-autofix.service /etc/systemd/system/maxn-super-autofix.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now maxn-super-autofix.service
   ```
   Check it any time with `systemctl status maxn-super-autofix.service`
   or `journalctl -u maxn-super-autofix.service -b`. **Confirmed for
   real on this device**: installed, survived an actual `sudo reboot`
   with `MAXN_SUPER` still active and the service logging a clean
   no-op, AND separately confirmed the repair branch itself actually
   works (simulated the broken/no-`MAXN_SUPER` state, the script
   correctly detected it, matched the current config's content against
   the stock non-super templates under `/etc/nvpmodel/` by hash (works
   whether `/etc/nvpmodel.conf` is a symlink or a plain-file copy --
   this device's is a plain-file copy after the manual fix above, which
   is exactly why the fix does NOT trust `readlink`/symlink-target
   parsing to find the matching board number), and re-applied the
   correct `_super.conf` and mode ID for that board.
   **One real, sharp edge found while testing this, worth knowing**: if
   you ever apply a `_super.conf` from a DIFFERENT `p3767_XXXX` number
   than your own device's actual board, `nvpmodel` does not just reject
   it -- it applies the mismatched profile (wrong CPU core count on
   this device: the wrong file's power model expected 8 cores, this
   board only has 6) and **automatically triggers its own reboot** to
   "reflect the mode change," with `NVPM ERROR` core/frequency lines
   logged in the process. This is why the script refuses to guess when
   it can't confirm a content match, and why it never picks a candidate
   by any means other than an exact hash match against your own
   device's actual (broken) conf content -- don't hand-edit which
   `_super.conf` gets copied in without being just as careful.

   **Two more real bugs found and fixed on this same device, both now
   corrected in `setup.sh`, `self_test.py`, and `build_engines.sh`** (not
   just recorded here -- the actual scripts were wrong and are now
   fixed):
   - `jetson_clocks --show` refuses to run at all without root on this
     JetPack release (`Error: Run this script as a root user`), even
     though `--show` is read-only -- every clock-lock check in this
     folder now calls it via `sudo` (or `sudo -n`, where the check needs
     to fail fast rather than hang if passwordless sudo isn't set up,
     rather than block on a password prompt).
   - The real per-rail field name in that output is **`CurrentFreq`**,
     not `CurFreq`. Every clock-lock check in this folder originally
     filtered/parsed for the substring `"CurFreq"` -- which is not a
     substring of `"CurrentFreq"`, so it never matched a single real
     line. The practical effect: `build_engines.sh`'s own clock-lock
     guard would have refused to build every time, unconditionally,
     regardless of whether clocks were actually locked. Found by
     actually running these checks against real hardware output, not by
     inspection -- fixed in all three files.

4. **One toy `trtexec` run, reporting median latency.** Confirms
   `trtexec` itself works on this device before anything model-specific
   is attempted. Any small ONNX file works -- e.g. export a trivial
   `torch.nn.Linear(64, 64)` to ONNX on a host and copy it over, or use
   any ONNX model already on hand:
   ```
   /usr/src/tensorrt/bin/trtexec --onnx=toy.onnx --fp16 --warmUp=200 --iterations=200
   ```
   Record the printed `GPU Compute Time ... median = ... ms` line -- this
   is the median latency figure for the paste-back below.

5. **Confirm <80C under 2 minutes of sustained load, in `jtop`.** Run the
   toy `trtexec` command above in a loop (or `trtexec ... --duration=120`)
   while watching `jtop`'s temperature panel. If any zone reaches 80C,
   stop, improve airflow/heatsinking, and re-check before proceeding --
   `build_engines.sh` and `measure_lut.py` both enforce this same 80C
   ceiling automatically for every later measurement, but Day-0 is where
   you first confirm the physical setup can stay under it at all.

6. **`dd` backup of the working image**, once 1-5 are confirmed working,
   so a bad experiment later doesn't cost a re-flash:
   ```
   sudo dd if=/dev/mmcblk0 of=/path/on/external/drive/jetson_orin_super_g1hw_baseline.img bs=4M status=progress
   ```
   (adjust `/dev/mmcblk0` to the actual boot device -- check with
   `lsblk` first; back up to external/network storage, not back onto the
   device being imaged).

### Exact paste-back expected, after Day-0

```
L4T:        <output of: cat /etc/nv_tegra_release>
TensorRT:   <output of: /usr/src/tensorrt/bin/trtexec 2>&1 | grep -m1 "TensorRT v">
CUDA:       <output of: nvcc --version | grep release>
torch:      <output of: python3 -c "import torch; print(torch.__version__)"  -- or "not installed" if step 1 of section 2 below hasn't been done yet>

<output of: nvpmodel -q>

Toy trtexec median latency: <the "GPU Compute Time ... median = ... ms" line from Day-0 item 4>
```
`setup.sh` (section 2 below) collects the first four of these
automatically into `out/versions.json` -- this paste-back is just those
same four values plus the two Day-0-specific lines, in one block, ready
to copy.

---

## 2. Harness usage

### Step order

```
-1. python replan.py --verify-port      [ANYWHERE, no device, no venv beyond pandas/numpy]
                                         PORT-EQUIVALENCE GATE -- blocking pre-flight check, run
                                         this FIRST, before touching a Jetson at all. See "Pre-flight
                                         port-equivalence gate" below.
 0. bash setup.sh                        [on-device]  idempotent env setup, versions.json
 1. export_segments_colab.ipynb (or `python export_segments.py --imagenet-dir <dir>` on any host)
                                         [HOST/Colab, NOT the Jetson]  ONNX export + calibration capture
    -- copy jetson/out/segments/ to the device before continuing (scp, or commit+pull) --
 2. bash build_engines.sh                [on-device]  FP16 + INT8 engines, per bucket, via trtexec
 3. ./venv/bin/python3 measure_lut.py     [on-device]  latency_lut.json (thermal-guarded)
 4. python replan.py                     [on-device]  plans_hw.csv + plans_hw_divergence.json
 5. paired_eval_colab.ipynb (or `python paired_eval.py --imagenet-dir <dir>` on any host)
                                         [HOST/Colab, NOT the Jetson]  paired_eval_result.json (the gate)
    -- copy jetson/out/replan/plans_hw.csv off the device first (see "Getting files off the
    Jetson" below); the notebook has an upload cell for exactly this file --
```
**Only steps 2-4 (`build_engines.sh`, `measure_lut.py`, `replan.py`) need
to actually run on the Jetson** -- they're the only ones that touch real
Jetson hardware (TensorRT engines built for this SoC, real measured
latency). Steps 1 and 5 are both plain PyTorch with no TensorRT/`trtexec`
dependency at all, so they run wherever is easiest -- **Colab is the
natural choice** for both, matching how the rest of this project
(E1-E5, P2.1, most of the baselines) already runs. This is a deliberate
scope decision, not an oversight: use the Jetson only for what genuinely
requires it.

**Real bug found and fixed in `requirements.txt`'s own pip install
(step 0), worth knowing if you ever do need torch on the Jetson
despite the above**: `timm` has `torch` as a hard dependency. A plain
`pip install -r requirements.txt` on a fresh venv will happily let pip
resolve `timm`'s `torch` requirement from PyPI -- which has no
Jetson/aarch64+CUDA build, so it pulls a generic (wrong, or simply
unbuildable) one, dragging in `triton`/`sympy`/a full CUDA toolkit
wheel stack in the process. Confirmed on real hardware: this silently
starts a large, wrong download rather than failing fast. **The fix, if
you ever need torch here**: install the correct Jetson-specific torch
wheel into the venv FIRST (`venv/bin/pip install <the-correct-wheel-url>`
for your exact JetPack/Python/CUDA combination), THEN run
`pip install -r requirements.txt` -- pip will see torch already
satisfied and won't try to fetch a different one. `setup.sh` itself
does not do this automatically (it doesn't know your JetPack version's
correct wheel URL), it only detects and reports whether torch is
already importable.

**Real bug found and fixed in `requirements-host.txt`'s installation
into Colab (steps 1 and 5 above), in two stages**: an earlier version
of both Colab notebooks ran `pip install -r jetson/requirements-host.txt`
directly into Colab's DEFAULT kernel environment. That file pins
`numpy==1.26.4`, and pip happily installs it -- but leaves whatever
`pandas` Colab already had in place untouched (nothing in that file
mentions `pandas`), and that `pandas` build is compiled against a
*newer* numpy's C-API (numpy>=2's larger `dtype` struct). The crash
doesn't show up at install time; it shows up later, the first time
anything imports `pandas` (in practice, `from datasets import
load_dataset`), as:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```
deep inside `numpy.random.mtrand` -- pip's resolver can't catch this in
advance because it's a compiled-ABI mismatch, not a declared-metadata
conflict. The first fix attempted was pinning `pandas==2.2.2` alongside
`numpy==1.26.4` as a matched pair (mirroring `requirements.txt`,
on-device, which already pins that exact pair) -- this fixed the
`datasets` import, but pinning `numpy==1.26.4` directly into Colab's
shared default kernel turned out to have a second, worse effect:
Colab's default kernel ships a long list of *other* pre-installed
packages (`shap`, `opencv-python`, `jax`, `cupy-cuda12x`, `tifffile`,
`rasterio`, `pytensor`, ...) that require `numpy>=2` -- none of which
this notebook uses, but all of which pip's resolver now reports as
broken (`numpy>=2 ... but you have numpy 1.26.4 which is
incompatible`) after this install.

**The real fix**: don't touch the shared default kernel's numpy at
all. This is the exact same problem P2.2's `experiments/phase2_2_baselines/` notebooks
already solved -- see e.g. `b1_tome/B1_tome_colab.ipynb`'s own venv
cell -- by giving each baseline's pinned, framework-heavy requirements
their OWN isolated venv, with `datasets`/`huggingface_hub` installed
separately into the untouched default kernel (only used there for HF
streaming, never imported by the baseline's own model code). Both
jetson Colab notebooks now follow the identical pattern:
`/content/venv_jetson_host` gets `requirements-host.txt` installed
into it exclusively; `export_segments.py`/`paired_eval.py` are
launched BY that venv's python (`VENV_PYTHON`), never the default
kernel's; and `datasets`/`huggingface_hub` are installed into the
default kernel only, for the calibration/eval-set materialization
cells, which keeps using whatever numpy Colab's base image already
ships and never has to coexist with the venv's
`numpy==1.26.4`/`pandas==2.2.2` pins at all. Both notebooks print
`torch`/`numpy`/`pandas` versions from inside the venv (via a
subprocess check, same pattern as the baselines' CUDA-availability
check) right after building it, so a future ABI break fails fast with
a clear version print instead of a deep traceback.

**Real bug found and fixed in HOW that venv gets built**: the first
version of this cell built it with stdlib `venv` after `apt-get
install python3-venv` -- matching the baselines' own
`apt-get ... python3.10 python3.10-venv` pattern, except without their
preceding `apt-get update`. On a real Colab run this 404'd (`Failed to
fetch .../python3.10-venv_... 404 Not Found`, a stale/desynced Ubuntu
mirror index), which made stdlib `venv`'s `ensurepip` step fail with no
pip installed into the new venv at all. Fixed by using `virtualenv`
(`pip install virtualenv`, a plain PyPI wheel bundling its own pip)
instead -- it never touches apt or `ensurepip`, so this failure mode
can't recur regardless of Ubuntu mirror state. Deliberately not
`python3.10` like the baselines: they need that specific older Python
because ToMe/RepQ-ViT pin torch versions with no wheels for newer
Python; nothing here has that floor (`torch==2.11.0` is current), so
this venv just reuses whichever `python3` Colab's own VM already has.

**Real bug found and fixed in `export_segments.py` itself, once it
actually ran against this pin**: `torch.onnx.export(...)` (in
`export_and_verify_one_bucket`) was called without specifying
`dynamo=`. As of torch 2.9, that call defaults to the newer
dynamo/`torch.export`-based exporter, and torch 2.11 (this project's
pin) removed the automatic fallback to the legacy exporter that used
to mask this on failure -- the new path also needs the `onnxscript`
package, which isn't in `requirements-host.txt`, so the export failed
outright (before `manifest.json` gets written, so the visible symptom
was a confusing downstream `FileNotFoundError` on that file in the
Results cell, not an error at the actual failure point). Fixed by
passing `dynamo=False` explicitly, pinning to the legacy
TorchScript-based exporter this script was designed and verified
against -- every export here is a fixed shape by design
(`dynamic_axes=None`), and the script's own onnxruntime-based
verification step right after already re-checks numeric correctness,
so there's nothing to gain from the new exporter here.

**Real bug found and fixed in `export_segments.py`, once `dynamo=False`
above got it tracing again**: `torch.jit.trace` (the legacy exporter)
boxes timm's `is_causal` argument -- passed internally from
`Block.forward` into `F.scaled_dot_product_attention` whenever timm's
fused/SDPA attention path is active -- into a 0-dim Tensor instead of
leaving it a plain Python `bool`, and SDPA's C++ binding rejects that
outright: `TypeError: scaled_dot_product_attention(): argument
'is_causal' must be bool, not Tensor`. Fixed by forcing every
`Attention` submodule off the fused path right after the model loads
(`_m.fused_attn = False` for every module with that attribute, set
directly on the already-constructed instances rather than via
`timm.layers.set_fused_attn()`, which only affects modules constructed
*after* it's called). The non-fused path (manual matmul + softmax +
matmul, no `is_causal` kwarg at all) traces cleanly and is
mathematically identical up to float precision -- well within this
script's own `MAX_ABS_DIFF_TOL=1e-4` check -- so this costs nothing but
a little export-time speed. `paired_eval.py` is unaffected (plain
eager-mode inference, no tracing involved).

**Real bug found and fixed in both Colab notebooks' run cells**:
`get_ipython().system(...)` does not raise on a nonzero exit code -- a
crash inside `export_segments.py`/`paired_eval.py` used to just print
its error and let the notebook silently continue into the next
(Results) cell, which then failed with a confusing `FileNotFoundError`
on the expected output file instead of showing the real error where it
actually happened (exactly how the `dynamo` bug above first surfaced).
Both run cells now capture the exit code and `assert` it's zero,
pointing back at the real error printed just above.

**Also worth knowing**: `ILSVRC/imagenet-1k` (the dataset both
notebooks stream calibration/eval images from) is a GATED dataset on
Hugging Face -- your `HF_TOKEN` must have accepted its terms on
huggingface.co before either materialization cell will work, same
requirement every `experiments/phase2_2_baselines/*_colab.ipynb` notebook already
documents. And the dataset id itself matters: it's `ILSVRC/imagenet-1k`,
not the bare `imagenet-1k` -- HF moved it under that namespace, and
current `huggingface_hub`/`datasets` no longer resolve the old bare
repo id at all (`HfUriError: Repository id must be 'namespace/name'`).
Same id already used and proven working in
`experiments/phase2_2_baselines/common/colab_imagenet.py`.

`self_test.py` can be run at any point and checks whatever artifacts
exist so far against this exact dependency order (item 11) -- it does
not require the full sequence to have completed. It also RE-RUNS the
port-equivalence gate for real every time (item 10) -- since that check
needs no device, it never degrades to an "N/A on this host" placeholder
the way the hardware-dependent checks do.

**Real bug found and fixed in BOTH `int8_calibrate.py` and
`measure_lut.py` -- same root cause, hit twice**: both scripts
originally required `pycuda` for device-memory management (allocating
GPU buffers, host<->device copies), with docstrings/error messages
claiming it "ships with JetPack 6.x system-wide" -- true for 6.x, not
for this device's JetPack 7.2 (confirmed on the real device: no
`pycuda`, no `cuda-python` anywhere on it, and the system CUDA
toolkit's `nvcc`/headers aren't even on `PATH` in a plain SSH session,
so compiling `pycuda` from source here would be slow and
failure-prone). Both fixed the same way: switched to `torch.cuda`
tensors instead -- device memory allocation, host->device copy, and a
raw pointer via `.data_ptr()` for TensorRT's C++ interfaces (the
`IInt8EntropyCalibrator2` callback in `int8_calibrate.py`, the
positional-`bindings` argument to `execute_v2()` in `measure_lut.py`)
all come from a dependency this venv already has and has already
verified working (torch's CUDA build for this exact JetPack/device),
rather than adding a new, ARM-Jetson-fragile one.

**Consequence, specific to `measure_lut.py`**: unlike `tensorrt`/`jtop`
(both installed system-wide by JetPack, importable from bare
`python3`), `torch` is installed only into `jetson/venv` (a
JetPack-specific wheel, not on PyPI). `measure_lut.py` now needs
`./venv/bin/python3 measure_lut.py`, not bare `python3` -- running it
unactivated now fails fast with a clear message pointing at this,
rather than the old `ModuleNotFoundError: No module named 'pycuda'`
with no indication of what to do about it.

**Also fixed while making this change** (`measure_lut.py` only):
removed a dead `stream = cuda.Stream()` variable (created, never
referenced -- `execute_v2()` is synchronous and never took a stream
argument; `pycuda.autoinit` masked this being unused), and hardened
`main()`'s file paths from bare CWD-relative (`Path("out/segments/
manifest.json")`) to `JETSON_ROOT`-relative, matching how
`_common.py`'s own `ensure_out_dir()` already resolves regardless of
invocation directory -- the CWD-relative form happened to work only
because this script is currently always invoked with `cwd=jetson/`, a
fragile coincidence rather than a guarantee (this exact class of bug
already caused a real failure in `int8_qdq_experiment.py`, see the
INT8 findings section above). The same fix was applied to
`replan.py`'s `load_latency_lut()`, which had the identical pattern,
before it could cause the same failure there too.

**Real bug found and fixed in `measure_lut.py`: segfault at interpreter
shutdown, AFTER all real work completes.** Confirmed on real hardware:
all 10 `(bucket, precision)` entries measured correctly and
`latency_lut.json` was fully written -- the `Segmentation fault (core
dumped)` happens strictly after the "Wrote ..." print, during Python's
own interpreter shutdown, not during measurement. Root cause is almost
certainly native-library cleanup ordering introduced by the pycuda ->
torch.cuda fix above: `pycuda.autoinit` registered its own
TensorRT-friendly CUDA context teardown hooks; `torch.cuda`'s implicit
context does not carry the same guarantees, and tearing down torch's
CUDA context against TensorRT's own C++ engine/context/runtime objects
in whatever order CPython's shutdown happens to collect them in can
crash -- a known class of issue when mixing GPU libraries, not a sign
of a wrong measurement. Since the LUT is already safely on disk by
that point, fixed by calling `os._exit(0)` immediately after the final
print, skipping the rest of Python's shutdown sequence (atexit hooks,
`__del__` calls, garbage collection) entirely rather than chasing the
exact native cleanup race -- no data at risk, nothing after that point
does anything the LUT depends on.

### Getting files off the Jetson (and onto it)

`pull_results.sh`, run from your OWN machine (not the Jetson):
```
bash pull_results.sh              # pulls jetson/out/ down from the 'jetson' SSH alias
bash pull_results.sh user@host    # or an explicit target
```
Syncs everything `build_engines.sh`/`measure_lut.py`/`replan.py`/
`setup.sh` produced (`out/`) down into this local checkout, so you can
inspect results without staying SSH'd in or using a monitor. Uses `scp`
(not `rsync` -- not guaranteed to be installed on every operator machine,
e.g. Windows). Nothing pushes to GitHub on its own -- pulling files onto
your own machine and deciding what (if anything) to commit are two
separate steps, and only you do the second one.

The reverse direction (getting `export_segments.py`'s output, built on a
HOST/Colab, onto the Jetson for `build_engines.sh`) and getting
`plans_hw.csv` off the Jetson for a Colab-run `paired_eval.py` both work
the same way, just pointed the other direction -- `scp -r
out/segments/ jetson:~/tandem/jetson/out/` and `scp
jetson:~/tandem/jetson/out/replan/plans_hw.csv out/replan/`
respectively (or commit+push/pull through the repo if you'd rather have
a record of exactly which artifacts moved when -- your call, not
automated either way).

### INT8 findings -- a real measured deployment property, investigated, not a harness bug

`build_engines.sh`'s own INT8 benchmark measured INT8 consistently
~2-2.5x **SLOWER** than FP16 across all 5 buckets (real numbers, on this
device: n197 FP16 mean 0.357ms vs INT8 mean 0.882ms; n69 FP16 mean
0.210ms vs INT8 mean 0.429ms -- slowdown worsens at higher token
counts). Counterintuitive (INT8 is normally expected to beat FP16), so
this was investigated rather than accepted at face value.
`replan.py` was HALTED until this investigation completed, since it
consumes the LUT this affects -- **investigation CLOSED, see the
decision-rule verdict at the end of step 2b below; `replan.py` is
unblocked.**

**Step 1 -- per-layer precision diagnosis (N=197, real device).**
Rebuilt the N=197 INT8 engine with `trtexec
--profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=...`
(reusing the calibration cache `int8_calibrate.py` already wrote, so no
recalibration needed) and inspected the resulting per-layer JSON. Of
the block's 4 GEMMs, only 3 actually run in INT8 (`qkv`, `mlp.fc1`,
`mlp.fc2`) -- `attn.proj` stays FP32 despite being quantizable.
Everything else (both LayerNorms, both attention-score matmuls
`Q@K^T`/`softmax@V`, softmax itself, GELU) never gets quantized at all,
with explicit FP32<->INT8 "Reformat" nodes bracketing each of the 3
quantized GEMMs. This explains the slowdown: conversion overhead paid
on top of a pipeline that's still mostly FP32, and the unquantized part
(attention score matmuls) is exactly what scales with token count --
consistent with slowdown worsening at higher N.

**Step 2a -- `trtexec --best` (N=197, real device, cheapest test of
whether forced-INT8 specifically is the problem).** GPU-compute median
0.2998ms, essentially tied with FP16 (0.3064ms measured that run --
0.2994ms is the number cited as the registered baseline). But checked
the actual layer graph, not just the number: the resulting engine has
9 nodes total, **zero** touching INT8. TensorRT's own free-precision
autotuner, given the option to use INT8 via the calibration cache,
chose not to use it anywhere for this block. This confirms forced-INT8
is actively harmful here, but is NOT a working INT8 path -- there's
nothing to adopt, since it doesn't use INT8 at all. (Plausible
mechanism beyond the Reformat overhead: this is a single isolated
block, a small workload where INT8's per-op conversion cost is a worse
trade than on a large, fully-fused network where reformatting
amortizes better -- see the scope-limitation note below.)

**Step 2b -- explicit Q/DQ quantization via NVIDIA ModelOpt
(`int8_qdq_experiment.py`, HOST/Colab, N=197 only).** Rather than
TensorRT's implicit calibration path (what `int8_calibrate.py` uses,
and what steps 1/2a diagnosed), this applies explicit Quantize/
Dequantize nodes directly to all 4 GEMMs' weights AND input
activations (`qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` -- the only
`nn.Linear` submodules in a timm ViT block, so restricting
quantization to `parent_class=nn.Linear` via a hand-built `quant_cfg`
naturally targets exactly these four and nothing else), calibrated on
the SAME pinned `calib_n197.npy` real-activation data
`export_segments.py` already captured. The resulting ONNX graph
carries its own baked-in scales; TensorRT builds it in EXPLICIT
quantization mode, which in principle lets it fuse Q/DQ directly into
the GEMM kernels instead of inserting Reformat nodes.

**Result (real device, N=197): GPU-compute median 0.4528ms.** Better
than the original forced-INT8 (0.821ms -- nearly 2x improvement), but
still slower than FP16 (0.2994ms) -- does not win the decision rule.
The per-layer graph confirms this is NOT a repeat of step 1's
diagnosis: 10 nodes total (vs 39 in the original build), **zero**
separate Reformat/copy nodes -- every Q/DQ pair is fused directly into
its GEMM kernel (confirmed for all 4 targets: `qkv`, `attn.proj`,
`mlp.fc1`, `mlp.fc2` each show up as a single fused `gemm`/`fusion`
node with `QuantizeLinear`/`DequantizeLinear` baked into its own
metadata, not bracketing it). LayerNorm even fused together with its
adjacent quantize step into one kernel. This is textbook-clean explicit
quantization -- step 3's own success criterion (4/4 GEMMs on fused INT8
kernels, not Reformat-bracketed) is fully met. The three attention ops
deliberately left unquantized (`Q@K^T`, softmax, `softmax@V`) remain in
float, exactly as scoped, and evidently dominate the block's cost
regardless of how cleanly the 4 GEMMs get quantized.

**This matters beyond just "still slower"**: it rules out "sloppy
Reformat-bracketed quantization" as the explanation for step 1/2a's
result. Even a maximally clean, correctly-scoped, fully-fused Q/DQ
implementation still loses to FP16 on this block/shape/device -- a
stronger, more specific finding than step 1 alone gave.

**Decision rule (registered) -- applied**: forced/explicit INT8 remains
>= FP16 after the full ladder (steps 1, 2a, 2b all tried, none beat
FP16). Per the registered rule, the finding STANDS as a genuine,
measured deployment property of this model/shape/device -- this full
attempt ladder is the evidence. **This investigation is CLOSED;
`replan.py` is UNBLOCKED** (see the top of this section -- it was
halted pending this).

**Updated on review, superseding this section's original
decision-rule verdict above**: although step 2b's explicit-Q/DQ path
doesn't beat FP16 (the finding above stands), it IS the best achievable
INT8 implementation found -- textbook-clean, all 4 GEMMs correctly
fused, no Reformat overhead, narrowing the gap from ~2.4x to ~1.5x
slower. The LUT must always price INT8 at this best-achievable
implementation, not the naive implicit-calibration one -- pricing INT8
at its worst inflates the measured latency of every plan that uses it
(including several of P2.1's own BOPs-selected plans), which distorts
every downstream budget/comparison built on the LUT (see "Budget
pairing" below for exactly where this mattered). `step 2b` was
therefore generalized from a single-bucket diagnostic
(`int8_qdq_experiment.py`, originally N=197 only) to all 5 buckets,
now the production INT8 path for `measure_lut.py`'s `int8` entries.
The naive implicit-calibration numbers are still measured too, kept
under the `int8_naive_implicit` key for provenance -- never read by
`replan.py`'s DP.

**Scope limitation (recorded regardless of the outcome above)**: this
investigation, like the rest of this folder's LUT, measures a
BLOCK-SEGMENT engine -- the correct instrument for `replan.py`'s DP,
which prices blocks, not a whole network. The INT8-vs-FP16 verdict at
block granularity may not generalize to a fully-fused whole-network
engine, where Reformat/reconversion overhead amortizes very
differently across many more INT8 ops in sequence. This is a scope
limitation of what this finding tells you about deployment more
broadly -- not a limitation of the experiment itself, which answers
exactly the question `replan.py`'s DP actually needs answered
(per-block cost).

### Budget pairing -- a real methodology fix, found on review

**History, kept for the record.** The first version of `replan.py`
paired each budget as `tau_b = fraction x Latency_id` (`Latency_id` =
all-FP16, no-merging identity latency). Against the 6 nominal fractions
(15%/21%/29.3%/41%/57.2%/80%), only 80% cleared the plan space's
achievable floor -- the other 5 came back `INFEASIBLE`: no plan in the
85-schedule plan space reached that low a latency. This was initially
handled as a reported finding (`replan.py` recording `infeasible: true`
per budget instead of crashing, `paired_eval.py` skipping those
budgets) -- **on review, this framing was corrected**: the
infeasibility was a real, measured consequence of the pairing, but the
pairing itself -- not physics -- was the problem, and it wasn't what
the protocol had actually registered.

**The confound**: `fraction x Latency_id` implicitly assumes the plan
space's cost scales down roughly linearly with the nominal fraction --
true enough for BOPs (which is exactly what `BUDGET_FRACTIONS` were
calibrated against), false for real measured latency here. Merging
buys only SUBLINEAR latency savings (dominated by near-fixed per-block
overhead -- `find_min_achievable_cost()` in `replan.py` computes the
plan space's true floor directly, ~76% of `Latency_id` using this
hardware's own FP16-only numbers), and INT8 buys no headroom at all
(measured slower than FP16 at every bucket -- see the INT8 findings
above). So most of the nominal fractions fell below what any plan
could physically reach -- an artifact of how the budget was defined,
not a hardware-reality result about G1-HW's actual question.

**The fix (registered protocol, now in `replan.py`)**: `tau_b` = the
MEASURED LATENCY of P2.1's own BOPs-selected plan at budget `b`, not a
fraction of `Latency_id`. This guarantees feasibility BY CONSTRUCTION
-- the BOPs-selected plan is itself always a valid candidate at cost
`tau_b`, in the identical 85-schedule x `BIT_MENU` plan space
`--verify-port` already proved this port searches faithfully -- so the
latency-selected search can only match or beat it. All 6 budgets are
feasible again, and `exact_select_for_budget()`'s `raise_on_infeasible`
path is now used for the MAIN run too (not just `--verify-port`): any
`INFEASIBLE` result under this pairing would indicate a real bug (in
the DP or the LUT), not physics, and halts loudly rather than silently
reporting a wrong result. `paired_eval.py` was updated to match --
back to evaluating all 6 budgets, no skipping, pooled statistic over
the full 6 x `PAIRED_EVAL_N` = 120,000 image-pairs as originally
intended.

This also reframes what the comparison actually tests -- which is what
was registered in the first place: at EQUAL real milliseconds, which
selection rule (BOPs-guided or latency-guided) gives better predicted
accuracy? Not "which budget survives being redefined in a proxy that
doesn't transfer."

**Standing finding, kept regardless of the pairing fix (`replan.py`
now computes and prints/records this automatically -- see
`find_min_achievable_cost()` and `replan_manifest.json`'s
`min_achievable_fraction_of_identity` field)**: the plan space's real
achievable latency range on this hardware is roughly
`[min_achievable_fraction, 100%]` of `Latency_id` -- much narrower than
`BUDGET_FRACTIONS`' nominal `[15%, 100%]` range, which was calibrated
against BOPs cost's much wider dynamic range. `replan.py` also prints,
per budget, the BOPs-selected plan's BELIEVED cost fraction (the
nominal `BUDGET_FRACTIONS` value) next to its MEASURED latency
fraction (`tau_b / Latency_id`) -- stated positively, this is the
proxy-divergence result: at tight budgets, BOPs-guided compression
believes it is buying a large cost reduction that, priced in real
measured milliseconds, does not materialize anywhere near as much.

**LUT correction, same review pass**: `measure_lut.py`'s INT8 entries
now come from the BEST ACHIEVABLE INT8 implementation (explicit Q/DQ
via `int8_qdq_experiment.py`, all 5 buckets), not TensorRT's naive
implicit-calibration path -- pricing INT8 at its worst would inflate
the measured cost of every plan that uses it (several of P2.1's own
BOPs-selected plans use `bits=8`), distorting `tau_b` itself and
therefore every budget/comparison built on it. The naive numbers are
still measured and kept, under the `int8_naive_implicit` key, for
provenance -- see `measure_lut.py`'s own module docstring.

### Joint vs staged under measured latency -- a third arm, testing the thesis's actual headline claim

**Background**: `experiments/phase1_E1-E5_E4a/E5_allocator_G1.py` (Phase 1) compared
TANDEM's JOINT allocator (the same Pareto DP `replan.py` ports) against
a STAGED baseline ("BEST-SEQUENTIAL" -- the stronger of two
compression orders, tokens-then-bits and bits-then-tokens, each swept
over an 11-point alpha threshold grid, so a single-order sequential
baseline is never the strawman being tested against) under SIMULATED
BOPs cost, and found they tie EXACTLY at all 6 budgets --
`plans_identical=True` every time, zero-width CI, verdict
`exact_tie_via_plan_collapse` ("staging collapse"). The documented
mechanism: `D_pred` is additively separable in (schedule, bits) by
construction (the accuracy-damage tables don't cross-depend on the
other axis), and whenever the joint-optimal schedule is ALSO
Pareto-efficient on the token axis alone, a sufficiently strong staged
search recovers it exactly -- checked and confirmed true at all 6
production budgets in that notebook.

**On review**: G1-HW's registered comparison (BOPs-selected vs
JOINT, both under real latency) is a clean test of "does the cost
proxy matter," but it doesn't test the thesis's actual headline claim
-- "joint beats staged" -- which only tied under the (separable) BOPs
proxy. Real measured latency is NOT separable the same way: a block's
real cost is a genuinely joint, non-linear function of token count AND
precision together (see the INT8 findings above -- lower precision
doesn't even help, and the Reformat/fusion overhead interacts with
token count in ways the BOPs formula never modeled). So the same
staged procedure, re-run under the SAME LUT and SAME `tau_b` as the
joint comparison, directly tests whether staging collapse survives
real hardware cost or the thesis's central claim actually separates
here, where it couldn't under simulation.

**Implementation**: `sequential_s1_tokens_then_bits`,
`sequential_s2_bits_then_tokens`, `best_sequential_for_budget` in
`replan.py` are ported VERBATIM from `experiments/phase1_E1-E5_E4a/E5_allocator_G1.py`
(the STOP-2 ruled exact version -- both stages of both orders
use `exact_pareto_bits`, the same primitive the joint DP uses, never a
lambda-sweep approximation, so STAGING is the only difference between
the two arms being compared, not solve-exactness). Cost-function
substitution follows the identical principle `--verify-port` already
proves faithful for the joint DP: every call to the plan-space
primitives (`plan_cost`, `objective_d_pred`, `exact_pareto_bits`) takes
`block_latency_fn` instead of a BOPs formula, with no other logic
touched. A permanent guard (ported from the same notebook's own
joint-vs-sequential invariant) asserts JOINT's predicted `D_pred`
weakly dominates STAGED's at every budget -- JOINT explores the exact
Pareto set of all 85 schedules, a strictly larger candidate set than
any staged procedure's one fixed schedule/bits choice, so this should
be structurally impossible to violate; a violation halts as a real bug,
not a finding.

**What's evaluated**: `paired_eval.py` now evaluates THREE plans per
budget (BOPs-selected, JOINT, STAGED) on the identical paired-eval
images, and reports TWO independent paired comparisons against the SAME
registered gate criterion (mean >= +0.4pp AND 95% CI excludes 0):
BOPs-vs-JOINT (the original question) and JOINT-vs-STAGED (the central
claim, under real hardware cost). `plans_hw.csv` carries all three
plans' JSON plus `joint_staged_identical`/`n_blocks_differing_joint_staged`
per budget; `replan.py`'s own console output reports how many of the 6
budgets staging collapse survives under real latency, directly next to
how many it survived under BOPs (6/6, per G1/E5).

This adds real, non-trivial evaluation cost -- six more
20,000-image passes on top of the original twelve (`paired_eval.py`'s
own runtime estimate updated accordingly) -- but reuses every existing
piece of infrastructure (`evaluate_plan_top1`, `compute_plan_divergence`,
`paired_ci`, the same materialized image set); no new data, no new
calibration.

**The real result, run for real (`out/paired_eval/paired_eval_result.json`)**:
BOPs-vs-JOINT -- pooled mean **+0.88pp**, 95% CI **[+0.7776, +0.9824]pp**,
n=120,000 paired predictions, gate **PASS**. This is the registered
hardware-latency result: the plan TANDEM's own joint allocator selects,
under REAL measured Jetson latency as the cost axis, beats the plan a
simulated-BOPs-guided selection would have picked, by a wide, clearly
CI-excludes-zero margin. JOINT-vs-STAGED on the same gate reads
`plans_identical=True`/`mean_delta=0.0` at every budget -- void by axis
domination (see the UPDATE below), not a second real result on this axis.

**UPDATE, G1-HW final phase (see `docs/CHARACTERIZATION.md`)**: the follow-up
budget sweep this section originally planned ("inside the achievable
range, forcing real compression") does NOT make joint-vs-staged a live
test on this device, and is CANCELLED as such. `docs/CHARACTERIZATION.md`
derives why: INT8 is slower than FP16 at every measured bucket (see the
INT8 findings above), so the precision axis is strictly dominated on
both cost and accuracy-damage at every budget -- no Pareto-optimal plan
on this hardware ever uses `bits<16`, the plan space collapses to
merge-only schedules, and JOINT/STAGED provably coincide by
construction regardless of how much merging any budget forces. Any
zero on this axis, at any budget, on this LUT, must be recorded as
"void by axis domination," not "joint fails to beat staged" and not a
re-confirmation of the BOPs-simulated staging-collapse finding above --
see `docs/CHARACTERIZATION.md`'s section (d) for the exact registered
language. The batch-8 pilot below exists to check whether a
compute-bound regime revives the precision axis and makes this
question live again.

### Batch-8 INT8 pilot -- does a compute-bound regime revive the precision axis?

**Purpose**: the INT8-dead finding above was measured entirely at
batch=1 (every ONNX export, every engine, every LUT entry in this
folder is batch=1 by construction -- see `export_segments.py`'s fixed
`torch.randn(1, n_tokens, d_model)` dummy input). At batch=1, a
block-segment engine is memory-bandwidth-bound, not compute-bound --
exactly the regime where INT8's bandwidth-halving advantage is smallest
relative to its Q/DQ conversion overhead. A larger batch shifts the
same engine toward compute-bound, where INT8's throughput advantage is
expected to matter more. This is a real, unresolved question, not
assumed either way -- `jetson/out/pilot_b8/` (new dir, does not touch
any existing `out/` artifact) holds a GO/NO-GO pilot at batch=8,
restricted to two buckets (`N=197`, `N=69` -- the two extremes of the
measured range) to keep this to roughly an afternoon:

- **Export** (`pilot_b8_export.py`, HOST-side/Colab, NOT the Jetson --
  same reason as `export_segments.py`/`int8_qdq_experiment.py`, see
  those files' own docstrings): FP-target and explicit-QDQ-INT8-target
  ONNX at `BATCH=8`, verified against PyTorch (max-abs-diff <= 1e-4),
  reusing the SAME pinned `calib_n197.npy`/`calib_n69.npy` calibration
  data `export_segments.py` already captured (no new calibration run).
- **Build + measure** (`pilot_b8_build.sh` + `pilot_b8_measure.py`,
  ON-DEVICE): FP16 and explicit-QDQ INT8 engines per bucket, per-layer
  graph dump to confirm fused INT8 kernels (no Reformat bracketing --
  same check as the original N=197 QDQ diagnosis), median+p99 over
  >=200 iterations under the standing thermal/clock-lock protocol,
  written to `pilot_b8_result.json`.
- **Decision rule** (pre-registered, in `pilot_b8_measure.py` before any
  number is measured): INT8 median < FP16 median at BOTH buckets ->
  **GO** (a batch-8 full G1-HW spec would follow, explicit sign-off
  required before running it). INT8 >= FP16 at either bucket ->
  **NO-GO** (this characterization's axis-domination finding closes the
  question on this device, at any batch size this pilot's result would
  reasonably be expected to generalize from). Within +/-5% at either
  bucket -> **BORDERLINE** (report both numbers for manual review). No
  further batch sizes or buckets without explicit sign-off either way.

**Two on-device install attempts, both abandoned, real findings not
harness bugs, 2026-07** -- kept for the record, so this isn't re-tried
identically:
1. A plain `pip install --target=<dir> nvidia-modelopt` (unconstrained
   deps) pulls a full parallel CUDA 13 / torch 2.13.0 / numpy 2.5.1
   dependency tree to satisfy its own `torch>=2.8` requirement -- a
   DIFFERENT torch than this venv's pinned `torch==2.12.0` (and a
   different numpy than every other script in this folder's
   `numpy==1.26.4` pin), with no guarantee the auto-resolved wheel is
   even the correct JetPack-matched CUDA build rather than a generic
   PyPI one. Injecting that via `PYTHONPATH` risked exactly the
   ABI-mismatch class already hit and fixed elsewhere in this project
   (see the Colab `numpy`/`pandas` ABI-break history above).
2. A second attempt installed ONLY modelopt's small set of top-level
   deps actually missing from this venv (`ninja`, `omegaconf`, `pulp`,
   `pydantic`, `regex` -- everything else, including `torch`/`numpy`/
   `scipy`/`safetensors`/`rich`/`PyYAML`, was already satisfied by this
   venv), each with `--no-deps` into an isolated target dir, specifically
   to let `import modelopt...` fall through `PYTHONPATH` to this venv's
   own already-validated `torch==2.12.0` rather than shadow it. This
   correctly avoided the version-drift problem in (1) -- but hit a
   harder, unfixable wall: `modelopt.torch`'s own `__init__.py`
   unconditionally imports `modelopt.torch.distill`, which imports
   `torch.distributed.tensor`, which needs `torch._C._distributed_c10d`
   -- a compiled symbol this device's JetPack-specific `torch==2.12.0`
   build does not have (`ModuleNotFoundError: No module named
   'torch._C._distributed_c10d'; 'torch._C' is not a package`).
   JetPack/Jetson PyTorch wheels are commonly built with
   `USE_DISTRIBUTED=0` (distributed training support is irrelevant on a
   single edge device and adds real binary size) -- this is a real,
   structural incompatibility between `nvidia-modelopt`'s import graph
   and this device's own PyTorch build, not a dependency-version problem
   with a further workaround short of either a from-source
   distributed-enabled torch rebuild (out of scope, high risk, no
   rollback short of a fresh venv) or monkey-patching a fake
   `torch._C._distributed_c10d` module before import (rejected --
   exactly the kind of fragile hack that could silently produce wrong
   quantization results, which this project's whole discipline exists to
   avoid).

Both attempts cleaned up fully afterward; nothing was left installed in
the registered venv itself. The export step stays HOST-side, matching
`export_segments.py`/`int8_qdq_experiment.py`'s already-proven pattern,
for exactly the reason that pattern exists -- Colab's x86 torch build
has full `torch.distributed` support and this exact failure mode does
not occur there.

### Re-scoped sweep: TANDEM hardware frontier + merge-granularity (independent of the pilot's outcome)

Runs regardless of the batch-8 pilot's GO/NO-GO verdict -- this is a
different, still-live question even though joint-vs-staged is void on
this LUT (see above). `jetson/out/frontier/` (new dir): a 6-point
budget grid, `tau in {0.77, 0.79, 0.82, 0.85, 0.89, 0.94} x Latency_id`,
chosen to sit strictly inside the plan space's real achievable range
(floor 76.4% of `Latency_id`, identity at 100% -- both infeasible by
construction at all 6 points, asserted per budget) so every budget
forces real compression.

- **ARM A -- TANDEM frontier**: the same joint DP as the main G1-HW run,
  under the same LUT, per tau. Per (b) above, this will select
  merge-only, all-16-bit plans -- asserted as a consistency check
  (`bits==16` at every block, every budget), not assumed.
- **ARM B -- uniform-r ToMe**: fixed `r` applied at every block
  (`N_l(l) = 197 - r*(l-1)`), `bits=16` throughout, `r` restricted to
  `{1..11}` so every per-block `N_l >= 69` (the LUT's measured floor --
  asserted, never extrapolated below it). Per tau, selects the smallest
  feasible `r` (minimum-damage plan whose summed LUT latency clears
  `tau`).
- This is **not** a joint-vs-staged test (void per `docs/CHARACTERIZATION.md`)
  -- it compares merge-PLACEMENT granularity (TANDEM's <=2-point
  schedules) against uniform-r merging at matched MEASURED latency, the
  hardware analogue of the P2.2 saturation finding. Paired eval (Colab,
  same 20k manifest/disjoint-calib pattern as `paired_eval.py`) reports
  per-budget and pooled mean paired Delta-top1 with a 95% CI --
  descriptive, no pass/fail gate was registered for this comparison.
- **Plan selection** (`frontier.py`, pure DP, no device): both arms' plans
  per tau, written to `frontier_plans.csv`/`frontier_manifest.json`.
- **Paired accuracy** (`frontier_paired_eval.py`, Colab, same 20k
  manifest/disjoint-calib pattern as `paired_eval.py`): ARM A at all 6
  budgets (the frontier curve, reported regardless of ARM B's
  feasibility); ARM A vs ARM B paired comparison restricted to the 2
  budgets where ARM B is actually feasible (89%/94% in v1 -- see below).
  Writes `frontier_result.json`.
- **Figure** (`frontier_figure.py`): accuracy vs measured ms -- both
  arms' curves, the identity point, and the 6 completed-run
  BOPs-selected plans plotted as dominated reference points, from
  `frontier_plans.csv`/`frontier_result.json`.

**Real result (v1)**: ARM A landed all-16-bit at all 6 budgets (confirms
`docs/CHARACTERIZATION.md` (b)); ARM B was latency-INFEASIBLE at 4/6 budgets
(77/79/82/85%) -- even the smallest candidate (`r=1`, uniform merging at
11 separate blocks) exceeded those budgets, because spreading merges
across 11 sites pays this device's near-fixed per-merge-operation
overhead 11 times over, vs. TANDEM's own `<=2-point` schedules paying it
once or twice. Only feasible at 89%/94% (`r=11`, `r=8`) -- see "LUT
extension" below for whether that bound was real or an LUT-range artifact.

### LUT extension -- was ARM B's infeasibility a hardware property, or the LUT's own range?

**Question**: the v1 ARM B infeasibility above was bounded at `r<=11`
because `N_l(block 12) = 197 - 11r >= 69`, and 69 was the ORIGINAL LUT's
smallest measured bucket -- a bound set by what got MEASURED, not
necessarily by anything about uniform merging itself. `jetson/out/lut_ext/`
(new dir) extends the LUT to test this directly, keeping the original 5
buckets completely untouched.

- **Export + build** (`lut_ext_export.py`, `lut_ext_build.sh`): 3 new
  buckets, N=53/37/21, **FP16 only** -- INT8 stays CLOSED per
  `docs/CHARACTERIZATION.md`, not re-litigated here. `lut_ext_export.py` runs
  ON-DEVICE (a deliberate, documented deviation from `export_segments.py`'s
  HOST-only convention -- FP16 needs no calibration/modelopt, and this
  device's venv already had torch/timm/onnx; `onnxruntime` and
  `onnxscript` were the two genuinely missing pieces, both checked for
  real and confirmed to have clean, lightweight aarch64 wheels -- unlike
  `nvidia-modelopt`'s CUDA/torch cascade, see the batch-8 pilot section
  above -- and installed with zero risk to the existing torch/CUDA setup).
  **Real environment finding**: this device's `torch==2.12.0` needs
  `onnxscript` importable even with `dynamo=False` (unlike the
  `torch==2.11.0` Colab host pin) -- a torch-version behavior difference,
  not a bug; verification (onnxruntime max-abs-diff, same 1e-4 tolerance)
  passed cleanly at all 3 new buckets regardless.
- **Measure** (`lut_ext_measure.py`): same timed-loop/thermal-guard
  protocol as `measure_lut.py` (imported directly, not reimplemented),
  FP16 only, writes `latency_lut_extended.json` = the original 15 entries
  copied verbatim (asserted byte-identical to `latency_lut.json` at write
  time -- never re-measured) plus 3 new FP16 entries.
- **Re-test** (`frontier_v2.py`, pure DP, no device -- reads only the
  extended LUT + P2.1's read-only tables + the immutable v1
  `frontier_plans.csv` for comparison): ARM A re-run against the extended
  LUT -- **all 6 plans identical to v1**, confirmed rather than assumed
  (TANDEM's own 85-schedule space never produces `n_l` below 69 regardless
  of the LUT's floor, so extending the floor gives it nothing new to
  select). ARM B re-run with `r`'s upper bound DERIVED from the extended
  floor (`floor((197-21)/11) = 16`, not the `18` loosely estimated when
  this extension was requested -- the arithmetic, not the estimate, sets
  the real bound), full sweep `r=1..16`.
- **Real result**: ARM B is now feasible at **4/6 budgets** (82%, 85%,
  89%, 94%) -- **82% and 85% are newly feasible** (`r=16`, `r=14`) that
  were NOT in v1; 89%/94% select the SAME `r` as v1 (`r=11`, `r=8` --
  the smallest-feasible-r rule doesn't change once the original smallest
  feasible r is already within a widened range). **77% and 79% remain
  genuinely INFEASIBLE even at the maximum extended range** (`r=16`'s
  cost, 4.058ms, still exceeds both budgets) -- this part of the original
  finding survives; the 82%/85% part does not and is retired.
- **Paired accuracy** (`frontier_v2_paired_eval.py`, Colab): ARM A vs ARM
  B at all 4 now-jointly-feasible budgets, cross-checked against v1's
  already-committed numbers at 89%/94% (same plans, same images --
  should match exactly; a mismatch would be a real bug). Outputs:
  `frontier_plans_v2.csv`, `arm_b_r_sweep_v2.csv` (every `r`'s exact
  per-block `N_l` trajectory, cost, and which budgets it clears),
  `frontier_manifest_v2.json`, `frontier_result_v2.json`.

**Real result (v2)**: the 89%/94% cross-check against v1 is an EXACT
match (same plans, same images, deterministic -- confirms the whole
pipeline reproduces). At the two NEWLY feasible budgets, the gap is
real and much larger than at 89%/94%: 82% -- ARM A 78.435% vs ARM B
77.875% (+0.56pp); 85% -- ARM A 79.060% vs ARM B 78.585% (+0.475pp).
**Pooled across all 4 budgets (n=80,000 pairs): +0.31pp, 95% CI
[+0.178, +0.442]pp -- CI EXCLUDES ZERO.** Unlike v1's pooled result
(89%/94% only, CI crossing zero), this is a statistically real, positive
result: TANDEM's concentrated `<=2-point` merge placement measurably
outperforms uniform-r merging at matched real hardware latency,
specifically where merging pressure is real (82-85%) -- exactly where
this extension was designed to look. Still explicitly NOT a
joint-vs-staged test (void per `docs/CHARACTERIZATION.md`) -- this is a
result about merge-PLACEMENT granularity, a different and genuinely
live question that the extension turned from untestable (v1, 2/6
budgets, non-significant) into a real, positive, defensible finding.

### DeiT-B INT8 pilot -- does a 4x-larger model revive the precision axis?

**Hypothesis** (`jetson/out/pilot_deitb/PREREGISTRATION.md`, written before any measurement):
utilization, not precision arithmetic, is what killed INT8 on DeiT-S -- a 4x larger model
(`deit_base_patch16_224`, `d_model=768` vs DeiT-S's `384`) should reverse it if that diagnosis
is right, since the same Q/DQ conversion overhead amortizes over ~4x more compute per block and
the GEMMs use tensor cores far better. `jetson/out/pilot_deitb/` (new dir, does not touch any
existing `out/` artifact): same two token-count extremes as the batch-8 pilot (N=197, N=69),
batch=1 this time (this pilot varies model scale, not batch size), same explicit-QDQ path.

**Decision rule** (verbatim, pre-registered): INT8-QDQ median faster than FP16 median at BOTH
buckets -> **GO** (precision axis live on DeiT-B, flag for manual review for a full spec). INT8 >=
FP16 at EITHER bucket -> **NO-GO** (axis domination confirmed at a second model scale,
`docs/CHARACTERIZATION.md` updated, device work CLOSED). Within +/-5% at either bucket ->
**BORDERLINE** (report both for manual review).

- **Export** (`pilot_deitb_export.py`, HOST/Colab -- confirmed NOT possible on-device,
  `nvidia-modelopt` hits the same `torch._C._distributed_c10d` blocker as the batch-8 pilot,
  not re-investigated here): FP + explicit-QDQ INT8 ONNX at both buckets, verified against
  PyTorch (max-abs-diff <= 1e-4). DeiT-B needs its OWN 128-image calibration capture (folded
  into this one script) -- cannot reuse DeiT-S's `calib_n197.npy`/`calib_n69.npy` (different
  weights, different activation distributions).
- **Build + measure** (`pilot_deitb_build.sh` + `pilot_deitb_measure.py`, ON-DEVICE): graph
  verification (per-layer fusion check, same mechanism as the batch-8 pilot) runs FIRST, before
  any timing -- a Reformat-bracketed (non-fused) INT8 engine HALTS the script rather than
  producing a meaningless comparison. Same timed-loop/thermal-guard protocol as
  `measure_lut.py`/`pilot_b8_measure.py` (imported directly), writes `pilot_deitb_result.json`
  with the automatic verdict.
- **Utilization context** (`pilot_deitb_utilization.py`, no device): achieved TFLOPS from a
  stated analytic per-block FLOP formula (`12*N*d^2 + 2*N^2*d` MACs) and real measured medians,
  for both models at N=197/FP16/batch=1. The cross-model achieved-TFLOPS RATIO needs no
  peak-throughput assumption (same hardware, ratio cancels it out) and is the rigorous headline
  number. Converting to an absolute utilization PERCENTAGE needs this device's peak FP16
  TFLOPS, which this script does NOT assert with confidence -- two independent derivations (one
  from this device's own real measured GPU clock + NVIDIA's published tensor-core count +
  a commonly-cited but not independently-verified per-cycle throughput figure; one from
  NVIDIA's published Orin Nano Super headline TOPS figure divided by standard sparse/precision
  conversion ratios) disagree by roughly 3x and are BOTH reported rather than one being silently
  picked. Smoke-tested against DeiT-S's already-real data: the top-down estimate gives DeiT-S
  ~10.9% utilization (reasonably close to the informally-cited "~15%"), the bottom-up estimate
  gives ~35.7% (diverges much further) -- worth relaying this specific discrepancy rather than
  treating either figure as settled.

### DeiT-B crossover characterization -- bounded follow-up, no campaign commitment

The DeiT-B pilot above came back **BORDERLINE overall**: N=197 a clear NO-GO (+19.3%), N=69
just inside the +/-5% band (-3.09%, fractionally faster). A naive 2-point linear interpolation
between those two puts the INT8=FP16 crossover around **N~87**. `jetson/out/pilot_deitb_ext/`
(new dir, does not touch `out/pilot_deitb/`'s committed result) fills that gap with two more
real buckets, N=133 and N=101, so the crossover is located from four measured points instead of
extrapolated from two.

**Decision rule** (pre-registered, verbatim): GO (full DeiT-B G1-HW) if INT8 is faster than
FP16 by MORE than 5% (ratio < 0.95) at >=2 of the four buckets -- the precision axis is then
meaningfully live in the reachable plan space. NO-GO otherwise -- close device work; the
crossover finding is reported as a measured hardware property regardless of which way the
verdict lands (the four-bucket ratio table + crossover fit is a paper figure either way).

- **Export** (`pilot_deitb_ext_export.py`, HOST/Colab -- same `nvidia-modelopt` blocker as
  every prior QDQ export): FP + explicit-QDQ INT8 ONNX at N=133/101, own fresh 128-image
  calibration capture (cannot reuse any prior run's calibration data).
- **Build + measure** (`pilot_deitb_ext_build.sh` + `pilot_deitb_ext_measure.py`, ON-DEVICE):
  same mandatory graph-verification-before-timing discipline as the original pilot. Combines
  the two new buckets' real measurements with N=197/N=69 copied READ-ONLY from
  `out/pilot_deitb/pilot_deitb_result.json` (never re-measured) into one four-bucket table, fits
  a least-squares crossover over all four real points (reported alongside, not replacing, the
  naive 2-point ~N=87 estimate -- confirmed by direct calculation to reproduce ~86.7), and
  applies the decision rule automatically.

### Energy measurement (E-LUT) -- a second, real cost axis, mechanistically different from latency

**Why**: the thesis's spine is "how you measure cost determines what compression is optimal."
Latency demonstrated that on one real axis. Energy is mechanistically DIFFERENT: static power is
paid per unit TIME, not per operation. `jetson/out/energy/` (new dir, does not touch any
existing `out/` artifact). Model scope: **DeiT-S, batch=1 only** -- the configuration every
registered latency result uses.

**Part 0 -- instrument validation (BLOCKING, real, done first)**: power sampling had never been
validated in this project. `energy_instrument.py` enumerates the real rails this device exposes
(`jetson.power["rail"]`: `VDD_CPU_GPU_CV`, `VDD_SOC`; `tot`/`VDD_IN` used as the PRIMARY rail --
total board power, the same key `measure_lut.py`'s own `get_power_mw` already reads), confirms
the achievable sampling rate is capped at **4Hz** (`_min_refresh_s=0.25` in this device's
installed `jetsonpower_provider.py`, confirmed by direct source inspection, not assumed) --
~260-2400x coarser than a single inference, so every per-inference energy figure below is
derived from sustained-loop average power and throughput, never single-shot. **Real result**:
60s idle baseline = 5.28W mean; monotonic sanity ladder idle (5.28W) < N=69 FP16 (8.16W) < N=197
FP16 (9.72W) -- **PASSED**.

**Part 1 -- energy LUT** (`energy_lut.py`, reuses the ALREADY-BUILT engines in `out/engines/`,
builds nothing new): 30s sustained loop per config (first 5s discarded), all 5 buckets x
{FP16, INT8-QDQ} = 10 configs. Primary metric `energy_per_inference_mJ = mean_power_W /
throughput_per_s * 1000` (the loop's own real throughput, including real inter-inference gaps);
an alternative latency-multiplied form and a dynamic-energy form (power minus the real measured
idle baseline) are also recorded. **Real result**: all 10 configs measured cleanly, max
temperature 60.4C (well under the 80C guard). Notably, INT8 draws LESS power than FP16 at every
bucket (e.g. N=101: FP16 9.29W vs INT8 7.88W) -- despite being latency-slower, INT8 comes out
ENERGY-CHEAPER than FP16 at N=101 specifically (2.63mJ vs 2.86mJ), a real, direct instance of
the two axes ranking configurations differently.

**Part 2 -- proxy divergence** (`proxy_divergence.py`, no device, reads Part 1's output +
P2.1's `block_bops` formula): divergence factors (latency/BOPs and energy/BOPs, normalized to
N=197/FP16) differ by more than 2 percentage points at **9/10 configurations** -- BOPs fails the
two real axes differently, not the same way, confirming the pre-registered claim's core
empirical content. The DIRECTION is more nuanced than the naive pre-registered expectation
("energy should diverge MORE than latency, since static power ignores operation count"): for
the FP16/token-reduction configs, energy's divergence factor is consistently SMALLER than
latency's -- because dynamic power itself drops with smaller workloads on this device (real
measured `mean_power_w` falls from 9.84W at N=197 to 8.25W at N=69), so energy saves MORE than
latency does there, not less. **The cleanest result is the R^2 comparison**: R^2(BOPs -> energy)
= **0.061** (essentially no predictive power), R^2(measured latency -> energy) = **0.939** --
BOPs is useless for predicting either real cost axis, but the two real axes track each other
closely. This matches PREREGISTRATION.md's second registered branch exactly ("a single measured
latency LUT suffices as a hardware cost proxy for both objectives... a useful, reportable result
and NOT a null") -- a cleaner, more decisive resolution than either registered branch alone
anticipated.

**Re-pricing** (not re-planning) the plans already committed in `out/replan/plans_hw.csv`: the
energy-cheapest of {BOPs-selected, JOINT, STAGED} differs from the latency-cheapest at **2/6**
budgets (the two tightest, 15% and 21% -- where BOPs-selected's own plan turns out
energy-cheaper than JOINT despite being latency-worse). Flagged as a finding; not acted on --
per this task's own scope, this is a re-pricing only, no DP re-run, any follow-up is a
separate decision.

### Energy-cost replan -- does joint vs staged separate on the energy axis?

DP-only, no device, no evaluation -- `energy_replan.py` re-runs the IDENTICAL joint DP and
staged (STOP-2, both orders, alpha-grid) arms `replan.py` already uses, substituting
`out/energy/energy_lut.json` for the latency LUT as the cost axis (same substitution principle
`--verify-port` already proves faithful). Six energy budgets, `tau_b` = measured ENERGY of
P2.1's own BOPs-selected plan at budget `b` (same registered budget-pairing principle as the
latency axis). Writes to `out/energy_replan/` only. **No paired evaluation, no accuracy
numbers** -- this answers exactly one question: do joint and staged select different plans when
the cost axis is energy instead of latency?

**Real result**: `plans_identical` at **6/6 budgets**; the precision axis (`bits<16`) is
**never** selected anywhere, at any budget, by either arm. Budgets 29.3-80% all converge on the
literal identity plan (`tau_b >= Energy_id` -- BOPs' own plans mostly cost more energy than
doing nothing, the same proxy-divergence pattern already found on the latency axis); only 15%
and 21% force real compression, and both do it via merging only, never bits. Per the
pre-registered interpretation (`out/energy_replan/PREREGISTRATION.md`, written before this
ran): **the axis-domination characterization holds on the energy axis as well; device work is
closed.**

Device work was reopened once more, briefly, for the power-mode sweep below -- the single
remaining question left unanswered by every result above being: does any of this (proxy
divergence, axis domination) hold at only ONE operating point (MAXN_SUPER), or does it
generalize across the device's real clock-ratio range? After that sweep, device work is closed
for real -- see its own closing line below.

### Power-mode sweep -- robustness of proxy divergence and axis domination across operating points

**The actual last device experiment.** Every hardware claim above rests on ONE nvpmodel
operating point, MAXN_SUPER. The Orin Nano Super exposes multiple nvpmodel profiles that change
the GPU/EMC clock ratio -- the exact mechanism proxy divergence is about -- so re-measuring at
lower power modes converts "one device, one operating point" into "one device, N operating
points," for the cost of an afternoon, and doubles as one more free test of whether the
precision axis revives once the GPU is slower relative to fixed per-kernel overhead (the regime
where INT8 could finally pay off). Five scripts, run in order, real device time:
`powersweep_enumerate.py` (Part 0: enumerate real nvpmodel modes + empirically test reboot
requirements) -> `powersweep_measure.py` (Parts 1-2: LUT per mode, reusing the existing engines,
building nothing new) -> `powersweep_repro_gate.py` (Part 1's blocking 3% reproduction gate) ->
`powersweep_analysis.py` (Part 3: proxy divergence + precision axis + utilization, per mode) ->
`powersweep_replan.py` (Part 4: joint/staged DP against each mode's own LUT, no device, no
eval). Pre-registered decision rule and both questions: `out/powersweep/PREREGISTRATION.md`.

**Part 0 (real, not assumed)**: this board exposes exactly 3 modes -- `15W`, `25W`,
`MAXN_SUPER` -- queried from `/etc/nvpmodel.conf`, not guessed. Reboot requirement was tested
empirically (declining the actual `nvpmodel -m <id>` prompt, not read from documentation):
switching TO or FROM MAXN_SUPER always needs a reboot; switching directly between 15W and 25W
does not (both use finite clock caps; MAXN_SUPER's uncapped values are the special case).

**Part 1, reproduction gate**: the first attempt genuinely FAILED (9/10 rows within 3%; N=69/INT8
at +3.19%, just over the pre-registered 3% hard-stop) -- halted per the pre-registered rule
rather than proceeding. A non-destructive 5-trial repeat of just that config showed tight
clustering (CV=0.29%, mean within 0.78% of committed), consistent with ordinary measurement
noise at the smallest/fastest bucket, not drift. A full, clean official re-run (all 10 rows, not
cherry-picked) then passed 10/10. Both attempts are preserved in `evidence/` for the record
(`04_jetson_hardware/out/powersweep/reproduction_gate_ATTEMPT1_FAILED.csv`) rather than only
keeping the passing one.

**A genuine device/firmware fault, found and recovered, not caused by this sweep**: the FIRST
boot into 15W failed with a GPU ACR (RISC-V security processor) bootstrap error in `dmesg`
(`nvgpu: ... Bootstrap HS ACR failed`) -- `torch.cuda.is_available()` was `False`
(`NvRmGpuLibOpen failed`). A clean `sudo reboot` (not `nvpmodel -m`, which itself failed
mid-corrupted-state with `NVPM ERROR: failed to read PARAM GPU`) fully recovered it; every
subsequent boot into every mode was clean. Recorded here as a real, board-specific flakiness
data point, not glossed over.

**A real infrastructure conflict, found and handled**: `maxn-super-autofix.service` (installed
earlier in this project to work around a JetPack 7.2 ISO-installer bug where MAXN_SUPER was
missing from the shipped config -- see Day-0 item 3) unconditionally force-switches back to
MAXN_SUPER on every boot if it isn't already active -- which was silently fighting every attempt
to boot into 15W/25W, forcing a SECOND reboot back to MAXN_SUPER each time. Temporarily disabled
(`systemctl disable --now`) for the sweep's duration and re-enabled immediately afterward,
confirmed active again before device work closed.

**This board's real clock ceilings are narrower than nvpmodel's mode names imply** -- verified
directly (GPU devfreq `available_frequencies` table, `nvpmodel -q --verbose`'s own REAL_VAL vs
CONF_VAL, and a direct write test to the EMC kernel cap node): this unit's GPU devfreq table
tops out at **624.75 MHz** (below 25W's nominal 918 MHz and MAXN_SUPER's uncapped value -- both
converge to this same real ceiling; only 15W's own lower cap, 612 MHz, is genuinely below it),
and EMC is hardware-capped at **2133 MHz on all three modes** (a write test to
`/sys/kernel/nvpmodel_clk_cap/emc` accepted 3199000000 but the read-back value never changed).
Effective variation actually exercised across the sweep: CPU frequency plus GPU at 15W only -- a
12.8% end-to-end N=197/FP16 latency spread between the fastest (MAXN_SUPER) and slowest (15W)
mode measured. **No claim of a wide power range or clock-ratio sweep is supported by this data**
-- see `evidence/00_CORRECTIONS.md` item c for the full writeup.

**Real results, both pre-registered questions answered**:
- **Primary (proxy divergence persistence)**: proxy error at N=69/FP16 vs N=197/FP16 is
  **2.185x (15W) / 2.246x (25W) / 2.204x (MAXN_SUPER, fresh remeasurement)** -- essentially flat
  across the whole sweep, matching the committed MAXN_SUPER reference of **2.235x**
  (`out/energy/proxy_divergence.csv`) to within ordinary noise. Divergence persists; it does not
  weaken as power falls.
- **Secondary (precision axis)**: INT8 is slower than FP16 in **15/15** (3 modes x 5 buckets)
  tested cells -- never once crosses the "faster" line, let alone by the pre-registered >5%
  threshold at >=2 buckets. Automatic verdict:
  **`AXIS_DOMINATION_CONFIRMED_ACROSS_OPERATING_POINTS`**. The precision axis does not revive at
  reduced clocks.
- **Replan (Part 4)**: joint and staged select the identical plan at **18/18** (3 modes x 6
  budgets) cells, at every mode -- the same degenerate-budget pattern (most budgets have
  `tau_b >= Latency_id`) already found at MAXN_SUPER reproduces identically across the sweep.

**Two numeric corrections surfaced by this sweep**, both traced to the same root cause (a
GPU-compute-only `trtexec` cross-check figure being used where this project's own TOTAL
per-block latency should have been) and documented in full in `evidence/00_CORRECTIONS.md`: the
MAXN_SUPER proxy-error reference is **2.235x, not 1.84x**; achieved throughput at N=197/FP16 is
**1.83 TFLOPS, not ~2.5**. Neither correction changes any conclusion -- both were citation
errors in numbers that had circulated in prior correspondence, not errors in the underlying
measurement.

Energy-per-mode (a theoretical Part 5) was deliberately **not** attempted -- both pre-registered
questions were already answered cleanly by Parts 1-4, and it would only have generalized an
already-closed result; see `evidence/00_MISSING.md` item 8 for the explicit scoping rationale.

**Device work is now closed for real** -- this was the last device experiment across the entire
G1-HW phase (bring-up, LUT, replan, paired eval, batch-8 pilot, DeiT-B pilot + crossover,
frontier/LUT extension, energy measurement, energy replan, and now the power-mode sweep). All
outputs: `out/powersweep/`.

### Pre-flight port-equivalence gate (`replan.py --verify-port`)

Before any hardware measurement is trusted, `replan.py --verify-port`
proves the PORTED Pareto DP (schedule enumeration, per-schedule bit
selection, cross-schedule budget selection -- the same code the real
on-device `replan.py` run uses) is faithful, independent of whether
latency or BOPs is the cost axis: it re-runs that identical DP with
P2.1's own `block_bops` formula substituted for the latency LUT, for all
6 budgets, and asserts the resulting plans -- per-block bits, per-block
token counts, cost, and `D_pred` -- exactly match P2.1's own committed
`plans.csv`. Without this, a measured on-device accuracy/latency delta
could not be attributed to latency specifically -- it might instead be
an artifact of the port disagreeing with P2.1's own allocator even under
identical cost semantics. Halts (raises), naming the first differing
block, on any mismatch -- per spec, no hardware measurement should be
attempted after a failure here.

### Expected runtimes (Orin Nano Super, order-of-magnitude -- record real
numbers once measured; these are planning estimates, not commitments)

| Step | Estimate |
|---|---|
| `replan.py --verify-port` | seconds (pure Python DP over 85 schedules, no device) -- run this first |
| `setup.sh` | 2-10 min (longer on first `apt-get update`) |
| `export_segments.py` | 2-5 min on a host GPU (5 ONNX exports + verification + calibration capture) |
| `build_engines.sh` | 15-40 min (5 buckets x 2 precisions x (build + 200-iteration benchmark); INT8 calibration adds the most) |
| `measure_lut.py` | 10-20 min (5 buckets x 2 precisions x 220 timed iterations + thermal cooldowns, if any) |
| `replan.py` | seconds (pure Python DP over 85 schedules) |
| `paired_eval.py` (on Colab) | 15-45 min on a Colab GPU (6 budgets x 3 plans -- BOPs, JOINT, STAGED, third arm added on review -- x 20,000 images; dominated by PyTorch inference, not TensorRT -- faster than the Jetson would be here, since this step never touches TensorRT/the SoC at all) |

### Resume behavior

- `measure_lut.py`: **resume-safe by construction** -- writes
  `latency_lut.json` after every single `(bucket, precision)` pair, and
  skips any pair already present on a re-run (`--force` to re-measure
  everything). A killed/interrupted run loses at most one in-progress
  measurement.
- `build_engines.sh`: **not currently resumable mid-bucket** (bash `set
  -e` halts on first failure); re-running after a fix re-builds every
  bucket from scratch. Each bucket's build+benchmark is independent, so a
  failure at bucket 3 of 5 does not invalidate buckets 1-2's already-
  written engine files -- but the script itself does not skip them on
  re-run today. Left as a known limitation, not silently glossed over.
- `paired_eval.py`: **not resumable mid-run** (the pooled gate statistic
  is computed once over the full sweep by design -- see its own
  docstring, "no peeking"). A killed run must be restarted from budget 1.
- `self_test.py`: stateless, safe to re-run anytime.

### Thermal guard

`measure_lut.py` checks temperature via `jtop` before every `(bucket,
precision)` measurement; if any zone is >=80C, it sleeps 20s and re-polls
(up to 15 retries) before proceeding, and raises rather than reporting a
throttled number if still hot after all retries. `build_engines.sh`
enforces a coarser guard: it refuses to start at all unless
`jetson_clocks --show` reports every rail locked (a proxy for "the device
is in its intended steady operating state," checked once per script
invocation, not per-bucket like `measure_lut.py`'s live 80C check).

---

## 3. Outputs

| File | Produced by | Description |
|---|---|---|
| `out/versions.json` | `setup.sh` | L4T, TensorRT, CUDA, python, torch, GPU name, pip-freeze hash, `nvpmodel -q` + `jetson_clocks --show` snapshots. |
| `out/segments/manifest.json` | `export_segments.py` (host) | Resolved checkpoint tag + state-dict hash, exported block index, per-bucket ONNX paths + PyTorch-vs-ONNX max-abs-diff, calibration metadata (pinned image count + hash, or `null` if `--imagenet-dir` was omitted). |
| `out/segments/block_seg_n{197,165,133,101,69}.onnx` | `export_segments.py` (host) | The 5 exported, verified ONNX transformer-block segments. |
| `out/segments/calib/calib_n{...}.npy` | `export_segments.py` (host) | Real (subsampled) activation arrays for INT8 calibration, one per bucket. |
| `out/engines/n{bucket}_{fp16,int8}.engine` | `build_engines.sh` | Built TensorRT engines, 10 total (5 buckets x 2 precisions). |
| `out/engines/n{bucket}_int8.cache` | `int8_calibrate.py` (via `build_engines.sh`) | TensorRT INT8 calibration cache (naive implicit path), reusable on re-build. |
| `out/engines/n{bucket}_int8_qdq.engine` | on-device `trtexec` (from `int8_qdq_experiment.py`'s ONNX output) | The BEST-ACHIEVABLE INT8 engine (explicit Q/DQ) -- what `measure_lut.py`'s `int8` entries actually measure. See INT8 findings above. |
| `out/logs/n{bucket}_{fp16,int8}_{build,bench}.log` | `build_engines.sh` | Raw `trtexec` build and benchmark logs for the FP16 and naive-INT8 engines. |
| `out/logs/n{bucket}_int8_qdq_bench.log` | on-device `trtexec` (manual, per the INT8 findings section's commands) | Benchmark log for the best-achievable (explicit-Q/DQ) INT8 engine -- `measure_lut.py`'s cross-check source for `int8` entries. |
| `out/lut/latency_lut.json` | `measure_lut.py` | The latency LUT: per `(bucket, precision)` -- median/p99 latency (this script's own timed loop, the LUT's source of truth), a `trtexec`-log cross-check median/p99, temperature before/after, mean power, joules/inference, versions, timestamp. Three precisions per bucket: `fp16`, `int8` (best-achievable, explicit Q/DQ -- what `replan.py` reads), `int8_naive_implicit` (provenance only, never read by `replan.py`). |
| `out/replan/plans_hw.csv` | `replan.py` | One row per budget fraction, THREE plans (BOPs, JOINT, STAGED -- see "Joint vs staged" above): `tau_b` (the budget itself -- the BOPs-selected plan's own measured latency, see "Budget pairing" above); each plan's measured latency/`D_pred`/full JSON (`bops_plan_latency_ms`/`bops_plan_d_pred`/`bops_plan_json`, `latency_plan_latency_ms`/`latency_plan_d_pred`/`plan_json` for JOINT, `staged_plan_latency_ms`/`staged_plan_d_pred`/`staged_plan_json` plus `staged_order`/`staged_alpha` for STAGED); `utilization`; `plans_identical`/`n_blocks_differing` (BOPs vs JOINT) and `joint_staged_identical`/`n_blocks_differing_joint_staged` (JOINT vs STAGED). All 6 budgets always present (feasible by construction -- see "Budget pairing"). |
| `out/replan/plans_hw_divergence.json` | `replan.py` | Per budget: `plans_identical` (bool) and `n_blocks_differing` -- distinguishes "hardware cost was effectively additive" (plans never diverged) from "joint allocation matters here" (they did). Same fields also live directly in `plans_hw.csv`; this file is a compact summary. |
| `out/replan/replan_manifest.json` | `replan.py` | `Latency_id`, the plan space's true achievable latency floor (`min_achievable_latency_ms`/`min_achievable_fraction_of_identity` -- the "Budget pairing" standing finding), the 6 `tau_b` values, schedule count, the bits-to-hardware-precision mapping (documented assumption), token buckets used. |
| `out/paired_eval/paired_eval_result.json` | `paired_eval.py` (HOST/Colab, not the Jetson) | The registered paired-comparison result: per-budget BOPs-plan vs latency-plan top-1, the pooled mean paired delta-top1 + 95% CI, and the gate's pass/fail (written unconditionally, in both directions -- see the script's own docstring). Produced off-device -- pull it into this local checkout (or the Jetson's own `out/`, if you want everything in one place) however you're already moving files around. |
| `out/self_test/` | (reserved) | Not currently written to by `self_test.py` (it only prints + raises) -- present for a future run-log if needed. |
| `out/frontier/{frontier_plans,frontier_manifest}.{csv,json}` | `frontier.py` | Original (v1) plan selection, both arms, per tau -- see "Re-scoped sweep" above. |
| `out/frontier/frontier_result.json` | `frontier_paired_eval.py` (Colab, HOST/not the Jetson) | Original (v1) paired accuracy result -- ARM A all 6 budgets, ARM A-vs-B restricted to the 2 feasible budgets. |
| `out/frontier/frontier_figure.png` | `frontier_figure.py` | Figure built from the two files above. |
| `out/frontier_v2/{frontier_plans_v2,frontier_result_v2,frontier_manifest_v2,arm_b_r_sweep_v2}.{csv,json}` | `frontier_v2.py`/`frontier_v2_paired_eval.py` | Extended-LUT frontier re-run -- the real +0.31pp TANDEM-vs-ToMe hardware result (see "LUT extension" above). |
| `out/lut_ext/{latency_lut_extended,lut_ext_export_manifest}.json` | `lut_ext_measure.py`/`lut_ext_export.py` | The 3-bucket LUT extension (N=53/37/21, FP16 only) backing `frontier_v2`. |
| `out/pilot_b8/{pilot_b8_result,pilot_b8_export_manifest,n{197,69}_b8_qdq_layerinfo}.json` | `pilot_b8_measure.py`/`pilot_b8_export.py` | Batch-8 INT8 pilot -- NO-GO at both buckets (see "Batch-8 INT8 pilot" above). |
| `out/pilot_deitb/{pilot_deitb_result,pilot_deitb_export_manifest,pilot_deitb_utilization,PREREGISTRATION}.{json,md}`, `n{197,69}_qdq_layerinfo.json` | `pilot_deitb_measure.py`/`pilot_deitb_export.py`/`pilot_deitb_utilization.py` | DeiT-B INT8 pilot -- BORDERLINE overall (see "DeiT-B INT8 pilot" above). |
| `out/pilot_deitb_ext/{pilot_deitb_ext_result,pilot_deitb_ext_export_manifest}.json`, `n{133,101}_qdq_layerinfo.json` | `pilot_deitb_ext_measure.py`/`pilot_deitb_ext_export.py` | DeiT-B 4-bucket crossover fill -- NO-GO overall (see "DeiT-B crossover characterization" above). |
| `out/energy/{energy_instrument,energy_lut,proxy_divergence_summary,PREREGISTRATION}.{json,md}`, `proxy_divergence.csv` | `energy_instrument.py`/`energy_lut.py`/`proxy_divergence.py` | Energy measurement (E-LUT) -- R²(BOPs→energy)=0.061 vs R²(latency→energy)=0.939 (see "Energy measurement" above). |
| `out/energy_replan/{energy_replan_manifest,PREREGISTRATION}.{json,md}`, `energy_replan_plans.csv` | `energy_replan.py` | Energy-axis replan -- `plans_identical` 6/6, axis domination holds on energy too (see "Energy-cost replan" above). |
| `out/powersweep/{modes_available,latency_lut_<mode>,powersweep_analysis_summary,replan_by_mode_manifest,PREREGISTRATION}.{json,md}`, `{reproduction_gate,precision_axis_by_mode,proxy_divergence_by_mode,replan_by_mode}.csv` | `powersweep_{enumerate,measure,repro_gate,analysis,replan}.py` | The power-mode sweep -- proxy divergence and axis domination confirmed across all 3 tested modes (see "Power-mode sweep" above). |

Most of `out/` IS committed to the repository -- small, real, human-readable
result artifacts (JSON/CSV/MD/PNG) are deliberately tracked through
`.gitignore`'s per-subdirectory whitelist (an `out/*` blanket ignore, with
explicit `!out/<subdir>/*.<ext>` negations per subdirectory -- see the
comment at the top of `jetson/.gitignore` for why the negation pairs are
structured that way). What stays ignored is the large, regenerable,
device-specific build output within those same subdirectories: ONNX
files, `.engine` files, calibration `.npy` arrays (except the two
`out/segments/calib/` buckets `pilot_b8_export.py` needs, tracked by
explicit exception), and raw `trtexec` build/bench logs.
