"""jetson/powersweep_enumerate.py -- ON-DEVICE. POWER-MODE SWEEP, Part 0 step 1-2. Queries
this board's REAL nvpmodel configuration (does not assume the mode set -- prints/records
whatever /etc/nvpmodel.conf + `nvpmodel -q --verbose` actually report) and empirically tests,
for every non-active mode, whether switching to it requires a reboot on THIS board/JetPack
build -- by issuing the real `nvpmodel -m <id>` switch command with stdin="no\\n" (a universal
decline: nvpmodel only accepts an exact "YES"/"yes" to proceed, anything else is treated as bad
input and the switch is aborted with the mode left unchanged -- confirmed safe by hand before
this script was written). This makes "which modes need a reboot" a real, reproducible,
evidence-backed finding rather than an assumption, without ever completing an unwanted switch.

Writes out/powersweep/modes_available.json. No engines/GPU touched -- config + CLI text only.

Usage: sudo -n python3 powersweep_enumerate.py   (needs passwordless sudo, same as jetson_clocks
--show elsewhere in this folder -- see self_test.py check 3's own note on why)
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ensure_out_dir, write_json  # noqa: E402

NVPMODEL_CONF = "/etc/nvpmodel.conf"


def run(cmd, input_text=None):
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True)


def parse_conf(conf_text):
    """Parses every `< POWER_MODEL ID=<id> NAME=<name> >` block up to the next `<` line or EOF.
    Extracts CPU_A78_0 MAX_FREQ (representative -- all 6 cores share the same value in every
    mode on this board, confirmed by inspection), GPU MAX_FREQ, EMC MAX_FREQ. -1 means
    "uncapped -- hardware ceiling", per nvpmodel.conf's own documented format (see its header
    comment: "-1 is taken as INT_MAX")."""
    blocks = re.split(r"\n(?=< POWER_MODEL)", conf_text)
    modes = []
    for b in blocks:
        m = re.search(r"< POWER_MODEL ID=(\d+) NAME=(\S+) >", b)
        if not m:
            continue
        mode_id, name = int(m.group(1)), m.group(2)

        def freq(pattern):
            fm = re.search(pattern, b)
            return int(fm.group(1)) if fm else None

        cpu_max = freq(r"CPU_A78_0 MAX_FREQ (-?\d+)")
        gpu_max = freq(r"GPU MAX_FREQ (-?\d+)")
        emc_max = freq(r"EMC MAX_FREQ (-?\d+)")
        modes.append({
            "id": mode_id, "name": name,
            "cpu_max_freq_khz": cpu_max, "cpu_max_freq_uncapped": cpu_max == -1,
            "gpu_max_freq_hz": gpu_max, "gpu_max_freq_uncapped": gpu_max == -1,
            "emc_max_freq_hz": emc_max, "emc_max_freq_uncapped": emc_max == -1,
        })
    default_m = re.search(r"< PM_CONFIG DEFAULT=(\d+)", conf_text)
    default_id = int(default_m.group(1)) if default_m else None
    return modes, default_id


def query_current_mode():
    r = run(["sudo", "-n", "nvpmodel", "-q", "--verbose"])
    text = r.stdout + r.stderr
    m = re.search(r"Current mode: NV Power Mode:\s*(\S+)\s*\n(\d+)", text)
    assert m, f"could not parse current mode from `nvpmodel -q --verbose` output:\n{text}"
    return m.group(1), int(m.group(2)), text


def test_reboot_required(mode_id, current_mode_id):
    """Empirically tests reboot requirement for a NON-active mode by issuing the real switch
    command and declining via stdin="no\\n" (see module docstring). Returns
    (reboot_required: bool, evidence: str). For the CURRENTLY active mode, no switch is
    attempted (it would be a no-op anyway) -- reboot_required is reported False with a note."""
    if mode_id == current_mode_id:
        return False, "already the active mode at enumeration time -- no switch attempted"
    r = run(["sudo", "-n", "nvpmodel", "-m", str(mode_id)], input_text="no\n")
    combined = (r.stdout + r.stderr).strip()
    reboot_required = "Reboot required" in combined
    return reboot_required, combined


def main():
    print("=== powersweep_enumerate.py: Part 0 -- real device mode enumeration ===")
    conf_r = run(["sudo", "-n", "cat", NVPMODEL_CONF])
    assert conf_r.returncode == 0, f"could not read {NVPMODEL_CONF}: {conf_r.stderr}"
    modes, default_id = parse_conf(conf_r.stdout)
    print(f"Parsed {len(modes)} POWER_MODEL block(s) from {NVPMODEL_CONF} (PM_CONFIG DEFAULT={default_id}).")

    current_name, current_id, verbose_text = query_current_mode()
    print(f"Currently active mode (nvpmodel -q --verbose): {current_name} (id={current_id})")
    if default_id is not None and default_id != current_id:
        print(f"NOTE: board's configured PM_CONFIG DEFAULT is id={default_id}, but the currently "
              f"active mode is id={current_id} ({current_name}) -- these differ, recorded as-is, "
              "not treated as an error (the active mode was almost certainly set explicitly by "
              "this project's own earlier setup, not left at the image's shipped default).")

    for mode in modes:
        reboot_required, evidence = test_reboot_required(mode["id"], current_id)
        mode["reboot_required"] = reboot_required
        mode["reboot_test_evidence"] = evidence
        # Re-verify the decline actually left the mode unchanged -- if this ever fails, the
        # "no" decline convention silently stopped working and this script must not proceed
        # pretending its own evidence is trustworthy.
        post_name, post_id, _ = query_current_mode()
        assert post_id == current_id, (
            f"SAFETY CHECK FAILED after probing mode {mode['id']} ({mode['name']}): active mode "
            f"changed from id={current_id} to id={post_id} despite declining with 'no' -- the "
            "decline convention this script relies on did not hold. HALTING -- do not trust any "
            "reboot_required value recorded so far; investigate manually."
        )
        print(f"  mode id={mode['id']} name={mode['name']}: reboot_required={reboot_required}")

    result = {
        "device": "Jetson Orin Nano Super Developer Kit",
        "nvpmodel_conf_path": NVPMODEL_CONF,
        "pm_config_default_id": default_id,
        "current_mode_at_enumeration": {"name": current_name, "id": current_id},
        "current_mode_verbose_raw": verbose_text,
        "modes": modes,
        "reboot_test_method": (
            "For each non-active mode, `sudo nvpmodel -m <id>` was actually invoked with "
            "stdin='no\\n' (a universal decline -- nvpmodel requires an exact 'YES'/'yes' to "
            "proceed, anything else aborts with the mode left unchanged). reboot_required=True "
            "iff the string 'Reboot required' appeared in the command's output. After every "
            "probe, the active mode was re-queried and asserted unchanged, so this is a safe, "
            "real, reproducible empirical test -- not an assumption from documentation."
        ),
    }
    out_path = ensure_out_dir("powersweep") / "modes_available.json"
    write_json(out_path, result)
    print(f"\nWrote {out_path} ({len(modes)} modes).")


if __name__ == "__main__":
    main()
