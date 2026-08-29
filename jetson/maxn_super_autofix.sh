#!/usr/bin/env bash
# jetson/maxn_super_autofix.sh -- idempotent, SAFE fix for the JetPack 7.2 MAXN SUPER
# ISO-installer bug (see README.md Day-0 item 3 for the full diagnosis). Re-applies the
# LOW-RISK config-swap fix (never the higher-risk /etc/nv_boot_control.conf edit, which
# has no easy rollback short of a full re-flash) every time it runs -- including
# automatically on every boot via maxn-super-autofix.service, because this bug has been
# reported to sometimes regenerate /etc/nvpmodel.conf from the still-uncorrected board
# identity on later reboots, silently dropping the MAXN SUPER mode again.
#
# Deliberately does NOT reboot or run jetson_clocks itself -- if the config file needed
# swapping this run, the NEXT boot re-runs this same idempotent script and confirms it
# stuck (self-healing over at most one extra boot, no risk of a reboot loop). Clock
# locking stays a separate, deliberate step (README Day-0 item 3 / build_engines.sh's own
# guard) -- this script's job is only "is MAXN SUPER available and active", not clocks.
#
# Must run as root. Safe to run by hand any time: `sudo bash maxn_super_autofix.sh`.
set -euo pipefail

log() { echo "[maxn-super-autofix] $*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "[maxn-super-autofix] must run as root (sudo $0)" >&2
    exit 1
fi

CONF=/etc/nvpmodel.conf

if [ ! -e "$CONF" ]; then
    log "ERROR: ${CONF} does not exist -- nothing to fix, this is not the known bug. Leaving alone."
    exit 1
fi

if grep -q 'NAME=MAXN_SUPER' "$CONF" 2>/dev/null; then
    log "MAXN_SUPER already present in ${CONF} -- config is fine, no swap needed."
else
    log "MAXN_SUPER missing from ${CONF} -- this is the known JetPack 7.2 ISO-installer bug."

    shopt -s nullglob
    candidates=(/etc/nvpmodel/nvpmodel_p3767_*_super.conf)
    shopt -u nullglob

    if [ ${#candidates[@]} -eq 0 ]; then
        log "ERROR: no *_super.conf found under /etc/nvpmodel/ -- cannot auto-fix. Needs manual investigation (see README.md Day-0 item 3)."
        exit 1
    fi

    super_conf=""
    if [ ${#candidates[@]} -eq 1 ]; then
        super_conf="${candidates[0]}"
    else
        # More than one *_super.conf candidate on this device -- match the CURRENT (broken)
        # conf against the non-super stock templates by CONTENT (sha256), not by symlink
        # target: /etc/nvpmodel.conf can be either a symlink to one of these templates OR a
        # plain-file copy of one (e.g. after this same script -- or the manual fix in
        # README.md -- already ran once), and a plain-file copy has no symlink target to
        # read at all. Content hashing works either way.
        current_hash=$(sha256sum "$CONF" | cut -d' ' -f1)
        matched_num=""
        shopt -s nullglob
        for f in /etc/nvpmodel/nvpmodel_p3767_*.conf; do
            case "$f" in *_super.conf) continue ;; esac
            if [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$current_hash" ]; then
                matched_num=$(basename "$f" | grep -oP 'p3767_\K[0-9]+' || true)
                break
            fi
        done
        shopt -u nullglob
        if [ -n "$matched_num" ] && [ -e "/etc/nvpmodel/nvpmodel_p3767_${matched_num}_super.conf" ]; then
            super_conf="/etc/nvpmodel/nvpmodel_p3767_${matched_num}_super.conf"
        fi
        if [ -z "$super_conf" ]; then
            log "ERROR: multiple *_super.conf candidates and the current ${CONF} content doesn't match any known non-super stock template -- refusing to guess. Candidates: ${candidates[*]}"
            exit 1
        fi
    fi

    log "Swapping in ${super_conf} -> ${CONF}"
    cp -f "$CONF" "${CONF}.pre-autofix-backup"
    cp -f "$super_conf" "$CONF"
fi

current_mode=$(nvpmodel -q 2>/dev/null | grep -oP '(?<=NV Power Mode: )\S+' || echo "")
if [ "$current_mode" = "MAXN_SUPER" ]; then
    log "MAXN_SUPER already the active mode -- nothing more to do."
    exit 0
fi

mode_id=$(grep 'NAME=MAXN_SUPER' "$CONF" | grep -oP 'ID=\K[0-9]+' | head -1 || true)
if [ -z "$mode_id" ]; then
    log "ERROR: could not parse MAXN_SUPER's mode ID out of ${CONF} -- refusing to guess an index. Run 'nvpmodel -q --verbose' and activate it manually."
    exit 1
fi

log "Activating MAXN_SUPER (mode ID ${mode_id}), currently: ${current_mode:-unknown}"
nvpmodel -m "$mode_id" --force
log "Done. nvpmodel -q now reports: $(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
