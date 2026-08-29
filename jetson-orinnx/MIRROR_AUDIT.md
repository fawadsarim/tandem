# MIRROR_AUDIT.md — jetson/ vs jetson-orinnx/, file for file

Generated 2026-08-26, Gate PB-0, before any device contact this session.
Method: recursive SHA-256 over every file in both trees (excluding
`__pycache__/`), matched by relative path. 132 files on each side.

## Headline result

**132/132 files byte-identical. Zero missing, zero extra, zero differing.**

This is not the same shape as the previous (now-void, pre-reflash)
`jetson_orinnx/` mirror, which excluded 4 `out/` subdirectories as a scope
decision. This commit (`a991935`, "Tandem Jetson NX Starter") copied
`jetson/` **in its entirety**, including `out/`.

## ⚠ Critical finding: `out/` is 100% Nano Super data, not yet regenerated

Because the copy is byte-identical everywhere, **every measured-data file
under `jetson-orinnx/out/` right now is the Orin Nano Super's own result**,
carried over verbatim:

- `out/lut/latency_lut.json`, `out/lut_ext/latency_lut_extended.json`
- `out/powersweep/latency_lut_{15W,25W,MAXN_SUPER}.json`,
  `precision_axis_by_mode.csv`, `proxy_divergence_by_mode.csv`,
  `replan_by_mode.csv`, `reproduction_gate.csv`, `modes_available.json`
- `out/energy/energy_lut.json`, `out/energy/proxy_divergence.csv`
- `out/energy_replan/*`
- `out/frontier/*`, `out/frontier_v2/*`
- `out/paired_eval/paired_eval_result.json` — **the registered hardware
  gate itself, still the Nano's +0.88pp number**
- `out/replan/plans_hw.csv`, `plans_hw_divergence.json`, `replan_manifest.json`
- `out/pilot_b8/`, `out/pilot_deitb/`, `out/pilot_deitb_ext/` (results, not
  just scripts/ONNX)
- `out/reviewer_release/*`
- `out/versions.json` — Nano's own L4T/CUDA/TensorRT/torch versions, not
  this device's

Per the governing reuse/rebuild policy, **every one of these is measured
data that must be rebuilt on this device, never copied.** None of it has
been touched, read as ground truth, or cited anywhere in this session.
Flagging it here so it cannot be mistaken for real Orin NX output later —
a script reading `out/lut/latency_lut.json` today would silently get Nano
Super numbers labeled as if they came from this device.

**Only `out/segments/` (the 5 production ONNX files) is correctly
REUSE-appropriate** — see ONNX hashes below. Everything else in `out/`
needs to be understood as placeholder/stale until PB-3 onward actually
regenerates it, at which point jetson-orinnx/out/ and jetson/out/ will
finally diverge in their numbers, as the governing principle expects
("diff and see only different numbers, never different structure").

This is listed as the #1 anticipated decision point below — not something
I've fixed unilaterally.

## ONNX file hashes — the 5 production segments, cross-checked

All REUSE-appropriate (device-independent) per the reuse/rebuild policy.
Confirmed byte-identical between `jetson/out/segments/` and
`jetson-orinnx/out/segments/`:

| file | sha256 | status |
|---|---|---|
| `block_seg_n69_qdq.onnx` | `e120e0a813cabb95d52ba2c79fc0ce4db66a70f6f40a58cb19fe5d3f542b6e66` | MATCH |
| `block_seg_n101_qdq.onnx` | `e74f3ab427a7c97838a93f33306d5fe3c411b91166c58546c9cab5572275fb2f` | MATCH |
| `block_seg_n133_qdq.onnx` | `ea90f7cb2bc8492bb2d7c9fb7851de0995d666345bb6e6a56c91351254c453f6` | MATCH |
| `block_seg_n165_qdq.onnx` | `00179de3497decfbcf998ef9d133197600a8fe6e19b9c9cc4599ad6c380332a8` | MATCH |
| `block_seg_n197_qdq.onnx` | `af81ce7a98fc775a9dcab4023e2a7ef74274da7bfdd2e5e7698e96d2200c31f8` | MATCH |

(Verified: each hash above is exactly 64 hex characters, confirmed with
`${#h}` after the fact — an initial eyeball miscount claimed 65 in an
earlier draft of this file; corrected here rather than left standing.)

`out/pilot_b8/`, `out/pilot_deitb/`, `out/pilot_deitb_ext/` also contain
`.onnx` files (also device-independent by the same reuse logic) — not
separately hashed here since those pilots are outside this campaign's
stated mode/stage scope (see decision point 3 below).

## Scripts and top-level files

All 55 top-level `.py`/`.sh`/`.md`/`.ipynb`/`.service` files plus
`.gitignore`: byte-identical, confirmed by the same sweep (they're part of
the 132/132 total above, not counted separately). `setup.sh` in particular
is **unmodified** — no `--break-system-packages` patch or any other
divergence from `jetson/setup.sh`, consistent with the "standard path, no
patches" ruling for this clean SDK-Manager-flashed device.

## git status (host repo, this gate)

```
?? jetson-orinnx/MIRROR_AUDIT.md
```

Zero modified tracked files. The only change is this new file itself
(untracked, as expected — `jetson-orinnx/` is a tracked directory from
commit `a991935`, but adding a new file inside it does not modify any
existing tracked file).

