"""Live-streaming subprocess helper. Framework-free (subprocess + stdlib
only).

`subprocess.run(..., capture_output=True)` fully buffers stdout/stderr and
only hands them back after the process exits -- for a long-running
subprocess (RepQ-ViT's/FQ-ViT's own `test_quant.py`, a 6+ minute FP-parity
pass) this means ZERO visible output for the entire duration, structurally
indistinguishable from a genuine hang (confirmed as a real, repeated point
of confusion on a live Colab run, 2026-07). `run_streaming` prints
stdout+stderr (merged, in the order they actually happened) live AS the
subprocess runs, while still returning the full text afterward so every
caller's existing regex/JSON-marker parsing needs no changes beyond the
call site itself -- a strict superset of what `capture_output=True`
provided, not a change to WHAT gets parsed or how success/failure is
determined.

Two things a naive rewrite gets wrong, handled here:
  1. **Child-side buffering.** Redirecting a Python subprocess's stdout to
     a pipe (not a real terminal) makes CPython default to FULL buffering,
     not line buffering -- without forcing this off, a parent-side
     line-by-line read loop still receives nothing until the child's own
     internal buffer fills or it exits, silently defeating streaming even
     though the code LOOKS correct. Fixed via `PYTHONUNBUFFERED=1` in the
     child's env -- an interpreter-level setting, not a modification of
     RepQ-ViT's/FQ-ViT's own unmodified scripts.
  2. **Two-pipe deadlock risk.** A naive implementation reading `stdout`
     and `stderr` as two SEPARATE pipes can deadlock if one fills its OS
     buffer while only the other is being read. Merging them
     (`stderr=subprocess.STDOUT`) means there is only ever one pipe to
     read continuously, which is deadlock-safe by construction.

`.stderr` is intentionally NOT a separate attribute on `StreamedResult`
(merged into `.stdout`) -- every caller that formatted an error message
from `result.stderr[-N:]` switched to `result.stdout[-N:]` at the call
site when adopting this helper, since that is now where that same
diagnostic text actually lives.
"""
import os
import subprocess


class StreamedResult:
    __slots__ = ("returncode", "stdout")

    def __init__(self, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout


def run_streaming(cmd, cwd=None, env=None):
    """Runs `cmd`, printing each line of its stdout+stderr (merged) to
    THIS process's own stdout as it arrives. Returns a `StreamedResult`
    with `.returncode` and `.stdout` (the full merged text, same shape as
    `subprocess.run(...).stdout` for every caller's existing parsing
    code). `env=None` inherits (a copy of) this process's own environment,
    matching `subprocess.run`'s own default."""
    run_env = dict(env) if env is not None else dict(os.environ)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return StreamedResult(proc.returncode, "".join(lines))
