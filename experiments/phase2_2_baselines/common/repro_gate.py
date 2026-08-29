"""The +-0.3% reproduction gate (fairness rule #3). Framework-free.

Three possible outcomes, never silently collapsed into a bare pass/fail:
  - "pass" / "fail": a published reference WAS located (cited in each
    baseline's published.json) and the measured number is within/outside
    +-0.3% of it.
  - "no_published_reference": no published number for this exact config
    was located in the upstream repo's own documentation (see
    deviations.md for which configs this applies to, and why) -- reported,
    never asserted against a guessed number.
"""

GATE_TOLERANCE_PCT = 0.3


def check_gate(measured_top1_pct, published_top1_pct):
    """Returns (gate_status, diff_pct_points). `published_top1_pct=None`
    means no reference was located -- returns ("no_published_reference", None)
    rather than raising, so callers can still record the measured number."""
    if published_top1_pct is None:
        return "no_published_reference", None
    diff = abs(measured_top1_pct - published_top1_pct)
    status = "pass" if diff <= GATE_TOLERANCE_PCT else "fail"
    return status, diff


def format_gate_banner(method, model, config, measured_top1_pct, published_top1_pct, gate_status, diff_pct_points):
    lines = [f"[{method} / {model} / {config}]"]
    if gate_status == "no_published_reference":
        lines.append(f"  measured={measured_top1_pct:.2f}%  published=NONE LOCATED  "
                      f"-> {gate_status.upper()} (reported, not gated -- see deviations.md)")
    else:
        lines.append(f"  measured={measured_top1_pct:.2f}%  published={published_top1_pct:.2f}%  "
                      f"diff={diff_pct_points:.3f}pp (tol={GATE_TOLERANCE_PCT}pp) -> {gate_status.upper()}")
    return "\n".join(lines)
