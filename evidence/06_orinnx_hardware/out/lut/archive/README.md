# Archive: Historical & Superseded Orin NX Benchmarks

This directory contains intermediate and audit artifacts preserved for provenance:

1. latency_lut_pb3_superseded_unverified_clocks.json: Initial PB-3 measurements taken before strict dynamic frequency governor verification was patched.
2. latency_lut_pb3_reference.json: Historical reference from PB-3.
3. 
eproduction_gate_orinnx.csv: First reproduction gate attempt under unverified clocks (9/10 passed, n101_int8 +3.26%).
4. latency_lut_reprogate_verified_a.json & latency_lut_reprogate_verified_b.json: Intermediate verified-clock validation sweeps.
5. 
eproduction_gate_orinnx_verified_clocks.csv: Verified-clock reproduction gate prior to final power-mode sweep.

**Canonical Active File**:
- out/lut/latency_lut.json: The authoritative 15-engine measured lookup table.
- out/powersweep/reproduction_gate.csv: The official 10/10 passing reproduction gate (max delta 1.88%).
