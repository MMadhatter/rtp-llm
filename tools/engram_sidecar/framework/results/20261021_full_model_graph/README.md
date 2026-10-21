# Full-model B actual-request Graph check (2026-10-21)

## Status

`GRAPH_PROVEN` for the external-EGM (B) path. **ABBA was not run.**

## Configuration

- DeepSeek V4.1 Flash, 40 layers, TP4 / EP4 / DP1, four ranks on one GB200 tray.
- B storage: the Owner process holds both Engram tables; the consumer maps them
  with `CU_MEM_LOCATION_TYPE_HOST_NUMA` + `CU_MEM_HANDLE_TYPE_FABRIC`.
- Runtime/source under test: `3ac1fbda79b3509b554b562775a46138d586afd3`
  (release cb42), unchanged native libraries.
- Startup: HTTP health OK, 32-token chat OK, four-rank capture and first
  replay synchronized.

## Actual-request evidence

One bounded 32-token chat request was issued while an explicit profiler window
was open. The diagnostic exported traces from all four ranks (0-3) and observed:

- request Graph dispatch on all four ranks;
- correlated Engram Graph execution on all four ranks;
- successful chat and four-rank stop/export;
- no profiler still active and no trace-metadata warning.

Result: `STANDALONE_GRAPH_PROVEN`, `formal_phase_eligible=false`,
`formal_requests=0`, `latency_100us_acceptance=NOT_MEASURED`.

This is a readiness/provenance check, not a performance run.

## Why ABBA is absent

The formal cycle requires the local-pinned A baseline first. Under the current
shared-host memory pressure the A keeper could not admit both full tables plus
its fixed headroom (it stopped before producing an A phase). With no A1 result
the formal gate cannot start B1/B2/A2, so no A/B decode comparison and no
per-step `<=100 us` claim is made. The B Graph result above does not substitute
for the formal phase gate.
