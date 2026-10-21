# Cross-host Engram full-model HTTP readiness, 2026-10-21

**The external-EGM full model returned a readable Chinese chat response.** This
is a successful standalone B storage-mode readiness check; local-pinned A,
formal A/B performance and actual request Graph execution remain unverified.
The model consumer was subsequently stopped and its owned-process and GPU-process
cleanup checks passed; the Owner remains a separate live service.

The running source was `3ac1fbda79b3509b554b562775a46138d586afd3`, based on
release `cb42b084a8a6350daa70897555f18d5f1b47f58e`. Publishing this report does
not change that runtime. See [machine-readable validation](validation.json)
and the [stage report](../../RELEASE_PERFORMANCE_20261021.md).

| Check | Observed evidence | What it establishes |
| --- | --- | --- |
| Cross-host storage | Owner and consumer run on separate hosts and in separate containers; both real Engram layers 1/14 use external EGM | The full model uses the shared Owner publication. |
| Shared allocation | Two Owner allocations, 202,758,946,816 physical bytes, about 188.83 GiB; four model ranks attach externally | The existing shared-backing design is exercised by the model. This readiness check is not a new system-wide memory census. |
| Fresh runtime | Four live rank processes map all five expected fresh artifacts, one backing per artifact; exact identities match those that later reach READY | Four release engine libraries plus the sharing bridge, with checked artifact/mapping identity. |
| Startup Graph | All four ranks log batch-1 capture and first-replay synchronization, then capture completion | Startup capture/warmup succeeded. This does not identify the execution path of a later HTTP request. |
| HTTP chat | Backend health OK; 12 prompt tokens, 32 completion tokens, finish reason `length` | A real request produced readable text. The short response is not a complete functional evaluation or performance benchmark. |

The returned text was:

> 你好！我是DeepSeek，由深度求索公司创造的AI助手，随时准备用热情和知识帮你解答问题、提供建议！😊

The private readiness archive has SHA-256
`3c18f888c157926f1f3c83535a22be57a7d2adf75ecf23027a45ab4e000f8059`.
Its archive hash and all 18 recorded member hashes were checked independently.
The earlier startup diagnosis had recorded progressing serialized JIT and no
HTTP readiness yet; the later READY/chat records establish the subsequent
successful startup. The same four process identities connect these observations.
Original archives remain private; this directory publishes selected fields only.

## Request Graph diagnosis remains open

The separate `graph-v1` diagnostic stopped in its environment precondition with
`PhaseError`, before invoking its diagnostic HTTP or profiler-start API. It sent
**zero diagnostic HTTP requests and did not arm the profiler**. Expected flags
were absent from the recorded process-environment observation for all four
ranks. That observation does not establish their effective runtime values.
A later, separate two-case CPU reproduction in the same runtime found that
`setproctitle 1.3.7` can erase the process-environment observation while all
eight values remain `0` in Python `os.environ`; `SPT_NOENV=1` preserved the
observation. Both small test children exited. This establishes an observation
limitation, not the stopped model workers' effective values. It does not
retroactively prove their request Graph execution.

This diagnostic attempt provides no request Graph trace and is not evidence
of a GPU or Graph execution failure. Startup markers and the successful chat
are retained as their own evidence.

## Remaining acceptance work

- Complete local-pinned A readiness and a matched A/B request cycle; this
  standalone B check is not a completed ABBA phase.
- Obtain explicitly correlated four-rank request Graph evidence separately
  from formal, unprofiled measurements.
- Measure request-average engine TPOT increments and TTFT. Neither this chat
  nor HTTP chunk intervals establish individual decode-step P99/max.
- Keep consumer process cleanup separate from native resource counters and
  Owner lifetime. This run is CLOSED: supervisor cleanup completed, independent
  exact-identity checks found no surviving owned process, and the consumer
  GPU-process query returned no compute processes. No eight-counter native
  resource audit was collected for this full-model shutdown; Owner shutdown
  is not claimed.

The earlier [release component measurements](../20261021_release_components/README.md)
remain unchanged: their Graph paired GPU increment reached **207.424 µs** and
exceeded 100 µs once. The full-model chat success does not resolve those tails
or satisfy the latency target. No fabric/OS/GPU cause is inferred from this
readiness result.
