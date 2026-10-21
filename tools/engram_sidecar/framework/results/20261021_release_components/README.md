# Release Engram component results: batch 1, two complete tables

**The additional-latency target has not passed.** The completed unprofiled normal run contains one Graph GPU event-interval increment of **207.424 µs** and 18 Graph wall-clock increments above 100 µs. Both real-table correctness and consumer cleanup checks passed. Full-model decode, HTTP performance, and actual model-request Graph validation remain separate work in progress.

The measured source is `3ac1fbda79b3509b554b562775a46138d586afd3`, based on release `cb42b084a8a6350daa70897555f18d5f1b47f58e`. This report changes no runtime code. Earlier main-based results and their failures remain under their original source identities.

## What was measured

The same consumer GPU and row-ID traces compare local host-pinned tables with cross-host EGM. Both paths read the original FP8/E8M0 layout and produce BF16 lookup outputs. Layers 1 and 14, totaling **202,758,032,400 logical table bytes** (188.83 GiB), execute serially inside one interval, with independent input traces per layer. This includes neither token hashing nor projection/WKV/gating, intervening model computation, real model dependency, or HTTP.

Both runs use batch 1, hot fixed IDs and uniformly sampled rows, 20 warmups, balanced local/remote order, per-path completion synchronization, and the same seed. Input preparation finishes before measurement. Cache state is **uncontrolled**; no L2 scratch disturbance is used. Complete table allocation is not evidence that the sampled working set covers the full table or models production cache behavior.

| Run | Pairs per mode × distribution | Total paired observations | Original timing records | Result |
|---|---:|---:|---:|---|
| screen-v1 | 100 | 400 | 1,600 | Completed; two positive-threshold pairs retained; screen only |
| normal-v1 | 10,000 | 40,000 | 160,000 | Completed; 150 positive-threshold pairs retained; target not passed |

Each run is a separate process execution; they share a seed and are not independent random trace replicates. Statistics are reported separately, with no trimming or instrumentation subtraction. Every timing record was paired and checked, every reported statistic independently recomputed, all seven source-file digests matched the measured commit, and both archives' member bytes/SHA-256 values were verified.

## Normal run

All values are µs. `Δ = remote − local` is computed per pair; **P99(Δ) is not the difference of two P99 values**. Counts are strict positive `Δ > 100`. P99 uses nearest rank. Full local/remote P50/P95/P99/max and signed delta statistics are in [metrics.tsv](metrics.tsv) and [metrics.json](metrics.json).

| Mode / distribution | Clock | Local P50 | Remote P50 | Δ P50 | Δ P99 | Δ max | Δ >100 / 10,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| eager / hot | GPU event interval | 90.720 | 115.744 | 25.248 | 75.456 | 1516.896 | 73 |
| eager / random | GPU event interval | 94.720 | 118.112 | 24.800 | 70.336 | 124942.403 | 57 |
| Graph / hot | GPU event interval | 12.480 | 18.144 | 5.760 | 7.680 | **207.424** | **1** |
| Graph / random | GPU event interval | 16.032 | 19.904 | 3.744 | 7.360 | 16.864 | 0 |
| eager / hot | CPU submit → event complete | 105.280 | 130.432 | 25.247 | 76.257 | 1517.507 | 72 |
| eager / random | CPU submit → event complete | 109.408 | 132.928 | 24.767 | 72.321 | 124957.333 | 56 |
| Graph / hot | CPU submit → event complete | 40.128 | 45.728 | 5.024 | 18.496 | 2405.480 | 14 |
| Graph / random | CPU submit → event complete | 42.496 | 46.144 | 3.040 | 16.448 | 324.865 | 4 |

The 20,000 Graph pairs therefore include **1 GPU interval** and **18 wall interval** positive exceedances; the GPU-exceeding pair is also one of the 18 wall pairs. These clock counts must not be added as distinct samples. A low P99 does not erase a maximum beyond the requested bound.

### Exact anomalous samples and limits of attribution

The table gives local / remote values from the same pair. `Submit` is the measured Python invoke/replay call; `Sync` is the subsequent completion synchronization call, not exclusive GPU wait. These intervals overlap GPU activity and must not be algebraically subtracted to infer kernel duration.

| Normal sample | GPU interval | Wall interval | Submit | Sync | Δ GPU | Δ wall |
|---|---:|---:|---:|---:|---:|---:|
| Graph hot 2309 | 26.112 / 233.536 | 70.528 / 310.657 | 44.416 / 272.129 | 26.112 / 38.528 | **207.424** | 240.129 |
| Graph hot 7435 | 34.272 / 22.784 | 118372.962 / 104.288 | 118340.130 / 91.328 | 32.832 / 12.960 | −11.488 | **−118268.674** |
| Graph hot 8760 | 12.192 / 21.952 | 40.960 / 2446.440 | 28.224 / 2436.232 | 12.736 / 10.208 | 9.760 | **2405.480** |
| eager random 1410 | 227.328 / 125169.731 | 253.249 / 125210.582 | 244.065 / 125192.342 | 9.184 / 18.240 | **124942.403** | **124957.333** |

- **Graph hot 2309:** the remote measuring thread records one involuntary context switch; local records zero. The GPU events are captured inside the graph, so a long outer submit call or the context-switch count alone cannot explain away the 233.536 µs event interval. Without correlated GPU node/API timing, long kernel execution versus an internal scheduling gap remains unknown. Neighboring pair GPU deltas 2307/2308/2310/2311 are 7.264 / 6.304 / −4.448 / 4.512 µs; the isolated observation is still retained.
- **Graph hot 7435:** local submit is 118.340 ms and records two involuntary context switches, while its GPU interval is 34.272 µs. The negative paired wall delta is a slow local observation, not evidence of a 118 ms remote-memory advantage. No sample was dropped or clamped to zero.
- **Graph hot 8760:** remote submit is 2.436 ms while its GPU interval is 21.952 µs; both paths have zero current-thread context-switch increments. Long wall time can occur without a comparable Graph event interval. Zero thread counters do not prove that all operating-system or driver interference was absent.
- **eager random 1410:** remote submit and GPU event interval both span about 125 ms, with one involuntary thread context switch. Eager records events around Python dispatch of the lookup chain, so GPU waiting for further host submissions can lie inside that event interval. The current evidence does not separate such gaps from execution or device delay.

No anomaly is assigned to Fabric, networking, the OS scheduler, or a particular kernel. A context switch is evidence of co-occurrence, not proof of causation. The public [neighborhoods](anomaly_neighborhoods.json) preserve ±2 pairs around these examples. [outliers.jsonl](outliers.jsonl) and [outlier_ids.json](outlier_ids.json) contain **every pair where either clock has |Δ| >100 µs**, including negative deltas and pairs whose clocks have opposite signs. In the normal run, 150 pairs have a positive exceedance, 31 have a negative exceedance, and their union is 179 pairs; two pairs belong to both sets.

### Screen remains a separate failed observation

| Screen sample | Δ GPU interval | Δ wall interval | Remote submit | Remote completion sync |
|---|---:|---:|---:|---:|
| eager hot 72 | 201.504 | 203.648 | 309.825 | 7.808 |
| eager random 90 | 1671.040 | 1671.398 | 1797.670 | 8.448 |

Both screen anomalies have zero voluntary/involuntary switch increments on both paths. They are retained with their neighborhoods and full raw-derived pairs. Screen Graph maxima are 8.224 µs (hot) and 7.328 µs (random), with no positive GPU exceedance in its 200 Graph pairs; the larger normal run nevertheless observes one. At 100 samples, nearest-rank P99 is only the second-highest value and can miss a single exceeding maximum. Neither run overwrites the other.

## Timing, correctness, memory and cleanup

Eager uses `begin.record(stream) → Python lookup chain → end.record(stream)`; its GPU event interval includes any command-starvation gaps caused by host dispatch. Graph captures external start/end event nodes around the lookup chain. Its event interval excludes outer CPU time outside those nodes but can include inter-node/device scheduling gaps, so it is not a sum of kernel durations. Wall time runs from invoking the path through event-completion synchronization. These clocks are intentionally distinct.

Both runs verify actual eager/Graph outputs against independent CPU references for two different inputs before timing and after each distribution. Both complete pinned tables are resident on GPU-local NUMA node 0; read-only source mappings were verified on NUMA node 1. Budget checks before prefault, after prefault and after copy passed. Weight/scale pointers matched registered host aliases, with zero additional **PyTorch-tracked allocated HBM** for wrapping; this does not claim zero total driver HBM. See [validation.json](validation.json). Source pages were not locked against later eviction, and source preparation wall duration was not recorded.

Each consumer run completed with all eight native counters zero: contexts, owner allocations, imports, mappings, reservations, view storages, deferred cleanups and cleanup failures. This refers to the measured consumer process; the separately READY Owner continues holding the shared data. It is not Owner shutdown evidence.

The normal Graph control measured 100 event-only and 100 intentionally empty replays: event-only GPU P99/max **3.424 / 3.488 µs**, event-only wall max **18.880 µs**, empty-graph wall max **12.576 µs**. The expected empty-graph warning is retained as a warning flag in [instrumentation.json](instrumentation.json). No control time or tail quantile was subtracted. Small-shape event instrumentation has observable overhead; an uneventful 100-sample control does not prove absence of rare instrumentation effects.

## Reproducible public evidence

- [paired_samples.jsonl.gz](paired_samples.jsonl.gz): all **40,400** raw-derived pairs from both runs, with both clocks, per-path submit/sync partitions, current-thread switch counters, order and sample IDs; no timing value is rounded in the machine-readable records. This is an explicit sanitized projection, not the original file payloads.
- [metrics.json](metrics.json), [metrics.tsv](metrics.tsv): separate-run statistics, including signed minima and exceedance counts.
- [outliers.jsonl](outliers.jsonl), [outlier_ids.json](outlier_ids.json), [anomaly_neighborhoods.json](anomaly_neighborhoods.json): all 182 signed-threshold pairs across the two runs and selected context.
- [provenance.json](provenance.json): original archive/member digests and measured source-file digests. Original complete archives remain preserved privately because their metadata includes private runtime identity. Nothing is silently replaced by this projection.
- [validation.json](validation.json), [instrumentation.json](instrumentation.json): memory, actual-output, cleanup and timing-control evidence within the limits above.

The original normal archive SHA-256 is `bf502f00520ca3c954ad1912ce1edcc60d4aafee674c6f30ac5fc94ae5668eb6`. This publication excludes machine addresses/names, process IDs, UUIDs, generation identifiers, absolute private paths and memory-sharing capabilities. A public SHA manifest covers the projected artifacts.

The next acceptance step is the separately gated full-model A1–B1–B2–A2 cycle. Its engine request-average TPOT increment, TTFT and actual request-Graph evidence must be reported separately. Neither this component interval nor a distribution of request averages establishes individual model decode-step P99/max.
