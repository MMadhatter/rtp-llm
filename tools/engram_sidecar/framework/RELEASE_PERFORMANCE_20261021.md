# Engram release performance validation — stage report, 2026-10-21

**The fresh Owner and consumer native runtime are READY. Component and
full-model validation remain IN_PROGRESS; the 100-microsecond additional-latency
target has not passed on this release.** The component screen has started;
full-model startup, serving and request-Graph gates remain pending. This is a
progress record, not a performance pass or a claim that historical tails have
been fixed.

The actual deployed/tested source remains
`3ac1fbda79b3509b554b562775a46138d586afd3`, based on release
`cb42b084a8a6350daa70897555f18d5f1b47f58e`. The commit adding this report changes
documentation only; it does not change that runtime identity. Development
continues on `codex/engram-sidecar-dsv41-release-20261019`.
See [release compatibility and CPU results](RELEASE_BASELINE.md) for the earlier
rebase validation, including its retained initial failures and targeted fixes.

## Completed in this stage

| Gate | Observed result | Limit of the evidence |
| --- | --- | --- |
| Owner bridge build from the deployed release source | PASS; new shared object, torch `2.11.0+cu130`; unresolved-library check passed | This standalone Owner bridge is separate from the consumer's four RTP engine libraries. |
| Owner bridge import with GPUs hidden | PASS; `torch.cuda.is_initialized()` was false; all eight bridge resource counters were zero; no RTP engine library was mapped | This is the pre-allocation import check, not the later live Owner's resource state or consumer ABI validation. |
| Fresh Owner generation and real tables | READY; two allocations, **202,758,946,816 physical bytes** (about 188.83 GiB); private publication ready | Both real Engram tables were loaded. READY does not prove consumer import, remote GPU reads, model correctness or performance. |
| Keeper/supervisor CPU contracts | **123 tests passed**, zero skips; independently rerun | Covers release identity, environment checks, ownership, cleanup and both host-PID proof methods; does not start a model. |
| Actual Linux lifecycle probe | PASS; exact-identity rejection/wake, pidfd signaling, exclusive lock and subreaper cleanup; final owned-child count zero; file-descriptor count 4 → 4 | Small CPU-only child processes, GPUs hidden, no torch or RTP native import. This validates OS lifecycle behavior, not the new model runtime. |
| Runtime-membership host-PID proof | PASS; actual runtime membership and host process identity collected twice; five identity-drift cases rejected; test child exited | Uses a small CPU child with synthetic READY/NVML records. Production must collect the real keeper's proof again; this is not large-table keeper or GPU-service validation. |
| Request Graph validator | **25 CPU tests passed** | Synthetic trace and HTTP boundary tests only. Actual consumer request traces remain pending. |
| Consumer native-build operator | **9 CPU tests passed** | Operator contracts only; the completed real build is recorded separately below. |
| CPU oracle for the fresh publication | PASS; three cases, **199 selected rows from each of the two real tables**; published samples matched; CUDA uninitialized | CPU checkpoint/hash reference preparation only, not remote GPU lookup verification. |
| Fresh consumer four-library build | PASS; CUDA 13 ARM configuration, **563.14 seconds**, return code zero, no remaining build children | Built from the recorded public release source; no internal source overlay or old engine library substitution. |
| Actual consumer ABI, imports and mappings | PASS; all four fresh libraries had one verified backing inode each; framework runtime CLI and actual service imports passed; `PyModelInputs.engram_token_windows_host` accepted a CPU tensor with the same data pointer | GPUs hidden and CUDA uninitialized. This proves import/ABI compatibility, not model construction, Graph replay or HTTP serving. |
| Fresh consumer bridge | PASS; newly built source/loader verified, unresolved-library check passed, all eight resource counters zero | Separate pre-allocation bridge import; it does not prove a consumer GPU has read either published table. |
| Build/import handoff | PASS; task compiler and import children exited, exclusive build cache had no live process, GPU compute-process count zero | Observed before the parent began the component screen; this is not cleanup evidence for that later GPU run. |

The Owner uses a standalone torch environment and a newly built bridge from
the recorded source. It does not load the old RTP engine libraries. It is not
being presented as a validation of the consumer's release dependency set.
While READY, the Owner intentionally holds two allocations, two mappings, two
reservations and two contexts; cleanup-failure and deferred-cleanup counters
are zero. Its later full shutdown/cleanup is still pending. The eight-zero
counter result in the table belongs only to the separate GPU-hidden import
check before these allocations existed.

The new consumer runtime and its four matching engine libraries have been
built and verified separately. Production launch guards require the exact new
virtual environment, its interpreter path before resolving symlinks, the release lock
digest and the two exact updated kernel-package versions. A resolved Python
executable path alone cannot distinguish virtual environments that share a
base interpreter. Current library content and installation evidence must also
match the release build; old engine-library checks cannot satisfy this gate.

The verified consumer dependency set includes `deep-gemm
2.8.2+ad62a90.cu132` and `rtp-kernel 0.1.0+94fd20eb.cu132`. Actual loaded package
paths belonged to the new environment, and 21 loaded third-party library
digests were checked against it. The GPU-hidden import emitted a caught
non-CUDA `FusedRopeKVCacheDecodeOp` fallback warning; that warning is retained
in the raw evidence and is not an observed failure of the GPU-visible V4.1
attention path.

The fresh consumer artifacts are identified below. A digest matching an older
artifact is not substituted provenance: the grammar library was rebuilt in
the new build root and its separately verified output happened to be identical.

| Artifact | SHA-256 |
| --- | --- |
| `libth_transformer_config.so` | `ce0e186ab39a7d784ea0b086a06cd2251e9fa05b9146d488d0fccd1a8d19f096` |
| `librtp_compute_ops.so` | `6d0efb140cfd39effdb8c2e532cb42ce1d1a75ea4e72650701d0bc60ff79d39f` |
| `libth_transformer.so` | `39c2f63f194529153042bcd267beada54390af2dae4450c43599e2d2f5aeb808` |
| `libth_grammar_tokenizer_info.so` | `0005d29292605d22eed27fd8cf68c03d6752668810fb99df126064612a27d8df` |
| Consumer sharing bridge | `679b0c495ae42fc4351e3c8b780efb40b08b98a2d7f8e5c7e12d1ea432ac0438` |

## Still pending

1. Finish the active component screen against the fresh Owner generation,
   validate actual cross-host GPU reads and both table outputs, then collect
   the unprofiled paired matrix and its resource-cleanup evidence.
2. Complete real full-model startup and chat correctness for both storage
   modes. The successful CPU/native runtime gate does not replace this gate.
3. Prove actual request Graph execution independently for both storage modes.
   Startup capture logs alone are insufficient. The diagnostic must bind four
   known rank processes to nested decode/Graph/replay scopes, an actual Graph
   launch and explicitly correlated GPU lookup nodes. Missing trace
   associations remain missing evidence.
4. Run the release's paired raw-lookup comparison and full-model A/B requests;
   preserve all raw samples, failures and outliers. Complete consumer shutdown
   and resource cleanup before closing the run.

No new consumer GPU latency, full-model HTTP performance or actual-request
Graph result is reported here. The Owner's READY state does not advance any
of those gates to PASS.

## Measurement boundaries and next run

Two comparisons remain separate:

- **Raw joint lookup:** the actual two tables, identical raw FP8/E8M0 layout,
  consumer GPU and row-ID trace; local host-pinned backing versus cross-host
  EGM. Measure both lookup outputs in one paired interval. This excludes
  hashing, projection, gating and intervening model computation. Never add
  separate layer P99 values to estimate a combined P99.
- **Full-model requests:** start with one sequential **A1–B1–B2–A2** cycle,
  where A is local pinned and B is external EGM. Each Graph service phase must
  pass chat and bounded request-Graph diagnostics, finish profiling/export,
  then run two fresh unprofiled warmups and eight sequential 256-output-token
  requests. Keep the actual rendered prompt length, corresponding seeds,
  generation settings, topology and non-storage environment identical. Eager
  full-model startup is a fallback for diagnosing Graph failures, not an
  additional default service matrix.

The paired full-model statistic is the **request-average engine TPOT
increment**, alongside TTFT and client completion time. HTTP chunk gaps are
not GPU step timings. A distribution of request averages does not establish
per-step decode P99/max. That stronger tail claim needs separate correlated
per-step completion evidence and remains unestablished by this protocol.

Request-Graph profiling is diagnostic only and must be fully stopped before
formal measurements. A profile cannot replace an unprofiled latency result.
If the first full cycle clearly fails the additional-latency target,
investigate it; expand to independent cycles only when useful to resolve a
preliminary pass or uncertainty. Do not delete unfavorable samples or rerun
only slow requests.

## Historical evidence remains valid within its original baseline

The previous main-based
[joint-table measurements](results/20261021_joint_numa_v2/README.md) include
GPU and wall-clock increments above 100 microseconds, including millisecond
tails. They have not been resolved by this rebase or by making the new Owner
READY. They remain preserved under their original source and run identities.

The subsequent
[profiled diagnostic](results/20261021_nsys_diagnostic/README.md) did not
reproduce those earlier GPU tails. It was PROFILE_ONLY; that absence was
neither a fix nor latency acceptance. This release stage does not overwrite
either historical result and does not attribute the unresolved tails to
fabric, the operating system or another component without further evidence.

## Public evidence policy

This summary intentionally excludes machine identities, addresses, process
identifiers, private filesystem locations and memory-sharing capabilities.
Original operational logs and trace files remain private. Public follow-up
results should contain sanitized statistics, sample identifiers, source and
artifact digests, and explicit measurement/cleanup status.
