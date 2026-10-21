# Complete-table pinned-host versus remote-EGM lookup latency

`checkpoint_latency.py` compares **the same real, complete FP8/E8M0 tables** on
one consumer GPU. Both paths call main's `HostEngramEmbedding` and unchanged
Triton lookup. The baseline uses main's `_PinnedSharedTable`, populated from
independent original checkpoint shards. The candidate uses the admitted remote
EGM generation. No full table is copied from remote EGM into the baseline.

**This measures selected layers' lookup component, not complete decode overhead
or a 100 µs whole-step pass.** The boundary is IDs ready → BF16 lookup rows ready. Hashing,
WKV quantization/projection, gated residual, other layers, scheduling and HTTP
are excluded. The real checkpoint has Engram at layers 1 and 14. `--layer-mode
single` measures each selected layer separately; `--layer-mode together` puts
both layers in one invocation/Graph and one event interval. Joint delta
statistics come from those actual joint samples, never by adding layer p99s.
The two lookup calls have independent inputs and are submitted serially on the
same stream. There is no modeled data dependence or intervening transformer
computation, so even the joint result is a component gate, not a full decode SLO.

## Inputs and capacity

- Use the real table-free consumer checkpoint, admitted private bundle, and
  private oracle produced by `checkpoint_lookup.py prepare`.
- Put original table shards and their config/tokenizer/index in a separate,
  task-private source checkpoint on the consumer. That directory is an explicit
  benchmark control, not the deployed consumer checkpoint. An original index
  may refer to unavailable backbone shards; this harness reads only the two
  Engram table shards.
- Keep the Owner alive until this consumer process exits. No capability is
  logged or included in results.
- Budget **one complete approximately 94.4 GiB local table** in `/dev/shm` and
  additional source file-cache / driver headroom. The source is mmap-backed;
  `_PinnedSharedTable` copies in 64 MiB chunks. That does not guarantee constant
  staging RSS or immediate eviction of source pages. Do not drop global caches.
- Together retains two complete baselines (approximately 188.83 GiB). A 256 GiB
  `/dev/shm` can hold them, but it does not establish enough cgroup/NUMA capacity:
  with both source shards cached, table + source pages can approach 377 GiB
  before runtime overhead. Under a 384 GiB memcg, that is insufficient margin.
  Before each table load, the harness checks actual cgroup-v2 usage/limits
  (including visible ancestors), `/dev/shm`, host available memory and local
  NUMA free memory. It conservatively reserves the complete table **plus the
  complete source shard(s)** and `--memory-headroom-gib` (default 16). After
  loading it checks actual remaining headroom again. This can refuse a run
  even when reclaim might make it possible; the harness does not promise that
  byte counters guarantee pinned-memory admission.
- The process selects a CPU on the chosen GPU's `HOST_NUMA_ID` before creating
  the table, then requires all observed baseline resident pages on that NUMA
  node. An admission failure is fatal; a remote NUMA baseline is not accepted.
- Use a prebuilt genuine native extension and an authorized GPU. Source and
  model hash ABI, geometry, bootstrap samples and independent oracle must agree.

## Minimal run

Start with the actual model's first Engram layer and M=1, 100 pairs. This is a
screen only; both eager and isolated guarded CUDA Graph paths are measured.

```bash
python tools/engram_sidecar/framework/checkpoint_latency.py \
  --checkpoint /absolute/consumer-checkpoint \
  --source-checkpoint /absolute/private-original-table-source \
  --bundle /absolute/private-consumer/bundle.json \
  --private-inputs /absolute/private-oracle/oracle.json \
  --output /absolute/results/fresh-M1-screen \
  --layers first --batches 1 --modes eager,graph \
  --samples 100 --distributions hot,random
```

After correctness and cleanup pass, use a **new output directory** and
`--samples 10000`. Run fresh processes/seeds as independent repeats. Extend to
`--batches 1,2,4,8,16,32,64` and `--layers 1,14` as resources allow. Layers are
measured sequentially, releasing the first local baseline before loading the
second; this is not a combined two-layer timing.

For a genuine joint lookup measurement, add `--layer-mode together --layers
1,14`. With `together` and no explicit layer list, all real configured Engram
layers are selected; with default `single`, the first layer remains the default.
The selected layers execute in model order, with different deterministic trace
seeds and buffers. Each backend invokes both layers inside the same measured
window. Both actual outputs are independently checked, including Graph replay.

The optional `--discard-private-source-cache` sends per-file
`POSIX_FADV_DONTNEED` **before and after each complete baseline load**. It does
not change main's 64 MiB copy loop or claim per-chunk cache eviction. Use it only
for an immutable copy created for this task: source directory mode 0700 and
current-UID, selected source files mode 0400, current-UID, regular, single-link,
without symlinks. Shared checkpoint files and writable copies are refused.
Files are opened read-only; file bytes and global cache settings are unchanged.
The operator must ensure no other same-UID task uses these copies. Successful
advice is not counted as reclaimed memory: the measured budget still must pass.
The harness records each before/after memory snapshot and advice result. These
snapshots are sequential observations, not atomic reservation against other jobs.

Repeat selected cases with `--l2-disturb`. Scratch is twice runtime L2 and is
finished before each path. This is a controlled disturbance, **not proof of
cold Grace memory, remote caches, fabric, page tables or TLBs**. Random IDs are
uniform over the complete real table with replacement, not a production token
distribution. Hot uses one fixed ID set. The full table capacity remains
present in both modes, unlike the earlier 264 MiB fixture.

## Timing and correctness

Each pair uses the same IDs on the same GPU, rotating/reversing path order.
Input copies, scratch, setup, validation and result writes are outside both
timing intervals. Each request is synchronized; this is isolated latency,
not a sustained queue/load/throughput benchmark.

| Measurement | Boundary and limit |
|---|---|
| Eager GPU interval | CUDA start/end event records around actual Python dispatch; includes launch gaps between records |
| Graph GPU interval | External timing nodes captured inside Graph; elapsed graph interval, not exclusive kernel/remote-memory time |
| Consumer wall | Python dispatch/replay entry → completion-event synchronization returns |
| Submit / synchronization partitions | Host call durations; neither is exclusive GPU work/wait |
| Paired overhead | Candidate minus baseline from each adjacent pair; negative values and all outliers retained |

The common `CapturedEngramGraph` wrapper pins storage for **both** paths (the
local path conservatively uses the same guard without reading remote storage).
This equalizes harness replay/lock overhead. Each Graph is drained/reset before
storage can close. This does **not** enable RTP engine-owned CUDA Graphs.
Eager measures the actual `HostEngramEmbedding` wrapper, including the remote
storage-state check/lock. Attach/import/map never happen in the timed region.

Before measurement, both paths' actual outputs are checked against independent
source-row byte decoding on two different random inputs. In Graph mode, those
are replay outputs, with no intervening eager overwrite. After each distribution,
the actual final outputs are checked again before graph destruction. Result
files record these checks; eager correctness alone cannot certify Graph.

Optional Graph event-only/empty-Graph calibration uses the existing harness and
is reported separately. No calibration value is subtracted. It does not cover
rare long-run timing noise or calibrate eager dispatch. Eager and Graph GPU
intervals have distinct labels and must not be conflated.

## Evidence and interpretation

Each case writes `samples.jsonl` incrementally, flushing/fsyncing every 128
pairs outside timing. `summary.json` appears only after measurement, correctness
and guarded Graph cleanup complete. `complete.json` appears only after all
cases and local unregister/remote close return successfully with zero native
counters. Failed cleanup retains references until process teardown and emits
failure, rather than unmapping memory still potentially referenced by a Graph.
After a failed worker exits, its logged task-created tmpfs data/lease paths may
need explicit cleanup. Process exit alone is not proof these files were removed;
verify the worker has exited and the paths belong to this run before cleanup.

Report per distribution/mode/batch/layer: absolute local and remote p50/p95/p99/
p99.9/max, paired delta statistics and count >100 µs, sample count, seed,
trace/source digests, complete table bytes, NUMA, software/GPU, calibration and
correctness/cleanup status. Preserve raw outliers. CPU context-switch deltas are
correlation only. A GPU event or sync spike cannot by itself identify EGM,
network, scheduler, driver or a particular kernel as the cause.

Any >100 µs observed paired delta fails that tested component condition. Zero
observed violations in finite samples establishes neither an every-request
guarantee nor whole-decode acceptance. Next gates are full two-Engram-stage (with WKV/gate)
and real decode-step comparison on the same model, real engine Graph lifecycle,
multiple independent consumers and EP traffic. The earlier known millisecond
outliers remain evidence and must not be replaced by a nicer short run.

CPU-only contracts:

```bash
python -B tools/engram_sidecar/framework/test_checkpoint_latency.py -v
```

These UTs do not execute GPU timing, complete-table allocation, Graphs or remote
reads. Actual GPU results must be recorded separately.
