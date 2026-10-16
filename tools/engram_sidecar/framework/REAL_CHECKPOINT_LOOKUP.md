# Real-checkpoint cross-host lookup

This experiment uses the original two large Engram tables from
`deepseek-ai/DeepSeek-V4.1-Flash`. It validates the complete backing allocation
and actual RTP eager hash/lookup path, without running attention, MoE, WKV,
the scheduler or an HTTP server. It is a separate prerequisite to the
[full-model HTTP experiment](FULL_MODEL_HTTP.md).

## Verified checkpoint preparation

The [staging evidence](CHECKPOINT_STAGING_RESULTS.json) records all 48 original
shard SHA256 values checked against the official ModelScope catalog. The Owner
copy contains complete shards 47–48. The consumer contains original shards 1–46
and repacked projection-only shards 47–48; neither its index nor its tensor
files contain the four full embedding tensors. All retained tensor shapes and
dtypes were compared with the source headers.

| Item | Size / count |
|---|---:|
| Engram layers | 1 and 14 |
| Complete logical FP8 weights + E8M0 scales | 202,758,032,400 bytes |
| Backing after 2 MiB CUDA allocation rounding | 202,758,946,816 bytes |
| Consumer tensor entries, excluding complete Engram tables | 96,081 |
| Consumer checkpoint files including metadata | 307,552,554,788 bytes |

The original config, index and tokenizer match the sampled official catalog.
The existing checkpoint's `encoding/encoding.py` is retained with its recorded
SHA256; this does not claim it equals the mutable remote `master` file today.
No weights or private capabilities are included in this repository.

## NUMA memory is a separate prerequisite

The full dataset needs about 188.83 GiB of pinned host backing in this design.
The development loader currently makes one approximately 94.4 GiB allocation
per layer on the selected local NUMA node. A large global `MemAvailable` value
does not prove that a particular NUMA allocation will succeed.

During preparation, `cuMemCreate(HOST_NUMA, FABRIC)` returned
`CUDA_ERROR_OUT_OF_MEMORY`. A subsequent isolated allocation of exactly one
real layer succeeded after private-file cache advice and changing background
memory pressure; the `after_create` test immediately rolled it back and left
all native resource counters at zero. This rules out a fixed 94.4 GiB bridge
limit in that environment. It does not isolate file cache as the only cause,
or establish a universally safe headroom threshold.

With streamed reads and private cache advice enabled, the next attempt created
and loaded the first complete layer, then failed at the second `cuMemCreate`.
At that second call, node 0 reported 107,031,744 KiB free, yet the approximately
94.4 GiB request still failed. Both Owner allocations were unpublished and all
native resource counters returned to zero. Free-byte totals alone therefore
remain insufficient admission evidence. Complete 188.83 GiB cross-host lookup
and full-model HTTP acceptance are still pending in this recorded stage.

A later read-only snapshot found 253.81 GiB of free buddy memory on that node,
but only 134.75 GiB in blocks of at least 2 MiB. On this 64 KiB-page kernel,
2 MiB corresponds to buddy order 5. The matching NVIDIA 580.95.05 allocator
uses a node-specific system-memory allocation path for EGM. Fragmentation or
insufficient reclaimable high-order pages is therefore a candidate, not a
proven root cause: the snapshot was taken after cleanup, and no failing-call
page-order trace was captured. See the official
[EGM allocator](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/src/kernel/mem_mgr/egm_mem.c)
and [page allocation implementation](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/kernel-open/nvidia/nv-vm.c).
The next full run should record free buddy blocks before each allocation and
use a suitable Owner host; changing global kernel memory settings is not part
of this experiment.

Use fresh NUMA, container and host measurements. Never drop global caches or
stop unrelated processes to make this test pass. The optional Owner argument
`--discard-private-checkpoint-cache` is only for an operator-controlled private
immutable checkpoint copy. It advises release of source file cache while
streaming; the kernel may retain pages, so the return value is not a proof of
reclaimed memory. Defaults do not discard checkpoint cache.

## Run order

Use the tested CUDA13 environment, the prebuilt native bridge supporting
128 GiB allocations, and the same model revision on both hosts. Select the
authorized GPU with `CUDA_VISIBLE_DEVICES`. All publication and oracle files
must be owned by the running user, mode `0600`, in mode `0700` directories.
Do not print their contents into logs.

1. Start `checkpoint_owner.py` with the real Owner checkpoint and explicit
   allocation/dataset limits as described in [FULL_MODEL_HTTP.md](FULL_MODEL_HTTP.md).
   On a private checkpoint copy, opt in to
   `--discard-private-checkpoint-cache`. Wait for **READY**; INSPECTED only
   proves metadata and budget validation.
2. Prepare a small independent CPU oracle on the Owner host:

   ```bash
   python tools/engram_sidecar/framework/checkpoint_lookup.py prepare \
     --checkpoint /workspace/checkpoints/owner \
     --bundle /workspace/private-owner/bundle.json \
     --private-inputs /workspace/private-oracle/oracle.json
   ```

   This reads only selected raw rows using seek/read, without materializing
   either full table. It binds the oracle to config/tokenizer hashes and the
   published dataset/generation/hash ABI. It cross-checks the Owner's published
   first/middle/last row hashes and verifies CUDA was not initialized.
3. Transfer the private bundle, manifest, capabilities and oracle through the
   existing authenticated channel. Do not copy Owner weight shards to the
   consumer.
4. Run the consumer in a fresh process:

   ```bash
   export RTP_ENGRAM_MAX_ALLOCATION_BYTES=137438953472
   export RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES=274877906944
   python tools/engram_sidecar/framework/checkpoint_lookup.py run \
     --checkpoint /workspace/checkpoints/consumer \
     --bundle /workspace/private-consumer/bundle.json \
     --private-inputs /workspace/private-oracle/oracle.json \
     --device 0
   ```

   The consumer revalidates the absence of full tables in actual file headers,
   attaches through `attach_from_bundle`, checks GPU reads against published
   source hashes, and invokes `HostEngramEmbedding.from_storage` with the
   existing Triton implementation. Cases cover first/middle/last rows, addresses
   around 4 GiB, negative indices, repeated indices, real-tokenizer prefill and
   decode-history windows, and both int64/int32 indices. An independent CPU
   FP8/E8M0 bit decoder supplies the expected BF16 result.
5. Require both PASS and CLOSED with zero native resource counters and confirm
   the consumer process has exited. Keep the Owner for the later full-model
   run, or create its STOP file only after **all** consumers have exited.

The probe reports PyTorch allocation changes, but those do not measure all
driver resources. No latency acceptance, CUDA Graph support or full-model
quality claim follows from a successful lookup probe.
