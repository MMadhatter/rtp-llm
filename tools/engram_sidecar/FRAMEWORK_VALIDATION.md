# Framework integration checkpoint

This checkpoint builds on the [shared-storage component checkpoint](CHECKPOINT_VALIDATION.md)
from public main `492ecf45b3ef55b22fd4882622abfdb2e2f664f2`.
It adds an experimental file bootstrap and model loading hook. It contains no
service layer, platform injection, Owner RPC, failover or live generation switch.

## Runtime path and trust boundary

```mermaid
sequenceDiagram
  participant L as Trusted launcher
  participant O as Test Owner, host A/container A
  participant C as RTP consumer, host B/container B
  participant G as Consumer GPU
  O->>O: Inspect safetensors; create and load HOST_NUMA/FABRIC allocations
  O->>L: Private bundle + manifest + opaque capabilities
  L->>C: Authenticated transfer into private files
  C->>C: Actual tokenizer/hash ABI + layout checks
  C->>G: Import/map and verify selected rows
  G-->>C: GPU samples match checkpoint-derived digests
  C->>C: Construct external Engram layers transactionally
  G->>O: Direct memory reads through mapped fabric path
  Note over G,O: No lookup RPC or Owner CPU request handler
```

`RTP_ENGRAM_EXTERNAL_BUNDLE` is an absolute path read once during initialization.
When absent, legacy loading remains the default. The trusted launcher must choose
the correct bundle, authenticate its transfer, protect its files and retain the
Owner until every consumer process has exited. Bundle files must be regular,
current-UID files with no group/other access; final-component symlinks are refused.
These checks do not replace authenticated distribution or a platform trust model.

The bundle pins a manifest digest, generation, dataset, model revision, hash ABI
and required read samples. The consumer recomputes actual tokenizer/hash
semantics and validates the complete layout before mapping. **Dataset and model
revision are supplied by the trusted bundle; they are not yet independently
pinned by local model configuration.** A launcher selecting the wrong bundle with
compatible geometry/hash semantics is outside the implemented protection.

`engram_model_attach.py` attaches once, constructs all selected model layers in a
temporary list and publishes only after complete success. Partial failure releases
views before closing storage; incomplete cleanup retains resources and fails.
There is no silent private-table fallback or in-process retry after failed init.

The model integration is **eager only**. External mode rejects configured engine
Graph use, Graph preparation and capture detected at forward entry. Successful
model mappings are retained for worker-process lifetime. There is no live detach,
model unload or engine-wide Graph cleanup contract. The older isolated guarded
Graph harness does not establish Graph support for this model hook.

## What was exercised on two hosts/containers

| Check | Result | Scope |
| --- | --- | --- |
| Real tokenizer and safetensors fixture | Passed | Actual tokenizer reload and tensor dtype/shape checks; two Engram layers |
| Bootstrap UT | 29 passed on each host, zero skipped | File/identity/hash and cleanup contracts; GPU mapping mocked where appropriate |
| Model attachment UT | 19 passed on each host, zero skipped | Transactional initialization, legacy behavior and eager-only guards |
| Model-hook harness CPU UT | 15 passed on the consumer host, zero skipped | Fixture/native preflight and fixed numerical-comparison bounds; no model forward |
| Cross-host bootstrap, hash and lookup | 12 layer/case checks passed exactly | Real mapped GPU reads and RTP hash/lookup; excludes WKV/model hook |
| Bad sample digest fault | Rejected with `GPU_READ_SAMPLE_MISMATCH` | Real remote GPU reads fail verification; consumer native counters return to zero |
| Actual model hook and layer forward | **Pending GPU acceptance** | Requires genuine RTP native libraries and original quantization/DeepGEMM path |
| Full model/engine/HTTP | **Not tested** | No attention/MoE/scheduler or serving acceptance |

The deterministic fixture has a real 32-token tokenizer, two layers with
4,097 and 514 embedding rows, 64 columns and group-32 scales. The owner checkpoint
contains 12 tensors. The consumer checkpoint contains only 8 local WKV/q/k
tensors plus tokenizer/configuration. Six input cases cover three decode
histories, mixed masked/boundary input, 33-token prefill and an empty batch.

At GPU-test time the consumer fixture had no `owner_checkpoint` directory or
embedding safetensors. During setup it had initially been generated locally;
its task-created owner checkpoint was hash-checked and removed before GPU tests.
This result must not be represented as proof no complete fixture table had ever
existed on the consumer. Retained consumer files matched the owner's corresponding
files byte for byte.

The actual cross-host `bootstrap_probe.py` run checked real tokenizer hash IDs
against saved CPU expectations, then checked both layers' lookup outputs against
independently generated selected-row expectations. All 6 × 2 checks matched
exactly. The Owner held **2 aligned allocations totaling 4 MiB**. Consumer attach
changed current `torch.cuda.memory_allocated()` by **0 bytes**. This is not peak
memory, driver/page-table overhead, full-model capacity or total-RSS accounting;
small verification/output buffers are allowed. Consumer cleanup and process exit
were confirmed, with every native resource/failure counter zero.

The fault run intentionally changed an expected sample digest. Real GPU reads
then failed admission with `GPU_READ_SAMPLE_MISMATCH`; cleanup returned every
native counter to zero and the probe process exited successfully after verifying
the expected failure. This tests incorrect-content rejection, not hardware fault
recovery. The same Owner remained alive for subsequent tests; these consumer
results do not imply Owner teardown has already occurred.

## Code and reproducibility

- `engram_bootstrap.py`: strict private-file bootstrap, hash ABI binding, sample
  verification and process-lifetime retention.
- `engram_model_attach.py` and `DeepSeekV41Model`: opt-in transactional model hook,
  repeated-initialization reuse and unsupported-Graph rejection.
- [Fixture setup](framework/FIXTURE.md): deterministic tokenizer and checkpoints.
- `framework/checkpoint_owner.py`: immutable allocation owner; explicit STOP only
  after confirmed consumer exit.
- `framework/bootstrap_probe.py`: the completed cross-host hash/lookup test.
- [Model-hook consumer](framework/CONSUMER.md): the next GPU test; missing real
  native libraries fails instead of using a projection shim or fake engine ops.

The publication tree preserves the exact tested Python source hashes. Its
19 model-attachment tests were also rerun locally on Python 3.10.20; all 10
changed/new Python files parsed as Python 3.10 and the diff passed whitespace
checks. Public documents summarize results without copying environment logs.

New Bazel CPU targets describe the narrow dependencies for bootstrap and model
attachment tests. **Bazel execution is not yet a reported validation result.**
Native build and dependency readiness are separate from Python UT results.
Environment-specific logs, host identifiers and private capabilities are kept
outside this public repository.

## Development limits and remaining gates

The native bridge still limits each allocation to 1 GiB and each load chunk to
64 MiB. File bootstrap allows at most 8 GiB total mapped allocations; the test
Owner additionally uses a 64 MiB loading budget. These are development bounds,
not production Engram capacity claims. No compilation occurs in forward: native
extensions must be built and importable before launching consumers.

Before reporting framework integration complete, run genuine model-hook GPU
acceptance through `Engram.from_checkpoint`, original lookup/quantization,
`V41MXFP8Linear`, DeepGEMM and gated residual. Then test the complete engine,
all required ranks, a real model checkpoint and two independent RTP deployments.
Engine Graph, sCR, unload/reload and Owner/host failure recovery need explicit
lifetimes and additional acceptance work.

The additional **100-microsecond** target remains unpassed. These new correctness
runs make no latency claim and do not replace same-format pinned-host paired
measurements, complete-step TTFT/TPOT or intended-concurrency testing.
