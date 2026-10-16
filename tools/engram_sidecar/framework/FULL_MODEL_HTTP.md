# Real DeepSeek V4.1 checkpoint / eager HTTP experiment

This step uses the complete checkpoint and the actual RTP server. The earlier
`checkpoint_fixture.py` / `consumer_model_hook.py` results prove the two-layer
Engram component path only; they do not prove an LLM, model quality, or HTTP.
These scripts do not implement service discovery, production service lifecycle,
automatic failover, performance acceptance, or CUDA Graph support.

## Supported source and real launch path

Baseline is public main `492ecf45b3ef55b22fd4882622abfdb2e2f664f2` plus this branch.
`MODEL_TYPE=deepseek_v41` selects `models/deepseek_v41.py:DeepSeekV41`;
its HF registration is exactly `DeepseekV41ForCausalLM`.

The full path is `start_server` → `start_backend_server` → `ModelFactory` →
`BaseModel.load` → `DeepSeekV41Model._load_extra_weights` →
`initialize_model_engram` → private bundle import / CUDA VMM mapping →
real attention/MoE/Engram forward → `RtpLLMOp` scheduler/RPC → OpenAI frontend.
The C++ `NormalModelInputGatherer` and `PyWrappedModel` provide the real token
history windows used by `_prepare_engram`, including prefill/decode transitions.

The unmodified main smoke test
`rtp_llm/test/smoke/deepseek_v41_contract_test.py` defaults to
`/mnt/nas1/hf/DeepSeek-V4.1-Flash`, and its BUILD target uses SM100 CUDA13,
TP4/EP4/DP1. It requires graph capture/replay, so it cannot be used unchanged for
the external eager-only path. The launcher here follows that four-GPU geometry,
with graph/speculation disabled and a single PDFUSION instance.

## Checkpoint contract and resources

Use the [streaming checkpoint staging tool](CHECKPOINT_STAGING.md) to construct
the Owner and consumer directories without copying full tables to the consumer.

The config needs real V4.1 `text_config`, `vision_config`, `quantization_config`,
`dtype`, token IDs, attention/compression/source-layer metadata and Engram
geometry. Text-only chat still uses the local ViT checkpoint loading path in
this configuration. Keep all backbone, MoE, vision, aligner and image delimiter
weights. Keep `tokenizer.json`, `tokenizer_config.json`, and particularly
`encoding/encoding.py`: the V4.1 OpenAI renderer imports that encoder.

For every configured Engram layer `L` the Owner loads:

| Tensor key | Shape | Safetensors source dtype |
|---|---|---|
| `layers.L.engram.embed.weight` | `[rows, head_dim]` | `F8_E4M3` |
| `layers.L.engram.embed.scale` | `[rows, head_dim/32]` | `U8` or `F8_E8M0` |

Each logical tensor is wholly in one safetensors shard. The current loader keeps
one EGM allocation per layer containing weight + scale; it does not split one
large table into several manifest shards. U8 scales are interpreted as E8M0 bits.
The consumer index omits both full-table entries and keeps per-layer
`engram.wkv.weight`, `engram.wkv.scale`, `engram.q_weight`, `engram.k_weight`,
plus **all** remaining model weights. WKV is FP8 E4M3 with 32×32 E8M0 scaling;
the q/k gate weights are BF16.

Compute Owner logical bytes exactly as
`sum(engram_num_embeddings) * (engram_head_dim + engram_head_dim // 32)`.
Add per-allocation CUDA granularity padding. The large default reference tables
are around 190 GiB in total; use the inspected checkpoint's exact numbers.
The new hard development ceilings are 128 GiB/allocation and 256 GiB/dataset,
not a statement that the selected host or container has that memory available.
Budget excludes file cache, loader RSS, driver metadata and all consumer model
weights. Neither a former 64 GiB container quota nor raising that quota proves
that the selected NUMA node can supply these tables. In the tested driver,
observed EGM RSS and container memory accounting differ; measure both.

For the consumer, four SM100 GPUs are the source-backed test configuration,
not a proved minimum. Check per-GPU HBM for TP/EP-sharded real weights, replicated
projection/vision tensors, runtime workspace and an explicit 8 GiB KV budget.
The launcher reserves 32 GiB runtime memory as in the main smoke target.
Owner and consumer must be in the validated NVLink fabric domain with matching
IMEX channel access and required devices visible. Roles are not tied to an SN.

## Run order

Use the prepared exact-main CUDA13 venv and its library paths on each host.
Use trusted, immutable checkpoint directories. The commands below are examples;
the operator provides the private publication directory and checkpoint identity.
No script creates the other host's environment or transfers private handles.

1. **Before the full HTTP experiment**, verify the complete runtime on the consumer.
   The separate [real-table lookup experiment](REAL_CHECKPOINT_LOOKUP.md) can run
   while the complete engine is being built:

   ```bash
   python tools/engram_sidecar/framework/full_model_http.py runtime --load-native
   ```

   This requires `rtp_llm/libs/libth_transformer_config.so`,
   `librtp_compute_ops.so`, `libth_transformer.so`, and
   `libth_grammar_tokenizer_info.so` from the same main build. The latter two
   targets are `//:th_transformer` and `//:th_grammar_tokenizer_info`.
   Grammar metadata is built at startup for every real language model, even
   when the request does not ask for a grammar. An older image's engine `.so`
   is not a valid substitute. The runtime command checks package discovery,
   ELF dependency resolution, prints SHA256 and optionally performs real native
   and server imports in a child with GPUs hidden. Import success still does
   not prove every later JIT/model path is usable.

2. Start the real Owner only after platform memory/NUMA/IMEX checks succeed:

   ```bash
   python tools/engram_sidecar/framework/checkpoint_owner.py \
     --checkpoint /workspace/engram_slice7/checkpoints/owner \
     --model-revision DeepSeek-V4.1-Flash-verified-checkpoint \
     --publication /workspace/engram_slice7/private-owner \
     --device 0 --numa 0 \
     --max-allocation-bytes 137438953472 \
     --budget-bytes 274877906944
   ```

   The revision is an explicit operator identity; it does not certify a public
   release. Record `INSPECTED` config/index SHA256 and real tokenizer/hash ABI.
   `READY` means the complete tables have loaded and the immutable manifest and
   private capabilities have been published. The manifest binds tensor bytes.
   Fixture mode remains `--fixture-root PATH`, with original defaults of 1 GiB
   per allocation and 64 MiB Owner budget; never call it a full checkpoint.
   For an operator-owned private checkpoint copy, the optional
   `--discard-private-checkpoint-cache` reduces retained source file cache while
   streaming. It is a kernel advice, not a capacity guarantee; keep monitoring
   NUMA and container memory, and leave it off for shared checkpoint files.

3. Transfer `bundle.json`, `manifest.json`, `capabilities.json` over the existing
   authenticated channel into the consumer's private directory. Keep files
   owned by the consumer UID, mode `0600`, directory `0700`. Do not print/copy
   capabilities into logs or reports. Verify checkpoint provenance against the
   Owner/source before supplying the same revision string.

4. Run the complete consumer screen (CPU tokenizer hashing, no EGM import):

   ```bash
   python tools/engram_sidecar/framework/full_model_http.py preflight \
     --checkpoint /workspace/engram_slice7/checkpoints/consumer \
     --bundle /workspace/engram_slice7/private-consumer/bundle.json \
     --model-revision DeepSeek-V4.1-Flash-verified-checkpoint \
     --max-allocation-bytes 137438953472 --budget-bytes 274877906944
   ```

   It rejects component fixtures, missing full-layer backbone/vision categories,
   missing encoder/tokenizer assets, missing local Engram projections, full-table
   entries in the consumer index, missing shard files, mismatching real hash ABI,
   revision mismatch and invalid private files. This metadata screen does not
   claim every tensor key/shape has already passed the real model loader.

5. Replace `preflight` with `launch` in the preceding command. Set the intended
   four GPU ordinals using `CUDA_VISIBLE_DEVICES` in the launcher environment.
   `launch` repeats preflight and the real native import probe before `exec` of
   `python -m rtp_llm.start_server`. It does not daemonize, auto-restart, detach
   EGM or create Owner STOP. It fixes TP4/EP4/PDFUSION, BF16, no graph, no draft,
   no layer truncation and no inherited service-discovery config.

   The experiment explicitly sets `FASTSAFETENSORS_NOGDS=1` while keeping
   `LOAD_METHOD=fastsafetensors`. Main's database adapter then selects the
   `nogds` file copier with `use_shm=False`. The locked `fastsafetensors` Python
   package can import without the separate `fast_safetensors` native shared-memory
   helper, which is absent in this development venv; the default shared-memory
   copier would require it at weight-loading time. This selects the existing
   non-GDS loading path, without installing a different native ABI. It affects
   checkpoint loading, not the external Engram CUDA VMM/NVLink lookup path.

   This is a colocated PDFUSION frontend/backend. With `MODEL_SERVICE_CONFIG`
   absent, `HostService.service_available` is false, so the request skips domain
   routing and `ModelRpcClient` uses local world-info rank RPC addresses. The
   non-separated frontend health check also checks its backend directly. Main
   does not synthesize a `MODEL_SERVICE_CONFIG`; `USE_LOCAL=1` alone does not
   configure `HostServiceArgs`, which reads that JSON's `use_local` field. The
   launcher fixes `USE_LOCAL=1` for compatible local paths and removes old
   remote RPC/ViT and CM2 settings, while leaving `MODEL_SERVICE_CONFIG` absent.

6. After `/health` is ready, send one short real OpenAI request:

   ```bash
   python tools/engram_sidecar/framework/full_model_http.py chat --port 18630
   ```

   HTTP 200 alone is insufficient: the helper rejects an error payload or empty
   user-visible message. The launcher fixes `THINK_MODE=0`; the request also sends
   `"thinking":{"type":"disabled"}`, the explicit schema accepted by main's
   V4.1 `resolve_thinking`. This prevents a 32-token smoke from spending its entire
   budget on reasoning without producing content. Inspect the returned text, then compare deterministic
   input/output token IDs and numerical behavior with the local-table baseline.
   Retain logs showing all configured Engram layers attached externally on each
   rank; verify consumer table files/entries were absent and EGM was shared.

7. Stop and confirm **every consumer worker has exited**. Only then create
   `private-owner/STOP`. Owner retains backing on timeout, SIGINT/SIGTERM after
   publication, or lost output pipe; these are not permission to free live data.
   There is no distributed lease proof in this experimental file protocol.

## Acceptance matrix

| Test | What it proves | What it cannot prove |
|---|---|---|
| New CPU Owner/preflight unit tests | Input mode, provenance, budgets, private publication, missing-asset/runtime rejection | GPU or LLM correctness |
| Existing v2 cross-host component suite | Real tokenizer/hash, EGM lookup, actual model hook, native WKV/gated residual | Full attention/MoE/scheduler/HTTP |
| Complete V4.1 eager HTTP + table-free consumer | Real full model and request plumbing over shared EGM | Local-baseline numerical equivalence or latency SLO by itself |
| Same checkpoint local-table vs external deterministic requests | Token/output comparison, cache/history regression evidence | General quality/performance at every batch |
| Multi-request decode/prefill/reuse and independent consumers | Real history transitions, lifetimes and concurrent sharing | Graph, PD, speculative decoding or failover |

Small Qwen models can verify the generic HTTP environment but do not execute
this Engram integration. No supported small full V4.1 checkpoint was identified
in the audited main tree. `HACK_LAYER_NUM` reduces `ModelConfig.num_layers` but
does not trim `deepseek_v41_config.engram_layer_ids`/the Owner dataset and is not
a full-model validation. Arbitrarily shrinking the fixture dimensions into a
purported LLM is unsafe: V4.1 attention/cache layouts and optimized kernels have
real geometry constraints.
