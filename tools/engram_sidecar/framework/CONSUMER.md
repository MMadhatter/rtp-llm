# Real model-hook consumer

GPU acceptance for this script is pending in this checkpoint. The completed
bootstrap/hash/lookup test is documented separately in
[framework validation](../FRAMEWORK_VALIDATION.md).

`consumer_model_hook.py` calls the real `DeepSeekV41Model._load_extra_weights()`
and `_prepare_engram()` on a small shell containing only their required fields.
Every tested layer is constructed by actual `Engram.from_checkpoint()` and its
actual `Engram.forward()` runs external lookup, `V41MXFP8Linear`, native RTP
group-32 quantization, DeepGEMM projection and gated residual.

This is an Engram framework integration test. It does not construct attention,
MoE, scheduler, full `DeepSeekV41Model.forward()`, or an HTTP service. There is no
projection shim and no fake `compute_ops`. Missing genuine native libraries is
a failure; the script never substitutes an approximate implementation.

Before invoking it, the launcher must start the Owner, transfer private bootstrap
files, and place only these fixture files on the consumer host/container:

- `consumer_checkpoint/`
- `config.json`, `fixture.json`, `oracle_inputs.safetensors`

The script refuses a fixture root containing `owner_checkpoint/`, embedding
entries in the consumer index, or undeclared safetensors files. It checks file
digests before model loading. CPU expectations generate only selected rows with
`checkpoint_fixture.table_rows()` and read small consumer-local WKV/q/k weights.

```bash
# Activate the validated CUDA 13 venv, then run from the repository root.
# Launcher sets CUDA_VISIBLE_DEVICES to an authorized idle GPU.
python tools/engram_sidecar/framework/consumer_model_hook.py \
  --fixture-root /absolute/private/engram-consumer-fixture-01 \
  --bundle /absolute/private/engram-private-bootstrap-01/bundle.json \
  --device 0
```

The genuine `libth_transformer_config` and `librtp_compute_ops` extensions must
be importable from the checkout `rtp_llm/libs`, top-level `bazel-bin`, or an
explicit `PYTHONPATH`. The prebuilt Engram native extension must also be available
as required by `engram_bootstrap`. Runtime dynamic dependencies must be resolved
by the existing development environment. Import failures are not skipped.

Checks cover repeated model loading without remapping, unmanaged Graph rejection,
three decode histories, mixed masked/boundary input, 33-token prefill and empty
batch. Hash IDs, masks, selected table rows and masked outputs must match exactly.

For WKV and gated output, the fixed per-element tolerance is two BF16 ULPs at the
reference exponent, with `1e-5` as the near-zero absolute floor. BF16 has seven
fraction bits; the ULP is `2**(frexp_exponent - 8)`. The two-ULP allowance covers
separate BF16 rounding and CPU/GPU FP32 reduction/transcendental differences;
it is set before measurements, not tuned to observed errors. Zero uses only the
absolute floor. Nonfinite values always fail. Every case reports maximum absolute
error and maximum error/bound ratio. This tolerance is a small-fixture criterion,
not an accuracy promise for the full model.

A successful run emits `consumer_complete` with `layer_cases=12`, the bound
dataset/generation, and explicit scope limitations. It makes no latency claim.
Mappings remain retained for process lifetime. The launcher must wait for the
consumer process to exit before permitting Owner teardown, including after a
failure. A success log alone is not proof that the consumer exited.

`test_consumer_model_hook.py` covers file/native preflight and comparison bounds
on CPU. It never mocks or executes production forward. GPU/remote results must
come from the real consumer script.
