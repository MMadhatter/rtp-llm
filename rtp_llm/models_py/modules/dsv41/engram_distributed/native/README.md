# Native EGM / ATen bridge

`engram_native.cc` implements HOST_NUMA + FABRIC allocations and imported uint8 ATen views.
CUDA13 BUILD targets compile the bridge into `rtp_llm/libs/libengram_native.so`,
which is included in the RTP wheel. Serving uses `rtp_llm.ops.load_engram_ops()`;
the standalone loader below is only for development tests.

Build in a Linux CUDA 13 environment with Torch `2.11.0+cu130`:

```bash
python rtp_llm/models_py/modules/dsv41/engram_distributed/native/load_native.py --verbose
```

`CUDA_HOME` selects driver headers. `CUDA_DRIVER_LINK_DIR` may override the link-only
`libcuda.so` stub directory. Do not add stubs to runtime library paths. Put the resulting
module directory on serving `PYTHONPATH`. Importing the loader does not build or initialize CUDA.

The native ceiling is 128 GiB/allocation and 64 MiB/load chunk. Actual local shard budgets
are checked before allocation; host/NUMA capacity still needs deployment verification.
Each publishing worker keeps its Owner until process exit. Imported tensor aliases retain
mapping references. Model decode Graph uses permanent storage pins; no standalone Graph
executor is used by RTP. Never unmap while consumers/graphs still reference the storage.

Retained validation tools:

- `../test/test_native.py`: nine real VMM create/export/import/view/release, fault-boundary and resource-counter tests.
  The locked Fabric gate supplies its already loaded bridge and fixtures; no second build or GPU process is needed.
  Its standalone `main()` still accepts exactly one visible GPU and builds through `load_native.py`.
- `../test/test_embedding_contract.py`: CPU Torch tests of the model's external tensor views and no-copy behavior.
  The standard `test.sh --torch` runner invokes its standalone `main()` and propagates failures.
  Bazel entry: `//rtp_llm/models_py/modules/dsv41/engram_distributed/test:test_embedding_contract` (CPU execution with the
  CUDA13 ARM Torch dependency set; no DeviceResource/pytest GPU wrapper).
- `../test/run_fabric_ci.py`: explicit two-GPU gate, compatible GPU file locks without PID cleanup,
  actual HOST_NUMA/Fabric export/import capability check and strict no-skip runner for all nine native
  cases plus the two lifecycle/lookup integration cases.
- `../test/test_fabric_lookup_cuda.py`: real P=2 Fabric owners → validated storage →
  `HostEngramEmbedding.from_storage` → production BF16 lookup kernel / CUDA Graph.
  Also covers P=2/P=3 contiguous mappings, aliases and raw CUDA Graph replay;
  the CI runner maps the second consumer onto a second GPU.
  It changes IDs across both shards and checks reference results for int32/int64 non-24-column IDs,
  with mapped scales and the default Blackwell GPU scale copy, then checks Graph pins, aliases and release.
  Its two independent owners are local fixtures; it does not establish cross-host transport correctness.
- `probe_static_exchange_cuda.py`: two-host synthetic startup/mapping probe; parent-held allocations
  are test fixtures, not a supported owner deployment. It needs a common `--cohort` of two distinct IPs.

Example opt-in two-GPU regression using an already built bridge:

```bash
CUDA_VISIBLE_DEVICES=0,1 ENGRAM_RUN_CUDA_TESTS=1 ENGRAM_TEST_NUMA=0 \
bash tools/engram_distributed/test.sh --cuda-fabric
```

The lookup test requires a compatible Torch/Triton environment, the prebuilt bridge on `PYTHONPATH`,
and working HOST_NUMA/FABRIC/IMEX access. It allocates at most 256 MiB of host backing plus a small
device scale copy. The runner refuses occupied/locked devices and never restarts services or
cleans up other GPU processes. `ENGRAM_TEST_BUILD_NATIVE=1` explicitly builds this source's
bridge in temporary test scratch.
Without `ENGRAM_RUN_CUDA_TESTS=1`, the runner refuses execution. Missing prerequisites return 2;
test failures, zero tests and skips return 1. Neither is reported as a pass.
GPU selection accepts full UUIDs or physical `nvidia-smi` indices; the runner keeps index-based
locks and PID checks, then passes the corresponding full UUIDs to CUDA to avoid ordinal mismatch.

The Bazel Fabric target is `manual`. An explicit invocation must override the default
process-cleaning wrapper with `--run_under=`; `run_fabric_ci.py` owns the safe GPU locks.
The target sets `GPU_COUNT=0` so an accidental invocation with the default wrapper hides CUDA
and fails its capability gate instead of cleaning up GPU processes. The runner probes real
Fabric capability rather than treating the `SM100_ARM_CU13` scheduling tag as sufficient.
CPU tests are selected by the existing CI wildcard and tag filters; no Engram-specific
pipeline parameters or jobs are required.

These tests do not establish full-model correctness or <100us decode overhead.
The small lookup regression disables the attention schedule while exercising production Graph ownership;
scheduled decode prefetch, DSpARK and cross-host model output still require the P/D end-to-end test.
