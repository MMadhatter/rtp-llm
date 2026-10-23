#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-container RPC -> real EGM mapping -> RTP lookup -> CUDA Graph probe.

Uses only checkpoint config/tokenizer on the consumer. Sample byte hashes bind
reads to the Owner checkpoint. This is an Engram component test, not full LLM
HTTP correctness or a controlled local-vs-remote latency benchmark.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    from rtp_llm.models_py.modules.dsv41.engram_mode import prepare_engram_startup

    begin = time.monotonic()
    prepare_engram_startup()
    import torch
    from rtp_llm.models_py.modules.dsv41.engram import (
        EngramLayout,
        NgramHashState,
        HostEngramEmbedding,
        _scale_to_float,
    )
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import (
        attach_from_rpc,
        _load_native,
    )
    from rtp_llm.models_py.modules.dsv41.engram_storage import CapturedEngramGraph
    from rtp_llm.models_py.modules.dsv41.engram_rpc import Client

    raw = json.loads((args.checkpoint / "config.json").read_text())
    config = {
        **raw,
        **raw.get("text_config", {}),
        "checkpoint_path": str(args.checkpoint),
    }
    layout = EngramLayout(config)
    state = NgramHashState(layout, str(args.checkpoint))
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    native = _load_native()
    assert all(n == 0 for n in native.resource_counters().values())
    storage = attach_from_rpc(layout=layout, hash_state=state, device=device)
    info = storage.info()
    print(
        json.dumps(
            dict(
                stage="RPC_GPU_ATTACH_OK",
                generation=info["generation"],
                dataset_id=info["dataset_id"],
                physical_bytes=info["physical_bytes"],
                allocation_count=info["allocation_count"],
                startup_seconds=time.monotonic() - begin,
            )
        ),
        flush=True,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("online lookup attempted control RPC")

    Client.call = forbidden
    tests = []
    for layer_id in layout.layer_ids:
        embedding = HostEngramEmbedding.from_storage(storage, layer_id)
        rows = embedding.table_shape[0]
        selected = [0, rows // 2, rows - 1]
        index = torch.tensor([selected + [-1]], dtype=torch.int64, device=device)
        # Imported bytes were already compared against checkpoint row hashes.
        # Independent CPU arithmetic is the dequantization oracle.
        weight, scale = storage.table_views(layer_id)
        ids = torch.tensor(selected, device=device)
        cpu_weight = weight.index_select(0, ids).cpu().view(torch.float8_e4m3fn).float()
        cpu_scale = _scale_to_float(scale.index_select(0, ids).cpu())
        expected = (cpu_weight * cpu_scale.repeat_interleave(32, dim=1)).to(
            torch.bfloat16
        )
        expected = torch.cat((expected, torch.zeros_like(expected[:1])), dim=0)[None]
        del weight, scale
        with torch.inference_mode():
            eager = embedding(index, device)
            torch.testing.assert_close(eager.cpu(), expected, rtol=0, atol=0)
            graph = CapturedEngramGraph.capture(
                lambda: embedding(index, device), [storage]
            )
            for _ in range(20):
                actual = graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            graph.close()
            del graph, actual, eager
        tests.append(
            dict(layer=layer_id, rows=selected, graph_replays=20, exact_bf16=True)
        )
        embedding.release_external_views()
        del embedding
    gc.collect()
    storage.close()
    counters = native.resource_counters()
    assert all(n == 0 for n in counters.values()), counters
    print(
        json.dumps(
            dict(
                stage="RPC_LOOKUP_GRAPH_OK",
                tests=tests,
                online_rpc_calls=0,
                counters=counters,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
