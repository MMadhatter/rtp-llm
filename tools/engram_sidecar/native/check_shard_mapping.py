#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU check for the sharded fabric mapping (run on a GB200 dev node).

Builds the development native bridge, publishes one or two small Owner
allocations, maps them through :class:`ShardedLayerMapping`, and verifies that
global row indexing lands in the right shard.

    python3 check_shard_mapping.py            # 1 shard
    python3 check_shard_mapping.py --shards 2 # 2 shards
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from load_native import load_native  # noqa: E402

from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards  # noqa: E402
from rtp_llm.models_py.modules.dsv41.engram_shard_storage import (  # noqa: E402
    ShardedLayerMapping,
)

DIM = 256
BLOCK = 32
ROWS_PER_SHARD = 262144  # one alignment unit: 64 MiB of weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    args = parser.parse_args()

    native = load_native(verbose=False)
    total_rows = ROWS_PER_SHARD * args.shards
    plan = plan_shards(total_rows, DIM, args.shards, block=BLOCK)
    print(f"plan: {plan.as_dict()}")

    owners, handles = [], []
    for shard in range(plan.shard_count):
        owner = native.Owner(plan.shard_bytes, args.numa, args.device)
        # Fill this shard's weight region with a per-shard byte pattern.
        pattern = bytes([shard + 1]) * (64 * 1024 * 1024)
        offset = 0
        while offset < plan.weight_bytes:
            chunk = min(len(pattern), plan.weight_bytes - offset)
            owner.load(pattern[:chunk], offset)
            offset += chunk
        owners.append(owner)
        handles.append(owner.export_handle())

    mapping = ShardedLayerMapping(plan, handles, args.device).attach()
    weight = mapping.weight_view()
    scale = mapping.scale_view()
    print(f"weight view: {tuple(weight.shape)} ptr=0x{weight.data_ptr():x}")
    print(f"scale  view: {tuple(scale.shape)} ptr=0x{scale.data_ptr():x}")

    ok = True
    for shard in range(plan.shard_count):
        start, end = plan.row_range(shard)
        probe_rows = [start, (start + end) // 2, end - 1]
        for row in probe_rows:
            got = int(weight[row, 0].item())
            if got != shard + 1:
                ok = False
                print(f"MISMATCH shard={shard} row={row} got={got} want={shard + 1}")
    # Padding rows must be readable (zeroed) and inside the last shard.
    last_padding = plan.padded_rows - 1
    print(f"padding row {last_padding} byte={int(weight[last_padding, 0].item())}")

    mapping.close()
    for owner in owners:
        owner.close()
    print("RESULT", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
