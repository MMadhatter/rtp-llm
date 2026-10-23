#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render TP4 P/D model env files; Engram RPC settings belong to consumer config.

Run from the source checkout with --pair PAIR.json --output NEW_DIRECTORY.
The existing paired_pd schema defines both hosts, frontend ports and GPU UUIDs.
Only decode captures a batch-1 CUDA Graph when explicitly enabled.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.engram_sidecar.framework.paired_pd import ROLES, launch_environment


def render(pair, role, *, cuda_graph=False):
    args = SimpleNamespace(
        checkpoint=Path("/checkpoint"),
        bundle=None,
        storage_mode="local-pinned",
        cuda_graph=cuda_graph,
        max_allocation_bytes=1,
        budget_bytes=1,
    )
    env = launch_environment(pair, role, args, inherited={})
    # Deployment chooses local/shared/auto and supplies RPC identity separately.
    # Never propagate a checkout's PYTHONPATH or historical capability file.
    env = {
        key: value
        for key, value in env.items()
        if not key.startswith("RTP_ENGRAM_") and key != "PYTHONPATH"
    }
    if any("\n" in value or "\r" in value for value in env.values()):
        raise ValueError("env values must be single-line")
    return "".join(f"{key}={value}\n" for key, value in sorted(env.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decode-cuda-graph", action="store_true")
    args = p.parse_args()
    pair = json.loads(args.pair.read_text())
    outputs = {
        role: render(pair, role, cuda_graph=args.decode_cuda_graph) for role in ROLES
    }
    args.output.mkdir(mode=0o700)  # Never overwrite an operator's active config.
    for role, text in outputs.items():
        (args.output / (role.lower() + ".env")).write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
