"""Launch either PD role from one shared, hashed experiment configuration.

The pair file contains addresses and device UUIDs, never Fabric capabilities.
Both roles use identical KV geometry. This is a bounded TP4/CP1 experiment,
not a general production deployment controller.
"""

import argparse
import hashlib
import ipaddress
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.engram_sidecar.framework import full_model_http

ROLES = ("PREFILL", "DECODE")
COMMON_KEYS = {
    "MODEL_TYPE",
    "ACT_TYPE",
    "FP8_KV_CACHE",
    "SEQ_SIZE_PER_BLOCK",
    "KERNEL_SEQ_SIZE_PER_BLOCK",
    "TP_SIZE",
    "EP_SIZE",
    "DP_SIZE",
    "WORLD_SIZE",
    "PREFILL_CP_SIZE",
    "PREFILL_CP_KV_CACHE_SHARDED",
    "CACHE_STORE_RDMA_MODE",
    "MAX_SEQ_LEN",
    "MAX_CONTEXT_BATCH_SIZE",
    "CONCURRENCY_LIMIT",
    "REUSE_CACHE",
    "DSV41_CPU_ENGRAM_LOOKUP",
    "DSV41_ASYNC_ENGRAM_LOOKUP",
}


def validate_pair(pair):
    if set(pair) != {"schema", "seq_size_per_block", "roles"} or pair["schema"] != 1:
        raise ValueError("unsupported pair schema or unknown keys")
    block = pair["seq_size_per_block"]
    if type(block) is not int or block not in (128, 256):
        raise ValueError("seq_size_per_block must be 128 or 256")
    if set(pair["roles"]) != set(ROLES):
        raise ValueError("both PREFILL and DECODE are required")
    for role in ROLES:
        endpoint = pair["roles"][role]
        if set(endpoint) != {"ip", "port", "gpu_uuids"}:
            raise ValueError("role overrides are limited to address, port and GPUs")
        # The current route renderer is IPv4-only; do not emit ambiguous IPv6 URLs.
        ipaddress.IPv4Address(endpoint["ip"])
        port = endpoint["port"]
        if type(port) is not int or not 1024 <= port <= 65490:
            raise ValueError("invalid service port")
        uuids = endpoint["gpu_uuids"]
        if not isinstance(uuids, list) or len(set(uuids)) != 4 or len(uuids) != 4:
            raise ValueError("this experiment requires four distinct GPU UUIDs")
        if any(not isinstance(x, str) or not x.startswith("GPU-") for x in uuids):
            raise ValueError("GPU UUIDs are required, not ordinal device IDs")
    if pair["roles"]["PREFILL"]["ip"] == pair["roles"]["DECODE"]["ip"]:
        raise ValueError("cross-host experiment requires distinct hosts")
    return hashlib.sha256(
        json.dumps(pair, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def launch_environment(pair, role, args, inherited=None):
    pair_sha = validate_pair(pair)
    if role not in ROLES:
        raise ValueError("invalid role")
    this = pair["roles"][role]
    peer = pair["roles"]["DECODE" if role == "PREFILL" else "PREFILL"]
    graph = role == "DECODE" and args.cuda_graph
    env = full_model_http.launch_environment(
        SimpleNamespace(
            checkpoint=args.checkpoint,
            bundle=args.bundle,
            storage_mode=args.storage_mode,
            cuda_graph=graph,
            max_allocation_bytes=args.max_allocation_bytes,
            budget_bytes=args.budget_bytes,
            port=this["port"],
        ),
        inherited=inherited,
    )
    endpoints = {
        r.lower()
        + "_endpoint": {
            "type": "Vipserver",
            "address": f'{e["ip"]}:{e["port"]}',
            "protocol": "http",
            "path": "/",
        }
        for r, e in pair["roles"].items()
    }
    env.update(
        {
            "ROLE_TYPE": role,
            "CUDA_VISIBLE_DEVICES": ",".join(this["gpu_uuids"]),
            "START_PORT": str(this["port"]),
            "REMOTE_SERVER_PORT": str(peer["port"]),
            "REMOTE_RPC_SERVER_IP": peer["ip"],
            "MODEL_SERVICE_CONFIG": json.dumps(
                {
                    "service_id": "engram_pd",
                    "use_local": True,
                    "role_endpoints": [{"group": "default", **endpoints}],
                }
            ),
            "SEQ_SIZE_PER_BLOCK": str(pair["seq_size_per_block"]),
            "KERNEL_SEQ_SIZE_PER_BLOCK": "128",
            "PREFILL_CP_SIZE": "1",
            "PREFILL_CP_KV_CACHE_SHARDED": "0",
            "CACHE_STORE_RDMA_MODE": "0",
            "CONCURRENCY_LIMIT": "4",
            "RTP_ENGRAM_PAIR_SHA256": pair_sha,
        }
    )
    return env


def check_local_devices(uuids):
    available = set(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).split()
    )
    if not set(uuids).issubset(available):
        raise ValueError("pair file contains GPU UUIDs absent from this host")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", type=Path, required=True)
    p.add_argument("--role", choices=ROLES, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument(
        "--storage-mode", choices=full_model_http.STORAGE_MODES, default="external-egm"
    )
    p.add_argument("--max-allocation-bytes", type=int, required=True)
    p.add_argument("--budget-bytes", type=int, required=True)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.storage_mode == "external-egm" and args.bundle is None:
        p.error("external-egm requires --bundle")
    pair = json.loads(args.pair.read_text())
    env = launch_environment(pair, args.role, args)
    check_local_devices(pair["roles"][args.role]["gpu_uuids"])
    # Keep capabilities, inherited credentials and bundle contents out of logs.
    print(
        json.dumps(
            {
                "role": args.role,
                "pair_sha256": env["RTP_ENGRAM_PAIR_SHA256"],
                "common": {k: env[k] for k in sorted(COMMON_KEYS)},
                "cuda_graph": env["ENABLE_CUDA_GRAPH"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.dry_run:
        os.execve(sys.executable, [sys.executable, "-m", "rtp_llm.start_server"], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
