#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Development coordination server; policy describes platform-authorized resources."""
import argparse
import json
import signal
import threading
from pathlib import Path

from rtp_llm.models_py.modules.dsv41.engram_registry import (
    RegistryServer,
    SqlitePlatformAdapter,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listen", required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--tls-ca", type=Path, required=True)
    p.add_argument("--tls-cert", type=Path, required=True)
    p.add_argument("--tls-key", type=Path, required=True)
    p.add_argument("--lease-seconds", type=int, default=30)
    args = p.parse_args()
    adapter = SqlitePlatformAdapter(
        args.database,
        json.loads(args.policy.read_text()),
        lease_seconds=args.lease_seconds,
    )
    server = RegistryServer(adapter)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        server.start(args.listen, ca=args.tls_ca, cert=args.tls_cert, key=args.tls_key)
        print(
            "REGISTRY_READY adapter=single_node_sqlite automatic_takeover=false",
            flush=True,
        )
        stop.wait()
    finally:
        server.stop()
        adapter.close()


if __name__ == "__main__":
    main()
