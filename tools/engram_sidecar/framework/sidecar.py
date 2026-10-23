#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One sidecar entrypoint: authorized producer or lightweight consumer observer.

Platform injects registry identity and TLS through env. RTP workers separately
resolve and import handles; sidecar never forwards per-request data or pointers.
"""
import argparse
import json
import signal
import threading

from rtp_llm.models_py.modules.dsv41.engram_producer import ProducerSession
from rtp_llm.models_py.modules.dsv41.engram_registry import RegistryClient


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--required-memory-bytes", type=int, required=True)
    p.add_argument("--role", choices=("auto", "producer", "consumer"), default="auto")
    p.add_argument("owner_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    session = None
    if args.role != "consumer":
        session, grant = ProducerSession.acquire(args.required_memory_bytes)
        if session is None and args.role == "producer":
            raise RuntimeError("PLATFORM_DID_NOT_GRANT_PRODUCER")
    if session is not None:
        try:
            from tools.engram_sidecar.framework.checkpoint_owner import (
                main as owner_main,
            )

            owner_args = (
                args.owner_args[1:]
                if args.owner_args[:1] == ["--"]
                else args.owner_args
            )
            if "--rpc-listen" not in owner_args:
                raise ValueError("coordinated producer requires Owner RPC")
            print(
                json.dumps({"role": "PRODUCER", "owner_epoch": session.epoch}),
                flush=True,
            )
            return owner_main(owner_args, producer_session=session)
        finally:
            session.close()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    client, previous = RegistryClient(), None
    try:
        while not stop.is_set():
            state = client.resolve()
            # Handles never enter logs; only readiness/identity changes do.
            view = {
                k: state[k]
                for k in ("state", "producer_instance_id", "owner_epoch", "generation")
                if k in state
            }
            if view != previous:
                print(json.dumps({"role": "CONSUMER", **view}), flush=True)
                previous = view
            stop.wait(5)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
