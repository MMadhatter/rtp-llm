#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One sidecar entrypoint: authorized producer or lightweight consumer observer.

Platform injects an explicit Owner assignment and TLS through env.
Legacy Registry coordination is retained for existing experiments. RTP workers separately
resolve and import handles; sidecar never forwards per-request data or pointers.
"""
import argparse
import json
import os
import signal
import threading

from rtp_llm.models_py.modules.dsv41.engram_producer import (
    AssignedProducerSession,
    ProducerSession,
)
from rtp_llm.models_py.modules.dsv41.engram_registry import RegistryClient


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--required-memory-bytes", type=int, required=True)
    p.add_argument("--role", choices=("auto", "producer", "consumer"))
    p.add_argument("owner_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    from rtp_llm.models_py.modules.dsv41.engram_assignment import configured_assignment

    platform = configured_assignment(os.environ) is not None
    configured_role = os.environ.get("RTP_ENGRAM_SIDECAR_ROLE")
    args.role = args.role or configured_role or "auto"
    if platform:
        if args.role not in ("producer", "consumer") or args.role != configured_role:
            raise ValueError("PLATFORM_EXPLICIT_ROLE_REQUIRED")
    session = None
    if platform and args.role == "producer":
        session = AssignedProducerSession(args.required_memory_bytes)
    elif args.role != "consumer":
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
    if platform:
        from rtp_llm.models_py.modules.dsv41.engram_rpc import Client

        client = Client()
    else:
        client = RegistryClient()
    previous = None
    try:
        while not stop.is_set():
            if platform:
                import grpc

                try:
                    response = client.describe()
                    state = dict(
                        state="READY",
                        generation=response["generation"],
                        producer_instance_id=response["assignment"][
                            "owner_instance_id"
                        ],
                        owner_epoch=response["assignment"]["owner_epoch"],
                    )
                except grpc.RpcError as error:
                    if error.code() not in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    ) and not (
                        error.code() == grpc.StatusCode.FAILED_PRECONDITION
                        and error.details() == "OWNER_NOT_READY"
                    ):
                        raise
                    state = {"state": "WAITING"}
            else:
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
