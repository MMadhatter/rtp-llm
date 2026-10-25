#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export gated Kubernetes review YAML from the same manual Pouch assignment.

Requires PyYAML only for export. Does not contact Kubernetes or inject devices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from c2_sim import ENTRY, RPC_CHECK, validate, require, write_private


def name_for(member):
    value = member["instance_id"]
    slug = re.sub(r"[^a-z0-9-]", "-", value.lower()).strip("-") or "instance"
    return "engram-" + slug[:32] + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def quantity(value):
    units = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
    match = re.fullmatch(r"([1-9][0-9]*)([bkmgBKMG]?)", value)
    require(match is not None, "invalid resource quantity")
    return str(int(match[1]) * units.get(match[2].lower(), 1))


def env(values):
    return [{"name": k, "value": str(v)} for k, v in sorted(values.items())]


def documents(spec):
    validate(spec)
    require(
        "owner_epoch" in spec, "use c2_sim render first; do not create another epoch"
    )
    c, o = spec["common"], spec["owner"]
    result = []
    for m in spec["instances"]:
        owner = m["instance_id"] == c["owner_instance_id"]
        name = name_for(m)
        identity = {
            "RTP_ENGRAM_" + k.upper(): c[k]
            for k in (
                "authorization_group",
                "service_id",
                "fabric_domain",
                "model_revision",
                "dataset_id",
                "owner_instance_id",
                "imex_channel",
            )
        }
        identity.update(
            RTP_ENGRAM_DISCOVERY="platform",
            RTP_ENGRAM_OWNER_EPOCH=spec["owner_epoch"],
            RTP_ENGRAM_OWNER_ENDPOINT=f"{o['ip']}:{o['port']}",
            RTP_ENGRAM_INSTANCE_ID=m["instance_id"],
            RTP_ENGRAM_STARTUP_TIMEOUT_S="1800",
            RTP_ENGRAM_TLS_CA="/run/engram-tls/ca.crt",
            RTP_ENGRAM_TLS_CERT="/run/engram-tls/consumer.crt",
            RTP_ENGRAM_TLS_KEY="/run/engram-tls/consumer.key",
            USER="rtp-engram",
            LOGNAME="rtp-engram",
            HOME="/tmp/rtp-engram",
            PYTHONUNBUFFERED="1",
        )
        result.append(
            dict(
                apiVersion="v1",
                kind="ConfigMap",
                metadata={"name": name + "-assignment"},
                data={k: str(v) for k, v in identity.items()},
            )
        )
        tls = {"name": "tls", "mountPath": "/run/engram-tls", "readOnly": True}
        checkpoint = {
            "name": "checkpoint",
            "mountPath": "/checkpoint",
            "readOnly": True,
        }
        security = dict(
            runAsNonRoot=True,
            runAsUser=m["uid"],
            runAsGroup=m["gid"],
            privileged=False,
            allowPrivilegeEscalation=False,
        )
        common = dict(
            image=c["image"],
            imagePullPolicy="IfNotPresent",
            securityContext=security,
            envFrom=[{"configMapRef": {"name": name + "-assignment"}}],
        )
        role = "producer" if owner else "consumer"
        args = [
            "sidecar",
            "--role",
            role,
            "--required-memory-bytes",
            str(o["required_memory_bytes"] if owner else 1),
        ]
        side_env = {"RTP_ENGRAM_SIDECAR_ROLE": role}
        side_mounts = [tls]
        volumes = [
            {
                "name": "tls",
                "secret": {"secretName": name + "-tls", "defaultMode": 288},
            },
            {
                "name": "checkpoint",
                "hostPath": {"path": m["checkpoint"], "type": "Directory"},
            },
            {
                "name": "shm",
                "emptyDir": {"medium": "Memory", "sizeLimit": quantity(m["shm_size"])},
            },
        ]
        if owner:
            args += [
                "--",
                "--checkpoint",
                "/checkpoint",
                "--model-revision",
                c["model_revision"],
                "--publication",
                "/run/engram/state",
                "--device",
                "0",
                "--numa",
                str(o["numa"]),
                "--max-allocation-bytes",
                "137438953472",
                "--budget-bytes",
                "274877906944",
                "--discard-private-checkpoint-cache",
                "--rpc-listen",
                f"{o['ip']}:{o['port']}",
                "--service-id",
                c["service_id"],
                "--tls-ca",
                "/run/engram-tls/ca.crt",
                "--tls-cert",
                "/run/engram-tls/owner.crt",
                "--tls-key",
                "/run/engram-tls/owner.key",
            ]
            for client in spec["instances"]:
                args += ["--allowed-client", client["instance_id"]]
            side_env.update(
                RTP_ENGRAM_RESERVATION_ID=o["reservation_id"],
                RTP_ENGRAM_RESERVED_MEMORY_BYTES=str(o["reserved_memory_bytes"]),
                CUDA_VISIBLE_DEVICES="0",
            )
            side_mounts += [
                checkpoint,
                {"name": "owner-state", "mountPath": "/run/engram"},
            ]
            volumes += [
                {
                    "name": "owner-state",
                    "hostPath": {"path": o["state_dir"], "type": "Directory"},
                }
            ]
        side_mem = quantity(o["memory_limit"]) if owner else str(1024**3)
        sidecar = dict(
            common,
            name="engram-sidecar",
            command=[ENTRY],
            args=args,
            env=env(side_env),
            volumeMounts=side_mounts,
            resources={
                "requests": {"memory": side_mem},
                "limits": {"memory": side_mem},
            },
            readinessProbe={
                "exec": {"command": [ENTRY, "shell", RPC_CHECK]},
                "periodSeconds": 10,
                "timeoutSeconds": 10,
            },
        )
        # A regular container waits; an init container waiting for a regular producer deadlocks.
        wait = (
            "deadline=$((SECONDS+1800)); until bash "
            + RPC_CHECK
            + " >/dev/null 2>&1; do "
            'if (( SECONDS >= deadline )); then echo "Owner readiness timed out" >&2; exit 1; fi; '
            "sleep 5; done; exec " + ENTRY + " serve"
        )
        rtp = dict(
            common,
            name="rtp",
            command=[ENTRY],
            args=["shell", "-c", wait],
            envFrom=common["envFrom"] + [{"configMapRef": {"name": name + "-model"}}],
            env=env(
                dict(
                    identity,
                    RTP_ENGRAM_MODE="shared",
                    RTP_ENGRAM_GRAPH_MODE="process_lifetime",
                    RTP_ENGRAM_MAX_ALLOCATION_BYTES="137438953472",
                    RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES="274877906944",
                    DSV41_CPU_ENGRAM_LOOKUP="0",
                    DSV41_ASYNC_ENGRAM_LOOKUP="0",
                )
            ),
            volumeMounts=[tls, checkpoint, {"name": "shm", "mountPath": "/dev/shm"}],
            resources={
                "requests": {"memory": quantity(m["memory_limit"])},
                "limits": {"memory": quantity(m["memory_limit"])},
            },
            readinessProbe={
                "httpGet": {"path": "/health", "port": m["http_port"]},
                "periodSeconds": 10,
                "timeoutSeconds": 5,
            },
        )
        requirements = dict(
            hostname=m["hostname"],
            rtp_gpu_uuids=m["gpu_uuids"],
            producer_gpu_uuid=o["gpu_uuid"] if owner else None,
            imex_device="/dev/nvidia-caps-imex-channels/channel"
            + str(c["imex_channel"]),
            rtp_rdma_devices=m.get("rdma_devices", []),
            model_env=m["model_env"],
            tls_dir=m["tls_dir"],
        )
        result.append(
            dict(
                apiVersion="v1",
                kind="Pod",
                metadata={
                    "name": name,
                    "annotations": {
                        "engram.example.com/review-only": "true",
                        "engram.example.com/device-requirements": json.dumps(
                            requirements, sort_keys=True
                        ),
                    },
                },
                spec=dict(
                    hostNetwork=True,
                    dnsPolicy="ClusterFirstWithHostNet",
                    restartPolicy="Never",
                    terminationGracePeriodSeconds=300,
                    securityContext={"fsGroup": m["gid"]},
                    schedulingGates=[
                        {
                            "name": "engram.example.com/platform-device-and-lifecycle-review"
                        }
                    ],
                    nodeSelector={
                        "kubernetes.io/hostname": "REPLACE_WITH_VERIFIED_K8S_NODE_LABEL"
                    },
                    containers=[sidecar, rtp],
                    volumes=volumes,
                ),
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import yaml  # Optional export dependency; not required for manual Pouch operation.

    spec = json.loads((args.run_dir / "assignment.json").read_text())
    content = (
        "# REVIEW TEMPLATE ONLY: device allocation and Owner lifecycle require platform integration.\n"
        "# The example.com annotations are documentation, not implemented C2 hooks.\n"
        "# Supply per-instance *-model ConfigMaps and *-tls Secrets. Never put keys in Git.\n"
        + yaml.safe_dump_all(documents(spec), sort_keys=False)
    )
    write_private(args.out, content)


if __name__ == "__main__":
    main()
