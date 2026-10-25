#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Operator-driven C2 simulation. Run --help; no SSH, election or rack discovery."""
import argparse
import contextlib
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from uuid import UUID, uuid4

HERE = Path(__file__).resolve().parent
ENTRY = "/opt/engram-runtime/repo/tools/engram_sidecar/image/entrypoint.sh"
RPC_CHECK = "/opt/engram-runtime/repo/tools/engram_sidecar/deploy/rpc-check.sh"
COMMON = (
    "image",
    "authorization_group",
    "service_id",
    "fabric_domain",
    "model_revision",
    "dataset_id",
    "owner_instance_id",
    "imex_channel",
)
MEMBER = (
    "instance_id",
    "hostname",
    "rtp_role",
    "checkpoint",
    "model_env",
    "tls_dir",
    "gpu_uuids",
    "memory_limit",
    "shm_size",
    "uid",
    "gid",
    "http_port",
)
OWNER = (
    "ip",
    "port",
    "gpu_uuid",
    "numa",
    "state_dir",
    "reservation_id",
    "reserved_memory_bytes",
    "required_memory_bytes",
    "memory_limit",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identifier(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value),
        "invalid identifier",
    )
    return value


def integer(value, lower, upper):
    require(type(value) is int and lower <= value <= upper, "invalid integer")


def absolute(value):
    require(
        isinstance(value, str)
        and Path(value).is_absolute()
        and not any(c in value for c in "\n\r\0"),
        "absolute single-line path required",
    )


def validate(spec):
    require(
        type(spec) is dict
        and type(spec.get("schema_version")) is int
        and spec.get("schema_version") == 1,
        "schema_version=1 required",
    )
    require(
        set(spec) <= {"schema_version", "common", "owner", "instances", "owner_epoch"},
        "unknown top-level field",
    )
    c, o, members = spec["common"], spec["owner"], spec["instances"]
    require(set(c) == set(COMMON), "common fields must match example exactly")
    require(set(o) == set(OWNER), "owner fields must match example exactly")
    for value in (c["authorization_group"], c["service_id"], c["owner_instance_id"]):
        identifier(value)
    for key in ("image", "fabric_domain", "model_revision"):
        require(
            isinstance(c[key], str)
            and c[key]
            and not any(x in c[key] for x in "\r\n\0"),
            "invalid " + key,
        )
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", c["dataset_id"]) is not None,
        "dataset_id requires real sha256 digest",
    )
    integer(c["imex_channel"], 0, 2**31 - 1)
    ipaddress.IPv4Address(o["ip"])
    integer(o["port"], 1, 65535)
    integer(o["numa"], 0, 1024)
    identifier(o["reservation_id"])
    absolute(o["state_dir"])
    for k in ("reserved_memory_bytes", "required_memory_bytes"):
        integer(o[k], 1, 2**63 - 1)
    require(
        o["required_memory_bytes"] <= o["reserved_memory_bytes"],
        "Owner reservation too small",
    )
    require(
        isinstance(members, list) and len(members) >= 2,
        "at least two paired instances required",
    )
    ids, ports, gpu_hosts = set(), set(), {}
    for m in members:
        require(
            set(MEMBER) <= set(m) <= set(MEMBER) | {"rdma_devices"},
            "instance fields must match example (rdma_devices is optional)",
        )
        devices = m.get("rdma_devices", [])
        require(isinstance(devices, list), "rdma_devices must be a list")
        require(
            all(
                isinstance(d, str)
                and re.fullmatch(r"/dev/infiniband/[A-Za-z0-9_-]+", d)
                for d in devices
            )
            and len(set(devices)) == len(devices),
            "rdma_devices must contain unique explicit /dev/infiniband device paths",
        )
        identifier(m["instance_id"])
        identifier(m["hostname"])
        require(m["instance_id"] not in ids, "duplicate instance ID")
        ids.add(m["instance_id"])
        require(
            m["rtp_role"] in ("PREFILL", "DECODE"), "RTP role must be PREFILL or DECODE"
        )
        for k in ("checkpoint", "model_env", "tls_dir"):
            absolute(m[k])
        integer(m["uid"], 1, 2**31 - 1)
        integer(m["gid"], 0, 2**31 - 1)
        integer(m["http_port"], 1, 65535)
        require(
            (m["hostname"], m["http_port"]) not in ports, "duplicate host HTTP port"
        )
        ports.add((m["hostname"], m["http_port"]))
        require(
            isinstance(m["gpu_uuids"], list) and m["gpu_uuids"],
            "GPU UUID list required",
        )
        for gpu in m["gpu_uuids"]:
            require(
                isinstance(gpu, str) and re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu),
                "explicit GPU UUID required",
            )
            require(
                (m["hostname"], gpu) not in gpu_hosts,
                "GPU assigned to multiple RTP instances",
            )
            gpu_hosts[m["hostname"], gpu] = m["instance_id"]
        for k in ("memory_limit", "shm_size"):
            require(
                re.fullmatch(r"[1-9][0-9]*[bkmgBKMG]?", m[k]) is not None,
                "invalid resource limit",
            )
    require(
        c["owner_instance_id"] in ids, "Owner must be one of the allocated instances"
    )
    own = next(m for m in members if m["instance_id"] == c["owner_instance_id"])
    require(
        o["gpu_uuid"] in own["gpu_uuids"],
        "Owner initialization GPU must belong to its paired RTP instance",
    )
    require(
        (own["hostname"], o["port"]) not in ports,
        "Owner RPC conflicts with model HTTP port",
    )
    require(
        re.fullmatch(r"[1-9][0-9]*[bkmgBKMG]?", o["memory_limit"]) is not None,
        "invalid Owner memory limit",
    )
    if "owner_epoch" in spec:
        require(
            str(UUID(spec["owner_epoch"])) == spec["owner_epoch"], "invalid owner epoch"
        )
    return spec


def config_for(spec, member, sidecar):
    c, o = spec["common"], spec["owner"]
    owner = member["instance_id"] == c["owner_instance_id"]
    suffix = spec["owner_epoch"].replace("-", "")[:12]
    cfg = {k.upper(): str(c[k]) for k in COMMON if k != "imex_channel"}
    cfg.update(
        DISCOVERY="platform",
        PLATFORM_IMEX_CHANNEL=str(c["imex_channel"]),
        OWNER_ENDPOINT=f"{o['ip']}:{o['port']}",
        OWNER_EPOCH=spec["owner_epoch"],
        INSTANCE_ID=member["instance_id"],
        CONTAINER_UID=str(member["uid"]),
        CONTAINER_GID=str(member["gid"]),
        TLS_DIR=member["tls_dir"],
        STARTUP_TIMEOUT_S="1800",
        CHECKPOINT=member["checkpoint"],
    )
    cfg["NAME"] = (
        "engram-sim-"
        + member["instance_id"]
        + "-"
        + ("sidecar-" if sidecar else "rtp-")
        + suffix
    )
    if sidecar:
        cfg.update(
            SIDECAR_ROLE="producer" if owner else "consumer",
            MEMORY_LIMIT=o["memory_limit"] if owner else "1g",
        )
        if owner:
            cfg.update(
                OWNER_IP=o["ip"],
                OWNER_PORT=str(o["port"]),
                GPU_UUID=o["gpu_uuid"],
                NUMA=str(o["numa"]),
                STATE_DIR=o["state_dir"],
                RESERVATION_ID=o["reservation_id"],
                RESERVED_MEMORY_BYTES=str(o["reserved_memory_bytes"]),
                REQUIRED_MEMORY_BYTES=str(o["required_memory_bytes"]),
                ALLOWED_CLIENT_NAMES=" ".join(
                    m["instance_id"] for m in spec["instances"]
                ),
            )
    else:
        cfg.update(
            MODEL_ENV=member["model_env"],
            GPU_UUIDS=",".join(member["gpu_uuids"]),
            MEMORY_LIMIT=member["memory_limit"],
            SHM_SIZE=member["shm_size"],
        )
    if not sidecar and member.get("rdma_devices"):
        cfg["RDMA_DEVICES"] = " ".join(member["rdma_devices"])
    return cfg


def shell_config(cfg):
    return (
        "# Generated from one platform assignment; regenerate as a new run to change it.\n"
        + "".join(k + "=" + shlex.quote(v) + "\n" for k, v in sorted(cfg.items()))
    )


def write_private(path, text):
    with path.open("x") as f:
        os.chmod(path, 0o600)
        f.write(text)


def render(source, output):
    spec = validate(json.loads(source.read_text()))
    require(
        "owner_epoch" not in spec,
        "input must not supply an epoch; reuse the generated run directory",
    )
    spec["owner_epoch"] = str(uuid4())
    output.mkdir(mode=0o700)  # No accidental reassignment or overwrite.
    write_private(output / "assignment.json", json.dumps(spec, indent=2) + "\n")
    for m in spec["instances"]:
        directory = output / m["instance_id"]
        directory.mkdir(mode=0o700)
        for part in ("sidecar", "rtp"):
            write_private(
                directory / (part + ".sh"),
                shell_config(config_for(spec, m, part == "sidecar")),
            )
    return spec


def run(cmd, *, timeout=30):
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
    ).stdout


def inspect(name):
    value = json.loads(run(["pouch", "inspect", name]))
    return value[0] if isinstance(value, list) else value


def receipt_path(directory, member, part):
    return directory / member["instance_id"] / (part + ".receipt.json")


def checked_container(directory, spec, member, part, *, running=True):
    receipt = json.loads(receipt_path(directory, member, part).read_text())
    require(receipt["epoch"] == spec["owner_epoch"], "stale container receipt")
    obj = inspect(receipt["container_id"])
    require(obj["Id"] == receipt["container_id"], "container identity changed")
    expected = config_for(spec, member, part == "sidecar")
    env = dict(
        v.split("=", 1) for v in obj.get("Config", {}).get("Env", []) if "=" in v
    )
    for key in (
        "INSTANCE_ID",
        "OWNER_EPOCH",
        "OWNER_ENDPOINT",
        "SERVICE_ID",
        "AUTHORIZATION_GROUP",
        "OWNER_INSTANCE_ID",
        "FABRIC_DOMAIN",
        "MODEL_REVISION",
        "DATASET_ID",
    ):
        require(
            env.get("RTP_ENGRAM_" + key) == expected[key],
            "container assignment mismatch: " + key,
        )
    if running:
        require(obj["State"]["Running"], part + " container has exited")
    return obj


def execute(obj, member, *args, timeout=30):
    return run(
        ["pouch", "exec", "-u", f"{member['uid']}:{member['gid']}", obj["Id"], *args],
        timeout=timeout,
    )


def preflight(member, cfg, part):
    require(
        os.geteuid() == member["uid"],
        "run start commands as the configured container UID; private state and TLS must remain accessible to that UID",
    )
    for path in (
        Path(member["checkpoint"]) / "config.json",
        Path(member["tls_dir"]) / "ca.crt",
        Path(member["tls_dir"]) / "consumer.crt",
        Path(member["tls_dir"]) / "consumer.key",
    ):
        require(path.is_file(), "missing input: " + str(path))
    if part == "rtp":
        values = {}
        for line in Path(member["model_env"]).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            require(
                not key.startswith("RTP_ENGRAM_"), "keep assignment out of model.env"
            )
            values[key] = value
        require(
            values.get("ROLE_TYPE") == member["rtp_role"],
            "model.env ROLE_TYPE mismatch",
        )
        require(
            values.get("START_PORT") == str(member["http_port"]),
            "model.env START_PORT mismatch",
        )
    # CN mismatch is diagnosed before launching expensive services.
    subject = run(
        [
            "openssl",
            "x509",
            "-in",
            str(Path(member["tls_dir"]) / "consumer.crt"),
            "-noout",
            "-subject",
            "-nameopt",
            "RFC2253",
        ]
    )
    require(
        re.search(
            r"(?:^|[=,])CN=" + re.escape(member["instance_id"]) + r"(?:,|$)",
            subject.strip(),
        )
        is not None,
        "client certificate CN must equal instance_id",
    )


def start(directory, spec, member, part):
    path = receipt_path(directory, member, part)
    if path.exists():
        obj = checked_container(directory, spec, member, part)
        return {"state": "ALREADY_RUNNING", "part": part, "container_id": obj["Id"]}
    cfg = config_for(spec, member, part == "sidecar")
    require(
        (directory / member["instance_id"] / (part + ".sh")).read_text()
        == shell_config(cfg),
        "generated configuration was modified; render a new run",
    )
    preflight(member, cfg, part)
    if part == "rtp":
        owner_ready(directory, spec, member)
    else:
        try:
            run(
                [
                    "pouch",
                    "run",
                    "--rm",
                    "--net=none",
                    "--user",
                    f"{member['uid']}:{member['gid']}",
                    "-e",
                    "USER=rtp-engram",
                    "-e",
                    "LOGNAME=rtp-engram",
                    "-e",
                    "HOME=/tmp/rtp-engram",
                    cfg["IMAGE"],
                    "python",
                    "-c",
                    'from rtp_llm.models_py.modules.dsv41.engram_producer import AssignedProducerSession; from rtp_llm.models_py.modules.dsv41.engram_assignment import configured_assignment; print("PLATFORM_ASSIGNMENT_OK")',
                ],
                timeout=120,
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                "Image runtime check failed: use a CI image containing platform assignment (780bd26a4 or later); no sidecar was launched"
            ) from None
    script = "sidecar-start.sh" if part == "sidecar" else "consumer-start.sh"
    run(
        [
            "bash",
            str(HERE / script),
            str(directory / member["instance_id"] / (part + ".sh")),
        ]
        + ([] if part == "sidecar" else ["shared"]),
        timeout=120,
    )
    obj = inspect(cfg["NAME"])
    write_private(
        path,
        json.dumps({"epoch": spec["owner_epoch"], "container_id": obj["Id"]}) + "\n",
    )
    return {"state": "STARTED", "part": part, "container_id": obj["Id"]}


def owner_ready(directory, spec, member):
    obj = checked_container(directory, spec, member, "sidecar")
    answer = json.loads(execute(obj, member, "bash", RPC_CHECK))
    a = answer["assignment"]
    require(
        a["owner_epoch"] == spec["owner_epoch"]
        and a["owner_instance_id"] == spec["common"]["owner_instance_id"],
        "Owner assignment mismatch",
    )
    return {
        "state": "OWNER_READY",
        "generation": answer["generation"],
        "owner_epoch": a["owner_epoch"],
    }


HTTP_PROBE = """import json,sys,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
port,action=sys.argv[1:3]
url='http://127.0.0.1:'+port
if action=='chat':
    payload={'model':sys.argv[3],'messages':[{'role':'user','content':'你好，请用一句话介绍自己。'}],'max_tokens':32,'temperature':0,'thinking':{'type':'disabled'}}
    req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    value=json.load(opener.open(req,timeout=90))
    text=value.get('choices',[{}])[0].get('message',{}).get('content','')
    if not isinstance(text,str) or not text.strip(): raise SystemExit('CHAT_RESPONSE_INVALID')
    print(json.dumps({'state':'CHAT_OK','content':text},ensure_ascii=False))
else:
    response=opener.open(url+'/health',timeout=5)
    if response.status!=200: raise SystemExit('HTTP_NOT_READY')
    print(json.dumps({'state':'HTTP_HEALTH_OK','http_status':response.status}))
"""


def http_probe(directory, spec, member, action, model):
    require(
        action != "chat" or member["rtp_role"] == "PREFILL",
        "send chat to prefill frontend, not decode",
    )
    obj = checked_container(directory, spec, member, "rtp")
    return json.loads(
        execute(
            obj,
            member,
            ENTRY,
            "python",
            "-c",
            HTTP_PROBE,
            str(member["http_port"]),
            action,
            model,
            timeout=100,
        )
    )


def wait_for(probe, seconds):
    end = time.monotonic() + seconds
    while True:
        try:
            return probe()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            details = getattr(error, "stderr", "") or ""
            if isinstance(details, bytes):
                details = details.decode("utf-8", "replace")
            if any(
                marker in details
                for marker in (
                    "PERMISSION_DENIED",
                    "DATASET_NOT_FOUND",
                    "ASSIGNMENT_MISMATCH",
                    "GENERATION_CHANGED",
                )
            ):
                raise ValueError(
                    "Owner identity/authorization mismatch; refusing readiness retry"
                ) from None
            if time.monotonic() >= end:
                raise RuntimeError(
                    "readiness timeout; inspect the paired container logs"
                ) from None
            time.sleep(min(2, max(0, end - time.monotonic())))


def pki(spec, output):
    # Development-only CA. Copy only the per-instance leaf directory to each host.
    output.mkdir(mode=0o700)

    def openssl(*args):
        run(["openssl", *map(str, args)])

    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "7",
        "-subj",
        "/CN=engram-simulation-ca",
        "-keyout",
        output / "ca.key",
        "-out",
        output / "ca.crt",
    )
    for m in spec["instances"]:
        d = output / m["instance_id"]
        d.mkdir(mode=0o700)
        (d / "ca.crt").write_bytes((output / "ca.crt").read_bytes())
        leaves = [("consumer", m["instance_id"], "extendedKeyUsage=clientAuth\n")]
        if m["instance_id"] == spec["common"]["owner_instance_id"]:
            leaves.append(
                (
                    "owner",
                    "engram-owner",
                    "extendedKeyUsage=serverAuth\nsubjectAltName=IP:"
                    + spec["owner"]["ip"]
                    + "\n",
                )
            )
        for name, cn, ext in leaves:
            openssl(
                "req",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-subj",
                "/CN=" + cn,
                "-keyout",
                d / (name + ".key"),
                "-out",
                d / (name + ".csr"),
            )
            (d / (name + ".ext")).write_text(ext)
            openssl(
                "x509",
                "-req",
                "-in",
                d / (name + ".csr"),
                "-CA",
                output / "ca.crt",
                "-CAkey",
                output / "ca.key",
                "-CAcreateserial",
                "-days",
                "7",
                "-extfile",
                d / (name + ".ext"),
                "-out",
                d / (name + ".crt"),
            )
            (d / (name + ".csr")).unlink()
            (d / (name + ".ext")).unlink()
    return {
        "state": "TEST_PKI_CREATED",
        "instruction": "Deliver only each instance directory; keep ca.key on the issuing host. Match ownership to the configured container UID.",
    }


@contextlib.contextmanager
def lock(directory, member):
    with (directory / member["instance_id"] / "operator.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser(
        "render",
        help="generate one epoch and both container configs per instance; no remote actions",
    )
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser(
        "test-pki",
        help="issue development-only per-instance certificates, never for production",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for command in (
        "plan",
        "start-sidecar",
        "wait-owner",
        "start-rtp",
        "wait-rtp",
        "status",
        "chat",
        "stop-rtp",
    ):
        p = sub.add_parser(command)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--instance", required=True)
        p.add_argument("--timeout", type=int, default=1800)
        p.add_argument("--model", default="deepseek_v41")
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("a command is required")
    os.umask(0o077)
    if args.command == "render":
        spec = render(args.spec, args.out)
        result = {
            "state": "CONFIGURED",
            "owner_epoch": spec["owner_epoch"],
            "run_dir": str(args.out),
            "owner_instance_id": spec["common"]["owner_instance_id"],
        }
    else:
        directory = args.run_dir.resolve()
        spec = validate(json.loads((directory / "assignment.json").read_text()))
        require("owner_epoch" in spec, "render the assignment first")
        if args.command == "test-pki":
            result = pki(spec, args.out)
        else:
            matches = [
                m for m in spec["instances"] if m["instance_id"] == args.instance
            ]
            require(len(matches) == 1, "unknown instance")
            member = matches[0]
            integer(args.timeout, 1, 3600)
            if args.command == "plan":
                result = {
                    "instance_id": member["instance_id"],
                    "hostname": member["hostname"],
                    "rtp_role": member["rtp_role"],
                    "sidecar_role": (
                        "producer"
                        if args.instance == spec["common"]["owner_instance_id"]
                        else "consumer"
                    ),
                    "owner_endpoint": f"{spec['owner']['ip']}:{spec['owner']['port']}",
                    "owner_epoch": spec["owner_epoch"],
                    "resource_reservation": "operator must reserve; simulator does not allocate resources",
                }
            else:
                require(
                    socket.gethostname() == member["hostname"],
                    "wrong host: run on configured hostname " + member["hostname"],
                )
                with lock(directory, member):
                    if args.command.startswith("start-"):
                        result = start(
                            directory, spec, member, args.command.split("-")[1]
                        )
                    elif args.command == "wait-owner":
                        result = wait_for(
                            lambda: owner_ready(directory, spec, member), args.timeout
                        )
                    elif args.command == "wait-rtp":
                        result = wait_for(
                            lambda: http_probe(
                                directory, spec, member, "health", args.model
                            ),
                            args.timeout,
                        )
                    elif args.command == "chat":
                        result = http_probe(directory, spec, member, "chat", args.model)
                    elif args.command == "status":
                        result = {}
                        for part in ("sidecar", "rtp"):
                            if not receipt_path(directory, member, part).exists():
                                result[part] = {"state": "NOT_STARTED"}
                                continue
                            obj = checked_container(
                                directory, spec, member, part, running=False
                            )
                            result[part] = {
                                "container_id": obj["Id"],
                                "running": obj["State"]["Running"],
                            }
                    else:
                        obj = checked_container(
                            directory, spec, member, "rtp", running=False
                        )
                        if obj["State"]["Running"]:
                            run(["pouch", "stop", "-t", "300", obj["Id"]], timeout=330)
                        require(
                            not checked_container(
                                directory, spec, member, "rtp", running=False
                            )["State"]["Running"],
                            "RTP still running",
                        )
                        result = {
                            "state": "LOCAL_RTP_STOPPED",
                            "container_id": obj["Id"],
                            "owner_retained": True,
                            "note": "This is not proof that other hosts are drained. Owner shutdown is deliberately not automated.",
                        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            json.dumps(
                {
                    "error": "COMMAND_FAILED",
                    "program": Path(error.cmd[0]).name,
                    "exit_code": error.returncode,
                    "hint": "inspect container logs; child output omitted to avoid credentials",
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        subprocess.TimeoutExpired,
    ) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        sys.exit(1)
