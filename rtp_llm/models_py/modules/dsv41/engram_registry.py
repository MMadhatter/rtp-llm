# SPDX-License-Identifier: Apache-2.0
"""Startup coordination contract and single-node durable development adapter.

No Fabric handles or weight bytes enter this service. Platform policy binds
certificate principals to instances, endpoints and pre-reserved capacity.
Expired claims are sticky: neither lease loss nor restart authorizes takeover.
Production C2 must implement the same atomic operations against its own store
and resource/lifecycle authority; SQLite is not an HA scheduler.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

from .engram_rpc import (
    OPTIONS,
    ControlError,
    _pem,
    configured_identity,
    decode,
    encode,
    validate_identity,
)

SERVICE = "rtp.engram.v1.Registry"
REGISTRY_ENV = "RTP_ENGRAM_REGISTRY_ENDPOINT"
RESOLVED_ENDPOINT_ENV = "RTP_ENGRAM_RESOLVED_OWNER_ENDPOINT"
METHOD_FIELDS = {
    "ResolveProducer": set(),
    "AcquireProducer": {"attempt_id", "required_memory_bytes"},
    "RenewProducer": {"attempt_id", "owner_epoch"},
    "PublishReady": {"attempt_id", "owner_epoch", "generation", "physical_bytes"},
}


class RegistryError(ControlError):
    def __init__(self, code, status="FAILED_PRECONDITION"):
        self.status = status
        super().__init__(code)


def endpoint(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[a-zA-Z0-9_.-]+:[0-9]{1,5}", value)
        or not 0 < int(value.rsplit(":", 1)[1]) <= 65535
    ):
        raise ControlError("RPC_ENDPOINT_REQUIRED")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,256}", value):
        raise ControlError("REGISTRY_IDENTIFIER_INVALID")
    return value


def uuid_value(value):
    try:
        if str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ControlError("REGISTRY_UUID_REQUIRED") from None
    return value


def positive_bytes(value):
    if type(value) is not int or not 0 < value <= (1 << 60):
        raise ControlError("REGISTRY_MEMORY_INVALID")
    return value


def validate_scope(scope):
    if type(scope) is not dict or set(scope) != {
        "fabric_domain",
        "authorization_group",
    }:
        raise ControlError("REGISTRY_SCOPE_REQUIRED")
    return {k: identifier(v) for k, v in scope.items()}


def canonical_group_key(scope, identity):
    return hashlib.sha256(
        json.dumps(
            {"scope": validate_scope(scope), "identity": validate_identity(identity)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class SqlitePlatformAdapter:
    """Reference adapter with serialized transactions, never implicit failover.

    Policy is platform provisioning input, not discovered from caller IPs.
    Memory reservations must already exist; this adapter does not reserve RAM.
    One group may authorize multiple candidates for CAS testing, or one fixed
    producer for first deployment. Policy changes fail closed for live claims.
    """

    def __init__(self, database, policy, *, lease_seconds=30, clock=time.time):
        if type(lease_seconds) is not int or not 3 <= lease_seconds <= 300:
            raise ValueError("lease_seconds must be in [3,300]")
        if set(policy) != {"schema_version", "groups"} or policy["schema_version"] != 1:
            raise ValueError("invalid platform policy")
        self.groups = {}
        for source in policy["groups"]:
            if set(source) != {"scope", "identity", "members"}:
                raise ValueError("invalid group policy")
            scope, identity = validate_scope(source["scope"]), validate_identity(
                source["identity"]
            )
            key = canonical_group_key(scope, identity)
            if key in self.groups or not source["members"]:
                raise ValueError("duplicate or empty group")
            members, principals = {}, set()
            for member in source["members"]:
                if set(member) != {
                    "instance_id",
                    "principal",
                    "node_id",
                    "owner_endpoint",
                    "reservation_id",
                    "reserved_memory_bytes",
                    "can_produce",
                }:
                    raise ValueError("invalid member policy")
                for field in ("instance_id", "principal", "node_id"):
                    identifier(member[field])
                if (
                    member["instance_id"] in members
                    or member["principal"] in principals
                ):
                    raise ValueError(
                        "principal and instance must be unique within a group"
                    )
                if type(member["can_produce"]) is not bool:
                    raise ValueError("can_produce must be boolean")
                if member["can_produce"]:
                    endpoint(member["owner_endpoint"])
                    identifier(member["reservation_id"])
                    positive_bytes(member["reserved_memory_bytes"])
                elif (
                    member["owner_endpoint"] is not None
                    or member["reservation_id"] is not None
                    or member["reserved_memory_bytes"] != 0
                ):
                    raise ValueError("consumer must not claim reserved resources")
                members[member["instance_id"]] = dict(member)
                principals.add(member["principal"])
            if not any(m["can_produce"] for m in members.values()):
                raise ValueError("group requires a platform-authorized candidate")
            self.groups[key] = (scope, identity, members)
        self.clock, self.lease_seconds = clock, lease_seconds
        self.lock = threading.Lock()
        path = Path(database)
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError("database needs an existing private absolute directory")
        if path.parent.stat().st_mode & 0o077:
            raise ValueError("database directory must be private (0700)")
        if path.is_symlink():
            raise ValueError("database symlinks not supported")
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        if path.stat().st_mode & 0o077:
            raise ValueError("database must be private (0600)")
        self.db = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None
        )
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS claims (group_key TEXT PRIMARY KEY, record TEXT NOT NULL)"
        )

    def close(self):
        with self.lock:
            self.db.close()

    def _member(self, request, principal):
        scope, identity = validate_scope(request["scope"]), validate_identity(
            request["identity"]
        )
        key = canonical_group_key(scope, identity)
        group = self.groups.get(key)
        member = group[2].get(request["instance_id"]) if group else None
        if not member or principal != member["principal"]:
            raise RegistryError("GROUP_OR_INSTANCE_NOT_AUTHORIZED", "PERMISSION_DENIED")
        return key, scope, identity, member

    def _view(self, record, group):
        if record is None:
            return {"state": "EMPTY"}
        member = group[2].get(record["producer_instance_id"])
        # Changing platform binding must not retarget already-published memory.
        if (
            member is None
            or not member["can_produce"]
            or record["principal"] != member["principal"]
            or record["reserved_memory_bytes"] > member["reserved_memory_bytes"]
            or any(
                record[k] != member[k]
                for k in ("node_id", "owner_endpoint", "reservation_id")
            )
        ):
            raise RegistryError("PLATFORM_BINDING_CHANGED")
        state = record["state"]
        if self.clock() >= record["expires_at"]:
            state = "STALE"
        result = {
            k: record[k]
            for k in (
                "producer_instance_id",
                "node_id",
                "owner_epoch",
                "reservation_id",
                "reserved_memory_bytes",
            )
        }
        result["state"] = state
        # An address is usable only after a complete generation was published.
        if state == "READY":
            result.update(
                owner_endpoint=record["owner_endpoint"],
                generation=record["generation"],
                physical_bytes=record["physical_bytes"],
            )
        return result

    def dispatch(self, method, request, principal):
        if method not in METHOD_FIELDS:
            raise ControlError("REGISTRY_METHOD_INVALID")
        expected = {
            "schema_version",
            "scope",
            "identity",
            "instance_id",
        } | METHOD_FIELDS[method]
        if (
            set(request) != expected
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise ControlError("REGISTRY_REQUEST_INVALID")
        identifier(request["instance_id"])
        if "attempt_id" in request:
            uuid_value(request["attempt_id"])
        if "owner_epoch" in request:
            uuid_value(request["owner_epoch"])
        if method == "AcquireProducer":
            positive_bytes(request["required_memory_bytes"])
        if method == "PublishReady":
            uuid_value(request["generation"])
            positive_bytes(request["physical_bytes"])
        key, scope, identity, member = self._member(request, principal)
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute(
                    "SELECT record FROM claims WHERE group_key=?", (key,)
                ).fetchone()
                record = json.loads(row[0]) if row else None
                view = self._view(record, self.groups[key])
                role = "CONSUMER"
                if method == "AcquireProducer":
                    required = request["required_memory_bytes"]
                    if record is None and member["can_produce"]:
                        if required > member["reserved_memory_bytes"]:
                            raise RegistryError(
                                "PRE_RESERVED_MEMORY_INSUFFICIENT", "RESOURCE_EXHAUSTED"
                            )
                        record = {
                            k: member[k]
                            for k in ("node_id", "owner_endpoint", "reservation_id")
                        }
                        record.update(
                            producer_instance_id=member["instance_id"],
                            principal=principal,
                            attempt_id=request["attempt_id"],
                            owner_epoch=str(uuid4()),
                            reserved_memory_bytes=required,
                            state="LOADING",
                            expires_at=self.clock() + self.lease_seconds,
                        )
                        role = "PRODUCER"
                    elif (
                        record
                        and record["producer_instance_id"] == member["instance_id"]
                    ):
                        self._check_holder(
                            record, request, principal, check_epoch=False
                        )
                        if required != record["reserved_memory_bytes"]:
                            raise RegistryError("IDEMPOTENCY_MISMATCH")
                        role = "PRODUCER"
                elif method in ("RenewProducer", "PublishReady"):
                    self._check_holder(record, request, principal)
                    if method == "RenewProducer":
                        record["expires_at"] = self.clock() + self.lease_seconds
                    else:
                        if request["physical_bytes"] > record["reserved_memory_bytes"]:
                            raise RegistryError(
                                "RESERVATION_EXCEEDED", "RESOURCE_EXHAUSTED"
                            )
                        if record["state"] == "READY" and (
                            record["generation"] != request["generation"]
                            or record["physical_bytes"] != request["physical_bytes"]
                        ):
                            raise RegistryError("GENERATION_IMMUTABLE")
                        record.update(
                            state="READY",
                            generation=request["generation"],
                            physical_bytes=request["physical_bytes"],
                        )
                    role = "PRODUCER"
                if record is not None and method != "ResolveProducer":
                    self.db.execute(
                        "INSERT OR REPLACE INTO claims VALUES (?,?)",
                        (key, json.dumps(record)),
                    )
                view = self._view(record, self.groups[key])
                result = dict(schema_version=1, scope=scope, identity=identity, **view)
                if method != "ResolveProducer":
                    result["role"] = role
                if role == "PRODUCER":
                    result["owner_endpoint"] = record["owner_endpoint"]
                    result["lease_remaining_s"] = max(
                        0, record["expires_at"] - self.clock()
                    )
                self.db.execute("COMMIT")
                return result
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _check_holder(self, record, request, principal, *, check_epoch=True):
        if (
            record is None
            or record["principal"] != principal
            or record["producer_instance_id"] != request["instance_id"]
            or record["attempt_id"] != request["attempt_id"]
            or (check_epoch and record["owner_epoch"] != request["owner_epoch"])
        ):
            raise RegistryError("CLAIM_NOT_HELD", "PERMISSION_DENIED")
        if self.clock() >= record["expires_at"]:
            raise RegistryError("CLAIM_EXPIRED_REQUIRES_PLATFORM_RECOVERY")


class RegistryServer:
    def __init__(self, adapter):
        self.adapter, self.server = adapter, None

    def _call(self, method, request, context):
        import grpc

        names = context.auth_context().get("x509_common_name", [])
        if len(names) != 1:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "CLIENT_IDENTITY_REQUIRED")
        try:
            return self.adapter.dispatch(method, request, names[0].decode("utf-8"))
        except RegistryError as error:
            context.abort(getattr(grpc.StatusCode, error.status), error.code)
        except (ControlError, KeyError, UnicodeError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST")
        except Exception:
            context.abort(grpc.StatusCode.INTERNAL, "PLATFORM_ADAPTER_FAILURE")

    def start(self, address, *, ca, cert, key):
        import grpc

        server = grpc.server(
            ThreadPoolExecutor(max_workers=16),
            options=OPTIONS,
            maximum_concurrent_rpcs=64,
        )
        handlers = {
            method: grpc.unary_unary_rpc_method_handler(
                lambda request, ctx, method=method: self._call(method, request, ctx),
                request_deserializer=decode,
                response_serializer=encode,
            )
            for method in METHOD_FIELDS
        }
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(SERVICE, handlers),)
        )
        credentials = grpc.ssl_server_credentials(
            [(Path(key).read_bytes(), Path(cert).read_bytes())],
            root_certificates=Path(ca).read_bytes(),
            require_client_auth=True,
        )
        port = server.add_secure_port(address, credentials)
        if not port:
            raise ControlError("RPC_BIND_FAILED")
        server.start()
        self.server = server
        return port

    def stop(self):
        if self.server is not None:
            self.server.stop(grace=5).wait()


class RegistryClient:
    def __init__(self, environ=None):
        import grpc

        env = os.environ if environ is None else environ
        self.identity = configured_identity(env)
        self.scope = validate_scope(
            {
                "fabric_domain": env.get("RTP_ENGRAM_FABRIC_DOMAIN"),
                "authorization_group": env.get("RTP_ENGRAM_AUTHORIZATION_GROUP"),
            }
        )
        self.instance_id = identifier(env.get("RTP_ENGRAM_INSTANCE_ID"))
        self.timeout = float(env.get("RTP_ENGRAM_RPC_TIMEOUT_S", "10"))
        if not 0 < self.timeout <= 120:
            raise ControlError("RPC_TIMEOUT_INVALID")
        credentials = grpc.ssl_channel_credentials(
            root_certificates=_pem(env, "CA"),
            private_key=_pem(env, "KEY"),
            certificate_chain=_pem(env, "CERT"),
        )
        self.channel = grpc.secure_channel(
            endpoint(env.get(REGISTRY_ENV)), credentials, options=OPTIONS
        )

    def close(self):
        self.channel.close()

    def call(self, method, **fields):
        request = dict(
            schema_version=1,
            identity=self.identity,
            scope=self.scope,
            instance_id=self.instance_id,
            **fields,
        )
        response = self.channel.unary_unary(
            "/" + SERVICE + "/" + method,
            request_serializer=encode,
            response_deserializer=decode,
        )(request, timeout=self.timeout)
        if (
            type(response.get("schema_version")) is not int
            or response.get("schema_version") != 1
            or response.get("identity") != self.identity
            or response.get("scope") != self.scope
            or response.get("state") not in ("EMPTY", "LOADING", "READY", "STALE")
        ):
            raise ControlError("REGISTRY_RESPONSE_MISMATCH")
        if response["state"] != "EMPTY":
            uuid_value(response.get("owner_epoch"))
            identifier(response.get("producer_instance_id"))
            positive_bytes(response.get("reserved_memory_bytes"))
        if response["state"] == "READY":
            endpoint(response.get("owner_endpoint"))
            uuid_value(response.get("generation"))
            positive_bytes(response.get("physical_bytes"))
        if method != "ResolveProducer" and response.get("role") not in (
            "PRODUCER",
            "CONSUMER",
        ):
            raise ControlError("REGISTRY_RESPONSE_MISMATCH")
        if response.get("role") == "PRODUCER":
            if response.get("producer_instance_id") != self.instance_id or response[
                "state"
            ] not in ("LOADING", "READY"):
                raise ControlError("REGISTRY_RESPONSE_MISMATCH")
            endpoint(response.get("owner_endpoint"))
            ttl = response.get("lease_remaining_s")
            if type(ttl) not in (int, float) or not 0 < ttl <= 300:
                raise ControlError("REGISTRY_LEASE_INVALID")
        return response

    def resolve(self):
        return self.call("ResolveProducer")

    def acquire(self, attempt_id, required_memory_bytes):
        return self.call(
            "AcquireProducer",
            attempt_id=attempt_id,
            required_memory_bytes=required_memory_bytes,
        )

    def renew(self, attempt_id, owner_epoch):
        return self.call(
            "RenewProducer", attempt_id=attempt_id, owner_epoch=owner_epoch
        )

    def publish(self, attempt_id, owner_epoch, generation, physical_bytes):
        return self.call(
            "PublishReady",
            attempt_id=attempt_id,
            owner_epoch=owner_epoch,
            generation=generation,
            physical_bytes=physical_bytes,
        )


def resolve_for_launcher(environ=None):
    """Resolve once before rank spawn; no handles, CUDA or per-token lookup."""
    env = os.environ if environ is None else environ
    timeout = float(env.get("RTP_ENGRAM_DISCOVERY_TIMEOUT_S", "600"))
    if not 0 < timeout <= 3600:
        raise ControlError("DISCOVERY_TIMEOUT_INVALID")
    deadline = time.monotonic() + timeout
    client = RegistryClient(env)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControlError("PRODUCER_READY_TIMEOUT")
            client.timeout = min(client.timeout, remaining)
            response = client.resolve()
            if response["state"] == "READY":
                previous = env.get("RTP_ENGRAM_RPC_GENERATION")
                if previous and previous != response["generation"]:
                    raise ControlError("RPC_GENERATION_CHANGED")
                env[RESOLVED_ENDPOINT_ENV] = response["owner_endpoint"]
                env["RTP_ENGRAM_RPC_GENERATION"] = response["generation"]
                env["RTP_ENGRAM_OWNER_EPOCH"] = response["owner_epoch"]
                return response
            if response["state"] == "STALE":
                raise ControlError("PRODUCER_STALE_REQUIRES_PLATFORM_RECOVERY")
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))
    finally:
        client.close()
