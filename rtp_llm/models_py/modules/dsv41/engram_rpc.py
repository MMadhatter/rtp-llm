# SPDX-License-Identifier: Apache-2.0
"""Startup-only, mutually authenticated Engram control RPC (no weight RPC).

JSON messages are carried by bounded gRPC unary calls. Consumers bind an
operator-selected service, revision, dataset and generation. Handles never
enter environment variables, logs or bootstrap files on this path.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import threading
from uuid import UUID

SERVICE = "rtp.engram.v1.Owner"
MAX_MESSAGE = 4 << 20
OPTIONS = (
    ("grpc.max_receive_message_length", MAX_MESSAGE),
    ("grpc.max_send_message_length", MAX_MESSAGE),
)
ENDPOINT_ENV = "RTP_ENGRAM_OWNER_ENDPOINT"
IDENTITY_KEYS = ("service_id", "model_revision", "dataset_id")


class ControlError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def encode(value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_MESSAGE:
        raise ControlError("RPC_MESSAGE_TOO_LARGE")
    return raw


def decode(raw):
    from .engram_bootstrap import _json

    if len(raw) > MAX_MESSAGE:
        raise ControlError("RPC_MESSAGE_TOO_LARGE")
    value = _json(raw)
    if type(value) is not dict:
        raise ControlError("RPC_OBJECT_REQUIRED")
    return value


def validate_identity(identity):
    if type(identity) is not dict or identity.keys() != set(IDENTITY_KEYS):
        raise ControlError("RPC_IDENTITY_REQUIRED")
    for key in IDENTITY_KEYS:
        value = identity[key]
        if (
            type(value) is not str
            or not 0 < len(value) <= 512
            or any(ord(c) < 32 for c in value)
        ):
            raise ControlError("RPC_IDENTITY_REQUIRED")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity["dataset_id"]):
        raise ControlError("RPC_DATASET_DIGEST_REQUIRED")
    return dict(identity)


def configured_identity(env):
    return validate_identity(
        {key: env.get("RTP_ENGRAM_" + key.upper()) for key in IDENTITY_KEYS}
    )


def _pem(env, name):
    path = env.get("RTP_ENGRAM_TLS_" + name)
    if not path or not Path(path).is_absolute():
        raise ControlError("RPC_TLS_CONFIGURATION_REQUIRED")
    return Path(path).read_bytes()


class Client:
    def __init__(self, environ=None):
        import grpc

        self.env = os.environ if environ is None else environ
        endpoint = self.env.get(ENDPOINT_ENV, "")
        if self.env.get("RTP_ENGRAM_REGISTRY_ENDPOINT"):
            if endpoint:
                raise ControlError("AMBIGUOUS_ENGRAM_SOURCE")
            endpoint = self.env.get("RTP_ENGRAM_RESOLVED_OWNER_ENDPOINT", "")
        # Explicit host:port; no resolver URI, userinfo or TLS-name override.
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+:[0-9]{1,5}", endpoint):
            raise ControlError("RPC_ENDPOINT_REQUIRED")
        if not 0 < int(endpoint.rsplit(":", 1)[1]) <= 65535:
            raise ControlError("RPC_ENDPOINT_REQUIRED")
        self.identity = configured_identity(self.env)
        self.timeout = float(self.env.get("RTP_ENGRAM_RPC_TIMEOUT_S", "10"))
        if not 0 < self.timeout <= 120:
            raise ControlError("RPC_TIMEOUT_INVALID")
        credentials = grpc.ssl_channel_credentials(
            root_certificates=_pem(self.env, "CA"),
            private_key=_pem(self.env, "KEY"),
            certificate_chain=_pem(self.env, "CERT"),
        )
        self.channel = grpc.secure_channel(endpoint, credentials, options=OPTIONS)

    def close(self):
        self.channel.close()

    def call(self, method, **fields):
        request = {"schema_version": 1, "identity": self.identity, **fields}
        response = self.channel.unary_unary(
            "/" + SERVICE + "/" + method,
            request_serializer=encode,
            response_deserializer=decode,
        )(request, timeout=self.timeout)
        if (
            response.get("schema_version") != 1
            or response.get("identity") != self.identity
        ):
            raise ControlError("RPC_RESPONSE_IDENTITY_MISMATCH")
        generation = response.get("generation")
        try:
            if str(UUID(generation)) != generation:
                raise ValueError()
        except (TypeError, ValueError, AttributeError):
            raise ControlError("RPC_GENERATION_INVALID") from None
        return response

    def describe(self):
        return self.call("Describe")

    def acquire(self, generation):
        response = self.call("Acquire", generation=generation)
        if response["generation"] != generation:
            raise ControlError("RPC_GENERATION_CHANGED")
        return response


def pin_generation(environ=None):
    """Launcher calls once before ranks spawn; no CUDA/handle import here."""
    env = os.environ if environ is None else environ
    client = Client(env)
    try:
        response = client.describe()
        previous = env.get("RTP_ENGRAM_RPC_GENERATION")
        if previous and previous != response["generation"]:
            raise ControlError("RPC_GENERATION_CHANGED")
        env["RTP_ENGRAM_RPC_GENERATION"] = response["generation"]
    finally:
        client.close()


class OwnerControl:
    """One immutable generation. RPC loss never releases published backing."""

    def __init__(self, service_id, allowed_client_names, *, admission=None):
        self.admission = admission
        self.service_id = service_id
        self.allowed = {name.encode() for name in allowed_client_names}
        if not self.allowed or not service_id:
            raise ControlError("RPC_AUTHORIZATION_REQUIRED")
        self._lock = threading.Lock()
        self._payload = None
        self.acquisitions = 0
        self.server = None

    def publish(self, bundle, manifest_bytes, capabilities):
        identity = validate_identity(
            dict(
                service_id=self.service_id,
                model_revision=bundle["model_revision"],
                dataset_id=bundle["dataset_id"],
            )
        )
        payload = dict(
            schema_version=1,
            identity=identity,
            generation=bundle["generation"],
            bundle=bundle,
            manifest=manifest_bytes.decode("utf-8"),
            capabilities=capabilities,
        )
        # Freeze a deep copy and enforce wire budget before becoming visible.
        raw = encode(payload)
        with self._lock:
            if self._payload is not None:
                raise ControlError("RPC_GENERATION_ALREADY_PUBLISHED")
            self._payload = raw

    def _request(self, request, context, acquire):
        import grpc

        names = set(context.auth_context().get("x509_common_name", []))
        if not names.intersection(self.allowed):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "CLIENT_NOT_AUTHORIZED")
        if self.admission is not None and not self.admission():
            context.abort(grpc.StatusCode.UNAVAILABLE, "PRODUCER_CLAIM_NOT_LIVE")
        fields = (
            {"schema_version", "identity", "generation"}
            if acquire
            else {"schema_version", "identity"}
        )
        if (
            request.keys() != fields
            or type(request.get("schema_version")) is not int
            or request["schema_version"] != 1
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST")
        try:
            identity = validate_identity(request["identity"])
        except ControlError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INVALID_IDENTITY")
        with self._lock:
            raw = self._payload
            if raw is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "OWNER_NOT_READY")
            payload = decode(raw)
            if identity != payload["identity"]:
                context.abort(grpc.StatusCode.NOT_FOUND, "DATASET_NOT_FOUND")
            if acquire:
                if request["generation"] != payload["generation"]:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION, "GENERATION_CHANGED"
                    )
                self.acquisitions += 1
                return payload
            return {
                key: payload[key]
                for key in ("schema_version", "identity", "generation")
            }

    def start(self, address, *, ca, cert, key):
        import grpc

        server = grpc.server(
            ThreadPoolExecutor(max_workers=8),
            options=OPTIONS,
            maximum_concurrent_rpcs=32,
        )
        handlers = {
            "Describe": grpc.unary_unary_rpc_method_handler(
                lambda request, ctx: self._request(request, ctx, False),
                request_deserializer=decode,
                response_serializer=encode,
            ),
            "Acquire": grpc.unary_unary_rpc_method_handler(
                lambda request, ctx: self._request(request, ctx, True),
                request_deserializer=decode,
                response_serializer=encode,
            ),
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
