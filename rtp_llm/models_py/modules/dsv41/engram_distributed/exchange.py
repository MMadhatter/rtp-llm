# SPDX-License-Identifier: Apache-2.0
"""One-shot static-cohort metadata exchange. No weights, CUDA or inference RPC.

One authenticated persistent connection per remote instance representative.
All payloads are bounded JSON. No pickle, discovery, election, or live reshard.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import UUID

MAX_MESSAGE = 4 << 20

# Only these protocol codes may cross the wire. An arbitrary exception message,
# including one wrapped in ExchangeError by an extension, is never public data.
_WIRE_ERRORS = frozenset(
    """
    MESSAGE_TOO_LARGE INVALID_MESSAGE INVALID_GROUP INVALID_GENERATION
    INVALID_REVISION INVALID_MEMBERS INVALID_MEMBER INVALID_INSTANCE_ID
    INVALID_WORLD_SIZE EVERY_INSTANCE_MUST_HOLD_SHARDS INVALID_SHARD
    INVALID_SHARD_INDEX DUPLICATE_OR_INVALID_HOST INVALID_PUBLISHER_RANK
    INCOMPLETE_SHARD_SET UNKNOWN_INSTANCE INCOMPLETE_INSTANCE_SHARDS
    INVALID_DESCRIPTOR DESCRIPTOR_IDENTITY_MISMATCH COHORT_TIMEOUT
    GROUP_MISMATCH CONFLICTING_REPUBLICATION ASSEMBLY_FAILED ACK_MISMATCH
    INVALID_FRAME TRUNCATED_FRAME UNAUTHORIZED_INSTANCE INVALID_METHOD
    INVALID_REQUEST REMOTE_EXCHANGE_FAILED UNSAFE_CHECKPOINT_PATH
""".split()
)

_MANIFEST_ERRORS = frozenset(
    """
    INVALID_LIMIT INVALID_LAYOUT DUPLICATE_LAYER UNSUPPORTED_LAYOUT
    UNSUPPORTED_BACKEND INVALID_INTEGER INVALID_TEXT INVALID_DIGEST INVALID_UUID
    INVALID_FIELDS UNSUPPORTED_SCHEMA INVALID_STATE TENSOR_LIMIT
    INVALID_TENSOR_NAME DUPLICATE_TENSOR INVALID_SHAPE UNSUPPORTED_DTYPE
    UNSUPPORTED_QUANTIZATION INVALID_STRIDE SIZE_OVERFLOW UNSUPPORTED_SHARDING
    DUPLICATE_SHARD INCOMPLETE_RANGE INVALID_RANGE INVALID_LOGICAL_BYTES
    INVALID_ALLOCATION_SIZE OUT_OF_BOUNDS UNEVEN_SHARDS ALLOCATION_CONFLICT
    MULTIPLE_OWNERS TOTAL_ALLOCATION_LIMIT OVERLAPPING_TENSORS
    INCOMPLETE_WEIGHT_SCALE INPUT_LIMIT DUPLICATE_KEY INVALID_JSON
    DATASET_DIGEST_MISMATCH DATASET_MISMATCH GENERATION_MISMATCH NOT_READY
    BACKEND_MISMATCH MODEL_CONTRACT_MISMATCH INCOMPLETE_GENERATION
""".split()
)

_ERROR_CLASSES = frozenset(
    (
        "exchange",
        "manifest",
        "file_not_found",
        "permission",
        "io",
        "import",
        "validation",
        "missing_field",
        "type",
        "internal",
    )
)
_CAUSE_CODES = (
    _WIRE_ERRORS
    | _MANIFEST_ERRORS
    | frozenset(
        (
            "FILE_NOT_FOUND",
            "PERMISSION_DENIED",
            "IO_ERROR",
            "IMPORT_FAILED",
            "VALIDATION_FAILED",
            "MISSING_FIELD",
            "INVALID_TYPE",
            "INTERNAL_ERROR",
            "MANIFEST_VALIDATION_FAILED",
        )
    )
)


def _allowed(value, known, fallback=None):
    return value if type(value) is str and value in known else fallback


def _log(level, message, *args):
    # Startup progress and failure propagation must not depend on log handlers.
    try:
        logging.getLogger(__name__).log(level, message, *args)
    except Exception:
        pass


class ExchangeError(RuntimeError):
    """Public protocol code plus optional, bounded diagnostic categories."""

    def __init__(self, code, *, error_class=None, cause_code=None):
        self.code = code
        self.error_class = _allowed(error_class, _ERROR_CLASSES)
        self.cause_code = _allowed(cause_code, _CAUSE_CODES)
        super().__init__(code)


def classify_failure(error):
    """Classify a startup exception without exporting arbitrary exception text."""
    from .manifest import ManifestError

    if isinstance(error, ExchangeError):
        category, code = "exchange", _allowed(
            error.code, _WIRE_ERRORS, "INTERNAL_ERROR"
        )
    elif isinstance(error, ManifestError):
        category, code = "manifest", _allowed(
            error.code, _MANIFEST_ERRORS, "MANIFEST_VALIDATION_FAILED"
        )
    else:
        # Match subclasses before their base classes; never inspect error text,
        # filenames, custom class names or descriptor values.
        category, code = "internal", "INTERNAL_ERROR"
        for kind, category_name, cause in (
            (FileNotFoundError, "file_not_found", "FILE_NOT_FOUND"),
            (PermissionError, "permission", "PERMISSION_DENIED"),
            (OSError, "io", "IO_ERROR"),
            (ImportError, "import", "IMPORT_FAILED"),
            (ValueError, "validation", "VALIDATION_FAILED"),
            (KeyError, "missing_field", "MISSING_FIELD"),
            (TypeError, "type", "INVALID_TYPE"),
        ):
            if isinstance(error, kind):
                category, code = category_name, cause
                break
    return ExchangeError("ASSEMBLY_FAILED", error_class=category, cause_code=code)


def _error_response(error):
    code = (
        _allowed(error.code, _WIRE_ERRORS) if isinstance(error, ExchangeError) else None
    )
    code = code or "INVALID_REQUEST"
    result = {"error": code}
    if isinstance(error, ExchangeError):
        if error.error_class is not None:
            result["error_class"] = error.error_class
        if error.cause_code is not None:
            result["cause_code"] = error.cause_code
    return result


def _remote_failure(data):
    """Accept old code-only responses and sanitize newer diagnostic fields."""
    try:
        value = decode(data)
    except (ValueError, ExchangeError):
        return ExchangeError("REMOTE_EXCHANGE_FAILED")
    code = _allowed(value.get("error"), _WIRE_ERRORS)
    if code is None:
        return ExchangeError("REMOTE_EXCHANGE_FAILED")
    return ExchangeError(
        code, error_class=value.get("error_class"), cause_code=value.get("cause_code")
    )


def encode(value):
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(data) > MAX_MESSAGE:
        raise ExchangeError("MESSAGE_TOO_LARGE")
    return data


def decode(data):
    from .bootstrap import _json

    if len(data) > MAX_MESSAGE:
        raise ExchangeError("MESSAGE_TOO_LARGE")
    value = _json(data)
    if type(value) is not dict:
        raise ExchangeError("INVALID_MESSAGE")
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_group(group):
    """Public immutable startup contract; no credentials or local filesystem paths."""
    from .config import validate_count

    if type(group) is not dict or set(group) != {
        "generation",
        "model_revision",
        "members",
    }:
        raise ExchangeError("INVALID_GROUP")
    try:
        if str(UUID(group["generation"])) != group["generation"]:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ExchangeError("INVALID_GENERATION") from None
    if (
        type(group["model_revision"]) is not str
        or not 0 < len(group["model_revision"]) <= 256
    ):
        raise ExchangeError("INVALID_REVISION")
    members = group["members"]
    if type(members) is not list or not 1 <= len(members) <= 64:
        raise ExchangeError("INVALID_MEMBERS")
    ids, hosts, indices = set(), set(), []
    for member in members:
        if type(member) is not dict or set(member) != {
            "instance_id",
            "world_size",
            "shards",
        }:
            raise ExchangeError("INVALID_MEMBER")
        ident = member["instance_id"]
        if type(ident) is not str or not 0 < len(ident) <= 128 or ident in ids:
            raise ExchangeError("INVALID_INSTANCE_ID")
        ids.add(ident)
        if (
            type(member["world_size"]) is not int
            or not 1 <= member["world_size"] <= 4096
        ):
            raise ExchangeError("INVALID_WORLD_SIZE")
        if type(member["shards"]) is not list or not member["shards"]:
            raise ExchangeError("EVERY_INSTANCE_MUST_HOLD_SHARDS")
        publishers = set()
        for shard in member["shards"]:
            if type(shard) is not dict or set(shard) != {
                "index",
                "host_id",
                "publisher_rank",
            }:
                raise ExchangeError("INVALID_SHARD")
            if type(shard["index"]) is not int or shard["index"] < 0:
                raise ExchangeError("INVALID_SHARD_INDEX")
            if (
                type(shard["host_id"]) is not str
                or not 0 < len(shard["host_id"]) <= 256
                or shard["host_id"] in hosts
            ):
                raise ExchangeError("DUPLICATE_OR_INVALID_HOST")
            rank = shard["publisher_rank"]
            if (
                type(rank) is not int
                or not 0 <= rank < member["world_size"]
                or rank in publishers
            ):
                raise ExchangeError("INVALID_PUBLISHER_RANK")
            hosts.add(shard["host_id"])
            publishers.add(rank)
            indices.append(shard["index"])
    validate_count(len(indices))
    if sorted(indices) != list(range(len(indices))):
        raise ExchangeError("INCOMPLETE_SHARD_SET")
    return digest(encode(group))


def validate_submission(group, instance_id, documents):
    members = {m["instance_id"]: m for m in group["members"]}
    if instance_id not in members or type(documents) is not list:
        raise ExchangeError("UNKNOWN_INSTANCE")
    expected = {s["index"]: s["host_id"] for s in members[instance_id]["shards"]}
    if len(documents) != len(expected):
        raise ExchangeError("INCOMPLETE_INSTANCE_SHARDS")
    seen = set()
    for doc in documents:
        if type(doc) is not dict or type(doc.get("shard")) is not int:
            raise ExchangeError("INVALID_DESCRIPTOR")
        index = doc["shard"]
        if (
            index in seen
            or index not in expected
            or doc.get("host_id") != expected[index]
            or doc.get("generation") != group["generation"]
            or doc.get("model_revision") != group["model_revision"]
        ):
            raise ExchangeError("DESCRIPTOR_IDENTITY_MISMATCH")
        seen.add(index)


class Cohort:
    """Static startup barrier; never retire physical backing on timeout/disconnect."""

    def __init__(self, group, build, timeout):
        self.group = decode(encode(group))
        self.group_digest = validate_group(group)
        self.members = {m["instance_id"] for m in group["members"]}
        self.build = build
        self.deadline = time.monotonic() + timeout
        self.condition = threading.Condition()
        self.documents = {}
        self.payload = None
        self.error = None
        self.acks = set()
        self.building = False

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ExchangeError("COHORT_TIMEOUT")
        return remaining

    def submit(self, instance_id, group_digest, documents):
        self.remaining()
        if group_digest != self.group_digest:
            raise ExchangeError("GROUP_MISMATCH")
        validate_submission(self.group, instance_id, documents)
        encoded = encode({"documents": documents})
        with self.condition:
            previous = self.documents.get(instance_id)
            if previous is not None and previous != encoded:
                raise ExchangeError("CONFLICTING_REPUBLICATION")
            self.documents[instance_id] = encoded
            _log(
                logging.INFO,
                "engram exchange: %s submitted (%d/%d instances)",
                instance_id,
                len(self.documents),
                len(self.members),
            )
            if len(self.documents) == len(self.members) and not self.building:
                self.building = True
                _log(
                    logging.INFO,
                    "engram exchange: all %d instances submitted, assembling package",
                    len(self.members),
                )
                all_docs = [
                    d
                    for ident in sorted(self.documents)
                    for d in decode(self.documents[ident])["documents"]
                ]

                def assemble():
                    try:
                        result = self.build(all_docs)
                        payload = encode(
                            {"group_digest": self.group_digest, "package": result}
                        )
                    except Exception as error:
                        failure = classify_failure(error)
                        with self.condition:
                            self.error = failure
                            self.condition.notify_all()
                        _log(
                            logging.ERROR,
                            "engram event=assembly_failed phase=assembly generation=%s "
                            "group_digest=%s instance_count=%d error_code=%s "
                            "error_class=%s cause_code=%s",
                            self.group["generation"],
                            self.group_digest[:12],
                            len(self.members),
                            failure.code,
                            failure.error_class,
                            failure.cause_code,
                        )
                    else:
                        with self.condition:
                            self.payload = payload
                            self.condition.notify_all()
                        _log(
                            logging.INFO,
                            "engram exchange: package assembled (%d bytes)",
                            len(payload),
                        )

                threading.Thread(target=assemble, daemon=True).start()
            while self.payload is None and self.error is None:
                self.condition.wait(self.remaining())
            if self.error:
                # Each waiter owns its traceback; never retain the builder's
                # exception, whose context may contain capabilities or paths.
                raise ExchangeError(
                    self.error.code,
                    error_class=self.error.error_class,
                    cause_code=self.error.cause_code,
                )
            return self.payload

    def ack(self, instance_id, payload_digest):
        with self.condition:
            self.remaining()
            if (
                instance_id not in self.documents
                or self.payload is None
                or digest(self.payload) != payload_digest
            ):
                raise ExchangeError("ACK_MISMATCH")
            self.acks.add(instance_id)
            self.condition.notify_all()

    def wait_acknowledged(self):
        with self.condition:
            while self.acks != self.members:
                self.condition.wait(self.remaining())
        _log(
            logging.INFO,
            "engram exchange: all %d instances acknowledged",
            len(self.members),
        )


class ExchangeServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, address, cohort, context, fingerprints):
        if context is not None:
            if (
                not isinstance(context, ssl.SSLContext)
                or context.verify_mode != ssl.CERT_REQUIRED
            ):
                raise ExchangeError("MUTUAL_TLS_REQUIRED")
            if not fingerprints:
                raise ExchangeError("MUTUAL_TLS_REQUIRED")
        self.cohort, self.context = cohort, context
        self.fingerprints = fingerprints or {}
        self.connections = 0
        self._slots = threading.BoundedSemaphore(len(cohort.members) + 2)
        super().__init__(address, ExchangeHandler)
        _log(
            logging.INFO,
            "engram exchange: listening on %s:%d for %d instances (tls=%s)",
            address[0],
            address[1],
            len(cohort.members),
            context is not None,
        )

    def get_request(self):
        raw, address = super().get_request()
        _log(logging.DEBUG, "engram exchange: connection from %s", address)
        try:
            # Bound every unauthenticated read: one wedged peer must not hold a
            # handler slot for the whole cohort window. The longest frame is 4 MiB.
            raw.settimeout(min(5, self.cohort.remaining()))
            if self.context is None:
                self.connections += 1
                return raw, address
            secure = self.context.wrap_socket(raw, server_side=True)
            secure.settimeout(self.cohort.remaining())
            self.connections += 1
            return secure, address
        except Exception as error:
            raw.close()
            # HTTPServer handles OSError from accept/handshake without leaking a
            # traceback or leaving the accepted descriptor open at the deadline.
            raise OSError("EXCHANGE_ACCEPT_FAILED") from error

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, address)

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        # Never log opaque capabilities or TLS peer details via tracebacks.
        pass


class ExchangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def respond(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_POST(self):
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise ExchangeError("INVALID_FRAME")
            length = int(lengths[0])
            if not 0 < length <= MAX_MESSAGE:
                raise ExchangeError("INVALID_FRAME")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ExchangeError("TRUNCATED_FRAME")
            request = decode(raw)
            ident = request.get("instance_id")
            # Plain-HTTP mode has no peer certificate; the group digest still
            # binds membership, but identity is self-asserted on a trusted link.
            if self.server.context is not None:
                cert_digest = digest(self.connection.getpeercert(binary_form=True))
                if self.server.fingerprints.get(ident) != cert_digest:
                    raise ExchangeError("UNAUTHORIZED_INSTANCE")
            if self.path == "/v1/exchange" and set(request) == {
                "instance_id",
                "group_digest",
                "documents",
            }:
                payload = self.server.cohort.submit(
                    ident, request["group_digest"], request["documents"]
                )
                self.respond(200, payload)
            elif self.path == "/v1/ack" and set(request) == {
                "instance_id",
                "payload_digest",
            }:
                self.server.cohort.ack(ident, request["payload_digest"])
                self.server.cohort.wait_acknowledged()
                self.close_connection = True
                self.respond(200, b'{"acknowledged":true}')
            else:
                raise ExchangeError("INVALID_METHOD")
        except Exception as error:
            self.close_connection = True
            try:
                self.respond(400, encode(_error_response(error)))
            except (OSError, ValueError):
                pass


def tls_context(tls, server=False):
    context = ssl.create_default_context(
        ssl.Purpose.CLIENT_AUTH if server else ssl.Purpose.SERVER_AUTH,
        cafile=tls["ca_file"],
    )
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(tls["cert_file"], tls["key_file"])
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def remote_exchange(
    address,
    context,
    instance_id,
    group_digest,
    documents,
    deadline,
    verify,
    server_fingerprint=None,
):
    """One connection, one payload transfer, one receipt ACK; no steady-state use."""
    _log(
        logging.INFO,
        "engram exchange: %s connecting to coordinator %s:%d",
        instance_id,
        address[0],
        address[1],
    )
    connection = None
    while connection is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExchangeError("CONNECT_TIMEOUT")
        if context is None:
            candidate = http.client.HTTPConnection(
                address[0], address[1], timeout=min(remaining, 5)
            )
        else:
            candidate = http.client.HTTPSConnection(
                address[0], address[1], context=context, timeout=min(remaining, 5)
            )
        try:
            candidate.connect()
            if (
                context is not None
                and server_fingerprint is not None
                and digest(candidate.sock.getpeercert(binary_form=True))
                != server_fingerprint
            ):
                candidate.close()
                raise ExchangeError("COORDINATOR_CERTIFICATE_MISMATCH")
            candidate.sock.settimeout(max(0.001, deadline - time.monotonic()))
            connection = candidate
            _log(logging.INFO, "engram exchange: %s connected", instance_id)
        except ssl.SSLError:
            candidate.close()
            raise ExchangeError("TLS_AUTHENTICATION_FAILED") from None
        except OSError:
            candidate.close()
            _log(
                logging.DEBUG,
                "engram exchange: %s connect retry in %.1fs",
                instance_id,
                min(0.1, max(0, deadline - time.monotonic())),
            )
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    try:

        def call(path, value):
            connection.request(
                "POST", path, encode(value), {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            length = response.getheader("Content-Length", "")
            if not length.isdecimal() or not 0 < int(length) <= MAX_MESSAGE:
                raise ExchangeError("INVALID_RESPONSE_LENGTH")
            data = response.read(int(length) + 1)
            if len(data) != int(length):
                raise ExchangeError("REMOTE_EXCHANGE_FAILED")
            if response.status != 200:
                failure = _remote_failure(data)
                _log(
                    logging.ERROR,
                    "engram event=remote_exchange_failed phase=exchange "
                    "group_digest=%s error_code=%s error_class=%s cause_code=%s",
                    group_digest[:12],
                    failure.code,
                    failure.error_class,
                    failure.cause_code,
                )
                raise failure from None
            return data

        payload = call(
            "/v1/exchange",
            dict(
                instance_id=instance_id, group_digest=group_digest, documents=documents
            ),
        )
        decoded = decode(payload)
        if decoded.get("group_digest") != group_digest:
            raise ExchangeError("GROUP_MISMATCH")
        verify(decoded["package"])
        call("/v1/ack", dict(instance_id=instance_id, payload_digest=digest(payload)))
        _log(
            logging.INFO,
            "engram exchange: %s received package (%d bytes) and acknowledged",
            instance_id,
            len(payload),
        )
        return decoded["package"]
    finally:
        connection.close()
