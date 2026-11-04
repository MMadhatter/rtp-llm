# SPDX-License-Identifier: Apache-2.0
"""Pure distributed descriptor validation shared by RPC and offline tooling."""
import base64
import json
from uuid import UUID, uuid4

from .placement import MAX_LOCAL_SEGMENTS, tensor_segments


def validate_documents(
    documents, shards, generation, model_revision=None, hash_abi=None
):
    from .config import validate_count
    from .shard import plan_shards

    validate_count(shards)
    if len(documents) != shards:
        raise ValueError("one descriptor per storage host required")
    if any(
        type(d.get("shard")) is not int
        or type(d.get("shards")) is not int
        or type(d.get("schema_version")) is not int
        or d.get("schema_version") not in (2, 3)
        for d in documents
    ):
        raise ValueError("version 2/3 descriptors with integer shard indices required")
    if len({d["schema_version"] for d in documents}) != 1:
        raise ValueError("all storage hosts must use the same NUMA placement mode")
    memory_kind = documents[0].get("memory_kind", "host")
    if memory_kind not in ("host", "gpu") or any(
        d.get("memory_kind", "host") != memory_kind for d in documents
    ):
        raise ValueError("all storage hosts must use the same memory kind")
    if memory_kind == "gpu" and documents[0]["schema_version"] != 3:
        raise ValueError("GPU backing requires segmented descriptors")
    if {d["shard"] for d in documents} != set(range(shards)):
        raise ValueError("descriptor shard indices are incomplete or duplicated")
    documents = sorted(documents, key=lambda d: d["shard"])
    model_revision = (
        documents[0]["model_revision"] if model_revision is None else model_revision
    )
    hash_abi = documents[0]["hash_abi"] if hash_abi is None else hash_abi
    if any(
        d["shards"] != shards
        or d["generation"] != generation
        or d["model_revision"] != model_revision
        or d["hash_abi"] != hash_abi
        for d in documents
    ):
        raise ValueError("descriptor generation/count/revision/hash mismatch")
    hosts = [d.get("host_id") for d in documents]
    if (
        any(type(host) is not str or not host.strip() for host in hosts)
        or len(set(hosts)) != shards
    ):
        raise ValueError("storage host IDs must be nonempty and distinct")
    if any(d["owner_id"] != d["host_id"] for d in documents):
        raise ValueError("owner identity must be the storage host ID")
    names = set(documents[0]["tensors"])
    if not names or any(set(d["tensors"]) != names for d in documents):
        raise ValueError("descriptor tensor sets differ")
    allocations, handles = set(), set()
    gpu_hosts = {}
    for name in names:
        role = name.rsplit(".", 1)[-1]
        if role not in ("weight", "scale"):
            raise ValueError("unsupported tensor role")
        weight_name = name.rsplit(".", 1)[0] + ".weight"
        scale_name = name.rsplit(".", 1)[0] + ".scale"
        if not {weight_name, scale_name} <= names:
            raise ValueError("missing weight/scale pair")
        first = documents[0]["tensors"][weight_name]
        shape = first["shape"]
        if type(shape) is not list or len(shape) != 2:
            raise ValueError("invalid weight shape")
        plan = plan_shards(
            *shape,
            shards,
            granularity=tensor_segments(documents[0], weight_name)[0][
                "allocation_granularity"
            ]
        )
        columns = plan.dim if role == "weight" else plan.dim // plan.block
        size = plan.weight_bytes if role == "weight" else plan.scale_bytes
        for index, document in enumerate(documents):
            if document["tensors"][name].get("shape") != [plan.total_rows, columns]:
                raise ValueError("descriptor tensor shape mismatch")
            pieces = tensor_segments(document, name)
            if not 1 <= len(pieces) <= MAX_LOCAL_SEGMENTS:
                raise ValueError("invalid NUMA segment count")
            start, end = plan.row_range(index)
            cursor, physical, nodes = start, 0, set()
            for entry in pieces:
                for key in (
                    "row_begin",
                    "row_end",
                    "logical_bytes",
                    "allocation_bytes",
                    "allocation_granularity",
                ):
                    if type(entry[key]) is not int:
                        raise ValueError("descriptor geometry must use integers")
                if (
                    type(entry["shape"]) is not list
                    or len(entry["shape"]) != 2
                    or any(type(n) is not int for n in entry["shape"])
                ):
                    raise ValueError("descriptor shape must use integers")
                if (
                    entry["shape"] != [plan.total_rows, columns]
                    or entry["row_begin"] != cursor
                    or not cursor < entry["row_end"] <= end
                    or entry["allocation_bytes"] <= 0
                    or entry["allocation_bytes"] % plan.granularity
                    or entry["allocation_bytes"] % columns
                    or entry["allocation_granularity"] != plan.granularity
                    or entry["logical_bytes"] != (entry["row_end"] - cursor) * columns
                    or entry["logical_bytes"] > entry["allocation_bytes"]
                    or (
                        entry["row_end"] != end
                        and entry["logical_bytes"] != entry["allocation_bytes"]
                    )
                ):
                    raise ValueError(
                        "descriptor geometry does not match storage host index"
                    )
                cursor = entry["row_end"]
                physical += entry["allocation_bytes"]
                if document["schema_version"] == 3:
                    node = (
                        entry.get("device_uuid")
                        if memory_kind == "gpu"
                        else entry.get("numa_node")
                    )
                    if memory_kind == "gpu":
                        if (
                            type(node) is not str
                            or str(UUID(node)) != node
                            or entry.get("numa_node") is not None
                        ):
                            raise ValueError("invalid GPU allocation location")
                    elif (
                        type(node) is not int
                        or not 0 <= node <= 65535
                        or entry.get("device_uuid") is not None
                    ):
                        raise ValueError("invalid NUMA allocation location")
                    if memory_kind == "gpu":
                        host = gpu_hosts.setdefault(node, document["host_id"])
                        if host != document["host_id"]:
                            raise ValueError(
                                "GPU UUID cannot belong to multiple storage hosts"
                            )
                    if node in nodes:
                        raise ValueError("duplicate physical segment location")
                    nodes.add(node)
                aid = entry["allocation_id"]
                if aid in allocations:
                    raise ValueError("duplicate allocation ID")
                allocations.add(aid)
                try:
                    raw = base64.b64decode(entry["handle"], validate=True)
                except (ValueError, TypeError):
                    raise ValueError("invalid capability encoding") from None
                if not 0 < len(raw) <= 768 or raw in handles:
                    raise ValueError("empty, oversized or duplicate capability")
                handles.add(raw)
            if cursor != end or physical != size:
                raise ValueError(
                    "NUMA segments must exactly partition the original shard"
                )
            weight_parts = tensor_segments(document, weight_name)
            scale_parts = tensor_segments(document, scale_name)
            geometry = lambda parts: [
                (e["row_begin"], e["row_end"], e.get("numa_node"), e.get("device_uuid"))
                for e in parts
            ]
            if geometry(weight_parts) != geometry(scale_parts):
                raise ValueError("weight and scale NUMA boundaries differ")
    return documents


def build_manifest(documents, shards, generation, model_revision, hash_abi):
    documents = validate_documents(
        documents, shards, generation, model_revision, hash_abi
    )
    names = sorted(documents[0]["tensors"])
    tensors = []
    for name in names:
        role = name.rsplit(".", 1)[1]
        columns = documents[0]["tensors"][name]["shape"][1]
        entries = []
        for document in documents:
            for entry in tensor_segments(document, name):
                part = dict(
                    shard_id=str(uuid4()),
                    row_begin=int(entry["row_begin"]),
                    row_end=int(entry["row_end"]),
                    logical_bytes=int(entry["logical_bytes"]),
                    byte_offset=0,
                    allocation_bytes=int(entry["allocation_bytes"]),
                    allocation_granularity=int(entry["allocation_granularity"]),
                    owner_id=document["owner_id"],
                    owner_incarnation=document["owner_incarnation"],
                    allocation_id=entry["allocation_id"],
                    handle_ref="capability-" + entry["allocation_id"],
                    sha256=entry["sha256"],
                )
                if document["schema_version"] == 3:
                    key = (
                        "device_uuid"
                        if document.get("memory_kind", "host") == "gpu"
                        else "numa_node"
                    )
                    part[key] = entry[key]
                entries.append(part)
        entries.sort(key=lambda item: item["row_begin"])
        rows = entries[-1]["row_end"]
        tensors.append(
            dict(
                name=name,
                shape=[rows, columns],
                dtype="float8_e4m3fn" if role == "weight" else "float8_e8m0fnu",
                storage_dtype="uint8",
                row_bytes=columns,
                quant_block=32,
                shards=entries,
            )
        )
    document = dict(
        schema_version=2 if documents[0]["schema_version"] == 3 else 1,
        model_revision=model_revision,
        generation=generation,
        layout_abi="engram-fp8-e8m0-v1",
        hash_abi=hash_abi,
        backend=(
            "fabric_gpu"
            if documents[0].get("memory_kind", "host") == "gpu"
            else "fabric_egm"
        ),
        state="READY",
        tensors=tensors,
    )
    from .manifest import compute_dataset_id

    document["dataset_id"] = compute_dataset_id(document)
    from .manifest import parse_manifest

    parse_manifest(json.dumps(document).encode())
    return document
