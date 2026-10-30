# SPDX-License-Identifier: Apache-2.0
"""Backend startup adapter: existing intra-instance Store + temporary leader RPC.

Only called before engine construction. No lookup/kernel/Graph code calls this
module. Holders remain owned by the launcher's static cohort lifecycle.
"""
from __future__ import annotations
import base64
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from .engram_exchange import (Cohort, ExchangeError, ExchangeServer, decode, digest, encode,
                              remote_exchange, tls_context, validate_group)
from .engram_bootstrap import _json, _private_read
from .engram_descriptors import build_manifest
from .engram_manifest import parse_manifest

CONFIG_ENV = 'RTP_ENGRAM_STATIC_GROUP_CONFIG'


def checkpoint_samples(checkpoint, manifest):
    """Hash every shard boundary from the trusted coordinator's checkpoint."""
    import torch
    from safetensors import safe_open
    root = Path(checkpoint).resolve(strict=True)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    result = {}
    for tensor in manifest.tensors:
        rows = sorted({0, tensor.shape[0] // 2, tensor.shape[0] - 1} |
                      {r for part in tensor.shards for r in (part.row_begin, part.row_end - 1)})
        relative = Path(index[tensor.name])
        if relative.is_absolute() or '..' in relative.parts:
            raise ExchangeError('UNSAFE_CHECKPOINT_PATH')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ExchangeError('UNSAFE_CHECKPOINT_PATH')
        with safe_open(path, framework='pt', device='cpu') as source:
            view = source.get_slice(tensor.name)
            payload = b''.join(view[r:r + 1].contiguous().view(torch.uint8).numpy().tobytes() for r in rows)
        result[tensor.name] = dict(row_ids=rows, sha256='sha256:' + digest(payload))
    return result


def assemble_package(group, documents, checkpoint, samples=checkpoint_samples):
    p = sum(len(m['shards']) for m in group['members'])
    manifest = build_manifest(documents, p, group['generation'], group['model_revision'], documents[0]['hash_abi'])
    raw = encode(manifest)
    sample_values = samples(checkpoint, parse_manifest(raw))
    capabilities = {entry['allocation_id']: entry['handle'] for d in documents for entry in d['tensors'].values()}
    return dict(manifest=manifest, capabilities=capabilities, samples=sample_values)


def verify_package(group, package):
    """Reject identity/membership/handle mismatches before publishing to workers."""
    if type(package) is not dict or set(package) != {'manifest', 'capabilities', 'samples'}:
        raise ExchangeError('INVALID_PACKAGE')
    manifest = parse_manifest(encode(package['manifest']))
    if manifest.generation != group['generation'] or manifest.model_revision != group['model_revision']:
        raise ExchangeError('PACKAGE_IDENTITY_MISMATCH')
    expected = {s['index']: s['host_id'] for m in group['members'] for s in m['shards']}
    handles = package['capabilities']
    ids = set()
    owners = None
    for tensor in manifest.tensors:
        if len(tensor.shards) != len(expected):
            raise ExchangeError('PACKAGE_SHARD_COUNT_MISMATCH')
        current = [(s.owner_id, s.owner_incarnation) for s in tensor.shards]
        if [v[0] for v in current] != [expected[i] for i in range(len(expected))]:
            raise ExchangeError('PACKAGE_HOST_MISMATCH')
        if owners is not None and current != owners:
            raise ExchangeError('PACKAGE_OWNER_INCONSISTENT')
        owners = current
        ids.update(s.allocation_id for s in tensor.shards)
    if type(handles) is not dict or set(handles) != ids:
        raise ExchangeError('PACKAGE_CAPABILITIES_MISMATCH')
    for value in handles.values():
        if type(value) is not str or not 0 < len(value) <= 1024:
            raise ExchangeError('INVALID_CAPABILITY')
        try:
            if not base64.b64decode(value, validate=True):
                raise ValueError()
        except (ValueError, TypeError):
            raise ExchangeError('INVALID_CAPABILITY') from None
    if type(package['samples']) is not dict or set(package['samples']) != {t.name for t in manifest.tensors}:
        raise ExchangeError('PACKAGE_SAMPLES_MISMATCH')
    # Full layout/hash ABI and GPU content are still checked by model bootstrap.
    return manifest


def materialize_package(group, package, root=None):
    """Local private cache only; never manually transferred between hosts."""
    manifest = verify_package(group, package)
    raw = encode(package['manifest'])
    directory = Path(tempfile.mkdtemp(prefix='rtp-engram-' + group['generation'] + '-', dir=root))
    bundle = dict(schema_version=1, manifest='manifest.json', capabilities='capabilities.json',
                  manifest_sha256='sha256:' + digest(raw), dataset_id=manifest.dataset_id,
                  generation=manifest.generation, model_revision=manifest.model_revision,
                  hash_abi=manifest.hash_abi, samples=package['samples'])
    try:
        for name, data in (('manifest.json', raw), ('capabilities.json', encode(package['capabilities'])),
                           ('bundle.json', encode(bundle))):
            fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
    except BaseException:
        for name in ('manifest.json', 'capabilities.json', 'bundle.json'):
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()
        raise
    return str(directory / 'bundle.json')


def _wait(store, keys, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExchangeError('INSTANCE_STARTUP_TIMEOUT')
    store.wait(keys, timedelta(seconds=remaining))


def exchange_leaders(config, documents, deadline, build=None):
    group = config['group']
    group_digest = validate_group(group)
    ident = config['instance_id']
    coordinator = group['members'][0]['instance_id']
    builder = build or (lambda docs: assemble_package(group, docs, config['checkpoint']))
    verifier = lambda package: verify_package(group, package)
    if len(group['members']) == 1:
        cohort = Cohort(group, builder, max(0.001, deadline - time.monotonic()))
        return decode(cohort.submit(ident, group_digest, documents))['package']
    address = config['coordinator']
    if type(address) is not dict or set(address) != {'host', 'port'} or type(address['port']) is not int or not 1 <= address['port'] <= 65535:
        raise ExchangeError('INVALID_COORDINATOR')
    tls = config['tls']
    fingerprints = tls['instance_cert_sha256']
    if (set(fingerprints) != {m['instance_id'] for m in group['members']}
            or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in fingerprints.values())
            or len(set(fingerprints.values())) != len(fingerprints)):
        raise ExchangeError('DISTINCT_MEMBER_CERTIFICATES_REQUIRED')
    if ident != coordinator:
        return remote_exchange((address['host'], address['port']), tls_context(tls), ident, group_digest,
                               documents, deadline, verifier, server_fingerprint=fingerprints[coordinator])
    cohort = Cohort(group, builder, max(0.001, deadline - time.monotonic()))
    server = ExchangeServer((address['host'], address['port']), cohort, tls_context(tls, server=True), fingerprints)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        payload = cohort.submit(ident, group_digest, documents)
        package = decode(payload)['package']
        verifier(package)
        cohort.ack(ident, digest(payload))
        cohort.wait_acknowledged()
        return package
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def prepare_static_engram(store, *, rank, world_size, environ=None, leader_exchange=exchange_leaders):
    """All ranks enter once. Only world rank 0 opens cross-instance connections."""
    env = os.environ if environ is None else environ
    path = env.get(CONFIG_ENV)
    if path is None:
        return None
    if 'RTP_ENGRAM_EXTERNAL_BUNDLE' in env:
        raise ExchangeError('SELECT_STATIC_EXCHANGE_OR_OFFLINE_BUNDLE')
    if not Path(path).is_absolute():
        raise ExchangeError('CONFIG_PATH_MUST_BE_ABSOLUTE')
    config = _json(_private_read(path, 1 << 20))
    group = config['group']
    group_digest = validate_group(group)
    members = {m['instance_id']: m for m in group['members']}
    member = members.get(config['instance_id'])
    if member is None or member['world_size'] != world_size or not 0 <= rank < world_size:
        raise ExchangeError('INSTANCE_WORLD_SIZE_MISMATCH')
    timeout = config.get('timeout_seconds', 1800)
    if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
        raise ExchangeError('INVALID_TIMEOUT')
    p = sum(len(m['shards']) for m in group['members'])
    if 'ENGRAM_SHARD_COUNT' in env and env['ENGRAM_SHARD_COUNT'] != str(p):
        raise ExchangeError('STORAGE_HOST_COUNT_MISMATCH')
    deadline = time.monotonic() + timeout
    if store is None:
        if world_size != 1:
            raise ExchangeError('INSTANCE_STORE_REQUIRED')
        from torch.distributed import HashStore
        store = HashStore()
    prefix = 'engram/' + group['generation'] + '/' + config['instance_id'] + '/'
    # Each worker must agree on the public contract, even if its config differs.
    store.set(prefix + f'config/{rank}', group_digest.encode())
    if rank == 0:
        config_keys = [prefix + f'config/{i}' for i in range(world_size)]
        _wait(store, config_keys, deadline)
        if any(store.get(k) != group_digest.encode() for k in config_keys):
            raise ExchangeError('INSTANCE_GROUP_CONFIG_MISMATCH')
    for shard in member['shards']:
        if shard['publisher_rank'] != rank:
            continue
        descriptor = Path(config['local_descriptors'][str(shard['index'])])
        if not descriptor.is_absolute():
            raise ExchangeError('DESCRIPTOR_PATH_MUST_BE_ABSOLUTE')
        while not descriptor.exists():
            if time.monotonic() >= deadline:
                raise ExchangeError('LOCAL_SHARD_NOT_READY')
            time.sleep(0.05)
        raw = _private_read(descriptor, 1 << 20)
        store.set(prefix + f'shard/{shard["index"]}', raw)
    if rank == 0:
        keys = [prefix + f'shard/{s["index"]}' for s in member['shards']]
        _wait(store, keys, deadline)
        documents = [decode(store.get(key)) for key in keys]
        package = leader_exchange(config, documents, deadline)
        verify_package(group, package)
        store.set(prefix + 'package', encode(package))
    _wait(store, [prefix + 'package'], deadline)
    package = decode(store.get(prefix + 'package'))
    bundle = materialize_package(group, package, root=config.get('cache_directory'))
    env['ENGRAM_SHARD_COUNT'] = str(p)
    env['RTP_ENGRAM_EXTERNAL_BUNDLE'] = bundle
    store.set(prefix + f'received/{rank}', b'1')
    if rank == 0:
        _wait(store, [prefix + f'received/{i}' for i in range(world_size)], deadline)
        # All workers now have private local caches. No serving-time Store reads.
        for key in ([prefix + 'package'] + keys + config_keys +
                    [prefix + f'received/{i}' for i in range(world_size)]):
            store.delete_key(key)
    return bundle
