#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a distributed storage launch plan, without creating containers or RPC.

Run each holder command on its declared host. After all descriptors are READY,
transfer them privately to the coordinator and run the bundle command. Transfer
that SAME bundle to every consumer host, then start their existing RTP commands
with the emitted env. Holders must outlive all consumers; STOP is manual only
once every consumer and CUDA graph has exited. This tool does not perform remote
execution, service discovery, topology attestation or live resharding.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from uuid import UUID

from rtp_llm.models_py.modules.dsv41.engram_shard_config import validate_count


def render_plan(config):
    """Storage host count P and the instance/worker topology are independent."""
    from rtp_llm.models_py.modules.dsv41.engram_capacity import validate_capacity
    allocation, total = validate_capacity(config.get('max_allocation_bytes', 128 << 30),
                                          config.get('max_total_allocation_bytes', 256 << 30))
    p = validate_count(config['storage_host_count'])
    hosts = config['storage_hosts']
    if type(hosts) is not list or len(hosts) != p:
        raise ValueError('storage_hosts length must equal P')
    ids = [h['host_id'] for h in hosts]
    if any(type(i) is not str or not i.strip() or len(i) > 256 for i in ids) or len(set(ids)) != p:
        raise ValueError('P distinct physical storage host IDs required')
    generation = config['generation']
    if str(UUID(generation)) != generation:
        raise ValueError('canonical generation UUID required')
    for key in ('checkpoint', 'publication_root'):
        if type(config[key]) is not str or not Path(config[key]).is_absolute():
            raise ValueError('checkpoint and publication_root must be absolute')
    if type(config['model_revision']) is not str or not config['model_revision'].strip():
        raise ValueError('model revision required')
    revision = config['model_revision']
    root = str(Path(config['publication_root']) / generation)
    framework = 'tools.engram_sidecar.framework.'
    shared = ['--checkpoint', config['checkpoint'], '--model-revision', revision,
              '--generation', generation, '--shards', str(p)]
    holders = []
    for index, host in enumerate(hosts):
        for key in ('device', 'numa'):
            if type(host[key]) is not int or host[key] < 0:
                raise ValueError('local device and NUMA ordinals must be nonnegative integers')
        command = ['python3', '-m', framework + 'sharded_checkpoint_owner', *shared,
                   '--host-id', host['host_id'], '--shard', str(index),
                   '--publication', f'{root}/shard-{index}',
                   '--device', str(host['device']), '--numa', str(host['numa']),
                   '--max-allocation-bytes', str(allocation), '--budget-bytes', str(total)]
        holders.append(dict(host_id=host['host_id'], shard_index=index, argv=command))
    instances = config['instances']
    if type(instances) is not list or not instances:
        raise ValueError('at least one instance required')
    n = config.get('instance_count', len(instances))
    if type(n) is not int or n < 1 or n != len(instances):
        raise ValueError('instance_count N must equal the complete static instance list')
    seen, consumers = set(), []
    for instance in instances:
        identifier = instance['instance_id']
        workers = instance['hosts']
        if type(identifier) is not str or not identifier or identifier in seen:
            raise ValueError('instance IDs must be nonempty and distinct')
        seen.add(identifier)
        if (type(workers) is not list or not workers
                or any(type(h) is not str or not h.strip() for h in workers)
                or len(set(workers)) != len(workers)):
            raise ValueError('each instance needs distinct consumer host IDs')
        consumers.append(dict(instance_id=identifier, hosts=workers,
            env={'RTP_ENGRAM_MAX_ALLOCATION_BYTES': str(allocation),
                 'RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES': str(total),
                 'ENGRAM_SHARD_COUNT': str(p), 'RTP_ENGRAM_EXTERNAL_BUNDLE': f'{root}/bundle/bundle.json'}))
    # Every declared instance participates in holding backing. This is a fixed
    # cohort, not an expandable consumer list; joining/removing requires a new
    # generation. One holder per physical storage host is the current contract.
    assignments = {identifier: [] for identifier in seen}
    for host, holder in zip(hosts, holders):
        candidates = [instance['instance_id'] for instance in instances
                      if host['host_id'] in instance['hosts']]
        owner = host.get('instance_id')
        if owner is None:
            if len(candidates) != 1:
                raise ValueError('storage host must belong to exactly one instance, or specify instance_id')
            owner = candidates[0]
        if owner not in candidates:
            raise ValueError('storage holder instance must contain its physical host')
        holder['instance_id'] = owner
        assignments[owner].append(holder['shard_index'])
    if any(not indices for indices in assignments.values()):
        raise ValueError('every static instance must hold at least one shard; consumer-only instances are not supported')
    for consumer in consumers:
        consumer['held_shards'] = assignments[consumer['instance_id']]
    bundle = ['python3', '-m', framework + 'sharded_bundle', *shared,
              '--publication', f'{root}/bundle']
    for index in range(p):
        bundle += ['--shard-json', f'{root}/shard-{index}/shard.json']
    return dict(instance_count=n, membership='static', storage_host_count=p, generation=generation, holders=holders,
                coordinator_argv=bundle, consumers=consumers,
                lifecycle='all N instances start their assigned P holders -> complete bundle -> all N instances attach; stop all N before retiring any holder')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_plan(json.loads(args.config.read_text())), indent=2))


if __name__ == '__main__':
    main()
