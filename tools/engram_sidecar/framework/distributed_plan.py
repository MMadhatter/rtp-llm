#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a static distributed Engram launch plan without remote deployment.

With exchange configuration, existing per-instance Store and temporary mTLS RPC
collect handles automatically during BackendManager startup. Without exchange,
the explicit offline bundle workflow remains available for diagnostics.
Start each holder on its declared host; retain all holders until every model
worker and CUDA graph in the fixed cohort has exited. No live reshard/election.
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
    result = dict(instance_count=n, membership='static', storage_host_count=p, generation=generation, holders=holders,
                coordinator_argv=bundle, consumers=consumers,
                lifecycle='all N instances start their assigned P holders -> complete bundle -> all N instances attach; stop all N before retiring any holder')
    if config.get('exchange') is not None:
        from rtp_llm.models_py.modules.dsv41.engram_exchange import validate_group
        exchange = config['exchange']
        members = []
        for instance in instances:
            local_size = instance.get('local_world_size')
            if type(local_size) is not int or local_size < 1:
                raise ValueError('RPC mode requires each instance local_world_size')
            member = dict(instance_id=instance['instance_id'],
                          world_size=local_size * len(instance['hosts']), shards=[])
            for holder in holders:
                if holder['instance_id'] == instance['instance_id']:
                    member['shards'].append(dict(index=holder['shard_index'], host_id=holder['host_id'],
                        publisher_rank=local_size * instance['hosts'].index(holder['host_id'])))
            members.append(member)
        group = dict(generation=generation, model_revision=revision, members=members)
        validate_group(group)
        result['startup_configs'] = {}
        result['coordinator_argv'] = None  # No manual bundle merge/copy in RPC mode.
        for member_index, consumer in enumerate(consumers):
            ident = consumer['instance_id']
            startup = dict(group=group, instance_id=ident, checkpoint=config['checkpoint'],
                coordinator=exchange.get('coordinator'),
                tls=exchange.get('tls', {}).get(ident),
                timeout_seconds=exchange.get('timeout_seconds', 1800),
                local_descriptors={str(index): f'{root}/shard-{index}/shard.json' for index in consumer['held_shards']})
            if n > 1 and (startup['coordinator'] is None or startup['tls'] is None):
                raise ValueError('multi-instance RPC requires coordinator address and per-instance TLS configuration')
            result['startup_configs'][ident] = startup
            consumer['env'].pop('RTP_ENGRAM_EXTERNAL_BUNDLE')
            consumer['env']['RTP_ENGRAM_STATIC_GROUP_CONFIG'] = f'{root}/startup-{member_index}.json'
        result['lifecycle'] = 'P holders READY -> existing instance Store -> one RPC connection per remote representative -> all-member ACK -> model attach'
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, help='Write private plan/startup configs for launcher distribution')
    args = parser.parse_args()
    plan = render_plan(json.loads(args.config.read_text()))
    if args.output_directory:
        import os
        destination = args.output_directory.resolve()
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        documents = {'launch-plan.json': plan}
        for index, consumer in enumerate(plan['consumers']):
            if 'startup_configs' in plan:
                documents[f'startup-{index}.json'] = plan['startup_configs'][consumer['instance_id']]
        for name, document in documents.items():
            fd = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(document, stream, indent=2)
        print(destination / 'launch-plan.json')
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
