# SPDX-License-Identifier: Apache-2.0
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from uuid import UUID

from rtp_llm.models_py.modules.dsv41 import engram_exchange as e
from rtp_llm.models_py.modules.dsv41 import engram_startup as s
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


def fixture(n=2, p=3, ranks=2):
    group = dict(generation=str(UUID(int=123)), model_revision='test-revision', members=[])
    docs = []
    plan = plan_shards(p * 4096 - 7, 64, p, granularity=4096)
    for instance in range(n):
        shards = [dict(index=i, host_id=f'host-{i}', publisher_rank=pos)
                  for pos, i in enumerate(range(instance, p, n))]
        group['members'].append(dict(instance_id=f'i{instance}', world_size=ranks, shards=shards))
    for index in range(p):
        begin, end = plan.row_range(index)
        tensors = {}
        for role, width, size in [('weight', 64, plan.weight_bytes), ('scale', 2, plan.scale_bytes)]:
            aid = str(UUID(int=1000 + 2 * index + (role == 'scale')))
            tensors['layers.1.engram.embed.' + role] = dict(shape=[plan.total_rows, width],
                row_begin=begin, row_end=end, allocation_bytes=size, allocation_granularity=4096,
                logical_bytes=(end - begin) * width, allocation_id=aid, sha256='sha256:' + 'a' * 64,
                handle=base64.b64encode(aid.encode()).decode())
        docs.append(dict(schema_version=2, shard=index, shards=p, generation=group['generation'],
            model_revision=group['model_revision'], hash_abi='sha256:' + 'b' * 64, host_id=f'host-{index}',
            owner_id=f'host-{index}', owner_incarnation=str(UUID(int=200 + index)), tensors=tensors))
    return group, docs


def samples(_, manifest):
    return {t.name: dict(row_ids=sorted({0, t.shape[0] // 2, t.shape[0] - 1} |
                                      {r for part in t.shards for r in (part.row_begin, part.row_end - 1)}),
                         sha256='sha256:' + 'c' * 64) for t in manifest.tensors}


def package(group, docs):
    return s.assemble_package(group, docs, None, samples=samples)


class GroupTest(unittest.TestCase):
    def test_supported_static_shapes(self):
        for n, p, ranks in [(1, 2, 2), (2, 3, 2), (3, 5, 2), (5, 5, 1)]:
            group, docs = fixture(n, p, ranks)
            self.assertEqual(len(e.validate_group(group)), 64)
            s.verify_package(group, package(group, docs))

    def test_invalid_memberships_and_host_assignments(self):
        mutations = [lambda g: g['members'][1].update(instance_id='i0'),
                     lambda g: g['members'][1].update(shards=[]),
                     lambda g: g['members'][0]['shards'][0].update(publisher_rank=True),
                     lambda g: g['members'][0]['shards'][0].update(index=1),
                     lambda g: g['members'][0]['shards'][0].update(host_id='host-1'),
                     lambda g: g['members'][0].update(world_size=0)]
        for mutate in mutations:
            group, _ = fixture(); mutate(group)
            with self.assertRaises((ValueError, e.ExchangeError)): e.validate_group(group)

    def test_cohort_waits_for_all_instances_and_rejects_changes(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: package(group, d), 2)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(cohort.submit, 'i0', cohort.group_digest, [docs[0], docs[2]])
            time.sleep(0.03)
            self.assertFalse(first.done())
            second = pool.submit(cohort.submit, 'i1', cohort.group_digest, [docs[1]])
            result = first.result(); self.assertEqual(result, second.result())
        self.assertEqual(cohort.submit('i0', cohort.group_digest, [docs[0], docs[2]]), result)
        changed = deepcopy(docs[0]); changed['owner_incarnation'] = str(UUID(int=999))
        with self.assertRaisesRegex(e.ExchangeError, 'REPUBLICATION'):
            cohort.submit('i0', cohort.group_digest, [changed, docs[2]])
        with self.assertRaisesRegex(e.ExchangeError, 'GROUP_MISMATCH'):
            cohort.submit('i1', 'wrong', [docs[1]])
        with self.assertRaisesRegex(e.ExchangeError, 'UNKNOWN'):
            cohort.submit('outsider', cohort.group_digest, [])
        cohort.ack('i0', e.digest(result)); cohort.ack('i1', e.digest(result))
        cohort.wait_acknowledged()

    def test_missing_member_and_failed_builder_do_not_hang(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 0.03)
        with self.assertRaisesRegex(e.ExchangeError, 'TIMEOUT'):
            cohort.submit('i0', cohort.group_digest, [docs[0], docs[2]])
        def fail(_): raise ValueError('private handle details')
        cohort = e.Cohort(group, fail, 2)
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(cohort.submit, 'i0', cohort.group_digest, [docs[0], docs[2]])
            with self.assertRaisesRegex(e.ExchangeError, '^ASSEMBLY_FAILED$'):
                cohort.submit('i1', cohort.group_digest, [docs[1]])
            with self.assertRaisesRegex(e.ExchangeError, '^ASSEMBLY_FAILED$'): pending.result()

    def test_slow_builder_and_missing_ack_are_bounded(self):
        group, docs = fixture(1, 2, 2)
        release = threading.Event()
        def slow(_):
            release.wait(2)
            return {}
        cohort = e.Cohort(group, slow, 0.03)
        try:
            with self.assertRaisesRegex(e.ExchangeError, 'TIMEOUT'):
                cohort.submit('i0', cohort.group_digest, docs)
        finally:
            release.set()
        cohort = e.Cohort(group, lambda d: package(group, d), 0.05)
        payload = cohort.submit('i0', cohort.group_digest, docs)
        with self.assertRaisesRegex(e.ExchangeError, 'ACK_MISMATCH'):
            cohort.ack('i0', 'wrong-payload')
        with self.assertRaisesRegex(e.ExchangeError, 'TIMEOUT'):
            cohort.wait_acknowledged()

    def test_bounded_frames_and_invalid_generation(self):
        with self.assertRaisesRegex(e.ExchangeError, 'MESSAGE_TOO_LARGE'):
            e.decode(b'x' * (e.MAX_MESSAGE + 1))
        with self.assertRaisesRegex(e.ExchangeError, 'MESSAGE_TOO_LARGE'):
            e.encode({'value': 'x' * e.MAX_MESSAGE})
        with self.assertRaises(ValueError):
            e.decode(b'{"instance_id":"a","instance_id":"b"}')
        for value in (None, 1, True, 'not-a-uuid'):
            group, _ = fixture(); group['generation'] = value
            with self.assertRaisesRegex(e.ExchangeError, 'INVALID_GENERATION'):
                e.validate_group(group)

    def test_private_local_cache_and_full_identity_checks(self):
        group, docs = fixture(); payload = package(group, docs)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(s.materialize_package(group, payload, directory))
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o600)
            self.assertEqual(bundle.parent.stat().st_mode & 0o777, 0o700)
            for name in ('manifest.json', 'capabilities.json'):
                self.assertEqual((bundle.parent / name).stat().st_mode & 0o777, 0o600)
        changed = deepcopy(payload); changed['capabilities'].pop(next(iter(changed['capabilities'])))
        with self.assertRaisesRegex(e.ExchangeError, 'CAPABILITIES'): s.verify_package(group, changed)
        changed_group = deepcopy(group); changed_group['generation'] = str(UUID(int=999))
        with self.assertRaisesRegex(e.ExchangeError, 'IDENTITY'): s.verify_package(changed_group, payload)


class MemoryStore:
    """Shared test Store with real blocking semantics and bounded waits."""
    def __init__(self): self.data = {}; self.condition = threading.Condition()
    def set(self, key, value):
        with self.condition:
            self.data[key] = value; self.condition.notify_all()
    def get(self, key):
        with self.condition: return self.data[key]
    def wait(self, keys, timeout):
        deadline = time.monotonic() + timeout.total_seconds()
        with self.condition:
            while not all(k in self.data for k in keys):
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise RuntimeError('store timeout')
                self.condition.wait(remaining)
    def delete_key(self, key):
        with self.condition: return self.data.pop(key, None) is not None


class IntraInstanceTest(unittest.TestCase):
    def test_only_rank_zero_exchanges_and_all_ranks_receive_identical_backing(self):
        group, docs = fixture(1, 2, 4)
        store = MemoryStore(); calls = []
        def exchange(config, local, deadline):
            calls.append(local)
            return package(group, local)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = {}
            for doc in docs:
                path = root / str(doc['shard']); path.write_bytes(e.encode(doc)); path.chmod(0o600)
                local[str(doc['shard'])] = str(path)
            config = dict(group=group, instance_id='i0', local_descriptors=local, timeout_seconds=3, cache_directory=directory)
            path = root / 'config.json'; path.write_bytes(e.encode(config)); path.chmod(0o600)
            envs = [{s.CONFIG_ENV: str(path)} for _ in range(4)]
            with ThreadPoolExecutor(4) as pool:
                futures = [pool.submit(s.prepare_static_engram, store, rank=i, world_size=4, environ=envs[i], leader_exchange=exchange) for i in range(4)]
                bundles = [f.result() for f in futures]
            self.assertEqual(len(calls), 1)
            caps = [Path(p).parent.joinpath('capabilities.json').read_bytes() for p in bundles]
            self.assertTrue(all(c == caps[0] for c in caps))
            self.assertEqual(store.data, {})
            self.assertTrue(all(env['ENGRAM_SHARD_COUNT'] == '2' for env in envs))

    def test_bad_p_fails_before_any_store_side_effect(self):
        group, _ = fixture(1, 2, 2)
        store = MemoryStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_bytes(e.encode(dict(group=group, instance_id='i0')))
            path.chmod(0o600)
            with self.assertRaisesRegex(e.ExchangeError, 'STORAGE_HOST_COUNT_MISMATCH'):
                s.prepare_static_engram(store, rank=0, world_size=2,
                    environ={s.CONFIG_ENV: str(path), 'ENGRAM_SHARD_COUNT': '3'})
        self.assertEqual(store.data, {})

    def test_inconsistent_worker_group_never_publishes_package(self):
        group, _ = fixture(1, 2, 2)
        store = MemoryStore()
        prefix = 'engram/' + group['generation'] + '/i0/'
        store.set(prefix + 'config/1', b'wrong-group')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_bytes(e.encode(dict(group=group, instance_id='i0', timeout_seconds=0.1)))
            path.chmod(0o600)
            with self.assertRaisesRegex(e.ExchangeError, 'INSTANCE_GROUP_CONFIG_MISMATCH'):
                s.prepare_static_engram(store, rank=0, world_size=2, environ={s.CONFIG_ENV: str(path)})
        self.assertNotIn(prefix + 'package', store.data)

    def test_unconfigured_models_do_nothing(self):
        self.assertIsNone(s.prepare_static_engram(None, rank=0, world_size=8, environ={}))


class MutualTLSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.engram_sidecar.framework.make_exchange_test_certs import create
        cls.temp = tempfile.TemporaryDirectory()
        cls.tls = create(Path(cls.temp.name) / 'pki', ['i0', 'i1', 'i2'])
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_two_remote_instances_reuse_exactly_two_connections(self):
        group, docs = fixture(3, 3, 1)
        cohort = e.Cohort(group, lambda d: package(group, d), 10)
        server = e.ExchangeServer(('127.0.0.1', 0), cohort, e.tls_context(self.tls['i0'], True), self.tls['i0']['instance_cert_sha256'])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True); thread.start()
        try:
            with ThreadPoolExecutor(3) as pool:
                futures = [pool.submit(e.remote_exchange, server.server_address, e.tls_context(self.tls[f'i{i}']),
                    f'i{i}', cohort.group_digest, [docs[i]], time.monotonic() + 8,
                    lambda p: s.verify_package(group, p), self.tls['i0']['instance_cert_sha256']['i0']) for i in (1, 2)]
                payload = cohort.submit('i0', cohort.group_digest, [docs[0]])
                cohort.ack('i0', e.digest(payload)); cohort.wait_acknowledged()
                results = [f.result() for f in futures]
            self.assertEqual(results[0], results[1])
            self.assertEqual(server.connections, 2)  # exchange + ACK on each connection
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_valid_ca_but_wrong_coordinator_certificate_rejected(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 3)
        server = e.ExchangeServer(('127.0.0.1', 0), cohort, e.tls_context(self.tls['i0'], True), self.tls['i0']['instance_cert_sha256'])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True); thread.start()
        try:
            with self.assertRaisesRegex(e.ExchangeError, 'COORDINATOR_CERTIFICATE_MISMATCH'):
                e.remote_exchange(server.server_address, e.tls_context(self.tls['i1']), 'i1', cohort.group_digest,
                    [docs[1]], time.monotonic() + 2, lambda p: None, self.tls['i0']['instance_cert_sha256']['i2'])
            self.assertEqual(cohort.documents, {})
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_certificate_cannot_claim_another_instance(self):
        group, docs = fixture()
        cohort = e.Cohort(group, lambda d: {}, 3)
        server = e.ExchangeServer(('127.0.0.1', 0), cohort, e.tls_context(self.tls['i0'], True), self.tls['i0']['instance_cert_sha256'])
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True); thread.start()
        try:
            with self.assertRaisesRegex(e.ExchangeError, 'REMOTE_EXCHANGE_FAILED'):
                e.remote_exchange(server.server_address, e.tls_context(self.tls['i2']), 'i1', cohort.group_digest,
                                  [docs[1]], time.monotonic() + 2, lambda p: None)
            self.assertEqual(cohort.documents, {})
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main(verbosity=2)
