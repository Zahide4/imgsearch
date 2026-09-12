#!/usr/bin/env python3
"""Tests for the crawl work queue. Stdlib only.

Run:  cd crawl_queue && python3 -m unittest test_queue_service -v
"""

import http.client
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'queue_service', os.path.join(ROOT, 'queue_service.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)

BUILD = 'celeb-test'
UNIT = {'name': 'Taylor Swift', 'source': 'commons', 'lic': 'CC-BY-4.0'}


class QueueServerTestCase(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        handle.close()
        os.unlink(handle.name)          # sqlite creates it fresh, 0600
        queue.DB_PATH = handle.name
        queue.init_db()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), queue.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        for suffix in ('', '-wal', '-shm'):
            try:
                os.unlink(queue.DB_PATH + suffix)
            except FileNotFoundError:
                pass

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        payload = None if body is None else json.dumps(body)
        connection.request(method, path, body=payload,
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response.status, (json.loads(raw) if raw else {})

    def seed(self, units):
        return self.request('POST', '/seed', {'build': BUILD, 'units': units})

    def claim(self, worker, lease=20):
        return self.request('POST', '/claim',
                            {'build': BUILD, 'worker': worker, 'lease_seconds': lease})


class TestSeedAndClaim(QueueServerTestCase):
    def test_seed_is_idempotent(self):
        status, body = self.seed([UNIT, {'name': 'Beyoncé', 'source': 'openverse'}])
        self.assertEqual((status, body['inserted']), (200, 2))
        status, body = self.seed([UNIT, {'name': 'Beyoncé', 'source': 'openverse'}])
        self.assertEqual((status, body['inserted']), (200, 0))

    def test_claim_hands_out_each_job_once(self):
        self.seed([UNIT, {'name': 'Beyoncé', 'source': 'openverse'}])
        status, first = self.claim('w0')
        self.assertEqual(status, 200)
        self.assertEqual(first['unit'], UNIT)
        status, second = self.claim('w0')
        self.assertEqual(status, 200)
        self.assertEqual(second['unit']['name'], 'Beyoncé')
        status, _ = self.claim('w0')
        self.assertEqual(status, 204)

    def test_builds_are_isolated(self):
        self.seed([UNIT])
        status, _ = self.request('POST', '/claim',
                                 {'build': 'other-build', 'worker': 'w0'})
        self.assertEqual(status, 204)
        status, _ = self.claim('w0')
        self.assertEqual(status, 200)

    def test_empty_claim_is_204_not_an_error(self):
        status, _ = self.claim('w0')
        self.assertEqual(status, 204)


class TestLeases(QueueServerTestCase):
    def _expire_now(self, job_id):
        with queue.connect() as database:
            database.execute('UPDATE jobs SET lease_until = ? WHERE id = ?',
                             (time.time() - 1, job_id))

    def test_expired_lease_returns_job_to_pool(self):
        self.seed([UNIT])
        _, job = self.claim('dead-worker')
        self._expire_now(job['id'])
        status, again = self.claim('fresh-worker')
        self.assertEqual(status, 200)
        self.assertEqual(again['id'], job['id'])
        self.assertEqual(again['attempts'], 2)

    def test_progress_banks_cursor_and_renews_lease(self):
        self.seed([UNIT])
        _, job = self.claim('w0', lease=1)
        status, _ = self.request('POST', '/progress', {
            'id': job['id'], 'worker': 'w0', 'lease_seconds': 600,
            'cursor': {'gsroffset': 50}, 'produced': 7})
        self.assertEqual(status, 200)
        self.request('POST', '/release',
                     {'id': job['id'], 'worker': 'w0'})
        _, again = self.claim('w1')
        self.assertEqual(again['cursor'], {'gsroffset': 50})

    def test_poison_job_retires_after_max_attempts(self):
        original = queue.MAX_ATTEMPTS
        queue.MAX_ATTEMPTS = 2
        try:
            self.seed([UNIT])
            _, job = self.claim('w0')
            self._expire_now(job['id'])
            _, job2 = self.claim('w1')
            self._expire_now(job2['id'])
            status, _ = self.claim('w2')
            self.assertEqual(status, 204)     # no longer offered
            with queue.connect() as database:
                row = database.execute('SELECT state, reason FROM jobs WHERE id = ?',
                                       (job['id'],)).fetchone()
            self.assertEqual(row['state'], 'failed')
            self.assertIn('lease expired', row['reason'])
        finally:
            queue.MAX_ATTEMPTS = original


class TestOwnership(QueueServerTestCase):
    def test_only_the_owner_may_progress_or_finish(self):
        self.seed([UNIT])
        _, job = self.claim('w0')
        status, _ = self.request('POST', '/progress',
                                 {'id': job['id'], 'worker': 'w1', 'produced': 1})
        self.assertEqual(status, 409)
        status, _ = self.request('POST', '/complete',
                                 {'id': job['id'], 'worker': 'w1'})
        self.assertEqual(status, 409)

    def test_lost_lease_rejects_the_old_owner(self):
        self.seed([UNIT])
        _, job = self.claim('w0')
        with queue.connect() as database:
            database.execute('UPDATE jobs SET lease_until = ? WHERE id = ?',
                             (time.time() - 1, job['id']))
        _, again = self.claim('w1')            # takes it over
        self.assertEqual(again['id'], job['id'])
        status, _ = self.request('POST', '/complete',
                                 {'id': job['id'], 'worker': 'w0'})
        self.assertEqual(status, 409)


class TestFinishAndStats(QueueServerTestCase):
    def test_complete_removes_work_and_counts_produced(self):
        self.seed([UNIT])
        _, job = self.claim('w0')
        self.request('POST', '/progress',
                     {'id': job['id'], 'worker': 'w0', 'produced': 12})
        self.request('POST', '/complete', {'id': job['id'], 'worker': 'w0'})
        status, body = self.request('GET', f'/stats?build={BUILD}')
        self.assertEqual(status, 200)
        self.assertEqual(body['done'], 1)
        self.assertEqual(body['pending'], 0)
        self.assertEqual(body['produced'], 12)

    def test_fail_and_requeue(self):
        self.seed([UNIT])
        _, job = self.claim('w0')
        self.request('POST', '/fail',
                     {'id': job['id'], 'worker': 'w0', 'reason': 'bad shard'})
        _, body = self.request('GET', f'/stats?build={BUILD}')
        self.assertEqual(body['failed'], 1)
        self.request('POST', '/requeue-failed', {'build': BUILD})
        _, body = self.request('GET', f'/stats?build={BUILD}')
        self.assertEqual(body['pending'], 1)
        self.assertEqual(body['failed'], 0)

    def test_release_puts_work_back_without_losing_attempts(self):
        self.seed([UNIT])
        _, job = self.claim('w0')
        self.request('POST', '/release', {'id': job['id'], 'worker': 'w0'})
        _, body = self.request('GET', f'/stats?build={BUILD}')
        self.assertEqual(body['pending'], 1)
        _, again = self.claim('w1')
        self.assertEqual(again['attempts'], 2)


class TestQueueClient(QueueServerTestCase):
    """The worker-side client against the live service."""

    def _client(self, worker):
        import sys
        sys.path.insert(0, os.path.dirname(ROOT))
        from crawl_queue.client import QueueClient
        return QueueClient(f'http://127.0.0.1:{self.port}', BUILD, worker, lease=60)

    def test_claim_progress_complete_roundtrip(self):
        self.seed([UNIT])
        client = self._client('client-w0')
        try:
            job = client.claim()
            self.assertEqual(job['unit'], UNIT)
            self.assertTrue(client.progress(job['id'], {'page': 2}, 5))
            self.assertTrue(client.complete(job['id']))
            self.assertIsNone(client.claim())
            stats = client.stats()
            self.assertEqual(stats['done'], 1)
            self.assertEqual(stats['produced'], 5)
        finally:
            client.close()

    def test_release_then_takeover_loses_ownership(self):
        self.seed([UNIT])
        mine = self._client('client-a')
        other = self._client('client-b')
        try:
            job = mine.claim()
            self.assertTrue(mine.release(job['id']))
            taken = other.claim()
            self.assertEqual(taken['id'], job['id'])
            # The old owner no longer may complete it (409 -> False).
            self.assertFalse(mine.complete(job['id']))
        finally:
            mine.close()
            other.close()

    def test_low_attempt_cap_surfaces_via_client(self):
        original = queue.MAX_ATTEMPTS
        queue.MAX_ATTEMPTS = 1
        try:
            self.seed([UNIT])
            client = self._client('client-w0')
            job = client.claim()
            with queue.connect() as database:
                database.execute('UPDATE jobs SET lease_until = ? WHERE id = ?',
                                 (time.time() - 1, job['id']))
            self.assertIsNone(client.claim())      # retired as failed
        finally:
            queue.MAX_ATTEMPTS = original


if __name__ == '__main__':
    unittest.main()
