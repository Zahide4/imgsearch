#!/usr/bin/env python3
"""Client for the crawl work queue.

Workers use this instead of deriving a static share: every unit of work is
claimed, progressed, and completed, so a finished or dead worker's units flow
to whoever is still running.
"""
from __future__ import annotations

import os

import httpx


class QueueClient:
    def __init__(self, url: str, build: str, worker: str, lease: int = 1200,
                 timeout: float = 30.0):
        self.url = url.rstrip('/')
        self.build = build
        self.worker = worker
        self.lease = lease
        headers = {'User-Agent': 'FrameDropCrawl/1'}
        app_key = os.environ.get('FRAMEDROP_APP_KEY')
        if app_key:
            headers['X-FrameDrop-Key'] = app_key
        self.http = httpx.Client(timeout=timeout, headers=headers)

    def _post(self, path: str, payload: dict) -> httpx.Response:
        return self.http.post(f'{self.url}{path}', json=payload)

    def claim(self) -> dict | None:
        r = self._post('/claim', {'build': self.build, 'worker': self.worker,
                                  'lease_seconds': self.lease})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    def progress(self, job_id: int, cursor: dict | None, produced: int) -> bool:
        r = self._post('/progress', {'id': job_id, 'worker': self.worker,
                                     'lease_seconds': self.lease,
                                     'cursor': cursor, 'produced': produced})
        return r.status_code == 200

    def complete(self, job_id: int) -> bool:
        r = self._post('/complete', {'id': job_id, 'worker': self.worker})
        return r.status_code == 200

    def release(self, job_id: int) -> bool:
        r = self._post('/release', {'id': job_id, 'worker': self.worker})
        return r.status_code == 200

    def fail(self, job_id: int, reason: str) -> bool:
        r = self._post('/fail', {'id': job_id, 'worker': self.worker,
                                 'reason': reason})
        return r.status_code == 200

    def stats(self) -> dict:
        r = self.http.get(f'{self.url}/stats', params={'build': self.build})
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.http.close()
