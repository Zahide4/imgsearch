#!/usr/bin/env python3
"""FrameDrop crawl work queue.

The 10M build handed every worker a static share (LPT bin packing over
estimated shard sizes). The estimates were rough and nobody could take over
work from a finished or dead colleague, so the last few workers chewed a
multi-day tail alone. This service is the lock server the old README said
the design needed: a tiny claim queue with leases, so work is pulled a small
unit at a time and every worker stays busy until the queue is empty.

Lifecycle of a job:

    pending -> claimed (lease) -> done
                     |  ^
                     |  +-- lease expiry or /release -> pending
                     +-----> failed (after max_attempts)

- `/claim` hands out the oldest pending job and starts a lease. A worker
  renews by calling `/progress`, which also banks its resume cursor and the
  images it has indexed, so a job killed mid-shard resumes from its last
  saved page instead of starting over.
- A worker that dies simply stops renewing; the job returns to pending when
  the lease expires and the next claimant continues from the saved cursor.
- Re-seeding is idempotent: every unit gets a stable sha256 and duplicates
  are ignored, so a restarted build can safely re-post its whole job list.

State lives in SQLite beside this file (queue.db). One file, one process,
one lock; the box is the only writer.

Run: python3 queue_service.py    (binds 127.0.0.1:8092; Caddy fronts it)
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = '127.0.0.1'
PORT = 8092
ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, 'queue.db')

DEFAULT_LEASE = 20 * 60        # seconds
MAX_ATTEMPTS = int(os.getenv('QUEUE_MAX_ATTEMPTS', '8'))
MAX_BODY = 4 * 1024 * 1024


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA busy_timeout=15000')
    return connection


def init_db() -> None:
    with connect() as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                build       TEXT NOT NULL,
                unit_hash   TEXT NOT NULL UNIQUE,
                unit        TEXT NOT NULL,
                cursor      TEXT,
                state       TEXT NOT NULL DEFAULT 'pending',
                worker      TEXT,
                lease_until REAL NOT NULL DEFAULT 0,
                attempts    INTEGER NOT NULL DEFAULT 0,
                produced    INTEGER NOT NULL DEFAULT 0,
                reason      TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(build, state, id);
            CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(state, lease_until);
            """
        )
    for path in (DB_PATH, DB_PATH + '-wal', DB_PATH + '-shm'):
        try:
            os.chmod(path, 0o600)
        except FileNotFoundError:
            pass


def unit_hash(build: str, unit: dict) -> str:
    canonical = json.dumps(unit, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256((build + '|' + canonical).encode()).hexdigest()


def expire(database: sqlite3.Connection, now: float, build: str | None = None) -> None:
    """Return dead leases to the pool; retire jobs that keep dying."""
    where = 'state = ? AND lease_until < ?'
    args: list = ['claimed', now]
    if build:
        where += ' AND build = ?'
        args.append(build)
    database.execute(
        f'UPDATE jobs SET state = ?, reason = ? , updated_at = ? '
        f'WHERE {where} AND attempts >= ?',
        ['failed', 'lease expired too many times', now, *args, MAX_ATTEMPTS],
    )
    database.execute(
        f'UPDATE jobs SET state = ?, worker = NULL, lease_until = 0, updated_at = ? '
        f'WHERE {where} AND attempts < ?',
        ['pending', now, *args, MAX_ATTEMPTS],
    )


def claim(build: str, worker: str, lease: int) -> dict | None:
    now = time.time()
    database = connect()
    try:
        database.execute('BEGIN IMMEDIATE')
        expire(database, now, build)
        row = database.execute(
            'SELECT id, unit, cursor, attempts FROM jobs '
            'WHERE build = ? AND state = ? ORDER BY id LIMIT 1',
            (build, 'pending'),
        ).fetchone()
        if row is None:
            database.execute('COMMIT')
            return None
        database.execute(
            'UPDATE jobs SET state = ?, worker = ?, lease_until = ?, '
            'attempts = attempts + 1, updated_at = ? WHERE id = ?',
            ('claimed', worker, now + lease, now, row['id']),
        )
        database.execute('COMMIT')
        return {'id': row['id'], 'unit': json.loads(row['unit']),
                'cursor': json.loads(row['cursor']) if row['cursor'] else None,
                'attempts': row['attempts'] + 1}
    except BaseException:
        database.execute('ROLLBACK')
        raise
    finally:
        database.close()


def owned_update(sql: str, args: list) -> int:
    database = connect()
    try:
        cursor = database.execute(sql, args)
        changed = cursor.rowcount
        database.commit()
        return changed
    finally:
        database.close()


def progress(job_id: int, worker: str, lease: int, cursor: dict | None,
             produced: int) -> bool:
    now = time.time()
    return owned_update(
        'UPDATE jobs SET cursor = COALESCE(?, cursor), produced = produced + ?, '
        'lease_until = ?, updated_at = ? '
        "WHERE id = ? AND worker = ? AND state = 'claimed'",
        [json.dumps(cursor) if cursor is not None else None, max(0, produced),
         now + lease, now, job_id, worker],
    ) > 0


def finish(job_id: int, worker: str, state: str, reason: str | None = None,
           requeue: bool = False) -> bool:
    now = time.time()
    if requeue:
        return owned_update(
            'UPDATE jobs SET state = ?, worker = NULL, lease_until = 0, '
            'reason = ?, updated_at = ? '
            "WHERE id = ? AND worker = ? AND state = 'claimed'",
            ['pending', reason, now, job_id, worker],
        ) > 0
    return owned_update(
        'UPDATE jobs SET state = ?, worker = NULL, lease_until = 0, reason = ?, '
        'updated_at = ? '
        "WHERE id = ? AND worker = ? AND state = 'claimed'",
        [state, reason, now, job_id, worker],
    ) > 0


def seed(build: str, units: list[dict]) -> int:
    now = time.time()
    inserted = 0
    database = connect()
    try:
        database.execute('BEGIN IMMEDIATE')
        for unit in units:
            cursor = database.execute(
                'INSERT OR IGNORE INTO jobs '
                '(build, unit_hash, unit, state, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (build, unit_hash(build, unit), json.dumps(unit),
                 'pending', now, now),
            )
            inserted += cursor.rowcount
        database.execute('COMMIT')
        return inserted
    except BaseException:
        database.execute('ROLLBACK')
        raise
    finally:
        database.close()


def stats(build: str) -> dict:
    now = time.time()
    database = connect()
    try:
        expire(database, now, build)
        rows = database.execute(
            'SELECT state, COUNT(*) AS n, COALESCE(SUM(produced), 0) AS produced '
            'FROM jobs WHERE build = ? GROUP BY state',
            (build,),
        ).fetchall()
        out = {row['state']: row['n'] for row in rows}
        out['produced'] = sum(row['produced'] for row in rows)
        out['pending'] = out.get('pending', 0)
        out['claimed'] = out.get('claimed', 0)
        out['done'] = out.get('done', 0)
        out['failed'] = out.get('failed', 0)
        return out
    finally:
        database.close()


def requeue_failed(build: str) -> int:
    return owned_update(
        "UPDATE jobs SET state = 'pending', attempts = 0, reason = NULL, "
        'updated_at = ? WHERE build = ? AND state = ?',
        [time.time(), build, 'failed'],
    )


class Handler(BaseHTTPRequestHandler):
    server_version = 'FrameDropQueue/1'

    def log_message(self, fmt, *args):
        sys.stdout.write('[%s] %s\n' % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def log_request(self, code='-', size='-'):
        path = urllib.parse.urlsplit(self.path).path
        self.log_message('%s %s %s', self.command, path, code)

    def json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict | None:
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def route(self) -> str:
        return urllib.parse.urlsplit(self.path).path

    def query(self) -> dict:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))

    def do_GET(self):
        path = self.route()
        if path == '/healthz':
            self.json_response(200, {'ok': True})
            return
        if path == '/stats':
            build = self.query().get('build', '')
            if not build:
                self.json_response(400, {'error': 'build_required'})
                return
            self.json_response(200, stats(build))
            return
        self.json_response(404, {'error': 'not_found'})

    def do_POST(self):
        path = self.route()
        body = self.read_json()
        if body is None:
            self.json_response(400, {'error': 'bad_request'})
            return
        if path == '/claim':
            build, worker = body.get('build', ''), body.get('worker', '')
            if not build or not worker:
                self.json_response(400, {'error': 'build_and_worker_required'})
                return
            lease = int(body.get('lease_seconds') or DEFAULT_LEASE)
            job = claim(build, worker, lease)
            if job is None:
                self.json_response(204, {})
            else:
                self.json_response(200, job)
            return
        if path == '/seed':
            build, units = body.get('build', ''), body.get('units') or []
            if not build or not isinstance(units, list):
                self.json_response(400, {'error': 'bad_seed'})
                return
            self.json_response(200, {'inserted': seed(build, units)})
            return
        if path == '/requeue-failed':
            build = body.get('build', '')
            if not build:
                self.json_response(400, {'error': 'build_required'})
                return
            self.json_response(200, {'requeued': requeue_failed(build)})
            return
        # The remaining endpoints act on one job, always as its owner.
        try:
            job_id = int(body.get('id'))
        except (TypeError, ValueError):
            self.json_response(400, {'error': 'bad_id'})
            return
        worker = body.get('worker', '')
        if not worker:
            self.json_response(400, {'error': 'worker_required'})
            return
        if path == '/progress':
            lease = int(body.get('lease_seconds') or DEFAULT_LEASE)
            ok = progress(job_id, worker, lease, body.get('cursor'),
                          int(body.get('produced') or 0))
        elif path == '/complete':
            ok = finish(job_id, worker, 'done', body.get('reason'))
        elif path == '/release':
            ok = finish(job_id, worker, 'pending', body.get('reason'), requeue=True)
        elif path == '/fail':
            ok = finish(job_id, worker, 'failed', body.get('reason'))
        else:
            self.json_response(404, {'error': 'not_found'})
            return
        if not ok:
            # The lease expired and someone else owns the job now. The worker
            # should drop it: continuing would only duplicate indexed images
            # (safe, but wasted), and its next /claim returns fresh work.
            self.json_response(409, {'error': 'not_owner'})
            return
        self.json_response(200, {'ok': True})


def main() -> None:
    os.umask(0o077)
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f'crawl queue on {HOST}:{PORT} (max attempts {MAX_ATTEMPTS})')
    server.serve_forever()


if __name__ == '__main__':
    main()
