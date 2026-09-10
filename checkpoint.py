#!/usr/bin/env python3
"""Crawl checkpoints in Qdrant.

The 10M build embeds on the crawl runners and parks nothing, which removes
both stores the checkpoint used to live in:

    coupled path      -> HuggingFace, one commit per save against a 128/hour
                         account-wide limit. Twenty workers saving every few
                         thousand images sits uncomfortably close to it, and
                         a rate-limited checkpoint costs a worker its range.
    crawl-only path   -> the S3 bucket, which is the thing we are deleting.

Qdrant is already open every batch, is already paid for, has no commit
ceiling, and is the one component the build cannot run without -- so a
checkpoint that lives there cannot be unavailable while the crawl is
otherwise fine.

State is tiny (a queue of cursors and a few counters), so it goes in the
payload of a single point per worker, in its own small collection.
"""
import os
import uuid

from qdrant_client import QdrantClient, models

COLLECTION = os.getenv('CRAWL_STATE_COLLECTION', 'crawl-state')
NAMESPACE = uuid.UUID('6f0a4c1e-9b3d-4a6f-8c2e-1d5b7a9e3f04')

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=os.environ['QDRANT_URL'],
                               api_key=os.environ.get('QDRANT_API_KEY'),
                               timeout=120)
    return _client


def ensure():
    """Create the state collection if it is missing. Idempotent."""
    qc = client()
    if qc.collection_exists(COLLECTION):
        return
    # Nothing is ever searched here; the vector exists only because a Qdrant
    # collection must have one. Size 1 and DOT keeps it as close to free as
    # the API allows -- COSINE would reject the zero vector.
    qc.create_collection(COLLECTION, vectors_config=models.VectorParams(
        size=1, distance=models.Distance.DOT))


def point_id(key):
    return str(uuid.uuid5(NAMESPACE, key))


def load(key):
    """The saved state for this worker, or None if it has never saved."""
    try:
        found = client().retrieve(COLLECTION, ids=[point_id(key)],
                                  with_payload=True, with_vectors=False)
    except Exception as exc:
        # Absent collection means a fresh build. Anything else is a real
        # failure and must not be mistaken for "nothing crawled yet": that
        # restarts the worker's range and re-crawls what it already indexed.
        if 'not found' in str(exc).lower() or 'doesn\'t exist' in str(exc).lower():
            return None
        raise
    return found[0].payload.get('state') if found else None


def save(key, state):
    client().upsert(COLLECTION, wait=True, points=[models.PointStruct(
        id=point_id(key), vector=[1.0], payload={'key': key, 'state': state})])
