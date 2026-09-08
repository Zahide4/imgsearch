#!/usr/bin/env python3
"""
push_qdrant.py -- ship vectors + payload to Qdrant Cloud.

THIS IS THE PRODUCTION VECTOR STORE, on its free tier. Same engine, same
HNSW index, same quantization you would run on a Hetzner box -- just hosted.

    Qdrant Cloud free cluster: 1 GB, no credit card, no expiry.
    768d int8-quantized: ~300k vectors fits comfortably (0.23 GB).

Scalar (int8) quantization is switched ON deliberately. It is 4x smaller than
fp32 with ~99% recall, and `always_ram=True` keeps the quantized vectors in
memory while originals sit on disk for rescoring -- exactly the pattern that
lets 100M vectors run on one box at full scale.

Setup (free, no card):
    1. cloud.qdrant.io -> create a free 1GB cluster
    2. copy the cluster URL and an API key
    3. export QDRANT_URL=https://xxx.cloud.qdrant.io:6333
       export QDRANT_API_KEY=xxx

Usage:
    python push_qdrant.py create      # make the collection
    python push_qdrant.py push        # upload everything embedded locally
    python push_qdrant.py info
"""

import os
import sqlite3
import sys
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "index.db"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "images")
DIM = 768
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def connect():
    url = os.environ.get("QDRANT_URL")
    if not url:
        sys.exit("Set QDRANT_URL and QDRANT_API_KEY (see docstring).")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"),
                        timeout=120, prefer_grpc=False)


def point_id(image_id: str) -> str:
    """Qdrant needs uint64 or UUID ids; ours are strings like 'ov:abc'.
    uuid5 is deterministic, so re-pushing updates rather than duplicates."""
    return str(uuid.uuid5(NAMESPACE, image_id))


def create(recreate=False):
    c = connect()
    if recreate:
        try:
            c.delete_collection(COLLECTION)
        except Exception:
            pass
    if c.collection_exists(COLLECTION):
        print(f"collection '{COLLECTION}' already exists")
    else:
        c.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=DIM, distance=models.Distance.COSINE,
                on_disk=True,                       # originals on disk
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,                # quantized stay in RAM
                ),
            ),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
        print(f"created '{COLLECTION}' (768d cosine, int8 quantized)")

    # payload indexes -> license filtering stays fast as the collection grows
    for field, schema in (("license_class", models.PayloadSchemaType.KEYWORD),
                          ("width", models.PayloadSchemaType.INTEGER)):
        try:
            c.create_payload_index(COLLECTION, field, field_schema=schema)
        except Exception:
            pass
    print("payload indexes ready")


def license_class(lic: str) -> str:
    """Bucket licences into what an editor actually needs to know."""
    s = (lic or "").lower()
    if "cc0" in s or "public domain" in s or "pdm" in s or "pd" == s.strip():
        return "public_domain"
    if "sa" in s.replace("-", " ").split() or "share" in s:
        return "share_alike"      # viral onto derivatives -- warn the user
    return "attribution"


def push(batch_size=256):
    import storage
    c = connect()
    if not c.collection_exists(COLLECTION):
        create()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT i.*, v.vec FROM images i
        JOIN vectors v ON v.id = i.id
        WHERE i.state = 'embedded'
    """).fetchall()
    con.close()
    if not rows:
        sys.exit("nothing embedded yet -- run: python ingest.py --embed-only")

    print(f"pushing {len(rows):,} points to {COLLECTION}...")
    sent = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        points = []
        for r in chunk:
            vec = np.frombuffer(r["vec"], dtype=np.float32)
            if vec.shape[0] != DIM:
                continue
            points.append(models.PointStruct(
                id=point_id(r["id"]),
                vector=vec.tolist(),
                payload={
                    "image_id": r["id"],
                    "title": r["title"] or "",
                    "creator": (r["creator"] or "")[:180],
                    "license": r["license"] or "",
                    "license_class": license_class(r["license"]),
                    "license_url": r["license_url"] or "",
                    "source_url": r["source_url"] or "",
                    "width": r["width"] or 0,
                    "height": r["height"] or 0,
                    "tags": r["tags"] or "",
                    "cdn": storage.cdn_url(r["id"], r["thumb_url"] or ""),
                },
            ))
        if points:
            c.upsert(collection_name=COLLECTION, points=points, wait=False)
            sent += len(points)
        print(f"\r  {sent}/{len(rows)}", end="", flush=True)
    print()
    info()


def info():
    c = connect()
    i = c.get_collection(COLLECTION)
    print(f"collection '{COLLECTION}': {i.points_count:,} points, status={i.status}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    {"create": lambda: create("--recreate" in sys.argv),
     "push": push, "info": info}[cmd]()
