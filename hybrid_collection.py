#!/usr/bin/env python3
"""Create and populate the production dense + BM25 Qdrant collection.

This moves vectors and metadata directly between Qdrant collections. It never
downloads image files. Migration is idempotent, so an interrupted run is safe
to repeat.
"""
import os
import sys

from qdrant_client import QdrantClient, models

from cloud_corpus import SPARSE_MODEL, search_text

SOURCE = os.getenv('QDRANT_SOURCE_COLLECTION', 'images')
DESTINATION = os.getenv('QDRANT_COLLECTION', 'images-v2')
DIM = 768


def connect():
    return QdrantClient(
        url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'],
        cloud_inference=True, timeout=180,
    )


def create(client):
    if not client.collection_exists(DESTINATION):
        client.create_collection(
            collection_name=DESTINATION,
            vectors_config={'image': models.VectorParams(
                size=DIM, distance=models.Distance.COSINE, on_disk=True,
            )},
            sparse_vectors_config={'bm25': models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=True),
            )},
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True,
                ),
            ),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            on_disk_payload=True,
        )
        print(f"created {DESTINATION}", flush=True)
    for field, schema in (
        ('license_class', models.PayloadSchemaType.KEYWORD),
        ('build_id', models.PayloadSchemaType.KEYWORD),
        ('worker', models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(DESTINATION, field, field_schema=schema, wait=True)


def migrate(client, batch_size=512):
    create(client)
    offset = None
    copied = 0
    while True:
        records, offset = client.scroll(
            SOURCE, limit=batch_size, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not records:
            break
        existing = client.retrieve(
            DESTINATION, ids=[record.id for record in records],
            with_payload=False, with_vectors=False,
        )
        existing_ids = {str(record.id) for record in existing}
        points = []
        for record in records:
            if str(record.id) in existing_ids:
                continue
            dense = record.vector
            if isinstance(dense, dict):
                dense = dense.get('image')
            if not dense or len(dense) != DIM:
                raise RuntimeError(f'invalid dense vector on point {record.id}')
            payload = record.payload or {}
            points.append(models.PointStruct(
                id=record.id,
                vector={
                    'image': dense,
                    'bm25': models.Document(text=search_text(payload), model=SPARSE_MODEL),
                },
                payload=payload,
            ))
        if points:
            client.upsert(DESTINATION, points=points, wait=True)
        copied += len(records)
        print(f'scanned {copied:,}', flush=True)
        if offset is None:
            break
    source_count = client.count(SOURCE, exact=True).count
    destination_count = client.count(DESTINATION, exact=True).count
    if destination_count < source_count:
        raise RuntimeError(f'incomplete migration: {destination_count:,}/{source_count:,}')
    print(f'migration complete: {destination_count:,} points', flush=True)


def info(client):
    data = client.get_collection(DESTINATION)
    print(f'{DESTINATION}: {data.points_count:,} points, {data.status}')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'info'
    client = connect()
    {'create': lambda: create(client), 'migrate': lambda: migrate(client),
     'info': lambda: info(client)}[command]()
