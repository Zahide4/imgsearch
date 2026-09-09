#!/usr/bin/env python3
"""Phase 2: embed what the crawl left in the bucket.

`cloud_corpus.py --no-embed` crawls, resizes, uploads derivatives and writes a
manifest. This reads that manifest, embeds the images on a GPU, and upserts
complete points to Qdrant. Nothing here touches the internet beyond your own
bucket -- the sources are crawled exactly once.

Why the split: coupled on a CI runner the two phases ran at 1-2 img/s per
worker, because every image waited on SigLIP on a shared CPU. Apart, the crawl
is bound only by how politely the sources will serve, and this runs at GPU
speed over a bucket it can re-read as often as it likes. Changing embedding
model later becomes a pass over your own storage rather than another crawl.

Free GPUs are enough: Kaggle gives 30 GPU-hours a week, Colab a T4.

    pip install open_clip_torch boto3 qdrant-client fastembed
    python embed_manifest.py --build pilot-v1

Resumable. Rows already present in Qdrant are skipped, so an interrupted run
picks up where it stopped.
"""
import argparse, io, json, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models

import sparse
import storage
from cloud_corpus import COLLECTION, point_id, search_text

MODEL_NAME, PRETRAINED = 'ViT-B-16-SigLIP', 'webli'


def manifests(build):
    """Every manifest shard the crawl wrote for this build."""
    paginator = storage.client().get_paginator('list_objects_v2')
    prefix = f'manifest/{build}/'
    for page in paginator.paginate(Bucket=storage.BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.jsonl'):
                yield obj['Key']


def rows_from(key):
    body = storage.client().get_object(Bucket=storage.BUCKET, Key=key)['Body'].read()
    for line in body.decode().splitlines():
        if line.strip():
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--batch', type=int, default=0, help='0 = auto by device')
    ap.add_argument('--loaders', type=int, default=32,
                    help='threads fetching from the bucket; the GPU starves below ~16')
    ap.add_argument('--force', action='store_true',
                    help='re-embed rows already in Qdrant. This is the path a '
                         'model change takes -- a pass over your own bucket '
                         'rather than another crawl of the internet -- and it '
                         'is how throughput gets measured on a corpus that is '
                         'already indexed. Upserts are idempotent by point id.')
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import open_clip

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')
    batch = a.batch or {'cuda': 256, 'mps': 64}.get(device, 32)
    print(f'device={device} batch={batch}', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    if device == 'cuda':
        model = model.half()          # ~2x throughput, no measurable recall cost

    qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'),
                      timeout=120)

    print('reading manifests...', flush=True)
    rows = [r for key in manifests(a.build) for r in rows_from(key)]
    print(f'  {len(rows):,} rows in the manifest', flush=True)

    if a.force:
        todo = rows
        print(f'  {len(todo):,} to re-embed (--force)\n', flush=True)
    else:
        # Resume: whatever is already in Qdrant does not need embedding again.
        todo, ids = [], [point_id(r['image_id']) for r in rows]
        for i in range(0, len(ids), 256):
            chunk = rows[i:i + 256]
            present = {str(p.id) for p in qc.retrieve(COLLECTION, ids=ids[i:i + 256],
                                                      with_payload=False, with_vectors=False)}
            todo.extend(r for r in chunk if point_id(r['image_id']) not in present)
        print(f'  {len(todo):,} still to embed\n', flush=True)
    if not todo:
        return

    pool = ThreadPoolExecutor(a.loaders)

    def load(row):
        """The derivative is already 384px WebP -- no resizing, just decode."""
        try:
            obj = storage.client().get_object(Bucket=storage.BUCKET, Key=row['cdn'])
            return row, Image.open(io.BytesIO(obj['Body'].read())).convert('RGB')
        except Exception as exc:
            print(f"load skipped {row.get('image_id')}: {type(exc).__name__}", flush=True)
            return row, None

    start, embedded = time.monotonic(), 0
    for i in range(0, len(todo), batch):
        loaded = [(r, im) for r, im in pool.map(load, todo[i:i + batch]) if im is not None]
        if not loaded:
            continue
        tensors = torch.stack([preprocess(im) for _, im in loaded]).to(device)
        if device == 'cuda':
            tensors = tensors.half()
        with torch.inference_mode():
            vectors = F.normalize(model.encode_image(tensors), dim=-1).float().cpu().numpy()
        if not np.isfinite(vectors).all() or vectors.shape[1] != 768:
            raise RuntimeError('Invalid image embeddings; refusing upload')

        sparse_vectors = sparse.documents(search_text(r) for r, _ in loaded)
        qc.upsert(COLLECTION, wait=True, points=[
            models.PointStruct(id=point_id(row['image_id']),
                               vector={'image': vec.tolist(), 'bm25': bm25},
                               payload=row)
            for (row, _), vec, bm25 in zip(loaded, vectors, sparse_vectors)])

        embedded += len(loaded)
        rate = embedded / max(time.monotonic() - start, 1e-9)
        print(f'\r  {embedded:,}/{len(todo):,}  {rate:6.0f} img/s  '
              f'eta {(len(todo)-embedded)/max(rate,1e-9)/60:5.1f} min', end='', flush=True)

    elapsed = time.monotonic() - start
    print(f'\n\nembedded {embedded:,} in {elapsed/60:.1f} min '
          f'({embedded/max(elapsed,1e-9):.0f} img/s)')
    print(f'10M at this rate: {10_000_000/max(embedded/max(elapsed,1e-9),1e-9)/3600:.1f} h')


if __name__ == '__main__':
    main()
