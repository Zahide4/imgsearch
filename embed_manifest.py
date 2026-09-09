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
import argparse, io, json, os, tarfile, time
from collections import OrderedDict, deque
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
    ap.add_argument('--loaders', type=int, default=8,
                    help='shard downloads in flight. A shard is one tar of ~500 '
                         'images, so this is not the knob it was when every '
                         'image cost its own request -- 8 is already 8 x 11 MB '
                         'in flight. Only rows from pre-shard builds still fetch '
                         'per object, and those use this as a thread count.')
    ap.add_argument('--lookahead', type=int, default=2,
                    help='shards decoded ahead of the GPU. Each costs ~220 MB '
                         'of decoded RGB, so this trades RAM for keeping the '
                         'GPU fed.')
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

    def decode(row, data):
        try:
            return row, Image.open(io.BytesIO(data)).convert('RGB')
        except Exception as exc:
            print(f"decode skipped {row.get('image_id')}: {type(exc).__name__}", flush=True)
            return row, None

    def fetch_shard(key, rows):
        """One GET for ~500 images. This is the whole point of the rewrite.

        Reading a derivative per image cost one Class B call per image; a full
        10M pass was 10M of them, against a free-tier allowance of 2,500 a day.
        """
        body = storage.client().get_object(Bucket=storage.BUCKET, Key=key)['Body'].read()
        members = {}
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            for info in tar:
                if info.name.endswith('.webp'):
                    members[info.name] = tar.extractfile(info).read()
        out = []
        for row in rows:
            data = members.get(point_id(row['image_id']) + '.webp')
            if data is None:
                print(f"missing from {key}: {row.get('image_id')}", flush=True)
                continue
            _, image = decode(row, data)
            if image is not None:
                out.append((row, image))
        return out

    def fetch_objects(_, rows):
        """Pre-shard builds: one request per image, the way it used to be."""
        def one(row):
            try:
                obj = storage.client().get_object(Bucket=storage.BUCKET, Key=row['cdn'])
                return decode(row, obj['Body'].read())
            except Exception as exc:
                print(f"load skipped {row.get('image_id')}: {type(exc).__name__}", flush=True)
                return row, None
        return [(r, im) for r, im in pool.map(one, rows) if im is not None]

    # Group by shard, preserving manifest order so a resumed run reads the
    # bucket roughly sequentially rather than seeking all over it.
    groups = OrderedDict()
    for row in todo:
        groups.setdefault(row.get('shard'), []).append(row)
    work = list(groups.items())
    sharded = sum(len(v) for k, v in work if k)
    print(f'  {len(work):,} shards, {sharded:,}/{len(todo):,} rows tarred '
          f'({len(todo) - sharded:,} pre-shard, fetched per object)\n', flush=True)

    # Bounded look-ahead: fetch the next shards while the GPU works, without
    # buffering the whole corpus into RAM.
    fetchers = ThreadPoolExecutor(a.loaders)
    queue, nxt = deque(), 0

    def submit():
        nonlocal nxt
        if nxt < len(work):
            key, rows = work[nxt]
            queue.append(fetchers.submit(fetch_shard if key else fetch_objects, key, rows))
            nxt += 1

    for _ in range(max(a.lookahead, 1)):
        submit()

    start, embedded = time.monotonic(), 0
    while queue:
        loaded = queue.popleft().result()
        submit()
        for i in range(0, len(loaded), batch):
            part = loaded[i:i + batch]
            if not part:
                continue
            tensors = torch.stack([preprocess(im) for _, im in part]).to(device)
            if device == 'cuda':
                tensors = tensors.half()
            with torch.inference_mode():
                vectors = F.normalize(model.encode_image(tensors), dim=-1).float().cpu().numpy()
            if not np.isfinite(vectors).all() or vectors.shape[1] != 768:
                raise RuntimeError('Invalid image embeddings; refusing upload')

            sparse_vectors = sparse.documents(search_text(r) for r, _ in part)
            qc.upsert(COLLECTION, wait=True, points=[
                models.PointStruct(id=point_id(row['image_id']),
                                   vector={'image': vec.tolist(), 'bm25': bm25},
                                   payload=row)
                for (row, _), vec, bm25 in zip(part, vectors, sparse_vectors)])

            embedded += len(part)
            rate = embedded / max(time.monotonic() - start, 1e-9)
            print(f'\r  {embedded:,}/{len(todo):,}  {rate:6.0f} img/s  '
                  f'eta {(len(todo)-embedded)/max(rate,1e-9)/60:5.1f} min', end='', flush=True)

    elapsed = time.monotonic() - start
    print(f'\n\nembedded {embedded:,} in {elapsed/60:.1f} min '
          f'({embedded/max(elapsed,1e-9):.0f} img/s)')
    print(f'10M at this rate: {10_000_000/max(embedded/max(elapsed,1e-9),1e-9)/3600:.1f} h')


if __name__ == '__main__':
    main()
