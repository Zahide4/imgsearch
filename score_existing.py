#!/usr/bin/env python3
"""Score a corpus that was indexed before safety scoring existed.

The 435k prototype rows carry no `safety` field, so the query filter passes
them through untouched -- which means the filter does nothing for exactly the
corpus that prompted it. Searching it for "baby being bathed" still returns
`Ejaculation spurt.jpg`.

Scoring needs no images. The vectors are already in Qdrant and safety is a
matmul against cached prompt vectors, so this is a pass over the database
rather than a re-crawl or a GPU job. No bucket reads, no downloads.

    python score_existing.py --limit 2000     # measure the rate first
    python score_existing.py                  # the whole collection
"""
import argparse, os, time
from pathlib import Path

for line in (Path(__file__).resolve().parent / '.env').read_text().splitlines():
    k, _, v = line.strip().partition('=')
    if k and not k.startswith('#'):
        os.environ.setdefault(k, v)

import numpy as np
from qdrant_client import QdrantClient, models

import safety
from cloud_corpus import COLLECTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='0 = whole collection')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    import open_clip
    print('loading SigLIP text tower...', flush=True)
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
    scorer = safety.Scorer(model.eval(), open_clip.get_tokenizer('ViT-B-16-SigLIP'))

    qc = QdrantClient(url=os.environ['QDRANT_URL'],
                      api_key=os.environ.get('QDRANT_API_KEY'), timeout=180)
    total = qc.count(COLLECTION, exact=True).count
    goal = min(a.limit, total) if a.limit else total
    print(f'{total:,} points in {COLLECTION}; scoring {goal:,}\n', flush=True)

    start, done, offset, flagged = time.monotonic(), 0, None, 0
    while done < goal:
        points, offset = qc.scroll(COLLECTION, limit=min(a.batch, goal - done),
                                   offset=offset, with_vectors=True, with_payload=False)
        if not points:
            break
        marks = scorer.payload(np.array([p.vector['image'] for p in points],
                                        dtype=np.float32))
        if not a.dry_run:
            # One operation per point: set_payload applies a single payload to
            # every id it is given, so distinct scores need distinct ops. They
            # still travel in one request.
            qc.batch_update_points(COLLECTION, update_operations=[
                models.SetPayloadOperation(set_payload=models.SetPayload(
                    payload=mark, points=[p.id]))
                for p, mark in zip(points, marks)], wait=False)
        flagged += int((np.array([m['safety'] for m in marks]) >= 0.005).sum())
        done += len(points)
        rate = done / max(time.monotonic() - start, 1e-9)
        print(f'\r  {done:,}/{goal:,}  {rate:,.0f} pts/s  '
              f'{flagged:,} over threshold  '
              f'eta {(goal-done)/max(rate,1e-9)/60:.1f} min', end='', flush=True)
        if offset is None:
            break

    elapsed = time.monotonic() - start
    print(f'\n\nscored {done:,} in {elapsed/60:.1f} min ({done/max(elapsed,1e-9):,.0f} pts/s)')
    print(f'{flagged:,} ({100*flagged/max(done,1):.2f}%) score >= 0.005 and will now be filtered')
    if a.limit and done:
        print(f'full {total:,} would take ~{total/(done/elapsed)/60:.0f} min')


if __name__ == '__main__':
    main()
