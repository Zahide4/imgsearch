"""What does the safety scorer actually say about the corpus we have?

Scores real vectors from the live 435k collection. Nothing is downloaded --
the vectors are already in Qdrant, so this costs no bucket reads and works
while the download cap is exhausted.

    python testdrive/safety_check.py [n]
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for line in (Path(__file__).resolve().parents[1] / '.env').read_text().splitlines():
    k, _, v = line.strip().partition('=')
    if k and not k.startswith('#'):
        os.environ.setdefault(k, v)

import numpy as np
from qdrant_client import QdrantClient

import safety
from cloud_corpus import COLLECTION

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

print('loading SigLIP text tower...', flush=True)
import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
model = model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
scorer = safety.Scorer(model, tokenizer)
print(f'{scorer.text.shape[0]} prompt vectors, logit scale {scorer.scale:.1f}\n', flush=True)

qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'),
                  timeout=120)
print(f'pulling {N:,} points with vectors...', flush=True)
points, offset = [], None
while len(points) < N:
    batch, offset = qc.scroll(COLLECTION, limit=min(256, N - len(points)),
                              offset=offset, with_vectors=True, with_payload=True)
    if not batch:
        break
    points.extend(batch)
    if offset is None:
        break
print(f'  got {len(points):,}\n', flush=True)

vectors = np.array([p.vector['image'] for p in points], dtype=np.float32)
scored = scorer.score(vectors)

print(f"{'group':>14}  {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}")
for name in ('unsafe', 'clinical', 'artistic_nude', 'safe'):
    v = scored[name]
    print(f'{name:>14}  {v.mean():7.4f} {np.percentile(v,50):7.4f} '
          f'{np.percentile(v,95):7.4f} {np.percentile(v,99):7.4f} {v.max():7.4f}')

print('\nfraction of corpus above each unsafe threshold:')
for t in (0.3, 0.5, 0.7, 0.9):
    n = int((scored['unsafe'] > t).sum())
    print(f'  > {t:.1f}   {n:5,} / {len(points):,}  ({100*n/len(points):5.2f}%)')

order = np.argsort(-scored['unsafe'])
print('\nhighest-scoring — the ones a threshold would remove:')
for i in order[:12]:
    p = points[i].payload
    title = (p.get('title') or p.get('image_id', ''))[:64]
    print(f"  {scored['unsafe'][i]:.3f}  {title}")
print('\nlowest-scoring — should look completely ordinary:')
for i in order[-5:]:
    p = points[i].payload
    print(f"  {scored['unsafe'][i]:.3f}  {(p.get('title') or '')[:64]}")

Path('testdrive/safety-check.json').write_text(json.dumps({
    'sampled': len(points),
    'thresholds': {str(t): int((scored['unsafe'] > t).sum()) for t in (0.3, 0.5, 0.7, 0.9)},
    'unsafe_mean': float(scored['unsafe'].mean()),
    'unsafe_p99': float(np.percentile(scored['unsafe'], 99)),
}, indent=2))
