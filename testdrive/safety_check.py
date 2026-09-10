"""Does the safety scorer separate anything, and where should the threshold go?

Eyeballing filenames is not a measurement. Commons titles and descriptions are
written by humans and are independent of the image embedding, so keyword
matches on that text give a weak but genuinely INDEPENDENT label for an
image-based scorer. Weak in both directions -- a file called "Nude study.jpg"
may be a marble statue, and plenty of explicit material is named IMG_4421 --
so treat these as a signal about separation, not as ground truth.

Scores real vectors already in Qdrant, so it costs no bucket reads and works
while the download cap is exhausted.

    python testdrive/safety_check.py [n]
"""
import json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for line in (Path(__file__).resolve().parents[1] / '.env').read_text().splitlines():
    k, _, v = line.strip().partition('=')
    if k and not k.startswith('#'):
        os.environ.setdefault(k, v)

import numpy as np
from qdrant_client import QdrantClient

import safety
from cloud_corpus import COLLECTION, search_text

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

EXPLICIT = re.compile(r'\b('
    r'nude|nudes|naked|topless|penis|vulva|vagina|genital|genitalia|erection|'
    r'ejaculat\w*|masturbat\w*|orgasm|porn\w*|erotic|fellatio|cunnilingus|'
    r'intercourse|coitus|breasts|nipples|buttocks|anus|scrotum|testicle\w*'
    r')\b', re.I)
GORE = re.compile(r'\b('
    r'corpse|cadaver|mutilat\w*|dismember\w*|amputat\w*|autopsy|massacre|'
    r'wound|wounds|gore|decapitat\w*|carcass|lesion|ulcer|gangrene|necrosis'
    r')\b', re.I)

print('loading SigLIP text tower...', flush=True)
import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
model = model.eval()
scorer = safety.Scorer(model, open_clip.get_tokenizer('ViT-B-16-SigLIP'))
print(f'{scorer.text.shape[0]} prompts, scale {scorer.scale:.1f}, bias {scorer.bias:.2f}\n',
      flush=True)

qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'),
                  timeout=180)
print(f'pulling {N:,} points...', flush=True)
points, offset = [], None
while len(points) < N:
    batch, offset = qc.scroll(COLLECTION, limit=256, offset=offset,
                              with_vectors=True, with_payload=True)
    if not batch:
        break
    points.extend(batch)
    if offset is None:
        break
    if len(points) % 2560 == 0:
        print(f'  {len(points):,}', flush=True)
print(f'  got {len(points):,}\n', flush=True)

vectors = np.array([p.vector['image'] for p in points], dtype=np.float32)
assert np.isfinite(vectors).all(), 'non-finite vectors in the corpus'
scored = scorer.score(vectors)
texts = [search_text(p.payload) for p in points]
explicit = np.array([bool(EXPLICIT.search(t)) for t in texts])
gore = np.array([bool(GORE.search(t)) for t in texts])
neutral = ~(explicit | gore)

print(f'weak labels from text: {explicit.sum()} explicit, {gore.sum()} gore, '
      f'{neutral.sum()} neither\n')

print(f"{'group':>14}  {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}")
for name in ('unsafe', 'clinical', 'artistic_nude', 'safe'):
    v = scored[name]
    print(f'{name:>14}  {v.mean():7.4f} {np.percentile(v,50):7.4f} '
          f'{np.percentile(v,95):7.4f} {np.percentile(v,99):7.4f} {v.max():7.4f}')

u = scored['unsafe']
if explicit.sum() >= 5:
    print(f'\nunsafe score, by weak label:')
    for label, mask in (('text says explicit', explicit), ('text says gore', gore),
                        ('neither', neutral)):
        if mask.sum():
            print(f'  {label:>18}  n={mask.sum():5}  mean {u[mask].mean():.4f}  '
                  f'median {np.median(u[mask]):.4f}  p90 {np.percentile(u[mask],90):.4f}')
    # AUC by rank: probability a random explicit image outranks a random neutral one.
    pos, neg = u[explicit], u[neutral]
    if len(pos) and len(neg):
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort().astype(np.float64) + 1
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f'\n  AUC vs neutral: {auc:.3f}   (0.5 = no signal, 1.0 = perfect)')

print('\nthreshold behaviour:')
print(f"  {'thr':>5} {'removed':>9} {'% corpus':>9} {'recall on explicit':>19}")
for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
    removed = int((u > t).sum())
    rec = (u[explicit] > t).mean() if explicit.sum() else float('nan')
    print(f'  {t:5.1f} {removed:9,} {100*removed/len(u):8.2f}% {100*rec:18.1f}%')

order = np.argsort(-u)
print('\nhighest scoring:')
for i in order[:10]:
    print(f"  {u[i]:.3f}  {(points[i].payload.get('title') or '')[:66]}")

Path('testdrive/safety-check.json').write_text(json.dumps({
    'sampled': len(points), 'explicit_labelled': int(explicit.sum()),
    'thresholds': {str(t): int((u > t).sum()) for t in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)},
    'unsafe_mean': float(u.mean()),
}, indent=2))
