"""Where should the safety threshold go?

Random sampling cannot answer this: the 435k corpus came from 481 curated
topics, so 10,000 random rows contained 8 text-labelled explicit images and the
resulting AUC rested on those 8. Useless.

So find the corpus's own worst material deliberately -- search for it with the
text tower, exactly as a user would stumble onto it -- and score THAT. Then
check what a legitimate search loses, because a threshold that removes
pornography and Renaissance painting equally is not a filter, it is a lobotomy.

    python testdrive/calibrate_safety.py
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for line in (Path(__file__).resolve().parents[1] / '.env').read_text().splitlines():
    k, _, v = line.strip().partition('=')
    if k and not k.startswith('#'):
        os.environ.setdefault(k, v)

import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient

import safety
from cloud_corpus import COLLECTION

# What a filter must catch.
HOSTILE = ['explicit sexual intercourse', 'erect penis close up', 'pornographic photograph',
           'ejaculation', 'naked woman spreading legs', 'a mutilated corpse',
           'severed limb with blood', 'graphic autopsy photograph']
# What it must NOT catch. These are the searches the product exists to serve.
LEGITIMATE = ['classical marble sculpture', 'renaissance oil painting', 'human anatomy diagram',
              'a surgeon operating in a hospital', 'ballet dancer', 'swimwear fashion photography',
              'baby being bathed', 'butcher shop meat counter', 'war memorial statue',
              'mountain landscape at sunrise', 'modern architecture facade', 'street food market']
TOP = 60

print('loading SigLIP...', flush=True)
import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
model = model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
scorer = safety.Scorer(model, tokenizer)

qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY'),
                  timeout=180)


def embed(text):
    with torch.inference_mode():
        v = F.normalize(model.encode_text(tokenizer([text])), dim=-1)
    return v[0].float().numpy()


def scores_for(queries, label):
    out, seen = [], set()
    for q in queries:
        hits = qc.query_points(COLLECTION, query=embed(q).tolist(), using='image',
                               limit=TOP, with_vectors=True, with_payload=True).points
        fresh = [h for h in hits if h.id not in seen]
        seen.update(h.id for h in fresh)
        if not fresh:
            continue
        vecs = np.array([h.vector['image'] for h in fresh], dtype=np.float32)
        s = scorer.score(vecs)['unsafe']
        out.extend(zip(s, (h.payload.get('title', '') for h in fresh), [q] * len(fresh)))
    print(f'{label}: {len(out)} distinct images from {len(queries)} queries')
    return out


hostile = scores_for(HOSTILE, 'hostile')
legit = scores_for(LEGITIMATE, 'legitimate')

# A neutral baseline, for what a blanket threshold costs the whole corpus.
pts, off, base = [], None, []
while len(pts) < 3000:
    b, off = qc.scroll(COLLECTION, limit=256, offset=off, with_vectors=True, with_payload=False)
    if not b:
        break
    pts.extend(b)
    if off is None:
        break
base = scorer.score(np.array([p.vector['image'] for p in pts], dtype=np.float32))['unsafe']

h = np.array([s for s, _, _ in hostile])
l = np.array([s for s, _, _ in legit])
print(f'\n{"set":>12} {"n":>6} {"median":>10} {"p75":>10} {"p90":>10} {"max":>10}')
for name, v in (('hostile', h), ('legitimate', l), ('random', base)):
    print(f'{name:>12} {len(v):6} {np.median(v):10.6f} {np.percentile(v,75):10.6f} '
          f'{np.percentile(v,90):10.6f} {v.max():10.6f}')

print(f'\n{"threshold":>12} {"catches hostile":>16} {"loses legit":>13} {"cuts corpus":>13}')
best = None
for t in (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5):
    caught, lost, cut = (h > t).mean(), (l > t).mean(), (base > t).mean()
    print(f'{t:12.5f} {100*caught:15.1f}% {100*lost:12.1f}% {100*cut:12.2f}%')
    # Prefer catching hostile material; treat collateral on legitimate
    # searches as three times as costly, since that is the product's job.
    value = caught - 3 * lost
    if best is None or value > best[1]:
        best = (t, value, caught, lost, cut)

t, _, caught, lost, cut = best
print(f'\nrecommended threshold {t:g}: catches {100*caught:.0f}% of hostile hits, '
      f'loses {100*lost:.1f}% of legitimate, removes {100*cut:.2f}% of the corpus')

# Which legitimate searches actually pay, and at what rate? A blanket loss
# figure hides the only thing that matters: losing results from "human anatomy
# diagram" is the filter working, losing them from "mountain landscape" is the
# filter broken.
print('\nper-query loss on legitimate searches:')
print(f'  {"query":>34} {"n":>4} {">0.001":>8} {">0.005":>8}')
for q in LEGITIMATE:
    v = np.array([s for s, _, qq in legit if qq == q])
    if not len(v):
        continue
    print(f'  {q:>34} {len(v):4} {100*(v>1e-3).mean():7.1f}% {100*(v>5e-3).mean():7.1f}%')

print('\nlegitimate images that would be removed (collateral):')
for s, title, q in sorted(legit, reverse=True)[:8]:
    if s > t:
        print(f'  {s:.6f}  [{q}]  {title[:52]}')

print('\nhostile hits that would survive (misses):')
for s, title, q in sorted(hostile)[:8]:
    if s <= t:
        print(f'  {s:.6f}  [{q}]  {title[:52]}')

Path('testdrive/safety-threshold.json').write_text(json.dumps({
    'threshold': t, 'hostile_recall': float(caught), 'legit_loss': float(lost),
    'corpus_removed': float(cut), 'n_hostile': len(h), 'n_legit': len(l)}, indent=2))
