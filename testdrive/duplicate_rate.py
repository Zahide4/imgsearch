"""How much of the corpus is the same picture twice?

`docs/10m-plan.html` lists this under open risks -- the prototype surfaced 365
scans of a single manuscript -- and nothing has measured it. It matters more
than it sounds: enumeration at 10M finds every page of every scanned book, and
a corpus of 10M images in which a tenth are near-duplicates of each other is
worth rather less than 10M.

Three angles, all read-only against the live collection:

  exact      identical bytes, by the sha1 MediaWiki reports
  near       cosine >= 0.95 to another image -- different scans of one page,
             crops, resolutions of the same photograph
  clustered  titles sharing a long prefix, which is how scanned documents and
             photo series are named

    .venv/bin/python testdrive/duplicate_rate.py [sample]
"""
import collections, json, os, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for line in open(Path(__file__).resolve().parents[1] / '.env'):
    key, _, value = line.strip().partition('=')
    if key:
        os.environ[key] = value

from qdrant_client import QdrantClient, models

SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NEAR, VERY_NEAR = 0.95, 0.98

qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=120)
C = os.getenv('QDRANT_COLLECTION', 'images-v2')
total = qc.count(C, exact=True).count
print(f'{total:,} points in {C}\n', flush=True)

# Scroll in id order. Qdrant's scroll offset is a point ID, not a row number,
# so a random integer there is meaningless -- and these ids are UUIDs derived
# from a hash of the image id, which means id order is already uncorrelated
# with subject, date or uploader. Sequential is both correct and unbiased.
print(f'sampling {SAMPLE} points...', flush=True)
sample, cursor = [], None
while len(sample) < SAMPLE:
    points, cursor = qc.scroll(C, limit=min(64, SAMPLE - len(sample)), offset=cursor,
                               with_payload=['title', 'sha1', 'source_url'], with_vectors=True)
    sample.extend(p for p in points if p.vector)
    print(f'  {len(sample)}/{SAMPLE}', flush=True)
    if cursor is None or not points:
        break
sample = sample[:SAMPLE]
print(f'  got {len(sample)}\n', flush=True)

print('measuring near-duplicates (each point queried against the whole corpus)...', flush=True)
near_counts, very_near_counts, examples = [], [], []
for i, p in enumerate(sample):
    hits = qc.query_points(C, query=p.vector['image'], using='image', limit=21,
                           with_payload=['title']).points
    others = [h for h in hits if h.id != p.id]
    near = [h for h in others if h.score >= NEAR]
    very = [h for h in others if h.score >= VERY_NEAR]
    near_counts.append(len(near))
    very_near_counts.append(len(very))
    if len(near) >= 5 and len(examples) < 5:
        examples.append((p.payload.get('title', '?'), len(near),
                         [h.payload.get('title', '?') for h in near[:3]]))
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(sample)}', flush=True)

def share(counts, threshold=1):
    return 100 * sum(1 for c in counts if c >= threshold) / max(len(counts), 1)

print(f'\nof {len(sample)} sampled images, within their top 20 neighbours:')
print(f'  {share(very_near_counts):5.1f}%  have at least one match at cosine >= {VERY_NEAR}')
print(f'  {share(near_counts):5.1f}%  have at least one match at cosine >= {NEAR}')
print(f'  {share(near_counts, 5):5.1f}%  have five or more at cosine >= {NEAR}')
print(f'  mean near-duplicates per image: {sum(near_counts)/max(len(near_counts),1):.2f}')

# Exact duplicates: MediaWiki dedupes uploads by sha1, so this should be ~0.
# If it is not, the crawler is writing the same file under two ids.
sha = collections.Counter(p.payload.get('sha1') for p in sample if p.payload.get('sha1'))
dupe_sha = sum(n - 1 for n in sha.values() if n > 1)
print(f'\nexact duplicates by sha1 in the sample: {dupe_sha}')

# Series naming: "Foo - page 12.jpg", "Foo (1).jpg" -- the shape scanned
# documents and photo runs arrive in.
def stem(title):
    t = re.sub(r'\.(jpg|jpeg|png|tif|tiff|webp|gif)$', '', title or '', flags=re.I)
    t = re.sub(r'[\s_\-–—]*(\(?\d+\)?|[ivxlc]+|p{1,2}\.?\s*\d+|page\s*\d+|no\.?\s*\d+)\s*$', '', t, flags=re.I)
    return re.sub(r'\s+', ' ', t).strip().lower()

stems = collections.Counter(stem(p.payload.get('title', '')) for p in sample)
series = {k: v for k, v in stems.items() if v > 1 and len(k) > 8}
print(f'title-series clusters in the sample: {len(series)} '
      f'covering {sum(series.values())} images')

if examples:
    print('\nexamples with five or more near-duplicates:')
    for title, n, neighbours in examples:
        print(f'  {title[:56]}  ({n} near)')
        for nb in neighbours:
            print(f'      ~ {nb[:56]}')

Path('testdrive/duplicate-rate.json').write_text(json.dumps({
    'sampled': len(sample), 'corpus': total,
    'pct_with_a_match_0.98': round(share(very_near_counts), 2),
    'pct_with_a_match_0.95': round(share(near_counts), 2),
    'pct_with_5plus_0.95': round(share(near_counts, 5), 2),
    'mean_near_per_image': round(sum(near_counts)/max(len(near_counts),1), 3),
    'exact_sha1_dupes': dupe_sha,
    'title_series_clusters': len(series),
}, indent=2))
print('\nwritten to testdrive/duplicate-rate.json')
