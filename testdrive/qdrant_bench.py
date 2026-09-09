"""Step D: does 10M actually fit, and does binary quantization hold recall?

Since Hetzner's June 2026 repricing this is the most valuable measurement left.
int8 needs ~11 GB at 10M and wants a CAX31 at EUR19.49/mo; binary with disk
rescore needs ~4.3 GB and fits a CAX21 at roughly half that. Every month.

Recall is measured against exact search on the same index -- brute force is the
ground truth, and anything else is comparing two approximations to each other.

    python qdrant_bench.py [target_url] [quantization]
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
from qdrant_client import QdrantClient, models

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:6333'
QUANT = (sys.argv[2] if len(sys.argv) > 2 else 'int8').lower()
COLLECTION = f'images-{QUANT}'
PROBES = 50
TARGET_CORPUS = 10_000_000

qc = QdrantClient(url=URL, timeout=180)
count = qc.count(COLLECTION, exact=True).count
scale = TARGET_CORPUS / count
print(f'{COLLECTION}: {count:,} points, projecting x{scale:.1f} to 10M\n')


def disk_bytes():
    out = subprocess.run(['du', '-sb', '/var/lib/qdrant/storage'],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def rss_bytes():
    out = subprocess.run(['bash', '-c',
                          "ps -eo rss,comm | grep -i qdrant | awk '{s+=$1} END {print s*1024}'"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


ram, disk = rss_bytes(), disk_bytes()
print(f'RAM  {ram/2**30:6.2f} GB  -> {ram*scale/2**30:6.2f} GB at 10M')
print(f'disk {disk/2**30:6.2f} GB  -> {disk*scale/2**30:6.2f} GB at 10M\n')

# Query vectors sampled from the corpus itself: a real embedding, not a random
# one. Random 768-d vectors are near-orthogonal to everything and would make
# any quantization look flawless.
sample, _ = qc.scroll(COLLECTION, limit=PROBES, with_vectors=True, with_payload=False)
queries = [p.vector['image'] for p in sample]

print(f'latency and recall over {len(queries)} real query vectors')
results = {}
for label, params in (
    ('quantized, rescore on',
     models.SearchParams(hnsw_ef=128, quantization=models.QuantizationSearchParams(
         rescore=True, oversampling=2.0))),
    ('quantized, rescore off',
     models.SearchParams(hnsw_ef=128, quantization=models.QuantizationSearchParams(
         rescore=False))),
):
    latencies, overlaps = [], []
    for vector in queries:
        t0 = time.monotonic()
        hits = qc.query_points(COLLECTION, query=vector, using='image',
                               limit=10, search_params=params).points
        latencies.append((time.monotonic() - t0) * 1000)
        truth = qc.query_points(COLLECTION, query=vector, using='image', limit=10,
                                search_params=models.SearchParams(exact=True)).points
        overlaps.append(len(set(h.id for h in hits) & set(t.id for t in truth)) / 10)
    latencies.sort()
    results[label] = {
        'p50_ms': round(latencies[len(latencies)//2], 1),
        'p99_ms': round(latencies[min(len(latencies)-1, int(len(latencies)*0.99))], 1),
        'recall_at_10': round(statistics.mean(overlaps), 4),
    }
    r = results[label]
    print(f"  {label:24} p50 {r['p50_ms']:6.1f} ms   p99 {r['p99_ms']:6.1f} ms   "
          f"recall@10 {r['recall_at_10']*100:5.1f}%")

report = {'collection': COLLECTION, 'quantization': QUANT, 'points': count,
          'ram_gb': round(ram/2**30, 2), 'disk_gb': round(disk/2**30, 2),
          'projected_ram_gb_10M': round(ram*scale/2**30, 2),
          'projected_disk_gb_10M': round(disk*scale/2**30, 2),
          'latency_recall': results}
Path(f'/tmp/step-d-{QUANT}.json').write_text(json.dumps(report, indent=2))

best = results['quantized, rescore on']
ram10 = ram * scale / 2**30
print(f'\ngates')
print(f"  RAM at 10M   < 11 GB   : {ram10:6.2f} GB  {'PASS' if ram10 < 11 else 'FAIL'}")
print(f"  disk at 10M  < 100 GB  : {disk*scale/2**30:6.2f} GB  "
      f"{'PASS' if disk*scale/2**30 < 100 else 'FAIL'}")
print(f"  p99          < 700 ms  : {best['p99_ms']:6.1f} ms  "
      f"{'PASS' if best['p99_ms'] < 700 else 'FAIL'}")
print(f"  recall@10    > 95%     : {best['recall_at_10']*100:5.1f}%  "
      f"{'PASS' if best['recall_at_10'] > 0.95 else 'FAIL'}")
print(f'\nwritten to /tmp/step-d-{QUANT}.json')
if QUANT == 'int8':
    print('\nnow run the same thing with binary -- if RAM at 10M drops under 8 GB')
    print('and recall stays above 90%, a CAX21 fits and the bill halves.')
