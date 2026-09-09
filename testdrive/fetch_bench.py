"""What can we actually pull out of the bucket, and does concurrency help?

The GPU pass measured 50, 76 and 58 img/s across three runs while its
concurrency went 32, 32 and 128. That is not a trend, it is noise around a
ceiling somewhere else -- so this takes the GPU, the model and the batching
out of the picture and measures one thing: object fetches per second against
thread count.

    python testdrive/fetch_bench.py [n_objects]

If throughput is flat across concurrency levels the limit is upstream -- the
host's egress, or the bucket throttling this key -- and no amount of threading
will move it. If it climbs, the embed pass simply needs more loaders.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
env = Path(__file__).resolve().parents[1] / '.env'
if env.exists():
    for line in env.read_text().splitlines():
        k, _, v = line.strip().partition('=')
        if k:
            os.environ.setdefault(k, v)

os.environ.setdefault('S3_POOL', '512')
import storage

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

client = storage.client()
keys = []
for page in client.get_paginator('list_objects_v2').paginate(
        Bucket=storage.BUCKET, Prefix='t/'):
    keys += [o['Key'] for o in page.get('Contents', [])]
    if len(keys) >= N:
        break
keys = keys[:N]
print(f'{len(keys)} objects to fetch\n')


def one(key):
    t0 = time.monotonic()
    body = client.get_object(Bucket=storage.BUCKET, Key=key)['Body'].read()
    return time.monotonic() - t0, len(body)


print(f"{'threads':>8} {'img/s':>8} {'MB/s':>8} {'median fetch':>14}")
results = {}
for threads in (8, 32, 64, 128, 256):
    with ThreadPoolExecutor(threads) as pool:
        start = time.monotonic()
        out = list(pool.map(one, keys))
    elapsed = time.monotonic() - start
    latencies = sorted(t for t, _ in out)
    megabytes = sum(n for _, n in out) / 1024 / 1024
    rate = len(keys) / elapsed
    results[threads] = round(rate, 1)
    print(f'{threads:8} {rate:8.1f} {megabytes/elapsed:8.2f} '
          f'{latencies[len(latencies)//2]*1000:11.0f} ms')

best = max(results, key=results.get)
print(f'\nbest: {results[best]:.0f} img/s at {best} threads')
print(f'10M at that rate: {10_000_000/results[best]/3600:.1f} h of fetching alone')
if results[max(results)] < results[min(results)] * 2:
    print('\nFlat across concurrency: the limit is upstream, not our threading.')
Path('testdrive/fetch-bench.json').write_text(json.dumps(results, indent=2))
