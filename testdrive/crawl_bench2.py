"""Step B2, second pass: does overlapping discovery with fetching close the gap?

The first pass measured 3.20 img/s sustained, zero failures in 8,650 fetches.
Zero failures says Wikimedia is not the constraint at this rate; the constraint
is our own loop, which discovers a page of 50 and only then starts fetching it.
About 7 seconds of every 15.6-second cycle has nothing downloading.

Rather than pipeline inside one loop -- which would mean restructuring the
continuation-cursor handling that killed two workers in the 500k run -- this
runs several independent streams over a shared range queue. While one stream
is waiting on `allimages`, the others are fetching. The HostLimiter is shared,
so total in-flight requests per host are capped exactly as before: this asks
Wikimedia for no more concurrency than the last run did, it just stops our own
fetchers idling.

    .venv/bin/python testdrive/crawl_bench2.py [cooldown_s] [measure_s] [streams] [per_host]

Defaults: 900s cooldown, 1800s measurement, 3 streams, 8 per host.
The cooldown is built in because throttling is cumulative over roughly ten
minutes -- a measurement taken straight after another one is worthless.
"""
import asyncio, collections, json, os, sys, time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

COOLDOWN = int(sys.argv[1]) if len(sys.argv) > 1 else 900
MEASURE = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
STREAMS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PER_HOST = int(sys.argv[4]) if len(sys.argv) > 4 else 8
BASELINE = 3.20                                    # img/s, measured 2026-09-09

# HostLimiter reads this at construction; upload.wikimedia.org stays at 2
# regardless, since that is the host that actually complained.
os.environ['FETCH_CONCURRENCY'] = str(PER_HOST)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import cloud_corpus as cc

ok, bad = collections.Counter(), collections.Counter()
by_bucket = collections.defaultdict(lambda: [0, 0])
discovered = collections.Counter()
pages = 0
BUCKET = 300


async def stream(client, limiter, queue, lock, deadline):
    """One range at a time: discover it, then fetch it. Several of these
    overlap, so a stream waiting on the API does not stall the others."""
    global pages
    while time.monotonic() < deadline:
        async with lock:
            if not queue:
                return
            job = queue.popleft()

        try:
            r = await cc.request(client, cc.COMMONS,
                                 params=cc.params_for(job['start'], job['end'], job['continue']))
            data = r.json()
        except Exception:
            bad['<discovery>'] += 1
            continue

        if data.get('error'):
            if data['error'].get('code') in ('maxlag', 'ratelimited'):
                bad['<discovery>'] += 1
                async with lock:
                    queue.append(job)
                await asyncio.sleep(30)
            continue

        rows = [row for p in data.get('query', {}).get('pages', [])
                if (row := cc.metadata(p, job['end']))]
        rows = list({row['image_id']: row for row in rows}.values())

        # Requeue the continuation before fetching, so another stream can pick
        # up the next page of this range while this one is still downloading.
        continuation = data.get('continue')
        if continuation and continuation != job['continue']:
            job = dict(job, **{'continue': continuation})
            async with lock:
                queue.append(job)

        for row in rows:
            discovered[urlparse(row['thumb_origin']).netloc] += 1

        results = await asyncio.gather(
            *(cc.fetch_image(client, row, limiter) for row in rows),
            return_exceptions=True)

        bucket = int((time.monotonic() - START) // BUCKET)
        for row, out in zip(rows, results):
            host = urlparse(row['thumb_origin']).netloc
            if isinstance(out, Exception) or out is None:
                bad[host] += 1
                by_bucket[bucket][1] += 1
            else:
                ok[host] += 1
                by_bucket[bucket][0] += 1
        pages += 1


async def report_progress(deadline):
    while time.monotonic() < deadline:
        await asyncio.sleep(15)
        elapsed = time.monotonic() - START
        total = sum(ok.values())
        print(f'\r  {elapsed/60:5.1f} min  {total:6,} fetched  '
              f'{total/max(elapsed,1):5.2f} img/s  {sum(bad.values()):4,} failed',
              end='', flush=True)


async def main():
    # The client is created and closed INSIDE the loop that uses it. Closing it
    # after asyncio.run() returns means closing a client bound to a loop that
    # no longer exists, which throws and takes the whole report with it.
    global client
    client = httpx.AsyncClient(headers={'User-Agent': cc.UA}, follow_redirects=True)
    try:
        await run_streams()
    finally:
        await client.aclose()


async def run_streams():
    limiter = cc.HostLimiter()
    queue = deque({'start': lo, 'end': hi, 'continue': {}} for lo, hi in cc.ranges(0, 20))
    lock = asyncio.Lock()
    deadline = START + MEASURE
    workers = [asyncio.create_task(stream(client, limiter, queue, lock, deadline))
               for _ in range(STREAMS)]
    ticker = asyncio.create_task(report_progress(deadline))
    await asyncio.gather(*workers, return_exceptions=True)
    ticker.cancel()


print(f'cooling down {COOLDOWN//60} min so this measures the service, not the last run')
for left in range(COOLDOWN, 0, -30):
    print(f'\r  {left//60:2d}:{left%60:02d} remaining', end='', flush=True)
    time.sleep(min(30, left))
print(f'\r  cooldown done                 ')
print(f'measuring {MEASURE//60} min: {STREAMS} streams, {PER_HOST} per host '
      f'(upload.wikimedia.org stays at 2)\n')

client = None
START = time.monotonic()
asyncio.run(main())
elapsed = time.monotonic() - START

total_ok, total_bad = sum(ok.values()), sum(bad.values())
rate = total_ok / max(elapsed, 1)

print('\n\nper host')
for host in sorted(set(ok) | set(bad), key=lambda h: -(ok[h] + bad[h])):
    n = ok[host] + bad[host]
    print(f'  {host:32} {ok[host]:7,} ok  {bad[host]:5,} failed  {100*bad[host]/max(n,1):5.1f}%')

print('\nper 5-minute bucket -- instantaneous rate, not the running average')
for b in sorted(by_bucket):
    good, fail = by_bucket[b]
    print(f'  {b*BUCKET//60:3d}-{(b+1)*BUCKET//60:3d} min  {good:6,} ok  {fail:5,} failed  '
          f'{100*fail/max(good+fail,1):5.1f}%   {good/BUCKET:5.2f} img/s')
settled = [good for b, (good, _) in sorted(by_bucket.items()) if b >= 2]
sustained = sum(settled) / (len(settled) * BUCKET) if settled else rate
print(f'\n  sustained after warm-up (from 10 min on): {sustained:.2f} img/s')

strict = discovered.get('upload.wikimedia.org', 0)
found = sum(discovered.values()) or 1
Path('testdrive/step-b2-pipelined.json').write_text(json.dumps({
    'streams': STREAMS, 'per_host': PER_HOST, 'seconds': round(elapsed, 1),
    'pages': pages, 'fetched': total_ok, 'failed': total_bad,
    'img_per_s_mean': round(rate, 2), 'img_per_s_sustained': round(sustained, 2), 'baseline_img_per_s': BASELINE,
    'failure_pct': round(100 * total_bad / max(total_ok + total_bad, 1), 2),
    'strict_host_share_pct': round(100 * strict / found, 1),
}, indent=2))

# Projected from the settled rate, not the running average: a fast first
# minute inflates a cumulative mean for the rest of the run.
hours = 10_000_000 / max(sustained * 20, 0.01) / 3600
print(f'\n  baseline           : {BASELINE:.2f} img/s   (1 stream, 6 per host)')
print(f'  now, running mean  : {rate:.2f} img/s')
print(f'  now, sustained     : {sustained:.2f} img/s   ({sustained/BASELINE:.1f}x)')
print(f'  projected 20 IPs   : {sustained*20:.0f} img/s')
print(f'  10M wall clock     : {hours:.1f} h   (plan says 11-30)')
print(f'  failure rate       : {100*total_bad/max(total_ok+total_bad,1):.2f}%   (was 0.00%)')
print(f'\n{"GO" if hours <= 30 and total_bad/max(total_ok,1) < 0.02 else "NO-GO"}: '
      f'the gate that matters is 10M inside 30 h with failures under 2%')
