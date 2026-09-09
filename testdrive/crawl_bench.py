"""Step B2: what does the *fixed* crawler sustain, per IP, fetch-only?

The 11-30 hour estimate for a 10M build rests on ~240 img/s across 20 IPs,
i.e. ~12 img/s each. The 500k run managed 1-2 img/s per worker, but that had
embedding on the same CPU runner and a global semaphore that concentrated
traffic on one host. This measures the real discovery-and-fetch path with
neither of those.

Three things it does that the original plan's Step 3 did not:

  * Time-boxed, not count-boxed, so it reports a *sustained* rate rather than
    a burst. Bursts are exactly what Wikimedia punishes.
  * Per-host accounting. Step A's 6.2% aggregate failure was 0% on one host
    and 39% on another; an average would have hidden the whole finding.
  * Error rate bucketed over time, because the question Step 3 was really
    asking is whether throttling is cumulative. Measured today: it is, over a
    short window, and it recovers in about ten minutes.

Nothing is uploaded. No credentials are needed and the corpus is untouched.

    .venv/bin/python testdrive/crawl_bench.py [seconds] [worker]

Run it from a cold IP -- no other Wikimedia traffic from this machine for
fifteen minutes beforehand -- or you are measuring your own previous run.
"""
import asyncio, collections, json, sys, time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import cloud_corpus as cc

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 2700        # 45 minutes
WORKER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BUCKET = 300                                                       # 5-minute bins

ok = collections.Counter()          # per host
bad = collections.Counter()         # per host
by_bucket = collections.defaultdict(lambda: [0, 0])
discovered_host = collections.Counter()
page_count = 0


async def main():
    global page_count
    limiter = cc.HostLimiter()
    queue = deque({'start': lo, 'end': hi, 'continue': {}}
                  for lo, hi in cc.ranges(WORKER, 20))
    start = time.monotonic()

    async with httpx.AsyncClient(headers={'User-Agent': cc.UA}, follow_redirects=True) as client:
        while queue and time.monotonic() - start < DURATION:
            job = queue[0]
            r = await cc.request(client, cc.COMMONS,
                                 params=cc.params_for(job['start'], job['end'], job['continue']))
            data = r.json()
            if data.get('error'):
                code = data['error'].get('code')
                if code in ('maxlag', 'ratelimited'):
                    bad['<discovery>'] += 1
                    await asyncio.sleep(30)
                    continue
                queue.popleft()
                continue

            rows = [row for p in data.get('query', {}).get('pages', [])
                    if (row := cc.metadata(p, job['end']))]
            rows = list({row['image_id']: row for row in rows}.values())
            for row in rows:
                discovered_host[urlparse(row['thumb_origin']).netloc] += 1

            results = await asyncio.gather(
                *(cc.fetch_image(client, row, limiter) for row in rows),
                return_exceptions=True)

            bucket = int((time.monotonic() - start) // BUCKET)
            for row, out in zip(rows, results):
                host = urlparse(row['thumb_origin']).netloc
                if isinstance(out, Exception) or out is None:
                    bad[host] += 1
                    by_bucket[bucket][1] += 1
                else:
                    ok[host] += 1
                    by_bucket[bucket][0] += 1

            page_count += 1
            queue.popleft()
            continuation = data.get('continue')
            if continuation and continuation != job['continue']:
                job['continue'] = continuation
                queue.append(job)

            elapsed = time.monotonic() - start
            total = sum(ok.values())
            print(f'\r  {elapsed/60:5.1f} min  {total:6,} fetched  '
                  f'{total/max(elapsed,1):5.2f} img/s  {sum(bad.values()):5,} failed',
                  end='', flush=True)

    return time.monotonic() - start


elapsed = asyncio.run(main())
total_ok, total_bad = sum(ok.values()), sum(bad.values())
rate = total_ok / max(elapsed, 1)

print('\n\nper host')
print(f'  {"host":32} {"ok":>7} {"failed":>7} {"fail %":>7}')
for host in sorted(set(ok) | set(bad), key=lambda h: -(ok[h] + bad[h])):
    n = ok[host] + bad[host]
    print(f'  {host:32} {ok[host]:7,} {bad[host]:7,} {100*bad[host]/max(n,1):6.1f}%')

print('\nfailure rate over time -- flat means throttling is not cumulative')
for b in sorted(by_bucket):
    good, fail = by_bucket[b]
    print(f'  {b*BUCKET//60:3d}-{(b+1)*BUCKET//60:3d} min  '
          f'{good:6,} ok  {fail:5,} failed  {100*fail/max(good+fail,1):5.1f}%')

strict = discovered_host.get('upload.wikimedia.org', 0)
found = sum(discovered_host.values()) or 1
report = {
    'seconds': round(elapsed, 1), 'pages': page_count,
    'fetched': total_ok, 'failed': total_bad,
    'img_per_s': round(rate, 2),
    'failure_pct': round(100 * total_bad / max(total_ok + total_bad, 1), 2),
    'strict_host_share_pct': round(100 * strict / found, 1),
    'per_host': {h: {'ok': ok[h], 'failed': bad[h]} for h in set(ok) | set(bad)},
}
Path('testdrive/step-b2-result.json').write_text(json.dumps(report, indent=2))

print(f'\n  sustained          : {rate:.2f} img/s on one IP')
print(f'  projected 20 IPs   : {rate*20:.0f} img/s   (plan assumes ~240)')
print(f'  10M at that rate   : {10_000_000/max(rate*20,0.01)/3600:.1f} h   (plan says 11-30)')
print(f'  strict-host share  : {report["strict_host_share_pct"]}%   (was ~16% at iiurlwidth=800)')
print(f'\n{"GO" if rate > 10 else "NO-GO"}: gate is >10 img/s per IP')
