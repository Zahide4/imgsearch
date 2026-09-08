#!/usr/bin/env python3
"""
warm_cache.py -- pay the proxy's cold-fetch cost once, at ingest, not per user.

A thumbnail nobody has requested before costs 624ms median (p90 2.1s) because
wsrv.nl must fetch the original and resize it. Slow origins dominate that tail:

    2,475ms  collections.museumsvictoria.com.au
    2,346ms  dms-cf-08.dimu.org
    2,237ms  images.metmuseum.org
      661ms  thumb.wikimedia.org

Once cached the same image serves in ~293ms regardless of how slow its origin
is. So the museum penalty is not really a museum problem -- it is a
"first requester pays" problem, and there is no rule saying the first
requester has to be a user.

This walks the corpus and requests every thumbnail once. Run it after each
corpus build. Idempotent and safe to re-run; already-cached images return
fast and cost almost nothing.

    python warm_cache.py            # whole collection
    python warm_cache.py --limit 500
"""
import argparse
import asyncio
import os
import time
import urllib.parse

import httpx
from qdrant_client import QdrantClient

COLLECTION = os.getenv('QDRANT_COLLECTION', 'images-v2')
UA = {'User-Agent': os.getenv('CRAWL_UA', 'ImgSearch/0.2 (+https://github.com/Zahide4/imgsearch)')}


def thumb_url(payload) -> str:
    """Resolve a thumbnail exactly the way the API does.

    Records carry EITHER a prebuilt `cdn` URL or a raw `thumb_origin` that the
    API wraps in the proxy at query time. Reading only `cdn` silently skips
    every record produced by cloud_corpus.py -- 400 of 12,345 today, and all
    of them at 500k, which would have made this script a no-op exactly when
    it matters most.
    """
    cdn = payload.get('cdn')
    if cdn:
        return cdn
    origin = payload.get('thumb_origin') or ''
    if not origin:
        return ''
    return ('https://wsrv.nl/?url=' + urllib.parse.quote(origin, safe='')
            + '&w=384&h=384&fit=inside&output=webp&q=80&maxage=1y')


def host_of(url: str) -> str:
    try:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get('url', [''])[0]
        return urllib.parse.urlparse(q).netloc or 'unknown'
    except Exception:
        return 'unknown'


async def warm(client, url, sem, stats):
    async with sem:
        t0 = time.time()
        try:
            r = await client.get(url, timeout=90, follow_redirects=True)
            ok = r.status_code == 200
        except Exception:
            ok = False
        ms = (time.time() - t0) * 1000
    h = host_of(url)
    s = stats.setdefault(h, [0, 0, 0.0])
    s[0] += 1
    s[1] += 1 if ok else 0
    s[2] += ms
    return ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='0 = whole collection')
    ap.add_argument('--concurrency', type=int, default=12,
                    help='wsrv.nl is a courtesy service; stay modest')
    args = ap.parse_args()

    qc = QdrantClient(url=os.environ['QDRANT_URL'],
                      api_key=os.environ['QDRANT_API_KEY'], timeout=120)
    urls, offset = [], None
    while True:
        recs, offset = qc.scroll(COLLECTION, limit=1000, offset=offset,
                                 with_payload=True, with_vectors=False)
        if not recs:
            break
        urls += [u for u in (thumb_url(r.payload or {}) for r in recs) if u]
        if offset is None or (args.limit and len(urls) >= args.limit):
            break
    if args.limit:
        urls = urls[:args.limit]
    print(f'warming {len(urls):,} thumbnails')

    sem = asyncio.Semaphore(args.concurrency)
    stats, done, t0 = {}, 0, time.time()
    async with httpx.AsyncClient(headers=UA) as client:
        for i in range(0, len(urls), 200):
            chunk = urls[i:i + 200]
            res = await asyncio.gather(*[warm(client, u, sem, stats) for u in chunk])
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            print(f'\r  {done}/{len(urls)}  ({rate:.0f}/s)', end='', flush=True)
    print()

    ok = sum(v[1] for v in stats.values())
    print(f'warmed {ok}/{len(urls)}')
    print('\nslowest origins (these are the ones users no longer wait on):')
    rows = sorted(stats.items(), key=lambda kv: -(kv[1][2] / max(kv[1][0], 1)))
    for h, (n, o, tot) in rows[:8]:
        print(f'   {tot / max(n, 1):7.0f}ms  n={n:5}  {h}')


if __name__ == '__main__':
    asyncio.run(main())
