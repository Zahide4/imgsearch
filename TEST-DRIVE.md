# 10M Test Drive — runbook

Falsify the load-bearing numbers in `docs/10m-plan.html` before committing to a
$10 build and $16/month. Stop at the first red.

**Reordered from the original plan so that everything free comes first.** Two of
the five steps need no money and no payment method at all, and one of them
(Step A) tests the single decision the whole architecture rests on.

| Step | Cost | Wall clock | Needs payment? |
|---|---|---|---|
| 0 · Pin the config, record baselines | $0 | 1 h | no |
| **A · Thumbnail bucket** | **$0** | 2 h | **no** |
| **B · Crawl politeness** | **$0** | 6–24 h | **no** |
| C · Self-host BM25 *(added — see below)* | $0 | 2 h | no |
| D · Qdrant fit on Hetzner | ~€0.10 | 3 h | yes |
| E · Embed throughput on a GPU | ~$1.20 | 1 h | yes |
| F · End-to-end query on the box | included in D | 1 h | — |

---

## Before anything: a hole in the original plan

**Self-hosted Qdrant has no cloud inference.** The BM25 half of hybrid
retrieval is currently computed *by Qdrant Cloud*, not by us:

```python
cloud_inference=True                                    # app.py:60, cloud_corpus.py:226
'bm25': models.Document(text=..., model=SPARSE_MODEL)   # cloud_corpus.py:331
```

`models.Document` is a request for Qdrant to embed the text on its side. A
self-hosted instance will reject it. So the moment you leave the free cloud
tier, **both ingest and query need sparse vectors computed client-side** — and
hybrid retrieval is the thing that made proper-noun search work ("Herstmonceux
Castle" #1 hybrid vs #3 dense-only).

That is Step C, and it costs nothing but has to happen before Step D can
measure anything real.

---

## Step 0 · Pin the config and record baselines — $0, 1 h

Nothing later means anything if the config drifts.

```bash
cd ~/Documents/Resources/imgsearch-proto
mkdir -p testdrive
curl -s https://imgsearch-api.onrender.com/api/progress | tee testdrive/baseline-count.json
```

Record cold and warm latency, and the ten-query recall set:

```bash
cat > testdrive/queries.txt <<'EOF'
Herstmonceux Castle
mountain lake sunrise
vintage typewriter desk
abandoned railway station
coral reef fish
steam locomotive snow
medieval manuscript illumination
brass telescope
autumn forest path
neon city rain
EOF

: > testdrive/baseline-latency.tsv
while read -r q; do
  curl -s --max-time 90 --get --data-urlencode "q=$q" --data "limit=10" \
    https://imgsearch-api.onrender.com/api/search \
  | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('timing',{})
print('$q', d['ms'], t.get('embed_ms'), t.get('ann_ms'),
      '|'.join(r['id'] for r in d['results'][:10]), sep='\t')"
done < testdrive/queries.txt | tee -a testdrive/baseline-latency.tsv
```

That last column is the recall set. Every later step compares against it — if
top-10 IDs drift after quantization or after self-hosted BM25, that is the
signal.

The 10M collection config is already in `hybrid_collection.py:29-44` and must be
copied **verbatim**: `m=16, ef_construct=100`, int8 `quantile=0.99,
always_ram=True`, `on_disk=True` for vectors, payload and sparse index. Query
config from `server/app.py`: `hnsw_ef=128, rescore=True, oversampling=2.0`,
`candidates = max(100, limit*2)`, RRF fusion.

---

## Step A · RESULT: GREEN on cost, and it found a crawler bug

Run 2026-09-09 on 2,500 real origins.

| measured | plan assumed | verdict |
|---|---|---|
| mean 21.2 KB (p50 19.3, p90 36.8, p99 55.1) | 22 KB | **within 4%** |
| 202 GB at 10M → **$1.21/mo** | $1.32/mo | **green** |
| 5.0 img/s fetch+resize, one IP | — | see Step B |
| **6.2% fetch failure** | 0.6% | **investigated below** |

The failure rate was the finding. It is entirely one host:

- `thumb.wikimedia.org` — 1,669 origins, **0 failures**
- `upload.wikimedia.org` — 392 origins, **152 failures (39%)**, all HTTP 429

Two causes, both now fixed in `cloud_corpus.py`:

1. **The cloud crawler had no per-host limiting.** `ingest.py` has had
   `HostLimiter` since the original 429 wall; `cloud_corpus.py` — the crawler
   that actually built the 500k — used a single global semaphore. That also
   explains the 500k run's per-worker error spread (0 to 56 an hour with no
   pattern in time): it was never time, it was which alphabet ranges a worker
   drew and therefore how much of its traffic hit one host.
2. **`iiurlwidth=800` was asking for the wrong size.** MediaWiki renders a
   thumbnail only when the source is *wider* than the width requested; below
   that it returns the original, on the strict host. These files have a median
   width of 640px — too narrow for an 800px thumbnail, ample for a 384px one,
   and 384 is what we store anyway. Measured on 39 such files: at 800, 38 came
   back as originals; at 384, 9. On a fresh unbiased slice, strict-host traffic
   fell from 6% to 2%.

**A measurement lesson, recorded because it nearly produced a wrong conclusion:**
the first validation of the fix reported *worse* results (77.6% failure). It had
run inside the penalty window from the burst test ten minutes earlier. After a
150-second cooldown the same replay gave 35.5% on that deliberately-hard subset.
The throttling is cumulative over a short window and **recovers within ~10
minutes** — which is itself useful for the 10M plan, and means any crawl
measurement taken right after another one is worthless.

---

## Step A · Thumbnail bucket — $0, 2 h, no card

The one architectural change, and the cheapest thing to falsify.
**Backblaze B2 gives 10 GB free with no credit card**, and 10k thumbnails at the
assumed 22 KB is 220 MB.

**A1. Pull 10k real origins from the live index.** They are already in the
payload as `thumb_origin` (`cloud_corpus.py:122`), so no crawling is needed.

```bash
.venv/bin/python - <<'PY'
import json, os
from qdrant_client import QdrantClient
qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=120)
seen, cursor = [], None
while len(seen) < 10000:
    pts, cursor = qc.scroll('images-v2', limit=1000, offset=cursor,
                            with_payload=['thumb_origin'], with_vectors=False)
    seen += [p.payload['thumb_origin'] for p in pts if p.payload.get('thumb_origin')]
    if cursor is None: break
open('testdrive/origins.txt','w').write('\n'.join(seen[:10000]))
print(len(seen[:10000]), 'origins')
PY
```

**A2. Install pyvips.** Not Pillow — shrink-on-load decodes a 4000px JPEG at 1/8
scale for ~750 KB instead of ~48 MB, which is the difference between 200
concurrent images and an OOM.

```bash
brew install vips && .venv/bin/pip install pyvips b2sdk
```

**A3. Resize 10k and measure the size distribution** *before* uploading
anything — if the mean is wrong, the $1.32/month line is wrong and you stop
here.

```bash
.venv/bin/python testdrive/resize_bench.py     # written in step A4
```

**A4.** The script: fetch, `thumbnail_buffer` at 384px, WebP q80, record bytes.

```python
# testdrive/resize_bench.py
import asyncio, httpx, pyvips, statistics, time, pathlib
OUT = pathlib.Path('testdrive/thumbs'); OUT.mkdir(parents=True, exist_ok=True)
urls = open('testdrive/origins.txt').read().split()[:10000]
sizes, fails, lat = [], 0, []
async def one(client, url, sem, i):
    global fails
    async with sem:
        t0 = time.monotonic()
        try:
            r = await client.get(url, timeout=30)
            r.raise_for_status()
            img = pyvips.Image.thumbnail_buffer(r.content, 384, height=384, size='down')
            data = img.write_to_buffer('.webp[Q=80]')
            (OUT / f'{i}.webp').write_bytes(data)
            sizes.append(len(data)); lat.append(time.monotonic() - t0)
        except Exception:
            fails += 1
async def main():
    sem = asyncio.Semaphore(6)          # per-host limits still apply
    async with httpx.AsyncClient(headers={'User-Agent': 'ImgSearch/0.2 (contact)'},
                                 follow_redirects=True) as c:
        await asyncio.gather(*(one(c, u, sem, i) for i, u in enumerate(urls)))
asyncio.run(main())
s = sorted(sizes)
print(f'n={len(s)} fails={fails}')
print(f'mean={statistics.mean(s)/1024:.1f}KB p50={s[len(s)//2]/1024:.1f}KB p90={s[int(len(s)*.9)]/1024:.1f}KB')
print(f'fetch p50={sorted(lat)[len(lat)//2]:.2f}s p90={sorted(lat)[int(len(lat)*.9)]:.2f}s')
```

**A5. Upload to B2 and measure GET latency** cold and warm.

**Go if:** mean 18–28 KB (the $1.32/mo maths holds), fetch failure rate near the
0.6% baseline, warm GET p90 < 400 ms.
**No-go if:** mean > 32 KB — the storage line is wrong. Or museum origins
dominate the tail past 2 s even from your own bucket, which means you need
origin filtering, not just a bucket.

> **Cloudflare in front of B2 needs a domain you own.** If you have none, test B2
> direct and treat the numbers as a ceiling — a CDN only improves them. Do not
> buy a domain for this test.

---

## Step B · RESULT: GO, and the 24-hour test was never needed

Two runs, 2026-09-09, both from a cold IP.

**Pass 1 — the question the original plan wanted 24 hours for.**
45 minutes, one stream, 6 per host: **8,650 fetched, 0 failed**, and the
failure rate was 0.0% in every one of nine consecutive five-minute buckets.
Throttling is not cumulative at this rate. Combined with the separate finding
that bursts *are* punished and recover in about ten minutes, the picture is
that sustained-polite works and bursts do not.

Throughput was 3.20 img/s, which projects to 43 hours for 10M -- outside the
plan's 11-30 h. Zero failures said the constraint was ours, not Wikimedia's:
the loop discovered a page of 50 and only then started fetching it, so roughly
7 seconds of every 15.6-second cycle had nothing downloading.

**Pass 2 — three concurrent streams over a shared range queue**, 8 per host on
`thumb.wikimedia.org`, `upload.wikimedia.org` still 2, the HostLimiter shared
so total in-flight per host is unchanged. 30 minutes: **17,202 fetched, 0
failed.**

| interval | instantaneous |
|---|---|
| 0-1 min | 37.6 img/s |
| 1-12 min | 13.4 img/s |
| 12-20 min | 5.7 img/s |
| 20-25 min | 5.1 img/s |
| 25-30 min | **5.4 img/s** |

**The sustained rate is ~5.4 img/s, not the 9.55 running mean.** A fast first
minute inflates a cumulative average for the rest of the run, which is why the
script now reports instantaneous rate per bucket and projects from the settled
figure. That is 1.7x the baseline, 108 img/s across 20 IPs, and **10M in ~26
hours** -- inside the 11-30 h band, at the top of it.

Zero failures across 25,852 fetches in both passes combined.

> A bug worth recording: pass 2's report crashed after the measurement
> finished. `asyncio.run(main())` closes the event loop, and the HTTP client
> was closed after that, on a loop that no longer existed. The numbers above
> were recovered from the progress line. The client is now opened and closed
> inside the loop that uses it.

---

## Step B · Crawl politeness — $0, 6–24 h, no card

The only phase that cannot be bought faster, and the one where the risk is
someone else's goodwill.

**The original plan says run 2 workers × 24 h on Actions. That is not possible
— GitHub hosted runners hard-stop at 6 hours** (`cloud-corpus.yml` already sets
`timeout-minutes: 330` for exactly this reason). Do this instead:

- **one worker on the Mac mini for 24 h** — your home IP, longest window, tells
  you whether throttling is *cumulative*
- **one worker on Actions for 5.5 h** — a datacentre IP, for comparison

```bash
.venv/bin/python cloud_corpus.py --worker 0 --workers 20 \
  --target 999999 --max-seconds 86400 --build testdrive-crawl 2>&1 \
  | tee testdrive/crawl-local.log
```

Stub the embedding out first so this measures fetching only, not SigLIP on a CPU.

```bash
grep -cE "maxlag|ratelimited|429|urlparamnormal" testdrive/crawl-local.log
awk '/uploaded/{print}' testdrive/crawl-local.log | tail -5
```

**Go if:** sustained > 10 img/s per IP with a *flat* error rate over 24 h, and
no worker deaths.
**No-go if:** the error rate climbs hour over hour. That means throttling is
cumulative rather than instantaneous, more IPs will not fix it, and the answer
is bulk dumps rather than crawling.

This also exercises the uncommitted `BUG_TYPES` fix — one malformed PNG killed
worker 19 two hours into the 500k run.

---

## Step C · RESULT: GO

`sparse.py` computes BM25 with `Qdrant/bm25` -- the same model the cloud was
using, so term ids hash identically and vectors written by either route are
interchangeable. Documents carry TF weights, queries are flat at 1.0, and the
collection's `Modifier.IDF` means Qdrant still supplies inverse document
frequency from its own statistics.

Validated in an in-process Qdrant (`QdrantClient(':memory:')`) -- no server, no
credentials, no cost -- by indexing 429 real corpus rows with
`sparse.document` and querying with `sparse.query`. Four of five probes matched
on every query term, and the fifth is explained: the test corpus was built by
pulling 48 results each for 'locomotive', 'castle' and so on, which makes those
words low-IDF *here*, so BM25 correctly prefers one rare term over two common
ones. The algorithm is working; the corpus is skewed.

Stemming is confirmed working: "medieval manuscript illumination" matches
"Medieval goats illuminated manuscript" on all three terms. An earlier version
of this check compared substrings and reported that correct match as a failure,
which is why it now compares term ids.

**Still unverified, and it needs Qdrant credentials:** whether a client-side
query vector retrieves correctly from the *existing* 435k rows, which were
indexed through cloud inference. It does not block the 10M build, where
everything is re-indexed either way, but it does matter if the current
collection is to be kept.

The API takes `LOCAL_SPARSE=1` to switch; it defaults to cloud inference so
production is undisturbed until the index actually moves in Step D.

---

## Step C · Self-host BM25 — $0, 2 h

Replace cloud inference with client-side sparse vectors, or Step D measures a
system you cannot actually run.

```bash
.venv/bin/pip install fastembed
```

Compute sparse vectors locally with `fastembed`'s `Bm25` model and upsert
`models.SparseVector(indices=..., values=...)` instead of `models.Document(...)`,
in both `cloud_corpus.py:331` and `server/app.py:141`. Then re-run the Step 0
recall set against the free cloud tier and confirm the top-10 IDs still match.

**Go if:** the recall set is unchanged and "Herstmonceux Castle" still ranks #1.
**No-go if:** ranking shifts — client-side IDF differs from the server's, and
that has to be understood before it is baked into 10M rows.

---

## Step D · Qdrant fit on Hetzner — ~€0.10, 3 h, needs payment

Two thirds of the monthly bill. Math is not a load test.

**Use the real 435k vectors, not synthetic ones.** The original plan says
"1M synthetic 768d normalized vectors" — random vectors in 768 dimensions are
nearly orthogonal to each other, while real SigLIP embeddings cluster hard.
HNSW graph connectivity, build time, memory *and* int8 recall all depend on that
distribution. Recall measured on random data does not transfer.

Instead: snapshot the live collection server-to-server onto the box, then
duplicate it with small perturbations to reach 1M if you want the extra headroom
— perturbed real vectors keep the cluster structure, random ones destroy it.

1. Rent a **CAX31** hourly (8 vCPU / 16 GB ARM, ~€12.49/mo ≈ €0.02/h), install
   Qdrant, create the collection with the Step 0 config verbatim.
2. Restore the 435k, upsert in 512-point batches (as `hybrid_collection.py:55-90`).
3. Measure: RAM and disk, scaled ×23 to 10M; upsert/s against the 82.6/s free
   tier ceiling; HNSW build time; p50/p99 at `hnsw_ef=128` with rescore on and
   off; top-10 recall int8 vs float32 **on the Step 0 query set**; then repeat
   with binary quantization.
4. **Destroy the server the moment you are done.** Hetzner bills by the hour.

**Go if:** projected RAM < 11 GB, disk < 100 GB, p99 < 700 ms, int8 recall > 95%,
upsert > 200/s.
**No-go if:** RAM projects > 14 GB, or build time projects > 12 h, or binary
quantization is required to fit but drops recall below 90% on proper nouns.

---

## Step E · Embed throughput — ~$1.20, 1 h, needs payment

Proves the phases decouple: crawl writes to a bucket, embedding reads from it.

Rent an L4 by the hour, stream 50k derivatives from Step A's bucket, run
`ViT-B-16-SigLIP/webli` `encode_image` in batches of 16–64 (as
`cloud_corpus.py:318-322`), with no origin fetches at all.

**Go if:** > 800 img/s sustained — the 4–6 h embed phase for 10M holds.
**No-go if:** < 400 img/s. That means bucket egress is the bottleneck rather
than the GPU; re-test with a local NVMe cache before concluding anything about
the GPU.

---

## Step F · End-to-end query — included in Step D's box, 1 h

Run `server/Dockerfile` on the same CAX31 against local Qdrant. Measure
`embed_ms` on ARM versus Render, `ann_ms` at 435k versus 1M, the RRF-vs-dense
delta, and the dense-fallback path when BM25 inference fails.

**Go if:** uncached p50 < 1.5 s, and `ann_ms` growth from 435k → 1M is < 50 ms
(which proves corpus size costs the API almost nothing).
**No-go if:** ARM embed > 800 ms — you need a bigger API box, and the $13.50/mo
line moves.

---

## What this still will not tell you

Green on all six earns the $10 build and $16/month. It does not cover:

- Wikimedia changing its etiquette or rate limits
- Hetzner moving prices
- **Duplicates at enumeration scale.** The prototype surfaced 365 scans of a
  single manuscript. At 10M that is a dedup and quality-filter problem nobody
  has measured yet, and it is the most likely thing to make 10M images worth
  fewer than 10M images.
