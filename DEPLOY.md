# Zero-cost cloud deployment

Every component below is either the real production component or its managed
equivalent, running on a free tier. Nothing is a toy substitute.

| Production design | Free tier used here | Capacity | Card? |
|---|---|---|---|
| Crawl fleet on rented VPS IPs | GitHub Actions, 20 shards | unlimited (public repo) | no |
| Rented L4 for bulk embed | Kaggle Notebooks | 30 GPU-hrs **per week** | no |
| Backblaze B2 + Cloudflare CDN | **wsrv.nl proxy** (display) + **HF Datasets** (archive) | unlimited display, ~free archive | **none** |
| Qdrant on a €12/mo Hetzner box | **Qdrant Cloud free cluster** | 1 GB = ~300k int8 vectors | no |
| FastAPI on that same box | **HF Spaces** (Docker) | 2 vCPU, 16 GB RAM | no |
| Postgres for metadata | Qdrant payload | included | no |

**Ceiling: ~300k images**, set by the Qdrant free cluster (300k × 768d int8
= 0.23 GB of 1 GB). Display storage is no longer a limit at all.

**No credit card is required anywhere in this stack.**

---

## 1. Storage — nothing to sign up for

Two jobs, split, because they have opposite requirements.

### Display thumbnails: wsrv.nl proxy (zero storage)

`wsrv.nl` is a free public image proxy on Cloudflare's edge. Give it an
origin URL and it resizes, converts to WebP, and caches the result:

    https://wsrv.nl/?url=<encoded origin>&w=384&h=384&fit=cover&output=webp&q=80&maxage=1y

You already store every image's origin URL in `images.thumb_url`, so the
display URL is derived, not stored. **No bucket, no account, no card.**

Measured on this corpus:

| | |
|---|---|
| 172 KB origins → | **28.5 KB** WebP |
| warm (edge cached) | **235 ms** median |
| cold | 887 ms median |
| success rate | 24/24, then 6/6 end-to-end |

It is a courtesy service — cache hard (`maxage=1y` is already set) and keep
request volume sane. Slower than a real bucket (~30–50 ms), free forever.

```bash
export STORAGE_BACKEND=proxy      # this is the default
```

### Archive: Hugging Face Datasets (tarballs)

You still need the 384px derivatives somewhere, for one reason: **changing
embedding models later.** Without an archive that means re-crawling the
whole internet; with one it is a 14-minute GPU job.

`cloud_sync.py` pushes them as **tarballs, not individual files** — a handful
of large objects rather than 300k small ones. That sidesteps HF's ~100k
files-per-repo guidance entirely, and uploads far faster.

```bash
export HF_TOKEN=hf_xxx HF_REPO=you/imgsearch-corpus
python cloud_sync.py push --shard 0
```

Free, no card, email signup only.

### If you later want a real bucket

`STORAGE_BACKEND=s3` switches to R2/B2 with no other code change. Worth it
when proxy latency starts bothering you — R2's free tier is 10 GB with zero
egress, and Backblaze B2's 10 GB tier may not require a card either (worth
two minutes to check).

## 2. Qdrant Cloud  (3 min)

1. cloud.qdrant.io → **Create free cluster** (1 GB, no card, no expiry)
2. Copy the cluster URL and create an API key

```bash
export QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
export QDRANT_API_KEY=<key>
python push_qdrant.py create
```

Creates a 768d cosine collection with **int8 scalar quantization,
`always_ram=True`** and originals on disk — the exact config that lets 100M
vectors run on one box at full scale, just smaller.

## 3. Crawl in parallel  (GitHub Actions)

Push this repo to GitHub **public** (free unlimited Actions minutes), then
Settings → Secrets → Actions:

| secret | value |
|---|---|
| `CRAWL_UA` | `imgsearch/0.1 (https://github.com/you/repo; you@mail.com)` |
| `HF_TOKEN` `HF_REPO` | for the derivative archive |

Actions → **crawl** → Run workflow. 20 runners, each with its own IP. Each
resizes to 384px, tars its shard, and pushes the tarball to HF before the
runner is destroyed (runners have no persistent disk).

⚠️ Fair use: legitimate as a bounded dataset build for a project you publish.
A permanent 24/7 crawl farm is Actions abuse. Keep runs manual.

## 4. Embed on a free GPU  (Kaggle)

New notebook → Accelerator **GPU**, Internet **On**:

```python
!pip install -q open_clip_torch boto3 qdrant-client
!git clone https://github.com/YOU/imgsearch-proto && cd imgsearch-proto
%env S3_ENDPOINT=...
%env QDRANT_URL=...
!python embed_gpu.py --from-s3 --push
```

| | throughput | 300k images |
|---|---|---|
| M4 local | ~44 img/s | 1.9 hours |
| Kaggle P100 (fp16) | ~350 img/s | **~14 min** |

## 5. Deploy the API  (HF Spaces)

huggingface.co → **New Space** → SDK **Docker** → push the `space/` folder.
Space Settings → Variables and secrets:

    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION=images

The Dockerfile bakes the text tower into the image so cold starts don't
re-download it.

---

## The one structural rule

**The API returns JSON only. Thumbnails go browser → CDN directly.**

```
browser ──query──> Space ──vector──> Qdrant Cloud
   │                                      │
   └──── thumbnails ◄── wsrv.nl edge ◄────┘
                            │
                       origin CDNs
```

Proxying images through the API would funnel every thumbnail byte through
one container — destroying latency, and burning the free tier in a day.
Each result's `cdn` field in the Qdrant payload is what the browser loads.

## Measured on 2,071 vectors

| | |
|---|---|
| Qdrant query (in-memory, int8) | **1.7–4.8 ms** |
| License-filtered query | 10 ms |
| Proxy thumbnail, warm | 235 ms, 28.5 KB |
| `"lonely figure in vast empty space"` | abandoned factory, empty metro station, Tate Modern interior |

That last row is the thesis: nothing was tagged "lonely" or "empty".
