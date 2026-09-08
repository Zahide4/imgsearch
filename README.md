# imgsearch-proto

Small-scale, zero-cost prototype of the universal image search app.
Architecturally identical to the production design — every stage maps 1:1,
only the backing service changes when you scale.

| Stage | Prototype (free, local) | Production |
|---|---|---|
| Sources | Wikimedia Commons API | Commons dumps + Openverse + museums |
| Crawl | httpx async | same, sharded on cheap VPSes |
| Resize | Pillow → 384px WebP | pyvips → 384px WebP (streaming) |
| Embed | SigLIP on Apple GPU (MPS) | same model, rented GPU |
| Blob store | `./data/thumbs/` | Backblaze B2 + Cloudflare CDN |
| Metadata | SQLite | Postgres |
| Vectors | `vectors.npy` + numpy dot | Qdrant, binary quantization + HNSW |
| API | FastAPI on localhost | same FastAPI, €12/mo Hetzner box |

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then **edit the `UA` constant at the top of `ingest.py`** with a real contact
URL or email. Wikimedia's User-Agent policy requires it and blocks generic ones.

## Build the index

```bash
python ingest.py --per-topic 60     # ~6k images, ~20 min
python ingest.py --per-topic 250    # ~25k images, ~90 min
```

Safe to re-run — it checkpoints in SQLite and skips what it already has.

## Search

```bash
python server.py     # http://127.0.0.1:8000
```

## What to measure

1. **Latency** — top-right of the search bar, split into client and server.
   Server-side should be single-digit ms. That is your "instant" proof.
2. **Coverage** — run your 100-query list. Score one binary judgement each:
   *would I use anything in these results?* Track it per category.
3. **Semantic wins** — search things nobody would ever tag, e.g.
   `lonely figure in vast empty space`, `warm nostalgic summer evening`.
   Keyword search cannot do these. That gap is the entire product thesis.

---

## Scaling up: multi-source + parallel cloud crawl

### The rate-limit fix (works locally, no cloud needed)

The original crawler pulled everything from `upload.wikimedia.org`, so all
concurrency hit one server that throttles per-IP. Result: 60% of fetches
died as 429s.

Two changes fixed it:

1. **Openverse as a second source** — 915M works across 52 providers
   (Flickr, iNaturalist, Wikimedia, Europeana, Smithsonian, Met, rawpixel,
   NASA, svgsilh...). We take each result's origin `url`, not Openverse's
   proxy thumbnail, so images come from ~14 different CDNs.
2. **`HostLimiter`** — one semaphore *per host* instead of one globally.
   6 concurrent to each of 14 hosts is 84 polite fetches in flight.

Measured: **1 host / 60% failures → 14 hosts / 0.6% failures.**

```bash
python ingest.py --per-topic 60 --source openverse,commons
```

⚠️ Openverse's own `license_type=commercial` still returns **ND**
(no-derivatives), which forbids cropping or compositing — useless for an
editor. We whitelist `cc0,pdm,by,by-sa` explicitly instead.

Openverse also returns **tags**, stored in `images.tags`. That is your
keyword field for hybrid search later, for free, no VLM required.

### Corpus size is bounded by topics, not runners

    images ≈ topics × sources(14) × per_topic × fill_rate(~0.5)

88 topics × 14 × 40 × 0.5 ≈ **25k**. To reach 300k you need ~500 topics,
not more machines. Expand `seeds.txt` first.

### Parallel crawl on GitHub Actions (free)

`.github/workflows/crawl.yml` runs 20 shards in parallel. Each runner has
its own IP, so the per-IP limit that caps one machine multiplies by 20.
Free and unlimited on **public** repos.

**Fair use:** legitimate as a bounded, occasional dataset build for a project
you publish. A permanent 24/7 crawl farm is Actions abuse. Keep runs manual
and shard counts sane.

Setup:

1. Push this repo to GitHub (public).
2. Free Hugging Face account → write token → new dataset repo.
3. Repo Settings → Secrets → Actions, add:
   - `HF_TOKEN` — your write token
   - `HF_REPO`  — e.g. `yourname/imgsearch-corpus`
   - `CRAWL_UA` — `imgsearch/0.1 (https://github.com/you/repo; you@mail.com)`
4. Actions tab → **crawl** → Run workflow.

Then locally:

```bash
export HF_TOKEN=hf_xxx HF_REPO=yourname/imgsearch-corpus
python cloud_sync.py pull      # merge all shards
python ingest.py --embed-only  # ~44 img/s on an M4
python server.py
```

### Why Hugging Face and not R2/B2

Both R2 and B2 have 10GB free tiers but want a credit card on file.
HF Datasets wants an email. At ~21KB per 384px WebP that is ~400k images
free. Outgrowing it is a one-function swap in `cloud_sync.py`:
`upload_file()` → `put_object()` against B2.

### Embedding without a Mac

Kaggle Notebooks give **30 free GPU hours/week** (P100/T4). At ~400 img/s
that is 300k images in ~12 minutes. Colab's free T4 works too.
`ingest.py --embed-only` runs unchanged; just point it at the pulled data.
