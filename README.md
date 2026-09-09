# imgsearch-proto — backend, corpus builder, and the record of the prototype

The macOS app lives in `../imgsearch-mac` (private GitHub repo `Zahide4/imgsearch-mac`).
This repo is everything behind it: the search API, the crawler, the embedding
pipeline, and `docs/10m-plan.html` — the architecture plan for the real build.

---

# Read this first if you are an AI agent

## Rules

**1. The corpus build is finished. Do not restart it.**
435,540 images are indexed and serving. The workflow that produced them is
rate-limit sensitive in three separate ways and one careless dispatch has
already killed most of a fleet mid-run.

**2. Never run `git checkout`, `git stash`, `git reset`, or `git clean`.**
There is uncommitted work here, including a crawler crash fix.

**3. Rate limits that have actually bitten this project.** Not hypotheticals —
each of these cost hours:

| Limit | What happened |
|---|---|
| **Hugging Face: 128 commits/hour, account-wide** | Scaling workers 4→20 scaled commit frequency 5× and killed 13 workers at once. Fixed by batching to one commit per save and raising the flush interval to 4000. |
| **MediaWiki per-IP throttling** | Hammering one host from one address produced a 60% failure rate that looked like missing images. Per-host semaphores took it to 0.6%. |
| **MediaWiki `urlparamnormal`** | It rejects non-NFC continuation cursors. Two workers died before params were NFC-normalised and the error was made cost a range, not a worker. |
| **`wsrv.nl` thumbnail proxy** | Blocked this IP outright for scripted requests. It still works from a browser and from the app's `URLSession`, so a failing `curl` does **not** mean thumbnails are broken. |

**4. Anything that looks like missing data is probably rate limiting.** A bare
`except` here hides one as the other. This repo has been burned by exactly that:
a swallowed `NameError` from a missing `urlparse` import made discovery return
zero Openverse results while looking perfectly healthy.

## Current state

- **435,540 images** in Qdrant Cloud, served by `server/app.py` on Render.
- The real target was **488,000** (24,400 × 20 workers), not 500,000 — the
  progress endpoint's "500000" is a display constant.
- The build ended at its own 4.5-hour wall-clock budget (`--max-seconds 16200`).
  Eleven workers saved resumable checkpoints; that is a designed stop, not a
  crash. Reusing `build=pilot-v1` resumes. **Only if asked.**
- **Uncommitted:** a fix in `cloud_corpus.py` for the one real crash. A single
  malformed PNG tripped Pillow's oversized-text-chunk guard, which raises a
  plain `ValueError` that `fetch_image` did not catch, and it killed a worker
  two hours in. `BUG_TYPES` now re-raises `NameError`/`AttributeError`/
  `ImportError` so a coding mistake can never again be mistaken for bad data.
  The workflow runs from the checked-out SHA, so it only takes effect once pushed.

## What the prototype proved

It was built to answer one question before any money was spent: does semantic
image search over a licence-filtered corpus actually work well enough to be a
product? It does.

- **Hybrid retrieval beats dense-only.** SigLIP dense vectors fused with BM25
  sparse by reciprocal rank. Validated on proper nouns, which is where dense
  embeddings are weakest: "Herstmonceux Castle" ranks #1 hybrid versus #3
  dense-only.
- **Warm queries take ~1.1–1.4s** — roughly 400ms SigLIP text embedding plus
  700ms ANN. The loading state is a real state.
- **Zero cost, throughout.** GitHub Actions for the crawl (20 concurrent
  runners, each with its own IP, which is the entire point), Hugging Face for
  archives, Qdrant Cloud free tier for vectors, Render free tier for the API.
  No credit card was used at any stage.

## Traps that fail silently

These do not raise. They quietly degrade search quality, which is worse.

- **4-bit quantization destroys SigLIP text embeddings.** Measured: cross-query
  similarity 0.86–0.97 under q4f16 versus 0.68–0.76 at float32. Everything
  looks like everything. There is a runtime gate for this — and the gate's own
  threshold was wrong once and would have rejected working backends.
- **Token padding must stay at 64.** Below it, cosine similarity collapses to
  0.45–0.63.
- **A hand-rolled tokenizer needs two fixes** or search degrades quietly:
  `pad_id` is 1, not 0, and SigLIP's canonicalisation (lowercase, strip
  punctuation, collapse whitespace) must be applied.
- **"Commercial use" filters still return no-derivatives licences.** Openverse's
  own `license_type=commercial` includes ND. Licences are whitelisted explicitly:
  `OV_LICENSES = "cc0,pdm,by,by-sa"`.
- **Photographer names do not belong in a lexical index.** They are
  high-cardinality tokens that crowd out real subject terms in BM25.
  `search_text()` deliberately excludes `creator`.
- **Markup leaks in through source metadata**, so `clean_text()` strips it.
- **An optimisation that works on one tower can reverse on the other.** ONNX
  Runtime wins on memory for the text tower; PyTorch is 3.4× faster for the
  image tower.

## The one thing that changes at 10M

Everything else in the prototype scales. This does not:

**Generate and store your own thumbnails.** The prototype serves images by
asking a free shared proxy to resize originals on demand, and that single
decision produced every serving problem encountered — broken images, slow
loads, copy failures, and finally an outright IP block when we tried to
pre-warm the cache. At 10M it is not a shortcut with rough edges; it is a
dependency that rate-limits you, fingerprints your tooling, and cannot be
warmed.

Resize once during ingest to a 384px WebP and PUT it to your own bucket. At the
22 KB average measured across this corpus that is **$1.32/month at 10M** and
$13/month at 100M. The thumbnails were never the expensive part — the vector
store is.

## The plan, in numbers

`docs/10m-plan.html` is the full architecture document. The shape of it:

| | 10M images | 100M images |
|---|---|---|
| Thumbnails (B2, free egress via Cloudflare) | $1.32 | $13 |
| Vector store + API (Hetzner ARM) | $13.50 | $27 |
| **Monthly** | **~$16** | **~$45** |
| Crawl + embed, one-time rented compute | ~$10 | ~$100 |

Wall clock for a 10M build is **~12–30 hours pipelined**, and crawling is the
only phase that cannot be accelerated: it is 10M HTTP round trips against a
donated service. More runners and faster machines do nothing for it. At 100M
that becomes the wall, and bulk dumps stop being a nicety and become the
architecture.

**A design lesson from the 500k run that is not yet in the plan document:**
static shard striping cannot drain a tail. Each worker owned a fixed slice of
alphabet ranges (`all_ranges[worker::workers]`) and a fixed quota, so a finished
worker exited rather than helping the stragglers, and a crashed worker left a
hole nothing filled. The run ended when the slowest quota ended, not when the
work did. The fix is a claim-based shared queue — ranges as
`unclaimed / in-flight / done` with expiring leases.

## Deferred, deliberately

Legal and compliance work: a `distribution_allowed` flag, a blocklist table, and
a takedown contact. The owner has seen the arguments and chosen to defer. Do not
re-litigate it unasked.

---

# Current deployment

The active app is `server/` on Render, backed by Qdrant Cloud.
Use `.github/workflows/cloud-corpus.yml` to crawl, embed, archive and index on
cloud workers; no image corpus is stored on your Mac. See [DEPLOY.md](DEPLOY.md)
for the current workflow, validation and capacity limits. The notes below
describe the earlier local prototype and experiments; their capacity and
throughput estimates are historical, not guarantees.

---

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
