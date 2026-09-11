#!/usr/bin/env python3
"""Bounded Commons dataset build. Run on cloud workers, never on the API host.

Downloads exist only on the worker; WebDataset archives and restart cursors go
into HF Datasets, vectors go directly to Qdrant. No image data returns to a Mac.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import random
import re
import string
import tarfile
import time
import pathlib
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import checkpoint
import safety
import sparse
import storage
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, urlparse

import httpx
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.errors import EntryNotFoundError
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ingest import COMMONS, license_ok, _clean
from push_qdrant import point_id, license_class

UA = os.getenv('CRAWL_UA', 'ImgSearch/0.2 (https://github.com/Zahide4/imgsearch)')
COLLECTION = os.getenv('QDRANT_COLLECTION', 'images-v2')
SPARSE_MODEL = 'qdrant/bm25'


_TAGS = re.compile(r'<[^>]+>')
_WS = re.compile(r'\s+')


def clean_text(value):
    """Strip markup and collapse whitespace.

    Openverse returns microformat HTML in some titles, e.g.
    "<div class='fn'> Water Drops</div>". Measured at 0.5% of a 12,345-point
    corpus (plus 0.4% empty). Left alone it renders as literal markup in the
    UI and, worse, feeds tokens like div/class/style/fn into the BM25 index
    where they compete with real subject terms.
    """
    return _WS.sub(' ', _TAGS.sub(' ', str(value or ''))).strip()


def search_text(row):
    """Compact lexical representation used by Qdrant's BM25 sparse index.

    Deliberately excludes `creator`. Photographer names are not what anyone
    searches an image library for, and indexing them means a query like
    "williams" matches every photo by Phil Williams. It also dilutes IDF:
    contributor names are high-cardinality tokens that crowd out real subject
    terms in the sparse index.
    """
    return ' '.join(filter(None, (
        clean_text(row.get('title')), clean_text(row.get('description')),
        clean_text(row.get('tags')),
    )))[:1400]


def pending_count(args, manifest_rows, archive, pending_points):
    # What save() would actually persist in this mode: manifest rows for
    # crawl-only, un-upserted points for no-archive, tarball pendings for
    # coupled+archive. Keying the flush off archive.pending in the modes
    # that never fill it meant no-archive workers uploaded exactly once --
    # at the very end -- so the indexed count never moved mid-run and a
    # dead worker lost hours, not minutes.
    if args.no_embed:
        return len(manifest_rows)
    if args.no_archive:
        return len(pending_points)
    return len(archive.pending)


def flush_every(args):
    default = '4000' if not args.no_embed and not args.no_archive else '500'
    return int(os.getenv('FLUSH_EVERY', default))


def shard_bounds():
    # Filename/sortkey prefix shards shared by the allimages walker and the
    # category walker: disjoint, shuffled once with a fixed seed so every
    # worker derives the same assignment without coordination.
    starts = sorted(set([''] + list(string.digits + string.ascii_uppercase) +
                        [a + b for a in string.ascii_uppercase for b in string.ascii_lowercase]))
    all_ranges = list(zip(starts, starts[1:] + [None]))
    random.Random(20260908).shuffle(all_ranges)
    return all_ranges


def ranges(worker, workers):
    # Hundreds of independent ranges distribute discovery across subjects,
    # rather than taking 125k nearly identical names from one letter.
    return shard_bounds()[worker::workers]


def category_jobs(categories, worker, workers):
    # One job per (category, sortkey-prefix shard) for a flat
    # generator=categorymembers crawl (the Quality/Featured/Valued layer).
    # Sortkeys default to file titles, so these shards partition category
    # members the way ranges() partitions allimages -- with the same seam
    # tolerance: dedupe by image_id absorbs overlaps, and the server-side
    # gcmendsortkey bounds each shard (metadata() gets end=None because the
    # bound is a sortkey, not a title). Strided assignment keeps every worker
    # on every category, so a small category finishes across all workers
    # instead of stalling on one.
    all_jobs = [{'cat': cat, 'start': lo, 'end': hi, 'continue': {}}
                for cat in categories for lo, hi in shard_bounds()]
    return all_jobs[worker::workers]


# Two width bands, not more: `filew:>N` and `filew:<N` BOTH include N
# (measured -- sunset+CC-Zero splits 17,186 / 4,193 against a 21,179 total, and
# `filew:3000` alone is 200), so bands must not share a boundary. `<2999` plus
# `>3000` sums to exactly 21,179: exhaustive and disjoint. Cirrus has no range
# form (`filew:1500..2999` returns nothing), so two is what the syntax allows --
# and two is enough. Eight licence shards times two bands is sixteen queries per
# topic at 10,000 each, which is more depth than any topic's clean pool holds.
WIDTH_BANDS = ['filew:>3000', 'filew:<2999']

# Commercially clean licence categories, in rough descending size. Sharding on
# these does three things at once: it keeps share-alike out of a corpus the app
# hides by default (56% of an unsharded harvest), it multiplies the 10,000-result
# cap by the number of shards, and it makes every crawled image one a default
# user can actually see.
CLEAN_LICENCES = ['CC-Zero', 'CC-BY-4.0', 'CC-BY-2.0', 'CC-BY-3.0',
                  'CC-PD-Mark', 'PD-old-100-expired', 'PD-self', 'PD-1996']


def job_weight(topic, licences, hits):
    """Rows a (topic, licence, band) shard is expected to yield.

    Only relative size matters, so the clean-licence fraction cancels out. The
    10,000 cap does not: it is where the search API stops paginating, and a
    200,000-hit topic is not twenty times the work of a 10,000-hit one.
    """
    shards = max(1, len(licences) * len(WIDTH_BANDS))
    return min(10000.0, max(hits.get(topic, 0), 0) / shards) or 1.0


def search_jobs(topics, licences, worker, workers, hits=None):
    """One job per (topic, licence, width band), balanced across workers.

    Search order is relevance order, so a partial job keeps the BEST of its
    slice rather than an arbitrary alphabetical one -- the whole reason this
    beats the category walk.

    Assignment is longest-processing-time-first bin packing, NOT the
    `jobs[worker::workers]` striding the other two walkers use. Measured on the
    Quality/Featured/Valued run, striding left NINE of forty lanes with zero
    rows while the busiest took 24,906 against a 10,387 average -- a 2.4x
    penalty on wall clock, because a lane that draws empty shards exits instead
    of helping. README.md records the same failure from the 500k run and
    prescribes a claim queue; that needs a lock server, and Qdrant is a poor
    one. Bin packing needs no coordination at all -- every worker derives the
    same assignment from the same inputs -- and it is available here for a
    reason the alphabet-shard walkers cannot match: shard size is KNOWN before
    the crawl, because every topic's Commons hit count was measured. LPT is
    within 4/3 of optimal makespan, and in practice far closer.

    Without measured hits it falls back to shuffled striding, which is the old
    behaviour and the old imbalance.
    """
    all_jobs = [{'topic': t, 'lic': lic, 'band': band, 'continue': {}}
                for t in topics for lic in licences for band in WIDTH_BANDS]
    if not hits:
        random.Random(20260910).shuffle(all_jobs)
        return all_jobs[worker::workers]
    # Deterministic order, so every worker packs the bins identically without
    # talking to any other worker.
    ordered = sorted(all_jobs,
                     key=lambda j: (-job_weight(j['topic'], licences, hits),
                                    j['topic'], j['lic'], j['band']))
    load = [0.0] * workers
    bins = [[] for _ in range(workers)]
    for job in ordered:
        light = min(range(workers), key=lambda i: (load[i], i))
        bins[light].append(job)
        load[light] += job_weight(job['topic'], licences, hits)
    return bins[worker]


def params_for_search(topic, lic, band, continuation):
    # Relevance harvest. Same imageinfo shape as the other two walkers, so
    # metadata(), fetch, embed, safety and upsert are untouched downstream.
    # gsrlimit maxes at 50 (500 is bot-only), which is ten times the requests
    # per image of the allimages walk and still negligible beside 50 image
    # fetches per page.
    params = dict(action='query', format='json', formatversion=2,
                  generator='search', gsrnamespace=6, gsrlimit=50,
                  gsrsearch=f'{topic} filetype:bitmap incategory:"{lic}" {band}',
                  prop='imageinfo', iiprop='url|size|extmetadata|mime|sha1',
                  iiurlwidth=384, maxlag=5)
    params.update(continuation)  # gsroffset
    return {k: unicodedata.normalize('NFC', v) if isinstance(v, str) else v
            for k, v in params.items()}


def params_for(start, end, continuation):
    params = dict(action='query', format='json', formatversion=2,
                  generator='allimages', gailimit=50, gaisort='name', gaifrom=start,
                  prop='imageinfo', iiprop='url|size|extmetadata|mime|sha1',
                  # 384, not 800, because 384 is what we store. MediaWiki only
                  # renders a thumbnail when the source is WIDER than the width
                  # asked for; below that it hands back the original, on
                  # upload.wikimedia.org, which rate-limits far harder than the
                  # thumbnail host. These files have a median width of 640px --
                  # too narrow for an 800px thumbnail, ample for a 384px one.
                  # Measured on 39 such files: at 800, 38 came back as
                  # originals; at 384, only 9 did.
                  iiurlwidth=384, maxlag=5)
    if end is not None:
        params['gaito'] = end
    params.update(continuation)  # retain gaifrom during imageinfo continuation
    # MediaWiki rejects parameters that are not NFC-normalized with
    # `urlparamnormal`. Continuation cursors are filenames handed back to us
    # by the API, and Commons contains names in decomposed form, so echoing
    # one back verbatim eventually kills the worker. Two workers died this
    # way ~23 pages in.
    return {k: unicodedata.normalize('NFC', v) if isinstance(v, str) else v
            for k, v in params.items()}


def params_for_category(cat, start, end, continuation):
    # Flat category enumeration for the Quality/Featured/Valued layer.
    # gcmtype=file keeps it to files (no subcategories, no recursion);
    # sortkey order plus prefix bounds shard the category the way gaifrom/
    # gaito shard allimages. Same imageinfo shape downstream, so metadata(),
    # fetch, embed, safety and upsert are untouched. Empty start means from
    # the very beginning; end None means to the very end.
    params = dict(action='query', format='json', formatversion=2,
                  generator='categorymembers', gcmtitle=cat, gcmtype='file',
                  gcmlimit=50, gcmsort='sortkey',
                  prop='imageinfo', iiprop='url|size|extmetadata|mime|sha1',
                  iiurlwidth=384, maxlag=5)
    if start:
        params['gcmstartsortkey'] = start
    if end is not None:
        params['gcmendsortkey'] = end
    params.update(continuation)  # retain sortkey position during imageinfo continuation
    return {k: unicodedata.normalize('NFC', v) if isinstance(v, str) else v
            for k, v in params.items()}


class Relevance:
    """Scores a harvested batch against the topic that retrieved it.

    The quality gate in the first corpus plan asked a generic question -- "is
    this a photograph?" -- which cannot tell a good photograph of a car park
    from a good photograph of a sunset when you searched for a sunset. Asking
    the specific question costs the same matmul, because the image vector is
    already in memory, and it drops scanned book pages for free: a book page
    scores near zero against "a photo of sunset" without anyone writing a
    prompt for book pages.

    Thresholds are RELATIVE to each topic's own opening batch, never absolute.
    Measured across topics, the same "clearly relevant" band lands anywhere
    from .03 to .53 depending only on how the prompt embeds, so one global
    constant would gate some topics to nothing and others not at all.
    """

    def __init__(self, model, tokenizer, device='cpu'):
        import torch
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.torch = torch
        self.cache = {}
        # Same sigmoid calibration as safety.py: SigLIP trains with a sigmoid
        # loss, so sigmoid(scale * cos + bias) is the model's own probability.
        # Softmaxing at this scale is a hard argmax and ranks nonsense first --
        # that mistake cost a rewrite once already.
        self.scale = float(model.logit_scale.exp().detach().cpu())
        self.bias = float(model.logit_bias.detach().cpu()) if hasattr(
            model, 'logit_bias') else 0.0

    def vector(self, topic):
        if topic not in self.cache:
            with self.torch.inference_mode():
                tokens = self.tokenizer([f'a photo of {topic}']).to(self.device)
                text = self.model.encode_text(tokens)
                text = text / text.norm(dim=-1, keepdim=True)
            self.cache[topic] = text.float().cpu().numpy().astype(np.float32)[0]
        return self.cache[topic]

    def score(self, topic, vectors):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            logits = self.scale * (vectors @ self.vector(topic)) + self.bias
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        if not np.isfinite(probability).all():
            raise RuntimeError('Non-finite relevance score; refusing to gate')
        return probability


def embed_chunk(model, preprocess, chunk):
    """Preprocess and run the image tower. Runs in a thread; no async here.

    torch is imported here rather than at module scope because the rest of this
    file does the same: the crawl-only and no-embed paths never load it, and on
    those runs importing it costs seconds and hundreds of megabytes for nothing.
    """
    import torch
    import torch.nn.functional as F
    batch = torch.stack([preprocess(im) for _, im, _ in chunk])
    with torch.inference_mode():
        return F.normalize(model.encode_image(batch), dim=-1).float().numpy()


async def batches_as_fetched(client, rows, limiter, size, encode_webp):
    """Yield batches of fetched images as they land, not after all of them do.

    The old shape was `await asyncio.gather(...)` for the whole page and only
    then the embed loop, so a page cost fetch-time PLUS embed-time. Measured on
    a finished lane: 46.2 images per productive page in 18.14s = 2.55 img/s,
    against README's bench of 5.4 fetching alone and 4.8 embedding alone. That
    2.55 is the bench's own "coupled, no overlap" figure of 2.5 -- the two
    halves were taking turns.

    Streaming batches out as they complete lets the caller embed batch N while
    batches N+1.. are still downloading. Paired with running the forward pass
    in a thread (so the event loop stays free to drive the sockets), the page
    cost goes from fetch+embed to max(fetch, embed).
    """
    tasks = [asyncio.create_task(fetch_image(client, row, limiter, encode_webp=encode_webp))
             for row in rows]
    buffer = []
    try:
        for done_task in asyncio.as_completed(tasks):
            got = await done_task
            if got:
                buffer.append(got)
            if len(buffer) >= size:
                yield buffer[:size]
                buffer = buffer[size:]
        if buffer:
            yield buffer
    finally:
        # A break upstream (target reached, window closed) must not leave
        # sockets open behind us.
        for task in tasks:
            if not task.done():
                task.cancel()


# Qdrant sits behind a proxy on a busy box, and a connection can drop halfway
# through a reply ("peer closed connection without sending complete message
# body"). One such blip killed worker 45 of build 10m-0911 an hour into its
# window, and a crashed worker raises no resume flag, so it sat idle until the
# next restart. Every Qdrant call here is safe to repeat: count, retrieve and
# the checkpoint load only read, the payload index and the state collection are
# idempotent, and upserts rewrite points under deterministic ids. About two
# minutes of retrying, then the error stands, as it always did.
QDRANT_RETRY_DELAYS = (2, 4, 8, 16, 32, 60)


def transient_qdrant_error(exc):
    """A dropped connection, a timeout or a server-side failure: worth another try."""
    if isinstance(exc, (ResponseHandlingException, httpx.TransportError)):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (exc.status_code >= 500 or exc.status_code == 429)
    return False


def with_retries(call, what, sleep=time.sleep, delays=QDRANT_RETRY_DELAYS):
    """Runs `call`, retrying transient Qdrant errors; any other error raises at once."""
    for attempt, delay in enumerate(delays):
        try:
            return call()
        except Exception as exc:
            if not transient_qdrant_error(exc):
                raise
            print(f'qdrant {what} failed ({type(exc).__name__}: {str(exc)[:120]}); '
                  f'retry {attempt + 1}/{len(delays)} in {delay}s', flush=True)
            sleep(delay)
    return call()


def goal_met(size, stop_at):
    """True once the whole collection holds `stop_at` points. 0 disables it."""
    return bool(stop_at) and size >= stop_at


def collection_size(qc):
    """Points in the whole collection -- every build's, not just this one's.
    Approximate, which is fine for a stopping line: each worker overshoots it
    by at most one flush."""
    return with_retries(lambda: qc.get_collection(COLLECTION).points_count or 0, 'collection size')


def exit_code(done, target, queue, goal_reached=False):
    """75 (EX_TEMPFAIL) only when the worker stopped with jobs still queued.

    The workflow restarts a build whenever a worker exits 75, so 75 has to mean
    "paused, more to do" and nothing else. A worker whose queue is empty has
    spent every shard in its bin -- capped, exhausted or abandoned -- and is
    finished even if it indexed fewer rows than its target. The first version
    exited 75 for that too, which would have made a self-restarting build keep
    relaunching finished workers until its restart cap ran out.
    """
    if goal_reached:
        # The whole corpus is done, whatever this worker's own share says.
        return 0
    return 75 if done < target and queue else 0


def clean_url(url):
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, '', ''))


def metadata(page, end=None):
    title = page.get('title', '').removeprefix('File:')
    if end is not None and title.replace(' ', '_') >= end:
        return None
    ii = (page.get('imageinfo') or [{}])[0]
    meta = ii.get('extmetadata') or {}
    lic = _clean(meta.get('LicenseShortName', {}).get('value', ''))
    if not license_ok(lic) or not ii.get('thumburl'):
        return None
    if ii.get('mime') not in ('image/jpeg', 'image/png', 'image/webp'):
        return None
    w, h = ii.get('width', 0), ii.get('height', 0)
    if min(w, h) < 320 or max(w, h) / min(w, h) > 8:
        return None
    return {
        'image_id': f"commons:{page['pageid']}", 'title': clean_text(title)[:300],
        'creator': _clean(meta.get('Artist', {}).get('value', ''))[:180],
        'license': lic, 'license_class': license_class(lic),
        'license_url': meta.get('LicenseUrl', {}).get('value', ''),
        'source_url': f"https://commons.wikimedia.org/?curid={page['pageid']}",
        'full_url': clean_url(ii['url']), 'thumb_origin': clean_url(ii['thumburl']),
        'width': w, 'height': h, 'mime': ii['mime'], 'sha1': ii.get('sha1', ''),
        'description': _clean(meta.get('ImageDescription', {}).get('value', ''))[:400],
        'tags': _clean(meta.get('Categories', {}).get('value', '')).replace('|', ', ')[:300],
    }


# Commons sometimes answers 200 with an error body instead of an HTTP error,
# so request()'s 429/5xx retry never sees it. internal_api_error_* means a
# backend hiccup -- internal_api_error_DBConnectionError killed worker 37 of
# build 10m-0911 2.5h into its window. Worth a wait and a retry, not a dead
# worker. Bounded: past COMMONS_MAX_BLIPS straight blips the error stands, so
# a truly broken query still dies loudly instead of burning its whole window.
COMMONS_MAX_BLIPS = 10


def transient_commons_error(code):
    """True when a Commons error-JSON code is a backend blip worth retrying."""
    if not code:
        return False
    if code in ('maxlag', 'ratelimited', 'readonly'):
        return True
    return code.startswith('internal_api_error')


def drop_label(job):
    """Short shard name for the drop line. Range shards carry start/end (which
    wins when present); topic shards carry topic/lic/band."""
    return job.get('start', {k: job.get(k) for k in ('topic', 'lic', 'band')})


async def request(client, url, **kwargs):
    for attempt in range(6):
        try:
            r = await client.get(url, timeout=60, **kwargs)
            if r.status_code == 429 or r.status_code >= 500:
                delay = min(float(r.headers.get('Retry-After', 2 ** (attempt + 1))), 120)
                await asyncio.sleep(delay + random.random())
                continue
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 5:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError('Remote service remained unavailable after six attempts')


# Broken code, not broken data. These must never be mistaken for a bad image.
BUG_TYPES = (NameError, AttributeError, ImportError)


def upload_batch(chunk):
    """Put a batch of derivatives in the bucket, in parallel, and never fail.

    boto3 is synchronous, so uploading 16 images one at a time would add well
    over a second per batch to a crawl already bound by politeness. A small
    thread pool overlaps them.

    An upload that fails returns an empty URL rather than raising. The row is
    still worth indexing: `result_from` falls back to the proxy when `cdn` is
    absent, so a bucket outage degrades serving rather than losing the crawl.
    """
    if not storage.enabled():
        return [''] * len(chunk)

    def one(entry):
        row, _, webp = entry
        if not webp:
            return ''
        try:
            return storage.put(row['image_id'], webp)
        except Exception as exc:
            print(f"upload skipped {row.get('image_id')}: {type(exc).__name__}", flush=True)
            return ''

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, chunk))


class HostLimiter:
    """One semaphore per host, and a tighter one for the hosts that throttle.

    `ingest.py` has had this since the 429 wall was first hit; the cloud
    crawler never got it, and kept a single global semaphore. That is why the
    500k run's per-worker error rates varied from 0 to 56 an hour with no
    pattern in time: it is not time, it is which alphabet ranges a worker drew
    and therefore how much of its traffic landed on one host.

    Measured on 2,500 real origins from the live index:

        thumb.wikimedia.org    1,669 fetched,   0 failures
        upload.wikimedia.org     392 fetched, 152 failures (39%)

    Those same URLs answer 200 when asked one at a time, and 429 for 16 of 30
    at concurrency 6. MediaWiki serves pre-rendered thumbnails generously and
    original files stingily -- reasonably, since originals cost it far more.

    About 16% of the corpus lands on the original: `iiurlwidth=800` returns
    `thumburl == url` for anything already narrower than 800px, so there is no
    thumbnail to hand back. At 10M that is 1.6M requests to the strictest host,
    which is why this cannot be left as it was.
    """
    DEFAULT = int(os.getenv('FETCH_CONCURRENCY', '6'))
    TIGHT = {'upload.wikimedia.org': int(os.getenv('ORIGIN_CONCURRENCY', '2'))}

    def __init__(self, per_host=None):
        self.per_host = per_host or self.DEFAULT
        self._sems = {}

    def limit_for(self, url):
        return self.TIGHT.get(urlparse(url).netloc, self.per_host)

    def get(self, url):
        host = urlparse(url).netloc or 'unknown'
        if host not in self._sems:
            self._sems[host] = asyncio.Semaphore(self.TIGHT.get(host, self.per_host))
        return self._sems[host]


async def fetch_image(client, row, limiter, encode_webp=True):
    url = row['thumb_origin']
    content = None
    for attempt in range(5):
        try:
            # API-generated thumbnail only: never construct Wikimedia paths.
            async with limiter.get(url):
                r = await client.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                # Backing off INSIDE the semaphore holds a slot on the very
                # host that just asked for less, and blocks every other image
                # queued behind it. Wait outside, then queue again.
                delay = min(float(r.headers.get('Retry-After', 2 ** (attempt + 1))), 120)
                await asyncio.sleep(delay + random.random())
                continue
            r.raise_for_status()
            content = r.content
            break
        except BUG_TYPES:
            raise
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 4:
                break
            await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            print(f"fetch skipped {row.get('image_id')}: {type(exc).__name__}", flush=True)
            return None

    if content is None:
        print(f"fetch skipped {row.get('image_id')}: unavailable after retries", flush=True)
        return None
    if len(content) > 15_000_000:
        return None

    # Decoding is CPU work and holds no network slot.
    try:
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert('RGBA' if 'A' in source.getbands() else 'RGB')
            image.thumbnail((384, 384), Image.Resampling.LANCZOS)
            # Encoding costs real CPU on a 4-vCPU runner and the standalone
            # build stores no derivative at all -- the browser is served
            # Wikimedia's own thumbnail. Skip it when nothing will keep it.
            webp = b''
            if encode_webp:
                buf = io.BytesIO()
                image.save(buf, 'WEBP', quality=80, method=4)
                webp = buf.getvalue()
            if image.mode == 'RGBA':
                rgb = Image.new('RGB', image.size, 'white')
                rgb.paste(image, mask=image.getchannel('A'))
            else:
                rgb = image.convert('RGB')
            return row, rgb, webp
    except BUG_TYPES:
        # A mistake inside this function would otherwise present as every
        # image being corrupt -- the failure mode that once made discovery
        # return nothing at all while looking healthy. Crash loudly instead.
        raise
    except Exception as exc:
        # One malformed file must never take down a worker that has been
        # crawling for hours; that is exactly how worker 19 died on a single
        # PNG. Pillow raises whatever the codec raises -- ValueError from the
        # oversized-text-chunk guard, zlib.error, struct.error, EOFError on a
        # truncated stream -- so the catch is broad and the type is logged.
        print(f"fetch skipped {row.get('image_id')}: {type(exc).__name__}", flush=True)
        return None


class Archive:
    def __init__(self, root, hf, repo):
        self.root, self.hf, self.repo = root, hf, repo
        self.pending = []

    def add(self, row, webp):
        self.pending.append((row, webp))

    def build(self):
        """Write the tarball and return a commit operation, without committing.

        Hugging Face allows 128 repository commits per hour. Committing the
        archive and the checkpoint separately means two commits per save per
        worker; at 20 workers that is ~300/hour and the run dies partway
        through. Returning the operation lets the caller put both in one
        commit.
        """
        if not self.pending:
            return None
        digest = hashlib.sha256(''.join(r['image_id'] for r, _ in self.pending).encode()).hexdigest()[:20]
        path = self.root / f'{digest}.tar'
        with tarfile.open(path, 'w') as tar:
            for row, webp in self.pending:
                key = point_id(row['image_id'])
                for name, data in [(key + '.webp', webp), (key + '.json', json.dumps(row).encode())]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        data = path.read_bytes()
        path.unlink()
        self.pending.clear()
        return CommitOperationAdd(path_in_repo=f'webdataset/{path.name}',
                                  path_or_fileobj=data)

    def to_bucket(self, key):
        """Write the pending images as one tar in the bucket. Returns the key.

        Crawl-only cannot use the HuggingFace path above: at 20 workers
        flushing every 500 rows that is ~780 commits/hour against a 128/hour
        limit. The bucket has no commit ceiling.

        The point of the tar is the READ side. Uploading a derivative per
        image is free -- writes are Class A -- but reading them back one at a
        time costs one Class B call per image, and a full embedding pass over
        10M images is then 10M calls. Backblaze's free tier allows 2,500 a
        day, and even paid this is 10M round trips of 22 KB each. Tarred at
        500, the same pass is 20,000 sequential reads of 11 MB. The per-image
        objects still exist for serving; this is what the GPU reads.
        """
        if not self.pending:
            return None
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as tar:
            for row, webp in self.pending:
                name = point_id(row['image_id'])
                for member, data in [(name + '.webp', webp),
                                     (name + '.json', json.dumps(row).encode())]:
                    info = tarfile.TarInfo(member)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        self.pending.clear()
        storage.client().put_object(Bucket=storage.BUCKET, Key=key,
                                    Body=buf.getvalue(),
                                    ContentType='application/x-tar')
        return key


async def run(args):
    # Crawling and embedding are separate phases. Coupled on a CI runner they
    # ran at 1-2 img/s per worker, because every image waited on SigLIP on a
    # shared CPU. Split, the crawl is bound only by how fast the sources will
    # politely serve -- measured 5.4 img/s -- and embedding runs later at GPU
    # speed over a bucket it can re-read as often as it likes. Changing model
    # later becomes a GPU pass, not another crawl of the internet.
    if not args.no_embed:
        import torch
        import torch.nn.functional as F
        import open_clip
        torch.set_num_threads(args.threads)
    # Crawl-only talks to exactly two things: Commons and the bucket. Its
    # checkpoint lives in the bucket too, so this phase needs no HuggingFace
    # token, no Qdrant credentials, and nothing that can rate-limit a commit.
    hf = repo = None
    if not args.no_embed and not args.no_archive:
        hf = HfApi(token=os.environ['HF_TOKEN'])
        repo = os.environ['HF_REPO']
    if args.no_archive:
        with_retries(checkpoint.ensure, 'checkpoint ensure')
    # No cloud_inference: sparse vectors are built here, so this same code
    # works against a self-hosted instance, which has no inference service.
    qc = None
    if not args.no_embed:
        # .get, not [..]: the 10M build targets a self-hosted Qdrant, which
        # may have no API key at all. checkpoint.py and embed_manifest.py
        # already treat it as optional; this was the one place that did not.
        qc = QdrantClient(url=os.environ['QDRANT_URL'],
                          api_key=os.environ.get('QDRANT_API_KEY'), timeout=120)
        for field, schema in [('build_id', models.PayloadSchemaType.KEYWORD),
                             ('worker', models.PayloadSchemaType.INTEGER),
                             ('safety', models.PayloadSchemaType.FLOAT)]:
            with_retries(lambda f=field, s=schema: qc.create_payload_index(
                COLLECTION, f, field_schema=s, wait=True), 'create_payload_index')
    checkpoint_key = f'checkpoints/{args.build}-{args.workers}-{args.worker}.json'
    categories = [c.strip() for c in (args.categories or '').split(',') if c.strip()]
    topics = []
    if args.topics:
        topics = [t.strip() for t in pathlib.Path(args.topics).read_text().splitlines()
                  if t.strip() and not t.strip().startswith('#')]
    licences = [l.strip() for l in (args.licences or '').split(',') if l.strip()] or CLEAN_LICENCES
    # Measured Commons hit counts, if they sit beside the topic list. Present
    # means bin-packed lanes; absent means the old striding, and the old tail.
    topic_hits = {}
    if args.topics:
        sidecar = pathlib.Path(args.topics).with_suffix('.hits.json')
        if sidecar.exists():
            topic_hits = {k: int(v) for k, v in json.loads(sidecar.read_text()).items()}
            print(f'worker {args.worker}: balancing lanes from {len(topic_hits)} '
                  f'measured topic sizes', flush=True)

    def read_checkpoint():
        if args.no_archive:
            return with_retries(lambda: checkpoint.load(checkpoint_key), 'checkpoint load')
        if args.no_embed:
            from botocore.exceptions import ClientError
            try:
                body = storage.client().get_object(Bucket=storage.BUCKET, Key=checkpoint_key)
                return json.loads(body['Body'].read())
            except ClientError as exc:
                # Absent means a fresh build; anything else must NOT be read as
                # one. A cap-exceeded 403 looks exactly like a missing object
                # to `except Exception`, and the worker would silently restart
                # its range from zero -- reusing manifest_seq and overwriting
                # the shards of the run it was supposed to resume.
                code = exc.response.get('Error', {}).get('Code', '')
                if code in ('NoSuchKey', 'NoSuchBucket', '404'):
                    return None
                # The read failed for a reason other than absence -- a download
                # cap, most likely, since those are the calls that get capped.
                # Listing is a different transaction class, so ask it instead:
                # a build with no manifests has nothing to overwrite and is
                # safe to start, whatever the checkpoint read did.
                print(f'checkpoint unreadable ({code}); checking for prior '
                      f'output by listing', flush=True)
                pages = storage.client().get_paginator('list_objects_v2')
                for page in pages.paginate(Bucket=storage.BUCKET,
                                           Prefix=f'manifest/{args.build}/'):
                    if page.get('Contents'):
                        raise RuntimeError(
                            f'Cannot read checkpoint {checkpoint_key} ({code}) and '
                            f'build {args.build!r} already has manifests. '
                            f'Refusing to start: resuming blind would reuse '
                            f'manifest_seq and overwrite existing shards.') from exc
                print(f'no manifests under {args.build!r}: safe fresh start', flush=True)
                return None
        try:
            return json.loads(Path(hf_hub_download(
                repo, checkpoint_key, repo_type='dataset',
                token=os.environ['HF_TOKEN'])).read_text())
        except EntryNotFoundError:
            return None
    if topics:
        queue = deque(search_jobs(topics, licences, args.worker, args.workers, topic_hits))
    elif categories:
        queue = deque(category_jobs(categories, args.worker, args.workers))
    else:
        queue = deque({'start': lo, 'end': hi, 'continue': {}}
                      for lo, hi in ranges(args.worker, args.workers))
    done = 0
    resume_seq = 0
    state = read_checkpoint()
    if state:
        queue = deque(state['queue'])
        # Crawl-only has no Qdrant to count, so progress rides in the
        # checkpoint that already carries the cursor.
        done = state.get('done', 0)
        # And so does the shard counter. Restarting it at zero made a resumed
        # worker overwrite its own earlier shards: 370 rows of manifest were
        # replaced by 1,791, leaving 370 derivatives in the bucket that nothing
        # referenced -- paid for, and invisible to the embedding pass.
        resume_seq = state.get('manifest_seq', 0)
    if qc is not None:
        flt = models.Filter(must=[models.FieldCondition(key='build_id', match=models.MatchValue(value=args.build)),
                                  models.FieldCondition(key='worker', match=models.MatchValue(value=args.worker))])
        done = with_retries(lambda: qc.count(COLLECTION, count_filter=flt, exact=True).count, 'count')
    if done >= args.target:
        print(f'already complete: {done}/{args.target}', flush=True)
        return
    if qc is not None and args.stop_at_total:
        size = collection_size(qc)
        if goal_met(size, args.stop_at_total):
            print(f'GOAL: the collection already holds {size:,} >= {args.stop_at_total:,}; '
                  f'nothing to do', flush=True)
            return
    # Jobs queued is the worker's whole future: 0 here means it will skip
    # everything and exit green within minutes (inherited an exhausted
    # checkpoint, or a category with nothing in its shards). Loud now so a
    # silent skip never again looks like a stall.
    print(f'worker {args.worker}: {len(queue)} jobs queued, {done}/{args.target} indexed', flush=True)
    if args.no_embed:
        if not storage.enabled():
            raise SystemExit('--no-embed needs a bucket: set STORAGE_BACKEND=s3 and the S3_* vars')
        print(f'crawl only, worker {args.worker}, {done}/{args.target} already written', flush=True)
        model = preprocess = None
        scorer = None
        relevance = None
    else:
        print(f'loading SigLIP, worker {args.worker}, {done}/{args.target} already uploaded', flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model.eval()
        # The text tower is already loaded, so the safety prompts cost one
        # forward pass at startup and a small matmul per batch thereafter.
        tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
        scorer = safety.Scorer(model, tokenizer)
        relevance = Relevance(model, tokenizer) if args.relevance > 0 else None
    root = Path('worker-data')
    root.mkdir(exist_ok=True)
    archive = Archive(root, hf, repo)
    pending_points = []
    manifest_rows = []
    # Set by save() once the collection reaches --stop-at-total.
    goal = {'reached': False}
    manifest_seq = resume_seq
    staged_ids = set()
    start_time, pages, failed, empty = time.monotonic(), 0, 0, 0
    # Fetch concurrency, per worker. Measured on a 50-image page cycle:
    # fetching was 6.7s of 15.7s at concurrency 3 -- the single largest block,
    # and self-inflicted rather than imposed by any origin.
    #
    # Each runner has its own IP, so 6 is 6 per address, not 120. Stopping at
    # 6 rather than 8 is deliberate: 20 workers x 4.3 img/s = 86/s already
    # exceeds the 82.6/s Qdrant upsert ceiling measured against the free
    # cluster, so anything higher just moves the queue from fetch to upsert
    # while putting more load on a donated service.
    limiter = HostLimiter()

    def save():
        nonlocal manifest_seq
        # Data first, so the durable cursor never runs ahead of what it points at.
        ops = []
        if args.no_embed:
            # The manifest is the handoff. Each line is a finished row whose
            # derivative is already in the bucket; the embedding pass reads
            # these, fetches the images by key, and writes the vectors.
            if manifest_rows:
                # Tar first, then the manifest that names it, then the
                # checkpoint. A crash can therefore orphan a shard, which
                # costs storage and nothing else; it can never leave a
                # manifest row pointing at a tar that was never written.
                shard = f'shard/{args.build}/{args.worker:02d}-{manifest_seq:05d}.tar'
                # Stamped before the tar is built so the copy of the row
                # inside the archive matches the copy in the manifest.
                for r in manifest_rows:
                    r['shard'] = shard
                archive.to_bucket(shard)
                body = ('\n'.join(json.dumps(r) for r in manifest_rows) + '\n').encode()
                key = f'manifest/{args.build}/{args.worker:02d}-{manifest_seq:05d}.jsonl'
                storage.client().put_object(Bucket=storage.BUCKET, Key=key, Body=body,
                                            ContentType='application/x-ndjson')
                print(f'manifest {key}: {len(manifest_rows)} rows', flush=True)
                manifest_rows.clear()
                manifest_seq += 1
        else:
            # Standalone keeps nothing: no tarball, so no HuggingFace commit
            # and no 128/hour ceiling to pace against.
            if not args.no_archive:
                archive_op = archive.build()
                if archive_op:
                    ops.append(archive_op)
            for i in range(0, len(pending_points), 128):
                batch = pending_points[i:i + 128]
                with_retries(lambda: qc.upsert(COLLECTION, points=batch, wait=True), 'upsert')
            pending_points.clear()
        staged_ids.clear()
        state = json.dumps({'queue': list(queue), 'uploaded': done, 'done': done,
                            'manifest_seq': manifest_seq, 'pages': pages}).encode()
        if args.no_archive:
            with_retries(lambda: checkpoint.save(checkpoint_key, json.loads(state)), 'checkpoint save')
            # After the save, so a worker that stops here leaves nothing
            # unsaved. Once per flush: cheap, and at most one flush of
            # overshoot per worker.
            if qc is not None and args.stop_at_total:
                goal['reached'] = goal_met(collection_size(qc), args.stop_at_total)
            return
        if args.no_embed:
            storage.client().put_object(Bucket=storage.BUCKET, Key=checkpoint_key,
                                        Body=state, ContentType='application/json')
            return
        ops.append(CommitOperationAdd(path_in_repo=checkpoint_key, path_or_fileobj=state))

        # A commit rate limit should pause a worker, not kill it. HF answers
        # 429 with "retry in about an hour", so back off long and hard rather
        # than losing the range this worker has already crawled.
        for attempt in range(6):
            try:
                hf.create_commit(repo_id=repo, repo_type='dataset', operations=ops,
                                 commit_message=f'Worker {args.worker}: {done} images')
                return
            except HfHubHTTPError as exc:
                if getattr(exc.response, 'status_code', None) != 429 or attempt == 5:
                    raise
                wait = min(900, 60 * 2 ** attempt)
                print(f'worker {args.worker}: HF 429, sleeping {wait}s', flush=True)
                time.sleep(wait)

    async with httpx.AsyncClient(headers={'User-Agent': UA}, follow_redirects=True) as client:
        blips = 0  # consecutive transient Commons error-JSONs; reset on any good page
        while (queue and done < args.target and not goal['reached']
               and time.monotonic() - start_time < args.max_seconds):
            job = queue[0]
            if 'topic' in job:
                params = params_for_search(job['topic'], job['lic'], job['band'], job['continue'])
                end = None  # relevance order; there is no title bound to enforce
            elif 'cat' in job:
                params = params_for_category(job['cat'], job['start'], job['end'], job['continue'])
                end = None  # the shard bound is a sortkey, enforced server-side
            else:
                params = params_for(job['start'], job['end'], job['continue'])
                end = job['end']
            r = await request(client, COMMONS, params=params)
            data = r.json()
            if data.get('error'):
                code = data['error'].get('code')
                if code in ('maxlag', 'ratelimited'):
                    await asyncio.sleep(30)
                    continue
                if transient_commons_error(code):
                    # Backend blip, not a bad query: wait it out, but not
                    # forever -- past the cap the error stands and the worker
                    # dies loudly instead of burning its window one minute at
                    # a time.
                    blips += 1
                    if blips > COMMONS_MAX_BLIPS:
                        raise RuntimeError(f"Commons API error: {code}")
                    print(f'worker {args.worker}: Commons blip {code} '
                          f'({blips}/{COMMONS_MAX_BLIPS}), sleeping 60s', flush=True)
                    await asyncio.sleep(60)
                    continue
                if code == 'cirrussearch-offset-too-large':
                    # The search API refuses offsets past 10,000. That is the
                    # shard exhausted, not a failure: licence and width shards
                    # exist so the clean pool is reachable across several of
                    # these, and page-id dedupe absorbs any overlap between
                    # them. Bank what it indexed and take the next shard.
                    print(f'worker {args.worker}: shard depth reached for '
                          f'{job.get("topic")!r} / {job.get("lic")!r}', flush=True)
                    queue.popleft()
                    blips = 0  # fresh job, fresh streak
                    save()
                    continue
                if code == 'urlparamnormal':
                    # One unrepresentable cursor should cost a job, not a
                    # worker. Drop this job, keep whatever it already indexed,
                    # and move to the next one. Job shape differs by mode
                    # (ranges have start/end, topic shards have topic/lic/band).
                    label = drop_label(job)
                    print(f'worker {args.worker}: dropping job '
                          f'{label!r} after {code}', flush=True)
                    queue.popleft()
                    blips = 0  # fresh job, fresh streak
                    save()
                    continue
                raise RuntimeError(f"Commons API error: {code}")
            blips = 0  # a clean page clears the blip streak
            rows = [row for p in data.get('query', {}).get('pages', []) if (row := metadata(p, end))]
            rows = list({row['image_id']: row for row in rows}.values())
            if rows:
                # Crawl-only has nothing to ask: the embed pass skips rows
                # already in Qdrant before it spends a GPU on them, so the
                # check belongs there rather than in a phase that runs hours
                # earlier and holds no database credentials.
                seen = set()
                if qc is not None:
                    wanted_ids = [point_id(r['image_id']) for r in rows]
                    seen = {str(p.id) for p in with_retries(lambda: qc.retrieve(
                        COLLECTION, ids=wanted_ids, with_payload=False, with_vectors=False), 'retrieve')}
                rows = [r for r in rows if point_id(r['image_id']) not in seen | staged_ids][:args.target - done]
            produced = 0
            fetched = 0
            async for chunk in batches_as_fetched(client, rows, limiter, args.batch,
                                                  encode_webp=not args.no_archive):
                fetched += len(chunk)
                if args.no_embed:
                    for (row, _, _) in chunk:
                        row.update(build_id=args.build, worker=args.worker)
                    for row, url in zip((r for r, _, _ in chunk), upload_batch(chunk)):
                        if url:
                            row['cdn'] = url
                    # Only rows whose derivative actually reached the bucket:
                    # the embedding pass has no other way to read the image.
                    # Each kept image also goes into the shard tar, which is
                    # what phase 2 actually reads -- see Archive.to_bucket.
                    kept = []
                    for row, _, webp in chunk:
                        if row.get('cdn'):
                            kept.append(row)
                            archive.add(row, webp)
                    manifest_rows.extend(kept)
                    staged_ids.update(point_id(r['image_id']) for r in kept)
                    done += len(kept)
                    produced += len(kept)
                    continue
                # preprocess + forward in a worker thread: torch releases the
                # GIL, so the event loop keeps draining the sockets for the
                # batches still in flight. Doing it inline blocks the loop and
                # the overlap above buys nothing.
                vectors = await asyncio.to_thread(embed_chunk, model, preprocess, chunk)
                if not np.isfinite(vectors).all() or vectors.shape[1] != 768:
                    raise RuntimeError('Invalid image embeddings; refusing upload')
                points = []
                for (row, _, _) in chunk:
                    row.update(build_id=args.build, worker=args.worker)
                # The derivative goes to our own bucket, and the row carries the
                # URL the browser will load it from. Without this the API falls
                # back to a free public resizing proxy -- which is what produced
                # every serving problem the prototype hit: broken images, slow
                # museum loads, copy failures, and finally an IP block when we
                # tried to warm it. It is the one architectural change in the
                # 10M plan, and it costs $1.21/month at the 21.2 KB measured.
                if not args.no_archive:
                    for row, url in zip((r for r, _, _ in chunk), upload_batch(chunk)):
                        if url:
                            row['cdn'] = url
                # One pass over the tokenizer for the batch, not one per row.
                sparse_vectors = sparse.documents(search_text(row) for row, _, _ in chunk)
                # The image vector is already here, so scoring it against the
                # cached safety prompts is one small matmul. Deferring it would
                # mean a second pass over a corpus this build does not keep.
                marks = scorer.payload(vectors) if scorer else [{}] * len(chunk)
                # The relevance gate. Search order is relevance order, so a
                # topic's usable depth is wherever its score falls off -- and
                # that is measured per topic at crawl time, not guessed. Ranged
                # from ~1,000 usable rows for `coffee` to past 10,000 for
                # `sunset` on identical hit counts, which is exactly why a
                # fixed harvest depth cannot work.
                relevant = np.ones(len(chunk), dtype=bool)
                if relevance is not None and 'topic' in job:
                    scores = relevance.score(job['topic'], vectors)
                    # A RUNNING HIGH-WATER MARK, not the opening batch.
                    # Calibration killed the opening-batch design: `sunset`
                    # opens at .0111 and then sits at .07-.09 for the next
                    # 9,000 ranks -- 7-9x its own first page -- so anchoring to
                    # page one set the bar far too low, while `autumn forest`
                    # opens at .2446 and set it far too high. The best batch
                    # seen so far is what the topic is actually capable of.
                    batch_median = float(np.median(scores))
                    job['ref'] = max(job.get('ref') or 0.0, batch_median)
                    # TWO thresholds, deliberately, because eye review showed
                    # they are different jobs. The per-image floor exists only
                    # to drop what is not the topic at all -- a coat of arms
                    # and an anatomy plate both score 0.0000 against "a photo
                    # of sunset" -- so it sits low. Cutting per-image at the
                    # STOP fraction instead put the knife inside a noise band:
                    # for `coffee` it kept shopfronts and a handful of gravel
                    # while dropping a cafe interior and a flat-lay with a cup,
                    # which are the same population either side of the line.
                    # The shard-level median is what actually tracks decay.
                    floor = args.relevance * max(job['ref'], 1e-6)
                    relevant = scores >= floor
                    if batch_median < args.relevance_stop * max(job['ref'], 1e-6):
                        job['decayed'] = job.get('decayed', 0) + 1
                    else:
                        job['decayed'] = 0
                for keep, (row, _, webp), vec, bm25, mark in zip(
                        relevant, chunk, vectors, sparse_vectors, marks):
                    if not keep:
                        continue
                    row.update(mark)
                    points.append(models.PointStruct(
                        id=point_id(row['image_id']),
                        vector={'image': vec.tolist(), 'bm25': bm25},
                        payload=row,
                    ))
                    if not args.no_archive:
                        archive.add(row, webp)
                pending_points.extend(points)
                staged_ids.update(p.id for p in points)
                done += len(points)
                produced += len(points)
            failed += len(rows) - fetched
            # A page that indexes nothing still costs an API round trip and a
            # Qdrant existence check. Measured on a finished lane: 2,014 of
            # 2,555 pages produced zero rows, 38 minutes -- 19% of that lane's
            # life -- because the shard had already been crawled. They arrive in
            # runs, since both sortkey and relevance order cluster, so a run of
            # them means the shard is spent rather than momentarily thin.
            # Depth cap. Search hands results back best-first, so taking the
            # top N of a shard IS a quality filter -- and unlike the relevance
            # gate it costs nothing, because the tail is never fetched rather
            # than fetched, embedded and then discarded. `coffee` and `sunset`
            # both hold up for the first few hundred; only `coffee` collapses
            # after that, and this stops both before it matters.
            job['kept'] = job.get('kept', 0) + produced
            if args.shard_depth and job['kept'] >= args.shard_depth:
                print(f'worker {args.worker}: {job.get("topic", job.get("cat"))!r} '
                      f'hit its {args.shard_depth}-row cap', flush=True)
                queue.popleft()
                save()
                continue
            job['barren'] = 0 if produced else job.get('barren', 0) + 1
            if job['barren'] >= args.barren_patience:
                print(f'worker {args.worker}: nothing new in {job["barren"]} pages, '
                      f'dropping shard', flush=True)
                queue.popleft()
                save()
                continue
            queue.popleft()
            continuation = data.get('continue')
            if job.get('decayed', 0) >= args.relevance_patience:
                # Spent: this topic has stopped returning itself. Bank the rows
                # and take the next shard rather than paying for the tail.
                print(f'worker {args.worker}: {job.get("topic")!r} exhausted at '
                      f'{job["continue"].get("gsroffset", 0)}', flush=True)
                save()
                continue
            if continuation:
                if continuation == job['continue']:
                    raise RuntimeError('Commons repeated a continuation cursor')
                job['continue'] = continuation
                queue.append(job)
            pages += 1
            # Ten pages that found rows and indexed none means something is
            # broken upstream. It must count what the page PRODUCED, not what
            # is left in `loaded`: crawl-only empties that list by design, so
            # keying the check on it stopped every worker after ten pages.
            empty = empty + 1 if rows and not produced else 0
            print(json.dumps({'worker': args.worker, 'uploaded': done, 'target': args.target,
                              'pages': pages, 'failed': failed, 'images_per_second': round(done / max(time.monotonic()-start_time, 1), 2)}), flush=True)
            # Flush cadence follows the cost of save(), which differs by mode:
            # coupled+archive saves through a HuggingFace commit against a
            # 128/hour ceiling (4000 keeps twenty workers near 36/hour);
            # crawl-only and no-archive save to a bucket / Qdrant with no
            # ceiling, so they flush often -- otherwise a dead worker loses
            # hours and, for no-archive, the indexed count never moves
            # mid-run because pending_points only upsert inside save().
            #
            # It also cannot key off `archive.pending` in the modes that
            # never fill it: crawl-only never adds to the archive, and
            # no-archive skips add, so the flush would never fire at all.
            # The pending thing is the manifest rows, or the points.
            if pending_count(args, manifest_rows, archive, pending_points) >= flush_every(args):
                save()
            if empty >= 10:
                save()
                raise RuntimeError('Ten batches failed to fetch: stopping instead of wasting the crawl')
            # Pacing for the Commons API. maxlag=5 already makes workers back
            # off when replication lag rises, which is the protection that
            # actually matters; this is a second, softer brake.
            await asyncio.sleep(0.3)
        save()
    if goal['reached']:
        print(f'GOAL: the collection reached {args.stop_at_total:,}; this worker stops at '
              f'{done}/{args.target} of its own share. Nothing to resume.', flush=True)
    if exit_code(done, args.target, queue, goal['reached']) == 75:
        # Not a failure. The worker ran out of wall clock before its quota and
        # has saved its cursor; the next run resumes from there. A 10M build is
        # six runs of twenty workers and MOST of those jobs end this way, so
        # raising here painted 120 jobs red and would have buried the ones that
        # broke for real. 75 is EX_TEMPFAIL -- "try again later" -- and the
        # workflow treats it as a note rather than an error.
        print(f'INCOMPLETE: {done}/{args.target} indexed, {len(queue)} jobs left, '
              f'cursor saved. The build restarts itself to resume.', flush=True)
        raise SystemExit(75)
    if done < args.target and not goal['reached']:
        print(f'EXHAUSTED: {done}/{args.target} indexed -- every job in this '
              f"worker's share is spent, so there is nothing to resume.", flush=True)
    if args.no_embed:
        print(f'COMPLETE: {done} images in the bucket, manifest written. '
              f'Run embed_manifest.py --build {args.build} on a GPU next.', flush=True)
    else:
        print(f'COMPLETE: {done} images embedded, archived, and searchable in Qdrant', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--build', default='commons-v1')
    p.add_argument('--topics', default='',
                   help='path to a topic list (one search string per line, # for '
                        'comments) for a relevance-ordered generator=search '
                        'harvest, sharded by licence and width band. Takes '
                        'precedence over --categories.')
    p.add_argument('--licences', default='',
                   help='comma-separated Commons licence categories to shard a '
                        '--topics harvest across. Defaults to the commercially '
                        'clean set, which keeps share-alike out of a corpus the '
                        'app hides by default.')
    p.add_argument('--relevance', type=float, default=0.0,
                   help='per-image floor, as a fraction of the running best '
                        'batch median for that shard. 0 disables the gate '
                        'entirely. Low on purpose: its job is dropping what is '
                        'not the topic at all, not ranking within it. Relative '
                        'rather than absolute because the SigLIP score for '
                        '"clearly relevant" varies by an order of magnitude '
                        'between prompts -- .0111 for sunset against .2446 for '
                        'autumn forest, measured.')
    p.add_argument('--relevance-stop', type=float, default=0.0,
                   help='a shard is decaying when its batch median falls below '
                        'this fraction of its running best. This is the number '
                        'that decides how deep a topic is crawled; measured, '
                        'it stops `coffee` around rank 3,000 and never stops '
                        '`sunset`.')
    p.add_argument('--relevance-patience', type=int, default=3,
                   help='consecutive pages below the floor before a shard is '
                        'treated as spent. Three, not two: `autumn forest` dips '
                        'below its floor at ranks 1500-2000 and recovers to '
                        '0.89x and 0.93x of reference at 4000 and 7000, so a '
                        'two-page fuse throws that away.')
    p.add_argument('--categories', default='',
                   help='comma-separated Commons categories for a flat '
                        'generator=categorymembers crawl (the Quality/Featured/'
                        'Valued layer). Empty means the default filename-range '
                        'allimages walk. Sharded by sortkey prefix across workers.')
    p.add_argument('--shard-depth', type=int, default=1600,
                   help='stop a (topic, licence, width) shard after this many '
                        'indexed rows. Replaces the relevance gate as the depth '
                        'control: search is relevance-ordered, so a cap keeps '
                        'the good part of every topic without paying to fetch '
                        'and embed the tail. 1600, not 800: measured per shard '
                        'across 40 topics in 8 hit-count strata, 800 held a '
                        'clean-only build to ~8.0M -- capped, not short of '
                        'supply, with ~193M clean in the list -- and 1600 '
                        'reaches ~11.9M. 0 disables the cap.')
    p.add_argument('--barren-patience', type=int, default=25,
                   help='consecutive pages indexing nothing before a shard is '
                        'abandoned. Zero-yield pages are cheap (0.93s measured) '
                        'but numerous; they were 19%% of one lane\'s wall clock.')
    # 8, not 16. The embed batch is now also the pipeline's fill unit: nothing
    # can be embedded until a whole batch has landed, so a big batch idles the
    # CPU at the start of every page. Simulated at the measured rates (fetch
    # 5.4 img/s, embed 4.8): batch 32 -> 1.17x, 16 -> 1.44x, 8 -> 1.63x, and
    # below 8 it flattens. Re-check on a runner; the model here assumes embed
    # time is linear in batch size, which real torch is not, quite.
    p.add_argument('--batch', type=int, default=8)
    # Measured on a GitHub runner (4 vCPU, AMD EPYC 7763): 2 threads
    # gave 4.8 img/s through SigLIP and 4 gave 4.2. More threads than
    # the forward pass can use just adds contention.
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-seconds', type=int, default=16200)
    p.add_argument('--stop-at-total', type=int, default=0,
                   help='stop every worker once the whole collection holds this '
                        'many points, whatever its own --target says, and exit 0 '
                        'so the build does not restart. 0 disables it. Lets lanes '
                        'run past a per-lane quota without overshooting the '
                        'corpus goal: a quota let fast lanes stop early and the '
                        'slow ones set the finish.')
    p.add_argument('--no-archive', action='store_true',
                   help='embed on this runner and keep nothing: no bucket, no '
                        'HuggingFace, no derivative stored anywhere. The browser '
                        'is served Wikimedia\'s own thumbnail URL, which every '
                        'row already carries. Checkpoints go to Qdrant. This is '
                        'the 10M build: it needs no object storage, so it needs '
                        'no payment method. The cost is that changing embedding '
                        'model means crawling again rather than re-reading a '
                        'bucket.')
    p.add_argument('--no-embed', action='store_true',
                   help='crawl only: derivatives to the bucket, metadata to a '
                        'manifest, no SigLIP and no Qdrant. Embedding then runs '
                        'separately on a GPU reading from that bucket.')
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers or args.target < 1:
        p.error('invalid worker/target configuration')
    if args.no_embed and args.no_archive:
        # --no-embed writes derivatives for a later GPU pass to read;
        # --no-archive keeps no derivative at all. Together they would crawl
        # the internet and throw every image away.
        p.error('--no-embed and --no-archive are contradictory: the first '
                'stores images for a GPU pass to read later, the second '
                'stores nothing')
    asyncio.run(run(args))
