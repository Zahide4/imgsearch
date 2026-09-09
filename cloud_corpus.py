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
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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


def ranges(worker, workers):
    # Hundreds of independent ranges distribute discovery across subjects,
    # rather than taking 125k nearly identical names from one letter.
    starts = sorted(set([''] + list(string.digits + string.ascii_uppercase) +
                        [a + b for a in string.ascii_uppercase for b in string.ascii_lowercase]))
    all_ranges = list(zip(starts, starts[1:] + [None]))
    random.Random(20260908).shuffle(all_ranges)
    return all_ranges[worker::workers]


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


async def fetch_image(client, row, limiter):
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
            buf = io.BytesIO()
            image.save(buf, 'WEBP', quality=80, method=4)
            if image.mode == 'RGBA':
                rgb = Image.new('RGB', image.size, 'white')
                rgb.paste(image, mask=image.getchannel('A'))
            else:
                rgb = image.convert('RGB')
            return row, rgb, buf.getvalue()
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
    if not args.no_embed:
        hf = HfApi(token=os.environ['HF_TOKEN'])
        repo = os.environ['HF_REPO']
    # No cloud_inference: sparse vectors are built here, so this same code
    # works against a self-hosted instance, which has no inference service.
    qc = None
    if not args.no_embed:
        qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'],
                          timeout=120)
        for field, schema in [('build_id', models.PayloadSchemaType.KEYWORD), ('worker', models.PayloadSchemaType.INTEGER)]:
            qc.create_payload_index(COLLECTION, field, field_schema=schema, wait=True)
    checkpoint = f'checkpoints/{args.build}-{args.workers}-{args.worker}.json'

    def read_checkpoint():
        if args.no_embed:
            try:
                body = storage.client().get_object(Bucket=storage.BUCKET, Key=checkpoint)
                return json.loads(body['Body'].read())
            except Exception:
                return None
        try:
            return json.loads(Path(hf_hub_download(
                repo, checkpoint, repo_type='dataset',
                token=os.environ['HF_TOKEN'])).read_text())
        except EntryNotFoundError:
            return None
    queue = deque({'start': lo, 'end': hi, 'continue': {}} for lo, hi in ranges(args.worker, args.workers))
    done = 0
    state = read_checkpoint()
    if state:
        queue = deque(state['queue'])
        # Crawl-only has no Qdrant to count, so progress rides in the
        # checkpoint that already carries the cursor.
        done = state.get('done', 0)
    if qc is not None:
        flt = models.Filter(must=[models.FieldCondition(key='build_id', match=models.MatchValue(value=args.build)),
                                  models.FieldCondition(key='worker', match=models.MatchValue(value=args.worker))])
        done = qc.count(COLLECTION, count_filter=flt, exact=True).count
    if done >= args.target:
        print(f'already complete: {done}/{args.target}', flush=True)
        return
    if args.no_embed:
        if not storage.enabled():
            raise SystemExit('--no-embed needs a bucket: set STORAGE_BACKEND=s3 and the S3_* vars')
        print(f'crawl only, worker {args.worker}, {done}/{args.target} already written', flush=True)
        model = preprocess = None
    else:
        print(f'loading SigLIP, worker {args.worker}, {done}/{args.target} already uploaded', flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model.eval()
    root = Path('worker-data')
    root.mkdir(exist_ok=True)
    archive = Archive(root, hf, repo)
    pending_points = []
    manifest_rows = []
    manifest_seq = 0
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
                body = ('\n'.join(json.dumps(r) for r in manifest_rows) + '\n').encode()
                key = f'manifest/{args.build}/{args.worker:02d}-{manifest_seq:05d}.jsonl'
                storage.client().put_object(Bucket=storage.BUCKET, Key=key, Body=body,
                                            ContentType='application/x-ndjson')
                print(f'manifest {key}: {len(manifest_rows)} rows', flush=True)
                manifest_rows.clear()
                manifest_seq += 1
        else:
            archive_op = archive.build()
            if archive_op:
                ops.append(archive_op)
            for i in range(0, len(pending_points), 128):
                qc.upsert(COLLECTION, points=pending_points[i:i + 128], wait=True)
            pending_points.clear()
        staged_ids.clear()
        state = json.dumps({'queue': list(queue), 'uploaded': done,
                            'done': done, 'pages': pages}).encode()
        if args.no_embed:
            storage.client().put_object(Bucket=storage.BUCKET, Key=checkpoint,
                                        Body=state, ContentType='application/json')
            return
        ops.append(CommitOperationAdd(path_in_repo=checkpoint, path_or_fileobj=state))

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
        while queue and done < args.target and time.monotonic() - start_time < args.max_seconds:
            job = queue[0]
            r = await request(client, COMMONS, params=params_for(job['start'], job['end'], job['continue']))
            data = r.json()
            if data.get('error'):
                code = data['error'].get('code')
                if code in ('maxlag', 'ratelimited'):
                    await asyncio.sleep(30)
                    continue
                if code == 'urlparamnormal':
                    # One unrepresentable cursor should cost a range, not a
                    # worker. Drop this job, keep whatever it already indexed,
                    # and move to the next range.
                    print(f'worker {args.worker}: dropping range '
                          f'{job["start"]!r} after {code}', flush=True)
                    queue.popleft()
                    save()
                    continue
                raise RuntimeError(f"Commons API error: {code}")
            rows = [row for p in data.get('query', {}).get('pages', []) if (row := metadata(p, job['end']))]
            rows = list({row['image_id']: row for row in rows}.values())
            if rows:
                # Crawl-only has nothing to ask: the embed pass skips rows
                # already in Qdrant before it spends a GPU on them, so the
                # check belongs there rather than in a phase that runs hours
                # earlier and holds no database credentials.
                seen = set()
                if qc is not None:
                    seen = {str(p.id) for p in qc.retrieve(
                        COLLECTION, ids=[point_id(r['image_id']) for r in rows],
                        with_payload=False, with_vectors=False)}
                rows = [r for r in rows if point_id(r['image_id']) not in seen | staged_ids][:args.target - done]
            loaded = [x for x in await asyncio.gather(*(fetch_image(client, row, limiter) for row in rows)) if x]
            failed += len(rows) - len(loaded)
            produced = 0
            if args.no_embed:
                for i in range(0, len(loaded), args.batch):
                    chunk = loaded[i:i + args.batch]
                    for (row, _, _) in chunk:
                        row.update(build_id=args.build, worker=args.worker)
                    for row, url in zip((r for r, _, _ in chunk), upload_batch(chunk)):
                        if url:
                            row['cdn'] = url
                    # Only rows whose derivative actually reached the bucket:
                    # the embedding pass has no other way to read the image.
                    kept = [r for r, _, _ in chunk if r.get('cdn')]
                    manifest_rows.extend(kept)
                    staged_ids.update(point_id(r['image_id']) for r in kept)
                    done += len(kept)
                    produced += len(kept)
                loaded = []
            for i in range(0, len(loaded), args.batch):
                chunk = loaded[i:i + args.batch]
                batch = torch.stack([preprocess(im) for _, im, _ in chunk])
                with torch.inference_mode():
                    vectors = F.normalize(model.encode_image(batch), dim=-1).float().numpy()
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
                for row, url in zip((r for r, _, _ in chunk), upload_batch(chunk)):
                    if url:
                        row['cdn'] = url
                # One pass over the tokenizer for the batch, not one per row.
                sparse_vectors = sparse.documents(search_text(row) for row, _, _ in chunk)
                for (row, _, webp), vec, bm25 in zip(chunk, vectors, sparse_vectors):
                    points.append(models.PointStruct(
                        id=point_id(row['image_id']),
                        vector={'image': vec.tolist(), 'bm25': bm25},
                        payload=row,
                    ))
                    archive.add(row, webp)
                pending_points.extend(points)
                staged_ids.update(p.id for p in points)
                done += len(points)
                produced += len(points)
            queue.popleft()
            continuation = data.get('continue')
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
            # The coupled path flushes rarely because each save is a HuggingFace
            # commit, and the repo allows 128 an hour: 4000 keeps twenty workers
            # near 36/hour. Crawl-only writes to a bucket instead, which has no
            # such ceiling, so it flushes far more often -- otherwise a worker
            # that dies has uploaded derivatives nothing knows the names of.
            #
            # It also cannot key off `archive.pending`: crawl-only never adds to
            # the archive, so that list stays empty and the flush would never
            # fire at all. The manifest is what is pending here.
            pending = len(manifest_rows) if args.no_embed else len(archive.pending)
            if pending >= int(os.getenv('FLUSH_EVERY', '500' if args.no_embed else '4000')):
                save()
            if empty >= 10:
                save()
                raise RuntimeError('Ten batches failed to fetch: stopping instead of wasting the crawl')
            # Pacing for the Commons API. maxlag=5 already makes workers back
            # off when replication lag rises, which is the protection that
            # actually matters; this is a second, softer brake.
            await asyncio.sleep(0.3)
        save()
    if done < args.target:
        raise RuntimeError(f'Checkpoint saved at {done}/{args.target}; rerun this build to resume')
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
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-seconds', type=int, default=16200)
    p.add_argument('--no-embed', action='store_true',
                   help='crawl only: derivatives to the bucket, metadata to a '
                        'manifest, no SigLIP and no Qdrant. Embedding then runs '
                        'separately on a GPU reading from that bucket.')
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers or args.target < 1:
        p.error('invalid worker/target configuration')
    asyncio.run(run(args))
