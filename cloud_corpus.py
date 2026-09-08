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
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

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
                  iiurlwidth=800, maxlag=5)
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


async def fetch_image(client, row, sem):
    async with sem:
        try:
            # API-generated thumbnail only: never construct Wikimedia paths.
            r = await request(client, row['thumb_origin'])
            if len(r.content) > 15_000_000:
                return None
            with Image.open(io.BytesIO(r.content)) as source:
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
        except (httpx.HTTPError, OSError, RuntimeError, Image.DecompressionBombError) as exc:
            print(f"fetch skipped {row['image_id']}: {type(exc).__name__}", flush=True)
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
    import torch
    import torch.nn.functional as F
    import open_clip
    torch.set_num_threads(args.threads)
    hf = HfApi(token=os.environ['HF_TOKEN'])
    repo = os.environ['HF_REPO']
    qc = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'],
                      cloud_inference=True, timeout=120)
    for field, schema in [('build_id', models.PayloadSchemaType.KEYWORD), ('worker', models.PayloadSchemaType.INTEGER)]:
        qc.create_payload_index(COLLECTION, field, field_schema=schema, wait=True)
    flt = models.Filter(must=[models.FieldCondition(key='build_id', match=models.MatchValue(value=args.build)),
                             models.FieldCondition(key='worker', match=models.MatchValue(value=args.worker))])
    done = qc.count(COLLECTION, count_filter=flt, exact=True).count
    checkpoint = f'checkpoints/{args.build}-{args.workers}-{args.worker}.json'
    queue = deque({'start': lo, 'end': hi, 'continue': {}} for lo, hi in ranges(args.worker, args.workers))
    try:
        saved = hf_hub_download(repo, checkpoint, repo_type='dataset', token=os.environ['HF_TOKEN'])
        queue = deque(json.loads(Path(saved).read_text())['queue'])
    except EntryNotFoundError:
        pass
    if done >= args.target:
        print(f'already complete: {done}/{args.target}', flush=True)
        return
    print(f'loading SigLIP, worker {args.worker}, {done}/{args.target} already uploaded', flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
    model.eval()
    root = Path('worker-data')
    root.mkdir(exist_ok=True)
    archive = Archive(root, hf, repo)
    pending_points = []
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
    sem = asyncio.Semaphore(int(os.getenv('FETCH_CONCURRENCY', '6')))

    def save():
        # Archive first, so the durable cursor never runs ahead of the data.
        ops = []
        archive_op = archive.build()
        if archive_op:
            ops.append(archive_op)
        for i in range(0, len(pending_points), 128):
            qc.upsert(COLLECTION, points=pending_points[i:i + 128], wait=True)
        pending_points.clear()
        staged_ids.clear()
        state = json.dumps({'queue': list(queue), 'uploaded': done, 'pages': pages}).encode()
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
                existing = qc.retrieve(COLLECTION, ids=[point_id(r['image_id']) for r in rows], with_payload=False, with_vectors=False)
                seen = {str(p.id) for p in existing}
                rows = [r for r in rows if point_id(r['image_id']) not in seen | staged_ids][:args.target - done]
            loaded = [x for x in await asyncio.gather(*(fetch_image(client, row, sem) for row in rows)) if x]
            failed += len(rows) - len(loaded)
            for i in range(0, len(loaded), args.batch):
                chunk = loaded[i:i + args.batch]
                batch = torch.stack([preprocess(im) for _, im, _ in chunk])
                with torch.inference_mode():
                    vectors = F.normalize(model.encode_image(batch), dim=-1).float().numpy()
                if not np.isfinite(vectors).all() or vectors.shape[1] != 768:
                    raise RuntimeError('Invalid image embeddings; refusing upload')
                points = []
                for (row, _, webp), vec in zip(chunk, vectors):
                    row.update(build_id=args.build, worker=args.worker)
                    points.append(models.PointStruct(
                        id=point_id(row['image_id']),
                        vector={'image': vec.tolist(),
                                'bm25': models.Document(text=search_text(row), model=SPARSE_MODEL)},
                        payload=row,
                    ))
                    archive.add(row, webp)
                pending_points.extend(points)
                staged_ids.update(p.id for p in points)
                done += len(points)
            queue.popleft()
            continuation = data.get('continue')
            if continuation:
                if continuation == job['continue']:
                    raise RuntimeError('Commons repeated a continuation cursor')
                job['continue'] = continuation
                queue.append(job)
            pages += 1
            empty = empty + 1 if rows and not loaded else 0
            print(json.dumps({'worker': args.worker, 'uploaded': done, 'target': args.target,
                              'pages': pages, 'failed': failed, 'images_per_second': round(done / max(time.monotonic()-start_time, 1), 2)}), flush=True)
            # 128 commits/hour across the repo. One commit per save, 20
            # workers, ~2 img/s each -> 4000 keeps the whole fleet near 36/hour.
            if len(archive.pending) >= int(os.getenv('FLUSH_EVERY', '4000')):
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
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers or args.target < 1:
        p.error('invalid worker/target configuration')
    asyncio.run(run(args))
