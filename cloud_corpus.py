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
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import HfApi, hf_hub_download
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
    return params


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

    def flush(self):
        if not self.pending:
            return
        digest = hashlib.sha256(''.join(r['image_id'] for r, _ in self.pending).encode()).hexdigest()[:20]
        path = self.root / f'{digest}.tar'
        with tarfile.open(path, 'w') as tar:
            for row, webp in self.pending:
                key = point_id(row['image_id'])
                for name, data in [(key + '.webp', webp), (key + '.json', json.dumps(row).encode())]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        self.hf.upload_file(path_or_fileobj=str(path), path_in_repo=f'webdataset/{path.name}',
                            repo_id=self.repo, repo_type='dataset', commit_message='Archive licensed image derivatives and provenance')
        path.unlink()
        self.pending.clear()


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
    sem = asyncio.Semaphore(3)

    def save():
        archive.flush()  # Don't advance durable cursor ahead of the archive.
        for i in range(0, len(pending_points), 128):
            qc.upsert(COLLECTION, points=pending_points[i:i + 128], wait=True)
        pending_points.clear()
        staged_ids.clear()
        state = json.dumps({'queue': list(queue), 'uploaded': done, 'pages': pages}).encode()
        hf.upload_file(path_or_fileobj=state, path_in_repo=checkpoint, repo_id=repo, repo_type='dataset',
                       commit_message=f'Checkpoint worker {args.worker}: {done} images')

    async with httpx.AsyncClient(headers={'User-Agent': UA}, follow_redirects=True) as client:
        while queue and done < args.target and time.monotonic() - start_time < args.max_seconds:
            job = queue[0]
            r = await request(client, COMMONS, params=params_for(job['start'], job['end'], job['continue']))
            data = r.json()
            if data.get('error'):
                if data['error'].get('code') in ('maxlag', 'ratelimited'):
                    await asyncio.sleep(30)
                    continue
                raise RuntimeError(f"Commons API error: {data['error'].get('code')}")
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
            if len(archive.pending) >= 1000:
                save()
            if empty >= 10:
                save()
                raise RuntimeError('Ten batches failed to fetch: stopping instead of wasting the crawl')
            await asyncio.sleep(1)  # Sequential, paced API requests per worker.
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
