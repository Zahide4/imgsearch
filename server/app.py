#!/usr/bin/env python3
"""Cloud search: SigLIP ONNX text encoder, Qdrant index, direct CDN images."""
import asyncio
import os
import re
import string
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from qdrant_client import AsyncQdrantClient, models
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
REPO, ONNX = 'Xenova/siglip-base-patch16-224', 'onnx/text_model_int8.onnx'
COLLECTION = os.getenv('QDRANT_COLLECTION', 'images-v2')
SPARSE_MODEL = 'qdrant/bm25'
DIM, MAXLEN, PAD_ID = 768, 64, 1
app = FastAPI(title='imgsearch')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET'], allow_headers=['*'])
S = {'sess': None, 'tok': None, 'qc': None, 'cache': OrderedDict(),
     'results': OrderedDict(), 'lock': None, 'stats': (0, 0)}
_PUNCT = str.maketrans('', '', string.punctuation)
ALLOWED_LICENSES = {'public_domain', 'attribution', 'share_alike'}


def canon(text):
    return re.sub(r'\s+', ' ', text.lower().translate(_PUNCT)).strip()


def remember(cache, key, value, maximum):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > maximum:
        cache.popitem(last=False)


@app.on_event('startup')
async def startup():
    def load():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv('ORT_THREADS', '1'))
        opts.inter_op_num_threads = 1
        S['sess'] = ort.InferenceSession(hf_hub_download(REPO, ONNX), opts, providers=['CPUExecutionProvider'])
        tok = Tokenizer.from_file(hf_hub_download(REPO, 'tokenizer.json'))
        tok.enable_truncation(MAXLEN)
        tok.enable_padding(length=MAXLEN, pad_id=PAD_ID, pad_token='</s>')
        S['tok'] = tok
    await asyncio.to_thread(load)
    S['lock'] = asyncio.Lock()
    if os.getenv('QDRANT_URL'):
        S['qc'] = AsyncQdrantClient(url=os.environ['QDRANT_URL'], api_key=os.getenv('QDRANT_API_KEY'),
                                    cloud_inference=True, timeout=20)


@app.on_event('shutdown')
async def shutdown():
    if S['qc']:
        await S['qc'].close()


def embed(text):
    if text in S['cache']:
        S['cache'].move_to_end(text)
        return S['cache'][text]
    ids = np.array([S['tok'].encode(text).ids], dtype=np.int64)
    # Preserve trained padding and the projected text output.
    vec = S['sess'].run(['pooler_output'], {'input_ids': ids})[0][0]
    norm = np.linalg.norm(vec)
    if not np.isfinite(vec).all() or norm <= 0:
        raise RuntimeError('Invalid query embedding')
    vec = (vec / norm).astype(np.float32).tolist()
    remember(S['cache'], text, vec, 512)
    return vec


def result_from(hit):
    p = hit.payload
    origin = p.get('thumb_origin', '')
    thumb = p.get('cdn', '')
    if not thumb and origin:
        thumb = 'https://wsrv.nl/?url=' + quote(origin, safe='') + '&w=384&h=384&fit=inside&output=webp&q=80&maxage=1y'
    return dict(id=p.get('image_id'), title=p.get('title', ''), creator=p.get('creator', ''),
                license=p.get('license', ''), license_class=p.get('license_class', ''),
                license_url=p.get('license_url', ''), source_url=p.get('source_url', ''),
                full_url=p.get('full_url', ''), width=p.get('width', 0), height=p.get('height', 0),
                thumb=thumb, score=round(float(hit.score), 4))


@app.get('/api/search')
async def search(request: Request, q: str = Query(..., max_length=300),
                 limit: int = Query(60, ge=1, le=100), license_class: str = ''):
    start = time.perf_counter()
    text = canon(q)
    licenses = tuple(sorted(set(filter(None, license_class.split(',')))))
    if set(licenses) - ALLOWED_LICENSES:
        raise HTTPException(400, 'Unknown license filter')
    if not text:
        return {'results': [], 'ms': 0}
    if S['qc'] is None:
        raise HTTPException(503, 'Search is not configured')
    key = (text, limit, licenses)
    cached = S['results'].get(key)
    if cached and time.monotonic() - cached[0] < 120:
        S['results'].move_to_end(key)
        return {**cached[1], 'cached': True, 'ms': 0, 'timing': {'embed_ms': 0, 'ann_ms': 0}}
    # One embedding at a time on a small CPU. Waiting clients that have
    # disconnected do not consume another expensive inference slot.
    async with S['lock']:
        if await request.is_disconnected():
            raise HTTPException(499, 'Search cancelled')
        vector = await asyncio.to_thread(embed, text)
    embedded = time.perf_counter()
    flt = models.Filter(must=[models.FieldCondition(key='license_class', match=models.MatchAny(any=list(licenses)))]) if licenses else None
    candidates = max(100, limit * 2)
    dense_prefetch = models.Prefetch(
        query=vector, using='image', limit=candidates, filter=flt,
        params=models.SearchParams(
            hnsw_ef=128,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0),
        ),
    )
    try:
        # The BM25 query vector is built by Qdrant Cloud inference. If that is
        # rate-limited or unavailable, fall back to dense-only rather than
        # failing the whole request: the semantic half needs no inference and
        # is the primary ranking signal. Losing exact-name matching degrades
        # results; returning 503 loses search entirely.
        hits = (await S['qc'].query_points(
            COLLECTION,
            prefetch=[
                dense_prefetch,
                models.Prefetch(
                    query=models.Document(text=text, model=SPARSE_MODEL),
                    using='bm25', limit=candidates, filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True,
        )).points
    except Exception:
        try:
            hits = (await S['qc'].query_points(
                COLLECTION, query=vector, using='image', limit=limit,
                with_payload=True, query_filter=flt,
                params=models.SearchParams(
                    hnsw_ef=128,
                    quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0),
                ),
            )).points
        except Exception:
            raise HTTPException(503, 'Search is temporarily unavailable. Please try again.')
    end = time.perf_counter()
    data = {'results': [result_from(hit) for hit in hits], 'ms': round((end-start)*1000, 1),
            'timing': {'embed_ms': round((embedded-start)*1000, 1), 'ann_ms': round((end-embedded)*1000, 1)}}
    remember(S['results'], key, (time.monotonic(), data), 128)
    return data


@app.get('/api/stats')
async def stats():
    if not S['qc']:
        return {'total': 0}
    stamp, count = S['stats']
    if time.monotonic() - stamp > 30:
        try:
            count = (await S['qc'].get_collection(COLLECTION)).points_count
            S['stats'] = (time.monotonic(), count)
        except Exception:
            return {'total': count, 'stale': True}
    return {'total': count}


@app.get('/api/progress')
async def progress(target: int = 500000):
    """Live corpus growth for the build dashboard.

    Keeps a small in-memory sample ring so rate and ETA are available on the
    first request rather than after the client has watched for a while. The
    ring resets when Render cycles the instance; that only costs the rate
    estimate, never the count, which is always read fresh from Qdrant.
    """
    if not S['qc']:
        return JSONResponse({'error': 'not configured'}, status_code=503)
    try:
        count = (await S['qc'].count(COLLECTION, exact=True)).count
    except Exception:
        raise HTTPException(503, 'Index unavailable')

    now = time.time()
    hist = S.setdefault('phist', [])
    if not hist or now - hist[-1][0] >= 5:
        hist.append((now, count))
        del hist[:-720]                       # ~1h at 5s resolution

    rate = None
    if len(hist) >= 2:
        # Use the widest window available, capped at 15 minutes, so a single
        # slow checkpoint does not swing the estimate.
        first = next((h for h in hist if now - h[0] <= 900), hist[0])
        dt, dn = now - first[0], count - first[1]
        if dt > 20 and dn > 0:
            rate = dn / dt

    remaining = max(0, target - count)
    return {
        'count': count,
        'target': target,
        'remaining': remaining,
        'pct': round(min(100.0, count / target * 100), 2) if target else 0,
        'per_second': round(rate, 3) if rate else None,
        'per_minute': round(rate * 60, 1) if rate else None,
        'eta_seconds': int(remaining / rate) if rate and remaining else None,
        'samples': [{'t': int(t), 'n': n} for t, n in hist[-180:]],
        'server_time': int(now),
    }


@app.get('/healthz')
def healthz():
    return {'ok': S['sess'] is not None}


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html', headers={'Cache-Control': 'no-cache'})
