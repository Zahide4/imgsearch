#!/usr/bin/env python3
"""
app.py -- the deployed search API. Runs on HF Spaces free tier.

This IS the production query path, unchanged in shape:

    browser ──query text──> this Space ──vector──> Qdrant Cloud
       │                                               │
       └────────────── thumbnails ◄── R2 / CDN ◄───────┘
                       (direct, never proxied)

The API returns JSON only. Images are fetched by the browser straight from
the CDN using the `cdn` field in each Qdrant payload. That is the single
most important structural detail: proxying images through the API would
put every thumbnail byte through one box and destroy both latency and any
free tier you are sitting on.

Env (set as Space secrets):
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
"""

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent
COLLECTION = os.environ.get("QDRANT_COLLECTION", "images")
MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"

app = FastAPI(title="imgsearch")
S = {"model": None, "tok": None, "qc": None, "cache": {}}


@app.on_event("startup")
def startup():
    import open_clip
    url = os.environ.get("QDRANT_URL")
    if url:
        S["qc"] = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"),
                               timeout=30)
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    S["model"] = model.eval()          # CPU: the text tower is tiny
    S["tok"] = open_clip.get_tokenizer(MODEL_NAME)
    torch.set_num_threads(2)
    print("ready")


def embed_text(q: str):
    """Cached because head queries repeat constantly. In the paid design this
    cache is Redis at the edge; here a dict is plenty and is most of why
    repeat queries feel instant."""
    if q in S["cache"]:
        return S["cache"][q]
    with torch.no_grad():
        v = F.normalize(S["model"].encode_text(S["tok"]([q])), dim=-1)[0].tolist()
    if len(S["cache"]) < 10_000:
        S["cache"][q] = v
    return v


@app.get("/api/search")
def search(q: str = Query(...), limit: int = 60, license_class: str = ""):
    t0 = time.perf_counter()
    if not S["qc"]:
        return JSONResponse({"error": "QDRANT_URL not configured"}, status_code=503)
    if not q.strip():
        return {"results": [], "ms": 0}

    qv = embed_text(q.strip())
    t_embed = time.perf_counter()

    flt = None
    if license_class:
        flt = models.Filter(must=[models.FieldCondition(
            key="license_class",
            match=models.MatchAny(any=[x for x in license_class.split(",") if x]))])

    hits = S["qc"].query_points(
        COLLECTION, query=qv, limit=limit, with_payload=True,
        query_filter=flt,
        # rescore against full-precision vectors: int8 first pass is 4x
        # cheaper, the rescore recovers the recall it costs
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)),
    ).points
    t_search = time.perf_counter()

    results = [{
        "id": h.payload.get("image_id"),
        "title": h.payload.get("title", ""),
        "creator": h.payload.get("creator", ""),
        "license": h.payload.get("license", ""),
        "license_class": h.payload.get("license_class", ""),
        "license_url": h.payload.get("license_url", ""),
        "source_url": h.payload.get("source_url", ""),
        "width": h.payload.get("width", 0),
        "height": h.payload.get("height", 0),
        "thumb": h.payload.get("cdn", ""),
        "score": round(float(h.score), 4),
    } for h in hits]

    return {
        "results": results,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "timing": {"embed_ms": round((t_embed - t0) * 1000, 1),
                   "ann_ms": round((t_search - t_embed) * 1000, 1)},
    }


@app.get("/api/stats")
def stats():
    if not S["qc"]:
        return {"total": 0}
    try:
        return {"total": S["qc"].get_collection(COLLECTION).points_count}
    except Exception as e:
        return {"total": 0, "error": str(e)}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
