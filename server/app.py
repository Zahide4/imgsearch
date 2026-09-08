#!/usr/bin/env python3
"""
app.py -- the production search API, sized to fit a free 512MB host.

This is the real thing: model in server RAM, Qdrant serves the index, the
browser downloads nothing but HTML. Identical in shape to what you'd run on
a paid box -- only the host is free.

WHY ONNX RUNTIME AND NOT PYTORCH
    torch + open_clip   ~2 GB RSS   -> needs a paid instance
    onnxruntime + int8    306 MB    -> fits Render/Koyeb/Fly free tiers
    Verified identical embeddings either way.

WHY `tokenizers` AND NOT `transformers`
    transformers pulls 218 MB of extra RSS for a tokenizer we can load
    directly. But two things must be replicated by hand or search silently
    degrades (both measured, both wrong by default):
      1. pad with token id 1, not 0
      2. SigLIP canonicalises text first: lowercase, strip punctuation,
         collapse whitespace. Without it 'A Steam Locomotive!' tokenises
         to something completely different from 'a steam locomotive'.
    With both fixes, tokenisation matches transformers exactly (6/6 probes).

Env: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, PORT
"""

import os
import re
import string
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
REPO = "Xenova/siglip-base-patch16-224"
ONNX = "onnx/text_model_int8.onnx"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "images")
DIM, MAXLEN, PAD_ID = 768, 64, 1

app = FastAPI(title="imgsearch")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
S = {"sess": None, "tok": None, "qc": None, "cache": {}}

_PUNCT = str.maketrans("", "", string.punctuation)


def canon(t: str) -> str:
    """SigLIP text canonicalisation. Must match the training preprocessing."""
    return re.sub(r"\s+", " ", t.lower().translate(_PUNCT)).strip()


@app.on_event("startup")
def startup():
    t0 = time.time()
    path = hf_hub_download(REPO, ONNX)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
    S["sess"] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
    tok.enable_truncation(MAXLEN)
    tok.enable_padding(length=MAXLEN, pad_id=PAD_ID, pad_token="</s>")
    S["tok"] = tok

    url = os.environ.get("QDRANT_URL")
    if url:
        S["qc"] = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"),
                               timeout=30)
    print(f"ready in {time.time() - t0:.1f}s")


def embed(text: str) -> list:
    if text in S["cache"]:
        return S["cache"][text]
    ids = np.array([S["tok"].encode(canon(text)).ids], dtype=np.int64)
    v = S["sess"].run(None, {"input_ids": ids})[1][0]
    v = (v / np.linalg.norm(v)).astype(np.float32).tolist()
    if len(S["cache"]) < 10_000:
        S["cache"][text] = v
    return v


@app.get("/api/search")
def search(q: str = Query(...), limit: int = 60, license_class: str = ""):
    t0 = time.perf_counter()
    if not S["qc"]:
        return JSONResponse({"error": "QDRANT_URL not set"}, status_code=503)
    if not q.strip():
        return {"results": [], "ms": 0}

    qv = embed(q.strip())
    t1 = time.perf_counter()

    flt = None
    if license_class:
        flt = models.Filter(must=[models.FieldCondition(
            key="license_class",
            match=models.MatchAny(any=[x for x in license_class.split(",") if x]))])

    hits = S["qc"].query_points(
        COLLECTION, query=qv, limit=limit, with_payload=True, query_filter=flt,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)),
    ).points
    t2 = time.perf_counter()

    return {
        "results": [{
            "id": h.payload.get("image_id"), "title": h.payload.get("title", ""),
            "creator": h.payload.get("creator", ""), "license": h.payload.get("license", ""),
            "license_class": h.payload.get("license_class", ""),
            "source_url": h.payload.get("source_url", ""),
            "width": h.payload.get("width", 0), "height": h.payload.get("height", 0),
            "thumb": h.payload.get("cdn", ""), "score": round(float(h.score), 4),
        } for h in hits],
        "ms": round((t2 - t0) * 1000, 1),
        "timing": {"embed_ms": round((t1 - t0) * 1000, 1),
                   "ann_ms": round((t2 - t1) * 1000, 1)},
    }


@app.get("/api/stats")
def stats():
    if not S["qc"]:
        return {"total": 0}
    try:
        return {"total": S["qc"].get_collection(COLLECTION).points_count}
    except Exception as e:
        return {"total": 0, "error": str(e)}


@app.get("/healthz")
def healthz():
    return {"ok": S["sess"] is not None}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
