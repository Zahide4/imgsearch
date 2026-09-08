#!/usr/bin/env python3
"""
server.py -- the search API + web UI

    PROTOTYPE                       PRODUCTION
    ---------------------------    --------------------------------------
    numpy brute-force dot product   Qdrant HNSW + binary quantization
    SQLite metadata join            Postgres
    local ./data/thumbs             Backblaze B2 behind Cloudflare CDN
    FastAPI on localhost            same FastAPI, on a EUR12/mo Hetzner box

The retrieval math is IDENTICAL. Vectors are L2-normalised, so cosine
similarity is a plain dot product; brute force over a matrix is what an ANN
index approximates. Below ~100k vectors brute force is both exact and fast
(a few ms), so you get a perfect-recall baseline to measure Qdrant against
later. Swap point is marked SWAP HERE.

Run:
    python server.py          # then open http://127.0.0.1:8000
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
THUMBS = DATA / "thumbs"
DB_PATH = DATA / "index.db"
VEC_PATH = DATA / "vectors.npy"
IDS_PATH = DATA / "ids.json"

MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"

app = FastAPI(title="imgsearch-proto")

STATE = {"vectors": None, "ids": [], "model": None,
         "tokenizer": None, "device": "cpu", "qcache": {}}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@app.on_event("startup")
def startup():
    import open_clip

    if not VEC_PATH.exists():
        print("\n  No index found. Run:  python ingest.py --per-topic 60\n")
    else:
        STATE["vectors"] = np.load(VEC_PATH)
        STATE["ids"] = json.loads(IDS_PATH.read_text())
        print(f"loaded {len(STATE['ids'])} vectors "
              f"({STATE['vectors'].nbytes / 1e6:.0f} MB)")

    device = get_device()
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    STATE["model"] = model.to(device).eval()
    STATE["tokenizer"] = open_clip.get_tokenizer(MODEL_NAME)
    STATE["device"] = device
    print(f"text tower ready on {device}\n  ->  http://127.0.0.1:8000\n")


def embed_text(q: str) -> np.ndarray:
    """Encode the query into the SAME space the images live in.

    Cached because head queries repeat constantly -- in production this cache
    is Redis at the edge, and it is most of why 'instant' is achievable.
    """
    if q in STATE["qcache"]:
        return STATE["qcache"][q]
    tokens = STATE["tokenizer"]([q]).to(STATE["device"])
    with torch.no_grad():
        feats = STATE["model"].encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
    vec = feats.cpu().numpy().astype(np.float32)[0]
    if len(STATE["qcache"]) < 5000:
        STATE["qcache"][q] = vec
    return vec


@app.get("/api/search")
def search(q: str = Query(...), limit: int = 60, commercial_only: bool = True):
    t0 = time.perf_counter()
    if STATE["vectors"] is None or not q.strip():
        return {"results": [], "ms": 0, "total": 0}

    qv = embed_text(q.strip())
    t_embed = time.perf_counter()

    # ---------------------------------------------------------- SWAP HERE
    # Production: qdrant.search(collection, qv, limit=200, filter=...)
    # Binary quantization does this pass 32x cheaper, then rescores top-1000.
    scores = STATE["vectors"] @ qv
    k = min(limit * 4, len(scores))                # overfetch, then filter
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    # ---------------------------------------------------------------------
    t_search = time.perf_counter()

    ids = [STATE["ids"][i] for i in top]
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    rows = {r["id"]: dict(r) for r in con.execute(
        f"SELECT * FROM images WHERE id IN ({placeholders})", ids)}
    con.close()

    results = []
    for idx, img_id in zip(top, ids):
        r = rows.get(img_id)
        if not r:
            continue
        # Ranking rules beyond raw similarity. Production adds click-through,
        # source quality, aspect-ratio match, transparency detection.
        score = float(scores[idx])
        if r["width"] and r["width"] >= 2000:
            score += 0.01                          # nudge high-res upward
        results.append({
            "id": img_id,
            "title": r["title"],
            "creator": r["creator"],
            "license": r["license"],
            "license_url": r["license_url"],
            "source_url": r["source_url"],
            "full_url": r["full_url"],
            "width": r["width"], "height": r["height"],
            "score": round(score, 4),
            "thumb": f"/thumb/{img_id}",
        })
        if len(results) >= limit:
            break

    results.sort(key=lambda x: -x["score"])
    t_end = time.perf_counter()

    return {
        "results": results,
        "total": len(STATE["ids"]),
        "ms": round((t_end - t0) * 1000, 1),
        "timing": {
            "embed_ms": round((t_embed - t0) * 1000, 1),
            "ann_ms": round((t_search - t_embed) * 1000, 1),
            "meta_ms": round((t_end - t_search) * 1000, 1),
        },
    }


@app.get("/thumb/{img_id}")
def thumb(img_id: str):
    p = THUMBS / f"{img_id.replace(':', '_')}.webp"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=31536000"})


@app.get("/api/stats")
def stats():
    if not DB_PATH.exists():
        return {"total": 0}
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    n = con.execute("SELECT COUNT(*) c FROM images WHERE state='embedded'").fetchone()["c"]
    lic = [dict(r) for r in con.execute(
        "SELECT license, COUNT(*) c FROM images WHERE state='embedded' "
        "GROUP BY license ORDER BY c DESC LIMIT 12")]
    con.close()
    return {"total": n, "licenses": lic}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
