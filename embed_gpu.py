#!/usr/bin/env python3
"""
embed_gpu.py -- the bulk embedding pass, for a free cloud GPU.

Kaggle Notebooks give 30 GPU-hours/WEEK free (P100 or T4x2), no card.
Colab's free T4 works too. This is the exact job you'd rent an L4 for in
production -- same model, same output -- just on someone else's free tier.

    M4 (local):     ~44 img/s   ->  300k images = 1.9 hours
    Kaggle P100:   ~350 img/s   ->  300k images = 14 minutes

On Kaggle: new notebook -> Settings -> Accelerator: GPU -> Internet: On.
Then in a cell:

    !pip install -q open_clip_torch boto3 qdrant-client
    !git clone https://github.com/YOURNAME/imgsearch-proto && cd imgsearch-proto
    %env S3_ENDPOINT=...
    %env QDRANT_URL=...
    !python embed_gpu.py --from-s3 --push

Reads thumbnails from S3/R2 (never re-crawls), embeds, pushes straight to
Qdrant. Nothing touches your machine.
"""

import argparse
import io
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "index.db"
THUMBS = ROOT / "data" / "thumbs"
MODEL_NAME, PRETRAINED, DIM = "ViT-B-16-SigLIP", "webli", 768


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto")
    ap.add_argument("--from-s3", action="store_true",
                    help="stream thumbnails from R2/B2 instead of local disk")
    ap.add_argument("--push", action="store_true", help="upsert to Qdrant when done")
    a = ap.parse_args()

    import open_clip
    dev = device()
    bs = a.batch_size or (256 if dev == "cuda" else 32)
    print(f"device={dev} batch={bs}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.half()          # fp16 ~2x throughput on T4/P100

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    todo = [r["id"] for r in con.execute("""
        SELECT i.id FROM images i LEFT JOIN vectors v ON v.id = i.id
        WHERE i.state IN ('stored','embedded') AND v.id IS NULL
    """)]
    if not todo:
        print("nothing to embed")
        return
    print(f"embedding {len(todo):,} images")

    if a.from_s3:
        import storage
        pool = ThreadPoolExecutor(32)

        def load(img_id):
            try:
                obj = storage.client().get_object(
                    Bucket=storage.BUCKET, Key=storage.key_for(img_id))
                return img_id, Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
            except Exception:
                return img_id, None
    else:
        pool = ThreadPoolExecutor(8)

        def load(img_id):
            p = THUMBS / f"{img_id.replace(':', '_')}.webp"
            try:
                return img_id, Image.open(p).convert("RGB")
            except Exception:
                return img_id, None

    t0, done = time.time(), 0
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        loaded = list(pool.map(load, chunk))
        tensors = [preprocess(im) for _, im in loaded if im is not None]
        ids = [k for k, im in loaded if im is not None]
        if not tensors:
            continue

        batch = torch.stack(tensors).to(dev)
        if dev == "cuda":
            batch = batch.half()
        with torch.no_grad():
            feats = F.normalize(model.encode_image(batch), dim=-1)
        feats = feats.float().cpu().numpy().astype(np.float32)

        con.executemany("INSERT OR REPLACE INTO vectors (id,vec) VALUES (?,?)",
                        [(k, v.tobytes()) for k, v in zip(ids, feats)])
        con.executemany("UPDATE images SET state='embedded' WHERE id=?",
                        [(k,) for k in ids])
        con.commit()
        done += len(ids)
        print(f"\r  {done}/{len(todo)}  ({done/max(time.time()-t0,1e-9):.0f} img/s)",
              end="", flush=True)
    print()
    con.close()

    if a.push:
        import push_qdrant
        push_qdrant.push()


if __name__ == "__main__":
    main()
