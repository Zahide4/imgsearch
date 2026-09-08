#!/usr/bin/env python3
"""
export_static.py -- bundle the index into a static site.

No backend at all: the browser downloads the vectors once, embeds the query
with transformers.js (verified cos=0.999998 against the open_clip vectors in
Qdrant, so the two are interchangeable), and does brute force in JS.

    2,071 vectors int8  ->  1.6 MB
    100k    vectors int8  ->   77 MB   (still fine, cached after first load)

Brute force over normalised vectors IS exact cosine search -- the same thing
Qdrant's HNSW approximates. Below ~100k it is both faster and more accurate
than an ANN index, and needs no server.
"""
import json, sqlite3, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "static_site"
DIM = 768


def main():
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(ROOT / "data" / "index.db")
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT i.*, v.vec FROM images i JOIN vectors v ON v.id = i.id
        WHERE i.state='embedded' ORDER BY i.id
    """).fetchall()
    con.close()
    if not rows:
        sys.exit("nothing embedded")

    mat = np.frombuffer(b"".join(r["vec"] for r in rows),
                        dtype=np.float32).reshape(len(rows), DIM)

    # Symmetric int8 with a PERCENTILE scale, not max.
    # Using max lets one outlier component set the scale, which wastes most
    # of the int8 range on values that never occur (measured: real values
    # only reached -59..+127 of the available -127..127). Clipping at the
    # 99.99th percentile spends the range where the data actually is.
    scale = float(np.quantile(np.abs(mat), 0.9999))
    q = np.clip(np.round(mat / scale * 127.0), -127, 127).astype(np.int8)

    # measure what the quantisation actually costs, don't assume
    rng = np.random.default_rng(0)
    probes = rng.choice(len(rows), size=min(200, len(rows)), replace=False)
    deq = q.astype(np.float32) * (scale / 127.0)
    deq /= np.linalg.norm(deq, axis=1, keepdims=True)
    overlap = []
    for i in probes:
        exact = np.argsort(-(mat @ mat[i]))[:10]
        approx = np.argsort(-(deq @ deq[i]))[:10]
        overlap.append(len(set(exact.tolist()) & set(approx.tolist())) / 10)
    print(f"int8 top-10 recall vs float32: {np.mean(overlap)*100:.1f}%")

    (OUT / "vectors.bin").write_bytes(q.tobytes())

    meta = [{
        "t": r["title"] or "",
        "c": (r["creator"] or "")[:120],
        "l": r["license"] or "",
        "k": license_class(r["license"]),
        "u": r["source_url"] or "",
        "o": r["thumb_url"] or "",          # origin; proxy url built in JS
        "w": r["width"] or 0, "h": r["height"] or 0,
    } for r in rows]
    (OUT / "meta.json").write_text(json.dumps({
        "n": len(rows), "dim": DIM, "scale": scale, "items": meta},
        separators=(",", ":")))

    vb = (OUT / "vectors.bin").stat().st_size / 1e6
    mb = (OUT / "meta.json").stat().st_size / 1e6
    print(f"{len(rows):,} vectors -> vectors.bin {vb:.1f} MB + meta.json {mb:.1f} MB")


def license_class(lic):
    s = (lic or "").lower()
    if "cc0" in s or "public domain" in s or "pdm" in s:
        return "public_domain"
    if "sa" in s.replace("-", " ").split() or "share" in s:
        return "share_alike"
    return "attribution"


if __name__ == "__main__":
    main()
