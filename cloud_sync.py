#!/usr/bin/env python3
"""
cloud_sync.py -- move crawl shards through Hugging Face Datasets.

Why HF and not R2/B2 for the prototype: it is free, generous, and needs NO
credit card. R2's free tier (10GB) and B2's (10GB) both want a payment method
on file. HF Datasets wants an email. For a prototype that matters.

    push   tar this shard's thumbnails + metadata JSONL -> HF dataset repo
    pull   download every shard -> merge into local SQLite + data/thumbs/

When you outgrow this (roughly 300k images), the swap is one function:
replace upload_file() with an S3 put_object() against B2. Same shape.

Setup (one time, free, no card):
    1. huggingface.co/join
    2. huggingface.co/settings/tokens  -> new token, role "write"
    3. huggingface.co/new-dataset      -> e.g. yourname/imgsearch-corpus
    4. export HF_TOKEN=hf_xxx  and  export HF_REPO=yourname/imgsearch-corpus
"""

import argparse
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
THUMBS = DATA / "thumbs"
DB_PATH = DATA / "index.db"


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract a shard archive without letting a member escape dest.

    Shards come from our own dataset repo, but a tampered or malformed
    archive must not be able to write through absolute paths, `..` or
    links. Only regular files are accepted; directories are created by
    extraction itself.
    """
    root = os.path.realpath(dest)
    for member in tar.getmembers():
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError(f"refusing archive member {member.name!r}: not a regular file")
        target = os.path.realpath(os.path.join(root, member.name))
        if not target.startswith(root + os.sep):
            raise ValueError(f"refusing archive member outside {dest}: {member.name!r}")
    tar.extractall(root)

REPO = os.environ.get("HF_REPO", "")
TOKEN = os.environ.get("HF_TOKEN", "")


def _api():
    from huggingface_hub import HfApi
    if not TOKEN or not REPO:
        sys.exit("Set HF_TOKEN and HF_REPO. See the docstring in this file.")
    return HfApi(token=TOKEN)


def push(shard: int):
    """Upload this shard's thumbnails (one tarball) + metadata (one JSONL).

    One tarball rather than N files on purpose: object stores and HF alike
    charge/throttle per request, and 50k individual PUTs is the single most
    common way people accidentally make this expensive. Same reasoning as
    packing thumbnails in the production design.
    """
    api = _api()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM images WHERE state IN ('stored','embedded')")]
    con.close()

    if not rows:
        print("nothing to push")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta = tmp / f"shard-{shard:03d}.jsonl"
        with meta.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        tar_path = tmp / f"shard-{shard:03d}.tar"
        n = 0
        with tarfile.open(tar_path, "w") as tar:
            for r in rows:
                p = THUMBS / f"{r['id'].replace(':', '_')}.webp"
                if p.exists():
                    tar.add(p, arcname=p.name)
                    n += 1

        size_mb = tar_path.stat().st_size / 1e6
        print(f"shard {shard}: {n} images, {size_mb:.0f} MB -> {REPO}")
        for local, remote in ((meta, f"meta/{meta.name}"),
                              (tar_path, f"thumbs/{tar_path.name}")):
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=REPO, repo_type="dataset")
        print("pushed")


def pull():
    """Download all shards and merge into the local SQLite + thumbs dir."""
    from huggingface_hub import HfApi, hf_hub_download
    api = _api()
    files = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    metas = sorted(f for f in files if f.startswith("meta/"))
    tars = sorted(f for f in files if f.startswith("thumbs/"))
    print(f"{len(metas)} metadata shards, {len(tars)} thumbnail tarballs")

    THUMBS.mkdir(parents=True, exist_ok=True)
    import ingest
    con = ingest.db_connect()

    total = 0
    for m in metas:
        p = hf_hub_download(REPO, m, repo_type="dataset", token=TOKEN)
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                con.execute("""
                    INSERT OR IGNORE INTO images
                    (id,title,creator,license,license_url,source_url,full_url,
                     thumb_url,width,height,topic,tags,state)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'stored')
                """, (r["id"], r.get("title"), r.get("creator"), r.get("license"),
                      r.get("license_url"), r.get("source_url"), r.get("full_url"),
                      r.get("thumb_url"), r.get("width"), r.get("height"),
                      r.get("topic"), r.get("tags", "")))
                total += 1
        con.commit()
        print(f"  merged {m}  ({total} rows)")

    for t in tars:
        p = hf_hub_download(REPO, t, repo_type="dataset", token=TOKEN)
        with tarfile.open(p) as tar:
            safe_extract(tar, THUMBS)
        print(f"  extracted {t}")

    con.close()
    n = len(list(THUMBS.glob("*.webp")))
    print(f"\n{total} rows, {n} thumbnails on disk")
    print("next:  python ingest.py --embed-only")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    push(a.shard) if a.cmd == "push" else pull()
