#!/usr/bin/env python3
"""
ingest.py -- crawl -> resize -> WebP -> embed -> index

Architecturally identical to the production pipeline. Every stage maps 1:1;
only the backing service changes when you scale up.

    PROTOTYPE (free, local)          PRODUCTION
    ----------------------------    -------------------------------------
    Wikimedia Commons search API    Commons dumps + Openverse + museums
    httpx async fetch               same, sharded across cheap VPSes
    Pillow -> 384px WebP            pyvips -> 384px WebP (streaming, 5x faster)
    SigLIP on MPS (Apple GPU)       same model, rented GPU for the bulk pass
    ./data/thumbs/                  Backblaze B2 + Cloudflare CDN
    SQLite                          Postgres
    vectors.npy + numpy dot         Qdrant, binary quantization + HNSW

Nothing here costs money and nothing leaves your machine except polite
HTTP GETs to Wikimedia.

Usage:
    python ingest.py --per-topic 60           # ~9k images, roughly 25 min
    python ingest.py --per-topic 200          # ~30k images, roughly 90 min
    python ingest.py --embed-only             # re-embed what's on disk
"""

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import time
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
THUMBS = DATA / "thumbs"
DB_PATH = DATA / "index.db"
VEC_PATH = DATA / "vectors.npy"
IDS_PATH = DATA / "ids.json"

MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"
EMBED_DIM = 768
THUMB_PX = 384          # doubles as display thumbnail AND embedding input
WEBP_QUALITY = 80

# CHANGE THIS. Wikimedia's User-Agent policy requires a real contact.
# Generic UAs get blocked. https://meta.wikimedia.org/wiki/User-Agent_policy
UA = os.environ.get("CRAWL_UA") or "ImgSearchPrototype/0.1 (https://github.com/yourname/imgsearch; sayeedshadab@gmail.com)"

COMMONS = "https://commons.wikimedia.org/w/api.php"

# Object storage is optional locally, required in CI (runners have no disk).
KEEP_LOCAL = os.environ.get("KEEP_LOCAL", "1") == "1"
try:
    import storage as _storage
    _S3 = _storage if _storage.enabled() else None
except Exception:
    _S3 = None

_SEEN_WARNINGS = set()

def _warn_once(msg: str):
    """Discovery errors used to vanish into bare excepts. Surface each
    distinct one once so a broken source is visible, not silently empty."""
    if msg not in _SEEN_WARNINGS:
        _SEEN_WARNINGS.add(msg)
        print(f"\n  [warn] {msg}", flush=True)


# ---------------------------------------------------------------- database

def db_connect():
    DATA.mkdir(exist_ok=True)
    THUMBS.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id           TEXT PRIMARY KEY,
            title        TEXT,
            creator      TEXT,
            license      TEXT,
            license_url  TEXT,
            source_url   TEXT,
            full_url     TEXT,
            thumb_url    TEXT,
            width        INTEGER,
            height       INTEGER,
            topic        TEXT,
            tags         TEXT,
            state        TEXT DEFAULT 'pending',  -- pending|stored|embedded|failed
            fail_reason  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_state ON images(state);

        CREATE TABLE IF NOT EXISTS vectors (
            id  TEXT PRIMARY KEY,
            vec BLOB NOT NULL
        );
    """)
    con.commit()
    return con


# ---------------------------------------------------------------- licensing

_BLOCK = re.compile(r"\bn[cd]\b|noncommercial|non-commercial|noderiv|non-free|fair\s*use")
_ALLOW = re.compile(r"cc0|cc[\s-]?by|public\s*domain|\bpd\b|pd-|pdm|attribution")

def license_ok(short_name: str) -> bool:
    """Commercial-use filter. Rejects NC/ND, accepts CC0/PD/BY/BY-SA.

    NOTE: BY-SA is share-alike and viral onto derivatives. It is allowed here
    so you can see it in results, but in the real product you'd default it OFF
    for video editors who won't read the terms.
    """
    s = (short_name or "").strip().lower()
    if not s:
        return False
    if _BLOCK.search(s):
        return False
    return bool(_ALLOW.search(s))


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()



# ---------------------------------------------------------------- enumeration

# Prefix points that split the Commons filename space for parallel walking.
# `aifrom` lets each worker start at a different point in the sorted namespace,
# which is what makes enumeration shardable across GitHub Actions runners.
ENUM_PREFIXES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","Ab","Am","B","Bo","C","Ch","Co","D","De","E","F","Fi","Fr","G","Go",
    "H","He","I","J","K","L","Li","M","Ma","Mo","N","O","P","Pa","Ph","Po",
    "Q","R","Ro","S","Sa","Sh","St","T","Th","To","U","V","W","Wi","X","Y","Z",
]


def commons_thumb(url: str, px: int = 800) -> str:
    """Turn a Commons original URL into its pre-rendered thumbnail URL.

    allimages returns only the master file, and those masters are huge -- the
    first enumeration run pulled 12,101px originals and lost 819/900 fetches
    to HTTP 429. Commons serves thumbnails at a derivable path:

        .../commons/a/ab/Name.jpg
        .../commons/thumb/a/ab/Name.jpg/800px-Name.jpg

    ~20x less bandwidth and far gentler on their servers.
    """
    # The API appends a tracking query string
    # (?utm_source=...&utm_content=original). Left on, it lands *inside* the
    # derived filename and every thumb 404s.
    url = url.split("?", 1)[0]
    marker = "/commons/"
    if marker not in url or "/thumb/" in url:
        return url
    head, tail = url.split(marker, 1)
    parts = tail.split("/")
    if len(parts) < 3:
        return url
    a, ab, fname = parts[0], parts[1], "/".join(parts[2:])
    return f"{head}{marker}thumb/{a}/{ab}/{fname}/{px}px-{fname}"


async def discover_enumerate(client, aifrom, want, limiter, batch=50, aito=None):
    """Walk Commons' full file list instead of searching it.

    Search only surfaces subjects you thought to type; enumeration gets
    everything. `gaifrom` starts each worker at a different point in the
    sorted namespace, so this shards cleanly across runners.

    Uses generator=allimages + iiurlwidth so the API hands back pre-rendered
    thumbnails. Two earlier approaches failed:
      - list=allimages returns only the master file. Fetching 12,101px
        originals lost 819/900 requests to HTTP 429.
      - Deriving thumb URLs by hand (/commons/thumb/a/ab/N.jpg/800px-N.jpg)
        is the correct MediaWiki form but Varnish answers 400.
    Asking the API for iiurlwidth=800 works, is 10x faster than
    list=allimages (2.1s vs 21.2s per batch), and is what the search path
    already does.
    """
    out = []
    seen = set()
    cont = {}
    retries = 0
    while len(out) < want:
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "allimages", "gailimit": str(batch), "gaisort": "name",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime", "iiurlwidth": "800",
            "gaifrom": aifrom, "maxlag": "5",
        }
        if aito is not None:
            params["gaito"] = aito
        if cont:
            params.update(cont)          # carries gaicontinue AND iicontinue
        else:
            params["gaifrom"] = aifrom
        try:
            async with limiter.get(COMMONS):
                r = await client.get(COMMONS, params=params, timeout=90)
            if r.status_code == 429:
                retries += 1
                if retries >= 6:
                    raise RuntimeError("Commons remained rate limited after six attempts")
                await asyncio.sleep(5 + random.random() * 5)
                continue
            if r.status_code != 200:
                break
            data = r.json()
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            retries = 0
        except Exception as e:
            _warn_once(f"enumerate {aifrom}: {type(e).__name__}: {e}")
            break

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            break

        for pg in pages:
            title = (pg.get("title") or "").removeprefix("File:")
            if aito is not None and title.replace(" ", "_") >= aito.replace(" ", "_"):
                continue
            image_id = f"commons:{pg['pageid']}"
            if image_id in seen:
                continue
            ii = (pg.get("imageinfo") or [{}])[0]
            thumb = ii.get("thumburl")
            if not thumb or ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if (ii.get("width") or 0) < 320 or (ii.get("height") or 0) < 320:
                continue                     # skip icons and scan fragments
            meta = ii.get("extmetadata", {}) or {}
            lic = _clean(meta.get("LicenseShortName", {}).get("value", ""))
            if not license_ok(lic):
                continue
            seen.add(image_id)
            out.append({
                "id": image_id,
                "title": title,
                "creator": _clean(meta.get("Artist", {}).get("value", ""))[:200],
                "license": lic,
                "license_url": _clean(meta.get("LicenseUrl", {}).get("value", "")),
                "source_url": ii.get("descriptionurl", ""),
                "full_url": ii.get("url", ""),
                "thumb_url": thumb,
                "width": ii.get("width", 0), "height": ii.get("height", 0),
                "topic": f"enum:{aifrom}", "tags": "",
            })

        cont = data.get("continue")
        if not cont:
            break
    return out[:want]


# ---------------------------------------------------------------- rate limiting

class HostLimiter:
    """One semaphore PER HOST, not one globally.

    This is the whole fix for the 429 wall. Every image used to come from
    upload.wikimedia.org, so all concurrency landed on one server that
    throttles per-IP. Spreading across Flickr / iNaturalist / Smithsonian /
    Met CDNs multiplies real throughput without hammering anyone: 8 in
    flight to each of 6 hosts is 48 concurrent fetches that all look polite.
    """
    def __init__(self, per_host=6):
        self.per_host = per_host
        self._sems = {}

    def get(self, url):
        host = urlparse(url).netloc or "unknown"
        if host not in self._sems:
            self._sems[host] = asyncio.Semaphore(self.per_host)
        return self._sems[host]


# ---------------------------------------------------------------- discovery

OPENVERSE = "https://api.openverse.org/v1/images/"

# Live tests found the standard tier still capped at 20 results per page,
# with no bulk throughput benefit from credentials. Use enumeration for bulk.
OV_TOKEN = os.environ.get("OPENVERSE_TOKEN", "")
OV_PAGE = 20  # Standard credentials did not lift this cap in live tests.

# Commercial-safe licences ONLY, enforced at the API. Note that Openverse's
# own `license_type=commercial` still returns ND (no-derivatives), which
# permits commercial use but forbids cropping or compositing -- useless for
# an editor. So we whitelist explicitly instead.
OV_LICENSES = "cc0,pdm,by,by-sa"

# Rotating these spreads load across completely different CDNs.
# Measured yield per topic over 88 topics / 12.6k images:
#   wikimedia 73, flickr 20, rawpixel 13, geograph 10, svgsilh 6, met 5,
#   museumsvictoria 4, smithsonian 1.4, inaturalist 1.0
# The tail costs the same API budget as the head and returns almost nothing,
# so keep the productive sources plus svgsilh/rawpixel (the only real sources
# of transparent cutouts, which the corpus is measurably short of).
OV_SOURCES = [
    "flickr", "wikimedia", "rawpixel", "geographorguk", "svgsilh",
    "met", "europeana",
]


async def discover_openverse(client, topic, source, limit, limiter):
    """Openverse aggregates 915M works across 52 providers.

    We take result['url'] (the origin CDN) rather than result['thumbnail']
    (an api.openverse.org proxy) precisely so the load spreads out.
    """
    out, page = [], 1
    max_pages = 3 if OV_TOKEN else 12
    headers = {"Authorization": f"Bearer {OV_TOKEN}"} if OV_TOKEN else None
    while len(out) < limit and page <= max_pages:
        params = {
            "q": topic, "license": OV_LICENSES, "source": source,
            "page_size": str(min(OV_PAGE, limit)), "page": str(page),
        }
        try:
            async with limiter.get(OPENVERSE):
                r = await client.get(OPENVERSE, params=params, timeout=45,
                                     headers=headers)
            if r.status_code == 429:
                await asyncio.sleep(5 + random.random() * 5)
                continue
            if r.status_code != 200:
                break
            results = r.json().get("results", [])
        except Exception as e:
            _warn_once(f"openverse {source}: {type(e).__name__}: {e}")
            break
        if not results:
            break

        for x in results:
            url = x.get("url")
            if not url:
                continue
            lic = f"CC {x.get('license','').upper()} {x.get('license_version','')}".strip()
            if not license_ok(lic):
                continue
            tags = ", ".join(t.get("name", "") for t in (x.get("tags") or [])[:25])
            out.append({
                "id": f"ov:{x['id']}",
                "title": (x.get("title") or "")[:300],
                "creator": (x.get("creator") or "")[:200],
                "license": lic,
                "license_url": x.get("license_url") or "",
                "source_url": x.get("foreign_landing_url") or "",
                "full_url": url,
                "thumb_url": url,          # fetched + downsized locally
                "width": x.get("width") or 0,
                "height": x.get("height") or 0,
                "topic": topic,
                "tags": tags,
            })
        page += 1
    return out[:limit]



async def discover(client, topic, limit, sem):
    """Query Commons search. Returns metadata dicts, license-filtered."""
    out, offset = [], 0
    while len(out) < limit and offset < 500:
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {topic}",
            "gsrnamespace": "6",
            "gsrlimit": str(min(50, limit - len(out))),
            "gsroffset": str(offset),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": "800",
        }
        try:
            async with sem.get(COMMONS):
                r = await client.get(COMMONS, params=params, timeout=30)
            if r.status_code == 429:
                await asyncio.sleep(5)
                continue
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
        except Exception as e:
            _warn_once(f"commons: {type(e).__name__}: {e}")
            break

        if not pages:
            break

        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii or not ii.get("thumburl"):
                continue
            if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            meta = ii.get("extmetadata", {}) or {}
            lic = _clean(meta.get("LicenseShortName", {}).get("value", ""))
            if not license_ok(lic):
                continue
            out.append({
                "id": f"commons:{p['pageid']}",
                "title": _clean(p.get("title", "")).replace("File:", ""),
                "creator": _clean(meta.get("Artist", {}).get("value", ""))[:200],
                "license": lic,
                "license_url": _clean(meta.get("LicenseUrl", {}).get("value", "")),
                "source_url": ii.get("descriptionurl", ""),
                "full_url": ii.get("url", ""),
                "thumb_url": ii.get("thumburl", ""),
                "width": ii.get("width", 0),
                "height": ii.get("height", 0),
                "topic": topic,
                "tags": "",
            })
        offset += 50
    return out[:limit]


# ---------------------------------------------------------------- fetch + resize

async def fetch_and_store(client, row, limiter, attempts=4):
    """Download -> resize to 384px -> WebP -> disk. Nothing hits a temp file.

    Commons WILL rate-limit you (429). Retrying with backoff is not optional:
    without it you silently lose ~60% of the corpus and mistake throttling for
    missing images. Honour Retry-After, release the semaphore while sleeping.

    In production this same function streams straight into an S3 PUT to B2 and
    hands the identical bytes to the embedder. One derivative, two uses: display
    thumbnail AND embedding input, so changing embedding models never means
    re-crawling the internet.
    """
    dest = THUMBS / f"{row['id'].replace(':', '_')}.webp"
    if dest.exists():
        return "stored", None

    reason = None
    for attempt in range(attempts):
        try:
            async with limiter.get(row["thumb_url"]):
                r = await client.get(row["thumb_url"], timeout=45)

            if r.status_code == 429:
                reason = "http_429"
                wait = float(r.headers.get("Retry-After") or 0) or min(2 ** attempt, 30)
                await asyncio.sleep(wait + random.random())   # jitter
                continue
            if r.status_code >= 500:
                reason = f"http_{r.status_code}"
                await asyncio.sleep(min(2 ** attempt, 15) + random.random())
                continue

            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, "WEBP", quality=WEBP_QUALITY, method=4)
            data = buf.getvalue()

            # Production: the crawler streams straight into object storage.
            # A GitHub Actions runner has no persistent disk, so this is the
            # ONLY path that works there.
            if _S3:
                await asyncio.to_thread(_S3.put, row["id"], data)
                if not KEEP_LOCAL:
                    return "stored", None
            dest.write_bytes(data)
            return "stored", None

        except Exception as e:
            reason = type(e).__name__
            await asyncio.sleep(min(2 ** attempt, 10) + random.random())

    return "failed", reason


async def crawl(topics, per_topic, per_host=6, retry_failed=False,
                sources=("commons", "openverse"), shard=0, shards=1,
                enum_target=0):
    con = db_connect()
    limiter = HostLimiter(per_host)
    headers = {"User-Agent": UA}

    # Sharding: each worker takes a disjoint slice of the topic list, so N
    # machines (or N GitHub Actions runners, each with its own IP) can crawl
    # in parallel without duplicating work.
    if shards > 1:
        topics = [t for i, t in enumerate(topics) if i % shards == shard]
        print(f"shard {shard}/{shards}: {len(topics)} topics")

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        # ---- discovery
        jobs = []
        if "enumerate" in sources:
            # Each walk has an exclusive upper boundary.
            mine = [x for k, x in enumerate(ENUM_PREFIXES) if k % shards == shard]
            per = max(1, enum_target // max(len(mine), 1))
            jobs += [("enumerate", pfx, per) for pfx in mine]
        if "commons" in sources:
            jobs += [("commons", t, None) for t in topics]
        if "openverse" in sources:
            # every (topic, provider) pair is its own job -> different CDNs
            jobs += [("openverse", t, src) for t in topics for src in OV_SOURCES]

        print(f"discovering: {len(jobs)} jobs across {len(topics)} topics...")
        found = 0

        async def run_job(kind, topic, src):
            if kind == "enumerate":
                pos = ENUM_PREFIXES.index(topic)
                end = ENUM_PREFIXES[pos + 1] if pos + 1 < len(ENUM_PREFIXES) else None
                return await discover_enumerate(client, topic, src, limiter, aito=end)
            if kind == "commons":
                return await discover(client, topic, per_topic, limiter)
            return await discover_openverse(client, topic, src, per_topic, limiter)

        tasks = [run_job(*j) for j in jobs]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            try:
                rows = await coro
            except Exception as e:
                _warn_once(f"job: {type(e).__name__}: {e}")
                rows = []
            for row in rows:
                con.execute("""
                    INSERT OR IGNORE INTO images
                    (id,title,creator,license,license_url,source_url,
                     full_url,thumb_url,width,height,topic,tags,state)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'pending')
                """, (row["id"], row["title"], row["creator"], row["license"],
                      row["license_url"], row["source_url"], row["full_url"],
                      row["thumb_url"], row["width"], row["height"],
                      row["topic"], row.get("tags", "")))
            found += len(rows)
            if i % 25 == 0 or i == len(jobs):
                con.commit()
                print(f"\r  {i}/{len(jobs)} jobs | {found} candidates",
                      end="", flush=True)
        con.commit()
        print()

        # ---- fetch + resize
        states = ("pending", "failed") if retry_failed else ("pending",)
        pending = con.execute(
            f"SELECT * FROM images WHERE state IN ({','.join('?' * len(states))})",
            states).fetchall()
        print(f"fetching {len(pending)} images...")
        t0, done = time.time(), 0
        for i in range(0, len(pending), 400):
            chunk = pending[i:i + 400]
            results = await asyncio.gather(
                *[fetch_and_store(client, dict(r), limiter) for r in chunk])
            for row, (state, reason) in zip(chunk, results):
                con.execute("UPDATE images SET state=?, fail_reason=? WHERE id=?",
                            (state, reason, row["id"]))
            con.commit()
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"\r  {done}/{len(pending)}  ({rate:.0f}/s)", end="", flush=True)
        print()

    ok = con.execute("SELECT COUNT(*) c FROM images WHERE state IN ('stored','embedded')").fetchone()["c"]
    print(f"stored: {ok}")
    fails = con.execute("SELECT fail_reason, COUNT(*) c FROM images "
                        "WHERE state='failed' GROUP BY fail_reason "
                        "ORDER BY c DESC LIMIT 8").fetchall()
    if fails:
        print("failures (re-run with --retry-failed to recover):")
        for f in fails:
            print(f"  {f['c']:5d}  {f['fail_reason']}")
    con.close()


# ---------------------------------------------------------------- embedding

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"          # Apple GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    import open_clip
    device = get_device()
    print(f"loading {MODEL_NAME} on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


def embed_all(batch_size=32):
    import numpy as np
    import torch
    import torch.nn.functional as F
    con = db_connect()
    todo = con.execute("""
        SELECT i.id FROM images i
        LEFT JOIN vectors v ON v.id = i.id
        WHERE i.state='stored' AND v.id IS NULL
    """).fetchall()
    if not todo:
        print("nothing to embed")
        con.close()
        return

    model, preprocess, _, device = load_model()
    print(f"embedding {len(todo)} images...")
    t0 = time.time()

    for i in range(0, len(todo), batch_size):
        chunk = [r["id"] for r in todo[i:i + batch_size]]
        tensors, ids = [], []
        for img_id in chunk:
            p = THUMBS / f"{img_id.replace(':', '_')}.webp"
            if not p.exists():
                continue
            try:
                tensors.append(preprocess(Image.open(p).convert("RGB")))
                ids.append(img_id)
            except Exception:
                continue
        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = F.normalize(feats, dim=-1)      # unit vectors -> cosine == dot
        feats = feats.cpu().numpy().astype(np.float32)

        con.executemany("INSERT OR REPLACE INTO vectors (id,vec) VALUES (?,?)",
                        [(i_, v.tobytes()) for i_, v in zip(ids, feats)])
        con.executemany("UPDATE images SET state='embedded' WHERE id=?",
                        [(i_,) for i_ in ids])
        con.commit()

        done = min(i + batch_size, len(todo))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"\r  {done}/{len(todo)}  ({rate:.0f} img/s)", end="", flush=True)
    print()
    con.close()


def build_matrix():
    import numpy as np
    """Collapse the vector table into one dense matrix the server mmaps.

    This is the prototype's stand-in for 'build the HNSW index and ship a
    Qdrant snapshot'. Same idea: index building is an offline batch job,
    serving just loads the artifact.
    """
    con = db_connect()
    rows = con.execute("""
        SELECT v.id, v.vec FROM vectors v
        JOIN images i ON i.id = v.id
        ORDER BY v.id
    """).fetchall()
    con.close()
    if not rows:
        print("no vectors yet")
        return
    ids = [r["id"] for r in rows]
    mat = np.frombuffer(b"".join(r["vec"] for r in rows),
                        dtype=np.float32).reshape(len(rows), EMBED_DIM)
    np.save(VEC_PATH, mat)
    IDS_PATH.write_text(json.dumps(ids))
    print(f"index built: {mat.shape[0]} vectors x {mat.shape[1]} dims "
          f"({mat.nbytes / 1e6:.1f} MB)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=60,
                    help="images to pull per seed topic")
    ap.add_argument("--topics", default=str(ROOT / "seeds.txt"))
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--crawl-only", action="store_true")
    ap.add_argument("--per-host", type=int, default=6,
                    help="concurrent fetches PER HOST (not global). 6 is polite.")
    ap.add_argument("--source", default="commons,openverse",
                    help="commons, openverse, enumerate (comma separated). "
                         "'enumerate' walks the full Commons file list instead "
                         "of searching -- the only path that scales past ~150k, "
                         "since Openverse sustains only ~0.4 req/s.")
    ap.add_argument("--enum-target", type=int, default=50000,
                    help="images this shard should enumerate from Commons")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1,
                    help="split topics across N parallel workers/IPs")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt rows that previously failed (mostly 429s)")
    args = ap.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        ap.error("require --shards >= 1 and 0 <= --shard < --shards")
    if args.per_host < 1 or args.enum_target < 1 or args.per_topic < 1:
        ap.error("concurrency and image targets must be positive")
    if set(args.source.split(",")) - {"commons", "openverse", "enumerate"}:
        ap.error("unknown source")

    if "sayeedshadab@gmail.com" in UA:
        print("!! Edit UA at the top of ingest.py with a real contact URL/email.")
        print("!! Wikimedia's User-Agent policy requires it and blocks generic UAs.\n")

    if not args.embed_only:
        topics = [l.strip() for l in Path(args.topics).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
        asyncio.run(crawl(topics, args.per_topic, args.per_host,
                          args.retry_failed,
                          tuple(x.strip() for x in args.source.split(",")),
                          args.shard, args.shards, args.enum_target))

    if not args.crawl_only:
        embed_all()
        build_matrix()


if __name__ == "__main__":
    main()
