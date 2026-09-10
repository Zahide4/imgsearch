#!/usr/bin/env python3
"""Bounded Openverse Class F discovery and human review; metadata only.

No Qdrant, storage backend, model, workflow or production crawler imports.
Page rows and cursor commit together. HTTP failures never exhaust a topic.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

import httpx

ROOT = Path(__file__).resolve().parent
API = "https://api.openverse.org/v1/"
SOURCES = ("flickr", "rawpixel")
LICENSES = ("cc0", "pdm", "by")
UA = "ImgSearch/0.2 (https://github.com/Zahide4/imgsearch)"


def plain(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value or ""))).strip()


def safe_url(value):
    p = urlsplit(value or "")
    return value if p.scheme == "https" and p.hostname and not p.username else ""


def normalize(x, topic, source, rank):
    """Validate API metadata independently of the request's licence filter."""
    if x.get("source") != source or source not in SOURCES:
        return None, "source"
    if x.get("license") not in LICENSES:
        return None, "license"
    if x.get("mature") or x.get("unstable__sensitivity"):
        return None, "sensitive_metadata"
    url = safe_url(x.get("url"))
    landing = safe_url(x.get("foreign_landing_url"))
    if not url or not landing or not x.get("id"):
        return None, "url_or_id"
    w, h = x.get("width"), x.get("height")
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or min(w, h) <= 0:
        return None, "dimensions_unknown"
    if min(w, h) < 480 or max(w, h) < 640:
        return None, "small"
    if max(w, h) / min(w, h) > 3:
        return None, "aspect"
    thumb = url
    p = urlsplit(url)
    match = re.fullmatch(r"/(?:[^/]+/)*(\d+)_([a-zA-Z0-9]+)(?:_[a-zA-Z0-9]+)?\.(jpg|png)", p.path)
    if source == "flickr" and (p.hostname == "live.staticflickr.com" or (p.hostname or "").endswith(".staticflickr.com")) and match:
        key = "flickr:" + match[1]
        # Derivative only. The API's _b URL is NOT a full-resolution original.
        thumb = urlunsplit((p.scheme, p.netloc, p.path.rsplit("/", 1)[0] + "/" + match[1] + "_" + match[2] + "_n." + match[3], "", ""))
    else:
        key = source + ":" + hashlib.sha256(landing.encode()).hexdigest()
    return dict(id="ov:" + x["id"], asset_key=key, title=plain(x.get("title"))[:300],
                creator=plain(x.get("creator"))[:200], license=x["license"],
                license_version=x.get("license_version"), license_url=safe_url(x.get("license_url")),
                attribution=plain(x.get("attribution")), source=source, source_url=landing,
                provider_url=url, thumb_origin=thumb, width=w, height=h,
                topic=topic, rank=rank, safety_status="not_scored"), None


def connect(path, topics):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (topic TEXT, source TEXT, page INTEGER DEFAULT 1,
          done INTEGER DEFAULT 0, PRIMARY KEY(topic,source));
        CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, asset_key TEXT UNIQUE, row TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages (topic TEXT, source TEXT, page INTEGER, raw INTEGER,
          added INTEGER, duplicates INTEGER, rejected TEXT, PRIMARY KEY(topic,source,page));
        CREATE TABLE IF NOT EXISTS requests (at REAL, topic TEXT, source TEXT, page INTEGER,
          status INTEGER, limits TEXT);
    """)
    signature = json.dumps({"topics": topics, "sources": SOURCES, "licenses": LICENSES, "schema": 1})
    old = db.execute("SELECT value FROM config WHERE key='signature'").fetchone()
    if old and old[0] != signature:
        db.close()
        raise ValueError("Pilot configuration changed. Use a new output directory to preserve the old sample.")
    with db:
        db.execute("INSERT OR IGNORE INTO config VALUES ('signature',?)", (signature,))
        db.execute("INSERT OR IGNORE INTO config VALUES ('pilot_id',?)", (str(uuid.uuid4()),))
        db.executemany("INSERT OR IGNORE INTO jobs(topic,source) VALUES (?,?)", [(t, s) for t in topics for s in SOURCES])
    return db


def commit_page(db, job, data):
    topic, source, page = job
    results = data.get("results")
    page_count = data.get("page_count")
    if not isinstance(results, list) or not isinstance(page_count, int) or data.get("page") != page:
        raise ValueError("Unexpected Openverse pagination response; cursor preserved")
    if len(results) > 20 or data.get("page_size") != 20:
        raise ValueError("Unexpected page size; cursor preserved")
    rejects, added, duplicates = Counter(), 0, 0
    with db:
        for i, x in enumerate(results):
            row, reason = normalize(x, topic, source, (page - 1) * 20 + i)
            if reason:
                rejects[reason] += 1
                continue
            n = db.execute("INSERT OR IGNORE INTO images VALUES (?,?,?)", (row["id"], row["asset_key"], json.dumps(row))).rowcount
            added += n
            duplicates += 1 - n
        db.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?)", (*job, len(results), added, duplicates, json.dumps(rejects)))
        db.execute("UPDATE jobs SET page=?, done=? WHERE topic=? AND source=?", (page + 1, int(not results or page >= page_count), topic, source))
    return added


def retry_at(value, now):
    try:
        return now + max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(now, parsedate_to_datetime(value).timestamp())
        except (TypeError, ValueError, OverflowError):
            return now + 60


def request_interval(headers):
    """Follow advertised burst limits with 10% headroom; never exceed 90/min."""
    intervals = []
    for key, value in headers.items():
        if key.startswith("x-ratelimit-limit-") and "burst" in key:
            match = re.fullmatch(r"([0-9]+)/min", value)
            if match and int(match[1]) > 0:
                intervals.append(60 / int(match[1]) * 1.1)
    return max([2 / 3, *intervals]) if intervals else 3.3


class Auth:
    def __init__(self, path):
        values = json.loads(path.read_text()) if path else {}
        self.token = values.get("access_token") or os.environ.get("OPENVERSE_TOKEN", "")
        self.client_id = values.get("client_id") or os.environ.get("OPENVERSE_CLIENT_ID", "")
        self.secret = values.get("client_secret") or os.environ.get("OPENVERSE_CLIENT_SECRET", "")
        if bool(self.client_id) != bool(self.secret):
            raise ValueError("Both Openverse client_id and client_secret are required")
        self.expires = float("inf") if self.token else 0

    def headers(self, client, refresh=False):
        if self.client_id and (refresh or time.time() >= self.expires):
            r = client.post(API + "auth_tokens/token/", data={"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.secret})
            if not r.is_success:
                raise RuntimeError(f"Openverse token endpoint HTTP {r.status_code}; credentials not logged")
            data = r.json()
            self.token = data["access_token"]
            self.expires = time.time() + float(data["expires_in"]) - 60
        return {"Authorization": "Bearer " + self.token} if self.token else {}


def crawl(db, args, auth):
    until = db.execute("SELECT value FROM config WHERE key='retry_at'").fetchone()
    if until and float(until[0]) > time.time():
        return "rate_limited", 75
    deadline = time.monotonic() + args.max_seconds
    last, requests, interval = 0, 0, 3.3
    with httpx.Client(headers={"User-Agent": UA}, timeout=45) as client:
        while db.execute("SELECT COUNT(*) FROM images").fetchone()[0] < args.target:
            if requests >= args.max_requests or time.monotonic() >= deadline:
                return "budget_paused", 75
            # Breadth first across both sources; no one topic consumes the sample.
            job = db.execute("SELECT topic,source,page FROM jobs WHERE done=0 ORDER BY page,rowid LIMIT 1").fetchone()
            if not job:
                return "sources_exhausted", 75
            topic, source, page = job
            headers = auth.headers(client)
            for attempt in range(3):
                if requests >= args.max_requests or time.monotonic() >= deadline:
                    return "budget_paused", 75
                time.sleep(max(0, interval - (time.monotonic() - last)))
                last = time.monotonic()
                requests += 1
                try:
                    r = client.get(API + "images/", params={"q": topic, "source": source, "license": ",".join(LICENSES), "page_size": 20, "page": page}, headers=headers)
                except httpx.TransportError as exc:
                    with db:
                        db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?)", (time.time(), *job, 0, json.dumps({"error": type(exc).__name__})))
                    print(f"transport_error={type(exc).__name__} topic={topic!r} page={page}", file=sys.stderr, flush=True)
                    if attempt == 2:
                        return "transport_error", 1
                    continue
                limits = {k: v for k, v in r.headers.items() if k.startswith("x-ratelimit") or k == "retry-after"}
                interval = request_interval(limits)
                with db:
                    db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?)", (time.time(), *job, r.status_code, json.dumps(limits)))
                print(json.dumps({"source": source, "topic": topic, "page": page, "http": r.status_code, "limits": limits}), file=sys.stderr, flush=True)
                if r.status_code == 401 and auth.client_id and attempt == 0:
                    headers = auth.headers(client, refresh=True)
                    continue
                if r.status_code == 429:
                    with db:
                        db.execute("INSERT OR REPLACE INTO config VALUES ('retry_at',?)", (str(retry_at(r.headers.get("retry-after"), time.time())),))
                    return "rate_limited", 75
                if r.status_code >= 500 and attempt < 2:
                    continue
                if r.status_code != 200:
                    # Includes pagination errors and expired credentials. Never count as no supply.
                    print(f"Openverse HTTP {r.status_code}: {r.text[:400]}", file=sys.stderr)
                    return "api_error", 1
                data = r.json()
                added = commit_page(db, job, data)
                print(f"added={added} total={db.execute('SELECT COUNT(*) FROM images').fetchone()[0]}", file=sys.stderr, flush=True)
                break
            else:
                return "api_error", 1
            if db.execute("SELECT COUNT(*) FROM images").fetchone()[0] >= args.target:
                return "target_reached", 0
            if any(k.startswith("x-ratelimit-available") and "sustained" in k and float(v) <= 0 for k, v in limits.items()):
                return "daily_allowance_used", 75
    return "target_reached", 0


def export(db, args, status):
    rows = [json.loads(x[0]) for x in db.execute("SELECT row FROM images ORDER BY rowid")]
    pages = db.execute("SELECT source,raw,added,duplicates,rejected FROM pages").fetchall()
    sources = {}
    for source in SOURCES:
        subset = [p for p in pages if p[0] == source]
        rejects = Counter()
        for p in subset:
            rejects.update(json.loads(p[4]))
        sources[source] = dict(pages=len(subset), raw=sum(p[1] for p in subset), candidates=sum(p[2] for p in subset), duplicates=sum(p[3] for p in subset), rejected=dict(rejects))
    pilot_id = db.execute("SELECT value FROM config WHERE key='pilot_id'").fetchone()[0]
    summary = dict(pilot_id=pilot_id, generated_at=datetime.now(timezone.utc).isoformat(), status=status,
                   target=args.target, required_review_count=10000, candidates=len(rows), sources=sources,
                   review_status="not_reviewed", decision="pending_human_review",
                   http_statuses=dict(db.execute("SELECT status,COUNT(*) FROM requests GROUP BY status")),
                   limitations=["Metadata sample; image bytes and true original resolution not verified.",
                                "Openverse provider URLs may be derivatives; source link required for original verification.",
                                "Metadata sensitivity filter only; no SigLIP safety or relevance scoring.",
                                "Exact source-asset dedup only; near-duplicates require human review."])
    for name, content in (("manifest.jsonl", "".join(json.dumps(r) + "\n" for r in rows)), ("summary.json", json.dumps(summary, indent=2))):
        temp = args.out / (name + ".tmp")
        temp.write_text(content)
        temp.replace(args.out / name)
    template = (ROOT / "plates-review.html").read_text()
    payload = json.dumps({"summary": summary, "rows": rows}).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    temp = args.out / "review.html.tmp"
    temp.write_text(template.replace("/*PILOT_DATA*/null", payload))
    temp.replace(args.out / "review.html")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "plates-pilot")
    ap.add_argument("--topics", type=Path, default=ROOT / "plates-topics.txt")
    ap.add_argument("--credentials", type=Path, help="Private JSON outside repository: client_id + client_secret, or access_token")
    ap.add_argument("--target", type=int, default=10000, help="Unique metadata candidates; may exceed by up to 19 to commit a full page")
    ap.add_argument("--max-requests", type=int, default=1000)
    ap.add_argument("--max-seconds", type=int, default=7200)
    ap.add_argument("--allow-anonymous", action="store_true", help="Explicitly allow a partial anonymous run within advertised limits")
    ap.add_argument("--report-only", action="store_true", help="Rebuild local report without network requests")
    args = ap.parse_args()
    if min(args.target, args.max_requests, args.max_seconds) < 1 or args.target > 10000:
        ap.error("Positive limits required; this pilot is capped at 10,000 candidates")
    topics = list(dict.fromkeys(t.strip() for t in args.topics.read_text().splitlines() if t.strip() and not t.lstrip().startswith("#")))
    if not topics:
        ap.error("No topics supplied")
    auth = Auth(args.credentials)
    if not args.report_only and not (auth.token or auth.client_id or args.allow_anonymous):
        ap.error("Credentials required for the 10k pilot; use --allow-anonymous only for a bounded partial probe")
    db = connect(args.out / "pilot.sqlite3", topics)
    status, code = "report_only", 0
    try:
        if not args.report_only:
            status, code = crawl(db, args, auth)
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    except (ValueError, RuntimeError, httpx.HTTPError) as exc:
        # Print type only: a malformed token response must never expose credentials.
        print(f"pilot_error={type(exc).__name__}; last page cursor preserved", file=sys.stderr)
        status, code = "error", 1
    finally:
        export(db, args, status)
        db.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
