# Class F pilot

The first implementation step from `docs/allocation.html`: discover 10,000
unique plate candidates from Flickr and Rawpixel through Openverse, then judge
their usefulness by eye. This is a **metadata and visual-review pilot**, not an
embedding or production-ingestion job. No workflow has been dispatched.

## What is prepared

- `plates_pilot.py`: bounded, resumable discovery. Uses only `httpx` and the
  Python standard library. Never imports production ingestion, Qdrant, or S3.
- `plates-topics.txt`: 100 surface/atmosphere topics. Each page is taken in
  source relevance order; jobs rotate across topic/source pairs before going
  deeper. An insufficient source pool is reported, not padded with other sources.
- `plates-review.html`: report template with 24 images per page, source/topic/
  review filters, usable/unusable/broken labels, browser persistence and JSON
  label export/import. Previews use the provider CDN directly.
- `tests/test_plates_pilot.py`: licence/source boundaries, derivative identity,
  deduplication, atomic checkpoints, failed-response handling, retry deadlines,
  resume ordering and safe report rendering.

Every successful page records raw rows, exact duplicates, candidates, and
rejections by reason. SQLite commits rows and cursor together. Re-running the
same output directory resumes; changing its topic list is rejected. The
candidate target can overshoot by at most 19 so a whole page commits atomically.

HTTP errors are separate from supply exhaustion. Each response records status
and rate-limit headers. Requests are serial and follow the advertised burst
limit with 10% headroom, capped at 90/minute. Without a recognized limit header,
the interval is 3.3 seconds. Transient failures have at most three attempts.
A 429 stops with its Retry-After deadline
saved. Never use extra runners, identities, or IPs to evade an allowance.

## Credentials and launch

Register/verify an Openverse application using the
[official API documentation](https://api.openverse.org/v1/).
Keep the resulting credentials **outside this repository**, for example
`~/.config/imgsearch/openverse.json`, readable only by your account:

```json
{"client_id": "YOUR_CLIENT_ID", "client_secret": "YOUR_CLIENT_SECRET"}
```

The script exchanges these for a token and refreshes it when needed. It also
accepts `OPENVERSE_CLIENT_ID` + `OPENVERSE_CLIENT_SECRET`, or a temporary
`OPENVERSE_TOKEN`. Neither credentials nor authorization headers are written to
the database, manifest, logs or review report. Do not put secrets in the command
line or in chat. Only the local file path is needed to run it.

From `imgsearch-proto`:

```sh
.venv/bin/python testdrive/plates_pilot.py \
  --credentials "$HOME/.config/imgsearch/openverse.json" \
  --target 10000 --max-requests 1000 --max-seconds 7200
```

Output is under ignored `testdrive/plates-pilot/`:

| File | Purpose |
|---|---|
| `pilot.sqlite3` | Durable rows, topic cursors, request history and counters |
| `manifest.jsonl` | All unique metadata candidates |
| `summary.json` | Source yield, filtering reasons and actual run status |
| `review.html` | Standalone report, with external provider previews |

Run the same command again after a budget pause. Exit 0 means discovery reached
the requested count; **it does not mean the visual pilot passed**. Exit 75 means
incomplete (budget, rate limit or source exhaustion); 1 means error; 130 means
interrupted. Read `summary.json` for the reason. Requests stop at the daily
allowance rather than attempting to spend past it. The max-request budget is
per invocation; token requests are separate from search requests.

To regenerate the report without network access:

```sh
.venv/bin/python testdrive/plates_pilot.py --report-only
```

Open `review.html` directly in a browser. If that browser does not retain labels
on local files, serve **only the output directory** on localhost:

```sh
.venv/bin/python -m http.server 8768 --bind 127.0.0.1 \
  --directory testdrive/plates-pilot
```

Then open `http://127.0.0.1:8768/review.html`. Export label backups each session;
they are checked against the pilot ID when imported. Regenerating a report does
not change its pilot ID or image IDs.

## Measurements made during preparation — 10 September 2026

A live anonymous probe returned HTTP 200 and advertised **20 requests/minute,
200/day**, with a 20-result page and a 240-result query cap. Even at perfect
fill that permits only 4,000 returned rows/day. A subsequent authenticated
probe reported a **verified standard application**, **100 requests/minute** and
**10,000/day**. Page 13 still returned HTTP 401 with the explicit error
`pagination depth may not exceed 240 for authenticated requests`.
Authentication raises request throughput, not per-query depth. The existing
`ingest.py` hard-codes only three authenticated pages; this pilot instead follows
the API response's page count and preserves any pagination errors explicitly.

The authenticated account also accepts 50-row pages: a probe returned 50 rows
with `result_count=240` and `page_count=4`. A request for 100 returned HTTP 401
stating the maximum is 50. Thus the old claim that authentication cannot lift
the 20-row page size is outdated for this account. This pilot retains 20-row
pages to preserve existing cursors and breadth-first sampling; it also avoids
the mismatch between 240 reported results and four advertised 50-row pages.
These responses are recorded in `plates-auth-preflight.json` without credentials.

The four-request smoke run queried concrete texture and paper texture from
both providers: **80 rows returned, 79 candidates** (40 Flickr, 39 Rawpixel),
one aspect-ratio rejection, zero exact duplicates, four HTTP 200 responses.
See `plates-smoke-result.json`. This validates discovery and report plumbing,
**not a 98.75% visual keep rate**. No usability judgments were retained.

The live sample report is in ignored `testdrive/plates-smoke/review.html`.
The source images are loaded by the browser, not downloaded into the repository.

Preparation validation: all 30 Python tests passed, `node tests/search-ui.cjs` passed, and
`git diff --check` passes. Browser checks covered both providers' previews,
source filtering, pagination, and label persistence across reload; test labels
were cleared afterward. An existing search test was updated to expect the
source thumbnail already returned by the API, rather than the retired wsrv proxy.

## Review and decision

Judge whether each image is actually useful as a plate: clean surface or
atmosphere, no distracting subject or visible watermark, sufficient source
resolution, and useful variation relative to already-reviewed images. A broken
preview needs source inspection; it is not automatically rejected.

Complete the 10k discovery and review before making the allocation decision.
Under 50% usable supports reducing Class F to 200k and considering background
removal. At least 50% supports continuing source tests; it does not establish
that 500k distinct usable plates are bulk-reachable. Source breakdowns and
unreviewed-count bounds stay visible throughout review.

## Findings that must carry forward

1. **Openverse provider URLs are not guaranteed full originals.** Flickr often
   returns `_b` (1024px); Rawpixel returns `editor_1024` even when metadata names
   a much larger work. The pilot calls these `provider_url`, never `full_url`.
   The app's original-copy requirement needs a separately verified source path.
2. **Equal dimensions do not imply a compatible embedding space.** SigLIP 2 may
   fit the 768-dimensional collection schema, but its queries cannot be assumed
   compatible with existing SigLIP image vectors. Benchmark both image/text
   towers together; use a separate collection or re-embed the corpus before
   cutover, and re-freeze/recalibrate safety prompts. No model was changed here.
3. API-reported dimensions and licences are metadata, not fetched-byte checks.
   Missing sizes, NC/ND/SA licences and flagged sensitive rows are excluded.
   SigLIP safety, relevance calibration, near-duplicate scoring and full-original
   availability are not established by this pilot preparation.
4. Licence-category shards can overlap; deduplication is still needed. The
   allocation's claim that all eight are disjoint must not guide ingest IDs.

The pilot leaves the finished 435k build, live collection and macOS app
untouched. Human review is required after authenticated discovery completes.
