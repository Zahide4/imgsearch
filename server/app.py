#!/usr/bin/env python3
"""Cloud search: SigLIP ONNX text encoder, Qdrant index, direct CDN images."""
import asyncio
import os
import re
import string
import time
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import NamedTuple
from urllib.parse import urlsplit

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
LOCAL_SPARSE = os.getenv('LOCAL_SPARSE', '').lower() in ('1', 'true', 'yes')
DIM, MAXLEN, PAD_ID = 768, 64, 1
app = FastAPI(title='imgsearch')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET'], allow_headers=['*'])
S = {'sess': None, 'tok': None, 'qc': None, 'cache': OrderedDict(),
     'results': OrderedDict(), 'lock': None, 'stats': (0, 0),
     'rankings': OrderedDict(), 'anchors': OrderedDict(), 'ranking_failures': OrderedDict(),
     'building': {}, 'build_slots': None}

# Ranked pages. Paging used to run a fresh fused query for every page, with a
# candidate pool that grew with the offset: 21 disk-bound queries to read one
# search to its end, and -- because each pool was different -- pages that could
# repeat or skip an image at their seams. Now page one is answered exactly as
# before, and alongside it one cheap query ranks the app's useful depth: eight
# pages of 48, or 384 results. Only the point ids and scores are kept; every
# later app page is a slice of that one order plus a
# payload lookup for just its own points. Page one's points are pinned to the
# head of the order, so the first page a client saw is never contradicted.
RANKED_DEPTH = 384
# Search results change only when the index does, and a repeated search is by
# far the cheapest one to serve: an uncached search costs ~90k disk reads.
RESULT_TTL = 1800
# Page-one search effort. Measured on the live index with 24 real queries
# against a high-effort reference (ef 256, 3x rescoring): ef 128 / 2x kept
# 99.4% of the top 48 at a median 2.05 s; ef 96 / 1.5x keeps 99.1% at 1.34 s,
# because every visited graph node is a disk read on this 8 GB box.
PAGE_EF = 96
PAGE_OVERSAMPLING = 1.5
RESULT_MAX = 512
RANKING_TTL = 1800
RANKING_MAX = 96
# A failed deep build must not be restarted by every page request. The normal
# page query remains available during this cooldown.
RANKING_FAILURE_TTL = 600
# Deeper rankings run at most this many at a time, so a burst of new searches
# cannot pile full-depth queries onto Qdrant in front of everyone's page one.
RANKING_BUILDS = 2
# How long a deeper page waits for its search's ranking before it gives up and
# runs the old single-page query instead. Never an error for the client.
RANKING_WAIT = 25
_PUNCT = str.maketrans('', '', string.punctuation)
ALLOWED_LICENSES = {'public_domain', 'attribution', 'share_alike'}

# Advanced filters: orientation, minimum resolution, source and file format.
# They read only what every point already carries -- image_id, width, height,
# mime and full_url -- so they need no migration and no new payload index.
# Licence and safety stay Qdrant conditions (both indexed). The advanced ones
# are applied to a deep, cheap candidate order whose payload is limited to
# those five fields, then the matches become the search's ranking. A point
# whose value is unknown (The Met and Wellcome record no pixel size) never
# matches a filter on that value; with no filter, nothing is ever excluded.
ALLOWED_ORIENTATIONS = {'landscape', 'portrait', 'square'}
ALLOWED_SOURCES = {'commons', 'openverse', 'nasa', 'met', 'wellcome'}
ALLOWED_FILE_TYPES = {'jpeg', 'png', 'webp'}
# Minimum pixel counts, not edge lengths, so orientation never matters:
# 4K UHD is 3840x2160 (8.3 MP) and 8K is 7680x4320 (33 MP).
MIN_PIXELS = {'': 0, '2mp': 2_000_000, '4k': 8_000_000, '8k': 30_000_000}
# Within 10% of 1:1 reads as square.
SQUARE_TOLERANCE = 1.1
# Candidates screened for a filtered search. Deep enough that ordinary filters
# fill all eight pages; a very narrow combination honestly returns fewer.
FILTER_DEPTH = 1200
FILTER_FIELDS = ['image_id', 'width', 'height', 'mime', 'full_url']
_SOURCE_PREFIXES = {'commons': 'commons', 'ov': 'openverse', 'openverse': 'openverse',
                    'nasa': 'nasa', 'met': 'met', 'wellcome': 'wellcome'}
_MIME_TYPES = {'image/jpeg': 'jpeg', 'image/jpg': 'jpeg', 'image/png': 'png', 'image/webp': 'webp'}
_EXTENSIONS = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.webp': 'webp'}


class FilterSpec(NamedTuple):
    orientations: tuple = ()
    min_resolution: str = ''
    sources: tuple = ()
    file_types: tuple = ()

    @property
    def active(self):
        return bool(self.orientations or self.min_resolution or self.sources or self.file_types)


def source_of(image_id):
    return _SOURCE_PREFIXES.get(str(image_id or '').split(':', 1)[0].lower(), '')


def dimensions(payload):
    try:
        width, height = int(payload.get('width') or 0), int(payload.get('height') or 0)
    except (TypeError, ValueError):
        return 0, 0
    return (width, height) if width > 0 and height > 0 else (0, 0)


def orientation_of(width, height):
    if width <= 0 or height <= 0:
        return ''
    ratio = width / height
    if ratio > SQUARE_TOLERANCE:
        return 'landscape'
    if ratio < 1 / SQUARE_TOLERANCE:
        return 'portrait'
    return 'square'


def file_type_of(mime, url):
    """The recorded MIME type when there is one, else the file's extension."""
    kind = _MIME_TYPES.get(str(mime or '').lower().split(';', 1)[0].strip())
    if kind:
        return kind
    try:
        suffix = PurePosixPath(urlsplit(str(url or '')).path).suffix.lower()
    except ValueError:
        return ''
    return _EXTENSIONS.get(suffix, '')


def matches(payload, spec):
    payload = payload or {}
    if spec.sources and source_of(payload.get('image_id')) not in spec.sources:
        return False
    if spec.orientations or spec.min_resolution:
        width, height = dimensions(payload)
        if spec.orientations and orientation_of(width, height) not in spec.orientations:
            return False
        if spec.min_resolution and width * height < MIN_PIXELS[spec.min_resolution]:
            return False
    if spec.file_types and file_type_of(payload.get('mime'), payload.get('full_url')) not in spec.file_types:
        return False
    return True


def choices(raw, allowed, label):
    values = tuple(sorted({value.strip() for value in raw.split(',') if value.strip()}))
    if set(values) - allowed:
        raise HTTPException(400, 'Unknown %s filter' % label)
    return values

# Adult-query refusal. The image safety filter is query-blind: it trims
# globally-high-scoring images, so a hostile query retrieves the most similar
# of whatever passed (measured: "porn" top-48 median safety 0.0001 against a
# 0.005 cutoff). The fix asks the other question -- "is this QUERY adult?" --
# after embedding (the vector already exists) and before Qdrant.
#
# Two layers, because neither works alone (testdrive/calibrate_query_gate.py):
# max-cosine to adult prompts cannot separate ("penis" 0.8253 scores below
# "classical nude sculpture" 0.8303), so an embedding-only gate either misses
# the exact complaint terms or eats legitimate art. Hence:
#   1. CORE_TERMS: whole-word match on unambiguous terms. Deliberately NOT
#      included: nude/naked (classical sculpture, mole rats), cock/ass/tit
#      (rooster, donkey, bird), breast (food, anatomy), nipple (grease
#      nipple), cracker (food), foursome (golf), tranny (transmission),
#      fag (cigarette), colored (civil-rights history), vibrator
#      (construction equipment), stripper (paint stripper), niger (the
#      country -- a deliberate non-match), abo (blood group), labia
#      (possible taxonomy collision). Deliberately included despite edge
#      uses: tits (birders write "great tits"), dick/pussy (names, cats),
#      cum ("cum laude"), retard (the verb), chink (the idiom), dyke
#      (vs dike), snuff (tobacco tins) -- image-search intent is
#      overwhelmingly the slur/slang, and the refusal message offers
#      recourse. Two legit phrases are carved out in REFUSAL_EXCEPTIONS
#      instead of dropping their terms. Tune from the refusal log, not taste.
#   2. Cosine backstop at 0.845: catches paraphrases ("naked girl" 0.8699,
#      "explicit sex" 0.9332, bare "naked" 0.8493) with zero legitimate
#      refusals on the 27-query battery (legit max 0.8303 -- margin 0.015,
#      thin; watch the refusal log for false positives). Re-run the
#      calibrator at any model swap.
REFUSAL_MESSAGE = ("We don't have that as it's NSFW. "
                   "If your query is falsely flagged, let us know.")
REFUSAL_COSINE = 0.845
REFUSAL_CORE = frozenset({
    # Explicit sexual content: industry, anatomy, acts, paraphilias, sites.
    'porn', 'porno', 'pornography', 'pornographic', 'pornhub', 'xvideos',
    'xhamster', 'redtube', 'youporn', 'brazzers', 'onlyfans', 'rule34',
    'xxx', 'hentai', 'erotic', 'erotica', 'nsfw', 'nsfl',
    'penis', 'penile', 'vagina', 'vaginal', 'vulva', 'clitoris', 'clit',
    'testicle', 'testicles', 'scrotum', 'semen', 'sperm', 'ejaculation',
    'ejaculate', 'ejaculating', 'orgasm', 'orgasmic', 'masturbation',
    'masturbate', 'masturbating', 'cum', 'cumming', 'cumshot', 'jizz',
    'blowjob', 'handjob', 'boob', 'boobs', 'tits', 'titties', 'pussy',
    'dick', 'dicks', 'cunt', 'orgy', 'gangbang', 'threesome', 'bukkake',
    'anal', 'dildo', 'fleshlight', 'upskirt', 'downblouse', 'creepshot',
    'lolicon', 'shotacon', 'snuff', 'incest', 'bestiality', 'rape',
    'bdsm', 'fetish', 'slut', 'whore', 'milf', 'dilf',
    'shemale', 'shemales', 'ladyboy', 'ladyboys',
    # Racial, ethnic, religious and anti-LGBT slurs.
    'nigger', 'niggers', 'nigga', 'niggas', 'niggah',
    'pajeet', 'paki', 'chink', 'chinks', 'chinky', 'gook', 'gooks',
    'spic', 'spics', 'spick', 'kike', 'kikes', 'yid',
    'coon', 'coons', 'wog', 'wogs', 'darkie', 'darkies', 'redskin',
    'squaw', 'injun', 'towelhead', 'towelheads', 'raghead', 'ragheads',
    'beaner', 'beaners', 'wetback', 'wetbacks', 'camel jockey', 'muzzie',
    'honky', 'honkies',
    'faggot', 'faggots', 'dyke', 'dykes',
    'retard', 'retarded',
    # Naked/nude phrases: bare "naked"/"nude" can never be core terms (they
    # would nuke "classical nude sculpture"), so the exact complaint shapes
    # are listed explicitly while the cosine backstop (0.845) holds the bare
    # words ("naked" calibrates at 0.8493). Child combinations are explicit
    # unconditionally: no cleverness where minors are concerned.
    'naked kids', 'nude kids', 'naked children', 'nude children',
    'naked child', 'nude child', 'naked baby', 'nude baby',
    'naked boy', 'nude boy', 'naked girl', 'nude girl',
    'naked teen', 'nude teen', 'naked women', 'nude women',
    'naked men', 'nude men', 'naked body', 'female nude', 'male nude',
})
# Whole-word terms with one legitimate phrase each. The term still fires
# everywhere else; the phrase passes. ("sperm whale" is Class B fauna;
# "spic and span" is cleaning products.)
REFUSAL_EXCEPTIONS = {
    'sperm': ('sperm whale',),
    'spic': ('spic and span',),
    # Slur terms with live legitimate uses (measured, not guessed):
    # Dutch water infrastructure is "dyke" in English; "chink of light" /
    # "chink in the armour" are idioms. The cosine backstop cannot cover
    # these -- it is calibrated on adult concepts, which slurs do not score
    # against -- so the phrases are carved out while the bare terms stay
    # refused. Honest cost, per review: this list never ends; the refusal
    # log must be READ for new collisions, not just written.
    'chink': ('chink of light', 'chink in the armour', 'chink in the armor'),
    'dyke': ('dyke landscape', 'dyke landscapes', 'sea dyke', 'dutch dyke'),
}
ADULT_PROMPTS = [
    'pornography',
    'explicit sexual activity',
    'a naked person in a sexual pose',
    'a photograph of exposed genitalia',
    'erotic imagery',
]
_ADULT = None  # frozen adult-concept vectors, loaded with the text tower

# Commons is an educational repository with no content policy of the kind a
# creative tool needs, and enumeration makes the proportion worse: the curated
# prototype came from 481 topics, while the 10M build walks the whole
# namespace. cloud_corpus.py scores every image against SigLIP's text tower at
# crawl time and stores the result in `safety`; this is where it takes effect.
#
# Calibrated in testdrive/calibrate_safety.py against the corpus's own worst
# material rather than a guess: search FOR the hostile content, score what
# comes back, then measure what legitimate searches lose.
#
# Loss is concentrated entirely in body-adjacent searches. At this threshold,
# landscape, architecture, street food, war memorial and marble sculpture lose
# 0.0% of their results; anatomy and ballet lose 18%. Dropping to 0.001 would
# catch more (86% of hostile hits rather than 73%) at the cost of a third of
# every "ballet dancer" search, which is too visible a regression for an
# innocent query.
#
# 0.005 sits far below the hostile median (0.0396) and far below the genuinely
# explicit images this corpus turned out to contain (0.26 to 0.92).
#
# Raise it to filter less, lower it to filter more; include_sensitive=true
# bypasses it per request.
SAFETY_MAX = float(os.getenv('SAFETY_MAX', '0.005'))


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
        # Cloud inference builds the sparse query on Qdrant's side and exists
        # only on the managed service. Self-hosted has to build it here. The
        # switch is an env var rather than a rewrite so production keeps its
        # current path until the index actually moves.
        S['qc'] = AsyncQdrantClient(url=os.environ['QDRANT_URL'], api_key=os.getenv('QDRANT_API_KEY'),
                                    cloud_inference=not LOCAL_SPARSE, timeout=20)


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


def ensure_adult_vectors():
    """Freeze adult-concept vectors with the production text tower.

    Idempotent; runs inside the embed lock's thread. A missing tower means
    fail-open (returns False): search must never break because refusal
    infrastructure is not loaded -- same philosophy as unscored rows passing
    the image filter.
    """
    global _ADULT
    if _ADULT is not None or S['sess'] is None or S['tok'] is None:
        return _ADULT is not None
    vecs = []
    for prompt in ADULT_PROMPTS:
        ids = np.array([S['tok'].encode(prompt).ids], dtype=np.int64)
        v = S['sess'].run(['pooler_output'], {'input_ids': ids})[0][0].astype(np.float64)
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(v).all() or not np.isfinite(nrm) or nrm <= 0:
            return False
        vecs.append(v / nrm)
    _ADULT = np.array(vecs)
    return True


def refusal_reason(text, vector):
    """Why this query is refused, or '' to let it through.

    `text` is already canon()icalised: lowercase, no punctuation. Core terms
    match on whole words only, so "titmouse" never trips on "tit".
    """
    padded = ' ' + text + ' '
    for term in REFUSAL_CORE:
        if ' ' + term + ' ' not in padded:
            continue
        exc = REFUSAL_EXCEPTIONS.get(term)
        if exc and any(' ' + phrase + ' ' in padded for phrase in exc):
            continue
        return 'core:' + term
    if _ADULT is not None:
        sims = np.asarray(vector, dtype=np.float64) @ _ADULT.T
        if bool(np.isfinite(sims).all()) and float(sims.max()) >= REFUSAL_COSINE:
            return 'cosine:%.4f' % float(sims.max())
    return ''


def sign(key):
    """Turn a stored object key into a URL the client can fetch."""
    try:
        import storage
        return storage.presigned_url(key)
    except Exception:
        # Signing failing must not empty the whole result set; the caller
        # falls through to the proxy for this row.
        return ''


# Wikimedia now rejects thumbnail widths outside its standard buckets with a
# 400 on hotlinks (20/40/60/120/250/330/500/960/1280/1920/3840), so the only
# useful large rendition is the 3840 bucket. It is also the whole reason the
# app can prepare a drag in ~1-5 MB instead of pulling a 20-30 MB original.
RENDITION_WIDTH = 3840
_THUMB_HOSTS = {'thumb.wikimedia.org', 'upload.wikimedia.org'}
# Animated GIF and SVG renditions are stills of the original; hand those
# originals over instead of silently flattening them.
_THUMB_STATIC_TAILS = ('.gif', '.svg', '.svg.png')


def rendition_url(payload):
    """The 4K rendition for this row, or '' when the original should be used.

    Only worth it when the original is actually bigger than 4K: requesting
    the 3840 bucket for a smaller image makes Wikimedia upscale it, measured
    at 2.3 MB where the 1.3 MB original was both smaller and truer. Skipped
    formats and non-Wikimedia sources fall through to the original, which the
    app already fetches today.
    """
    try:
        width = int(payload.get('width') or 0)
    except (TypeError, ValueError):
        return ''
    if width <= RENDITION_WIDTH:
        return ''
    return (_swap_thumb_width(payload.get('thumb_origin', ''))
            or _commons_thumb_from_original(payload.get('full_url', '')))


def _swap_thumb_width(url):
    """Replace the width token of an existing Wikimedia thumbnail URL."""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or parts.netloc.lower() not in _THUMB_HOSTS:
        return ''
    if '/thumb/' not in parts.path:
        return ''
    # The width token lives in the FINAL path segment, possibly after a
    # MediaWiki prefix (lossy-page1-500px-...), and is the first `\d+px-`
    # there -- the original's own name follows it.
    directory, _, name = parts.path.rpartition('/')
    match = re.search(r'\d+px-', name)
    if not match or parts.path.lower().endswith(_THUMB_STATIC_TAILS):
        return ''
    swapped = name[:match.start()] + f'{RENDITION_WIDTH}px-' + name[match.end():]
    return urlunsplit((parts.scheme, parts.netloc, f'{directory}/{swapped}', '', ''))


def _commons_thumb_from_original(url):
    """Canonical Commons thumbnail for rows with no usable thumb_origin."""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or parts.netloc.lower() != 'upload.wikimedia.org':
        return ''
    marker = '/wikipedia/commons/'
    if not parts.path.startswith(marker) or '/thumb/' in parts.path:
        return ''
    name = parts.path[len(marker):].rsplit('/', 1)[-1]
    if not name or name.lower().endswith(_THUMB_STATIC_TAILS):
        return ''
    path = f'{marker}thumb/{parts.path[len(marker):]}/{RENDITION_WIDTH}px-{name}'
    return urlunsplit((parts.scheme, parts.netloc, path, '', ''))


def sparse_query(text):
    """The BM25 half of the query, built here or by Qdrant Cloud."""
    if LOCAL_SPARSE:
        import sparse
        return sparse.query(text)
    return models.Document(text=text, model=SPARSE_MODEL)


def search_filter(licenses, include_sensitive):
    """License and safety conditions for the Qdrant query.

    Safety is expressed as must_not(safety >= SAFETY_MAX) rather than
    must(safety < SAFETY_MAX), and the difference matters: a point with NO
    `safety` field fails a `must` range condition and would be dropped. The
    435k rows crawled before scoring existed carry no such field, so the
    positive form would return an empty index. Unscored means unfiltered,
    which is honest -- those rows were never examined.
    """
    must = []
    if licenses:
        must.append(models.FieldCondition(
            key='license_class', match=models.MatchAny(any=list(licenses))))
    must_not = []
    if not include_sensitive:
        must_not.append(models.FieldCondition(
            key='safety', range=models.Range(gte=SAFETY_MAX)))
    return models.Filter(must=must, must_not=must_not) if (must or must_not) else None


def result_from(hit):
    """Where the browser loads this thumbnail from.

    Three cases, in preference order, and the order matters:

    1. `cdn` with no scheme is an object KEY in our own private bucket -- a
       real 384px derivative we generated. Best, when it exists. It gets
       signed on the way out because the address has an expiry in it.
    2. `thumb_origin` is the source's own thumbnail, generated by Wikimedia's
       API during the crawl and served from their CDN. This is what the
       standalone build produces, and it costs nothing to store.
    3. `cdn` WITH a scheme is a legacy wsrv.nl proxy URL, left in ~12k rows by
       the old proxy mode. It is deliberately last: wsrv.nl is the free shared
       proxy that produced every serving problem the prototype hit, and 2.7%
       of rows have no thumb_origin at all, so it stays as their fallback
       rather than being cleared.
    """
    p = hit.payload
    origin = p.get('thumb_origin', '')
    cdn = p.get('cdn', '')
    if cdn and not cdn.startswith('http'):
        thumb = sign(cdn) or origin
    else:
        thumb = origin or cdn
    return dict(id=p.get('image_id'), title=p.get('title', ''), creator=p.get('creator', ''),
                license=p.get('license', ''), license_class=p.get('license_class', ''),
                license_url=p.get('license_url', ''), source_url=p.get('source_url', ''),
                full_url=p.get('full_url', ''), width=p.get('width', 0), height=p.get('height', 0),
                thumb=thumb, rendition_url=rendition_url(p), score=round(float(hit.score), 4))


def fresh(cache, key, ttl):
    entry = cache.get(key)
    if entry is None:
        return None
    if time.monotonic() - entry[0] >= ttl:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return entry[1]


async def fused_query(text, vector, flt, offset, limit, with_payload=True):
    """Dense + BM25 fused by RRF; dense-only if the sparse half fails."""
    candidates = max(100, (offset + limit) * 2)
    dense_prefetch = models.Prefetch(
        query=vector, using='image', limit=candidates, filter=flt,
        params=models.SearchParams(
            hnsw_ef=PAGE_EF,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=PAGE_OVERSAMPLING),
        ),
    )
    try:
        # The BM25 query vector is built by Qdrant Cloud inference. If that is
        # rate-limited or unavailable, fall back to dense-only rather than
        # failing the whole request: the semantic half needs no inference and
        # is the primary ranking signal. Losing exact-name matching degrades
        # results; returning 503 loses search entirely.
        return (await S['qc'].query_points(
            COLLECTION,
            prefetch=[
                dense_prefetch,
                models.Prefetch(
                    query=sparse_query(text),
                    using='bm25', limit=candidates, filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, offset=offset, with_payload=with_payload,
        )).points
    except Exception:
        try:
            return (await S['qc'].query_points(
                COLLECTION, query=vector, using='image', limit=limit,
                offset=offset,
                with_payload=with_payload, query_filter=flt,
                # `search_params`, not `params`: the client rejects unknown
                # keywords, which turned every dense-only fallback into a 503.
                search_params=models.SearchParams(
                    hnsw_ef=PAGE_EF,
                    quantization=models.QuantizationSearchParams(rescore=True, oversampling=PAGE_OVERSAMPLING),
                ),
            )).points
        except Exception:
            raise HTTPException(503, 'Search is temporarily unavailable. Please try again.')


async def fast_ranking_query(text, vector, flt, depth=RANKED_DEPTH, with_payload=False):
    """One inexpensive approximate order for pages after the first.

    Page one keeps full-precision rescoring. Ranking 1,100 later results with
    that same path made Qdrant read thousands of original vectors from disk and
    regularly exceeded its 20-second timeout. The app only exposes 384 results,
    so later pages use the quantized index directly at that depth, with one
    candidate per result instead of two. The accurate page one is pinned back
    onto the front by build_ranking.
    """
    params = models.SearchParams(
        hnsw_ef=64,
        quantization=models.QuantizationSearchParams(rescore=False),
    )
    dense = models.Prefetch(
        query=vector, using='image', limit=depth, filter=flt, params=params,
    )
    try:
        return (await S['qc'].query_points(
            COLLECTION,
            prefetch=[
                dense,
                models.Prefetch(
                    query=sparse_query(text), using='bm25', limit=depth, filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=depth, with_payload=with_payload, timeout=20,
        )).points
    except Exception:
        try:
            # Sparse inference is allowed to fail without losing pagination.
            return (await S['qc'].query_points(
                COLLECTION, query=vector, using='image', limit=depth,
                with_payload=with_payload, query_filter=flt, search_params=params, timeout=20,
            )).points
        except Exception:
            raise HTTPException(503, 'Search is temporarily unavailable. Please try again.')


def pin_first_page(anchor, ranked):
    """The full order with the page one clients already saw at its head."""
    seen = {pid for pid, _ in anchor}
    return (list(anchor) + [entry for entry in ranked if entry[0] not in seen])[:RANKED_DEPTH]


async def build_ranking(rkey, text, vector, flt):
    started = time.perf_counter()
    if S['build_slots'] is None:
        S['build_slots'] = asyncio.Semaphore(RANKING_BUILDS)
    async with S['build_slots']:
        points = await fast_ranking_query(text, vector, flt)
    ranked = [(point.id, float(point.score)) for point in points]
    ranking = pin_first_page(fresh(S['anchors'], rkey, RANKING_TTL) or [], ranked)
    remember(S['rankings'], rkey, (time.monotonic(), ranking), RANKING_MAX)
    S['ranking_failures'].pop(rkey, None)
    print('ranking ready q=%r ids=%d ms=%.1f' %
          (text, len(ranking), (time.perf_counter() - started) * 1000), flush=True)
    return ranking


async def build_filtered_ranking(rkey, text, vector, flt, spec):
    """Every match for an advanced filter, best first, as one ranking.

    The deep candidate order is the cheap unrescored one; alongside it the
    accurate rescored top 100 runs, and its matches are pinned to the head, so
    the first page of a filtered search is ranked as well as an unfiltered one.
    Only the five fields the filters read come back with the candidates.
    """
    started = time.perf_counter()
    selector = models.PayloadSelectorInclude(include=FILTER_FIELDS)
    if S['build_slots'] is None:
        S['build_slots'] = asyncio.Semaphore(RANKING_BUILDS)
    async with S['build_slots']:
        accurate, deep = await asyncio.gather(
            fused_query(text, vector, flt, 0, 100, with_payload=selector),
            fast_ranking_query(text, vector, flt, depth=FILTER_DEPTH, with_payload=selector),
            return_exceptions=True)
    if isinstance(deep, BaseException):
        raise deep
    head = [] if isinstance(accurate, BaseException) else accurate
    anchor = [(point.id, float(point.score)) for point in head if matches(point.payload, spec)]
    ranked = [(point.id, float(point.score)) for point in deep if matches(point.payload, spec)]
    ranking = pin_first_page(anchor, ranked)
    remember(S['rankings'], rkey, (time.monotonic(), ranking), RANKING_MAX)
    S['ranking_failures'].pop(rkey, None)
    print('filtered ranking ready q=%r spec=%r matched=%d of %d ms=%.1f' %
          (text, tuple(spec), len(ranking), len(deep), (time.perf_counter() - started) * 1000), flush=True)
    return ranking


def ensure_ranking(rkey, text, vector, flt, spec=FilterSpec()):
    """The one in-flight build for this search, started if there is none."""
    if fresh(S['ranking_failures'], rkey, RANKING_FAILURE_TTL) is not None:
        return None
    task = S['building'].get(rkey)
    if task is None:
        build = (build_filtered_ranking(rkey, text, vector, flt, spec) if spec.active
                 else build_ranking(rkey, text, vector, flt))
        task = asyncio.create_task(build)
        S['building'][rkey] = task

        def finished(done):
            if S['building'].get(rkey) is done:
                S['building'].pop(rkey, None)
            if not done.cancelled() and done.exception() is not None:
                # Nobody may be awaiting a background build. Remember failure
                # so the app's remaining page requests cannot restart it.
                remember(S['ranking_failures'], rkey, (time.monotonic(), True), RANKING_MAX)
                print('ranking failed q=%r: %r' % (text, done.exception()), flush=True)
        task.add_done_callback(finished)
    return task


async def ranked_page(ranking, offset, limit):
    """One page of a ranking, with payloads read for just its own points."""
    window = ranking[offset:offset + limit]
    if not window:
        return [], False
    records = await S['qc'].retrieve(COLLECTION, ids=[pid for pid, _ in window],
                                     with_payload=True, with_vectors=False)
    by_id = {record.id: record for record in records}
    # A point deleted since the ranking was built is simply left out.
    hits = [SimpleNamespace(payload=by_id[pid].payload, score=score) for pid, score in window if pid in by_id]
    return [result_from(hit) for hit in hits], len(ranking) > offset + limit


@app.get('/api/search')
async def search(request: Request, q: str = Query(..., max_length=300),
                 limit: int = Query(60, ge=1, le=100), license_class: str = '',
                 offset: int = Query(0, ge=0, le=1000),
                 orientation: str = '', min_resolution: str = '',
                 source: str = '', file_type: str = '',
                 include_sensitive: bool = Query(
                     False, description='Return images the safety filter would '
                                        'exclude. Medical, anatomical and fine-art '
                                        'searches are legitimate and the filter '
                                        'cannot tell them apart perfectly.')):
    start = time.perf_counter()
    text = canon(q)
    licenses = choices(license_class, ALLOWED_LICENSES, 'license')
    if min_resolution not in MIN_PIXELS:
        raise HTTPException(400, 'Unknown resolution filter')
    spec = FilterSpec(choices(orientation, ALLOWED_ORIENTATIONS, 'orientation'), min_resolution,
                      choices(source, ALLOWED_SOURCES, 'source'),
                      choices(file_type, ALLOWED_FILE_TYPES, 'file type'))
    if not text:
        return {'results': [], 'ms': 0, 'has_more': False}
    if S['qc'] is None:
        raise HTTPException(503, 'Search is not configured')
    key = (text, limit, offset, licenses, include_sensitive, spec)
    cached = S['results'].get(key)
    if cached and time.monotonic() - cached[0] < RESULT_TTL:
        S['results'].move_to_end(key)
        return {**cached[1], 'cached': True, 'ms': 0, 'timing': {'embed_ms': 0, 'ann_ms': 0}}
    rkey = (text, licenses, include_sensitive, spec)
    cached_ranking = None
    if offset > 0:
        cached_ranking = fresh(S['rankings'], rkey, RANKING_TTL)
        if cached_ranking is not None and offset < len(cached_ranking):
            # Already ranked (and already past the refusal check, which is the
            # only way a ranking gets built): no embedding, no search.
            return await serve_ranked(key, cached_ranking, offset, limit, start, 0.0)
    # One embedding at a time on a small CPU. Waiting clients that have
    # disconnected do not consume another expensive inference slot.
    async with S['lock']:
        if await request.is_disconnected():
            raise HTTPException(499, 'Search cancelled')
        vector = await asyncio.to_thread(embed, text)
        await asyncio.to_thread(ensure_adult_vectors)
    embedded = time.perf_counter()
    reason = refusal_reason(text, vector)
    if reason:
        # Hard refusal, not an empty result: the app shows REFUSAL_MESSAGE
        # instead of searching. Logged (refused queries only, for tuning --
        # this is the interim false-flag channel until a report endpoint
        # exists) and cached like any other answer. Applies even with
        # include_sensitive=true: that flag is for legitimate edge searches,
        # not adult intent.
        print('refused q=%r reason=%s' % (text, reason), flush=True)
        data = {'results': [], 'ms': round((embedded - start) * 1000, 1),
                'timing': {'embed_ms': round((embedded - start) * 1000, 1), 'ann_ms': 0},
                'refusal': REFUSAL_MESSAGE, 'has_more': False}
        remember(S['results'], key, (time.monotonic(), data), RESULT_MAX)
        return data
    flt = search_filter(licenses, include_sensitive)
    embed_ms = round((embedded - start) * 1000, 1)
    if spec.active:
        return await filtered_search(key, rkey, text, vector, flt, spec, offset, limit, start, embed_ms)
    # Start the cheap deep order alongside page one, not after it. On the live
    # disk-bound index this makes the ranking ready around the same time the
    # first page appears. Page one still uses the accurate rescored query below.
    early_build = None
    if offset == 0 and fresh(S['rankings'], rkey, RANKING_TTL) is None:
        early_build = ensure_ranking(rkey, text, vector, flt)
    if offset > 0 and cached_ranking is None:
        try:
            build = ensure_ranking(rkey, text, vector, flt)
            if build is not None:
                ranking = await asyncio.wait_for(asyncio.shield(build), RANKING_WAIT)
                if offset < len(ranking):
                    return await serve_ranked(key, ranking, offset, limit, start, embed_ms)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # Too slow or failed: the single-page query below still answers.
    # Fetch one extra result so the client can disable Next on the last page.
    # Qdrant's limit tops out at 100; callers asking for 100 still get the
    # legacy exact-limit behavior and simply report no continuation.
    fetch_limit = min(limit + 1, 100)
    hits = await fused_query(text, vector, flt, offset, fetch_limit)
    end = time.perf_counter()
    has_more = len(hits) > limit
    data = {'results': [result_from(hit) for hit in hits[:limit]], 'has_more': has_more,
            'ms': round((end-start)*1000, 1),
            'timing': {'embed_ms': embed_ms, 'ann_ms': round((end-embedded)*1000, 1)}}
    remember(S['results'], key, (time.monotonic(), data), RESULT_MAX)
    if offset == 0 and has_more:
        # Pin what this client now sees, then rank the rest behind the reply.
        anchor = [(hit.id, float(hit.score)) for hit in hits[:limit]]
        remember(S['anchors'], rkey, (time.monotonic(), anchor), RANKING_MAX)
        # Usually the concurrent build is still running and will read this
        # anchor itself. If it won the race, repair its head before any later
        # page can observe it.
        ranking = fresh(S['rankings'], rkey, RANKING_TTL)
        if ranking is not None:
            remember(S['rankings'], rkey,
                     (time.monotonic(), pin_first_page(anchor, ranking)), RANKING_MAX)
        if fresh(S['rankings'], rkey, RANKING_TTL) is None:
            ensure_ranking(rkey, text, vector, flt)
    elif offset == 0:
        # A genuinely short result set does not need a deep ranking.
        if early_build is not None and not early_build.done():
            early_build.cancel()
        S['rankings'].pop(rkey, None)
    return data


async def filtered_search(key, rkey, text, vector, flt, spec, offset, limit, start, embed_ms):
    """Any page of a filtered search: always a slice of its one ranking.

    A cached ranking was served before embedding; this waits for the build.
    If the build fails or runs past RANKING_WAIT, page one still answers from
    the accurate top 100 (fewer results, no continuation) and later pages ask
    the client to retry, rather than showing images that break the filter.
    """
    ranking = fresh(S['rankings'], rkey, RANKING_TTL)
    if ranking is not None:
        return await serve_ranked(key, ranking, offset, limit, start, embed_ms)
    build = ensure_ranking(rkey, text, vector, flt, spec)
    if build is not None:
        try:
            ranking = await asyncio.wait_for(asyncio.shield(build), RANKING_WAIT)
            return await serve_ranked(key, ranking, offset, limit, start, embed_ms)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
    if offset > 0:
        raise HTTPException(503, 'Search is temporarily unavailable. Please try again.')
    hits = await fused_query(text, vector, flt, 0, 100)
    kept = [hit for hit in hits if matches(hit.payload, spec)][:limit]
    end = time.perf_counter()
    data = {'results': [result_from(hit) for hit in kept], 'has_more': False,
            'ms': round((end - start) * 1000, 1),
            'timing': {'embed_ms': embed_ms, 'ann_ms': round((end - start) * 1000 - embed_ms, 1)}}
    # Short-lived on purpose: the next attempt should get the full ranking.
    # Stamped as nearly expired, so it lives ~20 s and the next try ranks fully.
    remember(S['results'], key, (time.monotonic() - (RESULT_TTL - 20), data), RESULT_MAX)
    return data


async def serve_ranked(key, ranking, offset, limit, start, embed_ms):
    fetched = time.perf_counter()
    results, has_more = await ranked_page(ranking, offset, limit)
    end = time.perf_counter()
    # `total` is exact for this search: every result it can page to.
    data = {'results': results, 'has_more': has_more, 'total': len(ranking),
            'ms': round((end - start) * 1000, 1),
            'timing': {'embed_ms': embed_ms, 'ann_ms': round((end - fetched) * 1000, 1)}}
    remember(S['results'], key, (time.monotonic(), data), RESULT_MAX)
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
