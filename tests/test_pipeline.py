import asyncio
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import cloud_corpus
import ingest


async def _resolved(value):
    """An awaitable that is already done, for stubbing httpx's async get."""
    return value
from cloud_corpus import clean_text, metadata, params_for, ranges, search_text
from cloud_corpus import category_jobs, flush_every, params_for_category, pending_count, shard_bounds

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('search_app', ROOT / 'server/app.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_ranges_disjoint_and_complete(self):
        jobs = [pair for w in range(4) for pair in ranges(w, 4)]
        ordered = sorted(jobs)
        self.assertEqual(len(jobs), len(set(jobs)))
        self.assertEqual(ordered[0][0], '')
        self.assertIsNone(ordered[-1][1])
        for a,b in zip(ordered, ordered[1:]): self.assertEqual(a[1], b[0])

    def test_imageinfo_continuation_preserves_range(self):
        p = params_for('Ph', 'Pi', {'iicontinue': 'file|timestamp', 'continue': '||'})
        self.assertEqual((p['gaifrom'], p['gaito']), ('Ph', 'Pi'))
        self.assertEqual(p['iicontinue'], 'file|timestamp')

    def test_shard_refactor_preserves_ranges(self):
        full = shard_bounds()
        self.assertEqual(len(full), 1 + 10 + 26 + 26 * 26)
        self.assertEqual(ranges(0, 1), full)
        union = [pair for w in range(4) for pair in ranges(w, 4)]
        self.assertEqual(sorted(union), sorted(full))

    def test_category_jobs_stride_across_workers(self):
        cats = ['Category:Quality images', 'Category:Valued images']
        all_jobs = [j for w in range(4) for j in category_jobs(cats, w, 4)]
        self.assertEqual(len(all_jobs), 2 * len(shard_bounds()))
        self.assertEqual(len({(j['cat'], j['start'], j['end']) for j in all_jobs}), len(all_jobs))
        self.assertTrue(all(set(j) == {'cat', 'start', 'end', 'continue'} for j in all_jobs))
        self.assertTrue(all(j['continue'] == {} for j in all_jobs))

    def test_category_params_shape(self):
        p = params_for_category('Category:Valued images', 'Ab', 'Ac',
                                {'gcmcontinue': 'x', 'continue': '||'})
        self.assertEqual(p['generator'], 'categorymembers')
        self.assertEqual(p['gcmtitle'], 'Category:Valued images')
        self.assertEqual(p['gcmtype'], 'file')
        self.assertEqual((p['gcmstartsortkey'], p['gcmendsortkey']), ('Ab', 'Ac'))
        self.assertEqual(p['gcmcontinue'], 'x')
        self.assertEqual(p['iiurlwidth'], 384)
        edge = params_for_category('Category:Quality images', '', None, {})
        self.assertNotIn('gcmstartsortkey', edge)
        self.assertNotIn('gcmendsortkey', edge)

    def test_no_archive_flushes_on_pending_points(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        archive = MagicMock()
        archive.pending = []
        # no-archive: un-upserted points are what is pending (the bug was
        # keying off the never-filled archive list, so workers uploaded
        # exactly once, at the very end).
        args = SimpleNamespace(no_embed=False, no_archive=True)
        self.assertEqual(pending_count(args, [], archive, [1] * 499), 499)
        self.assertEqual(flush_every(args), 500)
        # crawl-only: manifest rows, as before.
        args = SimpleNamespace(no_embed=True, no_archive=False)
        self.assertEqual(pending_count(args, [1, 2], archive, [1] * 999), 2)
        self.assertEqual(flush_every(args), 500)
        # coupled+archive: unchanged 4000 behaviour.
        args = SimpleNamespace(no_embed=False, no_archive=False)
        archive.pending = [1] * 3999
        self.assertEqual(pending_count(args, [], archive, [1] * 999), 3999)
        self.assertEqual(flush_every(args), 4000)

    def test_license_boundary_and_stable_id(self):
        page = {'pageid': 42, 'title': 'File:Photo.jpg', 'imageinfo': [{'width': 1000, 'height': 800,
                'url': 'https://example.com/a.jpg?utm_source=test', 'thumburl': 'https://example.com/t.jpg',
                'mime': 'image/jpeg', 'extmetadata': {'LicenseShortName': {'value': 'CC BY 4.0'}}}]}
        row = metadata(page, 'Pi')
        self.assertEqual(row['image_id'], 'commons:42')
        self.assertEqual(row['full_url'], 'https://example.com/a.jpg')
        self.assertIsNone(metadata(page, 'Ph'))
        page['imageinfo'][0]['extmetadata']['LicenseShortName']['value'] = 'CC BY-ND 4.0'
        self.assertIsNone(metadata(page))

    def test_search_text_combines_discovery_fields(self):
        text = search_text({'title': 'Saturn V', 'creator': 'NASA',
                            'description': 'Launch vehicle', 'tags': 'Apollo'})
        self.assertEqual(text, 'Saturn V Launch vehicle Apollo')

    def test_clean_text_strips_openverse_markup(self):
        """Openverse returns microformat HTML in some titles. Left alone it
        renders as literal markup and feeds div/class/style tokens into the
        BM25 index, competing with real subject terms."""
        self.assertEqual(clean_text("<div class='fn'> Water Drops</div>"), 'Water Drops')
        self.assertEqual(clean_text('  a   b  '), 'a b')
        self.assertEqual(clean_text(None), '')

    def test_search_text_excludes_creator(self):
        """Photographer names are not search terms. Indexing them makes a
        query like "williams" match every photo by Phil Williams, and floods
        the sparse index with high-cardinality tokens that dilute IDF."""
        text = search_text({'title': 'Bridge', 'creator': 'Phil Williams'})
        self.assertNotIn('Williams', text)
        self.assertEqual(text, 'Bridge')

    async def test_topic_discovery_uses_host_limiter(self):
        called = []
        def respond(request):
            called.append(request)
            return httpx.Response(200, json={'query': {'pages': []}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as c:
            await ingest.discover(c, 'forest', 1, ingest.HostLimiter())
        self.assertEqual(len(called), 1)

    async def test_enumeration_continuation_and_deduplication(self):
        calls = []
        page={'pageid': 42, 'title': 'File:Photo.jpg', 'imageinfo': [{'thumburl':'https://example.com/t.jpg',
              'mime':'image/jpeg','width':800,'height':800,'extmetadata':{'LicenseShortName':{'value':'CC0'}}}]}
        def respond(request):
            calls.append(dict(request.url.params))
            data={'query':{'pages':[page]}}
            if len(calls)==1:data['continue']={'iicontinue':'Ph|timestamp','continue':'||'}
            return httpx.Response(200,json=data)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as c:
            rows=await ingest.discover_enumerate(c,'Ph',2,ingest.HostLimiter(),aito='Pi')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['id'],'commons:42')
        self.assertEqual(calls[1]['gaifrom'],'Ph')
        self.assertEqual(calls[1]['gaito'],'Pi')

class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls=[]
        async def query_points(*args,**kw):
            self.calls.append(kw)
            return SimpleNamespace(points=[SimpleNamespace(score=.12,payload={'image_id':'commons:1','title':'test',
                    'thumb_origin':'https://example.com/t.jpg','full_url':'https://example.com/original.jpg'})])
        api.S.update(qc=SimpleNamespace(query_points=query_points),lock=asyncio.Lock())
        api.S['results'].clear()
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test')
        self.patch=patch.object(api,'embed',return_value=[1.0]+[0.0]*767)
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.client.aclose()

    async def test_filter_aware_canonical_cache(self):
        first=await self.client.get('/api/search',params={'q':'A Steam Locomotive!','license_class':'public_domain'})
        again=await self.client.get('/api/search',params={'q':'a steam locomotive','license_class':'public_domain'})
        other=await self.client.get('/api/search',params={'q':'a steam locomotive','license_class':'attribution'})
        self.assertEqual(first.status_code,200)
        self.assertTrue(again.json()['cached'])
        self.assertEqual(other.status_code,200)
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.calls[0]['prefetch'][0].using, 'image')
        self.assertEqual(self.calls[0]['prefetch'][1].using, 'bm25')
        self.assertIn('full_url',first.json()['results'][0])
        self.assertEqual(first.json()['results'][0]['thumb'], 'https://example.com/t.jpg')

    async def test_limits_and_errors(self):
        for limit in [0,-1,101]:
            r=await self.client.get('/api/search',params={'q':'forest','limit':limit})
            self.assertEqual(r.status_code,422)
        r=await self.client.get('/api/search',params={'q':'forest','license_class':'invalid'})
        self.assertEqual(r.status_code,400)
        r=await self.client.get('/api/search',params={'q':'!!!'})
        self.assertEqual(r.json()['results'],[])

class RefusalTests(unittest.IsolatedAsyncioTestCase):
    """Adult queries get a hard refusal, never a result set. The embed tower
    is absent here (S['sess'] is None), which also proves the fail-open and
    core-term paths need no model."""
    async def asyncSetUp(self):
        self.calls=[]
        async def query_points(*args,**kw):
            self.calls.append(kw)
            return SimpleNamespace(points=[SimpleNamespace(score=.12,payload={'image_id':'commons:1','title':'test',
                    'thumb_origin':'https://example.com/t.jpg','full_url':'https://example.com/original.jpg'})])
        api.S.update(qc=SimpleNamespace(query_points=query_points),lock=asyncio.Lock())
        api.S['results'].clear()
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test')
        self.patch=patch.object(api,'embed',return_value=[1.0]+[0.0]*767)
        self.patch.start()
        self.old_adult=api._ADULT
        api._ADULT=None

    async def asyncTearDown(self):
        api._ADULT=self.old_adult
        self.patch.stop()
        await self.client.aclose()

    async def test_core_term_refused(self):
        r=await self.client.get('/api/search',params={'q':'porn'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['results'],[])
        self.assertEqual(r.json()['refusal'],api.REFUSAL_MESSAGE)
        self.assertEqual(self.calls,[])

    async def test_whole_word_only(self):
        r=await self.client.get('/api/search',params={'q':'titmouse'})
        self.assertNotIn('refusal',r.json())
        self.assertEqual(len(r.json()['results']),1)
        r=await self.client.get('/api/search',params={'q':'PENIS!'})
        self.assertEqual(r.json()['refusal'],api.REFUSAL_MESSAGE)

    async def test_expanded_core_terms(self):
        for q in ['ejaculation','sperm','cum shot','pajeet','nigger','gook',
                  'lolicon','upskirt','shemale','faggot','redskin','muzzie']:
            r=await self.client.get('/api/search',params={'q':q})
            self.assertEqual(r.json().get('refusal'),api.REFUSAL_MESSAGE,msg=q)

    async def test_naked_phrases_refused_as_core_terms(self):
        for q in ['naked kids','nude kids','naked children','nude boy',
                  'naked girl','naked women','nude men','female nude']:
            r=await self.client.get('/api/search',params={'q':q})
            self.assertEqual(r.json().get('refusal'),api.REFUSAL_MESSAGE,msg=q)

    async def test_backstop_threshold_is_calibrated_value(self):
        self.assertEqual(api.REFUSAL_COSINE, 0.845)

    async def test_legit_phrases_and_nonmatches_pass(self):
        for q in ['sperm whale','spic and span cleaning','niger river',
                  'fire retardant drop','ritz cracker','titmouse',
                  'classical nude sculpture','anatomy diagram']:
            r=await self.client.get('/api/search',params={'q':q})
            self.assertNotIn('refusal',r.json(),msg=q)

    async def test_cosine_backstop(self):
        import numpy as np
        api._ADULT=np.array([[1.0]+[0.0]*767])
        r=await self.client.get('/api/search',params={'q':'forest'})
        self.assertEqual(r.json()['refusal'],api.REFUSAL_MESSAGE)

    async def test_fail_open_without_tower(self):
        r=await self.client.get('/api/search',params={'q':'forest'})
        self.assertNotIn('refusal',r.json())

    async def test_refusal_ignores_include_sensitive(self):
        r=await self.client.get('/api/search',params={'q':'porn','include_sensitive':'true'})
        self.assertEqual(r.json()['refusal'],api.REFUSAL_MESSAGE)

class ManifestHandoff(unittest.TestCase):
    """The crawl writes a manifest; the GPU pass reads it. That file is the
    entire contract between two phases that now run hours apart."""

    def test_manifest_lines_round_trip(self):
        import embed_manifest
        rows = [{'image_id': 'commons:1', 'cdn': 't/aa/bb/commons_1.webp', 'title': 'One'},
                {'image_id': 'commons:2', 'cdn': 't/cc/dd/commons_2.webp', 'title': 'Two'}]
        body = ('\n'.join(json.dumps(r) for r in rows) + '\n').encode()

        class Body:
            @staticmethod
            def read(): return body

        class Client:
            @staticmethod
            def get_object(**kw): return {'Body': Body}

        with patch.object(embed_manifest.storage, 'client', return_value=Client):
            self.assertEqual(list(embed_manifest.rows_from('manifest/b/00-00000.jsonl')), rows)

    def test_a_blank_trailing_line_is_not_a_row(self):
        # Every shard ends with a newline; a naive split would yield an empty
        # row and the embed pass would try to fetch an image with no key.
        import embed_manifest

        class Body:
            @staticmethod
            def read(): return b'{"image_id": "commons:1"}\n\n'

        class Client:
            @staticmethod
            def get_object(**kw): return {'Body': Body}

        with patch.object(embed_manifest.storage, 'client', return_value=Client):
            self.assertEqual(len(list(embed_manifest.rows_from('k'))), 1)


class BucketUploads(unittest.TestCase):
    """The derivative goes to our own bucket, and a failure must not cost the row."""

    @staticmethod
    def chunk(n=3):
        return [({'image_id': f'commons:{i}'}, None, b'webp-bytes') for i in range(n)]

    def test_no_bucket_configured_yields_no_urls(self):
        with patch.object(cloud_corpus.storage, 'enabled', return_value=False):
            self.assertEqual(cloud_corpus.upload_batch(self.chunk()), ['', '', ''])

    def test_urls_come_back_in_the_order_they_went_in(self):
        # zip() pairs these against rows positionally; reordering would attach
        # each image's URL to a different image.
        with patch.object(cloud_corpus.storage, 'enabled', return_value=True), \
             patch.object(cloud_corpus.storage, 'put',
                          side_effect=lambda i, d: f'https://cdn/{i}.webp'):
            self.assertEqual(cloud_corpus.upload_batch(self.chunk()),
                             ['https://cdn/commons:0.webp',
                              'https://cdn/commons:1.webp',
                              'https://cdn/commons:2.webp'])

    def test_one_failed_upload_does_not_lose_the_others(self):
        def flaky(image_id, data):
            if image_id == 'commons:1':
                raise OSError('bucket said no')
            return f'https://cdn/{image_id}.webp'
        with patch.object(cloud_corpus.storage, 'enabled', return_value=True), \
             patch.object(cloud_corpus.storage, 'put', side_effect=flaky):
            urls = cloud_corpus.upload_batch(self.chunk())
        # The middle row keeps its place and simply has no CDN url; the API
        # falls back to the proxy for it rather than the row being dropped.
        self.assertEqual(urls, ['https://cdn/commons:0.webp', '',
                                'https://cdn/commons:2.webp'])


class HostLimiting(unittest.TestCase):
    """The fix ingest.py has had all along, finally in the cloud crawler.

    Measured on 2,500 real origins: thumb.wikimedia.org 0 failures in 1,669,
    upload.wikimedia.org 152 in 392, and 16 of 30 answering 429 at concurrency
    six. One global semaphore cannot express that difference.
    """

    def test_each_host_gets_its_own_budget(self):
        limiter = cloud_corpus.HostLimiter(per_host=6)
        a = limiter.get("https://thumb.wikimedia.org/x.jpg")
        b = limiter.get("https://images.metmuseum.org/y.jpg")
        self.assertIsNot(a, b)
        # Same host, same semaphore -- otherwise the limit means nothing.
        self.assertIs(a, limiter.get("https://thumb.wikimedia.org/z.jpg"))

    def test_the_origin_host_is_held_to_a_tighter_budget(self):
        limiter = cloud_corpus.HostLimiter(per_host=6)
        self.assertEqual(limiter.limit_for("https://upload.wikimedia.org/a.png"), 2)
        self.assertEqual(limiter.limit_for("https://thumb.wikimedia.org/b.jpg"), 6)
        self.assertLess(limiter.limit_for("https://upload.wikimedia.org/a.png"),
                        limiter.limit_for("https://live.staticflickr.com/c.jpg"))

    def test_a_429_does_not_hold_its_slot(self):
        """Backing off inside the semaphore blocks every image queued behind it."""
        calls, released = [], []

        class Recorder:
            def __init__(self, sem): self.sem = sem
            async def __aenter__(self): calls.append("acquire"); return self
            async def __aexit__(self, *a): released.append("release"); return False

        limiter = cloud_corpus.HostLimiter()
        limiter._sems["example.org"] = None
        original = limiter.get
        limiter.get = lambda url: Recorder(None)

        class Response:
            status_code = 429
            headers = {"Retry-After": "0"}
            content = b""
            def raise_for_status(self): pass

        class Client:
            def __init__(self): self.n = 0
            async def get(self, url, **kw):
                self.n += 1
                if self.n == 1:
                    return Response()
                ok = Response(); ok.status_code = 200
                ok.content = b"not an image"
                return ok

        async def run():
            return await cloud_corpus.fetch_image(
                Client(), {"thumb_origin": "https://example.org/a.jpg", "image_id": "x"}, limiter)

        asyncio.run(run())
        # Two attempts, and the slot was given back between them.
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(released), 2)


class ImageFetchResilience(unittest.IsolatedAsyncioTestCase):
    """A worker crawls for hours; one bad file must not end that.

    Worker 19 of the 500k build died at 2h25m on a single PNG whose text chunk
    tripped Pillow's decompression guard. That guard raises a plain ValueError,
    which the handler did not list, so it escaped and killed the shard.
    """

    @staticmethod
    def client(content=b'not-an-image', status=200):
        response = SimpleNamespace(content=content, status_code=status,
                                   headers={}, raise_for_status=lambda: None)
        return SimpleNamespace(get=lambda url, **kw: _resolved(response))

    async def fetch(self, error):
        row = {'image_id': 'commons:1', 'thumb_origin': 'https://example.org/a.png'}
        with patch.object(cloud_corpus.Image, 'open', side_effect=error):
            return await cloud_corpus.fetch_image(
                self.client(), row, cloud_corpus.HostLimiter())

    async def test_codec_failures_skip_the_image(self):
        import struct, zlib
        for error in (ValueError('Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK'),
                      zlib.error('invalid distance too far back'),
                      struct.error('unpack requires a buffer of 4 bytes'),
                      EOFError('truncated'),
                      OSError('cannot identify image file')):
            with self.subTest(error=type(error).__name__):
                self.assertIsNone(await self.fetch(error))

    async def test_undecodable_bytes_skip_the_image(self):
        row = {'image_id': 'commons:2', 'thumb_origin': 'https://example.org/a.png'}
        client = self.client(content=b'\x00\x01\x02 not an image')
        self.assertIsNone(await cloud_corpus.fetch_image(client, row, cloud_corpus.HostLimiter()))

    async def test_a_coding_mistake_is_never_mistaken_for_a_bad_image(self):
        # The broad catch above must not resurrect the bug where a NameError
        # from a missing import made discovery silently return nothing.
        for error in (NameError("name 'urlparse' is not defined"),
                      AttributeError('module has no attribute'),
                      ImportError('no module named PIL')):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    await self.fetch(error)


if __name__=='__main__':unittest.main()


class SearchHarvestTest(unittest.TestCase):
    """generator=search harvesting, sharded by licence and width band."""

    def test_jobs_partition_without_overlap_or_loss(self):
        topics = ['sunset', 'coffee', 'temple']
        jobs = [cloud_corpus.search_jobs(topics, cloud_corpus.CLEAN_LICENCES, w, 4)
                for w in range(4)]
        flat = [(j['topic'], j['lic'], j['band']) for js in jobs for j in js]
        expected = len(topics) * len(cloud_corpus.CLEAN_LICENCES) * len(cloud_corpus.WIDTH_BANDS)
        self.assertEqual(len(flat), expected)
        self.assertEqual(len(set(flat)), expected, 'a shard was issued twice')

    def test_every_worker_spans_every_topic(self):
        # Strided assignment, so one worker is never camped on one subject and
        # a finished topic does not leave a lane idle.
        topics = ['sunset', 'coffee', 'temple']
        for w in range(4):
            covered = {j['topic'] for j in cloud_corpus.search_jobs(
                topics, cloud_corpus.CLEAN_LICENCES, w, 4)}
            self.assertEqual(covered, set(topics))

    def test_width_bands_do_not_share_a_boundary(self):
        # `filew:>N` and `filew:<N` both INCLUDE N, so adjacent bands would
        # double-count the boundary. Measured: sunset+CC-Zero is 21,179, and
        # >3000 / <3000 return 17,186 / 4,193 -- 200 too many, which is exactly
        # `filew:3000`. The bands must be >N and <N-1.
        highs = [int(b.split('>')[1]) for b in cloud_corpus.WIDTH_BANDS if '>' in b]
        lows = [int(b.split('<')[1]) for b in cloud_corpus.WIDTH_BANDS if '<' in b]
        for hi in highs:
            for lo in lows:
                self.assertLess(lo, hi, 'width bands overlap at their boundary')

    def test_query_shape(self):
        params = cloud_corpus.params_for_search('sunset', 'CC-Zero', 'filew:>3000',
                                                {'gsroffset': 50})
        self.assertEqual(params['generator'], 'search')
        self.assertEqual(params['gsroffset'], 50)
        self.assertIn('incategory:"CC-Zero"', params['gsrsearch'])
        self.assertIn('filetype:bitmap', params['gsrsearch'])
        # 384 is what is stored, and asking for more pushes the fetch onto the
        # strict host. Shared with both other walkers.
        self.assertEqual(params['iiurlwidth'], 384)


    def test_bin_packing_beats_striding_on_makespan(self):
        # Striding ignores shard size. On the Quality/Featured/Valued run that
        # left 9 of 40 lanes with zero rows while the busiest took 24,906
        # against a 10,387 average. Shard size is known here, so it should not
        # happen again.
        topics = [f't{i}' for i in range(200)]
        # Deliberately lopsided: a few huge topics and a long thin tail, which
        # is the real shape (median 26,751 hits, max in the millions).
        hits = {t: (500_000 if i < 8 else 800) for i, t in enumerate(topics)}
        lic = cloud_corpus.CLEAN_LICENCES

        def spread(weights):
            loads = []
            for w in range(40):
                jobs = cloud_corpus.search_jobs(topics, lic, w, 40, weights)
                loads.append(sum(cloud_corpus.job_weight(j['topic'], lic, hits)
                                 for j in jobs))
            return max(loads) / max(min(loads), 1e-9)

        # Not "perfectly flat": shards are indivisible and capped at 10,000, so
        # with 128 full-size jobs over 40 lanes somebody must take four of them.
        # The guarantee LPT actually offers is 4/3 of optimal makespan.
        loads = []
        for w in range(40):
            jobs = cloud_corpus.search_jobs(topics, lic, w, 40, hits)
            loads.append(sum(cloud_corpus.job_weight(j['topic'], lic, hits)
                             for j in jobs))
        ideal = sum(loads) / len(loads)
        self.assertLess(max(loads) / ideal, 4 / 3, 'worse than the LPT bound')
        self.assertLess(spread(hits), spread(None), 'bin packing lost to striding')

    def test_every_worker_gets_work_when_sizes_are_lopsided(self):
        # The failure that actually cost wall clock was idle lanes, not uneven
        # ones. No lane may come back empty while jobs remain.
        topics = [f't{i}' for i in range(60)]
        hits = {t: (900_000 if i == 0 else 100) for i, t in enumerate(topics)}
        for w in range(40):
            self.assertTrue(cloud_corpus.search_jobs(
                topics, cloud_corpus.CLEAN_LICENCES, w, 40, hits),
                f'worker {w} drew no jobs')

    def test_packing_is_identical_on_every_worker(self):
        # No coordination: each worker derives the same assignment alone. If
        # two workers disagreed they would double-crawl and leave holes.
        topics = ['a', 'b', 'c', 'd']
        hits = {'a': 90_000, 'b': 40_000, 'c': 9_000, 'd': 900}
        lic = cloud_corpus.CLEAN_LICENCES
        seen = [tuple((j['topic'], j['lic'], j['band'])
                      for j in cloud_corpus.search_jobs(topics, lic, w, 6, hits))
                for w in range(6)]
        flat = [j for lane in seen for j in lane]
        self.assertEqual(len(flat), len(set(flat)), 'a shard landed on two workers')
        again = [tuple((j['topic'], j['lic'], j['band'])
                       for j in cloud_corpus.search_jobs(topics, lic, w, 6, hits))
                 for w in range(6)]
        self.assertEqual(seen, again, 'assignment is not deterministic')

    def test_licence_default_excludes_share_alike(self):
        # The app hides share-alike by default, so a harvest that includes it
        # spends crawl hours on images most users never see.
        self.assertTrue(cloud_corpus.CLEAN_LICENCES)
        for lic in cloud_corpus.CLEAN_LICENCES:
            self.assertNotIn('SA', lic.upper().replace('-', ''))
