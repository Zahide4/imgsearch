import asyncio
import importlib.util
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
        self.assertTrue(first.json()['results'][0]['thumb'].startswith('https://wsrv.nl/'))

    async def test_limits_and_errors(self):
        for limit in [0,-1,101]:
            r=await self.client.get('/api/search',params={'q':'forest','limit':limit})
            self.assertEqual(r.status_code,422)
        r=await self.client.get('/api/search',params={'q':'forest','license_class':'invalid'})
        self.assertEqual(r.status_code,400)
        r=await self.client.get('/api/search',params={'q':'!!!'})
        self.assertEqual(r.json()['results'],[])

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
