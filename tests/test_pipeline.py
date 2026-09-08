import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import ingest
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

if __name__=='__main__':unittest.main()
