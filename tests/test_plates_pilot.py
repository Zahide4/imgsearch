import importlib.util
import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plates_pilot", ROOT / "testdrive/plates_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def image(**kw):
    return dict(id="abc", title="Concrete", source="flickr", license="by", width=1024,
                height=768, url="https://live.staticflickr.com/1/123_abcdef_b.jpg",
                foreign_landing_url="https://www.flickr.com/photos/test/123", **kw)


def page(rows):
    return dict(results=rows, page_count=12, page_size=20, page=1)


class PlatesPilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.db = pilot.connect(self.out / "pilot.sqlite3", ["concrete texture"])

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_filters_do_not_trust_query_parameters(self):
        for key, value, reason in [("license", "by-sa", "license"), ("license", "by-nd", "license"),
                                   ("source", "wikimedia", "source"), ("width", None, "dimensions_unknown"),
                                   ("height", 40, "small"), ("width", 4000, "aspect"),
                                   ("mature", True, "sensitive_metadata"), ("url", "javascript:alert(1)", "url_or_id")]:
            x = image(); x[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(pilot.normalize(x, "texture", "flickr", 0)[1], reason)

    def test_flickr_derivative_and_asset_identity(self):
        row, reason = pilot.normalize(image(), "texture", "flickr", 0)
        self.assertIsNone(reason)
        self.assertEqual(row["asset_key"], "flickr:123")
        self.assertEqual(row["thumb_origin"], "https://live.staticflickr.com/1/123_abcdef_n.jpg")
        self.assertNotIn("full_url", row)
        self.assertEqual(row["provider_url"], image()["url"])

    def test_commit_deduplicates_across_openverse_ids_and_rolls_back_cursor(self):
        second = image(); second.update(id="other", url="https://live.staticflickr.com/2/123_abcdef_z.jpg")
        self.assertEqual(pilot.commit_page(self.db, ("concrete texture", "flickr", 1), page([image(), second])), 1)
        self.assertEqual(self.db.execute("SELECT duplicates FROM pages").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError):
            pilot.commit_page(self.db, ("concrete texture", "flickr", 1), page([image()]))
        self.assertEqual(self.db.execute("SELECT page FROM jobs WHERE source='flickr'").fetchone()[0], 2)

    def test_malformed_response_preserves_page(self):
        for response in [{}, {**page([]), "page": 2}, {**page([]), "page_size": 100}]:
            with self.assertRaises(ValueError):
                pilot.commit_page(self.db, ("concrete texture", "flickr", 1), response)
        self.assertEqual(self.db.execute("SELECT SUM(page) FROM jobs").fetchone()[0], 2)

    def test_429_preserves_cursor_and_records_retry_after(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429, headers={"retry-after": "3600"})))
        args = SimpleNamespace(target=10000, max_seconds=10, max_requests=2)
        with patch.object(pilot.httpx, "Client", return_value=client), patch.object(pilot.time, "sleep"):
            status, code = pilot.crawl(self.db, args, pilot.Auth(None))
        self.assertEqual((status, code), ("rate_limited", 75))
        self.assertEqual(self.db.execute("SELECT SUM(done) FROM jobs").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0], 0)
        with patch.object(pilot.httpx, "Client", side_effect=AssertionError("Must not retry before deadline")):
            self.assertEqual(pilot.crawl(self.db, args, pilot.Auth(None)), (status, code))

    def test_resume_uses_page_cursor_and_round_robin_sources(self):
        pilot.commit_page(self.db, ("concrete texture", "flickr", 1), page([image()]))
        calls = []
        def respond(request):
            calls.append(dict(request.url.params))
            return httpx.Response(200, json=page([]))
        client = httpx.Client(transport=httpx.MockTransport(respond))
        args = SimpleNamespace(target=10000, max_seconds=10, max_requests=1)
        with patch.object(pilot.httpx, "Client", return_value=client), patch.object(pilot.time, "sleep"):
            self.assertEqual(pilot.crawl(self.db, args, pilot.Auth(None)), ("budget_paused", 75))
        self.assertEqual(calls[0]["source"], "rawpixel")
        self.assertEqual(calls[0]["page"], "1")

    def test_configuration_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            pilot.connect(self.out / "pilot.sqlite3", ["unrelated topic"])

    def test_authentication_refreshes_without_persisting_tokens(self):
        path = self.out / "private.json"
        path.write_text(json.dumps({"client_id": "example-id", "client_secret": "example-secret"}))
        auth = pilot.Auth(path)
        calls = []
        def respond(request):
            calls.append(request)
            self.assertEqual(request.url.path, "/v1/auth_tokens/token/")
            self.assertIn(b"grant_type=client_credentials", request.content)
            return httpx.Response(200, json={"access_token": f"token-{len(calls)}", "expires_in": 3600})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.assertEqual(auth.headers(client), {"Authorization": "Bearer token-1"})
            self.assertEqual(auth.headers(client), {"Authorization": "Bearer token-1"})
            self.assertEqual(auth.headers(client, refresh=True), {"Authorization": "Bearer token-2"})
        self.assertEqual(len(calls), 2)
        self.assertNotIn("access_token", path.read_text())

    def test_request_interval_obeys_advertised_limits(self):
        self.assertAlmostEqual(pilot.request_interval({}), 3.3)
        self.assertAlmostEqual(pilot.request_interval({'x-ratelimit-limit-anon_burst':'20/min'}), 3.3)
        self.assertAlmostEqual(pilot.request_interval({'x-ratelimit-limit-oauth2_client_credentials_burst':'100/min'}), 2/3)
        self.assertAlmostEqual(pilot.request_interval({'x-ratelimit-limit-anon_burst':'10/min'}), 6.6)

    def test_export_cannot_inject_script_and_does_not_claim_review(self):
        x = image(); x["title"] = "</script><script>alert(1)</script>"
        pilot.commit_page(self.db, ("concrete texture", "flickr", 1), page([x]))
        args = SimpleNamespace(out=self.out, target=10000)
        pilot.export(self.db, args, "budget_paused")
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["decision"], "pending_human_review")
        self.assertNotIn("</script><script>alert", (self.out / "review.html").read_text())
        self.assertEqual(summary["sources"]["flickr"]["candidates"], 1)


if __name__ == "__main__":
    unittest.main()
