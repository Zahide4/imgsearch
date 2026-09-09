"""Collect distinct source-image URLs from the live API.

No credentials: the API's `thumb` field is a proxy URL with the real origin
url-encoded inside it, so the whole sample can be drawn from the public
endpoint. Topics come from seeds.txt so the sample spans the corpus rather
than one subject -- file size varies far more by subject (museum scans vs
snapshots) than by anything else, and a narrow sample would give a confident
wrong answer about the 22 KB assumption.
"""
import json, random, sys, time, urllib.parse, urllib.request

API = "https://imgsearch-api.onrender.com/api/search"
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 2500

topics = [l.strip() for l in open("seeds.txt")
          if l.strip() and not l.startswith("#")]
random.Random(20260909).shuffle(topics)

seen, origins, failures = set(), [], 0
for i, topic in enumerate(topics):
    if len(origins) >= WANT:
        break
    url = f"{API}?{urllib.parse.urlencode({'q': topic, 'limit': 48})}"
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            hits = json.load(r)["results"]
    except Exception as exc:
        failures += 1
        print(f"  ! {topic}: {type(exc).__name__}", flush=True)
        continue
    for hit in hits:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(hit["thumb"]).query)
        origin = q.get("url", [None])[0]
        if origin and origin not in seen:
            seen.add(origin)
            origins.append(origin)
    print(f"[{i+1}/{len(topics)}] {topic[:38]:38} total={len(origins)}", flush=True)
    time.sleep(0.3)                      # the API is one small free instance

open("testdrive/origins.txt", "w").write("\n".join(origins[:WANT]))
print(f"\n{len(origins[:WANT])} distinct origins, {failures} query failures")
