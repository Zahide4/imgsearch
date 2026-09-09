"""Step A: does a 384px WebP derivative really average ~22 KB?

The whole $1.32/month thumbnail line rests on that number. Measures the size
distribution and the fetch failure rate, and writes the derivatives so the
next step can upload them.
"""
import asyncio, json, pathlib, statistics, time
import httpx, pyvips

OUT = pathlib.Path("testdrive/thumbs"); OUT.mkdir(parents=True, exist_ok=True)
URLS = open("testdrive/origins.txt").read().split()
UA = "ImgSearch/0.2 (https://github.com/Zahide4/imgsearch)"

sizes, latency, failures = [], [], {}

async def fetch_one(client, url, index, sem):
    async with sem:
        started = time.monotonic()
        try:
            r = await client.get(url, timeout=30)
            r.raise_for_status()
            # shrink-on-load: a 4000px JPEG decodes at 1/8 scale for ~750 KB
            # rather than ~48 MB. Pillow has no equivalent and OOMs in bulk.
            img = pyvips.Image.thumbnail_buffer(r.content, 384, height=384, size="down")
            data = img.write_to_buffer(".webp[Q=80]")
            (OUT / f"{index}.webp").write_bytes(data)
            sizes.append(len(data))
            latency.append(time.monotonic() - started)
        except Exception as exc:
            name = type(exc).__name__
            failures[name] = failures.get(name, 0) + 1

async def main():
    # Six in flight per host, as the crawler does. A single global limit
    # concentrates every request on one origin and took failures to 60%.
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True) as c:
        await asyncio.gather(*(fetch_one(c, u, i, sem) for i, u in enumerate(URLS)))

start = time.monotonic()
asyncio.run(main())
elapsed = time.monotonic() - start

if not sizes:
    raise SystemExit(f"nothing succeeded: {failures}")

s = sorted(sizes); lat = sorted(latency)
pick = lambda xs, q: xs[min(len(xs) - 1, int(len(xs) * q))]
report = {
    "n": len(s), "attempted": len(URLS),
    "failure_rate_pct": round(100 * (len(URLS) - len(s)) / len(URLS), 2),
    "failures": failures,
    "mean_kb": round(statistics.mean(s) / 1024, 1),
    "p50_kb": round(pick(s, .50) / 1024, 1),
    "p90_kb": round(pick(s, .90) / 1024, 1),
    "p99_kb": round(pick(s, .99) / 1024, 1),
    "max_kb": round(s[-1] / 1024, 1),
    "fetch_p50_s": round(pick(lat, .50), 2),
    "fetch_p90_s": round(pick(lat, .90), 2),
    "wall_clock_s": round(elapsed, 1),
    "throughput_img_s": round(len(s) / elapsed, 1),
}
report["projected_10M_GB"] = round(report["mean_kb"] * 10_000_000 / 1024 / 1024, 0)
report["projected_10M_usd_month"] = round(report["projected_10M_GB"] * 0.006, 2)

open("testdrive/step-a-result.json", "w").write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print()
verdict = "GO" if 18 <= report["mean_kb"] <= 28 else "NO-GO"
print(f"{verdict}: mean {report['mean_kb']} KB "
      f"(gate 18-28 KB) -> ${report['projected_10M_usd_month']}/mo at 10M, "
      f"plan assumes $1.32")
