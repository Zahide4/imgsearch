"""How fast does SigLIP embed on a GitHub Actions runner's CPU?

This decides an architecture. Today the crawl runs on 20 free runners and
parks 229 GB of derivatives for a GPU to read later, which is the only reason
the project needs object storage at all. If the runners can embed as they
crawl, nothing is ever parked: no bucket, no egress, no card.

`cloud-corpus.yml` records ~1.75 img/s per worker for the coupled path, but
that predates the HostLimiter and iiurlwidth=384 fixes that took crawling from
39% failures to 5.4 img/s. So the coupled number is stale, and it is stale in
the direction that matters.

Measured here, on the real target hardware -- an M4 would flatter this by 3-5x
and answer the wrong question:

    decode+preprocess   PIL WebP -> normalized tensor, per image
    forward             SigLIP encode_image, batched
    combined            what a coupled worker would actually sustain

Crawl-only is 5.4 img/s per runner. If fetch and embed overlap perfectly a
coupled worker runs at min(5.4, embed); if they serialise it is the harmonic
sum. The truth is between, and both bounds are printed.
"""
import argparse, io, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CRAWL_RATE = 5.4          # measured, per runner, crawl-only


def sample_images(n, px=384):
    """Photograph-like WebP, so decode cost is representative.

    Flat colour compresses to almost nothing and would make decoding look
    free; smooth gradients plus noise land within a few KB of the 22.4 KB
    mean measured on the live bucket.
    """
    rng = np.random.default_rng(0)
    out = []
    base = np.linspace(0, 255, px, dtype=np.float32)
    for i in range(n):
        a = np.stack([(base[None, :] + base[:, None] * (1 + i % 3)) % 256] * 3, -1)
        a = np.clip(a + rng.normal(0, 18, a.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, 'WEBP', quality=80)
        out.append(buf.getvalue())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=int, default=192)
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()

    import open_clip
    print(f'torch {torch.__version__}  cpus={os.cpu_count()}', flush=True)

    blobs = sample_images(a.images)
    kb = sum(len(b) for b in blobs) / len(blobs) / 1024
    print(f'{len(blobs)} synthetic derivatives, mean {kb:.1f} KB '
          f'(live bucket measures 22.4 KB)\n', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16-SigLIP', pretrained='webli')
    model = model.eval()

    results = {}
    for threads in sorted({1, 2, 4, os.cpu_count() or 4}):
        torch.set_num_threads(threads)

        t0 = time.monotonic()
        tensors = [preprocess(Image.open(io.BytesIO(b)).convert('RGB')) for b in blobs]
        t_decode = time.monotonic() - t0

        t0 = time.monotonic()
        with torch.inference_mode():
            for i in range(0, len(tensors), a.batch):
                v = F.normalize(model.encode_image(torch.stack(tensors[i:i + a.batch])), dim=-1)
        t_forward = time.monotonic() - t0

        combined = len(blobs) / (t_decode + t_forward)
        results[threads] = round(combined, 2)
        print(f'threads={threads:2}  decode {len(blobs)/t_decode:6.1f} img/s   '
              f'forward {len(blobs)/t_forward:6.1f} img/s   '
              f'embed total {combined:6.2f} img/s', flush=True)

    best = max(results.values())
    overlapped = min(CRAWL_RATE, best)
    serial = 1 / (1 / CRAWL_RATE + 1 / best)
    print(f'\nbest embed rate      {best:.2f} img/s per runner')
    print(f'coupled, overlapped  {overlapped:.2f} img/s  (fetch hidden behind embed)')
    print(f'coupled, serialised  {serial:.2f} img/s  (no overlap at all)')
    for label, rate in (('overlapped', overlapped), ('serialised', serial)):
        print(f'  10M on 20 runners, {label:11}: '
              f'{10_000_000 / (rate * 20) / 3600:5.1f} h '
              f'({10_000_000 / (rate * 20) / 86400:.1f} days)')
    print(f'\nfor comparison, crawl-only + separate GPU pass needs 229 GB parked '
          f'somewhere, which is the only reason this project needs a bucket.')
    Path('testdrive/cpu-embed-bench.json').write_text(json.dumps(
        {'per_thread': results, 'crawl_rate': CRAWL_RATE,
         'coupled_overlapped': round(overlapped, 2),
         'coupled_serialised': round(serial, 2)}, indent=2))


if __name__ == '__main__':
    main()
