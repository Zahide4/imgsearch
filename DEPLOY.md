# Cloud deployment and corpus build

The public application runs `server/` on Render. Query text is embedded there;
Qdrant Cloud stores vectors and metadata. Browsers load images through wsrv.nl.
`server.py` and `static_site/` are older local/browser experiments, not the Render app.

## No image corpus on your Mac

Use the **Build searchable cloud corpus** GitHub Actions workflow. Each worker:

1. Walks disjoint Commons filename ranges, retaining both generator and image-info continuation.
2. Filters license metadata, image type, dimensions, and extreme aspect ratios.
3. Fetches API-provided thumbnails, makes 384px WebP derivatives, and embeds them
   with the same `ViT-B-16-SigLIP / webli` image model as the existing corpus.
4. Archives derivatives and license/source metadata as WebDataset tar files in
   Hugging Face, then writes dense image vectors and BM25 title/creator/description/
   tag vectors into the `images-v2` Qdrant collection.
5. Saves restart cursors in the dataset. No corpus is downloaded to your Mac or Render.

GitHub repository secrets: `HF_TOKEN`, `HF_REPO`, `QDRANT_URL`, `QDRANT_API_KEY`.
These must never appear in committed files or workflow inputs.

`hybrid_collection.py migrate` copies the older collection into `images-v2`
directly inside Qdrant's APIs. It is idempotent and transfers no image files.
Render and the workflow both select `images-v2` through `QDRANT_COLLECTION`.

Start with 100 images per worker (400 total):

```sh
gh workflow run cloud-corpus.yml -f target_per_worker=100 -f build=pilot-v1
```

Inspect the completed pilot before scaling. There are four workers. The target
is the number of **new images per worker for that build**, not the total collection
size. Reusing the same build identifier resumes its counters and cursors; retain
the same worker count. Existing Qdrant IDs are skipped. A time-limited run saves
its cursor and reports incomplete rather than claiming the target was reached.

After a pilot, use `(500000 - current_collection_count) / 4`, rounded down, for
a new build's target. The last few images can be added in a final small build.
Do not run unrelated ingestion jobs concurrently into the same collection.

## Capacity and verification

The Qdrant free plan advertises 1 GB RAM and 4 GB disk. Capacity depends on payloads,
vector originals, quantization, HNSW, and indexing headroom; 500k is a target, not a
measured guarantee. Keep scalar int8 quantization with full-precision rescoring.
Measure `/metrics`, `/telemetry`, collection health, and query latency as it grows.
HF public dataset storage is best-effort; never assume unlimited archive space.

The API fuses SigLIP image similarity with Qdrant BM25 using reciprocal-rank
fusion. This keeps visual/conceptual searches while making names, identifiers,
and technical terms discoverable. The API keeps the trained 64-token padding,
pad ID 1, and SigLIP canonicalization.
Do not shorten padding or substitute the failed four-bit text model for speed.
The 512-entry embedding cache and 128-entry, 120-second result cache are bounded.
The frontend cancels superseded searches and includes license filters in cache keys.

Before publishing code:

```sh
.venv/bin/python -B -m unittest discover -s tests -v
node tests/search-ui.cjs
git diff --check
```

Render deploys from the GitHub default branch via `render.yaml`. The model is
baked into the Docker image. Set `ORT_THREADS=1` initially and measure on the
actual host. New query latency is affected by its shared CPU; cached timings
are not an end-to-end latency guarantee. Health: `/healthz`; corpus count:
`/api/stats`. Image-load time must be measured separately from the API response.

Source documentation:
- https://qdrant.tech/documentation/cloud/create-cluster/
- https://qdrant.tech/documentation/inference/inference-bm25/
- https://qdrant.tech/documentation/search/hybrid-queries/
- https://www.mediawiki.org/wiki/API:Allimages
- https://www.mediawiki.org/wiki/API:Etiquette
- https://huggingface.co/docs/hub/storage-limits
