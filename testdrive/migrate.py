"""Copy the live collection into a Qdrant you control.

Run this ON the rented box, not on the Mac: it is a server-to-server copy and
the 435k vectors are ~1.4 GB, which is minutes over a datacentre link and an
evening over home broadband.

Resumable. It records the last scroll offset, so an interrupted run continues
rather than starting again.

    QDRANT_URL=... QDRANT_API_KEY=... \\
    python migrate.py [target_url] [quantization]

quantization: int8 (default) or binary -- the choice this whole step exists to
decide, since it is the difference between a box at ~EUR19/mo and one at half
that, every month, forever.
"""
import json, os, sys, time
from pathlib import Path
from qdrant_client import QdrantClient, models

SOURCE_COLLECTION = os.getenv('QDRANT_COLLECTION', 'images-v2')
TARGET_URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:6333'
QUANT = (sys.argv[2] if len(sys.argv) > 2 else 'int8').lower()
TARGET_COLLECTION = f'images-{QUANT}'
BATCH = 256
STATE = Path(f'/tmp/migrate-{QUANT}.json')

src = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=120)
dst = QdrantClient(url=TARGET_URL, timeout=120)

# Qdrant's own footprint, before any data. Only the vectors and the graph grow
# with the corpus; this does not. Scaling total RSS to 10M without subtracting
# it first overstates the requirement badly -- by 23x this constant.
import subprocess
BASELINE = Path('/tmp/qdrant-baseline.json')
if not BASELINE.exists():
    rss = subprocess.run(['bash', '-c',
        "for p in $(pgrep -f qdrant); do grep -h '^RssAnon' /proc/$p/status "
        "2>/dev/null; done | awk '{s+=$2} END {print s*1024}'"],
        capture_output=True, text=True).stdout.strip()
    BASELINE.write_text(json.dumps({'rss_bytes': int(rss or 0)}))
    print(f"recorded empty-Qdrant baseline: {int(rss or 0)/2**30:.2f} GB")

# int8-disk keeps int8's accuracy but memory-maps the quantized vectors instead
# of pinning them. RAM then holds little more than the HNSW graph, and the cost
# is a disk read per candidate -- paid out of a latency budget that measured
# 19 ms against a 700 ms gate.
QUANTIZATIONS = {
    'int8': models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
    'int8-disk': models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8, quantile=0.99, always_ram=False)),
    'binary': models.BinaryQuantization(binary=models.BinaryQuantizationConfig(
        always_ram=True)),
    'pq16': models.ProductQuantization(product=models.ProductQuantizationConfig(
        compression=models.CompressionRatio.X16, always_ram=True)),
    'pq8': models.ProductQuantization(product=models.ProductQuantizationConfig(
        compression=models.CompressionRatio.X8, always_ram=True)),
}
if QUANT not in QUANTIZATIONS:
    raise SystemExit(f'quantization must be one of {", ".join(QUANTIZATIONS)}')
quantization = QUANTIZATIONS[QUANT]

if not dst.collection_exists(TARGET_COLLECTION):
    dst.create_collection(
        TARGET_COLLECTION,
        # Identical to hybrid_collection.py, verbatim. Every measurement taken
        # against a drifted config describes a system nobody is going to run.
        vectors_config={'image': models.VectorParams(
            size=768, distance=models.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={'bm25': models.SparseVectorParams(
            modifier=models.Modifier.IDF,
            index=models.SparseIndexParams(on_disk=True))},
        quantization_config=quantization,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        on_disk_payload=True,
    )
    for field, schema in (('license_class', models.PayloadSchemaType.KEYWORD),
                          ('build_id', models.PayloadSchemaType.KEYWORD),
                          ('worker', models.PayloadSchemaType.INTEGER)):
        dst.create_payload_index(TARGET_COLLECTION, field, field_schema=schema, wait=True)
    print(f'created {TARGET_COLLECTION} with {QUANT} quantization')

offset = json.loads(STATE.read_text())['offset'] if STATE.exists() else None
moved = dst.count(TARGET_COLLECTION, exact=True).count
total = src.count(SOURCE_COLLECTION, exact=True).count
print(f'{moved:,} of {total:,} already present\n')

start = time.monotonic()
session_start = moved
while True:
    points, offset = src.scroll(SOURCE_COLLECTION, limit=BATCH, offset=offset,
                                with_payload=True, with_vectors=True)
    if not points:
        break
    dst.upsert(TARGET_COLLECTION, wait=False, points=[
        models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
    moved += len(points)
    elapsed = time.monotonic() - start
    rate = (moved - session_start) / max(elapsed, 1)
    print(f'\r  {moved:,}/{total:,}  {rate:6.0f} pts/s  '
          f'eta {(total-moved)/max(rate,1)/60:5.1f} min', end='', flush=True)
    STATE.write_text(json.dumps({'offset': str(offset) if offset else None}))
    if offset is None:
        break

# Disk measured straight after a bulk load counts segments the optimizer has
# not merged yet: binary read 4.11 GB that way and 1.96 GB once it settled.
print('\n\nwaiting for the optimizer to settle before anything measures disk')
for _ in range(120):
    info = dst.get_collection(TARGET_COLLECTION)
    if info.status == models.CollectionStatus.GREEN and not info.optimizer_status.ok is False:
        break
    time.sleep(5)
print(f'collection status: {dst.get_collection(TARGET_COLLECTION).status}')
print(f'\nmoved {moved:,} points in {(time.monotonic()-start)/60:.1f} min')
print(f'upsert rate: {(moved-session_start)/max(time.monotonic()-start,1):.0f} pts/s '
      f'(free tier measured 82.6/s; gate is >200)')
