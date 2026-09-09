#!/usr/bin/env bash
# Step D, start to finish, on a fresh Hetzner box.
#
#   export QDRANT_URL='https://....aws.cloud.qdrant.io'
#   export QDRANT_API_KEY='...'
#   bash testdrive/step_d.sh
#
# Runs both quantization passes and prints the two result files at the end.
# Destroy the server afterwards -- that is the only way this gets expensive.
set -euo pipefail

: "${QDRANT_URL:?set QDRANT_URL first}"
: "${QDRANT_API_KEY:?set QDRANT_API_KEY first}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "installing docker and python"
apt-get update -qq
apt-get install -y -qq docker.io python3-pip >/dev/null

say "starting qdrant"
mkdir -p /var/lib/qdrant/storage
if ! docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
    docker rm -f qdrant >/dev/null 2>&1 || true
    docker run -d --name qdrant -p 6333:6333 \
        -v /var/lib/qdrant/storage:/qdrant/storage qdrant/qdrant:latest >/dev/null
fi
# Wait for readiness rather than sleeping a guessed number of seconds.
for _ in $(seq 1 60); do
    curl -sf localhost:6333/healthz >/dev/null && break
    sleep 1
done
curl -sf localhost:6333/healthz >/dev/null || { echo "qdrant did not come up"; exit 1; }
echo "qdrant is up"

pip3 install -q --break-system-packages qdrant-client 2>/dev/null \
  || pip3 install -q qdrant-client

say "pass 1 of 2: int8 -- the current configuration"
python3 testdrive/migrate.py http://localhost:6333 int8
python3 testdrive/qdrant_bench.py http://localhost:6333 int8

say "dropping the int8 collection"
# Otherwise its resident memory is counted against the binary measurement and
# binary looks worse than it is.
python3 - <<'PY'
from qdrant_client import QdrantClient
QdrantClient(url='http://localhost:6333', timeout=120).delete_collection('images-int8')
print('dropped images-int8')
PY
sleep 5

say "pass 2 of 2: binary -- the half-price question"
python3 testdrive/migrate.py http://localhost:6333 binary
python3 testdrive/qdrant_bench.py http://localhost:6333 binary

say "results -- send both of these back"
cat /tmp/step-d-int8.json
echo
cat /tmp/step-d-binary.json
echo
printf '\n\033[1mNow delete the server in the Hetzner console.\033[0m\n'
