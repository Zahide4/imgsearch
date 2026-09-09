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

# Hetzner's Ubuntu images log you in as `ubuntu`, not root.
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
# Storage under $HOME so the benchmark can measure its size without sudo.
export QDRANT_STORAGE="${QDRANT_STORAGE:-$HOME/qdrant-storage}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "preparing qdrant"
# No root assumed. Hetzner's image gives the `ubuntu` user no sudo rights at
# all, so Docker is not installable -- but Qdrant ships a static binary, which
# needs nothing but a directory to write to.
if command -v docker >/dev/null && $SUDO -n true 2>/dev/null; then
    mkdir -p "$QDRANT_STORAGE"
    $SUDO docker ps --format '{{.Names}}' | grep -q '^qdrant$' || {
        $SUDO docker rm -f qdrant >/dev/null 2>&1 || true
        $SUDO docker run -d --name qdrant -p 6333:6333 \
            -v "$QDRANT_STORAGE":/qdrant/storage qdrant/qdrant:latest >/dev/null
    }
else
    if [ ! -x "$HOME/qdrant" ]; then
        ARCH=$(uname -m)
        VERSION=$(curl -sL https://api.github.com/repos/qdrant/qdrant/releases/latest \
                  | grep -oE '"tag_name": "[^"]*"' | head -1 | cut -d'"' -f4)
        curl -sL -o "$HOME/qdrant.tar.gz" \
          "https://github.com/qdrant/qdrant/releases/download/${VERSION}/qdrant-${ARCH}-unknown-linux-musl.tar.gz"
        tar xzf "$HOME/qdrant.tar.gz" -C "$HOME" && rm "$HOME/qdrant.tar.gz"
        chmod +x "$HOME/qdrant"
    fi
    if ! pgrep -f "$HOME/qdrant" >/dev/null; then
        mkdir -p "$QDRANT_STORAGE"
        ( cd "$HOME" && QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
          nohup "$HOME/qdrant" > "$HOME/qdrant.log" 2>&1 & )
    fi
fi

export PATH="$HOME/.local/bin:$PATH"
if ! python3 -c 'import qdrant_client' 2>/dev/null; then
    python3 -m pip --version >/dev/null 2>&1 || {
        curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
        python3 /tmp/get-pip.py --user --break-system-packages -q
    }
    python3 -m pip install --user --break-system-packages -q qdrant-client
fi

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
