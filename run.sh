#!/usr/bin/env bash
# Host-side one-liner for Linux + NVIDIA Docker.
# Usage:
#   ./run.sh                  # reproduce from committed caches
#   ./run.sh full             # cache HF_MODEL on GPU + aggregate + figures
#   ./run.sh pythia69         # EleutherAI/pythia-6.9b float16 on GPU
#   ./run.sh shell            # bash inside the container
#   MODE=full HF_MODEL=EleutherAI/pythia-6.9b DTYPE=float16 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-${MODE:-reproduce}}"
export MODE
shift || true

if ! command -v docker >/dev/null; then
  echo "docker not found"; exit 1
fi

# Prefer Compose v2 plugin; fall back to docker-compose.
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
elif command -v docker-compose >/dev/null; then
  DC=(docker-compose)
else
  echo "docker compose not found"; exit 1
fi

echo "Building and running mode=${MODE} ..."
"${DC[@]}" build
"${DC[@]}" run --rm minkpp "${MODE}" "$@"
