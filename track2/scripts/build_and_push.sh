#!/usr/bin/env bash
# Build and publish the standalone Track 2 scoring image.
set -euo pipefail

TAG="${1:-track2-v1}"
REGISTRY="${REGISTRY:-ghcr.io/alvnvnc}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$REGISTRY" == http://* || "$REGISTRY" == https://* ]]; then
  echo "REGISTRY must not include http:// or https://" >&2
  exit 2
fi

IMAGE="${REGISTRY%/}/terraceroute:${TAG}"

if docker buildx version >/dev/null 2>&1; then
  docker buildx build --platform linux/amd64 \
    -f "$ROOT/Dockerfile" -t "$IMAGE" --push "$ROOT"
else
  docker build --platform linux/amd64 \
    -f "$ROOT/Dockerfile" -t "$IMAGE" "$ROOT"
  docker push "$IMAGE"
fi

echo "PUSHED: $IMAGE"
echo "Track 2 submission reference: $IMAGE"
