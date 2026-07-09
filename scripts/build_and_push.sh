#!/usr/bin/env bash
# Build image linux/amd64 & push ke registry publik untuk submission Track 1.
#
# Pakai:
#   REGISTRY=ghcr.io/USERNAME ./scripts/build_and_push.sh v1
#   (login dulu: echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin)
#
# Referensi submission = <REGISTRY>/terraceroute:<TAG> (PUBLIC, sertakan tag, tanpa https://).
set -euo pipefail

TAG="${1:?tag wajib, mis: v1}"
REGISTRY="${REGISTRY:?set REGISTRY, mis: ghcr.io/username atau docker.io/username}"
IMAGE="${REGISTRY}/terraceroute:${TAG}"

cd "$(dirname "$0")/.."

# linux/amd64 wajib (aturan). buildx untuk cross-build bila di Apple Silicon.
if docker buildx version >/dev/null 2>&1; then
  docker buildx build --platform linux/amd64 -f Dockerfile.agent \
    -t "$IMAGE" --push .
else
  docker build --platform linux/amd64 -f Dockerfile.agent -t "$IMAGE" .
  docker push "$IMAGE"
fi

echo
echo "PUSHED: $IMAGE"
echo "Compressed size:"
docker manifest inspect "$IMAGE" >/dev/null 2>&1 && echo "  (cek di registry; batas 10 GB compressed)"
echo
echo "Submission reference (paste ke lablab): ${IMAGE}"
