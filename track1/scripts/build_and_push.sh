#!/usr/bin/env bash
# Build a linux/amd64 image & push it to a public registry for the Track 1 submission.
#
# Usage:
#   REGISTRY=ghcr.io/USERNAME ./scripts/build_and_push.sh v1
#   (log in first: echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin)
#
# Submission reference = <REGISTRY>/terraceroute:<TAG> (PUBLIC, include the tag, no https://).
set -euo pipefail

TAG="${1:?tag required, e.g. v1}"
REGISTRY="${REGISTRY:?set REGISTRY, e.g. ghcr.io/username or docker.io/username}"
IMAGE="${REGISTRY}/terraceroute:${TAG}"

cd "$(dirname "$0")/.."

# linux/amd64 is required (rule). Use buildx for cross-builds on Apple Silicon.
if docker buildx version >/dev/null 2>&1; then
  docker buildx build --platform linux/amd64 -f Dockerfile \
    -t "$IMAGE" --push .
else
  docker build --platform linux/amd64 -f Dockerfile -t "$IMAGE" .
  docker push "$IMAGE"
fi

echo
echo "PUSHED: $IMAGE"
echo "Compressed size:"
docker manifest inspect "$IMAGE" >/dev/null 2>&1 && echo "  (check on the registry; the limit is 10 GB compressed)"
echo
echo "Submission reference (paste into lablab): ${IMAGE}"
