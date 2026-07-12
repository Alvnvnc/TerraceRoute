#!/usr/bin/env bash
# Build and publish one event track. Keep the scored images independent.
set -euo pipefail

TRACK="${1:-}"
TAG="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$TRACK" in
  1|track1)
    exec "$ROOT/track1/scripts/build_and_push.sh" "${TAG:-track1-v1}"
    ;;
  2|track2)
    exec "$ROOT/track2/scripts/build_and_push.sh" "${TAG:-track2-v1}"
    ;;
  *)
    echo "usage: $0 <1|2> [tag]" >&2
    exit 2
    ;;
esac
