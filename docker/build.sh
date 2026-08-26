#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${OBJECT_TRACKER_IMAGE:-object-tracker:latest}"

docker build --network host \
  -f "$ROOT/docker/Dockerfile" \
  -t "$IMAGE" \
  "$ROOT"
