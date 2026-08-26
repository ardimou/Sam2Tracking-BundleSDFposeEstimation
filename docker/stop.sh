#!/usr/bin/env bash
set -euo pipefail

docker stop "${OBJECT_TRACKER_CONTAINER:-object-tracker}"
