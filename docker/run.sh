#!/usr/bin/env bash
set -euo pipefail

IMAGE="${OBJECT_TRACKER_IMAGE:-object-tracker:latest}"
CONTAINER="${OBJECT_TRACKER_CONTAINER:-object-tracker}"
DATA_DIR="${OBJECT_TRACKER_DATA:-$PWD/data}"

mkdir -p "$DATA_DIR"

docker run --rm -d \
  --name "$CONTAINER" \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$DATA_DIR:/data" \
  "$IMAGE" sleep infinity

echo "Container '$CONTAINER' is running."
echo "Open a shell with: docker exec -it $CONTAINER bash"
echo "The shell sources ROS Humble and /workspace/install automatically."
