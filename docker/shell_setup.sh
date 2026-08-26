#!/usr/bin/env bash

object_tracker_restore_nounset=0
case $- in
  *u*) object_tracker_restore_nounset=1; set +u ;;
esac

source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/BundleSDF:/opt/BundleSDF/mycuda:${PYTHONPATH:-}"

if [ "$object_tracker_restore_nounset" -eq 1 ]; then
  set -u
fi
unset object_tracker_restore_nounset
