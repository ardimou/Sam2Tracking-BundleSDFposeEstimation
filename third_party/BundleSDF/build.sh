#!/usr/bin/env bash
set -euo pipefail

ROOT=$(pwd)

# Set PyTorch library path
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_LIBRARIES="/usr/local/lib/python3.10/dist-packages/torch/lib"

# Additional PyTorch environment variables
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"

# Ensure PyTorch can be found
export PYTHONPATH="${ROOT}:${ROOT}/mycuda:/usr/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"

# Print debug info
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Testing PyTorch import..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Build in-place because this script runs in a disposable container. Installing
# into that container's site-packages would vanish when the container exits;
# in-place outputs persist in the bind-mounted BundleSDF checkout.
cd "${ROOT}/mycuda"
rm -rf build
python3 setup.py build_ext --inplace

# Keep successful objects between invocations. This makes recovery from a
# failure in a later verification/import step incremental instead of forcing a
# complete BundleTrack rebuild.
mkdir -p "${ROOT}/BundleTrack/build"
cd "${ROOT}/BundleTrack/build"
cmake ..
make -j11

# Fail the build here, with the real loader error, rather than later when the
# ROS node receives its first mask. Both outputs live in the bind-mounted
# checkout and therefore persist after this temporary build container exits.
cd "${ROOT}"
python3 -c "import sys; from mycuda import common; import gridencoder; sys.path.insert(0, '${ROOT}/BundleTrack/build'); import my_cpp; print('BundleSDF native extensions: OK')"
