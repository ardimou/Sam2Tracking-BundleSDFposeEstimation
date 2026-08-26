# Third-party software and models

This repository combines original ROS 2 integration code with third-party
software and pretrained model weights. The root `LICENSE` applies only to the
original code in this repository. It does not replace third-party terms.

## Important use restriction

The vendored BundleSDF source is licensed by NVIDIA for **non-commercial
research or evaluation use only**. Distributing or using this repository does
not remove that restriction. Review the complete license before use:

- Component: BundleSDF
- Source: https://github.com/NVlabs/BundleSDF
- Vendored revision: `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`
- License: NVIDIA Source Code License-NC
- License text: `third_party/BundleSDF/LICENSE.txt`

## Vendored components

The following components are included directly in `third_party/BundleSDF` and
retain their upstream license files and notices:

| Component | Upstream | License file |
|---|---|---|
| BundleTrack | https://github.com/wenbowen123/BundleTrack | `third_party/BundleSDF/BundleTrack/LICENSE` (BSD 3-Clause) |
| LoFTR and `outdoor_ds.ckpt` | https://github.com/zju3dv/LoFTR | `third_party/BundleSDF/BundleTrack/LoFTR/LICENSE` (Apache-2.0) |
| torch-ngp grid encoder code | https://github.com/ashawkey/torch-ngp | `third_party/BundleSDF/mycuda/torch_ngp_grid_encoder/LICENSE` (MIT) |
| NVIDIA CUDA sample-derived code | https://github.com/NVIDIA/cuda-samples | `third_party/BundleSDF/BundleTrack/src/cuda/LICENSE` |

The LoFTR checkpoint is approximately 46 MB and is redistributed with this
source tree. Its provenance is the download linked by the upstream BundleSDF
and LoFTR documentation.

## Downloaded while building the image

These projects or model snapshots are fetched by `docker/Dockerfile`; they are
not vendored in the Git source tree:

| Component | Pinned source/model | License |
|---|---|---|
| SAM 2 | https://github.com/facebookresearch/sam2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0; optional `cc_torch` code is BSD 3-Clause |
| SAM 2.1 Hiera Tiny checkpoint | https://huggingface.co/facebook/sam2.1-hiera-tiny | Apache-2.0 |
| Grounding DINO Tiny checkpoint | https://huggingface.co/IDEA-Research/grounding-dino-tiny | Apache-2.0 |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | BSD 3-Clause |
| OpenCV and opencv_contrib | https://github.com/opencv/opencv | Apache-2.0 |
| Point Cloud Library | https://github.com/PointCloudLibrary/pcl | BSD 3-Clause |
| Eigen | https://gitlab.com/libeigen/eigen | MPL-2.0 and component-specific licenses |
| pybind11 | https://github.com/pybind/pybind11 | BSD 3-Clause |
| yaml-cpp | https://github.com/jbeder/yaml-cpp | MIT |

Python and Ubuntu/ROS packages installed by `pip` and `apt` remain under their
respective licenses. Their package metadata and license files are available in
the built image.

## NVIDIA base container

The Dockerfile derives from
`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`. The Git repository contains
only build instructions and does not redistribute that base image. Pulling,
building, using, or publishing the resulting image is subject to the NVIDIA
DeepStream/NGC license displayed by the container and available in the image at
`/opt/nvidia/deepstream/deepstream/LicenseAgreement.pdf`.

Do not assume that the root Apache-2.0 license permits redistribution of a
built DeepStream-derived image. Check the license applicable to this exact NGC
tag before publishing the image to a registry.

## Trademarks

NVIDIA, DeepStream, ROS, Grounding DINO, SAM, and other names are trademarks or
project names of their respective owners. Their mention identifies compatible
or incorporated technology and does not imply endorsement.
