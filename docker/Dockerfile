FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

ARG CMAKE_VERSION_MAJOR=3
ARG CMAKE_VERSION_MINOR=25
ARG CMAKE_VERSION_PATCH=3

ARG EIGEN_VERSION_MAJOR=3
ARG EIGEN_VERSION_MINOR=4
ARG EIGEN_VERSION_PATCH=0

ARG OPENCV_VERSION_MAJOR=4
ARG OPENCV_VERSION_MINOR=11
ARG OPENCV_VERSION_PATCH=0

ARG PCL_VERSION_MAJOR=1
ARG PCL_VERSION_MINOR=10
ARG PCL_VERSION_PATCH=0

ARG PYBIND11_VERSION_MAJOR=2
ARG PYBIND11_VERSION_MINOR=13
ARG PYBIND11_VERSION_PATCH=0

ARG YAML_CPP_VERSION_MAJOR=0
ARG YAML_CPP_VERSION_MINOR=8
ARG YAML_CPP_VERSION_PATCH=0

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    # Python and development tools
    python3-pip \
    python3-dev \
    # Build tools and compilers
    build-essential \
    cmake \
    cmake-curses-gui \
    checkinstall \
    g++ \
    gcc \
    gfortran \
    # Version control and utilities
    git \
    vim \
    tmux \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    gnupg \
    software-properties-common \
    # Graphics and GUI libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    qtbase5-dev \
    # Math and science libraries
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # Security and networking
    libssl-dev \
    libzmq3-dev \
    # Boost libraries
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-all-dev \
    # Point cloud and 3D processing
    libflann-dev \
    # Image and video processing
    libjpeg8-dev \
    libtiff5-dev \
    pkg-config \
    yasm \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-dev \
    libxine2-dev \
    libv4l-dev \
    libtbb-dev \
    ffmpeg \
    # Audio/video codecs
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    x264 \
    v4l-utils \
    # Protocol buffers and logging
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    # Additional libraries
    libgphoto2-dev \
    libhdf5-dev \
    doxygen \
    proj-data \
    libproj-dev \
    libyaml-cpp-dev \
    libzmq3-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*


# Install cmake
RUN cd / &&\
wget http://www.cmake.org/files/v${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}/cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
tar xf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
cd cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH} &&\
./configure &&\
make &&\
make install


# Install Eigen
RUN cd / && \
    wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}/eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    tar xf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    cd eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install && \
    cd / && \
    rm -rf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}


# Install OpenCV
RUN cd / && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv_contrib && \
    mkdir -p /opencv/build && \
    cd /opencv/build && \
    cmake ..  -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA_STUBS=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_MATLAB=OFF \
        -Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF \
        -DCUDA_FAST_MATH=ON \
        -DMKL_WITH_OPENMP=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_OPENMP=ON \
        -DWITH_QT=ON \
        -DWITH_OPENEXR=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_FFMPEG=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DENABLE_CXX11=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DOPENCV_DNN_OPENCL=OFF \
        -DWITH_CUDA=ON \
        -DWITH_OPENCL=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DOPENCV_CUDA_OPTIONS_opencv_test_cudev=-std=c++17 \
        -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6 9.0" \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DINSTALL_PKGCONFIG=ON \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DPKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /opencv /opencv_contrib

# Install PCL
RUN cd / && \
    git clone --depth 1 --branch pcl-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}.${PCL_VERSION_PATCH} https://github.com/PointCloudLibrary/pcl && \
    mkdir -p /pcl/build && \
    cd /pcl/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_apps=OFF \
        -DBUILD_GPU=OFF \
        -DBUILD_CUDA=OFF \
        -DBUILD_examples=OFF \
        -DBUILD_global_tests=OFF \
        -DBUILD_simulation=OFF \
        -DCUDA_BUILD_EMULATION=OFF \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DPCL_ENABLE_SSE=ON \
        -DPCL_SHARED_LIBS=ON \
        -DWITH_VTK=OFF \
        -DPCL_ONLY_CORE_POINT_TYPES=ON \
        -DPCL_COMMON_WARNINGS=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /pcl

# Install Pybind11
RUN cd / && \
    git clone --depth 1 --branch v${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH} https://github.com/pybind/pybind11 && \
    mkdir -p /pybind11/build && \
    cd /pybind11/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /pybind11

# Install YAML-CPP
RUN cd / && \
    git clone --depth 1 --branch ${YAML_CPP_VERSION_MAJOR}.${YAML_CPP_VERSION_MINOR}.${YAML_CPP_VERSION_PATCH} https://github.com/jbeder/yaml-cpp && \
    mkdir -p /yaml-cpp/build && \
    cd /yaml-cpp/build && \
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DINSTALL_GTEST=OFF \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DYAML_BUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /yaml-cpp

SHELL ["/bin/bash", "--login", "-c"]

# Install Python packages directly with system Python to add additional conda env since docker should provide isolated environment
RUN apt-get remove -y python3-blinker || true && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch==2.6.0 torchvision==0.21.0

RUN pip3 install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html

RUN pip3 install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN pip3 install --break-system-packages --force-reinstall blinker && \
    pip3 install trimesh wandb matplotlib imageio tqdm open3d ruamel.yaml sacred kornia pymongo pyrender jupyterlab ninja "Cython>=0.29.37" yacs && \
    pip3 install scipy scikit-learn


ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV OPENCV_IO_ENABLE_OPENEXR=1
# Fix multiprocessing issues
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1


RUN imageio_download_bin freeimage
#### Kaolin will change numpy version
RUN pip3 install --break-system-packages --force-reinstall blinker && \
    pip3 install numpy==1.26.4 transformations einops scikit-image awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui pytinyrenderer cython-npm chardet openpyxl

# DeepStream ships an Intel RealSense source whose old signing key is not
# present. This image receives Xtion frames over ROS and does not use that
# repository, so remove it before subsequent apt operations.
RUN find /etc/apt/sources.list.d -maxdepth 1 -type f \
      -exec sed -i '\|librealsense\.intel\.com|d' {} + && \
    sed -i '\|librealsense\.intel\.com|d' /etc/apt/sources.list && \
    apt-get update --fix-missing && \
    apt install -y rsync lbzip2 pigz zip p7zip-full p7zip-rar

# ROS 2 Humble lives in this image so the BundleSDF tracking node can use the
# upstream in-process API and participate directly in the host DDS graph.
# cv_bridge loads Ubuntu's OpenCV image-codec library at runtime. The
# DeepStream records some HEVC codec packages as installed but strips their
# shared objects. Force dpkg to unpack them again; a plain install is a no-op.
RUN apt-get update && apt-get install -y --no-install-recommends \
      locales curl gnupg2 lsb-release && \
    locale-gen en_US en_US.UTF-8 && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && \
    apt-get remove -y libcurl4-gnutls-dev && \
    apt-get install -y --no-install-recommends libcurl4-openssl-dev && \
    apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-cv-bridge \
      ros-humble-geometry-msgs \
      ros-humble-message-filters \
      ros-humble-openni2-camera \
      ros-humble-python-qt-binding \
      ros-humble-rcl-interfaces \
      ros-humble-rclcpp-components \
      ros-humble-rviz2 \
      ros-humble-sensor-msgs \
      ros-humble-std-msgs \
      ros-humble-std-srvs \
      ros-humble-tf2-ros \
      ros-humble-visualization-msgs \
      ros-humble-rosidl-default-generators \
      libde265-0 \
      libx265-199 \
      libusb-1.0-0 \
      usbutils \
      python3-pyqt5 \
      python3-colcon-common-extensions \
      python3-rosdep

RUN apt-get install -y --reinstall libde265-0 libx265-199 && \
    ldconfig && \
    test -e /usr/lib/x86_64-linux-gnu/libde265.so.0 && \
    test -e /usr/lib/x86_64-linux-gnu/libx265.so.199 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble

ENV HF_HOME=/opt/models
ENV TRANSFORMERS_CACHE=/opt/models
ENV SAM2_BUILD_CUDA=0

RUN pip3 install --no-cache-dir \
      accelerate \
      huggingface-hub \
      hydra-core \
      iopath \
      omegaconf \
      pillow \
      transformers && \
    pip3 install --no-cache-dir --no-deps \
      "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny'); snapshot_download('facebook/sam2.1-hiera-tiny')"

RUN source /opt/ros/humble/setup.bash && \
    python3 -c "from python_qt_binding.QtCore import QT_VERSION_STR; print('ROS Qt:', QT_VERSION_STR)"

COPY third_party/BundleSDF /opt/BundleSDF
WORKDIR /opt/BundleSDF
RUN chmod +x build.sh && ./build.sh

COPY src /workspace/src
WORKDIR /workspace
RUN source /opt/ros/humble/setup.bash && \
    colcon build --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    ln -s /workspace/install /opt/object_tracker && \
    rm -rf /workspace/build /workspace/log

COPY docker/entrypoint.sh /usr/local/bin/object-tracker-entrypoint
COPY docker/shell_setup.sh /etc/object-tracker/setup.bash
COPY docker/bashrc /root/.bashrc
COPY LICENSE THIRD_PARTY_NOTICES.md CITATIONS.md /licenses/object_tracker/
RUN chmod +x /usr/local/bin/object-tracker-entrypoint && \
    mkdir -p /data/sessions && \
    test -f /workspace/install/setup.bash && \
    test -f /opt/object_tracker/setup.bash

ENV PYTHONPATH=/opt/BundleSDF:/opt/BundleSDF/mycuda
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/object-tracker-entrypoint"]
CMD ["bash"]
