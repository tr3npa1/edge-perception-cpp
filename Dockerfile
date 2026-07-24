# syntax=docker/dockerfile:1.7

# =============================================================================
# edge-perception-cpp
#
# Linux NVIDIA build and runtime image.
#
# Base stack:
#   Ubuntu 24.04
#   CUDA 12.8 generation
#   TensorRT 10.9 generation
#
# The image compiles the native C++ runtime, generates the deterministic ONNX
# test fixture, runs the complete CPU-safe CTest suite during docker build, and
# installs the release binaries into /opt/edge-perception.
#
# Models, datasets, TensorRT engines, benchmark inputs and generated output are
# intentionally not embedded. Mount them when the container runs.
# =============================================================================

ARG TENSORRT_CONTAINER=nvcr.io/nvidia/tensorrt:25.03-py3


# =============================================================================
# Builder
# =============================================================================

FROM ${TENSORRT_CONTAINER} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.26.0
ARG CUDA_ARCHITECTURES=75
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV YOLO_AUTOINSTALL=false

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# Native build dependencies
# -----------------------------------------------------------------------------

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libopencv-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python packages required only by build-time tests
#
# Keep these aligned with the validated repository pins.
# -----------------------------------------------------------------------------

RUN python3 -m pip install \
        --no-cache-dir \
        --break-system-packages \
        "numpy==2.2.6" \
        "protobuf==6.33.6" \
        "onnx==1.22.0" \
        "pytest==9.1.1" \
        "PyYAML==6.0.3"

# -----------------------------------------------------------------------------
# ONNX Runtime GPU C/C++ distribution
#
# ORT 1.26 is used in the coherent Linux CUDA-12 container stack. The portable
# ONNX artifact remains independent of this deployment package version.
# -----------------------------------------------------------------------------

RUN set -eux \
    && curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 3 \
        --output /tmp/onnxruntime.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz" \
    && mkdir -p /opt \
    && tar \
        --extract \
        --gzip \
        --file /tmp/onnxruntime.tgz \
        --directory /opt \
    && mv \
        "/opt/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}" \
        /opt/onnxruntime \
    && test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h \
    && test -f /opt/onnxruntime/lib/libonnxruntime.so \
    && rm -f /tmp/onnxruntime.tgz

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV ONNXRUNTIME_RUNTIME_DIR=/opt/onnxruntime/lib
ENV EDGE_ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV EDGE_ONNXRUNTIME_RUNTIME_DIR=/opt/onnxruntime/lib

ENV TENSORRT_ROOT=/usr
ENV EDGE_TENSORRT_ROOT=/usr

ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV EDGE_CUDA_ROOT=/usr/local/cuda

ENV OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
ENV EDGE_OPENCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# -----------------------------------------------------------------------------
# Source
# -----------------------------------------------------------------------------

WORKDIR /workspace/source

COPY . .

# -----------------------------------------------------------------------------
# Configure
#
# Direct CMake configuration is used instead of a Windows-oriented preset so
# every Linux dependency path is explicit inside the image.
# -----------------------------------------------------------------------------

RUN cmake \
        -S /workspace/source \
        -B /workspace/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/opt/edge-perception \
        -DONNXRUNTIME_ROOT=/opt/onnxruntime \
        -DONNXRUNTIME_RUNTIME_DIR=/opt/onnxruntime/lib \
        -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
        -DTENSORRT_ROOT=/usr \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DEDGE_BUILD_TESTS=ON \
        -DEDGE_ENABLE_TENSORRT=ON \
        -DEDGE_ENABLE_CUDA_PIPELINE=ON \
        -DEDGE_COPY_RUNTIME_DLLS=OFF \
        -DEDGE_ENABLE_IPO=OFF \
        -DEDGE_WARNINGS_AS_ERRORS=OFF

# -----------------------------------------------------------------------------
# Compile
# -----------------------------------------------------------------------------

RUN cmake \
        --build /workspace/build \
        --config Release \
        --parallel "$(nproc)"

# -----------------------------------------------------------------------------
# Verify that tests are registered
# -----------------------------------------------------------------------------

RUN set -eux; \
    test_count="$(ctest \
        --test-dir /workspace/build \
        --build-config Release \
        --show-only=json-v1 \
        | python3 -c 'import json, sys; print(len(json.load(sys.stdin)["tests"]))')"; \
    echo "Registered CTest tests: ${test_count}"; \
    test "${test_count}" -eq 4

# -----------------------------------------------------------------------------
# Run all CPU-safe tests during image creation
#
# The Docker build itself does not require an exposed GPU. Automatic provider
# selection must fall back to ORT CPU when accelerated initialization is not
# usable during the build.
# -----------------------------------------------------------------------------

RUN ctest \
        --test-dir /workspace/build \
        --build-config Release \
        --output-on-failure

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

RUN cmake \
        --install /workspace/build \
        --config Release \
        --prefix /opt/edge-perception

# Verify the installed executable exists before starting the runtime stage.
RUN test -x /opt/edge-perception/bin/edge_perception


# =============================================================================
# Runtime
# =============================================================================

FROM ${TENSORRT_CONTAINER} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="edge-perception-cpp"
LABEL org.opencontainers.image.description="Portable ONNX Runtime inference with optional CUDA, TensorRT and fused CUDA acceleration"
LABEL org.opencontainers.image.source="https://github.com/tr3npa1/edge-perception-cpp"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.vendor="tr3npa1"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_AUTOINSTALL=false

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV ONNXRUNTIME_RUNTIME_DIR=/opt/onnxruntime/lib
ENV EDGE_ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV EDGE_ONNXRUNTIME_RUNTIME_DIR=/opt/onnxruntime/lib

ENV TENSORRT_ROOT=/usr
ENV EDGE_TENSORRT_ROOT=/usr

ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV EDGE_CUDA_ROOT=/usr/local/cuda

ENV OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
ENV EDGE_OPENCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV LD_LIBRARY_PATH=/opt/edge-perception/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install only the runtime package set needed by the application.
#
# libopencv-dev is used here instead of Ubuntu ABI-specific component package
# names so the Dockerfile remains stable across compatible Ubuntu 24.04 package
# revisions.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /opt/edge-perception /opt/edge-perception

# trtexec is supplied by the NVIDIA TensorRT base image. Verify that either the
# command is already available through PATH or that a copy exists in a standard
# TensorRT location.
RUN command -v trtexec \
    || find /opt /usr /workspace \
        -type f \
        -name trtexec \
        -perm -111 \
        -print \
        -quit

RUN useradd \
        --create-home \
        --uid 10001 \
        --shell /bin/bash \
        edge \
    && mkdir -p \
        /workspace/cache \
        /workspace/inputs \
        /workspace/models \
        /workspace/outputs \
    && chown -R edge:edge \
        /workspace \
        /opt/edge-perception

WORKDIR /workspace

USER edge

ENTRYPOINT ["/opt/edge-perception/bin/edge_perception"]

CMD ["--help"]