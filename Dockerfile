# Base image with ROCm 6.1.2
FROM rocm/dev-ubuntu-22.04:6.1.2-complete

# Architecture-specific environment variables
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HCC_AMDGPU_TARGET=gfx1030
ENV GPU_ARCHS=gfx1030

# Install build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libnuma-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    rocm-device-libs \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with ROCm 6.1 compatibility
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1/

# Build bitsandbytes with architecture-specific optimizations
WORKDIR /app
RUN git clone -b multi-backend-refactor --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git && \
    cd bitsandbytes && \
    pip install -r requirements-dev.txt && \
    cmake -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1030" -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) && \
    pip install -e .

# Final environment setup
ENV PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
WORKDIR /workspace
