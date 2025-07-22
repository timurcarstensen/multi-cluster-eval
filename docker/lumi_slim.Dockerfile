# -------- Stage 1: obtain full ROCm stack (build stage) ---------------------
ARG ROCM_VER=6.4.1
FROM rocm/dev-ubuntu-24.04:${ROCM_VER}-complete AS rocm_binary

# -------- Stage 2: create minimal runtime image ----------------------------
ARG ROCM_VER=6.4.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.4
FROM ubuntu:24.04

# Keep image non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Copy only the ROCm runtime tree we need
COPY --from=rocm_binary /opt/rocm /opt/rocm
# Provide a versioned symlink for software that expects it
RUN ln -s /opt/rocm /opt/rocm-${ROCM_VER} || true

# Install minimal system libraries required by PyTorch & common ML tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-turbo8 \
        libtinfo6 \
        python3 \
        python3-pip \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set up ROCm environment paths (include uv once installed)
ENV PATH=/root/.local/bin:/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64

# Install uv Python package manager and then required Python wheels
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system --break-system-packages --no-cache-dir --pre \
        torch==2.7.1+rocm6.4.1 \
        torchvision \
        torchaudio \
        pytorch-triton-rocm \
        --extra-index-url ${TORCH_INDEX_URL} && \
    pip cache purge

# Allow non-root user access to GPU devices in many HPC environments
RUN groupadd -r render -g 109

WORKDIR /workspace
CMD ["python3"] 