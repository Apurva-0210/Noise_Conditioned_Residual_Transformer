# ==============================================================================
# NCRT & NCRN: Spatio-Temporal Massive MIMO Channel Estimation
# Multi-arch Dockerfile for CPU & GPU Reproducibility (x86_64 and arm64)
# ==============================================================================
FROM python:3.10-slim

# Prevent interactive prompts and Python output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies for scientific libraries and headless plotting
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository source, weights, and scripts
COPY . /workspace

# Set executable permissions for pipeline runner scripts
RUN chmod +x train_all.sh eval_all.sh

# Default command: interactive shell with reproduction instructions
CMD ["/bin/bash"]
