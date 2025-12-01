# Imagem baseada em Debian Bookworm, usando Python do sistema (compatível com armhf/aarch64)
FROM debian:bookworm-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python-is-python3 python3-venv \
    libatlas-base-dev tzdata python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtualenv with system site-packages so apt numpy is visible
RUN python3 -m venv /opt/venv --system-site-packages
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Install TFLite runtime (Coral repo provides multi-arch wheels) and requests in venv
RUN pip install --no-cache-dir \
    --extra-index-url https://google-coral.github.io/py-repo/ \
    tflite-runtime requests

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TZ=America/Sao_Paulo

# In development, prefer bind-mounting the repo with docker-compose
# COPY . /app