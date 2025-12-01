# Lightweight multi-arch Python image suitable for Raspberry Pi (armv7/aarch64)
# Installs tflite-runtime and minimal numeric libs to run TFLite inference.

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev tzdata python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install TFLite runtime (Coral repo provides multi-arch wheels) and common deps
RUN pip install --no-cache-dir \
    --extra-index-url https://google-coral.github.io/py-repo/ \
    tflite-runtime requests

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TZ=America/Sao_Paulo

# In development, prefer bind-mounting the repo with docker-compose
# COPY . /app