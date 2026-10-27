# Pin this image by digest for a published experiment and record that digest.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir triton==3.2.0
COPY sandbox/runner.py /opt/runner.py
ENV TRITON_CACHE_DIR=/tmp/triton-cache HOME=/tmp PYTHONDONTWRITEBYTECODE=1
WORKDIR /tmp
USER 65534:65534
