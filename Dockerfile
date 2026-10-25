# GPU image for Min-K%++ practical pipeline (Linux + NVIDIA).
# Torch/CUDA come from the base image; other deps from requirements-docker.txt.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

RUN chmod +x docker/entrypoint.sh run.sh \
    && mkdir -p /data/hf-cache cache results/figures notebooks

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["reproduce"]
