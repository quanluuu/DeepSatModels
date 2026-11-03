# Inference API: TSViT_DINOv3 on CPU behind FastAPI (GET /health, POST /predict).
# Needs weights/best.pth (GitHub Release weights-v1) and weights/dinov3 (facebookresearch/dinov3 at commit 6876159) in the context.
#   docker build -f docker/api.Dockerfile -t deepsat-api .
#   docker run --rm -p 8000:8000 deepsat-api          -> curl http://localhost:8000/health
FROM python:3.11-slim

ARG MODEL_VERSION=tsvit-dinov3-fold1-t9
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_VERSION=${MODEL_VERSION} \
    MODEL_CONFIG=configs/PASTIS24/TSViT_DINOv3_fold1.yaml \
    CHECKPOINT_PATH=weights/best.pth \
    DINOV3_REPO=weights/dinov3 \
    CONFIDENCE_THRESHOLD=0.80 \
    INFER_TIMEOUT_S=10
LABEL org.opencontainers.image.title="deepsat-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.licenses="see /srv/LICENSE.txt and /srv/weights/dinov3/LICENSE.md"

# CPU-only torch first, so requirements_dinov3.txt (torch>=2.4) does not pull the CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements_dinov3.txt api/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_dinov3.txt -r /tmp/requirements.txt

RUN useradd --create-home app
# the repo root is the working directory: data/__init__.py reads data/datasets.yaml relative to it
WORKDIR /srv
COPY --chown=app LICENSE.txt ./
COPY --chown=app models/ models/
COPY --chown=app utils/ utils/
COPY --chown=app data/ data/
COPY --chown=app configs/PASTIS24/TSViT_DINOv3_fold1.yaml configs/PASTIS24/
COPY --chown=app app/demo_core.py app/
COPY --chown=app api/ api/
COPY --chown=app weights/best.pth weights/
COPY --chown=app weights/dinov3/ weights/dinov3/
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "--factory", "api.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
