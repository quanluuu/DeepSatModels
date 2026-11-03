# Streamlit demo: shows the TSViT_DINOv3 predictions precomputed in app/data/demo_patches.npz; no model inside.
#   docker build -f docker/demo.Dockerfile -t deepsat-demo .
#   docker run --rm -p 8501:8501 deepsat-demo        -> http://localhost:8501
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home app
WORKDIR /srv
COPY --chown=app LICENSE.txt ./
COPY --chown=app app/crop_recognition_app.py app/demo_core.py app/
COPY --chown=app app/data/demo_patches.npz app/data/
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app/crop_recognition_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
