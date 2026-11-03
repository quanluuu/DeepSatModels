# Spec: containers

## Objective

Đóng gói hai thành phần thành Docker image. Mục đích là xóa câu "máy em chạy được, máy khác không" (slide buổi 8 và 10). Người có Docker chỉ cần một lệnh `docker run` là chạy được.

Acceptance criteria:

1. `docker build -f docker/api.Dockerfile -t deepsat-api .` thành công khi thư mục `weights/` đã có đủ file.
2. `docker run -p 8000:8000 deepsat-api`: trong vòng 120 giây, `GET /health` trả `{"status": "ok", ...}`.
3. Trong container API, `POST /predict` với `tests/fixtures/window.npz` đạt acceptance criterion 2 của `inference-api`. Đây là test AC-AI-08.
4. `docker run -p 8501:8501 deepsat-demo`: trong vòng 60 giây, `GET /_stcore/health` trả `ok`.
5. Cả hai container chạy bằng user không phải root và không cần Internet lúc chạy.
6. Image API dưới 2.5 GB. Image demo dưới 600 MB.
7. Image API có `LABEL org.opencontainers.image.version` và biến env `MODEL_VERSION`. Image có kèm file license của DINOv3 và của repo.

## Tech Stack

- Image API: base `python:3.11-slim`, torch CPU wheel, `uvicorn`.
- Image demo: base `python:3.13-slim`, `streamlit`.
- Mỗi image có `HEALTHCHECK`.

## Commands

```bash
# weights chuẩn bị sẵn (CI tải từ Release, xem delivery-pipeline)
ls weights/   # best.pth  dinov3/  dinov3_vits16_pretrain_lvd1689m-08c60483.pth  SHA256SUMS

docker build -f docker/api.Dockerfile  -t deepsat-api  .
docker build -f docker/demo.Dockerfile -t deepsat-demo .

docker run --rm -p 8000:8000 deepsat-api
curl -fsS http://localhost:8000/health
curl -fsS -X POST http://localhost:8000/predict -F "file=@tests/fixtures/window.npz"

docker run --rm -p 8501:8501 deepsat-demo
curl -fsS http://localhost:8501/_stcore/health

python scripts/smoke_api.py http://localhost:8000   # kiểm tra contract, thoát 1 nếu sai
```

## Project Structure

```text
+ docker/api.Dockerfile    -> COPY models/ utils/ data/PASTIS24/data_transforms.py configs/PASTIS24/TSViT_DINOv3_fold1.yaml api/ weights/
+ docker/demo.Dockerfile   -> COPY requirements.txt app/ (bỏ app/precompute.py)
+ .dockerignore            -> .venv, *.zip, *.zip.part, TSViT_fold1_full_a100_opt/, ignore/, tasks/, notebook, .git
+ scripts/smoke_api.py     -> gọi /health + /predict, kiểm contract, dùng trong CI
```

Chỉ copy đúng các file API cần, không copy cả repo. Mục đích là giữ image nhỏ và không chứa dữ liệu train.

## Code Style

Dockerfile ngắn. Mỗi lệnh `RUN` có một mục đích. Lệnh cài dependency đặt trước lệnh copy code để tận dụng layer cache.

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements_dinov3.txt api/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_dinov3.txt -r /tmp/requirements.txt
RUN useradd -m app
WORKDIR /srv
COPY --chown=app . .
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing Strategy

| Tầng | Kiểm tra |
|---|---|
| Container smoke | `/health` của cả hai image |
| System (AC-AI-08) | `scripts/smoke_api.py` gọi `/predict` thật trong container API |
| Kích thước | `docker image ls` so với giới hạn ở acceptance criterion 6 |

## Boundaries

- Always: chạy bằng user không phải root; dùng `--no-cache-dir`; verify `SHA256SUMS` của weights trước khi build.
- Ask first: đổi base image; thêm GPU image (CUDA).
- Never: copy `.env`, token, dataset PASTIS hay `.venv` vào image.

## Success Criteria

Tất cả lệnh ở mục Commands chạy thành công trên máy local (Docker Desktop). Hai image đạt giới hạn kích thước.

## Open Questions

1. Có cần build multi-arch (`linux/arm64` cho Mac M-series) không? Đề xuất: chỉ build `linux/amd64` để CI nhanh hơn.
