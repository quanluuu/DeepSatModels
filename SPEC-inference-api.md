# Spec: inference-api

## Objective

Đóng gói model `TSViT_DINOv3` thành một service HTTP. Phần mềm khác gọi model qua service này, đúng kiểu "Server-side AI" của slide buổi 8. Đầu ra theo slide buổi 4, 7 và 9B: có confidence, status OK/UNCERTAIN/FALLBACK, timeout, và `model_version` để truy vết.

Người dùng: sinh viên và nhà nghiên cứu cần dự đoán loại cây trồng cho một ô 24×24 pixel (240 m × 240 m) từ chuỗi ảnh Sentinel-2.

### API contract

`GET /health` trả:

```json
{"status": "ok", "model_version": "tsvit-dinov3-fold1-t9", "device": "cpu"}
```

`POST /predict` nhận multipart, field `file` là một file `.npz` gồm:

- `img`: `int16` hoặc `float32`, shape `(T, 10, 24, 24)`, với `1 <= T <= 60`. Thứ tự band giống PASTIS.
- `doy`: `int`, shape `(T,)`, giá trị trong khoảng 1..366.

Response 200:

```json
{
  "status": "OK",
  "confidence": 0.86,
  "pred": [[0, 2, ...], ...],
  "class_names": ["Background", "Meadow", "..."],
  "area_ha": {"Soft winter wheat": 1.84, "...": 0.0},
  "model_version": "tsvit-dinov3-fold1-t9",
  "latency_ms": 2140
}
```

| status | Điều kiện | pred |
|---|---|---|
| `OK` | confidence >= `CONFIDENCE_THRESHOLD` (mặc định 0.80) | ma trận 24×24 |
| `UNCERTAIN` | confidence < ngưỡng | ma trận 24×24; client phải cho người dùng kiểm tra |
| `FALLBACK` | inference lỗi hoặc vượt `INFER_TIMEOUT_S` (mặc định 10 s) | `null`, kèm `message` |

`confidence` là trung bình theo pixel của xác suất softmax lớn nhất.

Lỗi input trả 422 với `{"detail": "..."}` và không gọi model. Các trường hợp lỗi input: file không phải `.npz` hợp lệ, file chứa object pickle, thiếu key, sai shape, `T > 60`, file lớn hơn 5 MB.

### Acceptance criteria

1. `/health` trả 200 trong vòng 1 giây sau khi model đã load.
2. `/predict` với `tests/fixtures/window.npz` trả `status` thuộc `{OK, UNCERTAIN}`, `pred` có shape 24×24, mọi giá trị trong 0..18, và có `model_version`.
3. Các trường hợp AC-AI ở mục Testing Strategy đều pass.
4. Server khởi động mà không cần GPU và không cần Internet. Weights và snapshot DINOv3 nằm sẵn trên đĩa.
5. Preprocessing dùng lại `PASTIS_segmentation_transform(cfg['MODEL'], is_training=False)`, cùng code với lúc train (slide buổi 4: "Preprocess tách riêng"). Inference không có label, nên dùng `labels` giả bằng 0.

## Tech Stack

Python 3.11 (trùng Kaggle, thỏa yêu cầu của DINOv3), `fastapi`, `uvicorn`, `python-multipart`, torch CPU, cùng các dependency trong `requirements_dinov3.txt`.

## Commands

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_dinov3.txt -r api/requirements.txt -r requirements-dev.txt

# chạy local (weights đặt trong weights/)
set MODEL_CONFIG=configs/PASTIS24/TSViT_DINOv3_fold1.yaml
set CHECKPOINT_PATH=weights/best.pth
set DINOV3_REPO=weights/dinov3
set DINOV3_WEIGHTS=weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
uvicorn api.main:app --host 0.0.0.0 --port 8000

curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -F "file=@tests/fixtures/window.npz"

pytest -q tests/test_api.py
```

## Project Structure

```text
+ api/__init__.py
+ api/main.py          -> create_app(predictor) + biến app; route /health, /predict; validate input; timeout
+ api/predictor.py     -> load model từ env, preprocess, inference, confidence, status
+ api/requirements.txt -> fastapi, uvicorn, python-multipart
+ api/MODEL_CARD.md    -> model_name, model_version, trained_on, metrics, threshold, known_limitations (slide buổi 7)
+ .env.example         -> MODEL_CONFIG, CHECKPOINT_PATH, DINOV3_REPO, DINOV3_WEIGHTS, CONFIDENCE_THRESHOLD, INFER_TIMEOUT_S, MODEL_VERSION
+ tests/test_api.py
+ tests/fixtures/window.npz   -> 1 cửa sổ 24×24 cắt từ PASTIS test split, < 1 MB
```

Các biến env ghi đè `dinov3_repo` và `dinov3_weights` trong config. Nhờ vậy không phải sửa file YAML đang dùng trên Kaggle.

## Code Style

`create_app(predictor)` nhận predictor từ ngoài vào. Test truyền stub predictor nên không cần weights.

```python
def status_of(confidence, threshold):
    """OK when the mean max-softmax reaches the threshold, else UNCERTAIN (slide 9: unit test both branches)"""
    return 'OK' if confidence >= threshold else 'UNCERTAIN'
```

## Testing Strategy

| ID | Loại | Input / Điều kiện | Expected |
|---|---|---|---|
| AC-AI-01 | Unit | `status_of(0.80, 0.80)` / `status_of(0.79, 0.80)` | `OK` / `UNCERTAIN` |
| AC-AI-02 | Integration (stub) | stub trả confidence 0.94 | 200, `status=OK`, có `model_version` |
| AC-AI-03 | Integration (stub) | stub trả confidence 0.55 | 200, `status=UNCERTAIN` |
| AC-AI-04 | Integration (stub) | stub ngủ quá `INFER_TIMEOUT_S` hoặc raise | 200, `status=FALLBACK`, `pred=null` |
| AC-AI-05 | Negative | file `.exe` đổi tên `.npz` | 422, stub không bị gọi |
| AC-AI-06 | Negative (security) | `.npz` chứa object array | 422 |
| AC-AI-07 | Boundary | `T=61`, shape sai, file 6 MB | 422 |
| AC-AI-08 | System (model thật) | `tests/fixtures/window.npz` | Như acceptance criterion 2. Chạy trong container ở module `containers` |

Test AC-AI-01..07 dùng stub và chạy được ở mọi máy. AC-AI-08 cần weights.

## Boundaries

- Always: đọc `.npz` với `np.load(..., allow_pickle=False)`; trả `model_version` trong mọi response của `/predict`.
- Ask first: đổi API contract; đổi ngưỡng mặc định 0.80; thêm endpoint nhận cả patch 120×120.
- Never: nhận hoặc unpickle `.pickle` từ người dùng; commit weights vào git; sửa `models/TSViT/TSViT_dinov3.py` hay code train.

## Success Criteria

- `pytest -q tests/test_api.py` pass mà không cần weights.
- Khi có weights, chạy local thì `curl` ở mục Commands trả JSON đúng contract.
- `api/MODEL_CARD.md` có đủ 6 trường của slide buổi 7.

## Open Questions

1. Dùng chuỗi `model_version` là `tsvit-dinov3-fold1-t9` được không?
2. Ngưỡng 0.80 lấy theo slide. Có cần chọn lại ngưỡng trên eval split không? Slide buổi 7 ghi: "Threshold không phải chọn theo cảm tính".
