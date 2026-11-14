# Spec: ai-test-suite

Module 1/3 của [SPEC-ai-qa.md](SPEC-ai-qa.md). Phụ thuộc: không có module mới; dùng `inference-api` và `containers` đã có.

## Objective

Có 10 test case kiểm thử hệ thống AI (API suy luận + Docker image), đánh số TC-01..TC-10. Bộ test phủ ba loại theo đề bài: mock test, AI test, automation test. Bộ test cũng phủ hai tầng: API và system. Mỗi test case có ID, input, expected output và một lệnh chạy cho kết quả pass/fail thật. Module `qa-report-pdf` dùng kết quả này.

Người dùng: sinh viên chạy test, giảng viên đọc kết quả trong PDF.

### Acceptance criteria

| ID | Loại | Tầng | Input | Expected | Ở đâu |
|---|---|---|---|---|---|
| TC-01 | Mock | API | `GET /health`, stub predictor | 200, có `status=ok`, `model_version`, `device` | `tests/test_api.py` (có sẵn) |
| TC-02 | Mock | API | confidence 0.95 / 0.80 / 0.79 / 0.20 | OK / OK / UNCERTAIN / UNCERTAIN | `tests/test_api.py` (có sẵn) |
| TC-03 | Mock | API | stub ngủ lâu hơn `timeout_s` | 200, `status=FALLBACK`, `pred=null`; request kế tiếp vẫn được phục vụ | `tests/test_api.py` (có sẵn) |
| TC-04 | Mock | API | stub ném `RuntimeError` | 200, `status=FALLBACK`, không 500 | `tests/test_api.py` (có sẵn) |
| TC-05 | Mock | API | shape sai, `doy` ngoài 1..366, dtype sai, file không phải npz | 422, `stub.calls == 0` | `tests/test_api.py` (có sẵn) |
| TC-06 | AI | API, model thật | 300 cửa sổ fold 5 kèm nhãn | Trên confusion matrix gộp (bỏ nhãn void 19): OA ≥ 74.6%, mIoU ≥ 48.9% | `tests/test_ai_model.py` (mới) |
| TC-07 | AI | API, model thật | `tests/fixtures/window.npz` gửi 2 lần | `pred` giống hệt, `confidence` bằng nhau (API làm tròn 4 chữ số) | `tests/test_ai_model.py` (mới) |
| TC-08 | AI | API, model thật | 10 cửa sổ, nhân `img` với hệ số đều trong [0.99, 1.01], seed 0 | Trung bình tỉ lệ pixel giữ nguyên lớp ≥ 95% | `tests/test_ai_model.py` (mới) |
| TC-09 | Automation | System | image `deepsat-api:ci` chạy bằng `docker run` | `scripts/smoke_api.py` exit 0: `/health` ok, `/predict` 200, `pred` 24x24 với giá trị 0..18, file rác bị 422 | job `docker` trong `.github/workflows/ci.yml` (có sẵn) |
| TC-10 | Automation | System | k6 5 VUs × 2 phút trên `/predict`, uvicorn với model thật | p95 < 5 s, `http_req_failed` < 1%, `checks` > 99% | `tests/load/predict.js` (có sẵn) |

Thêm các điều kiện sau:

1. Mỗi test function có comment `# TC-xx` ngay trên dòng `def`, để tra ngược từ bảng về code.
2. `pytest -q tests/test_api.py` vẫn chạy được trên CI mà không cần torch hay weights.
3. Thiếu weights, `weights/dinov3` hoặc `tests/fixtures/eval_windows.npz` thì `tests/test_ai_model.py` skip kèm lý do rõ ràng, không fail.
4. Kết quả lưu dạng JUnit XML trong `docs/report/qa/`, làm bằng chứng cho PDF.

## Tech Stack

- pytest, FastAPI `TestClient`, numpy: đã có trong `requirements-dev.txt` và `api/requirements.txt`.
- Model thật: torch 2.11 trong `.venv`, `weights/best.pth`, snapshot `facebookresearch/dinov3` commit `6876159`.
- k6 cho TC-10. GitHub Actions + Docker cho TC-09.
- Không thêm dependency.

## Commands

```bash
# một lần: snapshot DINOv3 (chỉ code, weights backbone nằm trong best.pth)
git clone https://github.com/facebookresearch/dinov3 weights/dinov3
git -C weights/dinov3 checkout 6876159

# một lần: cắt 300 cửa sổ có nhãn từ fold 5 (khoảng 130 MB, không commit, nằm trong .gitignore)
.venv/Scripts/python -I scripts/make_fixture.py --zip pastis24.zip --eval 300 --out tests/fixtures/eval_windows.npz

# TC-01..05 (mock, chạy cả trên CI)
.venv/Scripts/python -m pytest -q tests/test_api.py --junitxml docs/report/qa/junit-mock.xml

# TC-06..08 (model thật, local, khoảng 3-4 phút trên CPU)
.venv/Scripts/python -m pytest -q tests/test_ai_model.py -rs --junitxml docs/report/qa/junit-ai.xml

# TC-10 (terminal 1: API thật; terminal 2: k6)
.venv/Scripts/uvicorn --factory api.main:create_app --port 8000
k6 run --summary-export docs/report/qa/k6-summary.json tests/load/predict.js

# TC-09: push nhánh, xem job docker trên GitHub Actions (bước "Smoke test API")
ruff check app tests api scripts
```

## Project Structure

```text
  tests/test_api.py               -> TC-01..05: thêm comment # TC-xx, không đổi logic
+ tests/test_ai_model.py          -> TC-06..08: TestClient(create_app()) với Predictor thật, app load một lần (fixture scope module)
  scripts/make_fixture.py         -> thêm --eval N: N cửa sổ chia đều trên fold_5_paths.csv, lưu img_i, doy_i, label_i (labels[0])
+ tests/fixtures/eval_windows.npz -> sinh ra, không commit (thêm vào .gitignore)
  scripts/smoke_api.py            -> TC-09, giữ nguyên
  tests/load/predict.js           -> TC-10, giữ nguyên
+ docs/report/qa/                 -> junit-*.xml, k6-summary.json: bằng chứng cho qa-report-pdf
```

## Code Style

Theo `tests/test_api.py`: hàm test ngắn, helper ở đầu file, docstring module nêu spec, comment chỉ giải thích lý do.

```python
"""real-model tests (SPEC-ai-test-suite TC-06..08): run locally, skipped when weights or the eval set are missing"""
EVAL = os.path.join(os.path.dirname(__file__), 'fixtures', 'eval_windows.npz')
missing = [p for p in ('weights/best.pth', 'weights/dinov3/hubconf.py', EVAL) if not os.path.exists(p)]
pytestmark = pytest.mark.skipif(bool(missing), reason='needs %s' % ', '.join(missing))


# TC-07
def test_same_input_gives_same_prediction(api):
    a, b = (post(api, read(FIXTURE)).json() for _ in range(2))
    assert a['pred'] == b['pred'] and a['confidence'] == b['confidence']
```

- Ruff, `line-length = 150`, rule E và F, như `pyproject.toml`.
- Tính OA và mIoU bằng `np.bincount` trên confusion matrix 19x19. mIoU lấy trung bình trên các lớp có mặt trong nhãn hoặc trong dự đoán.

## Testing Strategy

| Tầng | Cách | Chạy ở |
|---|---|---|
| Mock (TC-01..05) | `StubPredictor` thay model, không cần torch | CI job `test-api` và local |
| AI (TC-06..08) | Model thật qua `TestClient`, đi đúng đường của request thật | Chỉ local: CI không có snapshot DINOv3 và chạy CPU chậm |
| System (TC-09) | Image thật, `docker run`, HTTP thật | CI job `docker` |
| System (TC-10) | k6 bắn HTTP vào uvicorn local | Local |

Nếu TC-06 không đạt ngưỡng, giữ test fail và ghi số thật vào báo cáo. Không hạ ngưỡng khi chưa hỏi.

## Boundaries

- **Always:** chạy `ruff check` và `pytest -q tests/test_api.py` trước khi commit. Mọi số trong báo cáo lấy từ lần chạy thật. Đọc dữ liệu PASTIS bằng `pickle` chỉ từ `pastis24.zip` local.
- **Ask first:** đổi ngưỡng TC-06/TC-08/TC-10, thêm dependency, sửa `.github/workflows/ci.yml`, đổi logic `api/main.py` (logic API thuộc module `ai-attack-suite`).
- **Never:** commit `eval_windows.npz`, `weights/*` hay dữ liệu PASTIS. Xóa hoặc skip test đang fail để lấy màu xanh. Gắn attribution Claude vào commit.

## Success Criteria

1. `pytest -q tests/test_api.py` pass trên CI, với TC-01..05 gắn comment ID.
2. `pytest -q tests/test_ai_model.py -rs` chạy local với model thật. TC-06..08 có kết quả pass/fail thật, không skip. Số OA, mIoU và tỉ lệ ổn định được in ra.
3. Máy thiếu weights: `tests/test_ai_model.py` skip kèm lý do, exit 0.
4. Job `docker` của CI xanh, log có dòng `smoke test passed` (TC-09).
5. k6 chạy xong, `docs/report/qa/k6-summary.json` có p95 và error rate (TC-10).
6. `docs/report/qa/` có `junit-mock.xml` và `junit-ai.xml`.

## Open Questions

Không còn.
