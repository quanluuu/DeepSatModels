# Spec: ai-attack-suite

Module 2/3 của [SPEC-ai-qa.md](SPEC-ai-qa.md). Phụ thuộc: `ai-test-suite` (dùng chung helper; mọi bản sửa phải giữ TC-01..10 xanh).

## Objective

Có 10 test case tấn công hệ thống AI (API suy luận + model), đánh số ATK-01..ATK-10, mỗi test một loại tấn công khác nhau. Mỗi test case có:

- input tấn công,
- kỳ vọng an toàn,
- kết quả trước khi sửa và sau khi sửa, đo bằng lần chạy thật.

Lỗ hổng nhỏ được sửa trong `api/main.py`. Lỗ hổng cần thay đổi lớn, như huấn luyện đối kháng, chỉ ghi nhận. Module `qa-report-pdf` dùng kết quả của module này.

### Số đo trên dữ liệu thật (300 cửa sổ fold 5, 2026-11-14)

Luật chặn miền giá trị dựa trên các số này, để không chặn nhầm dữ liệu PASTIS hợp lệ:

| Đại lượng | Đo được | Hệ quả cho luật chặn |
|---|---|---|
| `img` | min -10000 (nodata, có trong 88/300 cửa sổ); ngoài -10000 thì min -3186, max 19080 | Nhận -10000 và khoảng [-5000, 25000]. Ngoài khoảng đó trả 422 |
| `doy` | Không giảm ở cả 300 cửa sổ; mỗi ngày lặp tối đa 2 lần (2 ảnh cùng ngày); không cửa sổ nào tăng ngặt | Luật là `doy` không giảm và mỗi giá trị lặp ≤ 2 lần. Không dùng luật "tăng ngặt" |
| Cửa sổ hằng số | Không có | `img` hằng số trên toàn cửa sổ trả 422 (không có dữ liệu) |

### Acceptance criteria

"Trước" là dự đoán khi đọc code. Cột này được thay bằng kết quả thật ở lần chạy đầu tiên.

| ID | Tấn công | Input | Kỳ vọng an toàn (sau sửa) | Trước (dự đoán) | Sửa |
|---|---|---|---|---|---|
| ATK-01 | Adversarial FGSM | 20 cửa sổ eval; nhiễu một bước theo dấu gradient, ε = 0.03 × std từng band, chỉ trên 10 band (không đụng kênh doy) | Đo: độ giảm pixel accuracy, tỉ lệ pixel đổi lớp, tỉ lệ cửa sổ chuyển thành UNCERTAIN. So với nhiễu ngẫu nhiên cùng ε. Tiêu chí an toàn: accuracy giảm ≤ 10 điểm, hoặc ≥ 80% cửa sổ bị tấn công thành UNCERTAIN | Không đạt | Không sửa (cần huấn luyện đối kháng). Đánh dấu `xfail(strict=True)` kèm số đo |
| ATK-02 | Adversarial PGD | Như ATK-01, 10 bước, bước 0.01 × std, chiếu về quả cầu L∞ ε | Như ATK-01 | Không đạt, nặng hơn FGSM | Như ATK-01 |
| ATK-03 | Giá trị ngoài miền | `img` toàn 32767; toàn -32768; một pixel 30000; `img` hằng số 0 | 422, model không được gọi | 200 (model chạy trên rác) | Kiểm miền giá trị trong `read_window` |
| ATK-04 | Tiêm NaN/Inf | `img` float32 có một NaN; có một +Inf | 422, model không được gọi | 200 với stub; model thật có thể trả 500 vì `confidence` NaN không serialize được JSON | Kiểm `np.isfinite` |
| ATK-05 | Decompression bomb | `.npz` nén khoảng 1 MB, bên trong `img` float64 (20000, 10, 24, 24) ≈ 920 MB | 422; bộ nhớ cấp phát đỉnh khi xử lý < 50 MB (đo bằng `tracemalloc`) | Cấp phát khoảng 920 MB rồi mới trả 422 vì shape sai | Kiểm kích thước giải nén khai báo của từng member zip (≤ 366 × 10 × 24 × 24 × 8 byte) trước khi đọc |
| ATK-06 | Pickle / object array (RCE) | `.npz` chứa object array mang payload `__reduce__` ghi file đánh dấu; file `.pickle` thô | 422, file đánh dấu không được tạo | Đã chặn (`allow_pickle=False`) | Không cần |
| ATK-07 | File giả mạo | zip bị cắt cụt; header PNG ghép trước zip; filename `../../etc/passwd`; content-type `image/png` cho npz hợp lệ | 3 ca đầu trả 422. Ca filename/content-type trả 200 và không file nào được ghi theo filename | Đã chặn | Không cần |
| ATK-08 | Upload quá cỡ | Body 50 MB có `Content-Length` | 413 trước khi đọc body | 422, nhưng toàn bộ body đã bị spool xuống đĩa tạm | Middleware kiểm `Content-Length` > 5 MB + 64 KB thì trả 413 |
| ATK-09 | Flood / DoS | 12 request đồng thời, stub ngủ 0.5 s | Request vượt quá 1 đang chạy + 4 chờ nhận 503 kèm `Retry-After`; `/health` trả lời < 1 s trong lúc flood | Mọi request xếp hàng tới timeout rồi trả FALLBACK | Đếm số suy luận đã submit mà chưa xong (giảm trong `future.add_done_callback`). Lớn hơn 5 thì trả 503 |
| ATK-10 | Thao túng chuỗi thời gian | `doy` đảo ngược; `doy` xáo trộn; một ngày lặp 3 lần | 422, model không được gọi | 200 | `doy` phải không giảm, mỗi giá trị lặp ≤ 2 lần |

Thêm các điều kiện sau:

1. Mỗi test có comment `# ATK-xx` ngay trên dòng `def` (hoặc trên decorator).
2. Lần chạy trước khi sửa lưu `docs/report/qa/junit-attack-before.xml`. Lần chạy sau khi sửa lưu `junit-attack-after.xml`.
3. Sau khi sửa: TC-01..05 và ATK-03..10 xanh trên CI. TC-06..08 và ATK-01..02 chạy local. TC-10 (k6 5 VUs) chạy lại, không có 503, p95 < 5 s.
4. Mỗi thông báo 422/413/503 là câu tiếng Anh ngắn, nêu lý do, không lộ stack trace hay đường dẫn.

## Tech Stack

- pytest, FastAPI `TestClient`, numpy, `tracemalloc` và `zipfile` (thư viện chuẩn): đã có.
- ATK-01/02: torch 2.11 trong `.venv`, gọi `predictor.model` và `predictor.transform` trực tiếp để lấy gradient theo input đã chuẩn hóa.
- Không thêm dependency.

## Commands

```bash
# ATK-03..10 (stub, chạy cả trên CI)
.venv/Scripts/python -m pytest -q tests/test_attack_api.py --junitxml docs/report/qa/junit-attack-api.xml

# ATK-01..02 (model thật, local)
.venv/Scripts/python -m pytest -q tests/test_attack_model.py -rA --junitxml docs/report/qa/junit-attack-model.xml

# bằng chứng trước/sau: chạy cả hai file, đổi tên output thành junit-attack-before.xml / junit-attack-after.xml
.venv/Scripts/python -m pytest -q tests/test_attack_api.py tests/test_attack_model.py -rA --junitxml docs/report/qa/junit-attack-after.xml

# hồi quy sau khi sửa
.venv/Scripts/python -m pytest -q
k6 run --summary-export docs/report/qa/k6-summary-after.json tests/load/predict.js
ruff check app tests api scripts
```

CI: thêm `tests/test_attack_api.py` vào bước pytest của job `test-api` trong `.github/workflows/ci.yml`. Đã có phê duyệt trong spec này.

## Project Structure

```text
  api/main.py                  -> read_window: kiểm kích thước member zip, isfinite, miền giá trị, img hằng số, doy không giảm;
                                  create_app: middleware 413 theo Content-Length, giới hạn số suy luận đang chờ (503 + Retry-After)
+ tests/test_attack_api.py     -> ATK-03..10 với StubPredictor (dùng lại helper của tests/test_api.py)
+ tests/test_attack_model.py   -> ATK-01..02 với model thật, skip khi thiếu weights/eval set như tests/test_ai_model.py
  tests/test_api.py            -> chỉ sửa nếu một ca 422 cũ đổi lý do (ví dụ dữ liệu toàn 0 trở thành "hằng số")
  .github/workflows/ci.yml     -> job test-api chạy thêm tests/test_attack_api.py
  api/MODEL_CARD.md            -> mục Input: luật miền giá trị, 413, 503
+ docs/report/qa/              -> junit-attack-before.xml, junit-attack-after.xml, k6-summary-after.json
```

## Code Style

Như `api/main.py` và `tests/test_api.py`: hằng số có comment một dòng nêu lý do, `ValueError` với thông báo cho người dùng, test ngắn.

```python
IMG_RANGE = (-5000, 25000)  # 300 fold 5 windows: -3186..19080 besides the -10000 nodata value
NODATA = -10000
MAX_MEMBER_BYTES = MAX_DATES * BANDS * WIN * WIN * 8    # largest valid array (float64); bigger means a decompression bomb

    if not np.isfinite(img).all():
        raise ValueError('img contains NaN or Inf')


# ATK-10
@pytest.mark.parametrize('doy', [np.arange(5, 0, -1), np.array([3, 1, 5, 2, 4]), np.array([1, 2, 2, 2, 3])])
def test_doy_manipulation_is_rejected(doy):
    stub = StubPredictor()
    assert post(client(stub), npz_bytes(img=np.zeros((5, 10, 24, 24), np.int16) + 100, doy=doy)).status_code == 422
    assert stub.calls == 0
```

## Testing Strategy

| Nhóm | Cách | Chạy ở |
|---|---|---|
| ATK-03..10 | `StubPredictor`, `TestClient`; ATK-09 bắn request đồng thời bằng thread | CI job `test-api` và local |
| ATK-01..02 | Model thật, gradient theo input, 20 cửa sổ đầu của `eval_windows.npz` | Local (khoảng 10 phút, ước lượng; đo thật ở bước đầu của plan) |
| Hồi quy | Toàn bộ pytest, TC-06..08, k6 TC-10 | Local + CI |

Thứ tự kiểu TDD: viết test tấn công trước. Lần chạy đầu là RED, đồng thời là bằng chứng "trước". Sau đó sửa từng lỗ hổng cho tới GREEN.

## Boundaries

- **Always:** chạy test tấn công trước khi sửa và lưu kết quả "trước". Giữ TC-01..10 xanh sau mỗi bản sửa. Payload tấn công chỉ chạy trên API local hoặc `TestClient`. File đánh dấu RCE ghi vào `tmp_path` của pytest.
- **Ask first:** đổi các ngưỡng miền giá trị, giới hạn hàng đợi hoặc ε. Đánh dấu `xfail` cho test nào ngoài ATK-01/02. Mọi thay đổi `.github/workflows/ci.yml` ngoài việc thêm `tests/test_attack_api.py`. Push lên GitHub.
- **Never:** tấn công hệ thống không thuộc dự án này (Docker Hub, GitHub, máy khác). Hạ ngưỡng hoặc xóa test để lấy màu xanh. Commit payload có thể chạy được ngoài test. Gắn attribution Claude vào commit.

## Ngoài phạm vi (ghi vào PDF là rủi ro còn lại)

- API không có xác thực; ai truy cập được cổng 8000 đều gọi được `/predict`.
- `/docs` và `/openapi.json` đang mở.
- Request không có `Content-Length` (chunked) vẫn bị spool xuống đĩa trước khi `read_window` cắt ở 5 MB. Uvicorn không có tùy chọn giới hạn body, cần reverse proxy.
- ATK-01/02 không được sửa. Muốn sửa cần huấn luyện đối kháng hoặc kiểm tra input bằng một detector riêng.

## Success Criteria

1. `tests/test_attack_api.py` có ATK-03..10 và chạy xanh trên CI sau khi sửa.
2. `junit-attack-before.xml` cho thấy các lỗ hổng thật. ATK nào dự đoán là lỗ hổng mà lần chạy thật lại pass thì ghi đúng như vậy, không sửa thêm.
3. ATK-05 đo được bộ nhớ đỉnh trước và sau khi sửa; sau khi sửa < 50 MB.
4. ATK-01/02 có số đo thật: accuracy sạch, accuracy sau tấn công, accuracy với nhiễu ngẫu nhiên cùng ε, tỉ lệ UNCERTAIN.
5. Toàn bộ pytest xanh. TC-06..08 vẫn pass sau khi sửa: luật miền giá trị không chặn cửa sổ thật nào trong 300 cửa sổ. k6 TC-10 chạy lại không có 503, p95 < 5 s.
6. `api/MODEL_CARD.md` mô tả luật input mới.

## Quyết định (2026-11-14)

1. Khoảng `img` [-5000, 25000] cộng giá trị nodata -10000: giữ nguyên.
2. ATK-01/02 không đạt tiêu chí an toàn thì đánh dấu `xfail(strict=True)` kèm số đo. Suite vẫn xanh, PDF ghi FAIL.

## Open Questions

Không còn.
