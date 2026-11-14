# Capability Map: Kiểm thử và tấn công hệ thống AI DeepSatModels

Đề bài (ảnh bảng, buổi học):

1. Xây dựng 10 test case để kiểm thử hệ thống AI: automation test, AI test, mock test. Test cả API và system.
2. Xây dựng 10 test case để tấn công hệ thống AI, tức 10 loại tấn công khác nhau.
3. Tổng hợp thành một file PDF để nộp.

Hệ thống được kiểm thử: API suy luận FastAPI (`GET /health`, `POST /predict`) chạy model `TSViT_DINOv3` thật, và Docker image `quanluuu/deepsat-api`. App demo Streamlit nằm ngoài phạm vi.

| Module id | Trách nhiệm | Phụ thuộc | Spec |
|---|---|---|---|
| ai-test-suite | 10 test case kiểm thử TC-01..10: mock (stub predictor), AI (model thật), automation (container + CI + k6) | — (dùng `inference-api`, `containers` đã có) | `SPEC-ai-test-suite.md` (viết sau khi map được duyệt) |
| ai-attack-suite | 10 test case tấn công ATK-01..10. Ghi kết quả trước khi sửa, sửa lỗ hổng nhỏ, rồi chạy lại | ai-test-suite (dùng chung helper; mọi bản sửa phải giữ TC-01..10 xanh) | `SPEC-ai-attack-suite.md` |
| qa-report-pdf | PDF tiếng Việt `docs/report/Testing_Attack_DeepSatModels.pdf`: bảng 20 test case, kết quả đo thật, trước và sau khi sửa | ai-test-suite, ai-attack-suite | `SPEC-qa-report-pdf.md` |

Build order: `ai-test-suite` → `ai-attack-suite` → `qa-report-pdf`

Module id cố định, không đổi tên giữa chừng.

## Quyết định đã chốt

| Câu hỏi | Quyết định |
|---|---|
| Đối tượng | API + container. Không gồm demo Streamlit |
| Khi phát hiện lỗ hổng | Ghi nhận rồi sửa. PDF trình bày trước và sau khi sửa |
| Tấn công đối kháng (FGSM/PGD) | Có, chạy local với model thật trên CPU. CI không chạy |
| PDF | File mới riêng, xuất bằng Chrome headless như `report-pdf` |
| Người đọc PDF | Giảng viên, tiếng Việt |
| TC-06: dữ liệu và ngưỡng | 300 cửa sổ fold 5 (ban đầu 50; đổi ngày 2026-11-14 vì với 50 cửa sổ, bootstrap 95% của OA là 65.0..78.4% và có lớp chỉ 1 pixel nhãn). Ngưỡng = model card trừ 5 điểm: OA ≥ 74.6%, mIoU ≥ 48.9% |
| ATK-09: khi quá tải | 503 kèm `Retry-After` khi đã có hơn 4 request chờ |
| ATK-03, ATK-10: input sai miền giá trị | Chặn ở `read_window`, trả 422 (reflectance ngoài miền, `doy` không tăng dần) |

## Danh sách test case đề xuất (duyệt cùng map)

Ký hiệu: **Loại** là Mock / AI / Automation. **Tầng** là API / System. **Chạy ở** là CI / local.

### Kiểm thử (ai-test-suite)

| ID | Kiểm tra gì | Loại | Tầng | Chạy ở | Đã có? |
|---|---|---|---|---|---|
| TC-01 | `/health` trả `status`, `model_version`, `device` | Mock | API | CI | có (`test_health_reports_model_version`) |
| TC-02 | Biên ngưỡng confidence: 0.80 cho OK, 0.79 cho UNCERTAIN | Mock | API | CI | có |
| TC-03 | Model treo quá `INFER_TIMEOUT_S` thì trả FALLBACK, service vẫn sống | Mock | API | CI | có |
| TC-04 | Model ném exception thì trả FALLBACK, không trả 500 | Mock | API | CI | có |
| TC-05 | Input sai (shape, doy, dtype, file không phải npz) trả 422 và không gọi model | Mock | API | CI | có |
| TC-06 | Độ chính xác: pixel accuracy và mIoU trên N cửa sổ test fold 1 so với nhãn PASTIS, không thấp hơn model card trừ biên sai số | AI | API (model thật) | local | mới |
| TC-07 | Tất định: cùng input gửi 2 lần cho cùng `pred` và cùng `confidence` | AI | API (model thật) | local | mới |
| TC-08 | Metamorphic: nhiễu ±1% reflectance giữ ít nhất 95% pixel không đổi lớp | AI | API (model thật) | local | mới |
| TC-09 | Container end-to-end: `docker run` image thật, `/health`, `/predict` trên fixture trả map 24x24 với lớp hợp lệ | Automation | System | CI (job docker) | gần có (`scripts/smoke_api.py`) |
| TC-10 | Tải: k6 5 VUs × 2 phút, p95 dưới ngưỡng, error rate 0 | Automation | System | local | có (`tests/load/predict.js`) |

### Tấn công (ai-attack-suite)

Cột "Dự đoán" là giả thuyết đọc từ code, chưa chạy. Kết quả thật sẽ thay thế cột này.

| ID | Loại tấn công | Cách làm | Kỳ vọng an toàn | Dự đoán hiện tại |
|---|---|---|---|---|
| ATK-01 | Adversarial FGSM | Nhiễu gradient ε nhỏ lên `img`, đo tỉ lệ pixel đổi lớp và độ giảm accuracy | Đo và báo cáo; confidence nên giảm thành UNCERTAIN | model bị lừa, cần đo |
| ATK-02 | Adversarial PGD (nhiều bước) | Như ATK-01, 10 bước | Đo và báo cáo | model bị lừa mạnh hơn |
| ATK-03 | Giá trị ngoài phân phối | `img` = int16 max, âm, toàn 0 | 422 hoặc UNCERTAIN, không trả OK | có thể trả OK tự tin |
| ATK-04 | Tiêm NaN/Inf | `img` float chứa NaN, Inf | 422 | có thể 500 (JSON không serialize được NaN) |
| ATK-05 | Decompression bomb | `.npz` nén vài KB, giải nén ra mảng hàng GB | 422 trước khi cấp phát bộ nhớ | **lỗ hổng**: `d['img']` giải nén toàn bộ rồi mới kiểm shape |
| ATK-06 | Pickle / object array (RCE) | `.npz` chứa object array, file pickle có payload | 422, payload không chạy | chặn (`allow_pickle=False`) |
| ATK-07 | File giả mạo | zip hỏng, polyglot, content-type sai, filename `../../etc/passwd` | 422, filename bị bỏ qua | chặn |
| ATK-08 | Upload quá cỡ | Body > 5 MB | 422, không đọc hết vào RAM | chặn ở `read_window`; cần kiểm bộ nhớ khi spool multipart |
| ATK-09 | Flood / DoS | 50 request đồng thời liên tục | `/health` vẫn trả lời; request vượt sức trả 429/503 hoặc FALLBACK nhanh | **lỗ hổng**: không giới hạn hàng đợi, request xếp hàng tới timeout |
| ATK-10 | Thao túng chuỗi thời gian | `doy` trùng lặp, đảo thứ tự, T=1 | 422 hoặc UNCERTAIN | có thể trả OK với dự đoán sai |

Ngoài 10 loại trên, PDF sẽ ghi 2 rủi ro mà không sửa: API không có xác thực, và `/docs` + `/openapi.json` đang mở.

## Giả định (sửa ngay nếu sai)

1. Test case AI và tấn công đối kháng (TC-06..08, ATK-01..02) chạy local với model thật. Cần clone `facebookresearch/dinov3` commit `6876159` vào `weights/dinov3`, giống bước build image.
2. Máy local chưa cài Docker, nên TC-09 chạy trong job `docker` của GitHub Actions. TC-10 dùng k6 chạy local với `uvicorn`.
3. Dữ liệu cho TC-06 cắt từ `pastis24.zip`, fold 5 (test split của fold 1), theo cách `scripts/make_fixture.py` đang làm.
4. Mỗi bản sửa (ATK-05, ATK-09, có thể ATK-03/04/10) chỉ dùng thư viện đã có, không thêm dependency.

## Open Questions

Không còn. Ba câu hỏi trước đã chốt ở bảng "Quyết định đã chốt" (2026-11-14).
