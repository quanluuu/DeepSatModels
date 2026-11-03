# Spec: report-pdf

## Objective

Viết một file PDF tiếng Việt, hướng dẫn step-by-step quy trình CI/CD và đóng gói Docker của dự án, để nộp giảng viên (buổi 10). Người đọc phải làm lại được từng bước mà không cần hỏi thêm. Cấu trúc theo slide buổi 10: mỗi bước có lệnh, vị trí chạy và expected output.

Acceptance criteria:

1. File `docs/report/CICD_DeepSatModels.pdf`, khổ A4, có mục lục và các phần sau, theo thứ tự:
   1. Project Overview (5 dòng)
   2. System Overview (sơ đồ: GitHub → Actions → Docker Hub → `docker run`)
   3. Environment (bảng OS, Python, Docker, version đã test)
   4. Chuẩn bị: tài khoản Docker Hub, Access Token, GitHub secrets, Release weights
   5. CI: lint + test (lệnh, ảnh Actions xanh và đỏ)
   6. Build image API và demo (Dockerfile giải thích từng khối)
   7. CD: push Docker Hub (tag `latest`, `sha-`, version)
   8. Run & Verify: `docker pull`, `docker run`, `curl /health` và `/predict`, kèm expected output
   9. AI Model: bảng model card (slide buổi 7)
   10. Kiểm thử: bảng test case AC-AI-01..08 (slide buổi 9), kết quả pass/fail
   11. Performance: k6 5 VUs × 2 phút trên `/predict`, bảng p50/p95/p99, error rate, unknown rate, kết luận pass/fail (slide buổi 9B)
   12. Troubleshooting (ít nhất 5 lỗi thường gặp)
   13. Giới hạn hiện tại
2. Mọi lệnh trong PDF đã được chạy thật. Mọi số liệu (latency, kích thước image, thời gian pipeline) là số đo thật, không phải số ví dụ.
3. Có ít nhất 4 ảnh chụp: Actions xanh, Actions đỏ, trang tag trên Docker Hub, response `/predict`.
4. Checklist README của slide buổi 10 (8 câu) được tick đủ ở cuối PDF.

## Tech Stack

- Nguồn: `docs/report/report.html` (HTML + CSS in, viết tay).
- Xuất PDF: Chrome headless, có sẵn trên máy tại `C:\Program Files\Google\Chrome\Application\chrome.exe`.
- Load test: k6.

## Commands

```bash
# đo tải (container API đang chạy ở :8000)
k6 run --summary-export docs/report/k6-summary.json tests/load/predict.js

# xuất PDF
"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --disable-gpu --no-pdf-header-footer ^
  --print-to-pdf="docs/report/CICD_DeepSatModels.pdf" docs/report/report.html
```

## Project Structure

```text
+ tests/load/predict.js          -> k6: vus 5, duration 2m, thresholds p(95) và rate lỗi, check model_version
+ docs/report/report.html        -> nguồn PDF
+ docs/report/img/*.png          -> ảnh chụp
+ docs/report/k6-summary.json    -> kết quả đo thô
+ docs/report/CICD_DeepSatModels.pdf
```

`.gitignore` đang chặn `docs/`. Cần mở cho `docs/report/` (Ask first).

## Code Style

Văn bản tiếng Việt, câu ngắn, theo bộ quy tắc "Không mơ hồ / Tái lập được / Có kết quả kỳ vọng" của slide buổi 10. Mỗi bước có cấu trúc:

```text
Bước 5.2 — Chạy test local
Chạy ở: thư mục gốc repo, venv đã bật
Lệnh:   pytest -q
Kết quả đúng: "NN passed in X.XXs"
```

## Testing Strategy

- Một người khác (hoặc chính bạn trên máy sạch) làm theo PDF từ bước 4 đến bước 8. Phải ra cùng kết quả.
- Đối chiếu từng con số trong PDF với log Actions và `k6-summary.json`.

## Boundaries

- Always: che token và email trong ảnh chụp.
- Ask first: mở `docs/` khỏi `.gitignore`; cài k6 (`winget install k6`).
- Never: chèn số liệu giả hoặc số liệu ví dụ của slide như thể là kết quả của dự án.

## Success Criteria

PDF mở được và đủ 13 phần, 4 ảnh. Checklist buổi 10 được tick 8/8.

## Open Questions

1. Ảnh chụp Actions và Docker Hub cần đăng nhập. Bạn tự chụp, hay để mình chụp qua trình duyệt của bạn (cần cho phép)?
2. Tiêu chí hiệu năng đặt trước khi đo: p95 < 5 s cho một cửa sổ 24×24 trên CPU, error rate < 1%. Có đồng ý không?
3. Có ghi tên nhóm hoặc thành viên ở trang bìa không?
