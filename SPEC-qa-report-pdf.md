# Spec: qa-report-pdf

Module 3 của [SPEC-ai-qa.md](SPEC-ai-qa.md). Phụ thuộc: [SPEC-ai-test-suite.md](SPEC-ai-test-suite.md), [SPEC-ai-attack-suite.md](SPEC-ai-attack-suite.md). Cả hai đã xong (2026-11-14).

## Objective

Tạo một file PDF tiếng Việt để nộp giảng viên, đáp ứng mục 3 của đề bài: "Tổng hợp thành một file PDF". PDF trình bày 10 test case kiểm thử (TC-01..10) và 10 test case tấn công (ATK-01..10) trên API suy luận và container của DeepSatModels. Mỗi con số trong PDF là số đo thật, lấy từ file bằng chứng trong `docs/report/qa/`.

Người đọc là giảng viên. Họ cần thấy:

1. Hệ thống nào được kiểm thử, và phạm vi.
2. Mỗi test case kiểm tra gì, thuộc loại nào (Mock / AI / Automation), ở tầng nào (API / System).
3. Cách chạy lại, và kết quả đúng phải như thế nào.
4. Với tấn công: kết quả trước khi sửa, bản sửa, kết quả sau khi sửa.
5. Rủi ro còn lại mà dự án chưa sửa.

Acceptance criteria:

1. File `docs/report/Testing_Attack_DeepSatModels.pdf`, khổ A4, có các phần sau, theo thứ tự:
   1. Bìa: tên đề tài, sinh viên, mã sinh viên, lớp, repository, commit được kiểm thử, ngày.
   2. Mục lục.
   3. Hệ thống được kiểm thử: sơ đồ luồng `client → /predict → read_window → hàng đợi → model → JSON`; phạm vi và ngoài phạm vi (demo Streamlit).
   4. Môi trường: bảng OS, Python, torch, CPU, dữ liệu (PASTIS fold 5, 300 cửa sổ), k6; nơi mỗi nhóm test chạy (CI hay local).
   5. Chiến lược: ma trận Loại × Tầng cho 10 TC; cách đặt ngưỡng (model card trừ 5 điểm, ε = 0.03 std...).
   6. Bảng tổng 10 TC: ID, kiểm tra gì, loại, tầng, chạy ở, kết quả kỳ vọng, kết quả thật, Pass/Fail.
   7. Chi tiết từng TC theo khuôn "Chạy ở / Lệnh / Kết quả đúng / Kết quả thật / Bằng chứng".
   8. Bảng tổng 10 ATK: ID, loại tấn công, mối đe dọa, trước khi sửa, sau khi sửa.
   9. Chi tiết từng ATK: cách tấn công, kết quả trước, đoạn code sửa (chỉ những dòng chính), kết quả sau. ATK-01/02 có biểu đồ cột accuracy: sạch, nhiễu ngẫu nhiên, FGSM, PGD.
   10. Hồi quy sau khi sửa: TC-06..08 và k6 trước/sau trong một bảng.
   11. Rủi ro còn lại: ATK-01/02 chưa sửa, không xác thực, `/docs` và `/openapi.json` mở, upload chunked không có `Content-Length`. Mỗi rủi ro có hướng khắc phục.
   12. Kết luận (5 dòng).
   13. Phụ lục: bảng "số liệu → file nguồn", danh sách commit.
2. Mọi số liệu khớp với file trong `docs/report/qa/` hoặc output pytest đã lưu. Không có số ví dụ.
3. Có 2 ảnh chụp: CI smoke (`tc09-ci-smoke.png`) và CI xanh cho ATK-03..10 (`atk-ci-green.png`).
4. PDF không chứa token, email hay đường dẫn tuyệt đối của máy local.

## Tech Stack

- Nguồn: `docs/report/testing_attack.html`, HTML + CSS viết tay, dùng lại khối `<style>` của `docs/report/report.html`.
- Biểu đồ: SVG inline viết tay. Không thêm thư viện.
- Xuất PDF: Chrome headless tại `C:\Program Files\Google\Chrome\Application\chrome.exe`.

## Commands

```bash
# xuất PDF (chạy ở thư mục gốc repo)
"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --disable-gpu --no-pdf-header-footer ^
  --print-to-pdf="docs/report/Testing_Attack_DeepSatModels.pdf" docs/report/testing_attack.html

# đối chiếu số liệu: chạy lại các bộ test rồi so với PDF
.venv/Scripts/python -m pytest -q tests/test_api.py tests/test_attack_api.py -rA
.venv/Scripts/python -m pytest -q tests/test_ai_model.py tests/test_attack_model.py -rA -s
```

## Project Structure

```text
+ docs/report/testing_attack.html               -> nguồn PDF
+ docs/report/Testing_Attack_DeepSatModels.pdf   -> file nộp
  docs/report/qa/*                               -> bằng chứng đã có, chỉ đọc
  docs/report/report.html, CICD_DeepSatModels.pdf -> báo cáo cũ, không sửa
```

## Code Style

Văn bản tiếng Việt, câu ngắn, mỗi câu một ý. Thuật ngữ và tên trong code giữ nguyên (`read_window`, `Retry-After`, FGSM). Mỗi test case theo khuôn:

```text
TC-06 — Độ chính xác trên dữ liệu thật
Chạy ở:       thư mục gốc repo, local, có weights và tests/fixtures/eval_windows.npz
Lệnh:         python -m pytest -q tests/test_ai_model.py -k accuracy -rA -s
Kết quả đúng: OA ≥ 74.6% và mIoU ≥ 48.9%
Kết quả thật: OA 79.4%, mIoU 51.6% (300 cửa sổ) — PASS
Bằng chứng:   docs/report/qa/junit-ai-after.xml
```

## Testing Strategy

- Đối chiếu từng con số trong PDF với bảng phụ lục "số liệu → file nguồn". Mỗi dòng phải mở được file nguồn và thấy đúng số đó.
- Mở PDF, kiểm tra: đủ 13 phần, bảng không bị cắt ngang trang, font tiếng Việt hiển thị đúng, ảnh hiện.
- Tìm trong text của PDF: không có `@gmail`, `token`, `C:\Users`.

## Boundaries

- Always: chỉ dùng số đo thật đã lưu; ghi rõ test nào chạy ở CI, test nào chạy local.
- Ask first: push lên GitHub; chạy lại bộ test model thật (khoảng 15 phút) nếu cần số mới.
- Never: sửa `docs/report/report.html` và `CICD_DeepSatModels.pdf` (đang có thay đổi chưa commit của bạn); commit `eval_windows.npz`, `weights/*`, dữ liệu PASTIS; ghi ATK-01/02 là "đã sửa".

## Success Criteria

1. `docs/report/Testing_Attack_DeepSatModels.pdf` mở được, đủ 13 phần, có 20 test case.
2. 100% số liệu có dòng trong bảng phụ lục và khớp file nguồn.
3. Bảng tấn công cho thấy 8/10 ATK đã chặn sau khi sửa, 2/10 (ATK-01/02) còn là rủi ro, kèm số đo.
4. Bạn duyệt bản PDF trước khi commit.

## Quyết định (2026-11-15)

| Câu hỏi | Quyết định |
|---|---|
| Bằng chứng CI cho ATK-03..10 | Push, rồi bạn chụp màn hình CI xanh có bước "API contract and attack tests". Ảnh lưu ở `docs/report/qa/atk-ci-green.png` |
| Hình bản đồ 24×24 cho ATK-02 | Không làm |
| Bìa | Cùng thông tin sinh viên như báo cáo CI/CD |

## Open Questions

Không còn.
