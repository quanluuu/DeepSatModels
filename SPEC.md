# Capability Map: CI/CD và Docker Hub cho DeepSatModels

Mục tiêu chung: mỗi lần push hoặc mở PR, GitHub Actions tự lint và test. Khi `main` xanh, Actions build hai Docker image và push public lên Docker Hub:

- `quanluuu/deepsat-api`: API suy luận chạy model `TSViT_DINOv3` thật trên CPU.
- `quanluuu/deepsat-demo`: app Streamlit hiển thị dự đoán tính sẵn.

Toàn bộ quy trình được tổng hợp thành một file PDF step-by-step để nộp giảng viên (buổi 10, "Triển khai, đóng gói, viết README").

Tài liệu tham chiếu: slide "Chuyên đề TTNT&KHDL" (Dat Tran). Các buổi được dùng:

- Buổi 7: model version, export, latency.
- Buổi 8: environment, container, server-side AI.
- Buổi 9 và 9B: test case, k6, AI acceptance OK/UNCERTAIN/FALLBACK.
- Buổi 10: cấu trúc README và expected output.

| Module id | Trách nhiệm | Phụ thuộc | Spec |
|---|---|---|---|
| demo-tests | ruff + pytest cho `app/demo_core.py`, AppTest cho app Streamlit | — | [SPEC-demo-tests.md](SPEC-demo-tests.md) |
| inference-api | FastAPI: `GET /health`, `POST /predict` nhận `.npz`, trả pred, confidence, status, model_version, latency_ms | — | [SPEC-inference-api.md](SPEC-inference-api.md) |
| containers | `docker/api.Dockerfile` (torch CPU + weights), `docker/demo.Dockerfile` (Streamlit), smoke test `/health` | inference-api | [SPEC-containers.md](SPEC-containers.md) |
| delivery-pipeline | GitHub Actions: lint, test, build, push Docker Hub; weights tải từ GitHub Release | demo-tests, containers | [SPEC-delivery-pipeline.md](SPEC-delivery-pipeline.md) |
| report-pdf | PDF step-by-step nộp giảng viên, có kết quả k6 p50/p95 và ảnh Actions + Docker Hub | delivery-pipeline | [SPEC-report-pdf.md](SPEC-report-pdf.md) |

Build order: `demo-tests`, `inference-api` → `containers` → `delivery-pipeline` → `report-pdf`

Module id cố định, không đổi tên giữa chừng. Plan và task chọn việc theo id này.

## Quyết định đã chốt

| Câu hỏi | Quyết định |
|---|---|
| Image chứa gì | Hai image: API suy luận và app demo |
| Weights lấy từ đâu khi build | GitHub Release asset của repo `quanluuu/DeepSatModels` |
| Registry | Docker Hub, namespace `quanluuu`, public |
| CI chạy khi nào | Push `main` và mọi PR; không bật branch protection |
| Python | 3.13 cho app demo và test; API theo yêu cầu của DINOv3 (>= 3.11) |
| Định dạng input API | Chỉ `.npz`, đọc với `allow_pickle=False`. Không nhận `.pickle` upload vì unpickle có thể chạy code tùy ý |
| Người đọc PDF | Giảng viên, tiếng Việt |

## Open Questions chung (chặn bước push public đầu tiên)

1. **License khi phân phối public.** Image API và Release chứa weights DINOv3 (DINOv3 License của Meta) và `best.pth` train trên PASTIS. Image demo chứa `demo_patches.npz` dẫn xuất từ PASTIS. Cần xác nhận hai license cho phép phân phối lại. Nếu được phép, image phải kèm file license tương ứng. Nếu chưa rõ, để repo Docker Hub ở chế độ private cho đến khi xác nhận.
2. **Commit DINOv3 cố định.** API cần snapshot `facebookresearch/dinov3` để gọi `torch.hub.load(source='local')`. Đề xuất pin theo commit SHA mà bạn dùng trên Kaggle. Bạn cung cấp SHA đó, hoặc lấy commit mới nhất của `main` tại thời điểm làm.
