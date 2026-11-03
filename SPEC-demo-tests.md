# Spec: demo-tests

## Objective

Tạo bộ kiểm tra tự động cho app demo để CI có cái để chạy. Hiện repo chưa có test nào. Module này là "Unit Test" và một phần "System Test" theo slide buổi 9.

Acceptance criteria:

1. `pytest` có test cho mọi hàm public của `app/demo_core.py`: `stitch`, `clip_roi`, `roi_from_geojson`, `area_table`, `overlay`, `panels`, `clearest_date`, `to_rgb_uint8`. Mọi assert trong khối `__main__` hiện có đều được chuyển sang pytest, không bỏ case nào.
2. `tests/test_app.py` dùng `streamlit.testing.v1.AppTest` chạy `app/crop_recognition_app.py` với `app/data/demo_patches.npz` thật. Test pass khi app render xong mà không có exception.
3. Khi `DEMO_DATA` trỏ tới file không tồn tại, app hiện `st.error` và có test cho trường hợp đó.
4. `ruff check app tests` không báo lỗi.
5. Lint và test chạy local dưới 1 phút.

## Tech Stack

Python 3.13, `pytest`, `ruff`, `streamlit>=1.40` (đã có trong `requirements.txt`).

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check app tests
pytest -q tests/test_demo_core.py tests/test_app.py
```

## Project Structure

```text
+ requirements-dev.txt   -> pytest, ruff
+ pyproject.toml         -> [tool.ruff] line-length = 130, include app/ tests/ api/; [tool.pytest.ini_options] pythonpath = ["app", "."]
+ tests/test_demo_core.py
+ tests/test_app.py
```

Cần `pythonpath` chứa `app` vì app import `from demo_core import ...`.

## Code Style

Viết giống code hiện có: hàm ngắn, comment cuối dòng giải thích con số, không dùng type hint. Mỗi test kiểm một hành vi.

```python
def test_area_table_counts_only_inside_roi():
    pred = np.zeros((120, 120), np.uint8)
    pred[:, 60:] = 2          # soft winter wheat on the right half
    pred[100:, 60:] = 8       # grapevine in the bottom-right corner
    rows, total, background = area_table(pred, (50, 90, 30, 30))   # x, y, w, h
    assert abs(total - 9.0) < 1e-9
    assert [r[0] for r in rows] == [8, 2]
```

## Testing Strategy

| Tầng | Nội dung |
|---|---|
| Unit (white-box) | Hàm numpy: shape, dtype, cắt ROI ở biên, diện tích theo ha, ROI nằm ngoài ảnh trả `None` |
| System (black-box) | AppTest render với data thật; thiếu file data thì hiện `st.error` |

Không test component `st_folium`, vì AppTest không tương tác được với custom component.

## Boundaries

- Always: giữ nguyên logic trong `app/`.
- Ask first: sửa code trong `app/` để test dễ hơn; xoá khối `__main__` trong `demo_core.py`.
- Never: lint hoặc reformat `models/`, `train_and_eval/`, `data/`, `utils/`; skip test để CI xanh.

## Success Criteria

- Hai lệnh `ruff` và `pytest` ở mục Commands đều thoát với mã 0.
- Sửa `PIXEL_HA` thành `0.02` làm ít nhất một test fail. Đây là bằng chứng test bắt được lỗi.

## Open Questions

1. Sau khi chuyển sang pytest, có xoá khối self-check `__main__` không? Đề xuất: xoá.
